"""
stream_server.py
================

The headless version of the project: it runs the camera, the model and the
counter on a Linux server with no screen attached, and publishes the annotated
video as a web page you open on your laptop.

    python server/stream_server.py

Then, on the laptop, browse to  http://<server-ip>:8000

What is different from main.py, and why
---------------------------------------
main.py is built around three things a server does not have: a window
(cv2.imshow), a keyboard (cv2.waitKey) and a person sitting in front of it
(the camera prompt, the drag-to-draw line screen). Everything below replaces
those three, and nothing else:

    window    -> an MJPEG stream over HTTP. Every browser on earth can show
                 one with a plain <img> tag, so there is nothing to install on
                 the laptop and nothing to configure.
    keyboard  -> buttons on that page, posting to a small JSON API.
    line setup-> you click the two ends of the line on a still frame in the
                 browser; the coordinates are posted back and saved to exactly
                 the same lines/<camera>.json the desktop program uses.

The counting itself is NOT reimplemented. CountPipeline, PersonDetector,
PersonCounter and CountingLine are imported from person_counter and used
unchanged, so the server and the laptop count identically - a change to the
counting rules reaches both.

The shape of the program
------------------------
One background thread (Engine) owns the camera and the model and runs flat
out, publishing the newest annotated frame into a single slot. The web server
threads only ever read that slot. That is the same decoupling camera.py
already does between the camera and inference, for the same reason: HTTP
clients come and go at their own speed, and none of them may be allowed to
slow the counting down or, worse, to call into a YOLO model from two threads
at once.
"""

# server_config must be imported before anything else: it puts the project
# root on sys.path, changes into it, and loads the .env file. The imports
# below depend on all three.
import server_config
from server_config import SERVER_DIR, settings

# person_counter's __init__ sets FFmpeg's RTSP transport to TCP, and FFmpeg
# only reads that setting when cv2 is first imported - so this import has to
# come before cv2, exactly as it does in main.py.
import person_counter  # noqa: F401  (imported for its side effect)

import hmac
import json
import logging
import os
import signal
import sys
import threading
import time
from functools import wraps
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

import config
from person_counter import drawing
from person_counter.app import CountPipeline, FpsMeter, TrackPipeline
from person_counter.camera import ThreadedVideoStream, open_stream
from person_counter.counting_line import CountingLine, line_path_for
from person_counter.detector import PersonDetector
from person_counter import source_prompt

log = logging.getLogger("person-counter")


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class _PyTorchDetector(PersonDetector):
    """PersonDetector running the .pt weights instead of the OpenVINO IR.

    PersonDetector always checks that the exported .xml exists and was built
    for this image size, because on the laptop that is always the thing being
    run. MODEL_BACKEND=pytorch deliberately is not running it, so the check
    has nothing to check - it would refuse to start over a file we are not
    about to open.

    Overriding it here rather than editing detector.py keeps the desktop
    program's behaviour exactly as it was: there, a missing export really is
    an error worth stopping for.
    """

    def _check_export(self):
        return


def build_detector() -> PersonDetector:
    if settings.model_backend == "pytorch":
        return _PyTorchDetector()
    return PersonDetector()


# ---------------------------------------------------------------------------
# The engine: camera + model + counter, in one background thread
# ---------------------------------------------------------------------------
class Engine:
    """Owns the camera and the model, and publishes annotated frames.

    Everything that touches the model happens on this one thread. The web
    layer never calls into YOLO; it reads `latest_jpeg()` and `stats()`, both
    of which are just guarded reads of plain data.
    """

    def __init__(self, source: str):
        self.source = source

        # The published frame, and the lock/condition guarding it. A Condition
        # rather than a sleep-and-poll loop so a browser receives each frame
        # the moment it exists, and burns nothing while waiting.
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0

        # The newest frame BEFORE anything was drawn on it, kept for the line
        # editor: drawing a tripwire over a picture already crossed by a red
        # tripwire is needlessly confusing.
        self._clean_frame = None
        self._clean_lock = threading.Lock()

        # State the web layer reports. Written here, read there; plain
        # assignments of immutable values, so no lock is needed for them.
        self.status = "starting"
        self.error: str | None = None
        self.frame_size = (0, 0)
        self.fps = 0.0
        self.lag_ms = 0.0
        self.dropped = 0
        self.drop_rate = 0.0
        self.started_at = time.time()
        self.connected_at: float | None = None

        self.detector: PersonDetector | None = None
        self.pipeline = None

        # Requests arriving from the web layer. Each is picked up by the loop
        # at a frame boundary, which is the only safe moment to rebuild a
        # pipeline that a half-finished inference might otherwise be using.
        self._pending_line: CountingLine | None = None
        self._clear_line = False
        self._pending_reset = False
        self._pending_track_point: str | None = None

        self._stop = threading.Event()
        self._restart = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ----------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run, name="engine", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._restart.set()          # break out of the frame loop immediately
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def switch_source(self, source: str):
        """Point the engine at a different camera, without restarting Python.

        The model stays loaded - it is the slow part, it takes no notice of
        where the frames come from, and reloading it would blank the stream
        for several seconds for no reason.
        """
        self.source = source
        self._restart.set()

    # -- requests from the web layer ---------------------------------------
    def set_line(self, line: CountingLine | None):
        if line is None:
            self._clear_line = True
            self._pending_line = None
        else:
            self._pending_line = line
            self._clear_line = False

    def request_reset(self):
        self._pending_reset = True

    def set_track_point(self, point: str):
        self._pending_track_point = point

    # -- the loop -----------------------------------------------------------
    def _run(self):
        while not self._stop.is_set():
            source = self.source
            try:
                self.status = "connecting"
                self._publish_message("Connecting to the camera ...")

                # open_stream picks the threaded reader for a live camera and
                # the sequential one for a file, and the `with` releases
                # either one on the way out - including when this body raises,
                # so a crash never leaves the camera holding a stream slot.
                with open_stream(source) as stream:
                    self.frame_size = (stream.width, stream.height)
                    self.connected_at = time.time()
                    self._build_pipeline()
                    self.status = "running"
                    self.error = None
                    log.info("Streaming %s at %dx%d",
                             config.mask_url(source), *self.frame_size)
                    self._frame_loop(stream)

            except RuntimeError as error:
                # VideoStream raises this with a troubleshooting checklist
                # when the source will not open at all.
                self.error = str(error).strip().splitlines()[0]
                log.warning("Camera error: %s", self.error)
            except Exception as error:                  # noqa: BLE001
                self.error = f"{type(error).__name__}: {error}"
                log.exception("Engine failed")

            self.connected_at = None

            if self._restart.is_set():
                # A deliberate restart (the camera was switched): go straight
                # round again rather than sitting out the retry delay.
                self._restart.clear()
                continue
            if self._stop.is_set():
                break

            self.status = "reconnecting"
            self._publish_message(
                f"Camera unavailable - retrying in {settings.retry_delay:.0f}s",
                self.error)
            # wait() returns early if stop() is called, so shutdown is prompt
            # rather than up to retry_delay seconds late.
            self._stop.wait(settings.retry_delay)

        self.status = "stopped"
        log.info("Engine stopped.")

    def _frame_loop(self, stream):
        fps_meter = FpsMeter()

        for frame in stream:
            if self._stop.is_set() or self._restart.is_set():
                return

            self._apply_pending_requests()

            # Copy before the pipeline draws on it: `frame` is annotated in
            # place, and the line editor wants the picture without overlays.
            with self._clean_lock:
                self._clean_frame = frame.copy()

            hud_lines = self.pipeline.process(frame)

            self.fps = fps_meter.tick()
            status_lines = [f"FPS: {self.fps:.1f}"]
            if isinstance(stream, ThreadedVideoStream):
                self.lag_ms = stream.last_frame_age * 1000
                self.dropped = stream.dropped
                self.drop_rate = stream.drop_rate
                status_lines.append(
                    f"Lag: {self.lag_ms:.0f} ms   "
                    f"Dropped: {self.dropped} ({self.drop_rate:.0f}%)")

            drawing.draw_hud(frame, hud_lines + status_lines)
            if hasattr(self.pipeline, "draw_extras"):
                self.pipeline.draw_extras(frame)

            self._publish(frame)

        # Falling out of the for loop means the source ran dry: a file that
        # ended, or a camera that dropped and would not come back.
        self.error = "The video source ended or dropped."

    def _apply_pending_requests(self):
        """Act on anything the web layer asked for, between frames."""
        if self._pending_track_point:
            point, self._pending_track_point = self._pending_track_point, None
            config.TRACK_POINT = point
            # Everyone's remembered side was decided by the OLD point, which
            # sat a person's height away from the new one. Keeping those would
            # book a phantom crossing for everybody on screen.
            if isinstance(self.pipeline, CountPipeline):
                self.pipeline.counter.states.clear()
            log.info("Counting by the %s now.", point)

        if self._clear_line:
            self._clear_line = False
            if isinstance(self.pipeline, CountPipeline):
                self._build_pipeline(line=None)

        if self._pending_line is not None:
            line, self._pending_line = self._pending_line, None
            self._build_pipeline(line=line)
            log.info("Counting line set to (%d,%d)-(%d,%d), dead zone %d px.",
                     line.x1, line.y1, line.x2, line.y2, line.effective_dead_zone)

        if self._pending_reset:
            self._pending_reset = False
            if hasattr(self.pipeline, "reset"):
                self.pipeline.reset()

    def _build_pipeline(self, line: CountingLine | None = "unset"):
        """(Re)build the pipeline, loading the model the first time only.

        `line` defaults to the sentinel "unset" rather than None because None
        is a meaningful value here - it means "no line, run in tracking mode"
        - and the default has to mean something else: "load whatever is saved
        for this camera".
        """
        if self.detector is None:
            self.status = "loading model"
            self._publish_message("Loading the model ...")
            self.detector = build_detector()

        if line == "unset":
            line = CountingLine.load(line_path_for(self.source),
                                     frame_size=self.frame_size)

        if line is None:
            # No line drawn for this camera yet. Tracking still runs, so the
            # browser shows live boxes and IDs while you place the line - you
            # can see people walking through the doorway you are about to put
            # the tripwire across.
            self.pipeline = TrackPipeline(self.detector)
        else:
            self.pipeline = CountPipeline(self.detector, line)

    # -- publishing ---------------------------------------------------------
    def _publish(self, frame):
        out = frame
        limit = settings.stream_max_width
        if limit and frame.shape[1] > limit:
            # Only ever shrinks, and only what is sent over the network. The
            # model, the line and the counts all worked on the full frame.
            height = int(frame.shape[0] * limit / frame.shape[1])
            out = cv2.resize(frame, (limit, height), interpolation=cv2.INTER_AREA)

        ok, buffer = cv2.imencode(
            ".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), settings.jpeg_quality])
        if not ok:
            return

        with self._condition:
            self._jpeg = buffer.tobytes()
            self._seq += 1
            self._condition.notify_all()

    def _publish_message(self, text: str, detail: str | None = None):
        """Publish a generated frame carrying a message.

        A browser showing an MJPEG stream that has stopped sending just keeps
        the last image on screen forever, which is indistinguishable from a
        camera where nothing is happening. Sending an actual picture that says
        what is wrong is the difference between "it is broken" and "it is
        reconnecting, wait".
        """
        width, height = (self.frame_size if self.frame_size[0] else (640, 360))
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (32, 32, 32)
        cv2.putText(frame, text, (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if detail:
            cv2.putText(frame, detail[:70], (20, height // 2 + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 160, 255), 1,
                        cv2.LINE_AA)
        self._publish(frame)

    def wait_for_jpeg(self, last_seq: int, timeout: float = 10.0):
        """Block until a frame newer than `last_seq` exists; return it and its
        sequence number, or (None, last_seq) if nothing arrived in time."""
        with self._condition:
            if self._seq == last_seq:
                self._condition.wait(timeout)
            if self._seq == last_seq:
                return None, last_seq
            return self._jpeg, self._seq

    def latest_jpeg(self):
        with self._condition:
            return self._jpeg

    # The line editor's backdrop never needs to be bigger than this. A 4K
    # camera's full frame is a megabyte of JPEG for a picture you are going to
    # drag a line across by eye, and on a slow link that is a visible pause
    # between pressing the button and seeing anything.
    SNAPSHOT_MAX_WIDTH = 1280

    def clean_jpeg(self):
        """The newest frame with nothing drawn on it, as JPEG bytes."""
        with self._clean_lock:
            frame = None if self._clean_frame is None else self._clean_frame.copy()
        if frame is None:
            return None

        if frame.shape[1] > self.SNAPSHOT_MAX_WIDTH:
            height = int(frame.shape[0] * self.SNAPSHOT_MAX_WIDTH / frame.shape[1])
            frame = cv2.resize(frame, (self.SNAPSHOT_MAX_WIDTH, height),
                               interpolation=cv2.INTER_AREA)

        # Quality 88 regardless of JPEG_QUALITY: this still is the background
        # you place the counting line on, and a smeared doorway edge is pixels
        # of error in where the tripwire ends up.
        ok, buffer = cv2.imencode(".jpg", frame,
                                  [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        return buffer.tobytes() if ok else None

    # -- what the web layer reports ----------------------------------------
    def stats(self) -> dict:
        pipeline = self.pipeline
        counting = isinstance(pipeline, CountPipeline)
        line = pipeline.counter.line if counting else None

        return {
            "status": self.status,
            "error": self.error,
            "source": source_prompt.describe(self.source),
            "frame_width": self.frame_size[0],
            "frame_height": self.frame_size[1],
            "mode": "count" if counting else ("track" if pipeline else "starting"),
            "entry": pipeline.counter.entry_count if counting else 0,
            "exit": pipeline.counter.exit_count if counting else 0,
            "inside": pipeline.counter.occupancy if counting else 0,
            "tracking": len(pipeline.trails) if isinstance(pipeline, TrackPipeline) else 0,
            "unique_ids": len(pipeline.seen_ids) if isinstance(pipeline, TrackPipeline) else 0,
            "events": list(reversed(pipeline.counter.recent_events)) if counting else [],
            "fps": round(self.fps, 1),
            "lag_ms": round(self.lag_ms),
            "dropped": self.dropped,
            "drop_rate": round(self.drop_rate, 1),
            "track_point": config.TRACK_POINT,
            "line": line.to_dict(self.frame_size) if line else None,
            "dead_zone": line.effective_dead_zone if line else config.DEAD_ZONE_PX,
            "uptime": round(time.time() - self.started_at),
            "connected_for": (round(time.time() - self.connected_at)
                              if self.connected_at else 0),
        }


# ---------------------------------------------------------------------------
# The web layer
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder=str(SERVER_DIR / "templates"))
engine: Engine | None = None


def requires_auth(view):
    """Ask for a username and password, if AUTH_USER was set.

    Browsers handle HTTP Basic themselves - they pop up a login box and then
    repeat the credentials on every request after it, including the <img> tag
    pulling the video - which is why it is used here rather than a login form
    we would have to build and then carry through to the stream.

    Basic sends the password in base64, which is encoding and not encryption.
    Over a LAN among people you work with that is a reasonable trade; over the
    internet, put it behind an SSH tunnel or a TLS reverse proxy. The README
    says the same thing at more length.
    """

    @wraps(view)
    def wrapper(*args, **kwargs):
        if not settings.auth_user:
            return view(*args, **kwargs)

        auth = request.authorization
        # compare_digest rather than == so the comparison takes the same time
        # whatever the password is, which is the standard precaution against
        # someone measuring their way to it character by character.
        ok = (auth is not None
              and hmac.compare_digest(auth.username or "", settings.auth_user)
              and hmac.compare_digest(auth.password or "", settings.auth_password))
        if not ok:
            return Response(
                "Login required.", 401,
                {"WWW-Authenticate": 'Basic realm="Person counter"'})
        return view(*args, **kwargs)

    return wrapper


@app.get("/")
@requires_auth
def index():
    return render_template("index.html")


@app.get("/video")
@requires_auth
def video():
    """The live view, as multipart MJPEG.

    MJPEG is simply JPEG after JPEG down one never-ending HTTP response, each
    separated by a boundary marker. It is not efficient - every frame is a
    whole picture, with none of the between-frame compression H.264 does - but
    it needs no plugin, no JavaScript, no WebRTC signalling and no transcoding,
    and a 640x480 stream at 10 FPS costs roughly 2 Mbit/s, which a LAN will
    not notice. If you ever need it cheaper, lower JPEG_QUALITY or set
    STREAM_MAX_WIDTH.
    """

    def generate():
        last_seq = 0
        while True:
            jpeg, last_seq = engine.wait_for_jpeg(last_seq, timeout=10.0)
            if jpeg is None:
                # Ten seconds with no new frame. Send the last one again so
                # the connection stays alive through a camera reconnect
                # instead of the browser giving up on a dead socket.
                jpeg = engine.latest_jpeg()
                if jpeg is None:
                    continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                   + jpeg + b"\r\n")

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-store, no-cache, must-revalidate",
                             "Pragma": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.get("/snapshot.jpg")
@requires_auth
def snapshot():
    """One still frame.

    `clean=1` gives the frame with nothing drawn on it - what the line editor
    uses as its backdrop.

    It may be scaled down, and that is safe: the editor converts a click into
    camera pixels from the size the picture is being DISPLAYED at and the
    camera's real dimensions (which it reads from /api/stats), never from this
    image's own pixel count. So the picture's resolution affects only how
    sharp the backdrop looks, not where the line ends up.
    """
    want_clean = request.args.get("clean") == "1"
    jpeg = engine.clean_jpeg() if want_clean else engine.latest_jpeg()
    if jpeg is None:
        return jsonify(error="No frame yet."), 503
    return Response(jpeg, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/stats")
@requires_auth
def api_stats():
    return jsonify(engine.stats())


@app.post("/api/reset")
@requires_auth
def api_reset():
    engine.request_reset()
    return jsonify(ok=True)


@app.get("/api/line")
@requires_auth
def api_get_line():
    stats = engine.stats()
    return jsonify(line=stats["line"],
                   frame_width=stats["frame_width"],
                   frame_height=stats["frame_height"])


@app.post("/api/line")
@requires_auth
def api_set_line():
    """Save a counting line drawn in the browser.

    The coordinates arrive in the camera's own pixels - the page converts
    from wherever you clicked on screen before posting - so they go to the
    same lines/<camera>.json the desktop program reads, and either program
    can then be run against this camera and find the line already placed.
    """
    data = request.get_json(silent=True) or {}
    try:
        x1, y1, x2, y2 = (int(data[key]) for key in ("x1", "y1", "x2", "y2"))
        entry_side = int(data.get("entry_side", 1))
        dead_zone = int(data.get("dead_zone", config.DEAD_ZONE_PX))
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Expected x1, y1, x2, y2 as whole numbers."), 400

    if entry_side not in (1, -1):
        return jsonify(error="entry_side must be 1 or -1."), 400
    if not 0 <= dead_zone <= config.DEAD_ZONE_MAX_PX:
        return jsonify(error=f"dead_zone must be 0-{config.DEAD_ZONE_MAX_PX}."), 400

    line = CountingLine(x1, y1, x2, y2, entry_side, dead_zone)
    if line.length < config.MIN_LINE_LENGTH:
        return jsonify(error=f"That line is only {line.length:.0f} px long - "
                             f"drag out at least {config.MIN_LINE_LENGTH}."), 400

    path = line_path_for(engine.source)
    line.save(path, frame_size=engine.frame_size)
    engine.set_line(line)
    return jsonify(ok=True, saved_to=path, line=line.to_dict(engine.frame_size))


@app.delete("/api/line")
@requires_auth
def api_delete_line():
    """Forget this camera's line and fall back to plain tracking."""
    path = Path(line_path_for(engine.source))
    if path.exists():
        path.unlink()
    engine.set_line(None)
    return jsonify(ok=True)


@app.post("/api/track-point")
@requires_auth
def api_track_point():
    point = (request.get_json(silent=True) or {}).get("track_point", "")
    if point not in config.TRACK_POINT_CHOICES:
        return jsonify(error="track_point must be 'foot' or 'head'."), 400
    engine.set_track_point(point)
    return jsonify(ok=True, track_point=point)


@app.get("/api/cameras")
@requires_auth
def api_cameras():
    """The remembered cameras, by index and WITHOUT their passwords.

    The page offers these as a list to switch between; it never receives the
    URLs themselves, so a password cannot leak into a browser history, a
    screenshot or somebody's shoulder.
    """
    cameras = source_prompt.load_cameras()
    current = engine.source
    return jsonify(cameras=[
        {"index": i, "label": source_prompt.describe(url), "current": url == current}
        for i, url in enumerate(cameras)
    ])


@app.post("/api/source")
@requires_auth
def api_source():
    """Switch camera: either {"index": n} from the list, or {"source": "..."}."""
    data = request.get_json(silent=True) or {}

    if "index" in data:
        cameras = source_prompt.load_cameras()
        try:
            source = cameras[int(data["index"])]
        except (ValueError, TypeError, IndexError):
            return jsonify(error="No remembered camera with that number."), 400
    else:
        try:
            source = source_prompt.parse_source(str(data.get("source", "")))
        except ValueError as error:
            return jsonify(error=str(error)), 400

    source_prompt.remember_camera(source)
    engine.switch_source(source)
    return jsonify(ok=True, source=source_prompt.describe(source))


@app.get("/healthz")
def healthz():
    """Unauthenticated, and deliberately so: it is for a monitoring tool or a
    `docker healthcheck`, and it says nothing a passer-by could use."""
    alive = engine is not None and engine.status in ("running", "connecting",
                                                     "reconnecting", "loading model")
    return (jsonify(status=engine.status if engine else "down"), 200 if alive else 503)


# ---------------------------------------------------------------------------
# Start-up
# ---------------------------------------------------------------------------
def resolve_source() -> str:
    """The camera this server watches: CAMERA_URL, or the one in config.py."""
    if settings.camera_url:
        return source_prompt.parse_source(settings.camera_url)
    return config.build_rtsp_url()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        # basicConfig does nothing at all if the root logger already has a
        # handler, and several of the libraries imported above are entitled to
        # have added one. force makes ours the format regardless, so the
        # journal reads consistently.
        force=True,
    )

    global engine
    try:
        source = resolve_source()
    except ValueError as error:
        # A typo in CAMERA_URL. Say so here, in one readable paragraph,
        # rather than letting a traceback be the first thing in the journal.
        print(f"\nCAMERA_URL is not usable:\n  {error}\n")
        print("Fix it in server/.env, then start again.")
        return 1
    engine = Engine(source)

    print("=" * 68)
    print(" PERSON COUNTER - server")
    print("=" * 68)
    for line in server_config.describe():
        print("  " + line)
    print("=" * 68)

    engine.start()

    # systemd stops a service by sending SIGTERM. Without a handler Python
    # dies on the spot, leaving the camera's RTSP session open until it times
    # out - and the camera only allows a few at once, so the next start can
    # fail for a reason that has nothing to do with the next start.
    def shutdown(signum, _frame):
        log.info("Signal %s - shutting down.", signum)
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    try:
        # waitress is a small production-grade WSGI server: it handles many
        # clients on a thread pool and streams a generator's output as it is
        # produced, which is what an endless MJPEG response needs. Flask's own
        # development server can do this too, but prints a warning about it on
        # every start and is not built to be left running for weeks.
        from waitress import serve

        # waitress announces the address it has bound itself, so there is no
        # line of our own here - two near-identical "Serving on ..." lines in
        # the log look like a bug and send you looking for one.
        serve(app, host=settings.host, port=settings.port,
              threads=16, channel_timeout=120, ident="person-counter")
    except ImportError:
        log.warning("waitress is not installed - falling back to Flask's "
                    "development server. `pip install waitress` for the real one.")
        app.run(host=settings.host, port=settings.port,
                threaded=True, debug=False)
    finally:
        engine.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
