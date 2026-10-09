# Person Counter

Detect, track and count people crossing a line on an RTSP camera.

## Run it

```bash
python main.py
```

That is the only file you run. It starts by asking which camera to use, then
runs counting mode on it.

## Choosing a camera

Several cameras are in use, so the camera is chosen at start-up rather than
hard-coded:

```
====================================================================
 WHICH CAMERA?
====================================================================
 Format:  rtsp://<username>:<password>@<host>:<port>/<stream-path>
 Example: rtsp://admin:Admin%400192@172.20.100.138:554/cam/realmonitor?channel=1&subtype=1

 Cameras used before:
   [1] admin@172.20.100.138:554/cam/realmonitor?channel=1&subtype=1
   [2] admin@172.20.100.140:554/cam/realmonitor?channel=1&subtype=1

 Or:  a bare IP (10.0.0.5) to reuse the login from config.py
      a video file path (sample.mp4) to run on a recording
      [d] the camera in config.py (172.20.100.138)
      [q] quit
====================================================================
Camera >
```

What it accepts:

| Typed | Means |
|---|---|
| `rtsp://admin:pass@10.0.0.5:554/stream1` | that camera, exactly |
| `1` | the first camera in the list above |
| `10.0.0.5` or `10.0.0.5:554` | that host, with the username, password and stream path from `config.py` — the usual case when several identical cameras sit on one network |
| `sample.mp4` | a recorded clip instead of a camera |
| Enter, or `d` | the camera in `config.py` |
| `q` | quit |

Every camera you use is remembered in `cameras.json`, newest first, so the
second run against a camera is one keypress. **That file holds passwords in
clear text**, exactly as `config.py` always has — keep both out of any shared
repository.

To skip the question: `--source <url>` runs a specific camera, and
`--no-prompt` uses the one in `config.py`. Launched without a terminal — from a
scheduler, or with its input piped — there is nobody to ask, so it falls back to
`config.py` rather than blocking.

### A password containing `@`

RTSP separates the credentials from the host with `@`, so a password that
contains one (`Admin@0192`) splits the URL in the wrong place and FFmpeg
reports a DNS failure for a host name that is really half a password. Type it
either way — plain or percent-encoded as `Admin%400192` — and the prompt
encodes it correctly before use.

### Stream paths, by vendor

The path after the host is vendor-specific, and getting it wrong is the other
common failure:

| Camera | Path |
|---|---|
| Dahua / Amcrest | `/cam/realmonitor?channel=1&subtype=1` (`subtype=0` main, `1` sub) |
| Hikvision | `/Streaming/Channels/102` (`101` main, `102` sub) |
| ONVIF-generic | `/onvif1`, `/stream1`, `/live` |

Prefer the **sub** stream. YOLO shrinks the frame to `INFERENCE_SIZE` anyway,
so decoding 1080p just to throw the pixels away is wasted CPU.

`--mode view` is the quickest way to prove a new camera's URL is right: it
loads no model, so it either shows the picture straight away or fails straight
away.

| Command | What it does |
|---|---|
| `python main.py` | Counting — draw a line, then tally entries and exits (default) |
| `python main.py --mode view` | Just the video. No model is loaded, so it starts instantly — the quickest way to prove the camera works |
| `python main.py --mode detect` | Boxes only, no IDs |
| `python main.py --mode track` | Tracking only — IDs, no counting |
| `python main.py --redraw-line` | Draw a new line over the saved one |

Useful flags:

```bash
python main.py --source rtsp://admin:pass@10.0.0.5:554/stream1  # skip the prompt
python main.py --no-prompt           # skip the prompt, use config.py
python main.py --source sample.mp4   # run against a recorded clip
python main.py --imgsz 320           # faster, slightly less accurate
python main.py --conf 0.3            # override the confidence threshold
python main.py --track-point head    # count by the head instead of the feet
python main.py --dead-zone 20        # widen the ignore band, this run only
python main.py --line 0,300,640,300  # set the line without drawing it
python main.py --no-trails           # hide the motion trails
python main.py --help                # all options
```

In the window: **`q`** or **Esc** quits, **`t`** toggles motion trails,
**`r`** resets the counts.

## Choosing what gets counted — feet or head

A person is a box, but a tripwire test needs a single point, and the two
sensible candidates sit at opposite ends of the same box. They disagree by the
height of a human — at a doorway, that is the difference between counting
somebody as they arrive and counting them as they leave. So it is the second
question at start-up:

```
====================================================================
 WHICH POINT ON EACH PERSON SHOULD BE COUNTED?
====================================================================
   [1] FOOT  bottom of the box - where the person meets the floor  (default)
         the usual choice: a camera above head height sees bodies lean
         into the frame, so a head reaches the line too early

   [2] HEAD  top of the box - the crown of the head
         steadier where feet are hidden: crowds, near-overhead cameras,
         or a desk or counter across the bottom of the picture

   [Enter] keeps foot          [q] quit
====================================================================
```

**Foot** is right for the usual bullet camera, angled down from above head
height: bodies lean into that kind of frame, and the head is the end that
leans, so it reaches the line while the person is still short of the doorway.

**Head** wins wherever feet are not reliably visible. In a crowd the lower body
is occluded first, and a box whose bottom edge is guessed from behind somebody
else's shoulders jitters far more than its top edge does. Heads stay in view.

Measured on a synthetic walk-through, a line at `y=300` and a 200 px-tall
person moving down the frame:

| `TRACK_POINT` | counts when the feet reach | one count, not two |
|---|---|---|
| `foot` | `y = 310` | yes |
| `head` | `y = 510` | yes |

Both register exactly one crossing — the choice moves *when*, not *whether*.
The dot drawn on each person is whichever point is in use, and the count HUD
names it, so a run started on the wrong one is visible rather than discovered
later in the totals. `--track-point foot|head` answers the question in advance;
`--no-prompt` takes the answer from `config.TRACK_POINT`.

## Drawing the counting line

On the first run of `count` mode you draw the tripwire yourself, on live video
so you can watch people walk through the scene while you place it:

| | |
|---|---|
| **click and drag** | draw the line |
| **`f`** | flip which side counts as ENTRY — watch the green arrow |
| **`+`** / **`-`** | widen or narrow the dead zone — the two orange lines |
| **`r`** | start over |
| **Enter** or **`c`** | confirm and begin counting |
| **`q`** / **Esc** | cancel |

The line **and its dead-zone width** are saved together and reused on every
later run. Pass `--redraw-line` to replace them.

**Each camera keeps its own line**, under `lines/`, named after its host and
stream path:

```
lines/172-20-100-138-554-cam-realmonitor-channel-1-subtype-1.json
lines/sample-mp4.json
```

A line is a position in one camera's field of view and means nothing in
another's, so one shared file would quietly put camera A's doorway over camera
B's car park — and the saved-resolution check cannot catch that when the two
cameras are the same model. Switching cameras therefore asks for a new line;
switching back finds the old one.

A line drawn before this split (a single `counting_line.json` in the project
root) is adopted by the first camera that runs and matches its resolution, and
the old file is renamed to `counting_line.json.migrated` so no second camera
can pick it up. If it was adopted by the wrong camera, run again with
`--redraw-line`.

**Check the green arrow before confirming.** It points the way an ENTRY goes.
Which side of a line the maths calls "positive" depends on the direction you
happened to drag it, so there is no way for the code to guess which side of
your doorway is indoors — press `f` if the arrow points the wrong way.

## Setup

```bash
pip install -r requirements.txt
```

The model file (`yolov8n.pt`, ~6 MB) downloads by itself on first run and is
cached in this folder, so later runs work offline.

Then build the model the project actually runs on:

```bash
python export_openvino.py
```

This converts the PyTorch weights into **OpenVINO IR** — a `yolov8n.xml`
holding the network topology and a `yolov8n.bin` holding the weights, written
to `yolov8n_openvino_model/`. Nothing at run time loads the `.pt` any more;
it is kept only as the source for this conversion.

It is the same network and, checked frame by frame against the `.pt`, the same
detections — but run by Intel's inference-only runtime, which compiles the
graph for this CPU instead of interpreting it layer by layer through PyTorch.
At the default `INFERENCE_SIZE` that is **32 ms per frame against 55 ms**.

Two settings earn that margin, both in `config.py` and both documented there:
`OPENVINO_DEVICE` (pinned to `CPU`, because ultralytics otherwise drifts onto
the integrated GPU, which is ~5x slower here) and `TORCH_THREADS` (capped at
1, because PyTorch's idle threads spin-wait and starve OpenVINO's). Undo
either and the IR ends up slower than the `.pt` it replaced.

Re-run the export after changing `INFERENCE_SIZE` — IR is compiled for one
fixed input shape. The detector compares the two at start-up and refuses to
run on a mismatch rather than letting OpenVINO fail obscurely mid-stream.

## Layout

```
main.py              the entry point — argument parsing, nothing else
config.py            every tunable setting: fallback camera, model, thresholds, colours
cameras.json         cameras used before, offered at the prompt (holds passwords)
lines/               one drawn counting line per camera
requirements.txt
export_openvino.py   one-off: converts yolov8n.pt into the OpenVINO .xml/.bin
yolov8n.pt           downloaded automatically on first run; only the export reads it
yolov8n_openvino_model/
    yolov8n.xml      the network the project actually runs
    yolov8n.bin      its weights
    metadata.yaml    class names and the imgsz it was compiled for

person_counter/
    __init__.py      sets FFmpeg's RTSP-over-TCP option (must run before cv2 loads)
    camera.py        opening the stream and reading frames
    detector.py      YOLO detection + ByteTrack tracking
    drawing.py       boxes, labels, trails, the heads-up display
    counting_line.py the tripwire: geometry, sides, dead zone, save/load
    line_setup.py    the click-and-drag screen for drawing the line
    source_prompt.py the "which camera?" prompt and the remembered camera list
    track_point_prompt.py  the "feet or head?" prompt
    counter.py       entry/exit tallying — the crossing state machine
    app.py           the display loop, plus one pipeline class per mode

legacy/              the original stage-by-stage scripts, kept for reference.
                     Safe to delete.
```

The shape to keep in mind: `app.py` holds **one** display loop, and each mode is
a small pipeline class with a `process(frame)` method. A new mode is a new
class, not another copy of the loop.

## How counting works

Three ideas do the work, and each one exists to defeat a specific way naive
counting goes wrong.

### 1. The signed distance

A line through points A and B splits the plane. For any point P, the 2D cross
product `(B−A) × (P−A)` is positive on one side, negative on the other, zero on
the line. Divide it by the line's length and it becomes the perpendicular
**distance in pixels**, carrying a sign.

One number answers both questions: the sign says *which side*, the magnitude
says *how far*.

### 2. The dead zone — `DEAD_ZONE_PX`

The obvious implementation compares each frame's side with the previous frame's
and counts whenever they differ. It is broken, and it breaks exactly where it
matters — at the line.

ByteTrack is resilient, but the YOLO box still breathes a few pixels per frame:
an arm swings, a leg is occluded, the box edge shifts, and the foot point moves
with it. Someone pausing on the tripwire flickers across it — left, right, left,
right — and a frame-to-frame comparison books a count for every flicker.

So sides are **sticky**, with a band of "no opinion" between them:

```
|<--- side -1 --->|<-- dead zone -->|<--- side +1 --->|
                    no opinion here
```

- Outside the band, the side is recorded as **confirmed**.
- Inside the band, nothing changes at all — the confirmed side stays put.
- A count happens only when a new *confirmed* side differs from the stored one.

Jitter inside the band is therefore structurally incapable of producing a count.
To register, a person must travel **completely clear** of the band on the far
side — which is what walking through a doorway does, and what standing on the
threshold does not.

Measured on a synthetic track standing on the line for 300 frames with ±8 px of
jitter:

| | Phantom crossings |
|---|---|
| `DEAD_ZONE_PX = 0` | **132** |
| `DEAD_ZONE_PX = 12` | **0** |

### 3. The segment bound

`signed_distance` treats the line as infinite — it will happily report which
side of the *extended* line a point is on, however far past the drawn ends it
sits. For a tripwire that is wrong: an operator who draws a line across a
doorway means that doorway, not its invisible continuation across the rest of
the scene. Without this check, somebody walking past at the far side of the
frame crosses the extension and books a count for a door they never went near.

So `covers()` projects the point onto the line and requires it to land between
the two endpoints. Off either end, the answer is "no opinion".

### Plus two smaller guards

- **A point at the edge of the box, never the centroid.** The feet by default,
  the head if you chose it at start-up — see *Choosing what gets counted*. On a
  bullet camera mounted above head height the body leans into the frame, so the
  centroid reaches the line while the person is still short of the doorway.
- **`MIN_TRACK_AGE_FRAMES`** — a track only a frame or two old may be a
  flickering false positive, or a real person whose ID was reissued mid-stride
  after an occlusion. Either way its first "crossing" is an artefact of being
  born. Its side is still recorded during that time, so it starts measuring from
  the right place once it matures.

## Tuning

Everything lives in `config.py`.

**The dead zone** — half-width of the ignore band, in pixels, drawn on screen
as two thin orange lines. Too narrow and counts climb while someone stands on
the line; too wide and it swallows the doorway, missing people who genuinely
cross near its edge.

You set it **where you draw the line**, on live video, with `+` and `-`:

```
Drag again to redraw  |  'f' flips the ENTRY arrow
dead zone 18 px  ('+' / '-' to resize)
Enter or 'c' to confirm  |  'r' clears  |  Esc cancels
```

The orange band resizes as you press, so you are judging the real width against
the real doorway rather than guessing a number. Widen it until it comfortably
covers the shuffling somebody does while standing on the threshold, and no
wider. It is **saved with the line, per camera** — which is the right place for
it, because the correct width depends on how large people are in *that*
picture. A line across a distant corridor and one across a doorway two metres
from the lens need very different bands, and nothing else in the configuration
can tell them apart.

Three places set it, narrowest scope first:

| | Scope |
|---|---|
| `--dead-zone 20` | this run only — overrides the saved width without replacing it, for trying a number |
| `+` / `-` in the setup screen | saved with that camera's line |
| `config.DEAD_ZONE_PX` (12) | the starting width for any line never adjusted |

A line file written before this existed simply has no saved width, so it keeps
following `config.DEAD_ZONE_PX` — including if you retune that later.

Measured on a loiterer wobbling ±8 px across the line for 40 frames, and on a
genuine walk-through:

| Dead zone | Phantom counts from the wobble | Real crossing still counted |
|---|---|---|
| 0 px | 18 entries + 18 exits | yes |
| 4 px | 18 entries + 18 exits | yes |
| 12 px | **0** | yes |
| 20 px | **0** | yes |
| 60 px | **0** | yes |

The band has to be wider than the jitter to do anything at all — 4 px against
an 8 px wobble changes nothing — and widening it well past that still does not
cost a real crossing. `config.DEAD_ZONE_MAX_PX` (200) caps the adjustment; a
band wider than that is a line drawn in the wrong place, not a dead zone set
too narrow.

**`INFERENCE_SIZE`** — the square each frame is resized to before the network
sees it. Cost scales with roughly the square of this. Measured on this machine
(CPU-only, `yolov8n`), with the pre-OpenVINO `.pt` figures for comparison:

| `INFERENCE_SIZE` | OpenVINO `.xml` | PyTorch `.pt` |
|---|---|---|
| 320 | 16 ms (~63 FPS) | ~21 FPS |
| 480 (default) | 32 ms (~32 FPS) | ~13 FPS |
| 640 | 56 ms (~18 FPS) | ~9 FPS |

Changing this requires re-running `python export_openvino.py`, since the IR is
compiled for one fixed input shape.

**`DETECT_CONF_THRESHOLD`** (0.4) — how confident YOLO must be to draw a box.
Lower catches partly hidden people but invents false boxes; higher misses
distant ones.

**`TRACK_CONF_THRESHOLD`** (0.15) — deliberately *lower*, and this is the most
important line in the config. Ultralytics applies `conf` **before** the tracker
sees anything, so a high threshold throws away the weak detections that
ByteTrack's second-round matching exists to recover. Measured on one identical
80-frame clip of this camera, same four people:

| Threshold | Unique IDs | Stable across the clip | Fragmented |
|---|---|---|---|
| `0.15` | 4 | **4** | 0 |
| `0.40` | 4 | 1 | 3 (died after 16, 7 and 9 frames) |

Every one of those rebirths would become a double count.

If people vanish behind a pillar and come back wearing a new ID, raise
`track_buffer` in the tracker config (see the comments in `config.py`).

## Measured performance

On this machine — CPU-only (`torch 2.5.1+cpu`), camera sub-stream at 640×480:

| Mode | FPS |
|---|---|
| view | 40+ (display-bound) |
| detect | ~11 |
| track | ~9–11 |
| count | ~13 |

Tracking is effectively free: on identical frames, detection and tracking both
ran 11.2 FPS, and counting adds only arithmetic. Frame reads cost 0.8 ms against
~90 ms of inference, so the pipeline is **inference-bound** — the camera and
network are nowhere near the limit. `INFERENCE_SIZE` is the knob that moves the
number.

## Live capture: why there are two readers

An RTSP camera pushes frames on its own schedule and will not wait for you.
Measured on this camera: a frame every **38 ms (~25 FPS)**, against **~90 ms**
for one YOLO pass (~11 FPS). Read them sequentially and you consume less than
half of what arrives — and the surplus does not evaporate, it queues inside
FFmpeg.

That queue is bounded, which matters for understanding the symptom. Measured by
draining it, the sequential pipeline settles about **36 frames — 1.4 seconds —
behind real time**. It does not grow past that, because once the buffer is full
frames are discarded *at the source*, outside your control. So the lag plateaus
and the camera starts dropping frames on you: late video plus missing frames,
which is exactly what a slow consumer looks like from the outside.

`CAP_PROP_BUFFERSIZE` does not fix this. It is a hint, honoured inconsistently,
and on FFmpeg/RTSP it generally is not.

So live streams get `ThreadedVideoStream`: a background thread reads at the
camera's pace into a **single slot** holding only the newest frame. Inference
takes from that slot whenever it is free, and everything that arrived while it
was busy is discarded rather than queued.

**Dropping frames is the point.** For counting people, a frame from two seconds
ago has no value, and the tracker does better with a current frame than a stale
one.

Measured over 30 s on the live camera:

| | FPS | Camera sent | Dropped | Backlog | Behind real time |
|---|---|---|---|---|---|
| sequential | 8.0–9.0 | ~766 / 30 s | by FFmpeg, uncontrolled | 36 frames | **1.4 s** |
| threaded | 5.6–8.3 | 766 / 30 s | 517 (67%), deliberate | 2 frames | **0.1 s** |

Read that carefully, because the FPS column is not the win. Threaded is
*slower* in raw throughput — measured between 7% and 30% slower across runs,
since the ingestion thread decodes every arriving frame and competes for CPU
with inference on a machine with no GPU.

**Throughput is not what this buys. Currency is.** The frame the model sees is
14× more recent, and the frames that get dropped are dropped by you rather than
by a full buffer. If you want the FPS number itself higher, that is
`INFERENCE_SIZE`, not threading.

Two implementation details that are easy to get wrong:

- **The ingestion thread must copy each frame.** OpenCV recycles a small pool of
  internal buffers — measured here, consecutive `read()` calls alternate between
  just *two* addresses. Without the copy, a frame handed to the inference thread
  is overwritten underneath it two frames later. It costs 0.03 ms.
- **`read()` must block for a frame *newer* than the last one served.** Without
  that sequence check, a consumer faster than the camera is handed the same
  frame repeatedly and wastes whole inference passes on it.

`grab()`-without-decode is not worth it here: `grab()` costs 38.06 ms against
`read()`'s 39.68 ms, because that time is spent *waiting on the camera*, not
decoding.

**Files always read sequentially.** A file has no clock of its own, so an
ingestion thread would race to the end while inference was still in the opening
seconds, discarding nearly everything. `open_stream()` routes by URL scheme;
`--no-threaded` forces sequential for debugging.

## Validation

Two complementary checks, because they catch opposite failures.

**Synthetic** (32 assertions, known answers): clean crossings both ways,
loitering on the line, hesitate-then-retreat vs hesitate-then-commit, segment
bounds, track age, simultaneous crossings, state cleanup. The headline case is a
track standing on the line for 300 frames with ±8 px of jitter — **132** phantom
crossings with `DEAD_ZONE_PX = 0`, **0** with it at 12.

**Real footage** (`sample.mp4`, 2560×1440, 592 frames), vertical line at x=550:

| Dead zone | Unique IDs | ENTRY | EXIT | Total |
|---|---|---|---|---|
| 0 px | 13 | 1 | 4 | 5 |
| 12 px | 13 | 1 | 4 | 5 |
| 48 px | 13 | 1 | 4 | 5 |

Ground truth from the trajectories: ID 37 crosses twice (out and back), IDs 23,
4 and 2 cross once each, all in the same direction — 4 one way, 1 the other.
The counter's 1 entry / 4 exits is exactly correct.

The dead zone changes nothing here, and that is the point: nobody in this clip
loiters on the line, so there is no jitter to suppress. It shows the other half
of the property — **the dead zone does not cost you real crossings**, even at
48 px on a 1440p frame.

A caution if you benchmark several passes in one process: setting
`model.predictor = None` does *not* reset ByteTrack state. IDs carry over and
the counts come out wrong. Run each pass in its own process.

## Pipeline status

- [x] **Stage 1 — Video → YOLO → person boxes** (`--mode detect`)
- [x] **Stage 2 — ByteTrack → stable IDs** (`--mode track`)
- [x] **Stage 3 — line crossing → `entry_count` / `exit_count`** (`--mode count`)

Worth knowing about the numbers it reports:

- **`Inside`** (entries − exits) assumes the space was empty at startup. Anyone
  already inside was never counted in, so it can legitimately go negative. Press
  `r` to reset when the space is known to be empty.
- Counting is only as good as the tracking under it. If `Unique IDs` in track
  mode climbs much faster than people actually walk past, fix that first — every
  spurious ID is a potential double count.
