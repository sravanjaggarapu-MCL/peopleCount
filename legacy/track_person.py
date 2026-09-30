"""
track_person.py
===============

STAGE 2 of the person-counting pipeline:

    RTSP video  ->  YOLO  ->  ByteTrack  ->  person boxes WITH STABLE IDs

What stage 1 (detect_person.py) could not do
--------------------------------------------
Detection is memoryless. Each frame is analysed from scratch, so the script had
no way to know that the box in frame 12 is the same human as the box in frame
11. That is fine for a headcount, but useless for entry/exit counting: to say
"this person crossed the line downwards" you must first be able to say "this
person" at all, across time.

Tracking supplies exactly that: every person gets an integer ID that stays with
them for as long as they remain in view.

How ByteTrack works, briefly
----------------------------
Each frame, the tracker predicts where every existing track should have moved
(a Kalman filter assuming roughly constant velocity), then matches those
predictions against this frame's detections by box overlap (IoU).

Its namesake trick is the two-round match. Most trackers throw away
low-confidence detections outright. ByteTrack does not:

  Round 1 - match existing tracks against HIGH confidence detections.
  Round 2 - take the tracks still unmatched and try them against the LOW
            confidence leftovers.

A person half-hidden behind someone else produces exactly such a weak
detection. Round 2 is what lets their track survive the occlusion instead of
dying and being reborn with a fresh ID - and every such ID change would
otherwise become a double count downstream.

Usage
-----
    python track_person.py                  # live camera
    python track_person.py sample.mp4       # a recorded clip

Press 'q' or Esc to quit. Press 't' to toggle the motion trails.
"""

import sys
import time
from collections import defaultdict, deque

# As in detect_person.py: importing this first sets the FFmpeg TCP option
# before cv2 loads, and builds the URL with the password encoded.
from rtsp_view import RTSP_URL

import cv2
import numpy as np
from ultralytics import YOLO

# Reuse stage 1's settings so both scripts stay in step - there is one place to
# change the model or the inference size, not two.
from detect_person import MODEL_PATH, PERSON_CLASS_ID, INFERENCE_SIZE, FONT

# ---------------------------------------------------------------------------
# Tracker settings
# ---------------------------------------------------------------------------

# The tracker config shipped inside the ultralytics package. To tune it, copy
# that file next to this script and point this at your copy. The settings worth
# knowing:
#
#   track_high_thresh: 0.25  confidence for round-1 matching (see above)
#   track_low_thresh:  0.1   floor for round-2 matching
#   new_track_thresh:  0.25  confidence needed to BIRTH a brand new ID
#   track_buffer:      30    frames a lost track is kept alive before deletion.
#                            At our ~13 FPS that is about 2.3 seconds of
#                            tolerance - raise it if people vanish behind a
#                            pillar for longer and come back with a new ID.
#   match_thresh:      0.8   how much box overlap counts as "the same person"
TRACKER_CONFIG = "bytetrack.yaml"

# NOTE - this is deliberately LOWER than stage 1's 0.4, and the reason matters.
#
# Ultralytics applies `conf` BEFORE handing detections to the tracker. Set it to
# 0.4 and every weaker detection is discarded before ByteTrack ever sees it,
# which silently disables the round-2 recovery described above - you would be
# paying for ByteTrack and getting a plain IoU tracker.
#
# So we pass a permissive threshold here and let the tracker's own
# track_high_thresh/track_low_thresh do the filtering they were designed to do.
# Weak detections that never associate with a track simply die out.
TRACK_CONF_THRESHOLD = 0.15

# How many past centre points to remember per person, for drawing the motion
# trail. Purely a visualisation here - but this same history is what stage 3
# reads to decide which way someone crossed the counting line.
TRAIL_LENGTH = 32

WINDOW_NAME = "Person tracking (ByteTrack) - stage 2"


def color_for_id(track_id: int):
    """Give each track ID its own stable, well-separated colour.

    Deterministic, so a person keeps one colour for their whole time on screen
    and you can follow them by eye. We spread hues using a large step (47) that
    shares no factor with 180, so consecutive IDs land far apart on the colour
    wheel instead of looking like near-identical shades.

    OpenCV's hue channel runs 0-179, not 0-359 - it has to fit in a byte.
    """
    hue = (track_id * 47) % 180
    hsv = np.uint8([[[hue, 200, 255]]])          # vivid, bright
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])  # cv2 wants plain Python ints


def draw_track(frame, x1, y1, x2, y2, track_id, confidence, color):
    """Draw one tracked person: their box, an ID label, and their foot point."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"ID {track_id}  {confidence:.2f}"
    (text_w, text_h), baseline = cv2.getTextSize(label, FONT, 0.5, 1)
    label_bottom = y1 if y1 - text_h - baseline > 0 else y1 + text_h + baseline

    cv2.rectangle(
        frame,
        (x1, label_bottom - text_h - baseline),
        (x1 + text_w, label_bottom),
        color,
        -1,  # filled
    )
    cv2.putText(frame, label, (x1, label_bottom - baseline), FONT, 0.5, (0, 0, 0), 1)

    # The foot point: bottom-centre of the box, where the person meets the floor.
    #
    # Stage 3 will test THIS point against the counting line rather than the box
    # centre. On a bullet camera mounted above head height, the body leans into
    # the frame, so the centroid reaches the line noticeably earlier than the
    # feet do - counting on the centroid registers the crossing while the person
    # is still short of the doorway. The feet are where the person actually is.
    foot_x, foot_y = (x1 + x2) // 2, y2
    cv2.circle(frame, (foot_x, foot_y), 4, color, -1)


def main() -> int:
    """Run the track-and-display loop. Returns a process exit code."""

    source = sys.argv[1] if len(sys.argv) > 1 else RTSP_URL

    print(f"Loading {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)

    # Warm-up, same reasoning as stage 1: keep PyTorch's slow first call out of
    # the display loop, where it looks like a freeze.
    model.predict(
        np.zeros((INFERENCE_SIZE, INFERENCE_SIZE, 3), dtype=np.uint8),
        imgsz=INFERENCE_SIZE,
        verbose=False,
    )
    print(f"Model ready. Tracker: {TRACKER_CONFIG}")

    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("Could not open the video source. Check rtsp_view.py works first.")
        return 1

    print(f"Source opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}"
          f"x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)
    print("Running. 'q'/Esc quits, 't' toggles trails.")

    # Per-ID history of foot points. defaultdict means a never-seen ID creates
    # its own deque on first access, with no "if id not in dict" dance.
    # maxlen makes the deque self-trimming: appending to a full one drops the
    # oldest point automatically, so memory cannot creep up over hours of video.
    trails = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))

    # Every ID ever issued. Its size is the total number of distinct people the
    # tracker believes it has seen - a useful sanity check: if this climbs far
    # faster than people actually walk past, IDs are being churned and stage 3
    # would over-count. Raising track_buffer usually settles it.
    seen_ids = set()

    show_trails = True
    fps = 0.0
    frames_this_window = 0
    window_started_at = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed - stream ended or dropped.")
            break

        # -------------------------------------------------------------------
        # Detect AND track, in one call
        # -------------------------------------------------------------------
        # track() runs the detector and then feeds its boxes to the tracker.
        #
        # persist=True is the critical argument: it tells ultralytics that this
        # frame continues the same video as the last call, so tracker state
        # carries over. Leave it out and the tracker is rebuilt from scratch on
        # every frame - every person would be "ID 1", newly born, forever.
        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            classes=[PERSON_CLASS_ID],
            conf=TRACK_CONF_THRESHOLD,
            imgsz=INFERENCE_SIZE,
            verbose=False,
        )
        boxes = results[0].boxes

        # boxes.id is None when the tracker produced no confirmed tracks at all
        # (an empty scene, or the first frames while tracks are still being
        # confirmed). Guard for it - indexing None would crash the loop.
        live_ids = []
        if boxes is not None and boxes.id is not None:
            # Pull the three parallel arrays off the GPU/tensor world into plain
            # NumPy in one go - cheaper than touching each box individually.
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.int().cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), track_id, conf in zip(xyxy, ids, confs):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                track_id = int(track_id)

                live_ids.append(track_id)
                seen_ids.add(track_id)
                color = color_for_id(track_id)

                draw_track(frame, x1, y1, x2, y2, track_id, float(conf), color)

                # Record this frame's foot point for the trail / stage 3.
                trails[track_id].append(((x1 + x2) // 2, y2))

                if show_trails:
                    # Join the remembered points into a polyline. Older points
                    # are thinner, giving the trail a natural fade so you can
                    # read the direction of travel at a glance - which is the
                    # whole question stage 3 has to answer.
                    points = trails[track_id]
                    for i in range(1, len(points)):
                        thickness = max(1, int(3 * i / len(points)))
                        cv2.line(frame, points[i - 1], points[i], color, thickness)

        # Forget the history of people who have left the scene, so `trails` does
        # not grow without bound across a long run. We keep `seen_ids` (just
        # integers) but drop the point lists, which are the bulky part.
        for dead_id in set(trails) - set(live_ids):
            del trails[dead_id]

        # -------------------------------------------------------------------
        # Heads-up display
        # -------------------------------------------------------------------
        cv2.putText(frame, f"Tracking now: {len(live_ids)}", (10, 30),
                    FONT, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Unique IDs so far: {len(seen_ids)}", (10, 60),
                    FONT, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90), FONT, 0.8, (0, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, frame)

        frames_this_window += 1
        elapsed = time.time() - window_started_at
        if elapsed >= 1.0:
            fps = frames_this_window / elapsed
            frames_this_window = 0
            window_started_at = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            print("Quit requested.")
            break
        if key == ord("t"):
            show_trails = not show_trails
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {len(seen_ids)} unique people tracked this session.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)
