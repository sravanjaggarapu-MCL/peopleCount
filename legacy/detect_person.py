"""
detect_person.py
================

STAGE 1 of the person-counting pipeline:

    RTSP video  ->  YOLO  ->  person bounding boxes  ->  display

This script only DETECTS people and draws a box around each one. It does not
yet track them across frames and it does not yet count entries/exits - those
are stage 2 (tracking) and stage 3 (line crossing). Detection is deliberately
built and verified on its own first, because every later stage consumes its
output: if the boxes are wrong, the counts can never be right.

Usage
-----
    python detect_person.py                 # live camera, default settings
    python detect_person.py "rtsp://..."    # a different stream or a video file
    python detect_person.py sample.mp4      # any file OpenCV can read

Press 'q' or Esc in the window to quit.

Requirements
------------
    pip install ultralytics opencv-python

The model file (yolov8n.pt, ~6 MB) downloads automatically on first run and is
cached next to this script, so later runs work offline.
"""

import sys
import time

# Import our existing RTSP helper FIRST. Two things happen as a side effect of
# this import, and the order matters:
#   1. It sets OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp, which FFmpeg
#      only reads at cv2 load time - so it must run before cv2 is imported.
#   2. It builds RTSP_URL with the password correctly percent-encoded.
# Its main() is guarded by `if __name__ == "__main__"`, so importing it does not
# open any window.
from rtsp_view import RTSP_URL

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------

# "n" = nano, the smallest YOLOv8 model (~6 MB, ~3.2M parameters). It is the
# right starting point on a CPU-only machine: fast enough for live video, and
# already trained on COCO, which includes "person" as class 0. Larger options,
# in increasing accuracy and cost: yolov8s.pt, yolov8m.pt, yolov8l.pt.
# Swap the filename here if you later move to a GPU box.
MODEL_PATH = "yolov8n.pt"

# COCO class index for "person". YOLO was trained on 80 classes (cars, dogs,
# chairs...); passing this to the model makes it discard everything else inside
# the network's post-processing, which is both faster and simpler than filtering
# the results ourselves afterwards.
PERSON_CLASS_ID = 0

# Minimum confidence for a detection to count. Think of it as the dial between
# two kinds of mistake:
#   lower (0.25) -> catches partly hidden people, but invents some false boxes
#   higher (0.6) -> only confident detections, but misses distant/occluded people
# 0.4 is a sane starting point for an indoor bullet camera. Tune it by watching
# the live window: if you see boxes on empty floor, raise it; if real people go
# unboxed, lower it.
CONFIDENCE_THRESHOLD = 0.4

# The square size each frame is letterboxed to before entering the network.
# Cost grows roughly with the square of this number, so 320 is ~4x cheaper than
# 640. Our sub-stream is only 640x480 and this machine has no GPU, so 480 is a
# good balance. Must be a multiple of 32 (the network's stride).
INFERENCE_SIZE = 480

# Draw settings.
BOX_COLOR = (0, 255, 0)        # BGR, not RGB - green
TEXT_COLOR = (0, 0, 0)         # black text on a filled green label
FONT = cv2.FONT_HERSHEY_SIMPLEX

WINDOW_NAME = "Person detection - stage 1"


def draw_detection(frame, x1, y1, x2, y2, confidence):
    """Draw one person's box plus a small filled label above it.

    Coordinates are pixel positions in the frame: (x1, y1) is the top-left
    corner of the box and (x2, y2) the bottom-right. Note that in image
    coordinates y grows DOWNWARDS, so y1 < y2 means y1 is the higher edge.
    """
    # The box itself. Thickness 2 stays visible without swallowing small people.
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

    label = f"person {confidence:.2f}"

    # Measure the text so the label background fits it exactly. getTextSize
    # returns ((width, height), baseline); the baseline is the few pixels that
    # descenders like 'p' hang below the line, which we add as padding.
    (text_w, text_h), baseline = cv2.getTextSize(label, FONT, 0.5, 1)

    # Put the label above the box, unless the box is already at the top of the
    # frame - in that case tuck the label just inside the box so it stays visible.
    label_bottom = y1 if y1 - text_h - baseline > 0 else y1 + text_h + baseline

    # Filled rectangle behind the text (thickness -1 means "fill"), so the label
    # stays readable over a busy background.
    cv2.rectangle(
        frame,
        (x1, label_bottom - text_h - baseline),
        (x1 + text_w, label_bottom),
        BOX_COLOR,
        -1,
    )
    cv2.putText(frame, label, (x1, label_bottom - baseline), FONT, 0.5, TEXT_COLOR, 1)


def main() -> int:
    """Run the detect-and-display loop. Returns a process exit code."""

    # First CLI argument wins; otherwise use the camera from rtsp_view.py.
    # OpenCV treats a plain path exactly like a URL, so "sample.mp4" also works -
    # handy for testing the detector on a recorded clip without the camera.
    source = sys.argv[1] if len(sys.argv) > 1 else RTSP_URL

    # -----------------------------------------------------------------------
    # Load the model
    # -----------------------------------------------------------------------
    # On the very first run this downloads yolov8n.pt (~6 MB) from GitHub and
    # caches it in this folder. Afterwards it loads from disk, in well under a
    # second. The weights are pretrained on COCO - we are NOT training anything.
    print(f"Loading {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)
    print(f"Model loaded. Detecting class {PERSON_CLASS_ID} ('{model.names[PERSON_CLASS_ID]}') only.")

    # Warm the model up on a dummy frame. The very first predict() call is much
    # slower than the rest - PyTorch allocates buffers, picks algorithms and
    # fills caches on the way through. Doing that here, before the window opens,
    # keeps it out of the display loop where it would otherwise look like the
    # video froze for a couple of seconds. Measured here: first call ~2-3 s,
    # every call after it ~0.08 s.
    model.predict(
        np.zeros((INFERENCE_SIZE, INFERENCE_SIZE, 3), dtype=np.uint8),
        imgsz=INFERENCE_SIZE,
        verbose=False,
    )

    # -----------------------------------------------------------------------
    # Open the video source
    # -----------------------------------------------------------------------
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # newest frame only - see rtsp_view.py

    if not cap.isOpened():
        print("Could not open the video source. Run rtsp_view.py first to")
        print("confirm the camera itself is reachable.")
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source opened: {width}x{height}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)
    print("Running. Press 'q' or Esc to quit.")

    # Rolling FPS measurement. We time how long a batch of frames takes rather
    # than timing each one, because a single frame's duration is too noisy to
    # read off the screen.
    fps = 0.0
    frames_this_window = 0
    window_started_at = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed - stream ended or dropped.")
            break

        # -------------------------------------------------------------------
        # Inference: the one line that turns pixels into detections
        # -------------------------------------------------------------------
        #   classes=[0]  -> only look for people
        #   conf=...     -> drop anything below our confidence threshold
        #   imgsz=...    -> resize input; smaller is faster
        #   verbose=False-> silence ultralytics' per-frame console spam
        # The result is a list with one entry per input image; we passed a
        # single frame, so results[0] is ours.
        results = model.predict(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=CONFIDENCE_THRESHOLD,
            imgsz=INFERENCE_SIZE,
            verbose=False,
        )
        boxes = results[0].boxes  # a Boxes object; len() gives the detection count

        # -------------------------------------------------------------------
        # Draw every detection
        # -------------------------------------------------------------------
        for box in boxes:
            # .xyxy holds corner coordinates as a tensor of shape (1, 4), in
            # pixels ON THE ORIGINAL FRAME - ultralytics already undoes the
            # letterboxing from imgsz for us, so no rescaling is needed here.
            # [0] unwraps the batch dimension; .tolist() moves it off the tensor
            # into plain Python floats; int() because pixels must be whole.
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            draw_detection(frame, x1, y1, x2, y2, confidence)

        # -------------------------------------------------------------------
        # Heads-up display
        # -------------------------------------------------------------------
        # Live person count. This is a PER-FRAME headcount, not an entry/exit
        # tally - without tracking, the same person appearing in the next frame
        # is an unrelated detection as far as this script knows. Turning these
        # boxes into stable identities is exactly what stage 2 adds.
        cv2.putText(frame, f"People in frame: {len(boxes)}", (10, 30),
                    FONT, 0.8, BOX_COLOR, 2)
        # FPS is the honest measure of whether this pipeline keeps up with the
        # camera. If it sits well below the camera's frame rate, lower
        # INFERENCE_SIZE, or process every Nth frame.
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), FONT, 0.8, BOX_COLOR, 2)

        cv2.imshow(WINDOW_NAME, frame)

        # Recompute FPS roughly once per second.
        frames_this_window += 1
        elapsed = time.time() - window_started_at
        if elapsed >= 1.0:
            fps = frames_this_window / elapsed
            frames_this_window = 0
            window_started_at = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # 'q' or Esc
            print("Quit requested.")
            break
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)
