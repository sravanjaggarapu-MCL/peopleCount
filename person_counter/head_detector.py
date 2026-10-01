"""
head_detector.py
================

Finds HEADS in a frame. A new, independent capability - it does not touch the
existing person detector, the tracker, or counting.

Purpose / why it exists
-----------------------
The existing model (yolov8n.pt, trained on COCO) knows "person" but has no
"head" class, so it cannot tell us where anyone's head is. This module runs a
SEPARATE model to get head boxes, which person_head_association.py then attaches
to the tracked people.

It solves one problem only: "where are the heads in this frame?" It knows
nothing about track IDs, counting lines or which person a head belongs to.

Two interchangeable backends (chosen by head_config "HEAD_BACKEND")
-------------------------------------------------------------------
    "pose"  Official Ultralytics pose model (yolov8n-pose.pt, auto-download).
            It predicts 17 body keypoints per person; the first five are the
            face: nose, left eye, right eye, left ear, right ear. We draw a box
            around the confident ones to get a head box.
            + works out of the box, no extra files
            - needs a visible face/ears; weak for people seen from behind or
              from directly overhead

    "yolo"  Any Ultralytics YOLO DETECTION model trained with a "head" class
            (e.g. on SCUT-HEAD / CrowdHuman). You supply the weights.
            + sees the back of a head and top-down views
            - you must obtain or train the model

Both return the same HeadDetection objects, so nothing downstream cares which
backend produced them. Adding a third backend means adding one method here.

Input / output
--------------
Input:  one BGR frame (numpy array), ideally a CLEAN copy with no overlays
        drawn on it - boxes and labels drawn by the person pipeline would
        otherwise be fed into the head model.
Output: list[HeadDetection] in original frame pixel coordinates.

Where it fits in the pipeline
-----------------------------
    frame ──► PersonDetector.track() ──► counting        (existing, untouched)
      └─(copy)─► HeadDetector.detect() ──► PersonHeadAssociator.associate()

Communicates with: head_config.py (settings), head_stage.py (its only caller).
It loads its OWN YOLO instance, so the person model and its ByteTrack state are
never shared with or disturbed by head inference.

Future use
----------
Head boxes are the natural input for face recognition (crop the head),
head-based counting on overhead cameras, crowd density, and head-pose /
attention estimation.
"""

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

import config
from . import head_config


# COCO keypoint indices of the face, in the order the pose model outputs them.
# Only these five are used; the other twelve (shoulders, elbows, ...) are body.
_FACE_KEYPOINTS = [0, 1, 2, 3, 4]   # nose, l_eye, r_eye, l_ear, r_ear
_EYE_KEYPOINTS = [1, 2]             # left eye, right eye
_EAR_KEYPOINTS = [3, 4]             # left ear, right ear
# Shoulders are body keypoints, but they give an independent head-size check:
# the eye-to-shoulder drop is about the same as the eye-to-crown distance.
_SHOULDER_KEYPOINTS = [5, 6]        # left shoulder, right shoulder


@dataclass
class HeadDetection:
    """One head found in one frame.

    Responsibility: a plain-data description of a head box, deliberately
    mirroring detector.Detection so both kinds of box are handled the same
    way. Holds no model tensors, so callers never need to know what
    ultralytics returned.

    State:  box corners in frame pixels, a confidence, and which backend made it.
    Output: convenience geometry (center, size, head_point) used by the
            association logic and by drawing.
    """

    x1: int          # left edge, pixels
    y1: int          # top edge (y grows DOWNWARDS in image coordinates)
    x2: int          # right edge
    y2: int          # bottom edge
    confidence: float
    source: str      # "pose" or "yolo" - kept for debugging / future logging
    # Body box of the pose skeleton this head was derived from ("pose" backend
    # only; None for "yolo"). It tells association WHICH body the head grew
    # out of - far more reliable than position alone when two people overlap.
    body_box: tuple[int, int, int, int] | None = None

    @property
    def center(self) -> tuple[int, int]:
        """Middle of the head box. This is what association compares against
        person boxes, because a single point is easy to reason about."""
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    @property
    def head_point(self) -> tuple[int, int]:
        """The single point that represents this head to the rest of the system.

        Defined as the box centre. It is a separate name from `center` on
        purpose: if a future feature wants a different anchor (e.g. top of the
        head for overhead counting), only this property has to change.

        NOTE: this point is NOT used for IN/OUT counting. Counting still uses
        Detection.foot_point, unchanged.
        """
        return self.center

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


class HeadDetector:
    """Loads the head model once and turns its output into HeadDetection objects.

    Responsibility: model loading, warm-up and per-frame inference for heads.
    Why a class: the model is expensive to load, so it is loaded once in
    __init__ and reused for every frame.

    State:  the YOLO model, which backend it is, and the inference size.
    Output: detect(frame) -> list[HeadDetection].

    Relationship: created and owned by head_stage.HeadAssociationStage, which
    wraps every call in error handling. This class itself is allowed to raise
    (e.g. a missing model file) - containing failures is the stage's job.
    """

    def __init__(self, backend: str | None = None, model_path: str | None = None,
                 inference_size: int | None = None):
        """Load and warm up the head model.

        Parameters:
            backend:        "pose" or "yolo". None = head_config HEAD_BACKEND.
            model_path:     weights file. None = head_config HEAD_MODEL_PATH.
            inference_size: square input size. None = HEAD_INFERENCE_SIZE, and
                            if that is None too, the person model's size.

        Raises:
            ValueError for an unknown backend, or if the weights do not match
            the backend (e.g. a detection model given to the "pose" backend).
            Whatever ultralytics raises if the file cannot be loaded.
        """
        self.backend = backend or head_config.get("HEAD_BACKEND")
        self.model_path = model_path or head_config.get("HEAD_MODEL_PATH")
        self.inference_size = (inference_size
                               or head_config.get("HEAD_INFERENCE_SIZE")
                               or config.INFERENCE_SIZE)

        # Reject a typo in the backend name up front, with a clear message,
        # rather than failing obscurely on the first frame.
        if self.backend not in ("pose", "yolo"):
            raise ValueError(f"HEAD_BACKEND must be 'pose' or 'yolo', got {self.backend!r}")

        print(f"Loading head model {self.model_path} (backend: {self.backend}) ...")
        # A brand-new YOLO instance, separate from PersonDetector's. This is
        # what guarantees head inference can never reset or advance the
        # person model's ByteTrack state, which lives on that other instance.
        self.model = YOLO(self.model_path)
        self._check_model_matches_backend()
        self._warm_up()
        print(f"Head model ready (imgsz={self.inference_size}).")

    def _check_model_matches_backend(self):
        """Fail early if the weights cannot serve the chosen backend.

        A plain detection model has no keypoints, so the "pose" backend would
        silently find zero heads forever. Catching it at start-up turns that
        silent failure into one clear message.
        """
        task = getattr(self.model, "task", None)
        if self.backend == "pose" and task != "pose":
            raise ValueError(
                f"Backend 'pose' needs a pose model (e.g. yolov8n-pose.pt), "
                f"but {self.model_path} is a '{task}' model.")
        if self.backend == "yolo" and task != "detect":
            raise ValueError(
                f"Backend 'yolo' needs a detection model with a head class, "
                f"but {self.model_path} is a '{task}' model.")

    def _warm_up(self):
        """Run one throwaway inference so the slow first call happens now.

        Same reason as PersonDetector._warm_up: PyTorch's first predict() is
        seconds slower than the rest, and doing it before the window opens
        keeps that stall out of the live display loop.
        """
        blank = np.zeros((self.inference_size, self.inference_size, 3), dtype=np.uint8)
        self.model.predict(blank, imgsz=self.inference_size, verbose=False)

    def detect(self, frame) -> list[HeadDetection]:
        """Find heads in one frame.

        Parameters:
            frame: BGR image. Should be a clean copy without overlays.

        Returns:
            list[HeadDetection], possibly empty, in frame pixel coordinates.

        Uses predict(), never track(): heads get their identity by being
        attached to an already-tracked person, so a second tracker would be
        redundant and could disagree with ByteTrack.
        """
        if self.backend == "pose":
            heads = self._detect_from_pose(frame)
        else:
            heads = self._detect_from_head_model(frame)
        # One box per physical head, whichever backend produced them.
        return suppress_duplicate_heads(heads)

    # ------------------------------------------------------------------
    # Backend: dedicated head-detection model
    # ------------------------------------------------------------------
    def _detect_from_head_model(self, frame) -> list[HeadDetection]:
        """Run a YOLO model that was trained with a "head" class.

        The simplest backend: the model's boxes ARE head boxes, so this is
        only a conversion into HeadDetection objects.
        """
        results = self.model.predict(
            frame,
            # Keep only the head class, in case the model also detects other
            # things (some CrowdHuman models have "person" and "head").
            classes=[head_config.get("HEAD_CLASS_ID")],
            conf=head_config.get("HEAD_CONF_THRESHOLD"),
            imgsz=self.inference_size,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        # .xyxy is already in ORIGINAL frame pixels (ultralytics undoes the
        # letterboxing), so no rescaling is needed.
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        return [HeadDetection(int(x1), int(y1), int(x2), int(y2), float(c), "yolo")
                for (x1, y1, x2, y2), c in zip(xyxy, confs)]

    # ------------------------------------------------------------------
    # Backend: head box derived from pose keypoints
    # ------------------------------------------------------------------
    def _detect_from_pose(self, frame) -> list[HeadDetection]:
        """Build a head box from each pose person's face + shoulder keypoints.

        Why not simply "a box around the face keypoints" (the first version):
        the nose, eyes and ears all sit on the FRONT and MIDDLE of the head.
        On this project's camera (seated people, three-quarter/side views,
        heads ~30 px tall) that box came out ~20 px - a third too small - and
        its centre sat on the cheek, below and in front of the real head
        centre. In profile the face points nearly coincide, so their spread
        says almost nothing about head size.

        So the head is built from body PROPORTIONS anchored on keypoints the
        model is good at, instead of from the spread of the face points:

            1. Keep the face keypoints that are confident (really visible).
               Need >= 2 of them, and at least one eye or ear - the nose alone
               cannot give eye level.

            2. EYE LEVEL = mean y of the visible eyes and ears. Eyes and ears
               sit at about the same height, roughly mid-head. The nose is
               excluded: it is lower and would drag the head down.

            3. "UP" = distance from eye level to the crown (top of the head).
               Primary estimate: the pose box TOP, because the head is the
               top-most part of an upright person, so
                   up = eye_level - pose_box_top.
               Cross-check: the eye-to-shoulder drop is about the same length
               as the eye-to-crown distance. If the box-top estimate disagrees
               wildly with it (e.g. arms raised above the head push the box
               top up), the shoulder estimate is used instead.
               Measured on real frames from this camera: the two estimates
               agreed within ~1 px for both people tested.

            4. HEAD SIZE from "up":  height = 1.6 x up    (crown -> chin)
                                     width  = 0.8 x height (heads are taller
                                                            than they are wide)

            5. HEAD CENTRE X:
                 both ears visible -> midpoint of the ears (true skull centre)
                 one ear visible   -> mostly the ear (in a side view the ear is
                                      near the middle of the skull), nudged a
                                      quarter of the way towards the eyes
                 no ears           -> midpoint of the eyes (frontal face)
               HEAD CENTRE Y: crown + height/2  =  eye_level - 0.2 x up

            6. Emit the box, clipped to the frame, remembering the pose body
               box it came from (association uses it to pick the right person).

        Measured on the real frame where the first version drew
        (353,124)-(373,145), the true head was about (342,108)-(373,145).
        """
        results = self.model.predict(
            frame,
            conf=head_config.get("HEAD_CONF_THRESHOLD"),
            imgsz=self.inference_size,
            verbose=False,
        )
        result = results[0]
        keypoints = result.keypoints
        if keypoints is None or result.boxes is None or len(result.boxes) == 0:
            return []

        # (N, 17, 2) pixel coordinates and (N, 17) per-keypoint confidence.
        # .conf can be None for models exported without visibility scores; in
        # that case treat every keypoint as visible (the box conf still gates).
        xy = keypoints.xy.cpu().numpy()
        kp_conf = (keypoints.conf.cpu().numpy() if keypoints.conf is not None
                   else np.ones(xy.shape[:2], dtype=np.float32))
        person_boxes = result.boxes.xyxy.cpu().numpy()
        person_confs = result.boxes.conf.cpu().numpy()

        min_kp_conf = head_config.get("HEAD_KEYPOINT_CONF")
        frame_h, frame_w = frame.shape[:2]
        heads = []

        for i in range(len(person_boxes)):
            pts = xy[i]
            # A keypoint counts as visible only if confident AND not (0, 0),
            # which is how ultralytics reports a keypoint it could not place.
            visible = ((kp_conf[i] >= min_kp_conf)
                       & (pts[:, 0] > 0) & (pts[:, 1] > 0))

            # Step 1: enough face evidence to place a head at all.
            face_vis = [k for k in _FACE_KEYPOINTS if visible[k]]
            eyes = [k for k in _EYE_KEYPOINTS if visible[k]]
            ears = [k for k in _EAR_KEYPOINTS if visible[k]]
            if len(face_vis) < 2 or not (eyes or ears):
                continue

            # Step 2: eye level from eyes/ears only (nose deliberately left out).
            eye_y = float(pts[eyes + ears, 1].mean())

            # Step 3: eye-to-crown distance from the pose box top, cross-checked
            # against the eye-to-shoulder drop when the shoulders are visible.
            box_x1, box_y1, box_x2, box_y2 = person_boxes[i]
            up = eye_y - float(box_y1)
            shoulders = [k for k in _SHOULDER_KEYPOINTS if visible[k]]
            if shoulders:
                up_from_shoulders = float(pts[shoulders, 1].mean()) - eye_y
                # Only trust the shoulders when they really are below the eyes;
                # a bending or lying person can put them level or above.
                if up_from_shoulders > 0 and not (
                        0.5 * up_from_shoulders <= up <= 1.8 * up_from_shoulders):
                    up = up_from_shoulders
            # Never let a degenerate estimate produce an invisible box.
            up = max(up, 3.0)

            # Step 4: head size from body proportions.
            head_h = 1.6 * up
            head_w = 0.8 * head_h

            # Step 5: head centre.
            if len(ears) == 2:
                cx = float(pts[ears, 0].mean())
            elif len(ears) == 1:
                ear_x = float(pts[ears[0], 0])
                # No eyes visible (back of the head): the ear is all we have.
                eyes_x = float(pts[eyes, 0].mean()) if eyes else ear_x
                cx = ear_x + 0.25 * (eyes_x - ear_x)
            else:
                cx = float(pts[eyes, 0].mean())
            cy = eye_y - up + head_h / 2

            # Step 6: clip to the frame so drawing and cropping are always safe.
            x1 = int(max(0, cx - head_w / 2))
            y1 = int(max(0, cy - head_h / 2))
            x2 = int(min(frame_w - 1, cx + head_w / 2))
            y2 = int(min(frame_h - 1, cy + head_h / 2))
            if x2 <= x1 or y2 <= y1:
                continue     # head lies entirely off-frame

            # Confidence: the person's detection confidence weighted by how
            # sure the model is about the face points we actually used.
            confidence = float(person_confs[i] * kp_conf[i, face_vis].mean())
            heads.append(HeadDetection(
                x1, y1, x2, y2, confidence, "pose",
                body_box=(int(box_x1), int(box_y1), int(box_x2), int(box_y2))))

        return heads


def box_iou(a: tuple, b: tuple) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes, from 0 to 1.

    1 = identical boxes, 0 = no overlap. Used here to remove duplicate heads,
    and by person_head_association.py to compare a head's pose body box with
    the tracked person boxes.
    """
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def suppress_duplicate_heads(heads: list[HeadDetection],
                             iou_threshold: float = 0.4) -> list[HeadDetection]:
    """Keep one head per physical head.

    Why: the pose model sometimes returns TWO skeletons for one person (seen on
    this camera: a full-body and an upper-body guess for the same seated man).
    Each yields a head box in the same place, and without this one copy gets
    matched while the other shows up as a stray "unmatched" head.

    How: highest confidence first; drop any later head that overlaps an
    already-kept head by more than iou_threshold, or whose centre lies inside
    one (catches a small duplicate nested inside a larger one).
    """
    kept = []
    for head in sorted(heads, key=lambda h: h.confidence, reverse=True):
        box = (head.x1, head.y1, head.x2, head.y2)
        cx, cy = head.center
        duplicate = any(
            box_iou(box, (k.x1, k.y1, k.x2, k.y2)) > iou_threshold
            or (k.x1 <= cx <= k.x2 and k.y1 <= cy <= k.y2)
            for k in kept)
        if not duplicate:
            kept.append(head)
    return kept
