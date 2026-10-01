"""
head_config.py
==============

Default settings for the optional head-detection / person-head association
feature, and the one function every head module uses to read them.

Why this module exists
----------------------
Every other setting in the project lives in config.py. But config.py is
deliberately kept OUT of git (it holds the camera password), so each machine
has its own private copy - and none of those copies know about the head
feature. If the head modules read `config.HEAD_...` directly, every existing
config.py would crash with AttributeError the moment the feature was used.

So the defaults live here, in a tracked file, and config.py stays the place to
OVERRIDE them:

    head_config.get("HEAD_CONF_THRESHOLD")
        -> config.HEAD_CONF_THRESHOLD   if you added it to your config.py
        -> DEFAULTS["HEAD_CONF_THRESHOLD"] otherwise

main.py's --heads / --head-backend / --head-model flags work the same way as
the existing flags: they assign onto the config module, and get() picks the
value up from there.

Input / output
--------------
Input:  a setting name.
Output: its value (config.py override if present, otherwise the default).

Where it fits
-------------
Read by head_detector.py, person_head_association.py, head_drawing.py and
head_stage.py. Nothing in the existing counting pipeline reads it, so no head
setting can change how people are detected, tracked or counted.

Future use
----------
Any later add-on feature (pose, Re-ID, ...) that must not depend on a private
config.py can follow the same "defaults here, overrides in config.py" pattern.
"""

import config


# ---------------------------------------------------------------------------
# Defaults. Copy any of these names into your config.py to override them.
# ---------------------------------------------------------------------------
DEFAULTS = {
    # ---- On / off ---------------------------------------------------------
    # Off by default: the head model is a SECOND neural network per frame, and
    # on a CPU-only machine that roughly halves the FPS. Turn it on with
    # `python main.py --heads`, or set this to True in config.py.
    "HEAD_DETECTION_ENABLED": False,

    # ---- Which head detector ---------------------------------------------
    # "pose" - derive a head box from the face keypoints (nose, eyes, ears)
    #          of the official Ultralytics pose model. Downloads by itself on
    #          first run, so it works out of the box. Weak when the face and
    #          ears are not visible (people seen from directly behind/above).
    # "yolo" - a YOLO DETECTION model trained with a "head" class, e.g. on
    #          SCUT-HEAD or CrowdHuman. Better for overhead cameras and people
    #          walking away, but you must supply the .pt file yourself.
    "HEAD_BACKEND": "pose",

    # Weights for the chosen backend. For "pose" this is an official name that
    # ultralytics downloads automatically (~6.5 MB). For "yolo" point it at
    # your own head-detection weights, e.g. "models/yolov8n-head.pt".
    "HEAD_MODEL_PATH": "yolov8n-pose.pt",

    # "yolo" backend only: the class index of "head" inside your model. Most
    # single-class head models use 0. Check model.names if unsure.
    "HEAD_CLASS_ID": 0,

    # Minimum confidence for a head (or, for "pose", for the pose person the
    # head comes from). Separate from the person thresholds on purpose.
    #
    # 0.2, not 0.3: on this project's camera a clearly visible seated person
    # scores only 0.30-0.35 in the pose model, so 0.3 made his head vanish in
    # ~1 frame in 12. Low-confidence extras are harmless here - duplicates are
    # suppressed in head_detector.py, and a head is only attached to a person
    # who passes the geometric gate in person_head_association.py.
    "HEAD_CONF_THRESHOLD": 0.2,

    # Inference size for the head model. None = same as config.INFERENCE_SIZE.
    # Heads are small, so a custom head model may want a LARGER size than the
    # person model to find distant heads.
    "HEAD_INFERENCE_SIZE": None,

    # "pose" backend only: a face keypoint must be at least this confident to
    # be used when building the head box. Below it, the keypoint is treated as
    # "not visible" (e.g. the far ear of someone in profile).
    "HEAD_KEYPOINT_CONF": 0.5,

    # ---- Association geometry (see person_head_association.py) -----------
    # The head centre must lie within the TOP fraction of the person box.
    # 0.45 = top 45%. A standing person's head is in the top ~15-20%; the
    # extra room covers sitting/bending people and loose person boxes.
    "HEAD_UPPER_REGION_FRACTION": 0.45,

    # How far the head centre may sit OUTSIDE the person box, sideways,
    # as a fraction of the box width. Person boxes are sometimes tighter
    # than the head (arms in, head tilted).
    "HEAD_HORIZONTAL_MARGIN": 0.15,

    # How far the head centre may sit ABOVE the person box top, as a fraction
    # of the box height. The person box usually contains the head, but a box
    # clipped at the top or a raised head can put the centre slightly above.
    "HEAD_TOP_MARGIN": 0.10,

    # ---- Visualization ----------------------------------------------------
    "HEAD_DRAW": True,          # draw head boxes and head points
    "HEAD_DRAW_LINK": True,     # draw a line from each head to its person
    # Colour (BGR) for heads that matched no person. Matched heads use the
    # same per-ID colour as their person's box, so the pairing is visible.
    "HEAD_UNMATCHED_COLOR": (200, 200, 200),

    # ---- Debugging ----------------------------------------------------------
    # Print every person's ID, person box, head box and head point to the
    # console every this many frames (0 = off). `python main.py --head-debug`
    # sets it to 10. Use it to check the association numerically.
    "HEAD_DEBUG_EVERY_N_FRAMES": 0,
}


def get(name: str):
    """Return a head setting: config.py's value if defined, else the default.

    Parameters:
        name: one of the keys of DEFAULTS.

    Returns:
        The setting's value.

    Read at CALL time (not cached at import) so that main.py's command-line
    overrides, which are assigned onto the config module after import, are
    always seen - the same convention the rest of the project follows.
    A KeyError for an unknown name is intentional: it catches typos instead
    of silently returning None.
    """
    return getattr(config, name, DEFAULTS[name])
