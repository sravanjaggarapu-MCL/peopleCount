"""
head_drawing.py
===============

On-screen overlays for the head feature: head boxes, head points, and the
optional link line from each head to its person.

Why a separate module
---------------------
drawing.py draws the existing person boxes, trails and HUD, and is left
completely untouched. Keeping the head overlays here means the head feature
can be removed, or drawn differently, without editing any existing drawing
code. It REUSES drawing.color_for_id (import only), so a head is drawn in the
same colour as its person's box and the pairing is obvious at a glance.

Input / output
--------------
Input:  the display frame (drawn on in place) and an AssociationResult.
Output: nothing returned; pixels on the frame.

Where it fits
-------------
Called by head_stage.py AFTER the existing pipeline has drawn its person boxes
and counting line, so head overlays sit on top and never hide anything the
existing UI drew underneath. Only called when head_config HEAD_DRAW is True.

Future use
----------
Other per-person add-ons (pose skeletons, face names, Re-ID labels) can follow
the same pattern: their own *_drawing.py, called from their own stage.
"""

import cv2

import config
from . import head_config
from .drawing import FONT, color_for_id
from .person_head_association import AssociationResult


def draw_associations(frame, result: AssociationResult):
    """Draw every head in the result: matched ones in their person's colour,
    unmatched ones in a neutral grey.

    Parameters:
        frame:  BGR display frame, modified in place.
        result: the AssociationResult for this frame.
    """
    draw_link = head_config.get("HEAD_DRAW_LINK")

    for assoc in result.associations:
        # Exactly the colour drawing.draw_detection used for this person's
        # box: the per-ID colour when tracked, plain BOX_COLOR in detect mode
        # (no ID). Matching colours are what make the pairing readable.
        color = (color_for_id(assoc.track_id) if assoc.track_id is not None
                 else config.BOX_COLOR)

        if draw_link:
            # Line from the head point to the person's box centre: makes the
            # association itself visible, the quickest way to spot a wrong
            # pairing when two people overlap. Drawn first so the head marker
            # sits on top of it.
            cv2.line(frame, assoc.head_point, assoc.person.centroid, color, 1, cv2.LINE_AA)

        # "head ID 17" names the track the head was attached to, so the
        # pairing can be read directly, not just inferred from colours.
        label = (f"head ID {assoc.track_id}" if assoc.track_id is not None
                 else "head")
        _draw_head(frame, assoc.head, color, label)

    for head in result.unmatched_heads:
        # "?" = a head was found but no tracked person fits it (e.g. the
        # person model or ByteTrack has no confirmed track there this frame).
        _draw_head(frame, head, head_config.get("HEAD_UNMATCHED_COLOR"), "head ?")


def _draw_head(frame, head, color, label: str):
    """Draw one head: box, head point and a small label.

    Parameters:
        frame: display frame, modified in place.
        head:  the HeadDetection.
        color: BGR colour (the person's colour if matched, grey otherwise).
        label: short text drawn beside the head box.

    Design choices, all for legibility on a 640x480 sub-stream where heads
    are only ~20-40 px:
        * 2 px box, the same weight as the person box, so it survives the
          window being scaled; it is told apart by being small and labelled.
        * The head point is a filled dot inside a black ring, so it reads as
          a different marker from the plain solid foot-point dot used for
          counting.
        * The label sits to the RIGHT of the head box, because the space
          above it is where drawing.draw_detection puts the "ID n" label.
    """
    cv2.rectangle(frame, (head.x1, head.y1), (head.x2, head.y2), color, 2)
    cv2.circle(frame, head.head_point, 6, (0, 0, 0), 2)
    cv2.circle(frame, head.head_point, 4, color, -1)

    # Black outline under coloured text keeps it readable on any background.
    org = (head.x2 + 4, head.y1 + 12)
    cv2.putText(frame, label, org, FONT, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, org, FONT, 0.4, color, 1, cv2.LINE_AA)
