"""
head_stage.py
=============

The single entry point app.py uses for the head feature. It runs head
detection, person-head association and head drawing for one frame - and
guarantees that NONE of it can ever break the existing pipeline.

Purpose / why it exists
-----------------------
The existing system (person detection -> ByteTrack -> counting) is the
product; heads are an optional extra. This module is the firewall between the
two:

    * If the head model fails to LOAD (missing file, wrong model type, no
      internet for the first download), the stage disables itself, prints one
      warning, and the app runs exactly as it did before.
    * If head inference fails on a FRAME, the exception is caught, the frame
      simply has no head info, and person tracking / counting carry on.

It also keeps app.py to orchestration only: app.py makes one call per frame
and does not need to know about models, geometry or overlays.

Input / output
--------------
Input (per frame):
    clean_frame  - a copy of the camera frame taken BEFORE any overlay was
                   drawn, so the head model never sees boxes or labels.
    draw_frame   - the display frame the existing pipeline has drawn on.
    persons      - the person Detections the existing pipeline produced this
                   frame. Only their new `head` field is written, after counting.
Output:
    HUD text lines, and self.last_result: the latest AssociationResult, which
    is where any future feature should read "track 17's head" from.

Where it fits in the pipeline
-----------------------------
    frame ─┬─► existing pipeline.process()  (person detect -> ByteTrack -> count)
           │          │ persons (only .head is set)
           └─(copy)───┴─► HeadAssociationStage.process()
                              ├─ HeadDetector.detect()          head_detector.py
                              ├─ PersonHeadAssociator.associate() person_head_association.py
                              └─ draw_associations()            head_drawing.py

It uses the frame app.py already read - it never opens a camera or stream.

Future use
----------
This "optional stage with failure isolation" shape is the template for later
add-ons (pose, Re-ID, face recognition): each gets its own stage object that
consumes the same frame + persons and can fail without consequence.
"""

import traceback

from . import head_config
from .head_detector import HeadDetector
from .head_drawing import draw_associations
from .person_head_association import AssociationResult, PersonHeadAssociator


# Print the full traceback for the first frame failure (so the cause can be
# diagnosed) and a one-line reminder every this many failures after that
# (so a persistent fault cannot flood the console at 10+ lines per second).
_REPORT_EVERY_N_FAILURES = 100


class HeadAssociationStage:
    """Optional, failure-isolated head detection + association for each frame.

    Responsibility: own the head detector and associator, run them per frame,
    draw the result, and swallow every error they raise.
    Why a class: the model must be loaded once and reused; failure counters
    and the last result must persist between frames.

    State:
        enabled      - False if loading failed; process() is then a no-op.
        detector     - HeadDetector (None if loading failed).
        associator   - PersonHeadAssociator.
        last_result  - AssociationResult from the most recent successful frame
                       (empty after a failed frame, so nothing stale is shown).
        failures     - count of frames whose head processing raised.

    Relationship: created by app.run() when the feature is switched on;
    consumes Detections from the existing pipeline and sets only their `head`
    field, after counting has already run for the frame.
    """

    def __init__(self):
        """Try to load the head model. Never raises.

        Any problem is reported once and leaves the stage disabled, so the
        caller does not need its own try/except around construction.
        """
        self.enabled = False
        self.detector = None
        self.associator = PersonHeadAssociator()
        self.last_result = AssociationResult()
        self.failures = 0
        # Frames processed, for the optional periodic debug print.
        self.frame_index = 0

        try:
            self.detector = HeadDetector()
            self.enabled = True
        except Exception as error:  # noqa: BLE001 - isolation is the whole point
            # Broad on purpose: a missing file, a bad download, a wrong model
            # type and a CUDA problem all raise different exception types, and
            # every one of them must leave counting running normally.
            print(f"WARNING: head detection disabled - could not load the head "
                  f"model: {error}")
            print("         Person detection, tracking and counting continue normally.")

    def process(self, clean_frame, draw_frame, persons) -> list[str]:
        """Detect heads, associate them with persons, draw. Never raises.

        Parameters:
            clean_frame: un-annotated copy of this frame, fed to the head model.
            draw_frame:  the display frame, drawn on in place.
            persons:     list[Detection] from the existing pipeline this frame.

        Returns:
            HUD lines to append to the existing ones (empty if disabled).
        """
        if not self.enabled:
            return []

        try:
            # 1. Heads from the CLEAN frame - the display frame already has
            #    person boxes, labels and the counting line painted on it.
            heads = self.detector.detect(clean_frame)

            # 2. Pair heads with the persons the existing tracker found.
            self.last_result = self.associator.associate(persons or [], heads)

            # 3. Attach each matched head to its person's existing track
            #    record (Detection.head), so anything holding the person -
            #    by track ID - now holds its head too. This is the ONLY write
            #    to a person, it touches only the new `head` field, and it
            #    happens after counter.update() has already run for this
            #    frame, so boxes, IDs, foot points and counts are unaffected.
            #    No new track is created for the head: it rides on the person's.
            for assoc in self.last_result.associations:
                assoc.person.head = assoc.head

            # 4. Overlay last, on top of the existing drawing.
            if head_config.get("HEAD_DRAW"):
                draw_associations(draw_frame, self.last_result)

            # 5. Optional numeric debug output, for checking pairings exactly.
            self.frame_index += 1
            debug_every = head_config.get("HEAD_DEBUG_EVERY_N_FRAMES")
            if debug_every and self.frame_index % debug_every == 0:
                self._print_debug(persons or [])

        except Exception:  # noqa: BLE001 - a head failure must not stop counting
            self._report_failure()
            # Clear the result so no stale head data from an earlier frame is
            # presented as belonging to this one.
            self.last_result = AssociationResult()
            return ["Heads: error (counting unaffected)"]

        result = self.last_result
        total_heads = len(result.associations) + len(result.unmatched_heads)
        return [f"Heads: {total_heads}   Matched: {len(result.associations)}"]

    def _print_debug(self, persons):
        """Print each person's ID, person box, head box and head point.

        One line per person, plus one per unmatched head, e.g.
            [heads] frame 120: ID 1 person=(334,107,375,209) head=(344,110,374,147) head_point=(359,128)
            [heads] frame 120: ID 7 person=(0,96,36,247) head=None
            [heads] frame 120: unmatched head=(600,40,620,64)
        Reads Detection.head, i.e. exactly what was attached to the track.
        """
        prefix = f"[heads] frame {self.frame_index}:"
        for person in persons:
            box = f"({person.x1},{person.y1},{person.x2},{person.y2})"
            head = person.head
            if head is None:
                print(f"{prefix} ID {person.track_id} person={box} head=None")
            else:
                print(f"{prefix} ID {person.track_id} person={box} "
                      f"head=({head.x1},{head.y1},{head.x2},{head.y2}) "
                      f"head_point={head.head_point}")
        for head in self.last_result.unmatched_heads:
            print(f"{prefix} unmatched head=({head.x1},{head.y1},{head.x2},{head.y2})")

    def _report_failure(self):
        """Log a frame failure without flooding the console."""
        self.failures += 1
        if self.failures == 1:
            print("WARNING: head processing failed on a frame; counting continues. "
                  "Details:")
            traceback.print_exc()
        elif self.failures % _REPORT_EVERY_N_FAILURES == 0:
            print(f"WARNING: head processing has failed on {self.failures} frames.")
