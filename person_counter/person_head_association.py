"""
person_head_association.py
==========================

Decides which detected head belongs to which detected (tracked) person.

Purpose / why it exists
-----------------------
The person model gives us people with track IDs. The head model gives us
heads, which have no identity at all. To say "person ID 17's head is HERE" we
must pair the two, and that pairing is this module's only job.

It is pure geometry: no model, no OpenCV, no frame. That makes it fast,
deterministic and easy to unit-test with hand-made boxes.

Input / output
--------------
Input:  list[detector.Detection]          (people, with track_id or None)
        list[head_detector.HeadDetection] (heads)
Output: AssociationResult
            .associations           one PersonHeadAssociation per matched pair
            .unmatched_heads        heads that fit no person
            .persons_without_head   people that got no head
            .by_track_id            {track_id: PersonHeadAssociation}

    Person ID 17
        ├── person_bbox   (from the existing Detection, unchanged)
        ├── head_bbox     (from HeadDetection)
        └── head_point    (HeadDetection.head_point)

Where it fits in the pipeline
-----------------------------
    PersonDetector.track() ─┐
                            ├──► PersonHeadAssociator.associate() ──► result
    HeadDetector.detect() ──┘

Called by head_stage.py. It only READS the person Detections - it never
modifies them - so the foot points and everything counter.py sees are
untouched.

The matching rule (first version - simple and explainable)
----------------------------------------------------------
    Head detected
        ↓
    Find candidate persons      - every person whose box could contain it
        ↓
    Gate: is the head centre inside / near the person's UPPER region?
          (horizontally within the box ± a margin, vertically between just
           above the box top and HEAD_UPPER_REGION_FRACTION of the way down)
        ↓
    Score each surviving (head, person) pair - lower cost is better:
          horizontal cost = how far the head centre is from the box's
                            vertical centre line, relative to half the width
          vertical cost   = how far the head centre is from where a head is
                            EXPECTED (just below the box top), relative to
                            the box height
          body cost       = (pose backend only) how poorly the pose skeleton
                            the head came from overlaps this person's box,
                            1 - IoU. Zero overlap rejects the pair outright.
        ↓
    Select best valid matches   - greedy, cheapest pair first, each head
                                  and each person used at most once
        ↓
    Create PersonHeadAssociation objects

Why greedy and one-to-one: two people standing close have overlapping boxes,
so one head may pass the gate for both. Handing out the cheapest pairs first,
and never reusing a head or a person, resolves that the way a human would -
the head goes to the person it is best centred on, and the other person gets
the other head. (An optimal Hungarian assignment is a possible upgrade; greedy
is easier to follow and almost always agrees in practice.)

Future use
----------
Head-to-ID pairing is the hook for face recognition / Re-ID (crop the head of
track 17), head-point trails, and pose or attention features per person.
"""

from dataclasses import dataclass, field

from . import head_config
from .detector import Detection
from .head_detector import HeadDetection, box_iou


# Where, inside the person box, a head centre is EXPECTED to be, as a fraction
# of box height from the top. A standing person's head occupies roughly the top
# 1/7-1/8 of their height, so its centre sits about 8% down. Used only for the
# vertical cost; the hard limit is HEAD_UPPER_REGION_FRACTION.
_EXPECTED_HEAD_Y_FRACTION = 0.08

# How much each cost term counts. Horizontal alignment is the stronger signal:
# a head is almost always centred over its body, whereas vertical position
# varies with posture (bending, sitting) and with how tight the box is.
_HORIZONTAL_WEIGHT = 0.6
_VERTICAL_WEIGHT = 0.4


@dataclass
class PersonHeadAssociation:
    """One person paired with one head.

    Responsibility: the output record of association - everything a later
    feature needs about "this person's head", in one object.

    State:  the original person Detection (shared, NOT copied - read-only by
            convention), the HeadDetection, and the match cost.
    Output: convenience accessors track_id / person_bbox / head_bbox /
            head_point, so callers do not dig through two objects.
    """

    person: Detection
    head: HeadDetection
    cost: float      # 0 = perfectly placed head; larger = less typical placement

    @property
    def track_id(self) -> int | None:
        """The ByteTrack ID of the person (None in detect mode)."""
        return self.person.track_id

    @property
    def person_bbox(self) -> tuple[int, int, int, int]:
        p = self.person
        return p.x1, p.y1, p.x2, p.y2

    @property
    def head_bbox(self) -> tuple[int, int, int, int]:
        h = self.head
        return h.x1, h.y1, h.x2, h.y2

    @property
    def head_point(self) -> tuple[int, int]:
        """The head's representative point. Not used for IN/OUT counting."""
        return self.head.head_point


@dataclass
class AssociationResult:
    """Everything one association pass produced, for one frame.

    Responsibility: hand callers matched pairs AND the leftovers, because the
    leftovers are informative too (a head with no person may be a person the
    person model missed; a person with no head may be facing away).

    Output: lists plus a by_track_id lookup for "give me track 17's head".
    """

    associations: list[PersonHeadAssociation] = field(default_factory=list)
    unmatched_heads: list[HeadDetection] = field(default_factory=list)
    persons_without_head: list[Detection] = field(default_factory=list)

    @property
    def by_track_id(self) -> dict[int, PersonHeadAssociation]:
        """Associations keyed by track ID. Untracked people (ID None) are left
        out, since None cannot identify anyone across frames."""
        return {a.track_id: a for a in self.associations if a.track_id is not None}


class PersonHeadAssociator:
    """Matches heads to persons using box geometry only.

    Responsibility: implement the gate -> score -> greedy-select rule described
    at the top of this file.
    Why a class: it holds the tunable thresholds, so a future variant (e.g. one
    that also uses track history) can subclass or replace it without touching
    the callers.

    State:  three geometry thresholds, read from head_config at construction.
    Output: associate(persons, heads) -> AssociationResult.

    Relationship: owned by head_stage.HeadAssociationStage. Stateless between
    frames - each call is independent.
    """

    def __init__(self, upper_region_fraction: float | None = None,
                 horizontal_margin: float | None = None,
                 top_margin: float | None = None):
        """Store the gate thresholds.

        Parameters (all fractions; None = take the value from head_config):
            upper_region_fraction: head centre must be within this top part of
                                   the person box height.
            horizontal_margin:     allowed overhang sideways, × box width.
            top_margin:            allowed overhang above the box, × box height.
        """
        self.upper_region_fraction = (upper_region_fraction if upper_region_fraction is not None
                                      else head_config.get("HEAD_UPPER_REGION_FRACTION"))
        self.horizontal_margin = (horizontal_margin if horizontal_margin is not None
                                  else head_config.get("HEAD_HORIZONTAL_MARGIN"))
        self.top_margin = (top_margin if top_margin is not None
                           else head_config.get("HEAD_TOP_MARGIN"))

    def match_cost(self, person: Detection, head: HeadDetection) -> float | None:
        """Score one (person, head) pair, or reject it.

        Parameters:
            person: a person Detection.
            head:   a HeadDetection.

        Returns:
            None if the head cannot belong to this person (fails the gate);
            otherwise a cost >= 0, where 0 means "exactly where a head should
            be". Costs are comparable across persons of different sizes
            because both terms are normalised by the person box.
        """
        width = person.x2 - person.x1
        height = person.y2 - person.y1
        # A degenerate box has no "upper region" to speak of.
        if width <= 0 or height <= 0:
            return None

        # The head's centre is the single point we compare with the person.
        hx, hy = head.center

        # --- Gate 1: horizontal. The head centre must be over the body,
        # allowing a small overhang for tight boxes or tilted heads.
        margin_x = width * self.horizontal_margin
        if not (person.x1 - margin_x <= hx <= person.x2 + margin_x):
            return None

        # --- Gate 2: vertical. Between slightly ABOVE the box top (box clipped
        # at the top, raised head) and the bottom of the allowed upper region.
        # This is what stops a head being matched to the legs of a taller
        # person standing in front of someone else.
        top_limit = person.y1 - height * self.top_margin
        bottom_limit = person.y1 + height * self.upper_region_fraction
        if not (top_limit <= hy <= bottom_limit):
            return None

        # --- Cost 1: horizontal alignment. 0 when the head is dead centre,
        # 1 at the box edge, a little over 1 inside the margin.
        person_cx = (person.x1 + person.x2) / 2
        horizontal_cost = abs(hx - person_cx) / (width / 2)

        # --- Cost 2: vertical placement. Distance from where a head centre is
        # normally found, measured in units of the allowed upper region, so it
        # is ~0 for a normal standing pose and ~1 at the edge of the region.
        expected_y = person.y1 + height * _EXPECTED_HEAD_Y_FRACTION
        vertical_cost = abs(hy - expected_y) / (height * self.upper_region_fraction)

        geometric_cost = _HORIZONTAL_WEIGHT * horizontal_cost + _VERTICAL_WEIGHT * vertical_cost

        # --- Body evidence (pose backend only). The head was derived from a
        # pose skeleton, and head.body_box is that skeleton's box. The tracked
        # person this head belongs to is the one whose box overlaps that body
        # box - which is decisive exactly where position alone is ambiguous:
        # two people side by side or one behind the other, where one head
        # passes the gate for BOTH person boxes.
        if head.body_box is not None:
            body_iou = box_iou(head.body_box, (person.x1, person.y1, person.x2, person.y2))
            # The head's own body does not touch this person's box at all, so
            # the head cannot be theirs, however well it happens to be placed.
            if body_iou == 0.0:
                return None
            # Equal parts placement and body overlap. (1 - IoU) is 0 for the
            # same body and approaches 1 for a body that barely overlaps.
            return 0.5 * geometric_cost + 0.5 * (1.0 - body_iou)

        return geometric_cost

    def associate(self, persons: list[Detection],
                  heads: list[HeadDetection]) -> AssociationResult:
        """Pair heads with persons, one-to-one, best matches first.

        Parameters:
            persons: this frame's person Detections (from the existing tracker).
            heads:   this frame's HeadDetections.

        Returns:
            AssociationResult with matched pairs and the leftovers.

        Logic:
            1. Build every VALID (cost, head, person) candidate pair.
            2. Sort them cheapest first.
            3. Walk the list, accepting a pair only if neither its head nor
               its person has been used yet.
        """
        # Step 1: every pair that passes the gate. With a handful of people
        # per frame this is a few dozen checks - negligible next to inference.
        candidates = []
        for h_idx, head in enumerate(heads):
            for p_idx, person in enumerate(persons):
                cost = self.match_cost(person, head)
                if cost is not None:
                    candidates.append((cost, h_idx, p_idx))

        # Step 2: best (lowest cost) first. Indices break ties so the order is
        # deterministic from run to run.
        candidates.sort()

        # Step 3: greedy one-to-one selection.
        used_heads, used_persons = set(), set()
        result = AssociationResult()
        for cost, h_idx, p_idx in candidates:
            if h_idx in used_heads or p_idx in used_persons:
                continue     # one of them already has a better partner
            used_heads.add(h_idx)
            used_persons.add(p_idx)
            result.associations.append(
                PersonHeadAssociation(person=persons[p_idx], head=heads[h_idx], cost=cost))

        # Leftovers on both sides, kept for diagnostics and future features.
        result.unmatched_heads = [h for i, h in enumerate(heads) if i not in used_heads]
        result.persons_without_head = [p for i, p in enumerate(persons) if i not in used_persons]
        return result
