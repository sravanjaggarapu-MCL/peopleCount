"""
track_point_prompt.py
=====================

Asking which point on a person the counter should follow, and nothing else.

A detection is a box, but a tripwire test needs a single point. The two
sensible candidates sit at opposite ends of the same box - the feet and the
head - and they disagree by the height of a person, which at a doorway is the
difference between counting someone as they arrive and counting them as they
leave. There is no universally right answer, only a right answer for a given
camera, so the program asks rather than assuming. config.TRACK_POINT carries
the full argument for each.

Like source_prompt, this imports nothing heavier than the standard library, so
the question appears the instant the program starts instead of after the model
has finished loading.
"""

import sys

import config

# What each choice is for, in one line, shown at the prompt. Kept next to the
# prompt rather than in config because this is sales copy for a decision being
# made right now - config.TRACK_POINT holds the reasoning in full.
DESCRIPTIONS = {
    "foot": ("bottom of the box - where the person meets the floor",
             "the usual choice: a camera above head height sees bodies lean",
             "into the frame, so a head reaches the line too early"),
    "head": ("top of the box - the crown of the head",
             "steadier where feet are hidden: crowds, near-overhead cameras,",
             "or a desk or counter across the bottom of the picture"),
}

# Everything accepted for each choice. Numbers match the menu order; the words
# are there because an operator who knows what they want should not have to
# read the menu to find the number.
ALIASES = {
    "1": "foot", "foot": "foot", "feet": "foot", "f": "foot",
    "2": "head", "head": "head", "h": "head",
}


def normalise(text: str) -> str:
    """Turn what was typed into "foot" or "head".

    Raises ValueError with the list of what is accepted, so the caller can
    print it either at the prompt or against a bad --track-point.
    """
    key = text.strip().strip('"').strip("'").lower()
    if key in ALIASES:
        return ALIASES[key]
    raise ValueError(
        f"{text!r} is not a tracking point. "
        f"Expected one of: {', '.join(config.TRACK_POINT_CHOICES)}.")


def prompt_for_track_point() -> str | None:
    """Ask which point to count by. Returns "foot"/"head", or None to quit.

    With no terminal to read from - launched from a scheduler, or piped - there
    is nobody to ask, so we take the default from config.py rather than block
    forever on stdin. Same reasoning as prompt_for_source.
    """
    default = config.TRACK_POINT
    if not sys.stdin or not sys.stdin.isatty():
        print(f"No terminal to prompt on - counting by the {default} point "
              f"(config.TRACK_POINT).")
        return default

    print()
    print("=" * 68)
    print(" WHICH POINT ON EACH PERSON SHOULD BE COUNTED?")
    print("=" * 68)
    for i, choice in enumerate(config.TRACK_POINT_CHOICES, start=1):
        headline, *why = DESCRIPTIONS[choice]
        marker = "  (default)" if choice == default else ""
        print(f"   [{i}] {choice.upper():5s} {headline}{marker}")
        for reason in why:
            print(f"         {reason}")
        print()
    print(f"   [Enter] keeps {default}          [q] quit")
    print("=" * 68)

    while True:
        try:
            answer = input("Track by > ").strip()
        except EOFError:
            # stdin closed mid-prompt - a cancel, not a crash.
            print()
            return None

        if answer.lower() in ("q", "quit", "exit"):
            return None
        if not answer:
            return default

        try:
            return normalise(answer)
        except ValueError as error:
            print(f"  {error}")


def resolve_track_point(choice: str | None, ask: bool = True) -> str | None:
    """Settle the tracking point for this run: the flag, the prompt, or config.

    Returns None only when the operator quit at the prompt - the same contract
    resolve_source uses, so main.py can treat both the same way.
    """
    if choice:
        try:
            return normalise(choice)
        except ValueError as error:
            raise SystemExit(f"--track-point: {error}")

    if not ask:
        return config.TRACK_POINT

    return prompt_for_track_point()
