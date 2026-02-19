"""Parse LLM text output into a flat Scout action integer."""

from __future__ import annotations

import json
import re
from typing import Any

from scouter.env.game_logic import (
    MAX_ACTIONS,
    N_SHOW,
    encode_scout,
    encode_show,
    is_scout_action,
    is_show_action,
)


def parse_action(text: str, hand_size: int) -> int:
    """Parse an LLM completion string into a flat action integer.

    Tries JSON parsing first, then regex fallback.

    Parameters
    ----------
    text:
        The raw text output from the LLM.
    hand_size:
        The current number of cards in the agent's hand (needed to validate
        insertion positions).

    Returns
    -------
    int
        A valid action integer in [0, MAX_ACTIONS), or -1 if parsing fails or
        the result is out of range.
    """
    action = _try_json_parse(text, hand_size)
    if action != -1:
        return action
    return _try_regex_parse(text, hand_size)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _try_json_parse(text: str, hand_size: int) -> int:
    """Extract the first JSON object from text and interpret it as an action."""
    # Find the first {...} block
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        return -1
    try:
        obj: dict[str, Any] = json.loads(match.group())
    except json.JSONDecodeError:
        return -1

    action_type = str(obj.get("action", "")).lower().strip()

    if action_type == "show":
        start = obj.get("start")
        end = obj.get("end")
        if start is None or end is None:
            return -1
        return _encode_show_safe(int(start) - 1, int(end) - 1)

    if action_type == "scout":
        side_raw = str(obj.get("side", "")).lower()
        side = 0 if side_raw in ("left", "0", "l") else 1
        flip_raw = obj.get("flip", False)
        if isinstance(flip_raw, str):
            flip = int(flip_raw.lower() in ("true", "1", "yes"))
        else:
            flip = int(bool(flip_raw))
        insert_raw = obj.get("insert", 1)
        insert_pos = int(insert_raw) - 1  # convert 1-based to 0-based
        return _encode_scout_safe(side, flip, insert_pos, hand_size)

    return -1


# ---------------------------------------------------------------------------
# Regex fallback
# ---------------------------------------------------------------------------

_SHOW_RE = re.compile(
    r"show\s+(?:cards?\s+)?(\d+)\s*[-–to]+\s*(\d+)", re.IGNORECASE
)
_SHOW_SINGLE_RE = re.compile(r"show\s+card\s+(\d+)", re.IGNORECASE)
_SCOUT_RE = re.compile(
    r"scout\s+(left|right)\s+.*?insert\s+(?:at\s+(?:pos(?:ition)?\s+)?)?(\d+)",
    re.IGNORECASE,
)
_FLIP_RE = re.compile(r"\bflip\b", re.IGNORECASE)


def _try_regex_parse(text: str, hand_size: int) -> int:
    m = _SHOW_RE.search(text)
    if m:
        return _encode_show_safe(int(m.group(1)) - 1, int(m.group(2)) - 1)

    m = _SHOW_SINGLE_RE.search(text)
    if m:
        pos = int(m.group(1)) - 1
        return _encode_show_safe(pos, pos)

    m = _SCOUT_RE.search(text)
    if m:
        side = 0 if m.group(1).lower() == "left" else 1
        insert_pos = int(m.group(2)) - 1
        flip = 1 if _FLIP_RE.search(text) else 0
        return _encode_scout_safe(side, flip, insert_pos, hand_size)

    return -1


# ---------------------------------------------------------------------------
# Safe encoders with bounds checks
# ---------------------------------------------------------------------------


def _encode_show_safe(start: int, end: int) -> int:
    from scouter.env.game_logic import MAX_HAND

    if not (0 <= start <= end < MAX_HAND):
        return -1
    action = encode_show(start, end)
    return action if 0 <= action < N_SHOW else -1


def _encode_scout_safe(side: int, flip: int, insert_pos: int, hand_size: int) -> int:
    if side not in (0, 1) or flip not in (0, 1):
        return -1
    if not (0 <= insert_pos <= hand_size):
        return -1
    action = encode_scout(side, flip, insert_pos)
    return action if is_scout_action(action) else -1
