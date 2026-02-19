"""Game logic for Scout: set classification, strength comparison, scoring, valid actions."""

from __future__ import annotations

from typing import Literal

from scouter.env.card import Card

SetType = Literal["match", "run", "invalid"]

# ---------------------------------------------------------------------------
# Set classification
# ---------------------------------------------------------------------------


def classify_set(values: list[int]) -> SetType:
    """Classify a proposed shown set by its active values.

    Rules:
    - Single card: always valid (treat as both match and run; returned as "match"
      so the strength comparison treats it as the higher type).
    - 2+ cards: must be all equal ("match") OR strictly ascending/descending ("run").
    """
    if not values:
        return "invalid"
    if len(values) == 1:
        return "match"
    if all(v == values[0] for v in values):
        return "match"
    # Check strictly ascending
    if all(values[i] + 1 == values[i + 1] for i in range(len(values) - 1)):
        return "run"
    # Check strictly descending
    if all(values[i] - 1 == values[i + 1] for i in range(len(values) - 1)):
        return "run"
    return "invalid"


# ---------------------------------------------------------------------------
# Strength comparison
# ---------------------------------------------------------------------------

_TYPE_RANK: dict[SetType, int] = {"match": 1, "run": 0, "invalid": -1}


def is_stronger(new_vals: list[int], old_vals: list[int]) -> bool:
    """Return True if new_vals beats old_vals as a Scout shown set.

    Strength rules (in priority order):
    1. More cards beats fewer cards.
    2. If same size: match beats run.
    3. If same size and same type: compare minimum value; higher minimum wins.
       Ties (equal minimum) cannot be shown — returns False.

    If old_vals is empty (no active set yet), any valid new set wins.
    """
    if not old_vals:
        return classify_set(new_vals) != "invalid"

    new_type = classify_set(new_vals)
    old_type = classify_set(old_vals)

    if new_type == "invalid":
        return False

    if len(new_vals) != len(old_vals):
        return len(new_vals) > len(old_vals)

    if _TYPE_RANK[new_type] != _TYPE_RANK[old_type]:
        return _TYPE_RANK[new_type] > _TYPE_RANK[old_type]

    return min(new_vals) > min(old_vals)


# ---------------------------------------------------------------------------
# Valid action enumeration
# ---------------------------------------------------------------------------


def valid_show_slices(
    hand: list[Card], active_set_values: list[int]
) -> list[tuple[int, int]]:
    """Return all (start, end) inclusive index pairs that form a valid shown set
    stronger than active_set_values.

    The shown cards must be consecutive in hand.
    """
    result: list[tuple[int, int]] = []
    n = len(hand)
    for start in range(n):
        for end in range(start, n):
            vals = [hand[i].value for i in range(start, end + 1)]
            if classify_set(vals) != "invalid" and is_stronger(vals, active_set_values):
                result.append((start, end))
    return result


def valid_scout_insertions(hand_size: int) -> list[int]:
    """Return all valid insert positions for a scouted card: 0..hand_size."""
    return list(range(hand_size + 1))


# ---------------------------------------------------------------------------
# Flat action encoding / decoding
# ---------------------------------------------------------------------------
#
# We use a flat integer action space for RLlib / TRL / LLM compatibility.
#
# Layout for a hand of at most MAX_HAND (11) cards:
#   SHOW actions:  indices 0 .. N_SHOW-1
#     encoded as: start * MAX_HAND + end  (start <= end, so only valid combos used)
#   SCOUT actions: indices N_SHOW .. N_SHOW + N_SCOUT - 1
#     encoded as: side * (MAX_HAND+1) * 2 + flip * (MAX_HAND+1) + insert_pos
#       side: 0=left, 1=right
#       flip: 0=no flip, 1=flip
#       insert_pos: 0..MAX_HAND
#
# MAX_HAND = 11  →  N_SHOW = 11*11 = 121  (many are structurally impossible but
#   the mask zeroes them out).  N_SCOUT = 2 * 2 * 12 = 48.
#   MAX_ACTIONS = 121 + 48 = 169.

# MAX_HAND is the largest possible hand at any point: 11 cards dealt + up to 3
# scouted cards (one per chip). Using 11 would overflow when hand exceeds that —
# encode_show(10, 11) == 121 == N_SHOW, which step() misclassifies as a scout.
MAX_HAND: int = 14
N_SHOW: int = MAX_HAND * MAX_HAND  # 196
N_SCOUT: int = 2 * 2 * (MAX_HAND + 1)  # 60
MAX_ACTIONS: int = N_SHOW + N_SCOUT  # 256


def encode_show(start: int, end: int) -> int:
    """Encode a Show action as a flat integer."""
    return start * MAX_HAND + end


def decode_show(action: int) -> tuple[int, int]:
    """Decode a Show action integer to (start, end)."""
    return divmod(action, MAX_HAND)


def encode_scout(side: int, flip: int, insert_pos: int) -> int:
    """Encode a Scout action as a flat integer.

    side: 0=left end of active set, 1=right end.
    flip: 0=keep as-is, 1=flip before inserting.
    insert_pos: position in hand to insert (0..hand_size).
    """
    return N_SHOW + side * 2 * (MAX_HAND + 1) + flip * (MAX_HAND + 1) + insert_pos


def decode_scout(action: int) -> tuple[int, int, int]:
    """Decode a Scout action integer to (side, flip, insert_pos)."""
    idx = action - N_SHOW
    side, remainder = divmod(idx, 2 * (MAX_HAND + 1))
    flip, insert_pos = divmod(remainder, MAX_HAND + 1)
    return side, flip, insert_pos


def is_show_action(action: int) -> bool:
    return 0 <= action < N_SHOW


def is_scout_action(action: int) -> bool:
    return N_SHOW <= action < MAX_ACTIONS


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def compute_round_scores(
    collected: dict[str, int],
    hands: dict[str, list[Card]],
    chips: dict[str, int],
    round_ender: str | None,
) -> dict[str, int]:
    """Compute per-player scores for one completed round.

    collected:    face-down card count per agent (from beating active sets).
    hands:        remaining hand per agent.
    chips:        unspent Scout chips per agent.
    round_ender:  agent who ended the round (emptied hand or played unbeatable
                  set).  This agent is exempt from hand-card penalties.

    Official 2-player scoring:
      +1 per collected card
      +1 per unspent Scout chip
      -1 per remaining hand card (exempt if you are the round_ender)
    """
    scores: dict[str, int] = {}
    for agent in collected:
        s = collected[agent]
        if agent != round_ender:
            s -= len(hands[agent])
        s += chips[agent]
        scores[agent] = s
    return scores
