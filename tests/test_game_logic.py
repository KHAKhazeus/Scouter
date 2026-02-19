"""Tests for scouter/env/game_logic.py"""

import pytest

from scouter.env.card import Card
from scouter.env.game_logic import (
    MAX_ACTIONS,
    N_SCOUT,
    N_SHOW,
    classify_set,
    compute_round_scores,
    decode_scout,
    decode_show,
    encode_scout,
    encode_show,
    is_stronger,
    valid_scout_insertions,
    valid_show_slices,
)


# ---------------------------------------------------------------------------
# classify_set
# ---------------------------------------------------------------------------


def test_classify_empty():
    assert classify_set([]) == "invalid"


def test_classify_single():
    assert classify_set([5]) == "match"


def test_classify_match():
    assert classify_set([4, 4, 4]) == "match"


def test_classify_run_ascending():
    assert classify_set([3, 4, 5]) == "run"


def test_classify_run_descending():
    assert classify_set([7, 6, 5]) == "run"


def test_classify_run_two_ascending():
    assert classify_set([2, 3]) == "run"


def test_classify_run_two_descending():
    assert classify_set([8, 7]) == "run"


def test_classify_invalid_mixed():
    assert classify_set([1, 3, 5]) == "invalid"


def test_classify_invalid_non_consecutive():
    assert classify_set([2, 4]) == "invalid"


def test_classify_invalid_same_with_different():
    assert classify_set([3, 3, 4]) == "invalid"


# ---------------------------------------------------------------------------
# is_stronger
# ---------------------------------------------------------------------------


def test_stronger_empty_active():
    """Any valid set beats an empty active set."""
    assert is_stronger([5], []) is True
    assert is_stronger([2, 3, 4], []) is True


def test_stronger_invalid_new_set_loses():
    assert is_stronger([1, 3], [5]) is False


def test_stronger_more_cards_wins():
    assert is_stronger([1, 2, 3], [4, 5]) is True
    assert is_stronger([7, 8], [1, 2, 3]) is False


def test_stronger_same_size_match_beats_run():
    # match [3,3] vs run [4,5] — same size 2, match wins
    assert is_stronger([3, 3], [4, 5]) is True
    assert is_stronger([4, 5], [3, 3]) is False


def test_stronger_same_type_higher_min():
    assert is_stronger([5, 6, 7], [3, 4, 5]) is True
    assert is_stronger([3, 4, 5], [5, 6, 7]) is False


def test_stronger_tie_returns_false():
    # Same size, same type, same min → tie → cannot show
    assert is_stronger([4, 5, 6], [4, 5, 6]) is False
    # Same min match
    assert is_stronger([4, 4], [4, 5]) is False  # match vs run, match wins → True
    # Same min match vs same min match
    assert is_stronger([3, 3], [3, 3]) is False


def test_stronger_single_match_beats_single_lower():
    assert is_stronger([9], [5]) is True
    assert is_stronger([5], [9]) is False
    assert is_stronger([7], [7]) is False


# ---------------------------------------------------------------------------
# valid_show_slices
# ---------------------------------------------------------------------------


def _make_hand(values):
    """Create a hand where each card's active value is the given value."""
    cards = []
    for v in values:
        if v < 10:
            cards.append(Card(v, v + 1))  # active = v (a side)
        else:
            cards.append(Card(9, 10, flipped=True))  # active = 10
    return cards


def test_valid_show_slices_no_active():
    hand = _make_hand([3, 5, 7])
    slices = valid_show_slices(hand, [])
    # All consecutive valid sub-sequences are fine
    starts_ends = {(s, e) for s, e in slices}
    # single cards: (0,0), (1,1), (2,2) should all be valid
    assert (0, 0) in starts_ends
    assert (1, 1) in starts_ends
    assert (2, 2) in starts_ends


def test_valid_show_slices_must_beat_active():
    hand = _make_hand([1, 2, 3, 6])
    active = [5]  # single 5
    slices = valid_show_slices(hand, active)
    # Any single card > 5 beats it, or any run of 2+
    starts_ends = {(s, e) for s, e in slices}
    assert (3, 3) in starts_ends   # 6 > 5 single
    assert (0, 1) in starts_ends   # run [1,2] beats single [5] by size
    assert (0, 0) not in starts_ends  # 1 does not beat 5 single


def test_valid_show_slices_invalid_slice_excluded():
    hand = _make_hand([1, 3, 5])  # no consecutive runs
    slices = valid_show_slices(hand, [])
    # (0,1) → [1,3] not consecutive → invalid; should not appear
    starts_ends = {(s, e) for s, e in slices}
    assert (0, 1) not in starts_ends


def test_valid_scout_insertions():
    positions = valid_scout_insertions(5)
    assert positions == [0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Action encoding / decoding round-trips
# ---------------------------------------------------------------------------


def test_encode_decode_show_all():
    from scouter.env.game_logic import MAX_HAND
    for start in range(MAX_HAND):
        for end in range(start, MAX_HAND):
            action = encode_show(start, end)
            assert action < N_SHOW
            s, e = decode_show(action)
            assert s == start and e == end


def test_encode_decode_scout_all():
    from scouter.env.game_logic import MAX_HAND
    for side in (0, 1):
        for flip in (0, 1):
            for insert in range(MAX_HAND + 1):
                action = encode_scout(side, flip, insert)
                assert N_SHOW <= action < MAX_ACTIONS
                s, f, i = decode_scout(action)
                assert s == side and f == flip and i == insert


# ---------------------------------------------------------------------------
# compute_round_scores
# ---------------------------------------------------------------------------


def _cards(n):
    deck = [Card(1, 2), Card(3, 4), Card(5, 6), Card(7, 8), Card(2, 9)]
    return deck[:n]


def test_score_basic():
    scores = compute_round_scores(
        collected={"player_0": 5, "player_1": 3},
        hands={"player_0": _cards(2), "player_1": _cards(4)},
        chips={"player_0": 2, "player_1": 1},
        round_ender=None,
    )
    # p0: 5 collected - 2 hand + 2 chips = 5
    assert scores["player_0"] == 5
    # p1: 3 collected - 4 hand + 1 chip = 0
    assert scores["player_1"] == 0


def test_score_round_ender_exempt():
    scores = compute_round_scores(
        collected={"player_0": 4, "player_1": 6},
        hands={"player_0": [], "player_1": _cards(3)},
        chips={"player_0": 1, "player_1": 2},
        round_ender="player_0",
    )
    # p0: 4 + 1 chip (no hand penalty) = 5
    assert scores["player_0"] == 5
    # p1: 6 - 3 + 2 = 5
    assert scores["player_1"] == 5


def test_score_all_chips_spent():
    scores = compute_round_scores(
        collected={"player_0": 0, "player_1": 0},
        hands={"player_0": _cards(3), "player_1": _cards(3)},
        chips={"player_0": 0, "player_1": 0},
        round_ender=None,
    )
    assert scores["player_0"] == -3
    assert scores["player_1"] == -3
