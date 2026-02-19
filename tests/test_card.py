"""Tests for scouter/env/card.py"""

import numpy as np
import pytest

from scouter.env.card import (
    Card,
    _REMOVED_2P_CARD,
    array_to_hand,
    build_2p_deck,
    build_deck,
    hand_to_array,
)


# ---------------------------------------------------------------------------
# Full 45-card deck
# ---------------------------------------------------------------------------


def test_deck_size():
    deck = build_deck()
    assert len(deck) == 45, f"Expected 45 cards, got {len(deck)}"


def test_deck_all_pairs_distinct():
    deck = build_deck()
    pairs = {(c.a, c.b) for c in deck}
    assert len(pairs) == 45, "Duplicate pairs in deck"


def test_deck_no_same_values():
    deck = build_deck()
    for c in deck:
        assert c.a != c.b, f"Card has same values: {c}"


def test_deck_canonical_order():
    """a < b always."""
    deck = build_deck()
    for c in deck:
        assert c.a < c.b, f"Card not in canonical order: {c}"


def test_deck_values_in_range():
    deck = build_deck()
    for c in deck:
        assert 1 <= c.a <= 10, f"Out-of-range a={c.a}"
        assert 1 <= c.b <= 10, f"Out-of-range b={c.b}"


# ---------------------------------------------------------------------------
# 2-player deck (44 cards)
# ---------------------------------------------------------------------------


def test_2p_deck_size():
    deck = build_2p_deck()
    assert len(deck) == 44, f"Expected 44 cards, got {len(deck)}"


def test_2p_deck_removed_card():
    deck = build_2p_deck()
    ra, rb = _REMOVED_2P_CARD
    for c in deck:
        assert not (c.a == ra and c.b == rb), f"Removed card still present: {c}"


def test_2p_deck_is_subset_of_full():
    full = set((c.a, c.b) for c in build_deck())
    two_p = set((c.a, c.b) for c in build_2p_deck())
    assert two_p.issubset(full)
    assert len(full - two_p) == 1


# ---------------------------------------------------------------------------
# Card.value and flipping
# ---------------------------------------------------------------------------


def test_card_value_not_flipped():
    c = Card(3, 7)
    assert c.value == 3
    assert c.other_value == 7


def test_card_value_flipped():
    c = Card(3, 7, flipped=True)
    assert c.value == 7
    assert c.other_value == 3


def test_card_flipped_copy():
    c = Card(2, 9)
    fc = c.flipped_copy()
    assert fc.value == 9
    assert fc.other_value == 2
    assert c.value == 2  # original unchanged


def test_card_double_flip_identity():
    c = Card(4, 6, flipped=True)
    assert c.flipped_copy().flipped_copy() == c


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_card_invalid_same_values():
    with pytest.raises(ValueError):
        Card(5, 5)


def test_card_invalid_wrong_order():
    with pytest.raises(ValueError):
        Card(8, 3)


def test_card_invalid_out_of_range():
    with pytest.raises(ValueError):
        Card(0, 5)
    with pytest.raises(ValueError):
        Card(1, 11)


# ---------------------------------------------------------------------------
# Numpy serialization
# ---------------------------------------------------------------------------


def test_card_to_array():
    c = Card(3, 7, flipped=True)
    arr = c.to_array()
    assert arr.dtype == np.int8
    assert list(arr) == [3, 7, 1]


def test_card_from_array():
    arr = np.array([2, 9, 0], dtype=np.int8)
    c = Card.from_array(arr)
    assert c.a == 2
    assert c.b == 9
    assert not c.flipped
    assert c.value == 2


def test_card_round_trip():
    original = Card(4, 10, flipped=True)
    recovered = Card.from_array(original.to_array())
    assert recovered == original


def test_hand_to_array_padding():
    hand = [Card(1, 2), Card(3, 8, flipped=True)]
    arr = hand_to_array(hand, max_size=5)
    assert arr.shape == (5, 3)
    assert list(arr[0]) == [1, 2, 0]
    assert list(arr[1]) == [3, 8, 1]
    assert list(arr[2]) == [0, 0, 0]  # padding


def test_array_to_hand_roundtrip():
    hand = [Card(2, 5), Card(4, 9, flipped=True), Card(1, 6)]
    arr = hand_to_array(hand)
    recovered = array_to_hand(arr, len(hand))
    assert recovered == hand
