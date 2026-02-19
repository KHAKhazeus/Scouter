"""Card dataclass and deck builders for Scout."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# The single card removed from the full 45-card deck for the 2-player game.
# Per the Scout rulebook, 2-player uses 44 cards (11 per player × 2 players × 2 rounds).
# We remove the (9, 10) card.
_REMOVED_2P_CARD: tuple[int, int] = (9, 10)


@dataclass
class Card:
    """A Scout double-number card.

    Canonical storage: a < b always.
    flipped=False → a is the active (up) value.
    flipped=True  → b is the active (up) value.
    """

    a: int  # smaller face value (1–10)
    b: int  # larger face value (1–10), always a != b
    flipped: bool = False

    def __post_init__(self) -> None:
        if not (1 <= self.a < self.b <= 10):
            raise ValueError(f"Invalid card values: a={self.a}, b={self.b}")

    @property
    def value(self) -> int:
        """Return the currently active (up) number."""
        return self.b if self.flipped else self.a

    @property
    def other_value(self) -> int:
        """Return the inactive (down) number."""
        return self.a if self.flipped else self.b

    def flipped_copy(self) -> "Card":
        """Return a new Card with the opposite side up."""
        return Card(self.a, self.b, not self.flipped)

    def to_array(self) -> NDArray[np.int8]:
        """Serialize to [a, b, flipped] as int8 array."""
        return np.array([self.a, self.b, int(self.flipped)], dtype=np.int8)

    @classmethod
    def from_array(cls, arr: NDArray[np.int8]) -> "Card":
        """Deserialize from [a, b, flipped] int8 array."""
        return cls(int(arr[0]), int(arr[1]), bool(arr[2]))

    def __repr__(self) -> str:
        up = self.value
        down = self.other_value
        return f"Card({up}^/{down})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.a == other.a and self.b == other.b and self.flipped == other.flipped

    def __hash__(self) -> int:
        return hash((self.a, self.b, self.flipped))


def build_deck() -> list[Card]:
    """Build the full 45-card Scout deck (all distinct pairs 1–10)."""
    return [Card(a, b) for a, b in combinations(range(1, 11), 2)]


def build_2p_deck() -> list[Card]:
    """Build the 44-card 2-player deck by removing the (9,10) card."""
    removed_a, removed_b = _REMOVED_2P_CARD
    return [c for c in build_deck() if not (c.a == removed_a and c.b == removed_b)]


def hand_to_array(hand: list[Card], max_size: int = 11) -> NDArray[np.int8]:
    """Encode a hand as a (max_size, 3) int8 array, zero-padded."""
    arr = np.zeros((max_size, 3), dtype=np.int8)
    for i, card in enumerate(hand[:max_size]):
        arr[i] = card.to_array()
    return arr


def array_to_hand(arr: NDArray[np.int8], size: int) -> list[Card]:
    """Decode a (max_size, 3) array back to a list of Cards."""
    return [Card.from_array(arr[i]) for i in range(size)]
