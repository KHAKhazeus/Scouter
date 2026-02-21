"""Convert a ScoutEnv observation dict to a natural-language LLM prompt."""

from __future__ import annotations

import textwrap

import numpy as np

from scouter.env.game_logic import (
    MAX_HAND,
    classify_set,
    encode_scout,
    encode_show,
    is_stronger,
    valid_show_slices,
)
from scouter.env.card import array_to_hand, Card


def _card_str(a: int, b: int, flipped: int) -> str:
    """Render a card as '[up^/down]'."""
    up = b if flipped else a
    down = a if flipped else b
    return f"[{up}^/{down}]"


def _active_set_description(active_set: list[list[int]]) -> str:
    if not active_set:
        return "(empty — no active set)"
    vals = [b if f else a for a, b, f in active_set]
    set_type = classify_set(vals)
    min_val = min(vals)
    val_str = " ".join(str(v) for v in vals)
    return f"{val_str}  ({set_type}, min={min_val})"


def format_observation(
    obs: dict,
    agent: str,
    active_owner: str | None = None,
) -> str:
    """Convert a ScoutEnv observation dict to a human-readable LLM prompt.

    Parameters
    ----------
    obs:
        The observation dict returned by ``env.observe(agent)``.
    agent:
        The agent ID whose perspective this is ("player_0" or "player_1").
    active_owner:
        The owner of the current active set (passed separately because the
        observation only stores the owner *index*, not the name).
    """
    round_num = int(obs["round_num"]) + 1
    hand_size = int(obs["hand_size"])
    active_set_size = int(obs["active_set_size"])
    opp_hand_size = int(obs["opp_hand_size"])

    # Decode hand
    hand_arr = obs["hand"]
    hand_cards = [
        (int(hand_arr[i, 0]), int(hand_arr[i, 1]), int(hand_arr[i, 2]))
        for i in range(hand_size)
    ]
    hand_str = "  ".join(_card_str(a, b, f) for a, b, f in hand_cards)

    # Indices for display (1-based for humans)
    idx_str = "  ".join(f" {i+1} " for i in range(hand_size))

    # Active set
    active_arr = obs["active_set"]
    active_cards = [
        (int(active_arr[i, 0]), int(active_arr[i, 1]), int(active_arr[i, 2]))
        for i in range(active_set_size)
    ]
    active_desc = _active_set_description(active_cards)
    owner_label = active_owner if active_owner else "unknown"

    # Chips
    chips = obs["scout_chips"]
    agent_idx = 0 if agent == "player_0" else 1
    opp_idx = 1 - agent_idx
    my_chips = int(chips[agent_idx])
    opp_chips = int(chips[opp_idx])

    # Collected
    collected = obs["collected_counts"]
    my_collected = int(collected[agent_idx])
    opp_collected = int(collected[opp_idx])

    # Collected card identities
    collected_cards = obs.get("collected_cards")
    my_cc_str = ""
    opp_cc_str = ""
    if collected_cards is not None:
        my_cc = [(int(collected_cards[agent_idx, i, 0]), int(collected_cards[agent_idx, i, 1]))
                 for i in range(my_collected)
                 if collected_cards[agent_idx, i, 0] != 0 or collected_cards[agent_idx, i, 1] != 0]
        opp_cc = [(int(collected_cards[opp_idx, i, 0]), int(collected_cards[opp_idx, i, 1]))
                  for i in range(opp_collected)
                  if collected_cards[opp_idx, i, 0] != 0 or collected_cards[opp_idx, i, 1] != 0]
        if my_cc:
            my_cc_str = " — cards: " + ", ".join(f"({a}/{b})" for a, b in my_cc)
        if opp_cc:
            opp_cc_str = " — cards: " + ", ".join(f"({a}/{b})" for a, b in opp_cc)

    # Opponent's scouted cards (both face values visible, but flip/position unknown)
    opp_scouted_count = int(obs.get("opp_scouted_count", 0))
    opp_scouted_arr = obs.get("opp_scouted_cards")
    opp_scouted_str = ""
    if opp_scouted_count > 0 and opp_scouted_arr is not None:
        scouted = [(int(opp_scouted_arr[i, 0]), int(opp_scouted_arr[i, 1]))
                   for i in range(opp_scouted_count)]
        opp_scouted_str = ", ".join(f"({a}/{b})" for a, b in scouted)

    # Valid actions from mask
    mask = obs["action_mask"]
    valid_indices = np.where(mask)[0].tolist()
    action_lines = _describe_valid_actions(valid_indices, hand_cards, active_cards)

    opp_scouted_section = ""
    if opp_scouted_str:
        opp_scouted_section = (
            f"\n        Opponent scouted cards (values known, orientation/position unknown): {opp_scouted_str}"
        )

    prompt = textwrap.dedent(f"""\
        === Scout Game — Round {round_num}, Your turn ({agent}) ===

        Your hand ({hand_size} cards, left to right):
          Pos:  {idx_str}
          Cards: {hand_str}
          (card notation: [active^/inactive])

        Active set (owned by {owner_label}, {active_set_size} card(s)): {active_desc}

        Scout chips: you={my_chips}/3  opponent={opp_chips}/3
        Collected cards: you={my_collected}{my_cc_str}  opponent={opp_collected}{opp_cc_str}
        Opponent hand size: {opp_hand_size}{opp_scouted_section}

        Valid actions:
        {chr(10).join('  ' + line for line in action_lines) if action_lines else '  (none — game over)'}

        Respond with a JSON action, for example:
          {{"action": "orient", "flip": false}}
          {{"action": "orient", "flip": true}}
          {{"action": "show", "start": 1, "end": 3}}
          {{"action": "scout", "side": "left", "flip": false, "insert": 2}}
        (positions are 1-based)
    """)
    return prompt


def _describe_valid_actions(
    valid_indices: list[int],
    hand_cards: list[tuple[int, int, int]],
    active_cards: list[tuple[int, int, int]],
) -> list[str]:
    from scouter.env.game_logic import (
        N_SHOW,
        decode_orientation,
        decode_scout,
        decode_show,
        is_orientation_action,
    )

    lines: list[str] = []
    seen_show: set[tuple[int, int]] = set()
    seen_scout: set[tuple[int, int, int]] = set()

    active_vals = [b if f else a for a, b, f in active_cards]

    for idx in valid_indices:
        if idx < N_SHOW:
            start, end = decode_show(idx)
            if (start, end) not in seen_show:
                seen_show.add((start, end))
                vals = [b if f else a for a, b, f in hand_cards[start : end + 1]]
                set_type = classify_set(vals)
                val_str = ",".join(str(v) for v in vals)
                lines.append(
                    f'SHOW cards {start+1}–{end+1}: [{val_str}] ({set_type})'
                )
        elif is_orientation_action(idx):
            choice = decode_orientation(idx)
            lines.append("ORIENT hand: keep current orientation" if choice == 0 else "ORIENT hand: flip whole hand")
        else:
            side, flip, insert_pos = decode_scout(idx)
            if (side, flip, insert_pos) not in seen_scout:
                seen_scout.add((side, flip, insert_pos))
                side_label = "left" if side == 0 else "right"
                if active_cards:
                    card = active_cards[0] if side == 0 else active_cards[-1]
                    a, b, f = card
                    current_val = b if f else a
                    other_val = a if f else b
                    flip_label = f" (flip → {other_val})" if flip else ""
                    lines.append(
                        f"SCOUT {side_label} card ({current_val}{flip_label}), insert at pos {insert_pos+1}"
                    )

    return lines
