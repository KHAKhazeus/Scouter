"""ScoutEnv: PettingZoo AEC environment for the 2-player Scout card game."""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

from scouter.env.card import (
    Card,
    array_to_hand,
    build_2p_deck,
    hand_to_array,
)
from scouter.env.game_logic import (
    MAX_ACTIONS,
    MAX_HAND,
    ORIENT_FLIP,
    ORIENT_KEEP,
    classify_set,
    compute_round_scores,
    decode_orientation,
    decode_scout,
    decode_show,
    encode_scout,
    encode_show,
    is_orientation_action,
    is_scout_action,
    is_show_action,
    is_stronger,
    valid_scout_insertions,
    valid_show_slices,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENTS = ["player_0", "player_1"]
NUM_ROUNDS = 2
INITIAL_CHIPS = 3
CARDS_PER_PLAYER = 11  # 44-card deck / 2 players / 2 rounds
MAX_COLLECTED = 22  # upper bound on cards any one player can collect in a round
MAX_SCOUTED = INITIAL_CHIPS  # max cards a player can scout per round (= chips)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ScoutEnv(AECEnv):
    """2-player Scout card game as a PettingZoo AEC environment.

    Observation and action spaces are identical for both agents.
    The flat Discrete(MAX_ACTIONS) action space is masked via both
    obs["action_mask"] and info["action_mask"] for framework compatibility.
    """

    metadata = {
        "render_modes": ["ansi"],
        "name": "scout_v0",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        num_rounds: int = NUM_ROUNDS,
        reward_mode: str = "raw",
    ) -> None:
        """Create a Scout environment.

        Parameters
        ----------
        render_mode : str | None
            "ansi" for text rendering, None to disable.
        num_rounds : int
            Number of rounds per game (default 2).  Use 1 for RL training
            where each round is an isolated episode.
        reward_mode : str
            "raw"        — each agent receives their own round score at round end.
            "score_diff" — each agent receives (my_score − opponent_score).
        """
        super().__init__()
        self.render_mode = render_mode
        self._num_rounds = num_rounds
        self._reward_mode = reward_mode

        self.possible_agents = list(AGENTS)
        self.agents = list(AGENTS)

        # Observation / action spaces (same for both agents)
        obs_space = spaces.Dict(
            {
                "hand": spaces.Box(0, 10, shape=(MAX_HAND, 3), dtype=np.int8),
                "hand_size": spaces.Discrete(MAX_HAND + 1),
                "active_set": spaces.Box(0, 10, shape=(MAX_HAND, 3), dtype=np.int8),
                "active_set_size": spaces.Discrete(MAX_HAND + 1),
                "active_set_owner": spaces.Discrete(2),
                "opp_hand_size": spaces.Discrete(MAX_HAND + 1),
                "scout_chips": spaces.Box(0, INITIAL_CHIPS, shape=(2,), dtype=np.int8),
                "collected_counts": spaces.Box(0, 44, shape=(2,), dtype=np.int8),
                # Full identity of collected cards: [player_idx, card_idx, (a, b)]
                "collected_cards": spaces.Box(
                    0, 10, shape=(2, MAX_COLLECTED, 2), dtype=np.int8
                ),
                # Cards scouted by the opponent this round: [card_idx, (a, b)].
                # Both face values are public (visible on the table when scouted),
                # but the flip choice and insertion position are hidden.
                "opp_scouted_cards": spaces.Box(
                    0, 10, shape=(MAX_SCOUTED, 2), dtype=np.int8
                ),
                "opp_scouted_count": spaces.Discrete(MAX_SCOUTED + 1),
                "round_num": spaces.Discrete(self._num_rounds + 1),
                "action_mask": spaces.Box(0, 1, shape=(MAX_ACTIONS,), dtype=np.int8),
            }
        )
        act_space = spaces.Discrete(MAX_ACTIONS)

        self.observation_spaces = {a: obs_space for a in self.possible_agents}
        self.action_spaces = {a: act_space for a in self.possible_agents}

        # Game state (initialised properly in reset())
        self._round: int = 0
        self._deck: list[Card] = []
        self._hands: dict[str, list[Card]] = {}
        self._active_set: list[Card] = []
        self._active_owner: str | None = None
        self._chips: dict[str, int] = {}
        self._collected: dict[str, int] = {}
        self._collected_cards: dict[str, list[tuple[int, int]]] = {}
        self._scouted_log: dict[str, list[tuple[int, int]]] = {}
        self._cumulative_scores: dict[str, int] = {}
        self._round_ender: str | None = None
        self._round_end_reason: str = ""
        self._full_deck: list[Card] = build_2p_deck()
        self._pre_round_phase: dict[str, bool] = {}
        self.final_info: dict = {}
        self._history: list[dict] = []
        self._round_results: list[dict] = []

    # ------------------------------------------------------------------
    # PettingZoo required interface
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.agents = list(self.possible_agents)
        self._cumulative_scores = {a: 0 for a in self.agents}
        self._round = 0
        self.final_info = {}
        self._history = []
        self._round_results = []
        self._pre_round_phase = {a: True for a in self.agents}
        # Initialise rewards before _start_round() (which does not touch rewards)
        self.rewards = {a: 0.0 for a in self.agents}
        self._start_round()

    def _start_round(self) -> None:
        """Deal cards for the current round and reset per-round state.

        Note: does NOT reset self.rewards — that is managed exclusively by
        reset() and step() so that round-end scores survive to be accumulated.
        """
        deck = list(self._full_deck)
        random.shuffle(deck)

        # Assign each half of the deck to the round
        round_cards = deck[self._round * 22 : (self._round + 1) * 22]

        self._hands = {
            AGENTS[0]: round_cards[:CARDS_PER_PLAYER],
            AGENTS[1]: round_cards[CARDS_PER_PLAYER:],
        }
        self._active_set = []
        self._active_owner = None
        self._chips = {a: INITIAL_CHIPS for a in self.agents}
        self._collected = {a: 0 for a in self.agents}
        self._collected_cards = {a: [] for a in self.agents}
        self._scouted_log = {a: [] for a in self.agents}
        self._round_ender = None
        self._round_end_reason = ""

        # Reset cumulative rewards for the new round cycle (rewards are NOT reset
        # here so round-end scores set by _end_round() survive until step() can
        # accumulate them into _cumulative_rewards).
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        # Pre-round flip window: each player may flip until their first action.
        self._pre_round_phase = {a: True for a in self.agents}

        # First player must Show (no active set exists yet)
        self._agent_selector = agent_selector.agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def observe(self, agent: str) -> dict[str, Any]:
        """Return the observation dict for the given agent."""
        # Use the fixed AGENTS list (not self.agents which may be empty after game-over)
        agent_idx = AGENTS.index(agent)
        opp = AGENTS[1 - agent_idx]

        hand = self._hands.get(agent, [])
        hand_arr = hand_to_array(hand, MAX_HAND)
        active_arr = hand_to_array(self._active_set, MAX_HAND)

        owner_idx = (
            self.agents.index(self._active_owner)
            if self._active_owner in self.agents
            else 0
        )
        opp_hand_size = len(self._hands.get(opp, []))

        chips_arr = np.array(
            [self._chips.get(AGENTS[0], 0), self._chips.get(AGENTS[1], 0)],
            dtype=np.int8,
        )
        collected_arr = np.array(
            [self._collected.get(AGENTS[0], 0), self._collected.get(AGENTS[1], 0)],
            dtype=np.int8,
        )

        # Full collected card identities: shape (2, MAX_COLLECTED, 2)
        collected_cards_arr = np.zeros((2, MAX_COLLECTED, 2), dtype=np.int8)
        for pi, a_name in enumerate(AGENTS):
            for ci, (ca, cb) in enumerate(self._collected_cards.get(a_name, [])):
                if ci < MAX_COLLECTED:
                    collected_cards_arr[pi, ci] = [ca, cb]

        # Opponent's scouted cards: shape (MAX_SCOUTED, 2)
        opp_scouted_arr = np.zeros((MAX_SCOUTED, 2), dtype=np.int8)
        opp_scouted_list = self._scouted_log.get(opp, [])
        for si, (sa, sb) in enumerate(opp_scouted_list):
            if si < MAX_SCOUTED:
                opp_scouted_arr[si] = [sa, sb]

        mask = self._build_action_mask(agent)

        return {
            "hand": hand_arr,
            "hand_size": np.int64(len(hand)),
            "active_set": active_arr,
            "active_set_size": np.int64(len(self._active_set)),
            "active_set_owner": np.int64(owner_idx),
            "opp_hand_size": np.int64(opp_hand_size),
            "scout_chips": chips_arr,
            "collected_counts": collected_arr,
            "collected_cards": collected_cards_arr,
            "opp_scouted_cards": opp_scouted_arr,
            "opp_scouted_count": np.int64(len(opp_scouted_list)),
            "round_num": np.int64(self._round),
            "action_mask": mask,
        }

    def last(
        self, observe: bool = True
    ) -> tuple[dict | None, float, bool, bool, dict]:
        agent = self.agent_selection
        obs = self.observe(agent) if observe else None
        reward = self._cumulative_rewards[agent]
        terminated = self.terminations[agent]
        truncated = self.truncations[agent]
        # Return a copy so callers cannot mutate self.infos[agent]
        info = dict(self.infos[agent])
        self._cumulative_rewards[agent] = 0.0
        return obs, reward, terminated, truncated, info

    def step(self, action: int | None) -> None:
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        if action is None:
            raise ValueError("action=None is only valid for terminated/truncated agents")
        if action < 0 or action >= MAX_ACTIONS:
            raise ValueError(f"Action {action} out of range [0, {MAX_ACTIONS})")

        mask = self._build_action_mask(agent)
        if not mask[action]:
            raise ValueError(
                f"Illegal action {action} for {agent}. "
                f"Valid actions: {np.where(mask)[0].tolist()}"
            )

        # Reset rewards at start of step (using all possible agents for safety)
        self.rewards = {a: 0.0 for a in self.possible_agents}

        if is_orientation_action(action):
            self._do_orientation(agent, action)
        elif is_show_action(action):
            self._do_show(agent, action)
        elif is_scout_action(action):
            self._do_scout(agent, action)
        else:
            raise ValueError(f"Action {action} out of range [0, {MAX_ACTIONS})")

        # Accumulate step rewards into cumulative rewards.
        # Use possible_agents because self.agents may be empty after game-over.
        for a in self.possible_agents:
            self._cumulative_rewards[a] += self.rewards.get(a, 0.0)

        # PettingZoo API requirement: rewards/terminations/etc. keys must == agents.
        self.rewards = {a: self.rewards.get(a, 0.0) for a in self.agents}

        if self.render_mode == "ansi":
            self.render()

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _do_orientation(self, agent: str, action: int) -> None:
        if not self._pre_round_phase.get(agent, False):
            raise ValueError(f"{agent} has already committed orientation this round")
        flip = decode_orientation(action)
        if flip:
            self._hands[agent] = [c.flipped_copy() for c in reversed(self._hands[agent])]
        self._pre_round_phase[agent] = False
        self._history.append(
            {
                "round": self._round,
                "action": "orientation",
                "player": agent,
                "choice": "flip" if flip else "keep",
            }
        )

        # Same turn continues after orientation commit. If no actions remain, round ends.
        post_mask = self._build_action_mask(agent)
        if not post_mask.any():
            self._round_ender = self._active_owner
            self._round_end_reason = (
                f"{agent} is stuck — cannot beat the active set and has no scout chips"
            )
            self._end_round()

    def _do_show(self, agent: str, action: int) -> None:
        self._pre_round_phase[agent] = False
        start, end = decode_show(action)
        hand = self._hands[agent]
        shown_cards = hand[start : end + 1]
        shown_vals = [c.value for c in shown_cards]
        set_type = classify_set(shown_vals)

        beaten_count = 0
        # The player who beats the active set collects those cards as score cards
        if self._active_set and self._active_owner is not None:
            beaten_count = len(self._active_set)
            self._collected[agent] += beaten_count
            for c in self._active_set:
                self._collected_cards[agent].append((c.a, c.b))

        prev_owner = self._active_owner
        self._active_set = shown_cards
        self._active_owner = agent
        self._hands[agent] = hand[:start] + hand[end + 1 :]

        entry: dict = {
            "round": self._round,
            "action": "show",
            "player": agent,
            "cards": shown_vals,
            "set_type": set_type,
            "size": len(shown_vals),
        }
        if beaten_count > 0:
            entry["beat"] = prev_owner
            entry["collected"] = beaten_count
        self._history.append(entry)

        # Condition (a): hand empty → round ends, this agent is exempt from penalties
        if len(self._hands[agent]) == 0:
            self._round_ender = agent
            self._round_end_reason = f"{agent} played all their cards"
            self._end_round()
            return

        # Advance to opponent
        self.agent_selection = self._agent_selector.next()

        # If the opponent can't Show AND can't Scout → round ends immediately.
        # The shower (active-set owner) is the round ender and exempt from penalties.
        opp = self.agent_selection
        opp_mask = self._build_action_mask(opp)
        if not opp_mask.any():
            self._round_ender = agent
            self._round_end_reason = (
                f"{opp} is stuck — cannot beat the active set and has no scout chips"
            )
            self._end_round()
            return

    def _do_scout(self, agent: str, action: int) -> None:
        self._pre_round_phase[agent] = False
        side, flip, insert_pos = decode_scout(action)

        # Take card from active set
        if side == 0:
            card = self._active_set.pop(0)
        else:
            card = self._active_set.pop()

        original_val = card.value
        # Record the scouted card's identity (both face values are public)
        self._scouted_log[agent].append((card.a, card.b))

        if flip:
            card = card.flipped_copy()

        # Insert into hand
        hand = self._hands[agent]
        hand.insert(insert_pos, card)
        self._hands[agent] = hand

        # Spend a scout chip (goes to center; opponent does NOT receive it in 2p)
        self._chips[agent] -= 1

        self._history.append({
            "round": self._round,
            "action": "scout",
            "player": agent,
            "side": "left" if side == 0 else "right",
            "card_value": card.value,
            "flipped": bool(flip),
            "original_value": original_val,
            "insert_pos": insert_pos,
            "chips_remaining": self._chips[agent],
        })

        # If active set is now empty, owner's set is gone
        if not self._active_set:
            self._active_owner = None

        # 2-player tempo rule: agent keeps the turn (does NOT advance).
        # They must continue scouting or showing until they Show or get stuck.
        # Check if they are now stuck (can't show AND no chips for scouting).
        mask = self._build_action_mask(agent)
        if not mask.any():
            self._round_ender = self._active_owner
            self._round_end_reason = (
                f"{agent} is stuck — cannot beat the active set and has no scout chips"
            )
            self._end_round()

    # ------------------------------------------------------------------
    # Round / game end
    # ------------------------------------------------------------------

    def _end_round(self) -> None:
        """Score the round, add to cumulative totals, advance or end game.

        The current active set does NOT score — it is discarded at round end.
        """
        round_scores = compute_round_scores(
            collected=self._collected,
            hands=self._hands,
            chips=self._chips,
            round_ender=self._round_ender,
        )

        round_result: dict = {
            "round": self._round,
            "round_ender": self._round_ender,
            "reason": self._round_end_reason,
            "scores": dict(round_scores),
            "breakdown": {},
        }
        for a in AGENTS:
            hand_size = len(self._hands.get(a, []))
            is_exempt = (a == self._round_ender)
            round_result["breakdown"][a] = {
                "collected": self._collected.get(a, 0),
                "hand_cards": hand_size,
                "hand_penalty": 0 if is_exempt else -hand_size,
                "unspent_chips": self._chips.get(a, 0),
                "total": round_scores[a],
                "exempt": is_exempt,
            }
        self._round_results.append(round_result)

        self._history.append({
            "round": self._round,
            "action": "round_end",
            "round_ender": self._round_ender,
            "reason": self._round_end_reason,
            "scores": dict(round_scores),
        })

        for a, s in round_scores.items():
            self._cumulative_scores[a] += s

        # Assign per-step rewards based on mode
        if self._reward_mode == "score_diff":
            agents_list = list(round_scores.keys())
            for a in agents_list:
                opp = [x for x in agents_list if x != a][0]
                self.rewards[a] = float(round_scores[a] - round_scores[opp])
        else:
            for a, s in round_scores.items():
                self.rewards[a] = float(s)

        self._round += 1

        if self._round >= self._num_rounds:
            # Game over — mark all agents terminated but keep them in self.agents
            # so each can receive a dead step and be removed one-by-one, keeping
            # all PettingZoo dict invariants (rewards/terminations/infos keys == agents).
            winner = max(self._cumulative_scores, key=lambda a: self._cumulative_scores[a])
            game_info = {
                "final_scores": dict(self._cumulative_scores),
                "winner": winner,
                "round_scores": round_scores,
            }
            # Persist final info so callers can read it after env.agents is empty.
            self.final_info = game_info
            self._history.append({
                "round": self._round - 1,
                "action": "game_over",
                "final_scores": dict(self._cumulative_scores),
                "winner": winner,
            })
            for a in self.agents:
                self.terminations[a] = True
                self.infos[a] = game_info
            # agent_selection stays on the acting agent so dead-step loop starts with them
        else:
            # Start next round — swap first player for the new round.
            # self.rewards still holds round1 scores; _start_round() does NOT reset
            # rewards, so step() can accumulate them before the next step resets.
            self.agents = list(reversed(self.possible_agents))
            self._start_round()

    # ------------------------------------------------------------------
    # Action mask
    # ------------------------------------------------------------------

    def _build_action_mask(self, agent: str) -> np.ndarray:
        mask = np.zeros(MAX_ACTIONS, dtype=np.int8)

        if self.terminations.get(agent) or self.truncations.get(agent):
            return mask

        # Force one orientation decision per player at round start.
        if self._pre_round_phase.get(agent, False):
            mask[ORIENT_KEEP] = 1
            mask[ORIENT_FLIP] = 1
            return mask

        hand = self._hands.get(agent, [])
        active_vals = [c.value for c in self._active_set]

        # Show actions: valid whenever this agent has a legal set to play.
        # NOTE: no agent_selection guard here — this method is also called on the
        # *opponent* inside _do_show() to check whether they would be stuck, and
        # that check must reflect the real board state, not whose turn it is.
        if not self._active_set:
            # No active set: any valid set is allowed (first show of the round, or
            # after the active set was emptied by scouting).
            for start, end in valid_show_slices(hand, []):
                mask[encode_show(start, end)] = 1
        else:
            for start, end in valid_show_slices(hand, active_vals):
                mask[encode_show(start, end)] = 1

        # Scout actions — only if active set exists and agent has chips
        if self._active_set and self._chips.get(agent, 0) > 0:
            hand_size = len(hand)
            for insert_pos in valid_scout_insertions(hand_size):
                for side in (0, 1):
                    for flip in (0, 1):
                        mask[encode_scout(side, flip, insert_pos)] = 1

        return mask

    # ------------------------------------------------------------------
    # Global state (for centralized training / QMIX)
    # ------------------------------------------------------------------

    def state(self) -> np.ndarray:
        """Return a flat global state array exposing all hidden info."""
        parts = []
        for a in AGENTS:
            hand = self._hands.get(a, [])
            parts.append(hand_to_array(hand, MAX_HAND).flatten())
            parts.append(np.array([len(hand)], dtype=np.int8))
        # Active set
        parts.append(hand_to_array(self._active_set, MAX_HAND).flatten())
        parts.append(np.array([len(self._active_set)], dtype=np.int8))
        # Owner index
        owner_idx = (
            AGENTS.index(self._active_owner) if self._active_owner in AGENTS else -1
        )
        parts.append(np.array([owner_idx], dtype=np.int8))
        # Chips (remaining)
        parts.append(
            np.array([self._chips.get(a, 0) for a in AGENTS], dtype=np.int8)
        )
        # Collected counts
        parts.append(
            np.array([self._collected.get(a, 0) for a in AGENTS], dtype=np.int8)
        )
        # Collected card identities: (2, MAX_COLLECTED, 2)
        for a in AGENTS:
            cc = np.zeros((MAX_COLLECTED, 2), dtype=np.int8)
            for i, (ca, cb) in enumerate(self._collected_cards.get(a, [])):
                if i < MAX_COLLECTED:
                    cc[i] = [ca, cb]
            parts.append(cc.flatten())
        # Scouted card log: (2, MAX_SCOUTED, 2)
        for a in AGENTS:
            sc = np.zeros((MAX_SCOUTED, 2), dtype=np.int8)
            for i, (sa, sb) in enumerate(self._scouted_log.get(a, [])):
                if i < MAX_SCOUTED:
                    sc[i] = [sa, sb]
            parts.append(sc.flatten())
        # Round
        parts.append(np.array([self._round], dtype=np.int8))
        return np.concatenate([p.flatten() for p in parts])

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None

        # Cap display: _round can exceed _num_rounds after game-over dead steps
        display_round = min(self._round + 1, self._num_rounds)
        lines = [f"=== Scout — Round {display_round} ==="]
        current = self.agent_selection if self.agents else "—"
        lines.append(f"Turn: {current}")
        for a in AGENTS:
            hand = self._hands.get(a, [])
            chips = self._chips.get(a, 0)
            cc = self._collected_cards.get(a, [])
            sc = self._scouted_log.get(a, [])
            cc_str = f", cards={cc}" if cc else ""
            sc_str = f", scouted={sc}" if sc else ""
            lines.append(
                f"  {a}: {[str(c) for c in hand]} "
                f"(chips={chips}, collected={self._collected.get(a,0)}{cc_str}{sc_str})"
            )
        active_vals = [c.value for c in self._active_set]
        lines.append(f"  Active set: {active_vals} (owner={self._active_owner})")
        out = "\n".join(lines)
        print(out)
        return out

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _other_agent(self, agent: str) -> str:
        return AGENTS[1 - AGENTS.index(agent)]

    def _was_dead_step(self, action: int | None) -> None:
        """Handle step(None) for a terminated/truncated agent.

        Removes the agent from self.agents and trims all dict attributes so
        the PettingZoo invariant (all dict keys == agents) is maintained.
        """
        if action is not None:
            raise ValueError(
                f"step() called with action={action} for terminated/truncated agent "
                f"{self.agent_selection}. Pass action=None instead."
            )
        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0.0

        # Remove the dead agent from the active list
        if agent in self.agents:
            self.agents.remove(agent)

        # Trim all dicts so keys == self.agents
        self.rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: self.terminations.get(a, False) for a in self.agents}
        self.truncations = {a: self.truncations.get(a, False) for a in self.agents}
        self.infos = {a: self.infos.get(a, {}) for a in self.agents}

        # Reinitialize the selector for remaining agents and advance
        if self.agents:
            self._agent_selector = agent_selector.agent_selector(self.agents)
            self.agent_selection = self._agent_selector.next()

    # ------------------------------------------------------------------
    # Rich state access (for API / frontend)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Pre-round hand-flip (players may flip their hand before first action)
    # ------------------------------------------------------------------

    def can_flip_hand(self, agent: str) -> bool:
        """True while this agent still needs orientation choice."""
        return (
            self._pre_round_phase.get(agent, False)
            and agent in self.agents
            and not self.terminations.get(agent, False)
        )

    def flip_player_hand(self, agent: str) -> bool:
        """Legacy helper: commit flipped orientation for this round."""
        if not self.can_flip_hand(agent):
            return False
        self._do_orientation(agent, ORIENT_FLIP)
        return True

    # ------------------------------------------------------------------
    # Rich state access (for API / frontend)
    # ------------------------------------------------------------------

    def get_rich_state(self) -> dict:
        """Return a JSON-serialisable full state dict (used by the API)."""
        return {
            "round": self._round,
            "current_player": self.agent_selection,
            "hands": {
                a: [[c.a, c.b, int(c.flipped)] for c in self._hands.get(a, [])]
                for a in AGENTS
            },
            "active_set": [[c.a, c.b, int(c.flipped)] for c in self._active_set],
            "active_owner": self._active_owner,
            "scout_chips": dict(self._chips),
            "collected": dict(self._collected),
            "collected_cards": {
                a: list(self._collected_cards.get(a, [])) for a in AGENTS
            },
            "scouted_log": {
                a: list(self._scouted_log.get(a, [])) for a in AGENTS
            },
            "cumulative_scores": dict(self._cumulative_scores),
            "action_mask": self._build_action_mask(self.agent_selection).tolist(),
            "can_flip_hand": {a: self.can_flip_hand(a) for a in AGENTS},
            "orientation_pending": {a: bool(self._pre_round_phase.get(a, False)) for a in AGENTS},
            "game_over": not bool(self.agents),
            "winner": self.final_info.get("winner") if not self.agents else None,
            "history": list(self._history),
            "round_results": list(self._round_results),
        }
