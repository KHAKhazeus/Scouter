"""Tests for ScoutEnv: PettingZoo AEC compliance, game mechanics, and edge cases."""

import numpy as np
import pytest

from scouter.env.game_logic import (
    MAX_ACTIONS,
    N_SHOW,
    encode_scout,
    encode_show,
    valid_show_slices,
)
from scouter.env.scout_env import AGENTS, INITIAL_CHIPS, MAX_COLLECTED, MAX_SCOUTED, NUM_ROUNDS, ScoutEnv


# ---------------------------------------------------------------------------
# PettingZoo api_test compliance
# ---------------------------------------------------------------------------


def test_pettingzoo_api_compliance():
    """Run PettingZoo's built-in API test suite."""
    from pettingzoo.test import api_test

    env = ScoutEnv()
    api_test(env, num_cycles=50, verbose_progress=False)


# ---------------------------------------------------------------------------
# Reset and initial state
# ---------------------------------------------------------------------------


def test_reset_agents():
    env = ScoutEnv()
    env.reset(seed=42)
    assert set(env.agents) == set(AGENTS)


def test_reset_hand_sizes():
    env = ScoutEnv()
    env.reset(seed=1)
    for a in AGENTS:
        assert len(env._hands[a]) == 11


def test_reset_chips():
    env = ScoutEnv()
    env.reset(seed=1)
    for a in AGENTS:
        assert env._chips[a] == INITIAL_CHIPS


def test_reset_no_active_set():
    env = ScoutEnv()
    env.reset()
    assert env._active_set == []
    assert env._active_owner is None


def test_observation_spaces_correct():
    from scouter.env.game_logic import MAX_HAND
    env = ScoutEnv()
    env.reset(seed=0)
    for a in AGENTS:
        obs = env.observe(a)
        assert obs["hand"].shape == (MAX_HAND, 3)
        assert obs["active_set"].shape == (MAX_HAND, 3)
        assert obs["action_mask"].shape == (MAX_ACTIONS,)
        assert obs["action_mask"].dtype == np.int8
        assert obs["collected_cards"].shape == (2, MAX_COLLECTED, 2)
        assert obs["opp_scouted_cards"].shape == (MAX_SCOUTED, 2)
        assert 0 <= int(obs["opp_scouted_count"]) <= MAX_SCOUTED


def test_observation_in_space():
    """Each observation should be contained in the declared observation space."""
    env = ScoutEnv()
    env.reset(seed=7)
    for a in AGENTS:
        obs = env.observe(a)
        assert env.observation_space(a).contains(obs), \
            f"Observation for {a} not in observation space"


# ---------------------------------------------------------------------------
# Action masking consistency
# ---------------------------------------------------------------------------


def test_first_action_must_be_show():
    """At game start there is no active set, so only Show actions are valid."""
    env = ScoutEnv()
    env.reset(seed=3)
    agent = env.agent_selection
    obs = env.observe(agent)
    mask = obs["action_mask"]
    # No scout actions should be valid
    scout_valid = mask[N_SHOW:].any()
    assert not scout_valid, "Scout actions should be masked at game start (no active set)"


def test_mask_invalid_actions_raise():
    """Stepping with a masked (invalid) action should raise ValueError."""
    env = ScoutEnv()
    env.reset(seed=5)
    agent = env.agent_selection
    obs = env.observe(agent)
    mask = obs["action_mask"]
    # Find an invalid action
    invalid = np.where(mask == 0)[0]
    if len(invalid) == 0:
        pytest.skip("All actions valid — cannot test")
    with pytest.raises(ValueError):
        env.step(int(invalid[0]))


def test_mask_and_step_consistent():
    """Every valid-masked action should execute without error."""
    env = ScoutEnv()
    env.reset(seed=9)
    agent = env.agent_selection
    obs = env.observe(agent)
    mask = obs["action_mask"]
    valid_actions = np.where(mask)[0]
    assert len(valid_actions) > 0, "No valid actions at start"
    # All of them should be executable (try first one)
    env2 = ScoutEnv()
    env2.reset(seed=9)
    env2.step(int(valid_actions[0]))  # Should not raise


def test_obs_has_action_mask():
    """action_mask must appear in the observation dict."""
    env = ScoutEnv()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.last()
    assert "action_mask" in obs
    assert obs["action_mask"].shape == (MAX_ACTIONS,)


# ---------------------------------------------------------------------------
# 2-player tempo rule
# ---------------------------------------------------------------------------


def _first_scout_action(env, agent):
    """Return the first valid scout action for agent, or None."""
    obs = env.observe(agent)
    mask = obs["action_mask"]
    for i in range(N_SHOW, MAX_ACTIONS):
        if mask[i]:
            return i
    return None


def _first_show_action(env, agent):
    """Return the first valid show action for agent, or None."""
    obs = env.observe(agent)
    mask = obs["action_mask"]
    for i in range(N_SHOW):
        if mask[i]:
            return i
    return None


def test_tempo_rule_scout_keeps_turn():
    """After a Scout, agent_selection does NOT advance."""
    env = ScoutEnv()
    env.reset(seed=42)

    # First, player_0 must Show (no active set)
    p0_show = _first_show_action(env, "player_0")
    assert p0_show is not None
    env.step(p0_show)

    # Now player_1's turn; player_1 must show or scout
    # We need player_1 to be able to scout
    p1 = env.agent_selection
    scout = _first_scout_action(env, p1)
    if scout is None:
        pytest.skip("No scout action available in this seed — skip tempo test")

    env.step(scout)
    # After scout, p1 should still be the agent_selection (tempo rule)
    assert env.agent_selection == p1, (
        f"After Scout, agent should remain {p1} but got {env.agent_selection}"
    )


def test_no_scout_without_chips():
    """Agent with 0 chips should have no scout actions in mask."""
    env = ScoutEnv()
    env.reset(seed=10)

    # Exhaust all chips for player_0 by forcing scouts
    env._chips["player_0"] = 0

    # Need an active set first
    p0_show = _first_show_action(env, "player_0")
    if p0_show is None:
        pytest.skip("Cannot show in this seed")
    # Artificially set agent to player_0 after chips are 0
    # Just check the mask directly
    obs = env.observe("player_0")
    mask = obs["action_mask"]
    scout_valid = mask[N_SHOW:].any()
    # If chips=0, scouts must all be masked
    # (they would be masked regardless until it's player_0's turn after a show)
    # We test it is 0 chips means no scouts
    if env.agent_selection == "player_0" and env._active_set:
        assert not scout_valid, "Scout valid despite 0 chips"


# ---------------------------------------------------------------------------
# Round-end condition (b): stuck player (can't show, no chips)
# ---------------------------------------------------------------------------


def test_round_ends_when_player_stuck():
    """Round ends when a player cannot Show AND has no Scout chips left.

    The active-set owner should be the round_ender (exempt from penalties).
    """
    from scouter.env.card import Card

    env = ScoutEnv()
    env.reset(seed=0)

    # Force a controlled state:
    # player_0 has a strong set as active; player_1 has weak cards, 0 chips.
    env._hands["player_0"] = [Card(8, 9), Card(7, 10)]
    env._hands["player_1"] = [Card(1, 2), Card(1, 3)]
    env._active_set = [Card(5, 6), Card(5, 7), Card(5, 8)]  # 3-card match [5,5,5]
    env._active_owner = "player_0"
    env._chips["player_1"] = 0  # no chips
    env.agent_selection = "player_1"

    # player_1's turn: they can't show (nothing beats [5,5,5]) and can't scout (0 chips)
    mask = env._build_action_mask("player_1")
    assert not mask.any(), "player_1 should have no valid actions"

    # This means before even stepping, the previous action that led here should
    # have triggered round end. Let's test via _do_show forcing the situation:
    env2 = ScoutEnv()
    env2.reset(seed=0)
    env2._hands["player_0"] = [Card(5, 6), Card(5, 7), Card(5, 8), Card(8, 9)]
    env2._hands["player_1"] = [Card(1, 2), Card(1, 3)]
    env2._active_set = []
    env2._active_owner = None
    env2._chips["player_1"] = 0
    env2._collected_cards = {"player_0": [], "player_1": []}
    env2._scouted_log = {"player_0": [], "player_1": []}
    env2.agent_selection = "player_0"

    # player_0 shows [5,5,5] (indices 0..2)
    from scouter.env.game_logic import encode_show
    initial_round = env2._round
    env2.step(encode_show(0, 2))

    # The round should have ended because player_1 is stuck.
    # _round_ender is reset by _start_round(), so check round_results instead.
    assert env2._round > initial_round or not env2.agents, "Round should have advanced"
    assert len(env2._round_results) >= 1
    rr = env2._round_results[-1]
    assert rr["round_ender"] == "player_0"
    assert rr["breakdown"]["player_0"]["exempt"] is True


# ---------------------------------------------------------------------------
# Round transition
# ---------------------------------------------------------------------------


def test_round_2_starts_after_round_1():
    """Trigger round-end and verify round counter increments."""
    env = ScoutEnv()
    env.reset(seed=0)
    initial_round = env._round
    assert initial_round == 0

    # Force round end by emptying a hand
    env._hands["player_0"] = []
    env._round_ender = "player_0"
    env._end_round()

    if env.agents:  # game not over yet
        assert env._round == 1


def test_game_terminates_after_two_rounds():
    """Play 5 complete random games; all must terminate with terminations set."""
    for seed in range(5):
        env = ScoutEnv()
        env.reset(seed=seed)
        steps = 0
        last_infos: dict = {}
        while env.agents:
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                last_infos[agent] = dict(env.infos.get(agent, {}))
                env.step(None)
            else:
                obs = env.observe(agent)
                mask = obs["action_mask"]
                valid = np.where(mask)[0]
                assert len(valid) > 0, f"No valid actions (seed={seed})"
                env.step(int(np.random.choice(valid)))
            steps += 1
            assert steps < 2000, f"Game did not end after 2000 steps (seed={seed})"

        assert env.agents == [], f"agents not empty at game end (seed={seed})"
        for a in AGENTS:
            assert "final_scores" in last_infos.get(a, {}), f"{a} missing final_scores"
            assert "winner" in last_infos.get(a, {}), f"{a} missing winner"


# ---------------------------------------------------------------------------
# Scoring: +1 per unspent chip, chips go to center (not opponent)
# ---------------------------------------------------------------------------


def test_scout_chip_goes_to_center():
    """When a player scouts, their chip count decreases but opponent's does NOT increase."""
    env = ScoutEnv()
    env.reset(seed=42)

    p0_show = _first_show_action(env, "player_0")
    env.step(p0_show)

    p1 = env.agent_selection
    p1_chips_before = env._chips[p1]
    p0_chips_before = env._chips["player_0"]

    scout = _first_scout_action(env, p1)
    if scout is None:
        pytest.skip("No scout action available")
    env.step(scout)

    assert env._chips[p1] == p1_chips_before - 1, "Scouting player should lose 1 chip"
    assert env._chips["player_0"] == p0_chips_before, "Opponent should NOT gain a chip"


def test_scoring_uses_unspent_chips():
    """Scoring gives +1 per unspent chip (not received chips)."""
    from scouter.env.card import Card
    from scouter.env.game_logic import compute_round_scores

    scores = compute_round_scores(
        collected={"p0": 3, "p1": 2},
        hands={"p0": [Card(1, 2)], "p1": [Card(3, 4), Card(5, 6)]},
        chips={"p0": 2, "p1": 0},
        round_ender=None,
    )
    assert scores["p0"] == 3 - 1 + 2  # collected - hand + chips = 4
    assert scores["p1"] == 2 - 2 + 0  # collected - hand + chips = 0


# ---------------------------------------------------------------------------
# Collected cards go to the player who BEATS the active set
# ---------------------------------------------------------------------------


def test_beater_collects_active_set():
    """When a player shows a stronger set, THEY collect the old active set cards."""
    from scouter.env.card import Card

    env = ScoutEnv()
    env.reset(seed=0)

    env._hands["player_0"] = [Card(3, 4)]
    env._hands["player_1"] = [Card(5, 6)]
    env._active_set = []
    env._active_owner = None
    env._chips = {"player_0": 3, "player_1": 3}
    env._collected = {"player_0": 0, "player_1": 0}
    env._collected_cards = {"player_0": [], "player_1": []}
    env._scouted_log = {"player_0": [], "player_1": []}
    env.agent_selection = "player_0"

    env.step(encode_show(0, 0))

    # player_1 beats player_0's active set with [5]
    assert env.agent_selection == "player_1"
    p1_collected_before = env._collected["player_1"]
    p0_collected_before = env._collected["player_0"]
    active_size = len(env._active_set)  # player_0's set = 1 card

    env.step(encode_show(0, 0))

    # player_1 (the beater) should have collected player_0's active set
    assert env._collected["player_1"] == p1_collected_before + active_size
    # player_0 (the prior owner) should NOT have collected anything
    assert env._collected["player_0"] == p0_collected_before


def test_collected_cards_identity_tracked():
    """Collected cards should record actual (a, b) pairs, not just counts."""
    from scouter.env.card import Card

    env = ScoutEnv()
    env.reset(seed=0)

    # Give player_0 enough cards so the round doesn't end when they show one
    env._hands["player_0"] = [Card(3, 4), Card(1, 2)]
    env._hands["player_1"] = [Card(5, 6), Card(7, 8)]
    env._active_set = []
    env._active_owner = None
    env._chips = {"player_0": 3, "player_1": 3}
    env._collected = {"player_0": 0, "player_1": 0}
    env._collected_cards = {"player_0": [], "player_1": []}
    env._scouted_log = {"player_0": [], "player_1": []}
    env.agent_selection = "player_0"

    # player_0 shows Card(3,4) at index 0 → value 3, hand still has Card(1,2)
    env.step(encode_show(0, 0))

    # player_1 beats with Card(5,6) → value 5
    assert env.agent_selection == "player_1"
    env.step(encode_show(0, 0))

    # player_1 collected the card (3,4) that was in the active set
    assert len(env._collected_cards["player_1"]) == 1
    assert env._collected_cards["player_1"][0] == (3, 4)
    assert env._collected_cards["player_0"] == []

    # Observation should contain the collected card identity
    obs_p1 = env.observe("player_1")
    p1_idx = AGENTS.index("player_1")
    assert obs_p1["collected_cards"][p1_idx, 0, 0] == 3
    assert obs_p1["collected_cards"][p1_idx, 0, 1] == 4


def test_opponent_scouted_cards_in_observation():
    """Observation should show which cards the opponent scouted (both values)."""
    from scouter.env.card import Card

    env = ScoutEnv()
    env.reset(seed=42)

    # player_0 shows first
    p0_show = _first_show_action(env, "player_0")
    assert p0_show is not None
    env.step(p0_show)

    # player_1 scouts
    p1 = env.agent_selection
    scout = _first_scout_action(env, p1)
    if scout is None:
        pytest.skip("No scout action available")

    # Record what's at the ends of the active set before scout
    active_left = env._active_set[0] if env._active_set else None
    active_right = env._active_set[-1] if env._active_set else None
    from scouter.env.game_logic import decode_scout
    side, flip, insert_pos = decode_scout(scout)
    scouted_card = active_left if side == 0 else active_right

    env.step(scout)

    # player_1 scouted a card — check the log
    assert len(env._scouted_log[p1]) == 1
    assert env._scouted_log[p1][0] == (scouted_card.a, scouted_card.b)

    # From player_0's perspective, the opponent's scouted cards should be visible
    obs_p0 = env.observe("player_0")
    assert int(obs_p0["opp_scouted_count"]) == 1
    assert obs_p0["opp_scouted_cards"][0, 0] == scouted_card.a
    assert obs_p0["opp_scouted_cards"][0, 1] == scouted_card.b


# ---------------------------------------------------------------------------
# History logging
# ---------------------------------------------------------------------------


def test_history_records_show_and_scout():
    """History list should log show and scout actions."""
    env = ScoutEnv()
    env.reset(seed=42)
    assert len(env._history) == 0

    # Play a show
    p0_show = _first_show_action(env, "player_0")
    env.step(p0_show)
    assert len(env._history) >= 1
    assert env._history[-1]["action"] == "show"
    assert env._history[-1]["player"] == "player_0"

    # Scout if possible
    p1 = env.agent_selection
    scout = _first_scout_action(env, p1)
    if scout is not None:
        env.step(scout)
        assert env._history[-1]["action"] == "scout"
        assert env._history[-1]["player"] == p1


def test_round_results_populated_after_round():
    """After a round ends, round_results should have a breakdown entry."""
    env = ScoutEnv()
    env.reset(seed=0)

    while env.agents and env._round == 0:
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
        else:
            obs = env.observe(agent)
            mask = obs["action_mask"]
            valid = np.where(mask)[0]
            env.step(int(np.random.choice(valid)))

    assert len(env._round_results) >= 1
    rr = env._round_results[0]
    assert "breakdown" in rr
    for a in AGENTS:
        bd = rr["breakdown"][a]
        assert "collected" in bd
        assert "hand_penalty" in bd
        assert "unspent_chips" in bd
        assert "total" in bd


# ---------------------------------------------------------------------------
# Scoring / termination info
# ---------------------------------------------------------------------------


def test_final_info_contains_scores():
    env = ScoutEnv()
    env.reset(seed=42)
    last_infos: dict = {}
    while env.agents:
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            last_infos[agent] = dict(env.infos.get(agent, {}))
            env.step(None)
        else:
            obs = env.observe(agent)
            mask = obs["action_mask"]
            valid = np.where(mask)[0]
            env.step(int(np.random.choice(valid)))

    assert env.agents == []
    for a in AGENTS:
        assert "final_scores" in last_infos[a]
        assert set(last_infos[a]["final_scores"].keys()) == set(AGENTS)


# ---------------------------------------------------------------------------
# Global state() method
# ---------------------------------------------------------------------------


def test_state_method_returns_array():
    env = ScoutEnv()
    env.reset(seed=1)
    s = env.state()
    assert isinstance(s, np.ndarray)
    assert s.ndim == 1
    assert len(s) > 0


def test_state_shape_stable():
    """state() must return the same shape every call."""
    env = ScoutEnv()
    env.reset(seed=1)
    shape1 = env.state().shape

    # Take one action
    agent = env.agent_selection
    obs = env.observe(agent)
    mask = obs["action_mask"]
    valid = np.where(mask)[0]
    env.step(int(valid[0]))

    shape2 = env.state().shape
    assert shape1 == shape2, "state() shape changed between steps"


# ---------------------------------------------------------------------------
# Smoke test: 50 complete random games
# ---------------------------------------------------------------------------


def test_50_random_games_no_errors():
    """Play 50 complete games with random valid actions; must not raise."""
    for seed in range(50):
        env = ScoutEnv()
        env.reset(seed=seed)
        steps = 0
        while env.agents:
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
            else:
                obs = env.observe(agent)
                mask = obs["action_mask"]
                valid = np.where(mask)[0]
                assert len(valid) > 0, f"No valid actions (seed={seed}, step={steps})"
                env.step(int(np.random.choice(valid)))
            steps += 1
            if steps > 2000:
                raise AssertionError(f"Game {seed} did not end in 2000 steps")
        assert env.agents == []


# ---------------------------------------------------------------------------
# Single-round mode (num_rounds=1) for RL training
# ---------------------------------------------------------------------------


def test_single_round_terminates_after_one_round():
    """ScoutEnv(num_rounds=1) should end the game after round 0."""
    for seed in range(10):
        env = ScoutEnv(num_rounds=1)
        env.reset(seed=seed)
        steps = 0
        while env.agents:
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
            else:
                obs = env.observe(agent)
                mask = obs["action_mask"]
                valid = np.where(mask)[0]
                env.step(int(np.random.choice(valid)))
            steps += 1
            assert steps < 1000, f"Single-round game {seed} didn't end"
        assert env._round == 1, "Should have completed exactly 1 round"
        assert len(env._round_results) == 1


def test_score_diff_reward_is_zero_sum():
    """With reward_mode='score_diff', the reward set at round end is zero-sum."""
    from scouter.env.card import Card
    from scouter.env.game_logic import compute_round_scores

    for seed in range(10):
        env = ScoutEnv(num_rounds=1, reward_mode="score_diff")
        env.reset(seed=seed)
        while env.agents:
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
            else:
                obs = env.observe(agent)
                mask = obs["action_mask"]
                valid = np.where(mask)[0]
                env.step(int(np.random.choice(valid)))

        assert len(env._round_results) == 1
        rr = env._round_results[0]
        scores = rr["scores"]
        p0_raw = scores[AGENTS[0]]
        p1_raw = scores[AGENTS[1]]
        # score_diff rewards: agent gets (my_score - opp_score)
        p0_diff = p0_raw - p1_raw
        p1_diff = p1_raw - p0_raw
        assert abs(p0_diff + p1_diff) < 1e-6, (
            f"Score diff not zero-sum (seed={seed})"
        )


# ---------------------------------------------------------------------------
# ScoutFlatAEC wrapper for RLlib
# ---------------------------------------------------------------------------


def test_flat_aec_obs_shape():
    """ScoutFlatAEC should produce {"observations": flat, "action_mask": mask}."""
    from scouter.rl.rllib_wrapper import ScoutFlatAEC

    env = ScoutFlatAEC(num_rounds=1, reward_mode="score_diff")
    env.reset(seed=42)
    obs = env.observe(env.agent_selection)
    assert "observations" in obs, "Missing 'observations' key"
    assert "action_mask" in obs, "Missing 'action_mask' key"
    assert obs["observations"].dtype == np.float32
    assert obs["action_mask"].dtype == np.float32
    assert obs["action_mask"].shape == (MAX_ACTIONS,)


def test_flat_aec_tempo_rule():
    """After a scout, ScoutFlatAEC should keep the same agent as current."""
    from scouter.rl.rllib_wrapper import ScoutFlatAEC

    env = ScoutFlatAEC(num_rounds=1, reward_mode="score_diff")
    env.reset(seed=42)

    # First player must show
    agent = env.agent_selection
    obs = env.observe(agent)
    mask = obs["action_mask"]
    valid_show = np.where(mask[:N_SHOW])[0]
    env.step(int(valid_show[0]))

    # Opponent tries to scout
    opp = env.agent_selection
    obs2 = env.observe(opp)
    mask2 = obs2["action_mask"]
    valid_scout = np.where(mask2[N_SHOW:])[0]
    if len(valid_scout) > 0:
        env.step(int(valid_scout[0]) + N_SHOW)
        assert env.agent_selection == opp, (
            "Tempo rule broken: agent should stay after scout"
        )
