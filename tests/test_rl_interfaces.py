"""Tests for RL training interfaces: obs_formatter, action_parser, grpo_rollout."""

import json

import numpy as np
import pytest

from scouter.env.game_logic import (
    MAX_ACTIONS,
    N_SHOW,
    encode_scout,
    encode_show,
)
from scouter.env.scout_env import AGENTS, ScoutEnv
from scouter.rl.action_parser import parse_action
from scouter.rl.obs_formatter import format_observation


# ---------------------------------------------------------------------------
# obs_formatter
# ---------------------------------------------------------------------------


def _get_obs(seed=0):
    env = ScoutEnv()
    env.reset(seed=seed)
    agent = env.agent_selection
    obs = env.observe(agent)
    return obs, agent, env


def test_format_observation_returns_string():
    obs, agent, env = _get_obs()
    prompt = format_observation(obs, agent, active_owner=None)
    assert isinstance(prompt, str)
    assert len(prompt) > 50


def test_format_observation_contains_round():
    obs, agent, env = _get_obs()
    prompt = format_observation(obs, agent)
    assert "Round 1" in prompt or "round_num" in prompt.lower() or "Round" in prompt


def test_format_observation_contains_agent_name():
    obs, agent, env = _get_obs()
    prompt = format_observation(obs, agent)
    assert agent in prompt


def test_format_observation_contains_chips():
    obs, agent, env = _get_obs()
    prompt = format_observation(obs, agent)
    assert "chip" in prompt.lower() or "3/3" in prompt


def test_format_observation_contains_hand_info():
    obs, agent, env = _get_obs()
    prompt = format_observation(obs, agent)
    assert "hand" in prompt.lower() or "card" in prompt.lower()


def test_format_observation_contains_legal_actions():
    obs, agent, env = _get_obs()
    prompt = format_observation(obs, agent)
    # Should mention show actions
    assert "SHOW" in prompt or "show" in prompt.lower()


def test_format_observation_changes_between_turns():
    env = ScoutEnv()
    env.reset(seed=5)
    agent0 = env.agent_selection
    obs0 = env.observe(agent0)
    prompt0 = format_observation(obs0, agent0)

    # Take a valid show action
    mask = obs0["action_mask"]
    valid = np.where(mask)[0]
    env.step(int(valid[0]))

    agent1 = env.agent_selection
    obs1 = env.observe(agent1)
    prompt1 = format_observation(obs1, agent1)

    assert prompt0 != prompt1


# ---------------------------------------------------------------------------
# action_parser
# ---------------------------------------------------------------------------


def test_parse_show_json():
    text = '{"action": "show", "start": 1, "end": 3}'
    action = parse_action(text, hand_size=11)
    assert action == encode_show(0, 2)  # 1-based → 0-based


def test_parse_show_single_json():
    text = '{"action": "show", "start": 5, "end": 5}'
    action = parse_action(text, hand_size=11)
    assert action == encode_show(4, 4)


def test_parse_scout_json_left_no_flip():
    text = '{"action": "scout", "side": "left", "flip": false, "insert": 3}'
    action = parse_action(text, hand_size=11)
    assert action == encode_scout(0, 0, 2)  # 1-based insert → 0-based


def test_parse_scout_json_right_flip():
    text = '{"action": "scout", "side": "right", "flip": true, "insert": 1}'
    action = parse_action(text, hand_size=11)
    assert action == encode_scout(1, 1, 0)


def test_parse_invalid_returns_minus_one():
    text = "I have no idea what to do"
    action = parse_action(text, hand_size=11)
    assert action == -1


def test_parse_malformed_json_falls_back():
    text = '{"action": "show", start: 1, "end": 2}'  # invalid JSON
    # Should try regex fallback or return -1 — just check no exception
    action = parse_action(text, hand_size=11)
    assert isinstance(action, int)


def test_parse_regex_show():
    text = "I will show cards 2-4"
    action = parse_action(text, hand_size=11)
    assert action == encode_show(1, 3)


def test_parse_regex_scout():
    text = "SCOUT left card, insert at pos 5"
    action = parse_action(text, hand_size=11)
    assert action == encode_scout(0, 0, 4)


def test_parse_out_of_bounds_returns_minus_one():
    text = '{"action": "show", "start": 20, "end": 25}'
    action = parse_action(text, hand_size=11)
    assert action == -1


def test_parse_all_valid_show_actions():
    """Every valid show action round-trips through JSON parse."""
    from scouter.env.game_logic import MAX_HAND

    for start in range(1, 5):
        for end in range(start, 5):
            text = json.dumps({"action": "show", "start": start, "end": end})
            action = parse_action(text, hand_size=11)
            assert action == encode_show(start - 1, end - 1), \
                f"Round-trip failed for show start={start} end={end}"


def test_parse_all_scout_json():
    for side in ("left", "right"):
        for flip in (True, False):
            for insert in range(1, 5):
                text = json.dumps({
                    "action": "scout",
                    "side": side,
                    "flip": flip,
                    "insert": insert,
                })
                action = parse_action(text, hand_size=11)
                expected_side = 0 if side == "left" else 1
                expected = encode_scout(expected_side, int(flip), insert - 1)
                assert action == expected


# ---------------------------------------------------------------------------
# grpo_rollout (smoke test — no real LLM needed)
# ---------------------------------------------------------------------------


def test_collect_grpo_dataset_structure():
    """collect_grpo_dataset returns records with correct keys."""
    from scouter.rl.grpo_rollout import collect_grpo_dataset

    def dummy_policy(prompt: str) -> str:
        # Return valid JSON that won't always parse — that's fine,
        # the fallback to random will kick in.
        return '{"action": "show", "start": 1, "end": 1}'

    records = collect_grpo_dataset(num_episodes=2, policy_fn=dummy_policy, seed=0)

    assert isinstance(records, list)
    assert len(records) > 0

    for r in records:
        assert "prompt" in r
        assert "completion" in r
        assert "reward" in r
        assert isinstance(r["prompt"], str)
        assert isinstance(r["completion"], str)
        assert isinstance(r["reward"], float)


def test_collect_grpo_dataset_custom_reward():
    from scouter.rl.grpo_rollout import collect_grpo_dataset

    def policy(prompt):
        return '{"action": "show", "start": 1, "end": 1}'

    def reward_fn(info):
        return 42.0

    records = collect_grpo_dataset(
        num_episodes=1,
        policy_fn=policy,
        reward_fn=reward_fn,
        seed=1,
    )
    assert all(r["reward"] == 42.0 for r in records)


# ---------------------------------------------------------------------------
# RLlib wrapper (does not require ray to be installed)
# ---------------------------------------------------------------------------


def test_rllib_wrapper_raises_without_ray():
    """ScoutRllibEnv raises ImportError gracefully when ray is not installed."""
    try:
        import ray  # noqa: F401
        pytest.skip("ray is installed; skipping ImportError test")
    except ImportError:
        pass

    from scouter.rl.rllib_wrapper import ScoutRllibEnv

    with pytest.raises(ImportError, match="ray\\[rllib\\]"):
        ScoutRllibEnv()
