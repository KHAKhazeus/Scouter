"""Tests for RLlib integration utilities in scouter.rl.rllib_wrapper."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from scouter.env.game_logic import MAX_ACTIONS, N_SHOW
from scouter.env.scout_env import ScoutEnv
from scouter.rl.rllib_wrapper import (
    FLAT_OBS_KEYS,
    FlatObsWrapper,
    ScoutActionMaskRLModule,
    register_scout_env,
)


def _first_show_action(env, agent: str) -> int | None:
    obs = env.observe(agent)
    valid_show = np.where(obs["action_mask"][:N_SHOW])[0]
    if len(valid_show) == 0:
        return None
    return int(valid_show[0])


def _first_scout_action(env, agent: str) -> int | None:
    obs = env.observe(agent)
    valid_scout = np.where(obs["action_mask"][N_SHOW:])[0]
    if len(valid_scout) == 0:
        return None
    return int(valid_scout[0]) + N_SHOW


def test_flat_obs_wrapper_obs_contract_and_size():
    env = FlatObsWrapper(ScoutEnv(num_rounds=1, reward_mode="score_diff"))
    env.reset(seed=42)
    obs = env.observe(env.agent_selection)

    assert set(obs.keys()) == {"observations", "action_mask"}
    assert obs["observations"].dtype == np.float32
    assert obs["action_mask"].dtype == np.float32
    assert obs["action_mask"].shape == (MAX_ACTIONS,)

    expected_size = sum(np.asarray(env.env.observe(env.agent_selection)[k]).size for k in FLAT_OBS_KEYS)
    assert obs["observations"].shape == (expected_size,)
    assert env.observation_space(env.agent_selection).contains(obs)


def test_flat_obs_wrapper_tempo_rule_after_scout():
    env = FlatObsWrapper(ScoutEnv(num_rounds=1, reward_mode="score_diff"))
    env.reset(seed=42)

    first_agent = env.agent_selection
    show = _first_show_action(env, first_agent)
    assert show is not None
    env.step(show)

    scout_agent = env.agent_selection
    scout = _first_scout_action(env, scout_agent)
    if scout is None:
        pytest.skip("No scout action available for this seed")
    env.step(scout)
    assert env.agent_selection == scout_agent


def test_register_scout_env_creates_pettingzoo_env():
    ray = pytest.importorskip("ray")
    register_scout_env("scout_v0_test")

    from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
    from ray.tune.registry import ENV_CREATOR, _global_registry

    creator = _global_registry.get(ENV_CREATOR, "scout_v0_test")
    env = creator({"num_rounds": 1, "reward_mode": "score_diff"})
    try:
        assert isinstance(env, PettingZooEnv)
    finally:
        env.close()


def test_action_mask_rlmodule_masks_invalid_logits():
    torch = pytest.importorskip("torch")

    obs_space = spaces.Dict(
        {
            "observations": spaces.Box(-np.inf, np.inf, shape=(188,), dtype=np.float32),
            "action_mask": spaces.Box(0.0, 1.0, shape=(MAX_ACTIONS,), dtype=np.float32),
        }
    )
    module = ScoutActionMaskRLModule(
        observation_space=obs_space,
        action_space=spaces.Discrete(MAX_ACTIONS),
        model_config={"head_fcnet_hiddens": [64, 64], "head_fcnet_activation": "relu"},
    )

    from ray.rllib.core.columns import Columns

    observations = torch.zeros((1, 188), dtype=torch.float32)
    action_mask = torch.ones((1, MAX_ACTIONS), dtype=torch.float32)
    action_mask[:, 0] = 0.0
    out = module._forward({Columns.OBS: {"observations": observations, "action_mask": action_mask}})
    logits = out[Columns.ACTION_DIST_INPUTS].detach().cpu().numpy()

    assert logits[0, 0] < -1e20
    assert np.isfinite(logits[0, 1])
