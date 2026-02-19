"""RLlib integration for ScoutEnv.

Uses RLlib's built-in PettingZooEnv wrapper with a thin AEC adapter that
flattens observations into the ``{"observations": flat, "action_mask": mask}``
format expected by RLlib's action-masking RLModules.

Provides:
- ``ScoutFlatAEC``          -- AEC wrapper that flattens obs for RLlib.
- ``register_scout_env()``  -- register the env with Ray's tune registry.
- ``get_ppo_config()``      -- example PPO self-play config with action masking.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from scouter.env.game_logic import MAX_ACTIONS
from scouter.env.scout_env import ScoutEnv


# ---------------------------------------------------------------------------
# AEC wrapper: flatten Dict obs -> {"observations": flat, "action_mask": mask}
# ---------------------------------------------------------------------------


class ScoutFlatAEC(AECEnv):
    """Thin AEC wrapper that restructures ScoutEnv observations.

    RLlib's ``PettingZooEnv`` passes observations straight through, and
    RLlib's ``ActionMaskingTorchRLModule`` expects a Dict with two keys:
        ``"observations"`` -- a flat float32 array of all game state features
        ``"action_mask"``  -- a float32 array (0/1) of legal actions

    This wrapper sits between ``ScoutEnv`` and ``PettingZooEnv`` to perform
    that restructuring without modifying ScoutEnv itself.
    """

    metadata = {
        "render_modes": ["ansi"],
        "name": "scout_flat_v0",
        "is_parallelizable": False,
    }

    def __init__(self, num_rounds: int = 1, reward_mode: str = "score_diff"):
        super().__init__()
        self._inner = ScoutEnv(num_rounds=num_rounds, reward_mode=reward_mode)

        self.possible_agents = list(self._inner.possible_agents)
        self.agents = list(self._inner.agents)

        flat_size = self._compute_flat_size()
        obs_space = spaces.Dict({
            "observations": spaces.Box(
                -np.inf, np.inf, shape=(flat_size,), dtype=np.float32
            ),
            "action_mask": spaces.Box(
                0.0, 1.0, shape=(MAX_ACTIONS,), dtype=np.float32
            ),
        })
        act_space = spaces.Discrete(MAX_ACTIONS)

        self.observation_spaces = {a: obs_space for a in self.possible_agents}
        self.action_spaces = {a: act_space for a in self.possible_agents}

    def _compute_flat_size(self) -> int:
        self._inner.reset(seed=0)
        sample = self._inner.observe(self.possible_agents[0])
        return sum(
            np.asarray(v).size for k, v in sample.items() if k != "action_mask"
        )

    @staticmethod
    def _flatten_obs(raw: dict) -> dict[str, np.ndarray]:
        parts = []
        for k, v in raw.items():
            if k == "action_mask":
                continue
            parts.append(np.asarray(v, dtype=np.float32).flatten())
        return {
            "observations": np.concatenate(parts),
            "action_mask": np.asarray(raw["action_mask"], dtype=np.float32),
        }

    # --- AECEnv interface (delegate to inner, transform obs) ---

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    @property
    def agent_selection(self):
        return self._inner.agent_selection

    @agent_selection.setter
    def agent_selection(self, val):
        self._inner.agent_selection = val

    @property
    def terminations(self):
        return self._inner.terminations

    @property
    def truncations(self):
        return self._inner.truncations

    @property
    def rewards(self):
        return self._inner.rewards

    @property
    def infos(self):
        return self._inner.infos

    @property
    def _cumulative_rewards(self):
        return self._inner._cumulative_rewards

    def reset(self, seed=None, options=None):
        self._inner.reset(seed=seed, options=options)
        self.agents = list(self._inner.agents)

    def observe(self, agent):
        raw = self._inner.observe(agent)
        return self._flatten_obs(raw)

    def step(self, action):
        self._inner.step(action)
        self.agents = list(self._inner.agents)

    def last(self, observe=True):
        agent = self._inner.agent_selection
        obs = self.observe(agent) if observe else None
        reward = self._inner._cumulative_rewards.get(agent, 0.0)
        terminated = self._inner.terminations.get(agent, False)
        truncated = self._inner.truncations.get(agent, False)
        info = dict(self._inner.infos.get(agent, {}))
        self._inner._cumulative_rewards[agent] = 0.0
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._inner.render()

    def close(self):
        self._inner.close()

    @property
    def unwrapped(self):
        return self._inner


# ---------------------------------------------------------------------------
# Registration + config helpers
# ---------------------------------------------------------------------------


def register_scout_env(env_name: str = "scout_v0") -> None:
    """Register the Scout env with Ray's tune registry."""
    from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
    from ray.tune.registry import register_env

    def _env_creator(config):
        nr = config.get("num_rounds", 1)
        rm = config.get("reward_mode", "score_diff")
        return PettingZooEnv(ScoutFlatAEC(num_rounds=nr, reward_mode=rm))

    register_env(env_name, _env_creator)


def get_ppo_config(
    env_name: str = "scout_v0",
    num_env_runners: int = 2,
    train_batch_size: int = 4000,
    sgd_minibatch_size: int = 128,
    num_sgd_iter: int = 10,
    lr: float = 3e-4,
) -> Any:
    """Return a PPO config for self-play Scout training with action masking.

    Uses the new API stack (RLModule + Learner) with
    ``ActionMaskingTorchRLModule`` from RLlib's examples.
    Both agents share one policy ("shared_policy"), following the
    waterworld parameter-sharing pattern.
    """
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
        ActionMaskingTorchRLModule,
    )

    register_scout_env(env_name)

    config = (
        PPOConfig()
        .environment(
            env=env_name,
            env_config={"num_rounds": 1, "reward_mode": "score_diff"},
        )
        .env_runners(num_env_runners=num_env_runners)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "shared_policy": RLModuleSpec(
                        module_class=ActionMaskingTorchRLModule,
                        model_config={
                            "head_fcnet_hiddens": [256, 256],
                            "head_fcnet_activation": "relu",
                        },
                    ),
                },
            ),
        )
        .training(
            train_batch_size_per_learner=train_batch_size,
            minibatch_size=sgd_minibatch_size,
            num_epochs=num_sgd_iter,
            lr=lr,
            vf_loss_coeff=0.5,
        )
    )
    return config
