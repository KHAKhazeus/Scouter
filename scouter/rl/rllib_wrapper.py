"""RLlib integration for ScoutEnv with PettingZooEnv + action masking."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper

from scouter.env.game_logic import MAX_ACTIONS
from scouter.env.scout_env import ScoutEnv

from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import TensorType

torch, _ = try_import_torch()


# Keep a fixed ordering so flattening is deterministic across Python versions.
FLAT_OBS_KEYS: tuple[str, ...] = (
    "hand",
    "hand_size",
    "active_set",
    "active_set_size",
    "active_set_owner",
    "opp_hand_size",
    "scout_chips",
    "collected_counts",
    "collected_cards",
    "opp_scouted_cards",
    "opp_scouted_count",
    "round_num",
)


class FlatObsWrapper(BaseWrapper):
    """Restructure ScoutEnv observations for RLlib action masking."""

    def __init__(self, env: ScoutEnv):
        super().__init__(env)
        self._flat_obs_size = self._compute_flat_obs_size()
        self._obs_space = spaces.Dict(
            {
                "observations": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._flat_obs_size,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(MAX_ACTIONS,),
                    dtype=np.float32,
                ),
            }
        )

    def _compute_flat_obs_size(self) -> int:
        obs_space = self.env.observation_space(self.env.possible_agents[0])
        size = 0
        for key in FLAT_OBS_KEYS:
            space = obs_space[key]
            shape = getattr(space, "shape", ())
            size += int(np.prod(shape, dtype=np.int64)) if shape else 1
        return size

    def observation_space(self, agent: str) -> spaces.Dict:
        return self._obs_space

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        raw = self.env.observe(agent)
        missing = [k for k in FLAT_OBS_KEYS if k not in raw]
        if missing:
            raise KeyError(f"Missing keys in observation for flattening: {missing}")

        parts = [np.asarray(raw[k], dtype=np.float32).reshape(-1) for k in FLAT_OBS_KEYS]
        return {
            "observations": np.concatenate(parts, dtype=np.float32),
            "action_mask": np.asarray(raw["action_mask"], dtype=np.float32),
        }


class ScoutActionMaskRLModule(DefaultPPOTorchRLModule):
    """PPO RLModule that applies action masking to action logits."""

    def __init__(
        self,
        *,
        observation_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
        inference_only: bool | None = None,
        learner_only: bool = False,
        model_config: dict | None = None,
        catalog_class=None,
        **kwargs,
    ):
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError(
                "ScoutActionMaskRLModule requires Dict obs space with "
                "'observations' and 'action_mask' keys."
            )
        self.observation_space_with_mask = observation_space
        super().__init__(
            observation_space=observation_space["observations"],
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_config,
            catalog_class=catalog_class,
            **kwargs,
        )

    @override(DefaultPPOTorchRLModule)
    def setup(self):
        super().setup()
        # PPO internals need Box obs space during network construction, but
        # the runtime batches still carry Dict observations with an action mask.
        self.observation_space = self.observation_space_with_mask
        self._checked_observations = False

    def _check_batch(self, batch: dict[str, TensorType]) -> None:
        if self._checked_observations:
            return

        obs = batch.get(Columns.OBS)
        if not isinstance(obs, dict):
            raise ValueError(
                "Expected dict observations with keys 'observations' and 'action_mask'."
            )
        if "action_mask" not in obs:
            raise ValueError("Missing 'action_mask' in observation dict.")
        if "observations" not in obs:
            raise ValueError("Missing 'observations' in observation dict.")
        self._checked_observations = True

    def _preprocess_batch(
        self, batch: dict[str, TensorType]
    ) -> tuple[TensorType, dict[str, TensorType]]:
        self._check_batch(batch)
        obs = batch[Columns.OBS]
        action_mask = obs["action_mask"]
        new_batch = batch.copy()
        new_batch[Columns.OBS] = obs["observations"]
        return action_mask, new_batch

    def _mask_logits(
        self, out: dict[str, TensorType], action_mask: TensorType
    ) -> dict[str, TensorType]:
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        out[Columns.ACTION_DIST_INPUTS] = out[Columns.ACTION_DIST_INPUTS] + inf_mask
        return out

    @override(DefaultPPOTorchRLModule)
    def _forward(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        action_mask, new_batch = self._preprocess_batch(batch)
        out = super()._forward(new_batch, **kwargs)
        return self._mask_logits(out, action_mask)

    @override(DefaultPPOTorchRLModule)
    def _forward_train(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        action_mask, new_batch = self._preprocess_batch(batch)
        out = super()._forward_train(new_batch, **kwargs)
        return self._mask_logits(out, action_mask)

    @override(ValueFunctionAPI)
    def compute_values(self, batch: dict[str, Any], embeddings=None):
        if isinstance(batch.get(Columns.OBS), dict):
            _, batch = self._preprocess_batch(batch)
        return super().compute_values(batch, embeddings)


def register_scout_env(env_name: str = "scout_v0") -> None:
    """Register Scout as an RLlib environment via PettingZooEnv."""
    from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
    from ray.tune.registry import register_env

    def _env_creator(config: dict[str, Any]):
        config = config or {}
        num_rounds = int(config.get("num_rounds", 1))
        reward_mode = str(config.get("reward_mode", "score_diff"))
        return PettingZooEnv(
            FlatObsWrapper(ScoutEnv(num_rounds=num_rounds, reward_mode=reward_mode))
        )

    register_env(env_name, _env_creator)
