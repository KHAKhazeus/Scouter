"""Managed deployed-agent pool for serving checkpoint policies in game sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.columns import Columns


@dataclass
class LoadedAgent:
    agent_id: str
    checkpoint_path: Path
    algo: Algorithm


class AgentPool:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._loaded: dict[str, LoadedAgent] = {}

    def list_deployed(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for p in sorted(self.root.iterdir(), key=lambda x: x.name):
            if not p.is_dir():
                continue
            manifest_path = p / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append(
                {
                    "agent_id": manifest.get("agent_id", p.name),
                    "name": manifest.get("name", p.name),
                    "created_at": manifest.get("created_at"),
                    "checkpoint_path": manifest.get("checkpoint_path"),
                    "source": manifest.get("source", {}),
                }
            )
        return out

    def _manifest(self, agent_id: str) -> dict[str, Any]:
        manifest_path = self.root / agent_id / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Agent manifest not found: {manifest_path}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def load(self, agent_id: str) -> LoadedAgent:
        if agent_id in self._loaded:
            return self._loaded[agent_id]

        manifest = self._manifest(agent_id)
        checkpoint_path = Path(manifest["checkpoint_path"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

        algo = Algorithm.from_checkpoint(str(checkpoint_path))
        loaded = LoadedAgent(agent_id=agent_id, checkpoint_path=checkpoint_path, algo=algo)
        self._loaded[agent_id] = loaded
        return loaded

    def compute_action(
        self,
        *,
        agent_id: str,
        obs: dict[str, np.ndarray],
        policy_id: str = "shared_policy",
    ) -> int:
        loaded = self.load(agent_id)
        module = loaded.algo.get_module(policy_id)

        obs_batch = {
            "observations": torch.from_numpy(obs["observations"][None, :]).float(),
            "action_mask": torch.from_numpy(obs["action_mask"][None, :]).float(),
        }
        with torch.no_grad():
            out = module.forward_inference({Columns.OBS: obs_batch})
        logits = out[Columns.ACTION_DIST_INPUTS]
        action_idx = int(torch.argmax(logits[0]).item())

        if action_idx < 0 or action_idx >= len(obs["action_mask"]):
            valid = np.where(obs["action_mask"] > 0)[0]
            return int(valid[0]) if len(valid) else 0
        if obs["action_mask"][action_idx] <= 0:
            valid = np.where(obs["action_mask"] > 0)[0]
            return int(valid[0]) if len(valid) else 0
        return action_idx

    def close(self) -> None:
        for loaded in self._loaded.values():
            try:
                loaded.algo.stop()
            except Exception:
                pass
        self._loaded.clear()
