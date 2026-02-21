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

from scouter.rl.rllib_wrapper import register_scout_env


@dataclass
class LoadedAgent:
    agent_id: str
    checkpoint_path: Path
    algo: Algorithm
    device: torch.device


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

        # Checkpoints reference scout_v0; ensure env is registered in serve process.
        register_scout_env("scout_v0")
        algo = Algorithm.from_checkpoint(checkpoint_path.resolve().as_uri())
        module = algo.get_module("shared_policy")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        module.to(device)
        module.eval()
        loaded = LoadedAgent(
            agent_id=agent_id,
            checkpoint_path=checkpoint_path,
            algo=algo,
            device=device,
        )
        self._loaded[agent_id] = loaded
        return loaded

    def compute_action(
        self,
        *,
        agent_id: str,
        obs: dict[str, np.ndarray],
        policy_id: str = "shared_policy",
        return_info: bool = False,
    ) -> int | tuple[int, dict[str, Any]]:
        loaded = self.load(agent_id)
        module = loaded.algo.get_module(policy_id)

        obs_batch = {
            "observations": torch.from_numpy(obs["observations"][None, :]).float().to(loaded.device),
            "action_mask": torch.from_numpy(obs["action_mask"][None, :]).float().to(loaded.device),
        }
        with torch.no_grad():
            out = module.forward_inference({Columns.OBS: obs_batch})
        logits = out[Columns.ACTION_DIST_INPUTS]
        raw_argmax_action = int(torch.argmax(logits[0]).item())
        action_idx = raw_argmax_action

        if action_idx < 0 or action_idx >= len(obs["action_mask"]):
            valid = np.where(obs["action_mask"] > 0)[0]
            chosen = int(valid[0]) if len(valid) else 0
            if return_info:
                return chosen, {
                    "source": "model_corrected_out_of_range",
                    "raw_argmax_action": raw_argmax_action,
                    "device": str(loaded.device),
                }
            return chosen
        if obs["action_mask"][action_idx] <= 0:
            valid = np.where(obs["action_mask"] > 0)[0]
            chosen = int(valid[0]) if len(valid) else 0
            if return_info:
                return chosen, {
                    "source": "model_corrected_masked",
                    "raw_argmax_action": raw_argmax_action,
                    "device": str(loaded.device),
                }
            return chosen
        if return_info:
            return action_idx, {
                "source": "model",
                "raw_argmax_action": raw_argmax_action,
                "device": str(loaded.device),
            }
        return action_idx

    def close(self) -> None:
        for loaded in self._loaded.values():
            try:
                loaded.algo.stop()
            except Exception:
                pass
        self._loaded.clear()
