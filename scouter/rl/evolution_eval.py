"""Evaluation helpers for snapshot evolution validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule

from scouter.env.scout_env import AGENTS, ScoutEnv
from scouter.rl.metrics import utc_now_iso
from scouter.rl.rllib_wrapper import FlatObsWrapper


PolicyFn = Callable[[dict[str, np.ndarray], str, np.random.Generator], int]
SeatMode = str


def _pick_valid_action(obs: dict[str, np.ndarray], rng: np.random.Generator) -> int:
    valid = np.where(obs["action_mask"] > 0.0)[0]
    if len(valid) == 0:
        return 0
    return int(rng.choice(valid))


def random_policy(obs: dict[str, np.ndarray], _agent: str, rng: np.random.Generator) -> int:
    return _pick_valid_action(obs, rng)


def make_algo_policy(algo: Algorithm, *, policy_id: str = "shared_policy") -> PolicyFn:
    """Create a policy function from an RLlib Algorithm checkpoint/instance."""
    module = algo.get_module(policy_id)

    def _policy(obs: dict[str, np.ndarray], _agent: str, rng: np.random.Generator) -> int:
        obs_batch = {
            "observations": torch.from_numpy(obs["observations"][None, :]).float(),
            "action_mask": torch.from_numpy(obs["action_mask"][None, :]).float(),
        }
        with torch.no_grad():
            out = module.forward_inference({Columns.OBS: obs_batch})

        logits = out[Columns.ACTION_DIST_INPUTS]
        if logits.ndim != 2 or logits.shape[0] != 1:
            return _pick_valid_action(obs, rng)
        action_idx = int(torch.argmax(logits[0]).item())

        if action_idx < 0 or action_idx >= len(obs["action_mask"]):
            return _pick_valid_action(obs, rng)
        if obs["action_mask"][action_idx] <= 0.0:
            return _pick_valid_action(obs, rng)
        return action_idx

    return _policy


def resolve_module_checkpoint(
    checkpoint_path: Path,
    *,
    policy_id: str = "shared_policy",
) -> Path:
    ckpt = checkpoint_path.resolve()
    return ckpt / "learner_group" / "learner" / "rl_module" / policy_id


def load_module_from_checkpoint(
    checkpoint_path: Path,
    *,
    policy_id: str = "shared_policy",
    device: str = "cpu",
) -> RLModule:
    module_ckpt = resolve_module_checkpoint(checkpoint_path, policy_id=policy_id)
    if not module_ckpt.exists():
        raise FileNotFoundError(f"RLModule checkpoint not found: {module_ckpt}")
    module = RLModule.from_checkpoint(module_ckpt)
    if device and device != "auto":
        module.to(device)
    return module


def module_device(module: RLModule) -> str:
    try:
        return str(next(module.parameters()).device)
    except (StopIteration, TypeError):
        return "unknown"


def make_module_policy(module: RLModule) -> PolicyFn:
    target_device = module_device(module)

    def _policy(obs: dict[str, np.ndarray], _agent: str, rng: np.random.Generator) -> int:
        obs_batch = {
            "observations": torch.from_numpy(obs["observations"][None, :]).float(),
            "action_mask": torch.from_numpy(obs["action_mask"][None, :]).float(),
        }
        if target_device.startswith("cuda"):
            obs_batch = {k: v.to(target_device) for k, v in obs_batch.items()}
        with torch.no_grad():
            out = module.forward_inference({Columns.OBS: obs_batch})

        logits = out[Columns.ACTION_DIST_INPUTS]
        if logits.ndim != 2 or logits.shape[0] != 1:
            return _pick_valid_action(obs, rng)
        action_idx = int(torch.argmax(logits[0]).item())

        if action_idx < 0 or action_idx >= len(obs["action_mask"]):
            return _pick_valid_action(obs, rng)
        if obs["action_mask"][action_idx] <= 0.0:
            return _pick_valid_action(obs, rng)
        return action_idx

    return _policy


def _candidate_is_p0(game_idx: int, seat_mode: SeatMode, seed: int) -> bool:
    if seat_mode == "p0":
        return True
    if seat_mode == "p1":
        return False
    if seat_mode == "random":
        rng = np.random.default_rng(seed + 99_999 + game_idx)
        return bool(rng.integers(0, 2) == 0)
    return (game_idx % 2) == 0


def evaluate_matchup(
    candidate_policy: PolicyFn,
    opponent_policy: PolicyFn,
    *,
    num_games: int,
    seed: int,
    seat_mode: SeatMode = "alternate",
) -> dict[str, float]:
    """Evaluate candidate policy against opponent over num_games episodes."""
    wins = 0
    draws = 0
    losses = 0
    score_diffs: list[float] = []

    for game_idx in range(num_games):
        env = FlatObsWrapper(ScoutEnv(num_rounds=1, reward_mode="score_diff"))
        env.reset(seed=seed + game_idx)

        # Alternate seats to avoid first-player bias.
        candidate_is_p0 = _candidate_is_p0(game_idx, seat_mode=seat_mode, seed=seed)
        rng = np.random.default_rng(seed + 10_000 + game_idx)

        while env.agents:
            agent = env.agent_selection
            if env.terminations.get(agent, False) or env.truncations.get(agent, False):
                env.step(None)
                continue

            obs = env.observe(agent)
            candidate_turn = (agent == AGENTS[0] and candidate_is_p0) or (
                agent == AGENTS[1] and not candidate_is_p0
            )
            action = (
                candidate_policy(obs, agent, rng)
                if candidate_turn
                else opponent_policy(obs, agent, rng)
            )
            env.step(action)

        final_scores = env.env.final_info.get("final_scores", {})
        p0 = float(final_scores.get(AGENTS[0], 0.0))
        p1 = float(final_scores.get(AGENTS[1], 0.0))
        candidate_score = p0 if candidate_is_p0 else p1
        opponent_score = p1 if candidate_is_p0 else p0

        diff = candidate_score - opponent_score
        score_diffs.append(diff)
        if diff > 0:
            wins += 1
        elif diff < 0:
            losses += 1
        else:
            draws += 1

    total = float(num_games)
    return {
        "win_rate": wins / total,
        "draw_rate": draws / total,
        "loss_rate": losses / total,
        "mean_score_diff": float(np.mean(score_diffs)) if score_diffs else 0.0,
    }


def evaluate_matchup_detailed(
    candidate_policy: PolicyFn,
    opponent_policy: PolicyFn,
    *,
    iteration: int,
    candidate_snapshot: str,
    opponent_type: str,
    opponent_snapshot: str | None,
    num_games: int,
    seed: int,
    num_rounds: int = 1,
    seat_mode: SeatMode = "alternate",
    game_idx_offset: int = 0,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate matchup and return aggregate metrics + per-game rows + replay docs."""
    wins = 0
    draws = 0
    losses = 0
    score_diffs: list[float] = []
    game_rows: list[dict[str, Any]] = []
    replay_docs: list[dict[str, Any]] = []

    for game_idx in range(num_games):
        abs_game_idx = game_idx_offset + game_idx
        env = FlatObsWrapper(ScoutEnv(num_rounds=num_rounds, reward_mode="score_diff"))
        env.reset(seed=seed + abs_game_idx)
        candidate_is_p0 = _candidate_is_p0(abs_game_idx, seat_mode=seat_mode, seed=seed)
        rng = np.random.default_rng(seed + 10_000 + abs_game_idx)
        game_id = (
            f"iter_{iteration}_{opponent_type}_{opponent_snapshot or 'none'}_{abs_game_idx}"
        )
        steps: list[dict[str, Any]] = []
        action_count = 0

        steps.append(
            {
                "step_idx": 0,
                "actor": None,
                "action": None,
                "state": env.env.get_rich_state(),
            }
        )

        while env.agents:
            agent = env.agent_selection
            if env.terminations.get(agent, False) or env.truncations.get(agent, False):
                env.step(None)
                steps.append(
                    {
                        "step_idx": len(steps),
                        "actor": agent,
                        "action": None,
                        "state": env.env.get_rich_state(),
                    }
                )
                continue

            obs = env.observe(agent)
            candidate_turn = (agent == AGENTS[0] and candidate_is_p0) or (
                agent == AGENTS[1] and not candidate_is_p0
            )
            action = (
                candidate_policy(obs, agent, rng)
                if candidate_turn
                else opponent_policy(obs, agent, rng)
            )
            env.step(action)
            action_count += 1
            steps.append(
                {
                    "step_idx": len(steps),
                    "actor": agent,
                    "action": int(action),
                    "state": env.env.get_rich_state(),
                }
            )

        final_scores = env.env.final_info.get("final_scores", {})
        p0 = float(final_scores.get(AGENTS[0], 0.0))
        p1 = float(final_scores.get(AGENTS[1], 0.0))
        candidate_score = p0 if candidate_is_p0 else p1
        opponent_score = p1 if candidate_is_p0 else p0
        diff = candidate_score - opponent_score
        score_diffs.append(diff)
        outcome: str
        if diff > 0:
            wins += 1
            outcome = "win"
        elif diff < 0:
            losses += 1
            outcome = "loss"
        else:
            draws += 1
            outcome = "draw"

        game_rows.append(
            {
                "timestamp": utc_now_iso(),
                "game_id": game_id,
                "iteration": int(iteration),
                "candidate_snapshot": candidate_snapshot,
                "opponent_type": opponent_type,
                "opponent_snapshot": opponent_snapshot,
                "seed": int(seed + abs_game_idx),
                "candidate_seat": AGENTS[0] if candidate_is_p0 else AGENTS[1],
                "candidate_score": candidate_score,
                "opponent_score": opponent_score,
                "score_diff": diff,
                "outcome": outcome,
                "num_steps": len(steps),
                "action_count": action_count,
                "num_games": int(num_games),
                "replay_path": None,
            }
        )

        replay_docs.append(
            {
                "game_id": game_id,
                "metadata": dict(game_rows[-1]),
                "steps": steps,
                "final": env.env.final_info,
            }
        )

    total = float(num_games)
    aggregate = {
        "win_rate": wins / total,
        "draw_rate": draws / total,
        "loss_rate": losses / total,
        "mean_score_diff": float(np.mean(score_diffs)) if score_diffs else 0.0,
    }
    return aggregate, game_rows, replay_docs


def build_eval_record(
    *,
    iteration: int,
    candidate_snapshot: str,
    opponent_type: str,
    num_games: int,
    metrics: dict[str, float],
    opponent_snapshot: str | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": utc_now_iso(),
        "iteration": int(iteration),
        "candidate_snapshot": candidate_snapshot,
        "opponent_type": opponent_type,
        "opponent_snapshot": opponent_snapshot,
        "num_games": int(num_games),
        **metrics,
    }


def load_algo_from_checkpoint(path: Path) -> Algorithm:
    resolved = path.resolve()
    return Algorithm.from_checkpoint(resolved.as_uri())
