"""Profile evaluation launch/runtime with per-game timing logs.

This script is for debugging slow/stuck eval phases.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scouter.env.scout_env import AGENTS, ScoutEnv
from scouter.rl.evolution_eval import load_algo_from_checkpoint, make_algo_policy, random_policy
from scouter.rl.rllib_wrapper import FlatObsWrapper, register_scout_env


@dataclass
class GameResult:
    game_idx: int
    elapsed_s: float
    steps: int
    candidate_score: float
    opponent_score: float
    score_diff: float
    outcome: str


def _pick_policy(ckpt: Path | None):
    if ckpt is None:
        return None, random_policy
    algo = load_algo_from_checkpoint(ckpt)
    policy = make_algo_policy(algo)
    return algo, policy


def _run_games(
    *,
    candidate_checkpoint: Path,
    opponent_checkpoint: Path | None,
    num_games: int,
    num_rounds: int,
    seed: int,
    progress_every: int,
) -> dict[str, Any]:
    register_scout_env("scout_v0")

    t0 = time.monotonic()
    candidate_algo = load_algo_from_checkpoint(candidate_checkpoint)
    candidate_load_s = time.monotonic() - t0

    opponent_algo = None
    if opponent_checkpoint is not None:
        t1 = time.monotonic()
        opponent_algo = load_algo_from_checkpoint(opponent_checkpoint)
        opponent_load_s = time.monotonic() - t1
        opponent_policy = make_algo_policy(opponent_algo)
        opponent_name = f"checkpoint:{opponent_checkpoint}"
    else:
        opponent_load_s = 0.0
        opponent_policy = random_policy
        opponent_name = "random"

    try:
        candidate_policy = make_algo_policy(candidate_algo)

        # Best-effort device detection.
        device = "unknown"
        try:
            module = candidate_algo.get_module("shared_policy")
            first_param = next(module.parameters(), None)
            if first_param is not None:
                device = str(first_param.device)
        except Exception:
            pass

        print(
            json.dumps(
                {
                    "phase": "setup",
                    "candidate_checkpoint": str(candidate_checkpoint),
                    "opponent": opponent_name,
                    "candidate_load_s": round(candidate_load_s, 3),
                    "opponent_load_s": round(opponent_load_s, 3),
                    "candidate_device": device,
                    "num_games": num_games,
                    "num_rounds": num_rounds,
                }
            )
        )

        wins = draws = losses = 0
        per_game: list[GameResult] = []
        global_start = time.monotonic()

        for game_idx in range(num_games):
            game_start = time.monotonic()
            env = FlatObsWrapper(ScoutEnv(num_rounds=num_rounds, reward_mode="score_diff"))
            env.reset(seed=seed + game_idx)

            candidate_is_p0 = (game_idx % 2) == 0
            rng = np.random.default_rng(seed + 10_000 + game_idx)
            steps = 0

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
                steps += 1

            final_scores = env.env.final_info.get("final_scores", {})
            p0 = float(final_scores.get(AGENTS[0], 0.0))
            p1 = float(final_scores.get(AGENTS[1], 0.0))
            candidate_score = p0 if candidate_is_p0 else p1
            opp_score = p1 if candidate_is_p0 else p0
            diff = candidate_score - opp_score

            if diff > 0:
                wins += 1
                outcome = "win"
            elif diff < 0:
                losses += 1
                outcome = "loss"
            else:
                draws += 1
                outcome = "draw"

            g = GameResult(
                game_idx=game_idx,
                elapsed_s=time.monotonic() - game_start,
                steps=steps,
                candidate_score=candidate_score,
                opponent_score=opp_score,
                score_diff=diff,
                outcome=outcome,
            )
            per_game.append(g)

            if progress_every > 0 and ((game_idx + 1) % progress_every == 0 or game_idx + 1 == num_games):
                print(
                    json.dumps(
                        {
                            "phase": "progress",
                            "completed_games": game_idx + 1,
                            "num_games": num_games,
                            "latest_game_elapsed_s": round(g.elapsed_s, 3),
                            "latest_steps": g.steps,
                            "wins": wins,
                            "draws": draws,
                            "losses": losses,
                            "elapsed_total_s": round(time.monotonic() - global_start, 3),
                        }
                    )
                )

        total_s = time.monotonic() - global_start
        mean_game_s = total_s / max(1, num_games)
        mean_steps = sum(g.steps for g in per_game) / max(1, num_games)
        mean_diff = sum(g.score_diff for g in per_game) / max(1, num_games)

        return {
            "status": "ok",
            "candidate_load_s": candidate_load_s,
            "opponent_load_s": opponent_load_s,
            "eval_total_s": total_s,
            "mean_game_s": mean_game_s,
            "mean_steps": mean_steps,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": wins / max(1, num_games),
            "mean_score_diff": mean_diff,
        }
    finally:
        candidate_algo.stop()
        if opponent_algo is not None:
            opponent_algo.stop()


def _worker(queue: mp.Queue, kwargs: dict[str, Any]) -> None:
    try:
        out = _run_games(**kwargs)
        queue.put({"ok": True, "result": out})
    except Exception as exc:  # noqa: BLE001
        queue.put({"ok": False, "error": repr(exc)})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile eval runtime with per-game logs")
    p.add_argument("--candidate-checkpoint", type=Path, required=True)
    p.add_argument("--opponent-checkpoint", type=Path, default=None)
    p.add_argument("--num-games", type=int, default=24)
    p.add_argument("--num-rounds", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--spawn-process", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--timeout-seconds", type=int, default=1800)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = {
        "candidate_checkpoint": args.candidate_checkpoint.resolve(),
        "opponent_checkpoint": args.opponent_checkpoint.resolve() if args.opponent_checkpoint else None,
        "num_games": int(args.num_games),
        "num_rounds": int(args.num_rounds),
        "seed": int(args.seed),
        "progress_every": int(args.progress_every),
    }

    if not args.spawn_process:
        out = _run_games(**kwargs)
        print(json.dumps({"phase": "done", **out}, indent=2))
        return

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_worker, args=(q, kwargs), daemon=True)
    t0 = time.monotonic()
    proc.start()
    proc.join(timeout=max(1, int(args.timeout_seconds)))

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        print(
            json.dumps(
                {
                    "phase": "timeout",
                    "elapsed_s": round(time.monotonic() - t0, 3),
                    "timeout_seconds": int(args.timeout_seconds),
                },
                indent=2,
            )
        )
        return

    if q.empty():
        print(
            json.dumps(
                {
                    "phase": "error",
                    "msg": "worker exited without payload",
                    "exit_code": proc.exitcode,
                },
                indent=2,
            )
        )
        return

    msg = q.get()
    if not msg.get("ok"):
        print(json.dumps({"phase": "error", "msg": msg.get("error")}, indent=2))
        return

    print(json.dumps({"phase": "done", **msg["result"]}, indent=2))


if __name__ == "__main__":
    main()
