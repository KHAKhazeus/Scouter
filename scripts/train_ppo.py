"""Train PPO self-play on Scout with monitoring and evolution evaluation."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from tensorboardX import SummaryWriter

from scouter.rl.evolution_eval import (
    build_eval_record,
    evaluate_matchup_detailed,
    load_algo_from_checkpoint,
    make_algo_policy,
    random_policy,
)
from scouter.rl.metrics import extract_train_metrics, summarize_eval_records, utc_now_iso
from scouter.rl.rllib_wrapper import ScoutActionMaskRLModule, register_scout_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO self-play for Scout.")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--num-rounds", type=int, default=1)
    parser.add_argument("--reward-mode", choices=["raw", "score_diff"], default="score_diff")
    parser.add_argument("--num-env-runners", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=4000)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-gpus", type=float, default=1.0)
    parser.add_argument("--num-learners", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tb-log-dir", type=Path, default=None)
    parser.add_argument("--snapshot-dir", type=Path, default=None)
    parser.add_argument("--eval-dir", type=Path, default=None)
    parser.add_argument("--dashboard-manifest", type=Path, default=None)

    parser.add_argument("--eval-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-games-random", type=int, default=100)
    parser.add_argument("--eval-games-history", type=int, default=50)
    parser.add_argument("--history-window", type=int, default=5)
    parser.add_argument("--snapshot-keep-last", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PPOConfig:
    env_name = "scout_v0"
    register_scout_env(env_name)

    config = (
        PPOConfig()
        .framework("torch")
        .environment(
            env=env_name,
            env_config={"num_rounds": args.num_rounds, "reward_mode": args.reward_mode},
            disable_env_checking=True,
        )
        .env_runners(num_env_runners=args.num_env_runners)
        .learners(
            num_learners=args.num_learners,
            num_gpus_per_learner=args.num_gpus,
        )
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda *_a, **_k: "shared_policy",
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "shared_policy": RLModuleSpec(
                        module_class=ScoutActionMaskRLModule,
                        model_config={
                            "head_fcnet_hiddens": [256, 256],
                            "head_fcnet_activation": "relu",
                        },
                    ),
                }
            )
        )
        .training(
            train_batch_size_per_learner=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            vf_loss_coeff=0.5,
        )
    )

    if args.seed is not None:
        config = config.debugging(seed=args.seed)
    return config


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")


def _ensure_dirs(args: argparse.Namespace) -> dict[str, Path]:
    base_output = args.output_dir or Path("outputs") / "rl_runs" / _run_id()
    tb_dir = args.tb_log_dir or (base_output / "tb")
    snapshot_dir = args.snapshot_dir or (base_output / "snapshots")
    eval_dir = args.eval_dir or (base_output / "eval")
    checkpoint_dir = args.checkpoint_dir or (base_output / "final_checkpoint")
    manifest = args.dashboard_manifest or (base_output / "manifest.json")

    for d in (base_output, tb_dir, snapshot_dir, eval_dir, checkpoint_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "output": base_output,
        "tb": tb_dir,
        "snapshots": snapshot_dir,
        "eval": eval_dir,
        "eval_replays": eval_dir / "replays",
        "checkpoint": checkpoint_dir,
        "manifest": manifest,
        "train_jsonl": base_output / "train_metrics.jsonl",
        "eval_jsonl": base_output / "eval_metrics.jsonl",
        "eval_games_jsonl": base_output / "eval_games.jsonl",
        "leaderboard": base_output / "leaderboard.json",
    }


def _json_dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, obj: Any) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def _save_checkpoint(algo, save_dir: Path) -> Path:
    result = algo.save(str(save_dir))
    if hasattr(result, "checkpoint"):
        return Path(result.checkpoint.path)
    if hasattr(result, "path"):
        return Path(result.path)
    return Path(str(result))


def _prune_snapshots(snapshot_rows: list[dict[str, Any]], keep_last: int) -> list[dict[str, Any]]:
    if keep_last <= 0:
        for row in snapshot_rows:
            path = Path(row["path"])
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        return []

    if len(snapshot_rows) <= keep_last:
        return snapshot_rows

    drop = snapshot_rows[:-keep_last]
    keep = snapshot_rows[-keep_last:]
    for row in drop:
        path = Path(row["path"])
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    return keep


def _log_train_to_tb(writer: SummaryWriter, train_record: dict[str, Any]) -> None:
    step = int(train_record["iteration"])
    if train_record.get("episode_return_mean") is not None:
        writer.add_scalar("train/episode_return_mean", train_record["episode_return_mean"], step)

    for agent, val in train_record.get("per_agent_returns", {}).items():
        if val is not None:
            writer.add_scalar(f"train/per_agent/{agent}_return_mean", val, step)

    for k, v in train_record.get("stability", {}).items():
        if v is not None:
            writer.add_scalar(f"train/stability/{k}", v, step)


def _log_eval_to_tb(writer: SummaryWriter, eval_summary: dict[str, Any]) -> None:
    step = int(eval_summary["iteration"])
    for key in (
        "win_rate_vs_random",
        "mean_score_diff_vs_random",
        "win_rate_vs_history_avg",
        "win_rate_vs_history_min",
    ):
        val = eval_summary.get(key)
        if val is not None:
            writer.add_scalar(f"evolution/{key}", float(val), step)


def _save_replays(
    *,
    replay_docs: list[dict[str, Any]],
    replay_dir: Path,
    game_rows: list[dict[str, Any]],
) -> None:
    replay_dir.mkdir(parents=True, exist_ok=True)
    row_by_id = {r["game_id"]: r for r in game_rows}

    for doc in replay_docs:
        game_id = doc["game_id"]
        path = replay_dir / f"{game_id}.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        row = row_by_id.get(game_id)
        if row is not None:
            row["replay_path"] = str(path)


def _run_evaluation(
    *,
    algo,
    iteration: int,
    snapshot_path: Path,
    history_snapshots: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_policy = make_algo_policy(algo)
    aggregate_rows: list[dict[str, Any]] = []
    game_rows: list[dict[str, Any]] = []
    replay_docs: list[dict[str, Any]] = []

    random_agg, random_games, random_replays = evaluate_matchup_detailed(
        candidate_policy,
        random_policy,
        iteration=iteration,
        candidate_snapshot=snapshot_path.name,
        opponent_type="random",
        opponent_snapshot=None,
        num_games=args.eval_games_random,
        seed=(args.seed or 0) + (iteration * 10_000),
        num_rounds=args.num_rounds,
    )
    aggregate_rows.append(
        build_eval_record(
            iteration=iteration,
            candidate_snapshot=snapshot_path.name,
            opponent_type="random",
            opponent_snapshot=None,
            num_games=args.eval_games_random,
            metrics=random_agg,
        )
    )
    game_rows.extend(random_games)
    replay_docs.extend(random_replays)

    history = history_snapshots[-args.history_window :]
    for idx, row in enumerate(history):
        opponent_path = Path(row["path"])
        if not opponent_path.exists():
            continue

        opponent_algo = load_algo_from_checkpoint(opponent_path)
        try:
            opponent_policy = make_algo_policy(opponent_algo)
            history_agg, history_games, history_replays = evaluate_matchup_detailed(
                candidate_policy,
                opponent_policy,
                iteration=iteration,
                candidate_snapshot=snapshot_path.name,
                opponent_type="history_checkpoint",
                opponent_snapshot=row["snapshot_id"],
                num_games=args.eval_games_history,
                seed=(args.seed or 0) + (iteration * 20_000) + idx,
                num_rounds=args.num_rounds,
            )
        finally:
            opponent_algo.stop()

        aggregate_rows.append(
            build_eval_record(
                iteration=iteration,
                candidate_snapshot=snapshot_path.name,
                opponent_type="history_checkpoint",
                opponent_snapshot=row["snapshot_id"],
                num_games=args.eval_games_history,
                metrics=history_agg,
            )
        )
        game_rows.extend(history_games)
        replay_docs.extend(history_replays)

    return (
        aggregate_rows,
        summarize_eval_records(aggregate_rows, iteration=iteration),
        game_rows,
        replay_docs,
    )


def main() -> None:
    args = parse_args()
    paths = _ensure_dirs(args)
    writer = SummaryWriter(log_dir=str(paths["tb"]))

    manifest: dict[str, Any] = {
        "run_id": paths["output"].name,
        "created_at": utc_now_iso(),
        "paths": {k: str(v) for k, v in paths.items() if k not in {"train_jsonl", "eval_jsonl", "eval_games_jsonl", "leaderboard"}},
        "config": {
            "iterations": args.iterations,
            "num_rounds": args.num_rounds,
            "reward_mode": args.reward_mode,
            "train_batch_size": args.train_batch_size,
            "minibatch_size": args.minibatch_size,
            "num_epochs": args.num_epochs,
            "eval_enabled": args.eval_enabled,
            "eval_interval": args.eval_interval,
            "eval_games_random": args.eval_games_random,
            "eval_games_history": args.eval_games_history,
            "history_window": args.history_window,
            "snapshot_keep_last": args.snapshot_keep_last,
        },
        "snapshots": [],
        "latest_iteration": 0,
        "latest_eval_iteration": None,
        "eval_game_count": 0,
    }
    _json_dump(paths["manifest"], manifest)

    algo = build_config(args).build_algo()

    try:
        for i in range(args.iterations):
            iteration = i + 1
            result = algo.train()
            train_record = extract_train_metrics(result, iteration)
            _append_jsonl(paths["train_jsonl"], train_record)
            _log_train_to_tb(writer, train_record)

            manifest["latest_iteration"] = iteration
            _json_dump(paths["manifest"], manifest)

            print(
                f"iter={iteration} reward_mean={train_record.get('episode_return_mean')} "
                f"timesteps={train_record.get('timesteps_total')} "
                f"per_agent_returns={train_record.get('per_agent_returns')}"
            )

            do_eval = args.eval_enabled and args.eval_interval > 0 and (iteration % args.eval_interval == 0)
            if not do_eval:
                continue

            snapshot_root = paths["snapshots"] / f"iter_{iteration}"
            snapshot_root.mkdir(parents=True, exist_ok=True)
            snapshot_path = _save_checkpoint(algo, snapshot_root)

            history_snapshots = list(manifest["snapshots"])
            aggregate_rows, eval_summary, game_rows, replay_docs = _run_evaluation(
                algo=algo,
                iteration=iteration,
                snapshot_path=snapshot_path,
                history_snapshots=history_snapshots,
                args=args,
            )

            replay_dir = paths["eval_replays"] / f"iter_{iteration}"
            _save_replays(replay_docs=replay_docs, replay_dir=replay_dir, game_rows=game_rows)

            for row in aggregate_rows:
                _append_jsonl(paths["eval_jsonl"], row)
            for row in game_rows:
                _append_jsonl(paths["eval_games_jsonl"], row)

            _log_eval_to_tb(writer, eval_summary)

            snapshot_row = {
                "snapshot_id": f"iter_{iteration}",
                "iteration": iteration,
                "created_at": utc_now_iso(),
                "path": str(snapshot_path),
            }
            manifest["snapshots"].append(snapshot_row)
            manifest["snapshots"] = _prune_snapshots(manifest["snapshots"], args.snapshot_keep_last)
            manifest["latest_eval_iteration"] = iteration
            manifest["eval_game_count"] = int(manifest.get("eval_game_count", 0)) + len(game_rows)

            leaderboard = {
                "run_id": manifest["run_id"],
                "timestamp": utc_now_iso(),
                "latest_iteration": iteration,
                "latest_eval": eval_summary,
                "latest_eval_games": game_rows[-20:],
                "snapshots": manifest["snapshots"],
            }
            _json_dump(paths["leaderboard"], leaderboard)
            _json_dump(paths["manifest"], manifest)

            print(
                f"eval@iter={iteration} "
                f"win_vs_random={eval_summary.get('win_rate_vs_random')} "
                f"win_vs_hist_avg={eval_summary.get('win_rate_vs_history_avg')} "
                f"games={len(game_rows)}"
            )
    finally:
        final_checkpoint = _save_checkpoint(algo, paths["checkpoint"])
        print(f"checkpoint={final_checkpoint}")
        algo.stop()
        writer.close()


if __name__ == "__main__":
    main()
