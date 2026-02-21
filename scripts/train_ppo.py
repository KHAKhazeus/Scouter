"""Train PPO self-play on Scout with monitoring and evolution evaluation."""

from __future__ import annotations

import argparse
import json
import statistics
import shutil
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.exceptions import GetTimeoutError
from tensorboardX import SummaryWriter

from scouter.rl.evolution_eval import (
    build_eval_record,
    evaluate_matchup_detailed,
    load_algo_from_checkpoint,  # used for resume + final save compatibility
    load_module_from_checkpoint,
    make_module_policy,
    module_device,
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
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-gpus", type=float, default=1.0)
    parser.add_argument("--num-learners", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample-timeout-s", type=float, default=120.0)
    parser.add_argument("--rollout-fragment-length", type=str, default="auto")
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--entropy-coeff", type=float, default=0.003)
    parser.add_argument("--vf-clip-param", type=float, default=30.0)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--gae-lambda", type=float, default=0.95)

    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tb-log-dir", type=Path, default=None)
    parser.add_argument("--snapshot-dir", type=Path, default=None)
    parser.add_argument("--eval-dir", type=Path, default=None)
    parser.add_argument("--dashboard-manifest", type=Path, default=None)

    parser.add_argument("--eval-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-games-random", type=int, default=100)
    parser.add_argument("--eval-games-history", type=int, default=50)
    parser.add_argument("--eval-num-workers", type=int, default=1)
    parser.add_argument(
        "--eval-seat-mode",
        choices=["alternate", "random", "p0", "p1"],
        default="alternate",
    )
    parser.add_argument(
        "--eval-policy-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--eval-worker-num-gpus", type=float, default=0.0)
    parser.add_argument(
        "--eval-device-fallback",
        choices=["cpu", "error"],
        default="cpu",
    )
    parser.add_argument("--history-window", type=int, default=5)
    parser.add_argument("--snapshot-keep-last", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)

    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--eval-timeout-seconds", type=int, default=1800)
    parser.add_argument("--eval-timeout-retries", type=int, default=1)
    parser.add_argument("--eval-timeout-action", choices=["skip", "fail"], default="skip")
    parser.add_argument("--max-consecutive-eval-timeouts", type=int, default=3)
    parser.add_argument("--state-checkpoint-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--state-checkpoint-interval", type=int, default=0)
    parser.add_argument("--tb-log-interval", type=int, default=1)
    args = parser.parse_args()
    rfl = str(args.rollout_fragment_length).strip().lower()
    if rfl == "auto":
        args.rollout_fragment_length = "auto"
    else:
        try:
            parsed = int(rfl)
        except ValueError as exc:
            raise ValueError("--rollout-fragment-length must be an int > 0 or 'auto'") from exc
        if parsed <= 0:
            raise ValueError("--rollout-fragment-length must be > 0 when set as int")
        args.rollout_fragment_length = parsed
    return args


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
        .env_runners(
            sample_timeout_s=args.sample_timeout_s,
            rollout_fragment_length=args.rollout_fragment_length,
        )
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
            clip_param=args.clip_param,
            grad_clip=args.grad_clip,
            entropy_coeff=args.entropy_coeff,
            vf_clip_param=args.vf_clip_param,
            kl_target=args.target_kl,
            lambda_=args.gae_lambda,
        )
        .reporting(
            log_gradients=True,
        )
    )

    if args.seed is not None:
        config = config.debugging(seed=args.seed)
    return config


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")


def _ensure_dirs(args: argparse.Namespace) -> dict[str, Path]:
    base_output = (args.output_dir or Path("outputs") / "rl_runs" / _run_id()).resolve()
    tb_dir = args.tb_log_dir or (base_output / "tb")
    snapshot_dir = args.snapshot_dir or (base_output / "snapshots")
    eval_dir = args.eval_dir or (base_output / "eval")
    checkpoint_dir = args.checkpoint_dir or (base_output / "final_checkpoint")
    state_checkpoint_dir = base_output / "state_checkpoints"
    manifest = args.dashboard_manifest or (base_output / "manifest.json")

    tb_dir = tb_dir.resolve()
    snapshot_dir = snapshot_dir.resolve()
    eval_dir = eval_dir.resolve()
    checkpoint_dir = checkpoint_dir.resolve()
    state_checkpoint_dir = state_checkpoint_dir.resolve()
    manifest = manifest.resolve()

    for d in (base_output, tb_dir, snapshot_dir, eval_dir, checkpoint_dir, state_checkpoint_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "output": base_output,
        "tb": tb_dir,
        "snapshots": snapshot_dir,
        "eval": eval_dir,
        "eval_replays": eval_dir / "replays",
        "checkpoint": checkpoint_dir,
        "state_checkpoints": state_checkpoint_dir,
        "manifest": manifest,
        "train_jsonl": base_output / "train_metrics.jsonl",
        "eval_jsonl": base_output / "eval_metrics.jsonl",
        "eval_games_jsonl": base_output / "eval_games.jsonl",
        "leaderboard": base_output / "leaderboard.json",
        "events_jsonl": base_output / "events.jsonl",
    }


def _json_dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, obj: Any) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def _save_checkpoint(algo, save_dir: Path) -> Path:
    result = algo.save(str(save_dir.resolve()))
    if hasattr(result, "checkpoint"):
        return Path(result.checkpoint.path).resolve()
    if hasattr(result, "path"):
        return Path(result.path).resolve()
    return Path(str(result)).resolve()


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


def _default_manifest(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "run_id": paths["output"].name,
        "created_at": utc_now_iso(),
        "paths": {
            k: str(v)
            for k, v in paths.items()
            if k
            not in {
                "train_jsonl",
                "eval_jsonl",
                "eval_games_jsonl",
                "leaderboard",
                "events_jsonl",
            }
        },
        "config": {
            "iterations": args.iterations,
            "num_rounds": args.num_rounds,
            "reward_mode": args.reward_mode,
            "train_batch_size": args.train_batch_size,
            "minibatch_size": args.minibatch_size,
            "num_epochs": args.num_epochs,
            "sample_timeout_s": args.sample_timeout_s,
            "rollout_fragment_length": args.rollout_fragment_length,
            "clip_param": args.clip_param,
            "grad_clip": args.grad_clip,
            "entropy_coeff": args.entropy_coeff,
            "vf_clip_param": args.vf_clip_param,
            "target_kl": args.target_kl,
            "gae_lambda": args.gae_lambda,
            "eval_enabled": args.eval_enabled,
            "eval_interval": args.eval_interval,
            "eval_games_random": args.eval_games_random,
            "eval_games_history": args.eval_games_history,
            "eval_num_workers": args.eval_num_workers,
            "eval_seat_mode": args.eval_seat_mode,
            "eval_policy_device": args.eval_policy_device,
            "eval_worker_num_gpus": args.eval_worker_num_gpus,
            "eval_device_fallback": args.eval_device_fallback,
            "history_window": args.history_window,
            "snapshot_keep_last": args.snapshot_keep_last,
            "eval_timeout_seconds": args.eval_timeout_seconds,
            "eval_timeout_retries": args.eval_timeout_retries,
            "eval_timeout_action": args.eval_timeout_action,
            "state_checkpoint_enabled": args.state_checkpoint_enabled,
            "state_checkpoint_interval": args.state_checkpoint_interval,
            "tb_log_interval": args.tb_log_interval,
            "resume": args.resume,
        },
        "snapshots": [],
        "latest_iteration": 0,
        "latest_eval_iteration": None,
        "eval_game_count": 0,
        "resume_count": 0,
        "training_state": {
            "phase": "idle",
            "phase_started_at": utc_now_iso(),
            "active_iteration": 0,
            "active_eval_opponent": None,
            "last_heartbeat_at": utc_now_iso(),
        },
        "timeouts": {
            "total_eval_timeouts": 0,
            "consecutive_eval_timeouts": 0,
            "last_timeout_iteration": None,
            "last_timeout_opponent": None,
        },
        "state_checkpoints": {
            "latest": None,
            "history": [],
        },
    }


def _merge_manifest_defaults(manifest: dict[str, Any], paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    defaults = _default_manifest(paths, args)
    out = defaults
    out.update(manifest)

    if not isinstance(out.get("training_state"), dict):
        out["training_state"] = defaults["training_state"]
    else:
        t = defaults["training_state"].copy()
        t.update(out["training_state"])
        out["training_state"] = t

    if not isinstance(out.get("timeouts"), dict):
        out["timeouts"] = defaults["timeouts"]
    else:
        t = defaults["timeouts"].copy()
        t.update(out["timeouts"])
        out["timeouts"] = t

    if not isinstance(out.get("state_checkpoints"), dict):
        out["state_checkpoints"] = defaults["state_checkpoints"]
    else:
        s = defaults["state_checkpoints"].copy()
        s.update(out["state_checkpoints"])
        if not isinstance(s.get("history"), list):
            s["history"] = []
        out["state_checkpoints"] = s

    if "paths" not in out:
        out["paths"] = defaults["paths"]
    else:
        path_rows = defaults["paths"].copy()
        path_rows.update(out["paths"])
        out["paths"] = path_rows

    out.setdefault("snapshots", [])
    out.setdefault("latest_iteration", 0)
    out.setdefault("latest_eval_iteration", None)
    out.setdefault("eval_game_count", 0)
    out.setdefault("resume_count", 0)
    out.setdefault("run_id", paths["output"].name)
    return out


def _emit_event(
    *,
    paths: dict[str, Path],
    manifest: dict[str, Any],
    event: str,
    iteration: int | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    row = {
        "timestamp": utc_now_iso(),
        "event": event,
        "iteration": iteration,
        "details": details or {},
    }
    _append_jsonl(paths["events_jsonl"], row)
    manifest.setdefault("last_event", row)


def _set_phase(
    *,
    manifest: dict[str, Any],
    paths: dict[str, Path],
    phase: str,
    iteration: int,
    opponent: str | None = None,
) -> None:
    state = manifest.setdefault("training_state", {})
    state["phase"] = phase
    state["phase_started_at"] = utc_now_iso()
    state["active_iteration"] = int(iteration)
    state["active_eval_opponent"] = opponent
    state["last_heartbeat_at"] = utc_now_iso()
    _json_dump(paths["manifest"], manifest)


def _heartbeat(manifest: dict[str, Any], paths: dict[str, Path]) -> None:
    state = manifest.setdefault("training_state", {})
    state["last_heartbeat_at"] = utc_now_iso()
    _json_dump(paths["manifest"], manifest)


def _log_train_to_tb(writer: SummaryWriter, train_record: dict[str, Any], *, interval: int = 1) -> None:
    step = int(train_record["iteration"])
    if interval > 1 and (step % interval) != 0:
        return

    if train_record.get("episode_return_mean") is not None:
        writer.add_scalar("train/returns/episode_return_mean", train_record["episode_return_mean"], step)

    for agent, val in train_record.get("per_agent_returns", {}).items():
        if val is not None:
            writer.add_scalar(f"train/self_play/{agent}_return_mean", val, step)

    derived = train_record.get("derived", {})
    if derived.get("seat_return_gap") is not None:
        writer.add_scalar("train/self_play/seat_return_gap", float(derived["seat_return_gap"]), step)
    if derived.get("clip_activity_proxy") is not None:
        writer.add_scalar("train/policy/clip_activity_proxy", float(derived["clip_activity_proxy"]), step)

    stability = train_record.get("stability", {})
    policy_keys = ("entropy", "policy_loss", "mean_kl_loss", "curr_kl_coeff")
    value_keys = ("vf_loss", "vf_loss_unclipped", "vf_explained_var")
    optim_keys = (
        "default_optimizer_learning_rate",
        "curr_entropy_coeff",
        "diff_num_grad_updates_vs_sampler_policy",
        "gradients_default_optimizer_global_norm",
    )
    batch_keys = ("module_train_batch_size_mean",)

    for k in policy_keys:
        v = stability.get(k)
        if v is not None:
            writer.add_scalar(f"train/policy/{k}", float(v), step)
    for k in value_keys:
        v = stability.get(k)
        if v is not None:
            writer.add_scalar(f"train/value/{k}", float(v), step)
    for k in optim_keys:
        v = stability.get(k)
        if v is not None:
            writer.add_scalar(f"train/optim/{k}", float(v), step)
    for k in batch_keys:
        v = stability.get(k)
        if v is not None:
            writer.add_scalar(f"train/batch/{k}", float(v), step)

    throughput = train_record.get("throughput", {})
    if throughput.get("sample_env_steps_per_s") is not None:
        writer.add_scalar("train/batch/sample_env_steps_per_s", float(throughput["sample_env_steps_per_s"]), step)
    if throughput.get("train_module_steps_per_s") is not None:
        writer.add_scalar("train/batch/train_module_steps_per_s", float(throughput["train_module_steps_per_s"]), step)


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


def _resolve_eval_policy_device(args: argparse.Namespace) -> str:
    if args.eval_policy_device != "auto":
        return str(args.eval_policy_device)
    if args.num_gpus and float(args.num_gpus) > 0.0:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
    return "cpu"


def _resolve_eval_runtime_settings(
    *,
    args: argparse.Namespace,
    num_workers: int,
) -> dict[str, Any]:
    requested_device = _resolve_eval_policy_device(args)
    fallback = str(args.eval_device_fallback)
    worker_gpus = float(args.eval_worker_num_gpus)
    workers = max(1, int(num_workers))
    use_ray = workers > 1 and ray.is_initialized()

    effective_device = requested_device
    fallback_applied = False
    fallback_reason = None
    ray_worker_num_gpus = 0.0

    if use_ray and requested_device == "cuda":
        if worker_gpus > 0.0:
            # Guard against GPU oversubscription/deadlock during eval.
            # If no GPU is currently allocatable for eval workers, fall back to CPU.
            available_gpu = float(ray.available_resources().get("GPU", 0.0)) if ray.is_initialized() else 0.0
            if available_gpu >= worker_gpus:
                ray_worker_num_gpus = worker_gpus
            elif fallback == "cpu":
                effective_device = "cpu"
                fallback_applied = True
                fallback_reason = (
                    "requested cuda for parallel eval but no allocatable GPU resources "
                    f"(available={available_gpu}, requested_per_worker={worker_gpus}); falling back to cpu"
                )
                ray_worker_num_gpus = 0.0
            else:
                raise ValueError(
                    "Parallel eval on cuda requested GPU workers, but no allocatable GPU resources were "
                    f"reported by Ray (available={available_gpu}, requested_per_worker={worker_gpus})."
                )
        elif fallback == "cpu":
            effective_device = "cpu"
            fallback_applied = True
            fallback_reason = (
                "requested cuda for parallel eval but --eval-worker-num-gpus <= 0; "
                "falling back to cpu"
            )
        else:
            raise ValueError(
                "Parallel eval on cuda requires --eval-worker-num-gpus > 0, "
                "or set --eval-device-fallback cpu."
            )

    return {
        "requested_device": requested_device,
        "effective_device": effective_device,
        "fallback": fallback,
        "fallback_applied": fallback_applied,
        "fallback_reason": fallback_reason,
        "ray_worker_num_gpus": ray_worker_num_gpus,
        "num_workers": workers,
        "use_ray": use_ray,
    }


def _evaluate_block_with_modules(
    *,
    candidate_checkpoint: str,
    iteration: int,
    candidate_snapshot: str,
    opponent_type: str,
    opponent_snapshot: str | None,
    num_games: int,
    seed: int,
    num_rounds: int,
    opponent_checkpoint: str | None,
    seat_mode: str,
    game_idx_offset: int = 0,
    policy_device: str = "cpu",
    device_fallback: str = "cpu",
    requested_policy_device: str | None = None,
) -> dict[str, Any]:
    worker_cuda_available = False
    effective_policy_device = policy_device
    fallback_applied = False
    fallback_reason = None
    if policy_device == "cuda":
        try:
            import torch

            worker_cuda_available = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            worker_cuda_available = False
        if not worker_cuda_available:
            if device_fallback == "cpu":
                effective_policy_device = "cpu"
                fallback_applied = True
                fallback_reason = "cuda requested in worker but cuda unavailable; fell back to cpu"
            else:
                raise RuntimeError("CUDA requested for eval worker, but CUDA is unavailable")

    candidate_module = load_module_from_checkpoint(
        Path(candidate_checkpoint), device=effective_policy_device
    )
    opponent_module = None
    try:
        candidate_policy = make_module_policy(candidate_module)
        candidate_dev = module_device(candidate_module)
        if opponent_type == "random":
            opponent_policy = random_policy
            opponent_dev = "random"
        else:
            if opponent_checkpoint is None:
                raise ValueError("opponent_checkpoint is required for history_checkpoint")
            opponent_module = load_module_from_checkpoint(
                Path(opponent_checkpoint), device=effective_policy_device
            )
            opponent_policy = make_module_policy(opponent_module)
            opponent_dev = module_device(opponent_module)

        aggregate, game_rows, replay_docs = evaluate_matchup_detailed(
            candidate_policy,
            opponent_policy,
            iteration=iteration,
            candidate_snapshot=candidate_snapshot,
            opponent_type=opponent_type,
            opponent_snapshot=opponent_snapshot,
            num_games=num_games,
            seed=seed,
            num_rounds=num_rounds,
            seat_mode=seat_mode,
            game_idx_offset=game_idx_offset,
        )
        return {
            "aggregate": aggregate,
            "game_rows": game_rows,
            "replay_docs": replay_docs,
            "candidate_device": candidate_dev,
            "opponent_device": opponent_dev,
            "requested_device": requested_policy_device or policy_device,
            "effective_device": effective_policy_device,
            "device_fallback_applied": fallback_applied,
            "device_fallback_reason": fallback_reason,
            "worker_cuda_available": worker_cuda_available,
        }
    finally:
        del opponent_module
        del candidate_module


def _aggregate_from_game_rows(game_rows: list[dict[str, Any]]) -> dict[str, float | None]:
    if not game_rows:
        return {
            "win_rate": None,
            "draw_rate": None,
            "loss_rate": None,
            "mean_score_diff": None,
        }
    wins = sum(1 for row in game_rows if row.get("outcome") == "win")
    draws = sum(1 for row in game_rows if row.get("outcome") == "draw")
    losses = sum(1 for row in game_rows if row.get("outcome") == "loss")
    diffs = [float(row.get("score_diff", 0.0)) for row in game_rows]
    total = float(len(game_rows))
    return {
        "win_rate": wins / total,
        "draw_rate": draws / total,
        "loss_rate": losses / total,
        "mean_score_diff": (sum(diffs) / total) if total > 0 else None,
    }


def _split_games(total_games: int, workers: int) -> list[tuple[int, int]]:
    n = max(1, int(workers))
    if total_games <= 0:
        return []
    n = min(n, total_games)
    base = total_games // n
    rem = total_games % n
    out: list[tuple[int, int]] = []
    start = 0
    for i in range(n):
        cnt = base + (1 if i < rem else 0)
        out.append((start, cnt))
        start += cnt
    return out


def _ray_eval_chunk_worker(kwargs: dict[str, Any]) -> dict[str, Any]:
    register_scout_env("scout_v0")
    return _evaluate_block_with_modules(**kwargs)


_RAY_EVAL_CHUNK_WORKER_REMOTE = ray.remote(num_cpus=1)(_ray_eval_chunk_worker)


def _run_eval_block_local(*, kwargs: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    try:
        register_scout_env("scout_v0")
        result = _evaluate_block_with_modules(**kwargs)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "elapsed_seconds": time.monotonic() - started,
            "error": repr(exc),
        }
    return {
        "status": "ok",
        "elapsed_seconds": time.monotonic() - started,
        **result,
    }


def _run_eval_block_ray(
    *,
    kwargs: dict[str, Any],
    timeout_seconds: int,
    num_workers: int,
    worker_num_gpus: float,
    progress_cb=None,
) -> dict[str, Any]:
    started = time.monotonic()
    chunks = _split_games(int(kwargs["num_games"]), num_workers)
    if not chunks:
        return {
            "status": "ok",
            "elapsed_seconds": 0.0,
            "aggregate": _aggregate_from_game_rows([]),
            "game_rows": [],
            "replay_docs": [],
            "candidate_device": None,
            "opponent_device": None,
            "requested_device": kwargs.get("policy_device"),
            "effective_device": kwargs.get("policy_device"),
            "device_fallback_applied": False,
            "device_fallback_reason": None,
            "worker_cuda_available": None,
        }

    refs = []
    for start_idx, count in chunks:
        chunk_kwargs = dict(kwargs)
        chunk_kwargs["num_games"] = int(count)
        chunk_kwargs["game_idx_offset"] = int(start_idx)
        refs.append(
            _RAY_EVAL_CHUNK_WORKER_REMOTE.options(
                num_cpus=1,
                num_gpus=max(0.0, float(worker_num_gpus)),
            ).remote(chunk_kwargs)
        )

    pending = list(refs)
    total_chunks = len(refs)
    completed_chunks = 0
    rows: list[dict[str, Any]] = []
    docs: list[dict[str, Any]] = []
    candidate_device: str | None = None
    opponent_device: str | None = None
    requested_device: str | None = None
    effective_device: str | None = None
    device_fallback_applied = False
    device_fallback_reason: str | None = None
    worker_cuda_available: bool | None = None
    deadline = started + max(1, int(timeout_seconds))

    try:
        while pending:
            remain = max(0.0, deadline - time.monotonic())
            if remain <= 0:
                for ref in pending:
                    ray.cancel(ref, force=True)
                return {
                    "status": "timeout",
                    "elapsed_seconds": time.monotonic() - started,
                    "error": "timeout",
                }
            done, pending = ray.wait(pending, num_returns=1, timeout=remain)
            if not done:
                continue
            result = ray.get(done[0])
            completed_chunks += 1
            rows.extend(result.get("game_rows", []))
            docs.extend(result.get("replay_docs", []))
            if candidate_device is None:
                candidate_device = result.get("candidate_device")
            if opponent_device is None:
                opponent_device = result.get("opponent_device")
            if requested_device is None:
                requested_device = result.get("requested_device")
            if effective_device is None:
                effective_device = result.get("effective_device")
            elif result.get("effective_device") not in {None, effective_device}:
                effective_device = "mixed"
            device_fallback_applied = bool(
                device_fallback_applied or result.get("device_fallback_applied", False)
            )
            if device_fallback_reason is None:
                device_fallback_reason = result.get("device_fallback_reason")
            if worker_cuda_available is None:
                worker_cuda_available = result.get("worker_cuda_available")
            if progress_cb is not None:
                progress_cb(
                    {
                        "type": "chunk_done",
                        "completed_chunks": completed_chunks,
                        "total_chunks": total_chunks,
                        "games_collected": len(rows),
                        "elapsed_seconds": time.monotonic() - started,
                        "effective_device": result.get("effective_device"),
                    }
                )
    except GetTimeoutError:
        for ref in pending:
            ray.cancel(ref, force=True)
        return {
            "status": "timeout",
            "elapsed_seconds": time.monotonic() - started,
            "error": "timeout",
        }
    except Exception as exc:  # noqa: BLE001
        for ref in pending:
            ray.cancel(ref, force=True)
        return {
            "status": "error",
            "elapsed_seconds": time.monotonic() - started,
            "error": repr(exc),
        }

    rows.sort(key=lambda row: str(row.get("game_id", "")))
    docs.sort(key=lambda doc: str(doc.get("game_id", "")))
    return {
        "status": "ok",
        "elapsed_seconds": time.monotonic() - started,
        "aggregate": _aggregate_from_game_rows(rows),
        "game_rows": rows,
        "replay_docs": docs,
        "candidate_device": candidate_device,
        "opponent_device": opponent_device,
        "requested_device": requested_device,
        "effective_device": effective_device,
        "device_fallback_applied": device_fallback_applied,
        "device_fallback_reason": device_fallback_reason,
        "worker_cuda_available": worker_cuda_available,
    }


def _run_eval_block(
    *,
    kwargs: dict[str, Any],
    timeout_seconds: int,
    num_workers: int,
    worker_num_gpus: float,
    progress_cb=None,
) -> dict[str, Any]:
    use_ray = num_workers > 1 and ray.is_initialized()
    if use_ray:
        return _run_eval_block_ray(
            kwargs=kwargs,
            timeout_seconds=timeout_seconds,
            num_workers=num_workers,
            worker_num_gpus=worker_num_gpus,
            progress_cb=progress_cb,
        )
    return _run_eval_block_local(kwargs=kwargs)


def _run_eval_block_with_retry(
    *,
    kwargs: dict[str, Any],
    timeout_seconds: int,
    retries: int,
    timeout_action: str,
    num_workers: int,
    worker_num_gpus: float,
    progress_cb=None,
) -> dict[str, Any]:
    attempts = max(1, retries + 1)
    last: dict[str, Any] | None = None

    for attempt in range(1, attempts + 1):
        out = _run_eval_block(
            kwargs=kwargs,
            timeout_seconds=timeout_seconds,
            num_workers=num_workers,
            worker_num_gpus=worker_num_gpus,
            progress_cb=progress_cb,
        )
        out["attempt"] = attempt
        last = out
        if out["status"] == "ok":
            return out
        if out["status"] == "timeout" and attempt < attempts:
            continue
        break

    if last is None:
        raise RuntimeError("evaluation block failed before producing status")

    final_status = "timeout_skipped" if last["status"] == "timeout" else "error_skipped"
    if timeout_action == "fail":
        raise RuntimeError(f"Eval block failed: {last.get('error', last['status'])}")

    return {
        "status": final_status,
        "attempt": last.get("attempt", attempts),
        "elapsed_seconds": float(last.get("elapsed_seconds", 0.0)),
        "error": last.get("error"),
        "aggregate": {
            "win_rate": None,
            "draw_rate": None,
            "loss_rate": None,
            "mean_score_diff": None,
        },
        "game_rows": [],
        "replay_docs": [],
        "candidate_device": last.get("candidate_device"),
        "opponent_device": last.get("opponent_device"),
        "requested_device": last.get("requested_device"),
        "effective_device": last.get("effective_device"),
        "device_fallback_applied": last.get("device_fallback_applied", False),
        "device_fallback_reason": last.get("device_fallback_reason"),
        "worker_cuda_available": last.get("worker_cuda_available"),
    }


def _run_evaluation(
    *,
    iteration: int,
    snapshot_path: Path,
    history_snapshots: list[dict[str, Any]],
    args: argparse.Namespace,
    manifest: dict[str, Any],
    paths: dict[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], int]:
    aggregate_rows: list[dict[str, Any]] = []
    game_rows: list[dict[str, Any]] = []
    replay_docs: list[dict[str, Any]] = []
    timeout_count = 0

    candidate_snapshot = snapshot_path.name
    candidate_checkpoint = str(snapshot_path)

    blocks: list[dict[str, Any]] = [
        {
            "opponent_type": "random",
            "opponent_snapshot": None,
            "num_games": args.eval_games_random,
            "seed": (args.seed or 0) + (iteration * 10_000),
            "opponent_checkpoint": None,
            "opponent_label": "random",
        }
    ]

    history = history_snapshots[-args.history_window :] if args.history_window > 0 else []
    for idx, row in enumerate(history):
        opponent_path = Path(row["path"])
        if not opponent_path.exists():
            continue
        blocks.append(
            {
                "opponent_type": "history_checkpoint",
                "opponent_snapshot": row["snapshot_id"],
                "num_games": args.eval_games_history,
                "seed": (args.seed or 0) + (iteration * 20_000) + idx,
                "opponent_checkpoint": str(opponent_path),
                "opponent_label": f"history:{row['snapshot_id']}",
            }
        )

    for block in blocks:
        opponent_type = block["opponent_type"]
        opponent_snapshot = block["opponent_snapshot"]
        opponent_label = block["opponent_label"]
        num_games = int(block["num_games"])
        runtime = _resolve_eval_runtime_settings(
            args=args,
            num_workers=max(1, int(args.eval_num_workers)),
        )

        _set_phase(
            manifest=manifest,
            paths=paths,
            phase="evaluating",
            iteration=iteration,
            opponent=opponent_label,
        )
        _emit_event(
            paths=paths,
            manifest=manifest,
            event="EVAL_BLOCK_START",
            iteration=iteration,
            details={
                "opponent": opponent_label,
                "num_games": num_games,
                "timeout_seconds": args.eval_timeout_seconds,
                "retries": args.eval_timeout_retries,
                "eval_num_workers": int(runtime["num_workers"]),
                "seat_mode": args.eval_seat_mode,
                "requested_eval_device": runtime["requested_device"],
                "effective_eval_device": runtime["effective_device"],
                "eval_worker_num_gpus": float(runtime["ray_worker_num_gpus"]),
                "eval_device_fallback": runtime["fallback"],
                "device_fallback_applied": bool(runtime["fallback_applied"]),
                "device_fallback_reason": runtime["fallback_reason"],
            },
        )

        kwargs = {
            "candidate_checkpoint": candidate_checkpoint,
            "iteration": iteration,
            "candidate_snapshot": candidate_snapshot,
            "opponent_type": opponent_type,
            "opponent_snapshot": opponent_snapshot,
            "num_games": num_games,
            "seed": int(block["seed"]),
            "num_rounds": args.num_rounds,
            "opponent_checkpoint": block["opponent_checkpoint"],
            "seat_mode": args.eval_seat_mode,
            "policy_device": runtime["effective_device"],
            "device_fallback": runtime["fallback"],
            "requested_policy_device": runtime["requested_device"],
        }

        def _on_eval_progress(payload: dict[str, Any]) -> None:
            _heartbeat(manifest, paths)
            _emit_event(
                paths=paths,
                manifest=manifest,
                event="EVAL_BLOCK_PROGRESS",
                iteration=iteration,
                details={
                    "opponent": opponent_label,
                    **payload,
                },
            )

        block_result = _run_eval_block_with_retry(
            kwargs=kwargs,
            timeout_seconds=args.eval_timeout_seconds,
            retries=args.eval_timeout_retries,
            timeout_action=args.eval_timeout_action,
            num_workers=int(runtime["num_workers"]),
            worker_num_gpus=float(runtime["ray_worker_num_gpus"]),
            progress_cb=_on_eval_progress,
        )

        status = str(block_result.get("status"))
        attempts = int(block_result.get("attempt", 1))
        elapsed_seconds = float(block_result.get("elapsed_seconds", 0.0))
        candidate_device = block_result.get("candidate_device")
        opponent_device = block_result.get("opponent_device")
        requested_device = block_result.get("requested_device", runtime["requested_device"])
        effective_device = block_result.get("effective_device", runtime["effective_device"])
        device_fallback_applied = bool(runtime["fallback_applied"]) or bool(
            block_result.get("device_fallback_applied", False)
        )
        device_fallback_reason = block_result.get("device_fallback_reason") or runtime["fallback_reason"]
        worker_cuda_available = block_result.get("worker_cuda_available")

        if status != "ok":
            if status == "timeout_skipped":
                timeout_count += 1
                _emit_event(
                    paths=paths,
                    manifest=manifest,
                    event="EVAL_TIMEOUT",
                    iteration=iteration,
                    details={
                        "opponent": opponent_label,
                        "attempts": attempts,
                        "elapsed_seconds": elapsed_seconds,
                    },
                )
            else:
                _emit_event(
                    paths=paths,
                    manifest=manifest,
                    event="EVAL_BLOCK_ERROR",
                    iteration=iteration,
                    details={
                        "opponent": opponent_label,
                        "attempts": attempts,
                        "error": block_result.get("error"),
                    },
                )

        metrics = block_result.get("aggregate") or {
            "win_rate": None,
            "draw_rate": None,
            "loss_rate": None,
            "mean_score_diff": None,
        }
        row = build_eval_record(
            iteration=iteration,
            candidate_snapshot=candidate_snapshot,
            opponent_type=opponent_type,
            opponent_snapshot=opponent_snapshot,
            num_games=num_games,
            metrics=metrics,
        )
        row["status"] = status
        row["attempts"] = attempts
        row["elapsed_seconds"] = elapsed_seconds
        if candidate_device is not None:
            row["candidate_device"] = str(candidate_device)
        if opponent_device is not None:
            row["opponent_device"] = str(opponent_device)
        if requested_device is not None:
            row["requested_eval_device"] = str(requested_device)
        if effective_device is not None:
            row["effective_eval_device"] = str(effective_device)
        row["eval_worker_num_gpus"] = float(runtime["ray_worker_num_gpus"])
        row["device_fallback_applied"] = device_fallback_applied
        if device_fallback_reason:
            row["device_fallback_reason"] = str(device_fallback_reason)
        if worker_cuda_available is not None:
            row["worker_cuda_available"] = bool(worker_cuda_available)
        if block_result.get("error"):
            row["error"] = str(block_result["error"])
        aggregate_rows.append(row)

        rows = block_result.get("game_rows") or []
        for game_row in rows:
            game_row["num_games"] = num_games
        replays = block_result.get("replay_docs") or []
        game_rows.extend(rows)
        replay_docs.extend(replays)

        _heartbeat(manifest, paths)
        _emit_event(
            paths=paths,
            manifest=manifest,
            event="EVAL_BLOCK_DONE",
            iteration=iteration,
            details={
                "opponent": opponent_label,
                "status": status,
                "attempts": attempts,
                "elapsed_seconds": elapsed_seconds,
                "num_games_written": len(rows),
                "candidate_device": candidate_device,
                "opponent_device": opponent_device,
                "requested_eval_device": requested_device,
                "effective_eval_device": effective_device,
                "eval_worker_num_gpus": float(runtime["ray_worker_num_gpus"]),
                "device_fallback_applied": device_fallback_applied,
                "device_fallback_reason": device_fallback_reason,
                "worker_cuda_available": worker_cuda_available,
            },
        )

    return (
        aggregate_rows,
        summarize_eval_records(aggregate_rows, iteration=iteration),
        game_rows,
        replay_docs,
        timeout_count,
    )


def _state_checkpoint_interval(args: argparse.Namespace) -> int:
    if args.state_checkpoint_interval and args.state_checkpoint_interval > 0:
        return int(args.state_checkpoint_interval)
    return int(args.eval_interval)


def _should_save_state_checkpoint(iteration: int, args: argparse.Namespace) -> bool:
    if not args.state_checkpoint_enabled:
        return False
    interval = _state_checkpoint_interval(args)
    return interval > 0 and (iteration % interval == 0)


def _resolve_resume_checkpoint(args: argparse.Namespace, manifest: dict[str, Any]) -> Path | None:
    if args.resume_from_checkpoint is not None:
        return args.resume_from_checkpoint.resolve()

    state_row = (manifest.get("state_checkpoints") or {}).get("latest")
    if isinstance(state_row, dict):
        path = state_row.get("path")
        if path:
            return Path(path).resolve()

    snapshots = manifest.get("snapshots") or []
    if snapshots:
        latest = snapshots[-1].get("path")
        if latest:
            return Path(latest).resolve()
    return None


def _check_stability_alerts(
    *,
    train_record: dict[str, Any],
    rolling: dict[str, deque[float]],
    paths: dict[str, Path],
    manifest: dict[str, Any],
) -> None:
    iteration = int(train_record["iteration"])
    stability = train_record.get("stability", {})
    derived = train_record.get("derived", {})

    vf_exp = stability.get("vf_explained_var")
    if vf_exp is not None:
        rolling["vf_explained_var"].append(float(vf_exp))
        if len(rolling["vf_explained_var"]) >= 25 and all(v < 0.02 for v in rolling["vf_explained_var"]):
            _emit_event(
                paths=paths,
                manifest=manifest,
                event="STABILITY_ALERT",
                iteration=iteration,
                details={
                    "type": "low_vf_explained_var",
                    "window": 25,
                    "threshold": 0.02,
                    "latest": float(vf_exp),
                },
            )

    clip_proxy = derived.get("clip_activity_proxy")
    if clip_proxy is not None:
        rolling["clip_activity_proxy"].append(float(clip_proxy))
        if len(rolling["clip_activity_proxy"]) >= 10 and all(v > 0.35 for v in rolling["clip_activity_proxy"]):
            _emit_event(
                paths=paths,
                manifest=manifest,
                event="STABILITY_ALERT",
                iteration=iteration,
                details={
                    "type": "high_clip_activity_proxy",
                    "window": 10,
                    "threshold": 0.35,
                    "latest": float(clip_proxy),
                },
            )

    grad_norm = stability.get("gradients_default_optimizer_global_norm")
    if grad_norm is not None:
        grad = float(grad_norm)
        if len(rolling["grad_norm"]) >= 20:
            baseline = statistics.median(rolling["grad_norm"])
            if baseline > 0 and grad > (3.0 * baseline):
                _emit_event(
                    paths=paths,
                    manifest=manifest,
                    event="STABILITY_ALERT",
                    iteration=iteration,
                    details={
                        "type": "grad_norm_spike",
                        "threshold_multiplier": 3.0,
                        "latest": grad,
                        "baseline_median": baseline,
                    },
                )
        rolling["grad_norm"].append(grad)


def main() -> None:
    args = parse_args()
    paths = _ensure_dirs(args)
    writer = SummaryWriter(log_dir=str(paths["tb"]))

    existing_manifest = None
    if paths["manifest"].exists():
        existing_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    if existing_manifest is not None and args.resume:
        manifest = _merge_manifest_defaults(existing_manifest, paths, args)
    else:
        manifest = _default_manifest(paths, args)

    _json_dump(paths["manifest"], manifest)

    resume_checkpoint = _resolve_resume_checkpoint(args, manifest) if args.resume else None
    start_iteration = int(manifest.get("latest_iteration", 0)) + 1

    algo = None
    resumed = False
    if resume_checkpoint is not None:
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")

        _emit_event(
            paths=paths,
            manifest=manifest,
            event="RESUME_START",
            iteration=start_iteration,
            details={"checkpoint": str(resume_checkpoint)},
        )
        _json_dump(paths["manifest"], manifest)

        register_scout_env("scout_v0")
        algo = load_algo_from_checkpoint(resume_checkpoint)
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        resumed = True

        _emit_event(
            paths=paths,
            manifest=manifest,
            event="RESUME_OK",
            iteration=start_iteration,
            details={"checkpoint": str(resume_checkpoint)},
        )
        _json_dump(paths["manifest"], manifest)
    else:
        algo = build_config(args).build_algo()

    try:
        rolling: dict[str, deque[float]] = {
            "vf_explained_var": deque(maxlen=25),
            "clip_activity_proxy": deque(maxlen=10),
            "grad_norm": deque(maxlen=50),
        }
        if start_iteration > args.iterations:
            print(
                f"resume_noop latest_iteration={manifest.get('latest_iteration')} target_iterations={args.iterations}"
            )

        for iteration in range(start_iteration, args.iterations + 1):
            _set_phase(manifest=manifest, paths=paths, phase="training", iteration=iteration)
            result = algo.train()
            train_record = extract_train_metrics(result, iteration, target_kl=args.target_kl)
            _append_jsonl(paths["train_jsonl"], train_record)
            _log_train_to_tb(writer, train_record, interval=max(1, int(args.tb_log_interval)))
            _check_stability_alerts(
                train_record=train_record,
                rolling=rolling,
                paths=paths,
                manifest=manifest,
            )

            manifest["latest_iteration"] = iteration
            _heartbeat(manifest, paths)

            print(
                f"iter={iteration} reward_mean={train_record.get('episode_return_mean')} "
                f"timesteps={train_record.get('timesteps_total')} "
                f"per_agent_returns={train_record.get('per_agent_returns')} "
                f"kl={train_record.get('stability', {}).get('mean_kl_loss')} "
                f"entropy={train_record.get('stability', {}).get('entropy')} "
                f"grad_norm={train_record.get('stability', {}).get('gradients_default_optimizer_global_norm')} "
                f"clip_proxy={train_record.get('derived', {}).get('clip_activity_proxy')}"
            )

            _emit_event(
                paths=paths,
                manifest=manifest,
                event="TRAIN_ITER_DONE",
                iteration=iteration,
                details={
                    "timesteps_total": train_record.get("timesteps_total"),
                    "episode_return_mean": train_record.get("episode_return_mean"),
                    "mean_kl_loss": train_record.get("stability", {}).get("mean_kl_loss"),
                    "grad_norm": train_record.get("stability", {}).get("gradients_default_optimizer_global_norm"),
                    "clip_activity_proxy": train_record.get("derived", {}).get("clip_activity_proxy"),
                },
            )

            state_checkpoint_path: Path | None = None
            if _should_save_state_checkpoint(iteration, args):
                _set_phase(manifest=manifest, paths=paths, phase="saving_checkpoint", iteration=iteration)
                state_root = paths["state_checkpoints"] / f"iter_{iteration}"
                state_root.mkdir(parents=True, exist_ok=True)
                state_checkpoint_path = _save_checkpoint(algo, state_root)

                row = {
                    "checkpoint_id": f"iter_{iteration}",
                    "iteration": iteration,
                    "created_at": utc_now_iso(),
                    "path": str(state_checkpoint_path),
                }
                history = list((manifest.get("state_checkpoints") or {}).get("history", []))
                history.append(row)
                history = _prune_snapshots(history, args.snapshot_keep_last)
                manifest.setdefault("state_checkpoints", {})["history"] = history
                manifest.setdefault("state_checkpoints", {})["latest"] = row
                _emit_event(
                    paths=paths,
                    manifest=manifest,
                    event="STATE_CHECKPOINT_SAVED",
                    iteration=iteration,
                    details={"path": str(state_checkpoint_path)},
                )
                _json_dump(paths["manifest"], manifest)

            do_eval = args.eval_enabled and args.eval_interval > 0 and (iteration % args.eval_interval == 0)
            if not do_eval:
                _json_dump(paths["manifest"], manifest)
                continue

            _set_phase(manifest=manifest, paths=paths, phase="saving_checkpoint", iteration=iteration)
            if state_checkpoint_path is not None:
                snapshot_path = state_checkpoint_path
            else:
                snapshot_root = paths["snapshots"] / f"iter_{iteration}"
                snapshot_root.mkdir(parents=True, exist_ok=True)
                snapshot_path = _save_checkpoint(algo, snapshot_root)

            history_snapshots = list(manifest["snapshots"])
            aggregate_rows, eval_summary, game_rows, replay_docs, timeout_count = _run_evaluation(
                iteration=iteration,
                snapshot_path=snapshot_path,
                history_snapshots=history_snapshots,
                args=args,
                manifest=manifest,
                paths=paths,
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

            timeouts = manifest.setdefault("timeouts", {})
            total = int(timeouts.get("total_eval_timeouts", 0))
            consecutive = int(timeouts.get("consecutive_eval_timeouts", 0))
            if timeout_count > 0:
                total += timeout_count
                consecutive += timeout_count
                timeouts["last_timeout_iteration"] = iteration
                timeouts["last_timeout_opponent"] = "multiple" if timeout_count > 1 else "single"
            else:
                consecutive = 0

            timeouts["total_eval_timeouts"] = total
            timeouts["consecutive_eval_timeouts"] = consecutive

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
                f"games={len(game_rows)} timeouts={timeout_count}"
            )

            _emit_event(
                paths=paths,
                manifest=manifest,
                event="EVAL_DONE",
                iteration=iteration,
                details={
                    "games": len(game_rows),
                    "win_vs_random": eval_summary.get("win_rate_vs_random"),
                    "timeouts": timeout_count,
                },
            )

            if consecutive >= int(args.max_consecutive_eval_timeouts):
                raise RuntimeError(
                    "Exceeded max consecutive eval timeouts: "
                    f"{consecutive} >= {args.max_consecutive_eval_timeouts}"
                )

        _set_phase(
            manifest=manifest,
            paths=paths,
            phase="idle",
            iteration=int(manifest.get("latest_iteration", 0)),
            opponent=None,
        )
    finally:
        if algo is not None:
            _set_phase(
                manifest=manifest,
                paths=paths,
                phase="saving_checkpoint",
                iteration=int(manifest.get("latest_iteration", 0)),
                opponent=None,
            )
            final_checkpoint = _save_checkpoint(algo, paths["checkpoint"])
            print(f"checkpoint={final_checkpoint}")
            _emit_event(
                paths=paths,
                manifest=manifest,
                event="FINAL_CHECKPOINT_SAVED",
                iteration=int(manifest.get("latest_iteration", 0)),
                details={"path": str(final_checkpoint), "resumed": resumed},
            )
            manifest.setdefault("training_state", {})["phase"] = "idle"
            manifest.setdefault("training_state", {})["active_eval_opponent"] = None
            manifest.setdefault("training_state", {})["last_heartbeat_at"] = utc_now_iso()
            _json_dump(paths["manifest"], manifest)
            algo.stop()
            if ray.is_initialized():
                ray.shutdown()

        writer.close()


if __name__ == "__main__":
    main()
