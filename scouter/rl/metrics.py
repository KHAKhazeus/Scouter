"""Utilities for normalizing RLlib training/evaluation metrics."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


TRAIN_STABILITY_KEYS = (
    "entropy",
    "curr_kl_coeff",
    "kl",
    "policy_loss",
    "vf_explained_var",
    "vf_loss",
    "total_loss",
    "mean_kl_loss",
    "curr_entropy_coeff",
    "vf_loss_unclipped",
    "default_optimizer_learning_rate",
    "diff_num_grad_updates_vs_sampler_policy",
    "module_train_batch_size_mean",
    "num_module_steps_trained_lifetime_throughput",
    "gradients_default_optimizer_global_norm",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def extract_train_metrics(
    result: dict[str, Any],
    iteration: int,
    *,
    target_kl: float | None = None,
) -> dict[str, Any]:
    """Extract a stable subset of useful training metrics from RLlib result dict."""
    env_runners = result.get("env_runners", {})
    learners = result.get("learners", {})
    policy_stats = learners.get("shared_policy", {})

    record: dict[str, Any] = {
        "timestamp": utc_now_iso(),
        "iteration": int(iteration),
        "timesteps_total": _as_float(result.get("num_env_steps_sampled_lifetime")),
        "episode_return_mean": _as_float(
            env_runners.get("episode_return_mean", result.get("episode_reward_mean"))
        ),
        "per_agent_returns": {
            k: _as_float(v)
            for k, v in env_runners.get("agent_episode_returns_mean", {}).items()
        },
        "throughput": {
            "sample_env_steps_per_s": _as_float(
                env_runners.get("num_env_steps_sampled_lifetime_throughput")
            ),
            "train_module_steps_per_s": _as_float(
                policy_stats.get("num_module_steps_trained_lifetime_throughput")
            ),
        },
    }

    stability = {}
    for key in TRAIN_STABILITY_KEYS:
        if key in policy_stats:
            stability[key] = _as_float(policy_stats.get(key))
    record["stability"] = stability

    p0 = record["per_agent_returns"].get("player_0")
    p1 = record["per_agent_returns"].get("player_1")
    seat_return_gap = None
    if p0 is not None and p1 is not None:
        seat_return_gap = p0 - p1

    mean_kl = stability.get("mean_kl_loss")
    clip_proxy = None
    if mean_kl is not None and target_kl and target_kl > 0:
        # Proxy only: high values usually correlate with entering PPO clipped regime.
        clip_proxy = max(0.0, mean_kl / target_kl)

    record["derived"] = {
        "seat_return_gap": seat_return_gap,
        "clip_activity_proxy": clip_proxy,
    }
    return record


def summarize_eval_records(
    eval_records: list[dict[str, Any]],
    *,
    iteration: int,
) -> dict[str, Any]:
    """Summarize random/history eval records into one leaderboard-friendly row."""
    random_rows = [r for r in eval_records if r.get("opponent_type") == "random"]
    history_rows = [r for r in eval_records if r.get("opponent_type") == "history_checkpoint"]

    random_win = _as_float(random_rows[0]["win_rate"]) if random_rows else None
    random_diff = _as_float(random_rows[0]["mean_score_diff"]) if random_rows else None

    history_wins = [
        _as_float(r.get("win_rate"))
        for r in history_rows
        if _as_float(r.get("win_rate")) is not None
    ]
    history_avg = sum(history_wins) / len(history_wins) if history_wins else None
    history_min = min(history_wins) if history_wins else None

    return {
        "timestamp": utc_now_iso(),
        "iteration": int(iteration),
        "win_rate_vs_random": random_win,
        "mean_score_diff_vs_random": random_diff,
        "win_rate_vs_history_avg": history_avg,
        "win_rate_vs_history_min": history_min,
        "num_history_opponents": len(history_rows),
    }
