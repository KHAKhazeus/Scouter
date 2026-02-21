"""Read-only API endpoints for RL training/evolution monitoring outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/rl", tags=["rl-monitor"])


def _runs_root() -> Path:
    return Path(os.getenv("RL_RUNS_ROOT", "outputs/rl_runs"))


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if limit is not None and limit > 0:
        rows = rows[-limit:]
    return rows


def _run_dir(run_id: str) -> Path:
    root = _runs_root().resolve()
    path = (root / run_id).resolve()
    # Prevent path traversal outside the configured runs root.
    if root not in path.parents and path != root:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return path


@router.get("/runs")
def list_runs() -> dict[str, Any]:
    root = _runs_root()
    if not root.exists():
        return {"runs": []}

    runs = []
    candidates = sorted(
        [p.parent for p in root.rglob("manifest.json") if p.is_file()],
        key=lambda p: p.as_posix(),
        reverse=True,
    )
    for run_dir in candidates:
        manifest = _load_json(run_dir / "manifest.json", default={})
        if not manifest:
            continue
        rel_run_id = run_dir.relative_to(root).as_posix()
        runs.append(
            {
                "run_id": rel_run_id,
                "created_at": manifest.get("created_at"),
                "latest_iteration": manifest.get("latest_iteration", 0),
                "latest_eval_iteration": manifest.get("latest_eval_iteration"),
                "num_snapshots": len(manifest.get("snapshots", [])),
                "eval_game_count": manifest.get("eval_game_count", 0),
            }
        )

    return {"runs": runs}


@router.get("/runs/{run_id:path}/summary")
def run_summary(run_id: str) -> dict[str, Any]:
    run_dir = _run_dir(run_id)
    manifest = _load_json(run_dir / "manifest.json", default={})
    if not manifest:
        raise HTTPException(status_code=404, detail="Manifest not found")

    leaderboard = _load_json(run_dir / "leaderboard.json", default={})
    train_latest = _load_jsonl(run_dir / "train_metrics.jsonl", limit=1)
    eval_latest = _load_jsonl(run_dir / "eval_metrics.jsonl", limit=20)

    return {
        "run_id": run_id,
        "manifest": manifest,
        "latest_train": train_latest[-1] if train_latest else None,
        "latest_eval_rows": eval_latest,
        "latest_eval_summary": leaderboard.get("latest_eval") if leaderboard else None,
        "latest_eval_games": leaderboard.get("latest_eval_games", []),
        "training_state": manifest.get("training_state", {}),
        "timeouts": manifest.get("timeouts", {}),
    }


@router.get("/runs/{run_id:path}/train")
def run_train(run_id: str, limit: int = 1000) -> dict[str, Any]:
    run_dir = _run_dir(run_id)
    limit = max(1, min(limit, 100000))
    return {"run_id": run_id, "rows": _load_jsonl(run_dir / "train_metrics.jsonl", limit=limit)}


@router.get("/runs/{run_id:path}/evolution")
def run_evolution(run_id: str, limit: int = 1000) -> dict[str, Any]:
    run_dir = _run_dir(run_id)
    limit = max(1, min(limit, 100000))
    return {"run_id": run_id, "rows": _load_jsonl(run_dir / "eval_metrics.jsonl", limit=limit)}


@router.get("/runs/{run_id:path}/snapshots")
def run_snapshots(run_id: str) -> dict[str, Any]:
    run_dir = _run_dir(run_id)
    manifest = _load_json(run_dir / "manifest.json", default={})
    if not manifest:
        raise HTTPException(status_code=404, detail="Manifest not found")
    return {
        "run_id": run_id,
        "snapshots": manifest.get("snapshots", []),
    }


@router.get("/runs/{run_id:path}/eval-games")
def run_eval_games(
    run_id: str,
    limit: int = 2000,
    iteration: int | None = None,
    opponent_type: str | None = None,
    opponent_snapshot: str | None = None,
) -> dict[str, Any]:
    run_dir = _run_dir(run_id)
    rows = _load_jsonl(run_dir / "eval_games.jsonl", limit=None)
    limit = max(1, min(limit, 200000))

    if iteration is not None:
        rows = [r for r in rows if int(r.get("iteration", -1)) == int(iteration)]
    if opponent_type is not None:
        rows = [r for r in rows if r.get("opponent_type") == opponent_type]
    if opponent_snapshot is not None:
        rows = [r for r in rows if r.get("opponent_snapshot") == opponent_snapshot]

    rows = rows[-limit:]
    return {"run_id": run_id, "rows": rows}


@router.get("/runs/{run_id:path}/eval-games/{game_id}")
def run_eval_game_replay(run_id: str, game_id: str) -> dict[str, Any]:
    run_dir = _run_dir(run_id)
    rows = _load_jsonl(run_dir / "eval_games.jsonl", limit=None)
    match = next((r for r in rows if r.get("game_id") == game_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Game '{game_id}' not found")

    replay_path = match.get("replay_path")
    if not replay_path:
        raise HTTPException(status_code=404, detail="Replay path missing for selected game")

    replay_file = Path(replay_path)
    if not replay_file.exists():
        raise HTTPException(status_code=404, detail=f"Replay file not found: {replay_file}")

    replay = _load_json(replay_file, default={})
    return {
        "run_id": run_id,
        "game": match,
        "replay": replay,
    }


@router.get("/runs/{run_id:path}/status")
def run_status(run_id: str) -> dict[str, Any]:
    run_dir = _run_dir(run_id)
    manifest = _load_json(run_dir / "manifest.json", default={})
    if not manifest:
        raise HTTPException(status_code=404, detail="Manifest not found")
    return {
        "run_id": run_id,
        "latest_iteration": manifest.get("latest_iteration", 0),
        "latest_eval_iteration": manifest.get("latest_eval_iteration"),
        "training_state": manifest.get("training_state", {}),
        "timeouts": manifest.get("timeouts", {}),
        "resume_count": manifest.get("resume_count", 0),
    }


@router.get("/runs/{run_id:path}/events")
def run_events(run_id: str, limit: int = 500) -> dict[str, Any]:
    run_dir = _run_dir(run_id)
    limit = max(1, min(limit, 200000))
    return {"run_id": run_id, "rows": _load_jsonl(run_dir / "events.jsonl", limit=limit)}
