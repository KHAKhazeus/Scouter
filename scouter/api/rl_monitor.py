"""Read-only API endpoints for RL training/evolution monitoring outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

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
    path = _runs_root() / run_id
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return path


@router.get("/runs")
def list_runs() -> dict[str, Any]:
    root = _runs_root()
    if not root.exists():
        return {"runs": []}

    runs = []
    for run_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True):
        manifest = _load_json(run_dir / "manifest.json", default={})
        if not manifest:
            continue
        runs.append(
            {
                "run_id": run_dir.name,
                "created_at": manifest.get("created_at"),
                "latest_iteration": manifest.get("latest_iteration", 0),
                "latest_eval_iteration": manifest.get("latest_eval_iteration"),
                "num_snapshots": len(manifest.get("snapshots", [])),
            }
        )

    return {"runs": runs}


@router.get("/runs/{run_id}/summary")
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
    }


@router.get("/runs/{run_id}/train")
def run_train(run_id: str, limit: int = Query(default=1000, ge=1, le=100000)) -> dict[str, Any]:
    run_dir = _run_dir(run_id)
    return {"run_id": run_id, "rows": _load_jsonl(run_dir / "train_metrics.jsonl", limit=limit)}


@router.get("/runs/{run_id}/evolution")
def run_evolution(run_id: str, limit: int = Query(default=1000, ge=1, le=100000)) -> dict[str, Any]:
    run_dir = _run_dir(run_id)
    return {"run_id": run_id, "rows": _load_jsonl(run_dir / "eval_metrics.jsonl", limit=limit)}


@router.get("/runs/{run_id}/snapshots")
def run_snapshots(run_id: str) -> dict[str, Any]:
    run_dir = _run_dir(run_id)
    manifest = _load_json(run_dir / "manifest.json", default={})
    if not manifest:
        raise HTTPException(status_code=404, detail="Manifest not found")
    return {
        "run_id": run_id,
        "snapshots": manifest.get("snapshots", []),
    }
