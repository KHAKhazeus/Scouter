import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from scouter.api import rl_monitor


def _write_json(path: Path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_rl_monitor_readers(tmp_path, monkeypatch):
    runs_root = tmp_path / "rl_runs"
    run = runs_root / "run_test"
    run.mkdir(parents=True)

    _write_json(
        run / "manifest.json",
        {
            "run_id": "run_test",
            "created_at": "2026-02-20T00:00:00Z",
            "latest_iteration": 10,
            "latest_eval_iteration": 10,
            "snapshots": [{"snapshot_id": "iter_10", "path": "/tmp/snap"}],
        },
    )
    _write_json(run / "leaderboard.json", {"latest_eval": {"win_rate_vs_random": 0.5}})
    _write_jsonl(run / "train_metrics.jsonl", [{"iteration": 10, "episode_return_mean": 0.0}])
    _write_jsonl(run / "eval_metrics.jsonl", [{"iteration": 10, "opponent_type": "random", "win_rate": 0.5}])

    monkeypatch.setenv("RL_RUNS_ROOT", str(runs_root))

    runs = rl_monitor.list_runs()
    assert runs["runs"][0]["run_id"] == "run_test"

    summary = rl_monitor.run_summary("run_test")
    assert summary["latest_eval_summary"]["win_rate_vs_random"] == 0.5

    train = rl_monitor.run_train("run_test", limit=100)
    assert train["rows"][0]["iteration"] == 10

    evo = rl_monitor.run_evolution("run_test", limit=100)
    assert evo["rows"][0]["opponent_type"] == "random"

    snaps = rl_monitor.run_snapshots("run_test")
    assert snaps["snapshots"][0]["snapshot_id"] == "iter_10"



def test_run_dir_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_RUNS_ROOT", str(tmp_path / "missing"))
    with pytest.raises(HTTPException):
        rl_monitor.run_summary("not_exist")
