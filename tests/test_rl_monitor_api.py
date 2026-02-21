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
            "training_state": {"phase": "idle"},
            "timeouts": {"total_eval_timeouts": 0, "consecutive_eval_timeouts": 0},
        },
    )
    _write_json(run / "leaderboard.json", {"latest_eval": {"win_rate_vs_random": 0.5}})
    _write_jsonl(run / "train_metrics.jsonl", [{"iteration": 10, "episode_return_mean": 0.0}])
    _write_jsonl(run / "eval_metrics.jsonl", [{"iteration": 10, "opponent_type": "random", "win_rate": 0.5}])
    _write_jsonl(
        run / "events.jsonl",
        [
            {
                "timestamp": "2026-02-20T00:00:01Z",
                "event": "TRAIN_ITER_DONE",
                "iteration": 10,
                "details": {"timesteps_total": 8192},
            }
        ],
    )
    replay_file = run / "replay_game_1.json"
    _write_json(replay_file, {"game_id": "game_1", "steps": [{"step_idx": 0}]})
    _write_jsonl(
        run / "eval_games.jsonl",
        [
            {
                "game_id": "game_1",
                "iteration": 10,
                "opponent_type": "random",
                "opponent_snapshot": None,
                "score_diff": 2.0,
                "candidate_score": 8.0,
                "opponent_score": 6.0,
                "outcome": "win",
                "replay_path": str(replay_file),
            }
        ],
    )

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

    games = rl_monitor.run_eval_games("run_test", limit=100)
    assert games["rows"][0]["game_id"] == "game_1"
    replay = rl_monitor.run_eval_game_replay("run_test", "game_1")
    assert replay["replay"]["steps"][0]["step_idx"] == 0

    status = rl_monitor.run_status("run_test")
    assert status["latest_iteration"] == 10
    assert status["training_state"]["phase"] == "idle"

    events = rl_monitor.run_events("run_test", limit=100)
    assert events["rows"][0]["event"] == "TRAIN_ITER_DONE"



def test_run_dir_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_RUNS_ROOT", str(tmp_path / "missing"))
    with pytest.raises(HTTPException):
        rl_monitor.run_summary("not_exist")


def test_rl_monitor_nested_run_discovery_and_readers(tmp_path, monkeypatch):
    runs_root = tmp_path / "rl_runs"
    run = runs_root / "overnight_sweep_abc123" / "A0_baseline_stable"
    run.mkdir(parents=True)

    _write_json(
        run / "manifest.json",
        {
            "run_id": "A0_baseline_stable",
            "created_at": "2026-02-21T00:00:00Z",
            "latest_iteration": 25,
            "latest_eval_iteration": 20,
            "snapshots": [{"snapshot_id": "iter_25", "path": "/tmp/snap25"}],
            "training_state": {"phase": "training"},
            "timeouts": {"total_eval_timeouts": 0, "consecutive_eval_timeouts": 0},
        },
    )
    _write_json(run / "leaderboard.json", {"latest_eval": {"win_rate_vs_random": 0.8}})
    _write_jsonl(run / "train_metrics.jsonl", [{"iteration": 25, "episode_return_mean": 0.0}])
    _write_jsonl(
        run / "eval_metrics.jsonl",
        [{"iteration": 20, "opponent_type": "random", "win_rate": 0.8}],
    )

    nested_id = "overnight_sweep_abc123/A0_baseline_stable"
    monkeypatch.setenv("RL_RUNS_ROOT", str(runs_root))

    runs = rl_monitor.list_runs()
    run_ids = {r["run_id"] for r in runs["runs"]}
    assert nested_id in run_ids

    summary = rl_monitor.run_summary(nested_id)
    assert summary["run_id"] == nested_id
    assert summary["latest_eval_summary"]["win_rate_vs_random"] == 0.8

    train = rl_monitor.run_train(nested_id, limit=100)
    assert train["rows"][0]["iteration"] == 25

    evo = rl_monitor.run_evolution(nested_id, limit=100)
    assert evo["rows"][0]["opponent_type"] == "random"

    snaps = rl_monitor.run_snapshots(nested_id)
    assert snaps["snapshots"][0]["snapshot_id"] == "iter_25"

    status = rl_monitor.run_status(nested_id)
    assert status["latest_iteration"] == 25

    events = rl_monitor.run_events(nested_id, limit=100)
    assert events["rows"] == []


def test_run_dir_rejects_path_traversal(tmp_path, monkeypatch):
    runs_root = tmp_path / "rl_runs"
    run = runs_root / "run_ok"
    run.mkdir(parents=True)
    _write_json(run / "manifest.json", {"run_id": "run_ok"})
    monkeypatch.setenv("RL_RUNS_ROOT", str(runs_root))
    with pytest.raises(HTTPException):
        rl_monitor.run_summary("../outside")
