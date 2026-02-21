from scouter.rl.metrics import extract_train_metrics, summarize_eval_records


def test_extract_train_metrics_basic_shape():
    result = {
        "num_env_steps_sampled_lifetime": 1234,
        "env_runners": {
            "episode_return_mean": 1.25,
            "agent_episode_returns_mean": {
                "player_0": -0.4,
                "player_1": 0.4,
            },
        },
        "learners": {
            "shared_policy": {
                "entropy": 0.5,
                "policy_loss": -0.1,
            }
        },
    }

    record = extract_train_metrics(result, iteration=7)
    assert record["iteration"] == 7
    assert record["timesteps_total"] == 1234.0
    assert record["episode_return_mean"] == 1.25
    assert record["per_agent_returns"]["player_0"] == -0.4
    assert record["stability"]["entropy"] == 0.5


def test_summarize_eval_records_history_fields():
    rows = [
        {"opponent_type": "random", "win_rate": 0.62, "mean_score_diff": 1.3},
        {"opponent_type": "history_checkpoint", "win_rate": 0.4},
        {"opponent_type": "history_checkpoint", "win_rate": 0.6},
    ]

    summary = summarize_eval_records(rows, iteration=25)
    assert summary["iteration"] == 25
    assert summary["win_rate_vs_random"] == 0.62
    assert summary["win_rate_vs_history_avg"] == 0.5
    assert summary["win_rate_vs_history_min"] == 0.4
    assert summary["num_history_opponents"] == 2


def test_extract_train_metrics_extended_stability_and_derived():
    result = {
        "num_env_steps_sampled_lifetime": 2048,
        "env_runners": {
            "episode_return_mean": 0.0,
            "num_env_steps_sampled_lifetime_throughput": 512.0,
            "agent_episode_returns_mean": {
                "player_0": -0.6,
                "player_1": 0.6,
            },
        },
        "learners": {
            "shared_policy": {
                "entropy": 1.1,
                "mean_kl_loss": 0.01,
                "default_optimizer_learning_rate": 1e-4,
                "gradients_default_optimizer_global_norm": 0.42,
                "vf_loss_unclipped": 12.4,
                "num_module_steps_trained_lifetime_throughput": 128.0,
            }
        },
    }

    record = extract_train_metrics(result, iteration=5, target_kl=0.02)
    assert record["throughput"]["sample_env_steps_per_s"] == 512.0
    assert record["throughput"]["train_module_steps_per_s"] == 128.0
    assert record["stability"]["mean_kl_loss"] == 0.01
    assert record["stability"]["gradients_default_optimizer_global_norm"] == 0.42
    assert record["derived"]["seat_return_gap"] == -1.2
    assert record["derived"]["clip_activity_proxy"] == 0.5
