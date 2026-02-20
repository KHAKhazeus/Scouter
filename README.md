# Scouter RL Monitoring

## Train With Evolution Monitoring

Run training with output artifacts for TensorBoard and dashboard:

```bash
.venv/bin/python scripts/train_ppo.py \
  --iterations 200 \
  --eval-interval 25 \
  --eval-games-random 100 \
  --eval-games-history 50 \
  --history-window 5
```

Outputs are created under `outputs/rl_runs/<run_id>/` by default:

- `train_metrics.jsonl`
- `eval_metrics.jsonl`
- `manifest.json`
- `leaderboard.json`
- `tb/`
- `snapshots/`

## TensorBoard

```bash
.venv/bin/python -m tensorboard.main --logdir outputs/rl_runs
```

## Dashboard

Start backend and frontend, then open `/rl-dashboard`.

Backend serves monitoring APIs under `/rl/*`:

- `/rl/runs`
- `/rl/runs/{run_id}/summary`
- `/rl/runs/{run_id}/train`
- `/rl/runs/{run_id}/evolution`
- `/rl/runs/{run_id}/snapshots`
