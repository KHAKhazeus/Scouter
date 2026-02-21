#!/usr/bin/env bash
set -euo pipefail

BASE_TS="$(date -u +%Y%m%d_%H%M%S)"
ROOT="outputs/rl_runs/overnight_sweep_${BASE_TS}"
mkdir -p "$ROOT"

COMMON=(
  --iterations 220
  --num-rounds 1
  --reward-mode score_diff
  --num-env-runners 4
  --num-learners 1
  --num-gpus 1
  --train-batch-size 8192
  --minibatch-size 512
  --clip-param 0.2
  --grad-clip 0.5
  --entropy-coeff 0.003
  --vf-clip-param 30.0
  --gae-lambda 0.95
  --eval-enabled
  --eval-interval 10
  --eval-games-random 24
  --eval-games-history 8
  --history-window 5
  --snapshot-keep-last 20
  --eval-num-workers 4
  --eval-seat-mode alternate
  --eval-policy-device auto
  --eval-worker-num-gpus 0
  --eval-device-fallback cpu
  --sample-timeout-s 300
  --rollout-fragment-length auto
)

run_cfg() {
  local name="$1"; shift
  local outdir="${ROOT}/${name}"
  mkdir -p "$outdir"
  echo "=== START ${name} $(date -u) ===" | tee -a "${ROOT}/sweep_progress.log"
  .venv/bin/python scripts/train_ppo.py \
    "${COMMON[@]}" \
    "$@" \
    --output-dir "$outdir" \
    > "${outdir}/nohup.log" 2>&1
  echo "=== END ${name} $(date -u) ===" | tee -a "${ROOT}/sweep_progress.log"
}

# A0: current-ish baseline
run_cfg A0_baseline_stable --lr 1e-4 --num-epochs 3 --target-kl 0.02 --clip-param 0.2 --grad-clip 0.5

# A1: lower step size
run_cfg A1_lr5e5_kl1e2_e2 --lr 5e-5 --num-epochs 2 --target-kl 0.01 --clip-param 0.15 --grad-clip 0.3

# A2: middle conservative
run_cfg A2_lr7e5_kl12e3_e2 --lr 7e-5 --num-epochs 2 --target-kl 0.012 --clip-param 0.18 --grad-clip 0.35

# A3: very conservative
run_cfg A3_lr3e5_kl8e3_e1 --lr 3e-5 --num-epochs 1 --target-kl 0.008 --clip-param 0.15 --grad-clip 0.25

# A4: keep lr, reduce clipping pressure
run_cfg A4_lr1e4_kl1e2_e2 --lr 1e-4 --num-epochs 2 --target-kl 0.01 --clip-param 0.15 --grad-clip 0.3

echo "Sweep completed: $ROOT"
