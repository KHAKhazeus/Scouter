#!/usr/bin/env bash
set -euo pipefail

TS="$(date -u +%Y%m%d_%H%M%S)"
OUTDIR="outputs/rl_runs/train_eval_stability_${TS}"
LOGFILE="${OUTDIR}/nohup_${TS}.log"

mkdir -p "${OUTDIR}"

nohup uv run python scripts/train_ppo.py \
  --iterations 1200 \
  --num-rounds 1 \
  --reward-mode score_diff \
  --num-env-runners 4 \
  --num-learners 1 \
  --num-gpus 1 \
  --train-batch-size 8192 \
  --minibatch-size 512 \
  --num-epochs 3 \
  --lr 1e-4 \
  --clip-param 0.2 \
  --grad-clip 0.5 \
  --entropy-coeff 0.003 \
  --vf-clip-param 30.0 \
  --target-kl 0.02 \
  --gae-lambda 0.95 \
  --eval-enabled \
  --eval-interval 25 \
  --eval-games-random 24 \
  --eval-games-history 8 \
  --history-window 5 \
  --snapshot-keep-last 20 \
  --eval-num-workers 4 \
  --eval-seat-mode alternate \
  --eval-policy-device auto \
  --eval-worker-num-gpus 0 \
  --eval-device-fallback cpu \
  --sample-timeout-s 300 \
  --rollout-fragment-length auto \
  --output-dir "${OUTDIR}" \
  > "${LOGFILE}" 2>&1 &

echo "Started training."
echo "OUTDIR: ${OUTDIR}"
echo "LOG:    ${LOGFILE}"
