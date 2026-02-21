#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TS="$(date -u +%Y%m%d_%H%M%S)"
LOG_DIR="outputs/monitoring"
PID_DIR="${LOG_DIR}/pids"

TB_PORT="${TB_PORT:-6006}"
API_PORT="${API_PORT:-8000}"

mkdir -p "${LOG_DIR}" "${PID_DIR}"

TB_LOG="${LOG_DIR}/tensorboard_${TS}.log"
API_LOG="${LOG_DIR}/rl_dashboard_api_${TS}.log"
TB_PID_FILE="${PID_DIR}/tensorboard.pid"
API_PID_FILE="${PID_DIR}/rl_dashboard_api.pid"

if [[ -f "${TB_PID_FILE}" ]] && kill -0 "$(cat "${TB_PID_FILE}")" 2>/dev/null; then
  echo "TensorBoard already running with PID $(cat "${TB_PID_FILE}")."
else
  nohup .venv/bin/python -m tensorboard.main \
    --logdir outputs/rl_runs \
    --host 0.0.0.0 \
    --port "${TB_PORT}" \
    > "${TB_LOG}" 2>&1 &
  echo $! > "${TB_PID_FILE}"
  echo "Started TensorBoard PID $(cat "${TB_PID_FILE}") log=${TB_LOG}"
fi

if [[ -f "${API_PID_FILE}" ]] && kill -0 "$(cat "${API_PID_FILE}")" 2>/dev/null; then
  echo "RL dashboard API already running with PID $(cat "${API_PID_FILE}")."
else
  nohup .venv/bin/python -m uvicorn scouter.api.server:app \
    --host 0.0.0.0 \
    --port "${API_PORT}" \
    > "${API_LOG}" 2>&1 &
  echo $! > "${API_PID_FILE}"
  echo "Started RL dashboard API PID $(cat "${API_PID_FILE}") log=${API_LOG}"
fi

echo "TensorBoard:  http://localhost:${TB_PORT}"
echo "RL Dashboard: http://localhost:${API_PORT}/rl-dashboard"
