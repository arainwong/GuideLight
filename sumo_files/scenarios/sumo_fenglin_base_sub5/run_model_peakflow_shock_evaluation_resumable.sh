#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:-model_4520.pt}
NUM_EPS=${2:-50}
PORT_START=${3:-34520}

case "$NUM_EPS" in
  ''|*[!0-9]*) echo "NUM_EPS must be a positive integer" >&2; exit 2 ;;
esac
if [ "$NUM_EPS" -lt 1 ]; then
  echo "NUM_EPS must be a positive integer" >&2
  exit 2
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
SUB4_SCENARIO="sumo_files/scenarios/sumo_fenglin_base_sub4"
SUB5_SCENARIO="sumo_files/scenarios/sumo_fenglin_base_sub5"
PYTHON=${PYTHON:-/home/ge95qur/.conda/envs/GuideLight/bin/python}
MODEL_TAG=${MODEL%.pt}
MODEL_PATH="$ROOT/tsc/eval_models/$MODEL"
BASELINE_ROUTE_DIR="$ROOT/$SUB5_SCENARIO/baseline_real_h00_h24"
SHOCK_2P0_ROUTE_DIR="$ROOT/$SUB5_SCENARIO/shock_peak2x_h07_h08"
SHOCK_2P5_ROUTE_DIR="$ROOT/$SUB5_SCENARIO/shock_peak2p5_h07_h08"
LOG_DIR="$ROOT/$SUB4_SCENARIO/eval_logs"

if [ ! -x "$PYTHON" ]; then
  echo "Missing Python executable: $PYTHON" >&2
  exit 2
fi
if [ ! -f "$MODEL_PATH" ]; then
  echo "Missing checkpoint: $MODEL_PATH" >&2
  exit 2
fi
for dir in "$BASELINE_ROUTE_DIR" "$SHOCK_2P0_ROUTE_DIR" "$SHOCK_2P5_ROUTE_DIR"; do
  if [ ! -d "$dir" ]; then
    echo "Missing route dir: $dir" >&2
    exit 2
  fi
done

export SUMO_HOME=${SUMO_HOME:-/home/ge95qur/.conda/envs/GuideLight/lib/python3.9/site-packages/sumo}
export PATH="$SUMO_HOME/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [ "$NUM_EPS" -eq 1 ]; then
  ROUTES="route_1"
else
  ROUTES="routes_1_${NUM_EPS}"
fi
INDICES="1-${NUM_EPS}"
BASELINE_PORT=$PORT_START
SHOCK_2P0_PORT=$((PORT_START + 1))
SHOCK_2P5_PORT=$((PORT_START + 2))
BASELINE_CONDITIONS=""
SHOCK_2P0_CONDITIONS=""
SHOCK_2P5_CONDITIONS=""

completed_count() {
  if [ ! -d "$LOG_DIR/$1" ]; then
    echo 0
    return
  fi
  find "$LOG_DIR/$1" -path '*/trial_*/plt_*.pkl' -type f 2>/dev/null | wc -l
}

run_condition() {
  local condition=$1
  local route_dir=$2
  local port=$3
  local start=$4
  local duration=$5
  local existing
  local log_path="$LOG_DIR/${condition}.log"

  existing=$(completed_count "$condition")
  if [ "$existing" -ge "$NUM_EPS" ]; then
    echo "[skip] $condition already has $existing/$NUM_EPS policy records"
    return
  fi

  echo "[run] $condition: existing $existing/$NUM_EPS, start_episode $existing, port $port, log $log_path"
  if ! "$PYTHON" -u tsc/eval_model.py --condition "$condition" \
      --model "$MODEL" --num-eps "$NUM_EPS" --start-episode "$existing" \
      --start-time "$start" --duration "$duration" \
      --route-dir "$route_dir" --port-start "$port" --no-tripinfo \
      --skip-pre-start-departures --sumo-seed 777 \
      > "$log_path" 2>&1; then
    echo "[failed] $condition; last log lines:" >&2
    tail -n 30 "$log_path" >&2
    return 1
  fi
  echo "[done] $condition"
}

mkdir -p "$LOG_DIR"
cd "$ROOT"
echo "[start] model=$MODEL routes=$NUM_EPS ports=$BASELINE_PORT/$SHOCK_2P0_PORT/$SHOCK_2P5_PORT"
echo "[routes] baseline=$BASELINE_ROUTE_DIR"
echo "[routes] shock_2p0=$SHOCK_2P0_ROUTE_DIR"
echo "[routes] shock_2p5=$SHOCK_2P5_ROUTE_DIR"

for SEGMENT in h00_h06 h06_h12 h12_h18 h18_h24; do
  case "$SEGMENT" in
    h00_h06) START=0; DURATION=21600 ;;
    h06_h12) START=20700; DURATION=22500 ;;
    h12_h18) START=42300; DURATION=22500 ;;
    h18_h24) START=63900; DURATION=22500 ;;
  esac

  BASELINE_CONDITION="sub5_peakflow_baseline_real_${ROUTES}_${MODEL_TAG}_${SEGMENT}"
  SHOCK_2P0_CONDITION="sub5_peakflow_shock_2x_${ROUTES}_${MODEL_TAG}_${SEGMENT}"
  SHOCK_2P5_CONDITION="sub5_peakflow_shock_2p5_${ROUTES}_${MODEL_TAG}_${SEGMENT}"
  BASELINE_CONDITIONS="${BASELINE_CONDITIONS:+${BASELINE_CONDITIONS},}${BASELINE_CONDITION}"
  SHOCK_2P0_CONDITIONS="${SHOCK_2P0_CONDITIONS:+${SHOCK_2P0_CONDITIONS},}${SHOCK_2P0_CONDITION}"
  SHOCK_2P5_CONDITIONS="${SHOCK_2P5_CONDITIONS:+${SHOCK_2P5_CONDITIONS},}${SHOCK_2P5_CONDITION}"

  run_condition "$BASELINE_CONDITION" "$BASELINE_ROUTE_DIR" "$BASELINE_PORT" "$START" "$DURATION"
  run_condition "$SHOCK_2P0_CONDITION" "$SHOCK_2P0_ROUTE_DIR" "$SHOCK_2P0_PORT" "$START" "$DURATION"
  run_condition "$SHOCK_2P5_CONDITION" "$SHOCK_2P5_ROUTE_DIR" "$SHOCK_2P5_PORT" "$START" "$DURATION"
done

plot_case() {
  local shock_label=$1
  local shock_conditions=$2
  local output_dir="$ROOT/$SUB5_SCENARIO/analysis_plots/${ROUTES}_h00_h24_5min_${MODEL_TAG}_peakflow_${shock_label}"
  mkdir -p "$output_dir"
  echo "[plot] $shock_label -> $output_dir"
  "$PYTHON" "$SUB4_SCENARIO/plot_shock_analysis.py" cycle \
    --indices "$INDICES" --shock-conditions "$shock_conditions" \
    --baseline-conditions "$BASELINE_CONDITIONS" \
    --window-begin 0 --window-end 86400 --output-dir "$output_dir"
  "$PYTHON" "$SUB4_SCENARIO/plot_shock_analysis.py" sync \
    --indices "$INDICES" --shock-conditions "$shock_conditions" \
    --baseline-conditions "$BASELINE_CONDITIONS" \
    --window-begin 0 --window-end 86400 --output-dir "$output_dir"
}

plot_case shock_2p0 "$SHOCK_2P0_CONDITIONS"
plot_case shock_2p5 "$SHOCK_2P5_CONDITIONS"

echo "[complete] wrote results under $ROOT/$SUB5_SCENARIO/analysis_plots"
