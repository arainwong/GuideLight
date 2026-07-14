#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?"Usage: run_model_shock_evaluation_resumable.sh MODEL.pt [NUM_EPS] [PORT_START]"}
NUM_EPS=${2:-50}

case "$NUM_EPS" in
  ''|*[!0-9]*) echo "NUM_EPS must be a positive integer" >&2; exit 2 ;;
esac
if [ "$NUM_EPS" -lt 1 ]; then
  echo "NUM_EPS must be a positive integer" >&2
  exit 2
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
SCENARIO="sumo_files/scenarios/sumo_fenglin_base_sub4"
TAG=${MODEL%.pt}
MODEL_PATH="$ROOT/tsc/eval_models/$MODEL"
MODEL_STEP=${TAG#model_}

if [ ! -f "$MODEL_PATH" ]; then
  echo "Missing checkpoint: $MODEL_PATH" >&2
  exit 2
fi
if [ -z "${SUMO_HOME:-}" ]; then
  echo "SUMO_HOME is not set; activate GuideLight and source setup_env39.sh first." >&2
  exit 2
fi

if [ "$#" -ge 3 ]; then
  PORT_START=$3
elif [ "$MODEL_STEP" != "$TAG" ] && [ "$MODEL_STEP" -eq "$MODEL_STEP" ] 2>/dev/null; then
  PORT_START=$((24969 + MODEL_STEP))
else
  PORT_START=24969
fi
case "$PORT_START" in
  ''|*[!0-9]*) echo "PORT_START must be an integer" >&2; exit 2 ;;
esac

if [ "$NUM_EPS" -eq 1 ]; then
  ROUTES="route_1"
else
  ROUTES="routes_1_${NUM_EPS}"
fi
INDICES="1-${NUM_EPS}"
LOG_DIR="$ROOT/$SCENARIO/eval_logs"
OUTPUT_DIR="$ROOT/$SCENARIO/analysis_plots/${ROUTES}_h00_h24_5min_${TAG}"
BASELINE_PORT=$((PORT_START + 1))
SHOCK_CONDITIONS=""
BASELINE_CONDITIONS=""

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
  echo "[run] $condition: existing $existing/$NUM_EPS, port $port, log $log_path"
  if ! python -u tsc/eval_model.py --condition "$condition" \
      --model "$MODEL" --num-eps "$NUM_EPS" --start-time "$start" --duration "$duration" \
      --route-dir "$route_dir" --port-start "$port" --no-tripinfo \
      --skip-pre-start-departures --sumo-seed 777 \
      > "$log_path" 2>&1; then
    echo "[failed] $condition; last log lines:" >&2
    tail -n 30 "$log_path" >&2
    return 1
  fi
  echo "[done] $condition"
}

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$ROOT"
echo "[start] model=$MODEL routes=$NUM_EPS ports=$PORT_START/$BASELINE_PORT"

for SEGMENT in h00_h06 h06_h12 h12_h18 h18_h24; do
  case "$SEGMENT" in
    h00_h06) START=0; DURATION=21600 ;;
    h06_h12) START=20700; DURATION=22500 ;;
    h12_h18) START=42300; DURATION=22500 ;;
    h18_h24) START=63900; DURATION=22500 ;;
  esac
  SHOCK_CONDITION="shock_routes_1_${NUM_EPS}_${TAG}_${SEGMENT}"
  BASELINE_CONDITION="baseline_routes_1_${NUM_EPS}_${TAG}_${SEGMENT}"
  SHOCK_CONDITIONS="${SHOCK_CONDITIONS:+${SHOCK_CONDITIONS},}${SHOCK_CONDITION}"
  BASELINE_CONDITIONS="${BASELINE_CONDITIONS:+${BASELINE_CONDITIONS},}${BASELINE_CONDITION}"
  run_condition "$SHOCK_CONDITION" sumo_fenglin_base_sub4 "$PORT_START" "$START" "$DURATION"
  run_condition "$BASELINE_CONDITION" sumo_fenglin_base_sub1 "$BASELINE_PORT" "$START" "$DURATION"
done

echo "[plot] cycle-time statistics and synchronization figures"
python "$SCENARIO/plot_shock_analysis.py" cycle \
  --indices "$INDICES" --shock-conditions "$SHOCK_CONDITIONS" \
  --baseline-conditions "$BASELINE_CONDITIONS" \
  --window-begin 0 --window-end 86400 --output-dir "$OUTPUT_DIR"
python "$SCENARIO/plot_shock_analysis.py" sync \
  --indices "$INDICES" --shock-conditions "$SHOCK_CONDITIONS" \
  --baseline-conditions "$BASELINE_CONDITIONS" \
  --window-begin 0 --window-end 86400 --output-dir "$OUTPUT_DIR"

echo "[complete] wrote statistics and plots to $OUTPUT_DIR"
