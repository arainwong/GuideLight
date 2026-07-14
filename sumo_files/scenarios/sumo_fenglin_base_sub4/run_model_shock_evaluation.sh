#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?"Usage: run_model_shock_evaluation.sh MODEL.pt [NUM_EPS] [PORT_START]"}
NUM_EPS=${2:-50}
PORT_START=${3:-24969}

case "$NUM_EPS" in
  ''|*[!0-9]*) echo "NUM_EPS must be a positive integer" >&2; exit 2 ;;
esac
if [ "$NUM_EPS" -lt 1 ]; then
  echo "NUM_EPS must be a positive integer" >&2
  exit 2
fi
case "$PORT_START" in
  ''|*[!0-9]*) echo "PORT_START must be an integer" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
SCENARIO="sumo_files/scenarios/sumo_fenglin_base_sub4"
TAG=${MODEL%.pt}
MODEL_PATH="$ROOT/tsc/eval_models/$MODEL"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Missing checkpoint: $MODEL_PATH" >&2
  exit 2
fi
if [ -z "${SUMO_HOME:-}" ]; then
  echo "SUMO_HOME is not set; activate GuideLight and source setup_env39.sh first." >&2
  exit 2
fi

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

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$ROOT"

for SEGMENT in h00_h06 h06_h12 h12_h18 h18_h24; do
  case "$SEGMENT" in
    h00_h06) START=0; DURATION=21600 ;;
    h06_h12) START=20700; DURATION=22500 ;;
    h12_h18) START=42300; DURATION=22500 ;;
    h18_h24) START=63900; DURATION=22500 ;;
  esac

  SHOCK_CONDITION="shock_routes_1_${NUM_EPS}_${TAG}_${SEGMENT}"
  BASELINE_CONDITION="baseline_routes_1_${NUM_EPS}_${TAG}_${SEGMENT}"
  if [ -n "$SHOCK_CONDITIONS" ]; then
    SHOCK_CONDITIONS="$SHOCK_CONDITIONS,$SHOCK_CONDITION"
    BASELINE_CONDITIONS="$BASELINE_CONDITIONS,$BASELINE_CONDITION"
  else
    SHOCK_CONDITIONS="$SHOCK_CONDITION"
    BASELINE_CONDITIONS="$BASELINE_CONDITION"
  fi

  python -u tsc/eval_model.py --condition "$SHOCK_CONDITION" \
    --model "$MODEL" --num-eps "$NUM_EPS" --start-time "$START" --duration "$DURATION" \
    --route-dir sumo_fenglin_base_sub4 --port-start "$PORT_START" --no-tripinfo \
    --skip-pre-start-departures --sumo-seed 777 \
    > "$LOG_DIR/${SHOCK_CONDITION}.log" 2>&1
  python -u tsc/eval_model.py --condition "$BASELINE_CONDITION" \
    --model "$MODEL" --num-eps "$NUM_EPS" --start-time "$START" --duration "$DURATION" \
    --route-dir sumo_fenglin_base_sub1 --port-start "$BASELINE_PORT" --no-tripinfo \
    --skip-pre-start-departures --sumo-seed 777 \
    > "$LOG_DIR/${BASELINE_CONDITION}.log" 2>&1
done

python "$SCENARIO/plot_shock_analysis.py" cycle \
  --indices "$INDICES" --shock-conditions "$SHOCK_CONDITIONS" \
  --baseline-conditions "$BASELINE_CONDITIONS" \
  --window-begin 0 --window-end 86400 --output-dir "$OUTPUT_DIR"
python "$SCENARIO/plot_shock_analysis.py" sync \
  --indices "$INDICES" --shock-conditions "$SHOCK_CONDITIONS" \
  --baseline-conditions "$BASELINE_CONDITIONS" \
  --window-begin 0 --window-end 86400 --output-dir "$OUTPUT_DIR"

echo "Wrote statistics and plots to $OUTPUT_DIR"
