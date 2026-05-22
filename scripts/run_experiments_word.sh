#!/usr/bin/env bash
set -euo pipefail

# Runs the recommended experiment suite for logit-gap claim detection in word mode.
# Usage:
#   ./scripts/run_experiments_word.sh
# Optional env overrides:
#   MODEL="FAST.gpt-oss:120b" PROMPTS_FILE="prompts/basic_prompts.txt" RUNS=3 ./scripts/run_experiments_word.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
else
  PYTHON="python"
fi

MODEL="${MODEL:-FAST.gpt-oss:120b}"
PROMPTS_FILE="${PROMPTS_FILE:-prompts/basic_prompts.txt}"
TOP_LOGPROBS="${TOP_LOGPROBS:-5}"
MAX_TOKENS="${MAX_TOKENS:-300}"
RUNS="${RUNS:-10}"
GRANULARITY="${GRANULARITY:-token}"
DISPLAY_GRANULARITY="${DISPLAY_GRANULARITY:-word}"

if [[ ! -f "$PROMPTS_FILE" ]]; then
  echo "Missing prompts file: $PROMPTS_FILE"
  exit 1
fi

echo "Using Python: $PYTHON"
echo "Using model: $MODEL"
echo "Using prompts file: $PROMPTS_FILE"
echo "Using repeated runs: $RUNS"
echo "Using signal granularity: $GRANULARITY"
echo "Using display granularity: $DISPLAY_GRANULARITY"

echo ""
echo "[1/2] Main experiment (temperature sweep, repeated)"
"$PYTHON" uncertainty_logit_gap.py \
  --model "$MODEL" \
  --prompts-file "$PROMPTS_FILE" \
  --temperatures "0.0,0.2,0.5" \
  --num-runs "$RUNS" \
  --top-logprobs "$TOP_LOGPROBS" \
  --max-tokens "$MAX_TOKENS" \
  --granularity "$GRANULARITY" \
  --display-granularity "$DISPLAY_GRANULARITY" \
  --experiment-name "exp_main_temp_sweep_word"

echo ""
echo "[2/2] Deterministic run (T=0.0)"
"$PYTHON" uncertainty_logit_gap.py \
  --model "$MODEL" \
  --prompts-file "$PROMPTS_FILE" \
  --temperatures "0.0" \
  --num-runs 1 \
  --top-logprobs "$TOP_LOGPROBS" \
  --max-tokens "$MAX_TOKENS" \
  --granularity "$GRANULARITY" \
  --display-granularity "$DISPLAY_GRANULARITY" \
  --experiment-name "exp_deterministic_t0_word"

echo ""
echo "All experiments completed."
echo "Artifacts are in: $ROOT_DIR/experiments"
