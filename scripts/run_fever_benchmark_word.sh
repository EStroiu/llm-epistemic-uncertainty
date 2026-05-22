#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
else
  PYTHON="python"
fi

MODEL="${MODEL:-FAST.gpt-oss:120b}"
FEVER_CONFIG="${FEVER_CONFIG:-v1.0}"
FEVER_SPLIT="${FEVER_SPLIT:-labelled_dev}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"
NUM_RUNS="${NUM_RUNS:-3}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_LOGPROBS="${TOP_LOGPROBS:-5}"
MAX_TOKENS="${MAX_TOKENS:-256}"
SEED="${SEED:-42}"
GRANULARITY="${GRANULARITY:-token}"
DISPLAY_GRANULARITY="${DISPLAY_GRANULARITY:-word}"

if [[ ! -d "$ROOT_DIR/external/fever" ]]; then
  echo "Note: FEVER submodule folder not found."
  echo "The benchmark still works because loading uses the Hugging Face Hub dataset ID."
fi

echo "Using Python: $PYTHON"
echo "Model: $MODEL"
echo "FEVER config/split: $FEVER_CONFIG / $FEVER_SPLIT"
echo "Max examples: $MAX_EXAMPLES"
echo "Runs: $NUM_RUNS"
echo "Signal granularity: $GRANULARITY"
echo "Display granularity: $DISPLAY_GRANULARITY"

echo ""
"$PYTHON" evals/fever_benchmark.py \
  --model "$MODEL" \
  --config "$FEVER_CONFIG" \
  --split "$FEVER_SPLIT" \
  --max-examples "$MAX_EXAMPLES" \
  --num-runs "$NUM_RUNS" \
  --temperature "$TEMPERATURE" \
  --top-logprobs "$TOP_LOGPROBS" \
  --max-tokens "$MAX_TOKENS" \
  --granularity "$GRANULARITY" \
  --display-granularity "$DISPLAY_GRANULARITY" \
  --seed "$SEED"

echo ""
echo "Done. FEVER benchmark artifacts are in: $ROOT_DIR/experiments"
