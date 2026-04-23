#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
MAX_TOKENS="${MAX_TOKENS:-180}"
SEED="${SEED:-42}"
QUICK_RUN="${QUICK_RUN:-0}"
QUICK_EXAMPLES="${QUICK_EXAMPLES:-10}"

echo "Using Python: $PYTHON"
echo "Model: $MODEL"
echo "FEVER config/split: $FEVER_CONFIG / $FEVER_SPLIT"

if [[ "$QUICK_RUN" == "1" ]]; then
  echo "Quick run enabled (~1 minute depending on API latency)."
  "$PYTHON" evals/label_ambiguity_benchmark.py \
    --model "$MODEL" \
    --config "$FEVER_CONFIG" \
    --split "$FEVER_SPLIT" \
    --max-examples "$MAX_EXAMPLES" \
    --num-runs "$NUM_RUNS" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --seed "$SEED" \
    --quick-run \
    --quick-examples "$QUICK_EXAMPLES"
else
  "$PYTHON" evals/label_ambiguity_benchmark.py \
    --model "$MODEL" \
    --config "$FEVER_CONFIG" \
    --split "$FEVER_SPLIT" \
    --max-examples "$MAX_EXAMPLES" \
    --num-runs "$NUM_RUNS" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --seed "$SEED"
fi

echo ""
echo "Done. Label-ambiguity benchmark artifacts are in: $ROOT_DIR/experiments"
