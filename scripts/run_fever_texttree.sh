#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
else
  PYTHON="python"
fi

RUN_DIR="${RUN_DIR:-$ROOT_DIR/experiments/fever_simple_v1.0_labelled_dev_2026-05-21_15-26-08}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/experiments}"

echo "Using Python: $PYTHON"
echo "Output directory: $OUTPUT_DIR"
echo "Model: ${MODEL:-FAST.gpt-oss:120b}"
echo "FEVER config/split: ${FEVER_CONFIG:-v1.0} / ${FEVER_SPLIT:-labelled_dev}"
echo "Max examples: ${MAX_EXAMPLES:-200}"
echo "Test size: ${TEST_SIZE:-0.2}"
echo "Runs: ${NUM_RUNS:-3}"

echo ""
"$PYTHON" fever_texttree_uncertainty.py \
  --model "${MODEL:-FAST.gpt-oss:120b}" \
  --config "${FEVER_CONFIG:-v1.0}" \
  --split "${FEVER_SPLIT:-labelled_dev}" \
  --max-examples "${MAX_EXAMPLES:-200}" \
  --num-runs "${NUM_RUNS:-3}" \
  --test-size "${TEST_SIZE:-0.2}" \
  --temperature "${TEMPERATURE:-0.0}" \
  --max-tokens "${MAX_TOKENS:-256}" \
  --seed "${SEED:-42}" \
  --output-dir "$OUTPUT_DIR"

echo ""
echo "Done. Text-tree artifacts are in: $OUTPUT_DIR"
