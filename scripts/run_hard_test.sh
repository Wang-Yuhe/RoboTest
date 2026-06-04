#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data/hard"
OUTPUT_DIR="outputs/eval_hard"
PREDICTION="outputs/prediction_hard.png"
NUM_SAMPLES="${1:-600}"
SEED="${2:-3}"

cd "$(dirname "$0")/.."

echo "==> Generating hard dataset"
python3 scripts/generate_dataset.py \
  --output-dir "$DATA_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --seed "$SEED" \
  --difficulty hard

echo
echo "==> Evaluating template grounding model"
python3 scripts/evaluate.py \
  --data-dir "$DATA_DIR" \
  --mode template \
  --output-dir "$OUTPUT_DIR"

echo
echo "==> Evaluating color baseline"
python3 scripts/evaluate.py \
  --data-dir "$DATA_DIR" \
  --mode baseline \
  --output-dir "$OUTPUT_DIR"

echo
echo "==> Saving one hard prediction visualization"
python3 scripts/predict.py \
  --data-dir "$DATA_DIR" \
  --mode template \
  --output "$PREDICTION"

echo
echo "==> Hard test complete"
echo "Template metrics: $OUTPUT_DIR/template/metrics.json"
echo "Baseline metrics: $OUTPUT_DIR/baseline/metrics.json"
echo "Prediction image: $PREDICTION"

