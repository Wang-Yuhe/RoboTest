#!/usr/bin/env bash
set -euo pipefail

PHOTO_ROOT="${1:-data/photo_objects}"
OUTPUT_DIR="${2:-data/photo_grid}"
NUM_SAMPLES="${3:-800}"
EPOCHS="${4:-20}"
MODEL_SIZE="${5:-small}"
HARD_AUGMENT="${6:-true}"
CHECKPOINT="outputs/photo_model.pt"

cd "$(dirname "$0")/.."

echo "==> Building photo-grid dataset from $PHOTO_ROOT"
python3 scripts/build_photo_grid_dataset.py \
  --photo-root "$PHOTO_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --num-samples "$NUM_SAMPLES" \
  $([ "$HARD_AUGMENT" = "true" ] && echo "--hard-augment")

echo
echo "==> Training neural model on photo-grid dataset"
python3 scripts/train.py \
  --data-dir "$OUTPUT_DIR" \
  --output "$CHECKPOINT" \
  --epochs "$EPOCHS" \
  --batch-size 64 \
  --lr 0.001 \
  --aux-weight 0.7 \
  --patience 6 \
  --model-size "$MODEL_SIZE"

echo
echo "==> Evaluating neural model on photo-grid dataset"
python3 scripts/evaluate.py \
  --data-dir "$OUTPUT_DIR" \
  --mode model \
  --checkpoint "$CHECKPOINT" \
  --output-dir outputs/eval_photo_grid

echo
echo "==> Saving one photo-grid prediction"
python3 scripts/predict.py \
  --data-dir "$OUTPUT_DIR" \
  --mode model \
  --checkpoint "$CHECKPOINT" \
  --output outputs/photo_grid_prediction.png

echo
echo "==> Done"
echo "Dataset: $OUTPUT_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "Metrics: outputs/eval_photo_grid/model/metrics.json"
echo "Prediction: outputs/photo_grid_prediction.png"
