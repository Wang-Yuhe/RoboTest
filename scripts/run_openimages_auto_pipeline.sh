#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="${1:-data/openimages_raw}"
PHOTO_ROOT="${2:-data/photo_objects}"
GRID_DIR="${3:-data/photo_grid}"
MAX_SAMPLES="${4:-600}"
CROPS_PER_CLASS="${5:-80}"
GRID_SAMPLES="${6:-800}"
EPOCHS="${7:-20}"
MODEL_SIZE="${8:-small}"

cd "$(dirname "$0")/.."

echo "==> Downloading Open Images subset and exporting object crops"
python3 scripts/download_openimages_subset.py \
  --output-dir "$RAW_DIR" \
  --photo-root "$PHOTO_ROOT" \
  --max-samples "$MAX_SAMPLES" \
  --crops-per-class "$CROPS_PER_CLASS"

echo
echo "==> Building photo-grid dataset and training model"
bash scripts/run_photo_grid_pipeline.sh \
  "$PHOTO_ROOT" \
  "$GRID_DIR" \
  "$GRID_SAMPLES" \
  "$EPOCHS" \
  "$MODEL_SIZE"

echo
echo "==> Open Images auto pipeline complete"
echo "Raw Open Images data: $RAW_DIR"
echo "Cropped object photos: $PHOTO_ROOT"
echo "Photo grid dataset: $GRID_DIR"
