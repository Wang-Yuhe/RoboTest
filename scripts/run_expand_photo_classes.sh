#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="${1:-data/openimages_broad_raw}"
PHOTO_ROOT="${2:-data/photo_objects}"
MAX_SAMPLES="${3:-20000}"
CROPS_PER_CLASS="${4:-100}"
CLASS_PRESET="${5:-broad}"
MIN_IMAGES="${6:-100}"
PYTHON="${PYTHON:-python}"

cd "$(dirname "$0")/.."

echo "==> Downloading Open Images '$CLASS_PRESET' class preset"
"$PYTHON" scripts/download_openimages_subset.py \
  --output-dir "$RAW_DIR" \
  --photo-root "$PHOTO_ROOT" \
  --class-preset "$CLASS_PRESET" \
  --max-samples "$MAX_SAMPLES" \
  --crops-per-class "$CROPS_PER_CLASS"

echo
echo "==> Reporting class coverage"
"$PYTHON" scripts/report_photo_objects.py \
  --photo-root "$PHOTO_ROOT" \
  --min-images "$MIN_IMAGES" \
  --json-output outputs/photo_object_report.json

echo
echo "==> Done. Next build a larger photo-grid dataset:"
echo "$PYTHON scripts/build_photo_grid_dataset.py --photo-root $PHOTO_ROOT --output-dir data/photo_grid_100cls --num-samples 10000 --min-images-per-class $MIN_IMAGES --max-classes 120 --hard-augment"
