#!/usr/bin/env bash
set -euo pipefail

PHOTO_ROOT="${1:-data/photo_objects}"
RAW_DIR="${2:-data/openimages_gap_fill_raw}"
TARGET_CLASSES="${3:-100}"
MIN_IMAGES="${4:-100}"
MAX_SAMPLES="${5:-50000}"
PASSES="${6:-2}"
SPLIT="${7:-train}"
PYTHON="${PYTHON:-python}"

cd "$(dirname "$0")/.."

echo "==> Filling underrepresented Open Images classes"
"$PYTHON" scripts/fill_photo_class_gaps.py \
  --photo-root "$PHOTO_ROOT" \
  --output-dir "$RAW_DIR" \
  --target-classes "$TARGET_CLASSES" \
  --min-images "$MIN_IMAGES" \
  --max-samples "$MAX_SAMPLES" \
  --passes "$PASSES" \
  --split "$SPLIT"

echo
echo "==> Reporting class coverage"
"$PYTHON" scripts/report_photo_objects.py \
  --photo-root "$PHOTO_ROOT" \
  --min-images "$MIN_IMAGES" \
  --json-output outputs/photo_object_report_100.json

echo
echo "==> Done. If usable classes are >= $TARGET_CLASSES, build the grid dataset:"
echo "$PYTHON scripts/build_photo_grid_dataset.py --photo-root $PHOTO_ROOT --output-dir data/photo_grid_100cls --num-samples 10000 --min-images-per-class $MIN_IMAGES --max-classes 120 --hard-augment"
