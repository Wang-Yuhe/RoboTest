#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-outputs/photo_model_100cls_attn.pt}"
EXPORT_DIR="${2:-release_artifacts/photo_model_100cls_attn}"
ARCHIVE="${3:-release_artifacts/photo_model_100cls_attn_artifacts.tar.gz}"

LOG_FILE="${CHECKPOINT%.pt}.log.jsonl"
METRICS_FILE="outputs/eval_100cls_attn/model/metrics.json"
DEMO_FILE="outputs/photo_demo_100cls_model.html"

cd "$(dirname "$0")/.."

if [ ! -f "$CHECKPOINT" ]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$EXPORT_DIR"
cp "$CHECKPOINT" "$EXPORT_DIR/"

if [ -f "$LOG_FILE" ]; then
  cp "$LOG_FILE" "$EXPORT_DIR/"
fi

if [ -f "$METRICS_FILE" ]; then
  mkdir -p "$EXPORT_DIR/eval"
  cp "$METRICS_FILE" "$EXPORT_DIR/eval/metrics.json"
fi

if [ -f "$DEMO_FILE" ]; then
  cp "$DEMO_FILE" "$EXPORT_DIR/"
fi

cat > "$EXPORT_DIR/README_MODEL_PACKAGE.md" <<'EOF'
# Trained Model Package

This package contains the trained multimodal CAPTCHA locator checkpoint and related experiment artifacts.

## Files

- `photo_model_100cls_attn.pt`: trained PyTorch checkpoint
- `photo_model_100cls_attn.log.jsonl`: training log, if included
- `eval/metrics.json`: validation metrics, if included
- `photo_demo_100cls_model.html`: standalone HTML demo, if included

## How to use after cloning the repo

Copy the checkpoint into the repo's `outputs/` directory:

```bash
mkdir -p outputs
cp photo_model_100cls_attn.pt outputs/
```

Then run prediction on a custom 9-grid image:

```bash
python scripts/predict_image.py \
  --image path/to/your_grid.png \
  --prompt "请点击汽车" \
  --checkpoint outputs/photo_model_100cls_attn.pt \
  --output outputs/custom_prediction.png
```

To regenerate HTML examples, the teammate also needs the corresponding dataset under `data/photo_grid_100cls/`.
EOF

mkdir -p "$(dirname "$ARCHIVE")"
tar -czf "$ARCHIVE" -C "$(dirname "$EXPORT_DIR")" "$(basename "$EXPORT_DIR")"

echo "Exported artifacts to: $EXPORT_DIR"
echo "Archive: $ARCHIVE"
