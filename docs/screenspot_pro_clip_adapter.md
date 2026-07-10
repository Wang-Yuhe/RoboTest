# ScreenSpot-Pro CLIP Adapter

This adapter gives RoboTest a ScreenSpot-Pro-compatible local baseline without training a GUI model.

It reads ScreenSpot-Pro annotation JSON files, divides each screenshot into a regular grid, scores each patch with CLIP against the instruction, predicts the best patch center as a click point, and writes a JSON report with `metrics.overall.action_acc` and per-sample `details`.

## Data Layout

Download ScreenSpot-Pro from Hugging Face or prepare the same structure manually:

```text
data/screenspot_pro/
  images/
    ...
  annotations/
    excel_macos.json
    photoshop_windows.json
    ...
```

The official dataset is large, so the adapter does not download it automatically.

## Smoke Test

The repository tests use `--mock-oracle` and do not load CLIP:

```bash
python -m unittest tests.test_screenspot_pro_clip_adapter -v
```

## Run CLIP Grid Baseline

Install the optional CLIP dependency first (it is also included in the full `requirements.txt`):

```bash
python -m pip install transformers
```

```bash
python scripts/screenspot_pro_clip_adapter.py \
  --screenspot-imgs data/screenspot_pro/images \
  --screenspot-test data/screenspot_pro/annotations \
  --task all \
  --output outputs/screenspot_pro_clip_grid8.json \
  --model-name openai/clip-vit-base-patch32 \
  --device auto \
  --grid-rows 8 \
  --grid-cols 8 \
  --language en \
  --progress-every 25
```

For a quick local check:

```bash
python scripts/screenspot_pro_clip_adapter.py \
  --screenspot-imgs data/screenspot_pro/images \
  --screenspot-test data/screenspot_pro/annotations \
  --task excel_macos \
  --output outputs/screenspot_pro_clip_excel_macos_20.json \
  --max-samples 20 \
  --device auto
```

## Output

The output JSON has:

- `meta`: adapter config and model name.
- `metrics.overall.action_acc`: fraction of predicted points inside the target bbox.
- `details`: one row per sample with `pred`, `pred_norm`, `bbox`, `correctness`, and the selected patch metadata.

This is intended as a reproducible local baseline and adapter path. It is not an official leaderboard submission package.

## Notes

- Increasing `--grid-rows/--grid-cols` gives finer localization but more CLIP calls.
- CLIP patch retrieval is a weak baseline for professional GUI grounding; strong ScreenSpot-Pro systems use GUI-specialized VLMs, search-region reduction, or coordinate/action heads.
- The official positive-sample rule is simple: a predicted click point is correct if it falls inside the annotated target bbox.
