from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image, ImageOps


def load_records(data_dir: Path, split: str, max_samples: int) -> list[dict]:
    manifest = data_dir / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"Manifest does not exist: {manifest}")
    records = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("split") == split:
            records.append(record)
            if max_samples > 0 and len(records) >= max_samples:
                break
    if not records:
        raise SystemExit(f"No records found for split '{split}' in {data_dir}.")
    return records


def crop_grid_cells(image: Image.Image) -> list[Image.Image]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    cell = image.size[0] // 3
    cells = []
    for idx in range(9):
        row, col = divmod(idx, 3)
        cells.append(image.crop((col * cell, row * cell, (col + 1) * cell, (row + 1) * cell)))
    return cells


def target_text(record: dict) -> str:
    class_key = str(record.get("target_class_key") or record.get("target_object") or "object")
    return f"a photo of a {class_key.replace('_', ' ')}"


def mock_scores(record: dict) -> list[float]:
    targets = set(map(int, record.get("target_indices", [])))
    return [1.0 if idx in targets else 0.0 for idx in range(9)]


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")
    return requested


def load_clip(model_name: str, device: str):
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise SystemExit("CLIP zero-shot eval requires transformers. Install transformers first.") from exc
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, processor


def feature_tensor(output):
    if torch.is_tensor(output):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if isinstance(output, tuple) and output:
        return output[0]
    raise TypeError(f"Unsupported CLIP feature output type: {type(output).__name__}")


def clip_scores(model, processor, cells: list[Image.Image], text: str, device: str) -> list[float]:
    inputs = processor(text=[text], images=cells, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        image_features = feature_tensor(model.get_image_features(pixel_values=inputs["pixel_values"]))
        text_features = feature_tensor(
            model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
        )
    image_features = torch.nn.functional.normalize(image_features, dim=-1)
    text_features = torch.nn.functional.normalize(text_features, dim=-1)
    return (image_features @ text_features[0]).detach().cpu().tolist()


def topk_indices(scores: list[float], k: int) -> list[int]:
    return sorted(idx for idx, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:k])


def threshold_indices(scores: list[float], threshold: float) -> list[int]:
    return [idx for idx, score in enumerate(scores) if float(score) >= threshold]


def update_counts(counts: dict[str, int], predicted: list[int], target: list[int]) -> None:
    pred_set = set(predicted)
    target_set = set(target)
    counts["samples"] += 1
    counts["cell_exact"] += int(pred_set == target_set)
    counts["true_positive"] += len(pred_set & target_set)
    counts["predicted_positive"] += len(pred_set)
    counts["target_positive"] += len(target_set)


def finalize(counts: dict[str, int]) -> dict:
    return {
        "samples": counts["samples"],
        "cell_exact_match": counts["cell_exact"] / max(counts["samples"], 1),
        "cell_precision": counts["true_positive"] / max(counts["predicted_positive"], 1),
        "cell_recall": counts["true_positive"] / max(counts["target_positive"], 1),
    }


def parse_threshold_grid(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise SystemExit("--threshold-grid must contain at least one value.")
    return values


def score_record(record: dict, data_dir: Path, model, processor, device: str, mock: bool) -> list[float]:
    if mock:
        return mock_scores(record)
    image = Image.open(data_dir / record["image"])
    return clip_scores(model, processor, crop_grid_cells(image), target_text(record), device)


def score_threshold(records: list[dict], scores_by_record: list[list[float]], threshold: float) -> dict:
    counts = {"samples": 0, "cell_exact": 0, "true_positive": 0, "predicted_positive": 0, "target_positive": 0}
    for record, scores in zip(records, scores_by_record):
        target = sorted(map(int, record.get("target_indices", [])))
        update_counts(counts, threshold_indices(scores, threshold), target)
    return finalize(counts)


def select_threshold(records: list[dict], scores_by_record: list[list[float]], grid: list[float]) -> tuple[float, list[dict]]:
    candidates = []
    for threshold in grid:
        metrics = score_threshold(records, scores_by_record, threshold)
        candidates.append({"threshold": threshold, **metrics})
    best = max(
        candidates,
        key=lambda item: (
            item["cell_exact_match"],
            item["cell_precision"],
            item["cell_recall"],
            -abs(float(item["threshold"]) - 0.5),
        ),
    )
    return float(best["threshold"]), candidates


def evaluate(args: argparse.Namespace) -> dict:
    data_dir = Path(args.data_dir)
    device = resolve_device(args.device)
    records = load_records(data_dir, args.split, args.max_samples)
    model = processor = None
    if not args.mock_scores:
        model, processor = load_clip(args.model_name, device)

    counts = {"samples": 0, "cell_exact": 0, "true_positive": 0, "predicted_positive": 0, "target_positive": 0}
    fixed_counts = {"samples": 0, "cell_exact": 0, "true_positive": 0, "predicted_positive": 0, "target_positive": 0}
    threshold_counts = {"samples": 0, "cell_exact": 0, "true_positive": 0, "predicted_positive": 0, "target_positive": 0}
    threshold_source = "none"
    threshold_candidates = None
    calibration_samples = 0
    if args.threshold == "none":
        threshold = None
    elif args.threshold == "auto":
        calibration_limit = args.max_calibration_samples if args.max_calibration_samples > 0 else args.max_samples
        calibration_records = load_records(data_dir, args.calibration_split, calibration_limit)
        calibration_scores = [
            score_record(record, data_dir, model, processor, device, args.mock_scores)
            for record in calibration_records
        ]
        threshold, threshold_candidates = select_threshold(
            calibration_records,
            calibration_scores,
            parse_threshold_grid(args.threshold_grid),
        )
        threshold_source = f"auto_{args.calibration_split}"
        calibration_samples = len(calibration_records)
    else:
        threshold = float(args.threshold)
        threshold_source = "fixed"
    examples = []
    for idx, record in enumerate(records):
        target = sorted(map(int, record.get("target_indices", [])))
        scores = score_record(record, data_dir, model, processor, device, args.mock_scores)
        predicted = topk_indices(scores, max(len(target), 1))
        fixed_predicted = topk_indices(scores, args.fixed_topk)
        update_counts(counts, predicted, target)
        update_counts(fixed_counts, fixed_predicted, target)
        threshold_predicted = []
        if threshold is not None:
            threshold_predicted = threshold_indices(scores, threshold)
            update_counts(threshold_counts, threshold_predicted, target)
        if len(examples) < args.max_examples:
            examples.append(
                {
                    "image": record["image"],
                    "target_class_key": record.get("target_class_key"),
                    "target_object": record.get("target_object"),
                    "text": target_text(record),
                    "target_indices": target,
                    "predicted_indices_oracle_topk": predicted,
                    "predicted_indices_fixed_topk": fixed_predicted,
                    "predicted_indices_threshold": threshold_predicted,
                    "scores": [round(float(score), 6) for score in scores],
                }
            )
    return {
        "mode": "mock" if args.mock_scores else "transformers_clip",
        "model_name": None if args.mock_scores else args.model_name,
        "split": args.split,
        "samples": len(records),
        "oracle_topk": finalize(counts),
        "fixed_topk": {"k": args.fixed_topk, **finalize(fixed_counts)},
        "threshold_policy": (
            {
                "threshold": threshold,
                "threshold_source": threshold_source,
                "calibration_samples": calibration_samples,
                "candidates": threshold_candidates,
                **finalize(threshold_counts),
            }
            if threshold is not None
            else None
        ),
        "examples": examples,
        "note": "oracle_topk uses the true target count, so it is a diagnostic upper-bound signal rather than a deployable click policy.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CLIP zero-shot cell ranking on click-all action data.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--fixed-topk", type=int, default=3)
    parser.add_argument("--threshold", default="none", help="Fixed CLIP cosine threshold, 'auto', or 'none'.")
    parser.add_argument("--calibration-split", default="val", help="Split used when --threshold auto.")
    parser.add_argument("--max-calibration-samples", type=int, default=0, help="Calibration sample limit. 0 reuses --max-samples.")
    parser.add_argument("--threshold-grid", default="0.18,0.2,0.22,0.24,0.26,0.28,0.3")
    parser.add_argument("--mock-scores", action="store_true", help="Use target-derived mock scores for fast tests.")
    args = parser.parse_args()
    payload = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("mode", "split", "samples", "oracle_topk")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
