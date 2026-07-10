from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image, ImageOps


def candidate_grid_boxes(width: int, height: int, rows: int, cols: int) -> list[dict]:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive.")
    boxes = []
    for row in range(rows):
        y0 = round(row * height / rows)
        y1 = round((row + 1) * height / rows)
        for col in range(cols):
            x0 = round(col * width / cols)
            x1 = round((col + 1) * width / cols)
            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2
            boxes.append(
                {
                    "index": len(boxes),
                    "bbox": [x0, y0, x1, y1],
                    "center_pixel": [center_x, center_y],
                    "center_norm": [center_x / width, center_y / height],
                }
            )
    return boxes


def point_in_bbox(point: Iterable[float], bbox: Iterable[float]) -> bool:
    x, y = [float(value) for value in point]
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return x0 <= x <= x1 and y0 <= y <= y1


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
        raise SystemExit("ScreenSpot-Pro CLIP adapter requires transformers. Install transformers first.") from exc
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


def load_task_records(test_path: Path, task: str, max_samples: int) -> list[dict]:
    if test_path.is_file():
        paths = [test_path]
    elif task == "all":
        paths = sorted(test_path.glob("*.json"))
    else:
        paths = [test_path / f"{task}.json"]
    records: list[dict] = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"ScreenSpot-Pro task file does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise SystemExit(f"Expected a JSON list in {path}")
        for record in payload:
            item = dict(record)
            item["_task"] = path.stem
            records.append(item)
            if max_samples > 0 and len(records) >= max_samples:
                return records
    if not records:
        raise SystemExit(f"No ScreenSpot-Pro records found in {test_path}")
    return records


def crop_box(image: Image.Image, bbox: list[int]) -> Image.Image:
    x0, y0, x1, y1 = bbox
    return image.crop((x0, y0, x1, y1))


def clip_patch_scores(model, processor, patches: list[Image.Image], instruction: str, device: str) -> list[float]:
    inputs = processor(text=[instruction], images=patches, return_tensors="pt", padding=True)
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


def select_prompt(record: dict, language: str) -> str:
    if language == "cn" and record.get("instruction_cn"):
        return str(record["instruction_cn"])
    if language == "en" and record.get("instruction"):
        return str(record["instruction"])
    return str(record.get("instruction") or record.get("instruction_cn") or "")


def predict_with_clip_grid(
    image: Image.Image,
    instruction: str,
    model,
    processor,
    device: str,
    rows: int,
    cols: int,
) -> tuple[list[float], dict]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    boxes = candidate_grid_boxes(image.width, image.height, rows, cols)
    patches = [crop_box(image, box["bbox"]) for box in boxes]
    scores = clip_patch_scores(model, processor, patches, instruction, device)
    best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
    best = boxes[best_idx]
    return best["center_pixel"], {
        "candidate_index": best_idx,
        "candidate_bbox": best["bbox"],
        "candidate_score": float(scores[best_idx]),
        "rows": rows,
        "cols": cols,
    }


def predict_mock_oracle(record: dict) -> tuple[list[float], dict]:
    bbox = [float(value) for value in record["bbox"]]
    point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    return point, {"mode": "mock_oracle"}


def evaluate_results(details: list[dict]) -> dict:
    total = len(details)
    correct = sum(1 for item in details if item.get("correctness") == "correct")
    wrong_format = sum(1 for item in details if item.get("correctness") == "wrong_format")

    metrics = {
        "overall": {
            "num_total": total,
            "num_correct_action": correct,
            "wrong_format_num": wrong_format,
            "action_acc": correct / max(total, 1),
        }
    }
    by_ui_type: dict[str, list[dict]] = defaultdict(list)
    for item in details:
        by_ui_type[str(item.get("ui_type") or "unknown")].append(item)
    for ui_type, items in sorted(by_ui_type.items()):
        ui_correct = sum(1 for item in items if item.get("correctness") == "correct")
        metrics["overall"][f"{ui_type}_acc"] = ui_correct / max(len(items), 1)
        metrics["overall"][f"{ui_type}_num"] = len(items)
    return {"metrics": metrics, "details": details}


def run_adapter(args: argparse.Namespace) -> dict:
    device = resolve_device(args.device)
    records = load_task_records(Path(args.screenspot_test), args.task, args.max_samples)
    model = processor = None
    if not args.mock_oracle:
        model, processor = load_clip(args.model_name, device)

    details = []
    image_root = Path(args.screenspot_imgs)
    for idx, record in enumerate(records):
        instruction = select_prompt(record, args.language)
        bbox = record.get("bbox")
        img_size = record.get("img_size")
        if not bbox or not img_size:
            details.append(
                {
                    "sample_id": record.get("id", idx),
                    "img_filename": record.get("img_filename"),
                    "instruction": instruction,
                    "bbox": bbox,
                    "pred": None,
                    "pred_norm": None,
                    "correctness": "wrong_format",
                    "raw_response": "missing bbox or img_size",
                    "ui_type": record.get("ui_type"),
                    "task": record.get("_task"),
                }
            )
            continue

        image_path = image_root / str(record["img_filename"])
        if args.mock_oracle:
            pred, raw = predict_mock_oracle(record)
        else:
            image = Image.open(image_path)
            pred, raw = predict_with_clip_grid(
                image,
                instruction,
                model,
                processor,
                device,
                rows=args.grid_rows,
                cols=args.grid_cols,
            )
        width, height = float(img_size[0]), float(img_size[1])
        pred_norm = [float(pred[0]) / width, float(pred[1]) / height]
        correctness = "correct" if point_in_bbox(pred, bbox) else "wrong"
        details.append(
            {
                "sample_id": record.get("id", idx),
                "img_filename": record.get("img_filename"),
                "instruction": instruction,
                "bbox": bbox,
                "img_size": img_size,
                "pred": [float(pred[0]), float(pred[1])],
                "pred_norm": pred_norm,
                "correctness": correctness,
                "raw_response": raw,
                "ui_type": record.get("ui_type"),
                "group": record.get("group"),
                "platform": record.get("platform"),
                "application": record.get("application"),
                "task": record.get("_task"),
            }
        )
        if args.progress_every > 0 and (idx + 1 == 1 or (idx + 1) % args.progress_every == 0 or idx + 1 == len(records)):
            print(json.dumps({"status": "progress", "done": idx + 1, "total": len(records)}, ensure_ascii=False), flush=True)

    report = evaluate_results(details)
    report["meta"] = {
        "benchmark": "ScreenSpot-Pro",
        "adapter": "clip_grid_patch",
        "model_name": None if args.mock_oracle else args.model_name,
        "device": device,
        "task": args.task,
        "language": args.language,
        "grid_rows": args.grid_rows,
        "grid_cols": args.grid_cols,
        "mock_oracle": args.mock_oracle,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CLIP grid-patch adapter on ScreenSpot-Pro style data.")
    parser.add_argument("--screenspot-imgs", required=True, help="Directory containing ScreenSpot-Pro images.")
    parser.add_argument("--screenspot-test", required=True, help="Directory of task JSON files or one JSON file.")
    parser.add_argument("--task", default="all", help="Task JSON stem, or 'all'.")
    parser.add_argument("--output", required=True, help="Output JSON log with metrics and details.")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--language", choices=["en", "cn", "auto"], default="en")
    parser.add_argument("--grid-rows", type=int, default=8)
    parser.add_argument("--grid-cols", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--mock-oracle", action="store_true", help="Testing mode: predict bbox center without loading CLIP.")
    args = parser.parse_args()

    payload = run_adapter(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["metrics"]["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
