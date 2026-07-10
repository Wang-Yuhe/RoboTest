from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from src.multimodal_captcha.action_checkpoint import build_action_model_from_checkpoint
from src.multimodal_captcha.action_sequence import (
    actions_to_tokens,
    cell_logits_to_actions,
    cell_logits_to_topk_actions,
    compute_action_metrics,
    target_indices_to_actions,
)
from src.multimodal_captcha.dataset import ActionSequenceDataset
from src.multimodal_captcha.model import ActionCellSelector
from src.multimodal_captcha.trajectory import generate_mouse_trajectory, random_point_in_cell
from src.multimodal_captcha.visualize import draw_trajectory


DEFAULT_THRESHOLD_GRID = "0.03,0.05,0.07,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7"


def parse_threshold_grid(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("--threshold-grid must contain at least one value.")
    return values


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but CUDA is not available.")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def load_action_model(checkpoint_path: str | Path, device: str) -> tuple[ActionCellSelector, dict[str, int], dict[str, int], dict]:
    return build_action_model_from_checkpoint(checkpoint_path, device)


def collect_predictions(
    model: ActionCellSelector,
    dataset: ActionSequenceDataset,
    device: str,
    batch_size: int,
    progress_every: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, list[list[int]], list[dict]]:
    loader = DataLoader(dataset, batch_size=batch_size)
    logits_batches = []
    count_logits_batches = []
    target_batches = []
    action_targets: list[list[int]] = []
    start_time = time.time()
    consumed = 0
    total = len(dataset)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            if getattr(model, "count_head", None) is not None:
                logits, _, count_logits = model(batch["image"].to(device), batch["text"].to(device), return_aux=True)
                count_logits_batches.append(count_logits.cpu())
                logits = logits.cpu()
            else:
                logits = model(batch["image"].to(device), batch["text"].to(device)).cpu()
            logits_batches.append(logits)
            target_batches.append(batch["cell_targets"].cpu())
            action_targets.extend(batch["action_targets"].tolist())
            consumed += int(logits.shape[0])

            if progress_every > 0 and (batch_idx == 1 or consumed % progress_every == 0 or consumed == total):
                elapsed = time.time() - start_time
                examples_per_sec = consumed / elapsed if elapsed > 0 else 0.0
                remaining = (total - consumed) / examples_per_sec if examples_per_sec > 0 else 0.0
                print(
                    json.dumps(
                        {
                            "status": "progress",
                            "done": consumed,
                            "total": total,
                            "percent": round(consumed / max(total, 1) * 100, 2),
                            "examples_per_sec": round(examples_per_sec, 2),
                            "eta_sec": round(remaining, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    if logits_batches:
        logits_all = torch.cat(logits_batches, dim=0)
        targets_all = torch.cat(target_batches, dim=0)
    else:
        logits_all = torch.empty((0, 9), dtype=torch.float32)
        targets_all = torch.empty((0, 9), dtype=torch.float32)
    count_logits_all = torch.cat(count_logits_batches, dim=0) if count_logits_batches else None
    return logits_all, targets_all, count_logits_all, action_targets, dataset.records


def clicked_cells_from_actions(actions: list[dict]) -> list[int]:
    clicked = []
    pending_cell = None
    for action in actions:
        if action.get("type") == "move_to_cell":
            pending_cell = int(action["cell"])
        elif action.get("type") == "click" and pending_cell is not None:
            clicked.append(pending_cell)
            pending_cell = None
        elif action.get("type") == "done":
            break
    return clicked


def class_name(record: dict) -> str:
    return str(record.get("target_object") or record.get("target_class_key") or "unknown")


def empty_class_stat() -> dict[str, int]:
    return {
        "total": 0,
        "cell_exact": 0,
        "click_order": 0,
        "true_positive": 0,
        "predicted_positive": 0,
        "target_positive": 0,
    }


def finalize_binary_metrics(stats: dict[str, int]) -> dict[str, float | int]:
    total = max(int(stats["total"]), 1)
    return {
        "total": int(stats["total"]),
        "cell_exact_match": stats["cell_exact"] / total,
        "click_order_accuracy": stats["click_order"] / total,
        "cell_precision": stats["true_positive"] / max(int(stats["predicted_positive"]), 1),
        "cell_recall": stats["true_positive"] / max(int(stats["target_positive"]), 1),
    }


def score_predictions(
    logits: torch.Tensor,
    cell_targets: torch.Tensor,
    action_targets: list[list[int]],
    records: list[dict],
    threshold: float,
    decode_policy: str = "threshold",
    count_logits: torch.Tensor | None = None,
    max_count: int = 4,
) -> dict:
    if decode_policy == "threshold":
        actions = cell_logits_to_actions(logits, threshold=threshold)
    elif decode_policy == "topk_count":
        if count_logits is None:
            raise ValueError("decode_policy='topk_count' requires count_logits.")
        actions = cell_logits_to_topk_actions(logits, count_logits, min_count=1, max_count=max_count)
    else:
        raise ValueError(f"Unknown decode policy: {decode_policy}")
    action_predictions = [actions_to_tokens(row) for row in actions]
    action_metrics = compute_action_metrics(action_predictions, action_targets)
    pred_cells = torch.zeros_like(cell_targets)
    for row_idx, row_actions in enumerate(actions):
        for cell in clicked_cells_from_actions(row_actions):
            pred_cells[row_idx, cell] = 1.0

    exact_rows = (pred_cells == cell_targets).all(dim=1)
    true_positive = int(((pred_cells == 1) & (cell_targets == 1)).sum())
    predicted_positive = int((pred_cells == 1).sum())
    target_positive = int((cell_targets == 1).sum())

    per_class_raw: dict[str, dict[str, int]] = defaultdict(empty_class_stat)
    predicted_count_histogram: dict[str, int] = defaultdict(int)
    target_count_histogram: dict[str, int] = defaultdict(int)
    count_correct = 0
    for idx, record in enumerate(records):
        label = class_name(record)
        stats = per_class_raw[label]
        pred = pred_cells[idx]
        target = cell_targets[idx]
        predicted_count = int(pred.sum().item())
        target_count = int(target.sum().item())
        predicted_count_histogram[str(predicted_count)] += 1
        target_count_histogram[str(target_count)] += 1
        count_correct += int(predicted_count == target_count)
        stats["total"] += 1
        stats["cell_exact"] += int(bool(exact_rows[idx]))
        stats["click_order"] += int(clicked_cells_from_actions(actions[idx]) == list(map(int, record.get("target_indices", []))))
        stats["true_positive"] += int(((pred == 1) & (target == 1)).sum())
        stats["predicted_positive"] += int((pred == 1).sum())
        stats["target_positive"] += int((target == 1).sum())

    metrics = {
        **action_metrics,
        "cell_exact_match": int(exact_rows.sum()) / max(len(records), 1),
        "cell_precision": true_positive / max(predicted_positive, 1),
        "cell_recall": true_positive / max(target_positive, 1),
        "per_class": {
            key: finalize_binary_metrics(value)
            for key, value in sorted(per_class_raw.items(), key=lambda item: (-item[1]["total"], item[0]))
        },
        "predicted_count_histogram": dict(sorted(predicted_count_histogram.items())),
        "target_count_histogram": dict(sorted(target_count_histogram.items())),
        "count_accuracy": count_correct / max(len(records), 1),
    }
    return metrics


def select_threshold(
    model: ActionCellSelector,
    data_dir: Path,
    vocab: dict[str, int],
    object_vocab: dict[str, int],
    device: str,
    batch_size: int,
    grid: list[float],
) -> tuple[float, list[dict]]:
    val_set = ActionSequenceDataset(data_dir, split="val", vocab=vocab, object_vocab=object_vocab)
    if len(val_set) == 0:
        raise SystemExit(f"No records found for split 'val' in {data_dir}.")
    logits, targets, _, action_targets, records = collect_predictions(
        model,
        val_set,
        device=device,
        batch_size=batch_size,
        progress_every=0,
    )
    candidates = []
    for threshold in grid:
        metrics = score_predictions(logits, targets, action_targets, records, threshold)
        candidates.append(
            {
                "threshold": threshold,
                "cell_exact_match": metrics["cell_exact_match"],
                "click_order_accuracy": metrics["click_order_accuracy"],
                "cell_precision": metrics["cell_precision"],
                "cell_recall": metrics["cell_recall"],
            }
        )
    best = max(
        candidates,
        key=lambda item: (
            item["cell_exact_match"],
            item["click_order_accuracy"],
            item["cell_precision"],
            -abs(item["threshold"] - 0.5),
        ),
    )
    return float(best["threshold"]), candidates


def draw_cell_boxes(image: Image.Image, indices: Iterable[int], color: tuple[int, int, int], width: int = 5) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    cell = out.size[0] // 3
    for index in indices:
        row, col = divmod(int(index), 3)
        x0, y0 = col * cell + 4, row * cell + 4
        x1, y1 = x0 + cell - 8, y0 + cell - 8
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
    return out


def draw_action_trajectory(image: Image.Image, clicked_cells: list[int], seed: int = 0) -> Image.Image:
    rng = random.Random(seed)
    current = None
    out = image
    for step, cell in enumerate(clicked_cells):
        target = random_point_in_cell(cell, image.size[0], rng)
        points = generate_mouse_trajectory(target, start=current, seed=seed + step, image_size=image.size[0])
        current = (int(points[-1][0]), int(points[-1][1]))
        out = draw_trajectory(out, points)
    return out


def save_failure_visualizations(
    data_dir: Path,
    records: list[dict],
    logits: torch.Tensor,
    cell_targets: torch.Tensor,
    threshold: float,
    failure_dir: Path,
    max_failures: int,
    decode_policy: str = "threshold",
    count_logits: torch.Tensor | None = None,
    max_count: int = 4,
) -> list[dict]:
    failure_dir.mkdir(parents=True, exist_ok=True)
    if max_failures <= 0:
        return []

    if decode_policy == "threshold":
        actions = cell_logits_to_actions(logits, threshold=threshold)
    elif decode_policy == "topk_count":
        if count_logits is None:
            raise ValueError("decode_policy='topk_count' requires count_logits.")
        actions = cell_logits_to_topk_actions(logits, count_logits, min_count=1, max_count=max_count)
    else:
        raise ValueError(f"Unknown decode policy: {decode_policy}")
    failures = []
    for idx, record in enumerate(records):
        true_cells = [int(value) for value in record.get("target_indices", [])]
        pred_cells = clicked_cells_from_actions(actions[idx])
        target = cell_targets[idx]
        pred = torch.zeros_like(target)
        for cell in pred_cells:
            pred[cell] = 1.0
        if bool((pred == target).all()):
            continue

        image = Image.open(data_dir / record["image"]).convert("RGB")
        vis = draw_cell_boxes(image, true_cells, (34, 160, 90), width=4)
        vis = draw_cell_boxes(vis, pred_cells, (220, 54, 46), width=5)
        vis = draw_action_trajectory(vis, pred_cells, seed=idx + 123)
        failure_path = failure_dir / f"failure_{len(failures):03d}.png"
        vis.save(failure_path)
        failures.append(
            {
                "image": str(failure_path),
                "source_image": record["image"],
                "prompt": record["prompt"],
                "target_object": class_name(record),
                "target_indices": true_cells,
                "predicted_indices": pred_cells,
                "actions": actions[idx],
            }
        )
        if len(failures) >= max_failures:
            break
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a click-all action model on train/val/test splits.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output-dir", default="outputs/action_eval")
    parser.add_argument("--threshold", default="0.5", help="Float threshold or 'auto' to choose on validation split.")
    parser.add_argument(
        "--threshold-grid",
        default=DEFAULT_THRESHOLD_GRID,
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--decode-policy", choices=["threshold", "topk_count"], default="threshold")
    parser.add_argument("--max-failures", type=int, default=24)
    parser.add_argument("--max-samples", type=int, default=0, help="Limit evaluated records for smoke tests. 0 uses all.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model, vocab, object_vocab, config = load_action_model(args.checkpoint, device)
    threshold_candidates = None
    if args.threshold == "auto":
        threshold, threshold_candidates = select_threshold(
            model,
            data_dir,
            vocab,
            object_vocab,
            device=device,
            batch_size=args.batch_size,
            grid=parse_threshold_grid(args.threshold_grid),
        )
        threshold_source = "auto_val"
    else:
        threshold = float(args.threshold)
        threshold_source = "fixed"

    dataset = ActionSequenceDataset(
        data_dir,
        split=args.split,
        vocab=vocab,
        object_vocab=object_vocab,
        max_action_len=int(config.get("max_action_len", 10)),
    )
    if len(dataset) == 0:
        raise SystemExit(f"No records found for split '{args.split}' in {data_dir}.")
    if args.max_samples > 0:
        dataset.records = dataset.records[: args.max_samples]
    if len(dataset) == 0:
        raise SystemExit(f"No records left for split '{args.split}' after applying --max-samples.")
    print(
        json.dumps(
            {
                "status": "start",
                "split": args.split,
                "total": len(dataset),
                "threshold": threshold,
                "threshold_source": threshold_source,
                "decode_policy": args.decode_policy,
                "device": device,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    logits, targets, count_logits, action_targets, records = collect_predictions(
        model,
        dataset,
        device=device,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
    )
    metrics = score_predictions(
        logits,
        targets,
        action_targets,
        records,
        threshold,
        decode_policy=args.decode_policy,
        count_logits=count_logits,
        max_count=int(config.get("max_count", 4)),
    )
    failures = save_failure_visualizations(
        data_dir=data_dir,
        records=records,
        logits=logits,
        cell_targets=targets,
        threshold=threshold,
        failure_dir=output_dir / "failures",
        max_failures=args.max_failures,
        decode_policy=args.decode_policy,
        count_logits=count_logits,
        max_count=int(config.get("max_count", 4)),
    )
    metrics = {
        "split": args.split,
        "total": len(dataset),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "decode_policy": args.decode_policy,
        "failures_saved": len(failures),
        "failures": failures,
        **metrics,
    }
    if threshold_candidates is not None:
        metrics["threshold_candidates"] = threshold_candidates

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: value for key, value in metrics.items() if key not in {"failures", "per_class"}}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
