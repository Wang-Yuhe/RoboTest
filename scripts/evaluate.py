from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from src.multimodal_captcha.baseline import color_grounding_predict
from src.multimodal_captcha.dataset import CaptchaDataset, build_vocab
from src.multimodal_captcha.generator import draw_prediction_overlay, generate_dataset
from src.multimodal_captcha.model import MultimodalGridLocator, build_model_from_checkpoint, predict_index
from src.multimodal_captcha.template_matcher import template_grounding_predict
from src.multimodal_captcha.trajectory import cell_center, generate_mouse_trajectory, random_point_in_cell
from src.multimodal_captcha.visualize import draw_trajectory


def click_distance(predicted_index: int, target_index: int, image_size: int) -> float:
    px, py = cell_center(predicted_index, image_size)
    tx, ty = cell_center(target_index, image_size)
    return math.hypot(px - tx, py - ty)


def load_model(data_dir: Path, checkpoint_path: Path) -> tuple[MultimodalGridLocator, dict[str, int]]:
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        vocab = checkpoint["vocab"]
        object_vocab = checkpoint.get("object_vocab", {})
        model = build_model_from_checkpoint(checkpoint, len(vocab), len(object_vocab))
        model.load_state_dict(checkpoint["model"])
        return model, vocab

    vocab = build_vocab(data_dir / "manifest.jsonl")
    model = MultimodalGridLocator(vocab_size=len(vocab))
    return model, vocab


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate grounding accuracy on the synthetic CAPTCHA dataset.")
    parser.add_argument("--data-dir", default="data/synthetic_captcha")
    parser.add_argument("--checkpoint", default="outputs/model.pt")
    parser.add_argument("--mode", choices=["baseline", "template", "model"], default="baseline")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--max-failures", type=int, default=12)
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N examples. 0 disables progress.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "manifest.jsonl").exists():
        generate_dataset(data_dir, 600, seed=41, image_size=192)

    model, vocab = load_model(data_dir, Path(args.checkpoint))
    dataset = CaptchaDataset(data_dir, split=args.split, vocab=vocab, train_ratio=args.train_ratio)

    output_dir = Path(args.output_dir) / args.mode
    failure_dir = output_dir / "failures"
    failure_dir.mkdir(parents=True, exist_ok=True)

    correct = 0
    top3_correct = 0
    distances = []
    failures = []
    confusion = np.zeros((9, 9), dtype=int)
    start_time = time.time()
    total = len(dataset)
    print(
        json.dumps(
            {
                "status": "start",
                "mode": args.mode,
                "split": args.split,
                "total": total,
                "data_dir": str(data_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for idx, record in enumerate(dataset.records):
        sample = dataset[idx]
        image = Image.open(data_dir / record["image"]).convert("RGB")
        target = int(record["target_index"])

        if args.mode == "model":
            pred, probs = predict_index(model, sample["image"], sample["text"])
            probs_arr = probs.numpy()
        elif args.mode == "template":
            pred, probs_arr = template_grounding_predict(image, record["prompt"])
        else:
            pred, probs_arr = color_grounding_predict(image, record["prompt"])

        top3 = np.argsort(probs_arr)[-3:][::-1].tolist()
        is_correct = pred == target
        correct += int(is_correct)
        top3_correct += int(target in top3)
        distance = click_distance(pred, target, image.size[0])
        distances.append(distance)
        confusion[target, pred] += 1

        if not is_correct and len(failures) < args.max_failures:
            overlay = draw_prediction_overlay(image, target, pred)
            rng_seed = idx + int(target) * 997
            import random

            click = random_point_in_cell(pred, image.size[0], random.Random(rng_seed))
            trajectory = generate_mouse_trajectory(click, seed=rng_seed, image_size=image.size[0])
            vis = draw_trajectory(overlay, trajectory)
            failure_path = failure_dir / f"failure_{len(failures):03d}.png"
            vis.save(failure_path)
            failures.append(
                {
                    "prompt": record["prompt"],
                    "target_index": target,
                    "predicted_index": pred,
                    "top3": top3,
                    "distance": round(distance, 3),
                    "image": str(failure_path),
                }
            )
        done = idx + 1
        if args.progress_every > 0 and (done == 1 or done % args.progress_every == 0 or done == total):
            elapsed = time.time() - start_time
            examples_per_sec = done / elapsed if elapsed > 0 else 0.0
            remaining = (total - done) / examples_per_sec if examples_per_sec > 0 else 0.0
            print(
                json.dumps(
                    {
                        "status": "progress",
                        "done": done,
                        "total": total,
                        "percent": round(done / max(total, 1) * 100, 2),
                        "accuracy_so_far": round(correct / max(done, 1), 4),
                        "examples_per_sec": round(examples_per_sec, 2),
                        "eta_sec": round(remaining, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    metrics = {
        "mode": args.mode,
        "split": args.split,
        "total": total,
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "mean_click_distance": float(np.mean(distances)) if distances else 0.0,
        "median_click_distance": float(np.median(distances)) if distances else 0.0,
        "failures_saved": len(failures),
        "failures": failures,
        "confusion_matrix": confusion.tolist(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k not in {"failures", "confusion_matrix"}}, ensure_ascii=False, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
