from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

from src.multimodal_captcha.baseline import color_grounding_predict
from src.multimodal_captcha.dataset import CaptchaDataset
from src.multimodal_captcha.generator import draw_prediction_overlay, generate_dataset
from src.multimodal_captcha.model import MultimodalGridLocator, build_model_from_checkpoint, predict_index
from src.multimodal_captcha.template_matcher import template_grounding_predict
from src.multimodal_captcha.trajectory import generate_mouse_trajectory, random_point_in_cell
from src.multimodal_captcha.visualize import draw_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one prediction and save a visualization.")
    parser.add_argument("--data-dir", default="data/synthetic_captcha")
    parser.add_argument("--checkpoint", default="outputs/model.pt")
    parser.add_argument("--mode", choices=["baseline", "template", "model"], default="template")
    parser.add_argument("--output", default="outputs/prediction.png")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "manifest.jsonl").exists():
        generate_dataset(data_dir, 60, seed=11)

    checkpoint_path = Path(args.checkpoint)
    if args.mode == "model" and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        vocab = checkpoint["vocab"]
        object_vocab = checkpoint.get("object_vocab", {})
        model = build_model_from_checkpoint(checkpoint, len(vocab), len(object_vocab))
        model.load_state_dict(checkpoint["model"])
    else:
        from src.multimodal_captcha.dataset import build_vocab

        vocab = build_vocab(data_dir / "manifest.jsonl")
        model = MultimodalGridLocator(vocab_size=len(vocab))

    dataset = CaptchaDataset(data_dir, split="val", vocab=vocab)
    idx = random.randrange(len(dataset))
    sample = dataset[idx]
    record = dataset.records[idx]
    image = Image.open(data_dir / record["image"]).convert("RGB")
    if args.mode == "model":
        pred, probs = predict_index(model, sample["image"], sample["text"])
    elif args.mode == "template":
        pred, probs = template_grounding_predict(image, record["prompt"])
    else:
        pred, probs = color_grounding_predict(image, record["prompt"])
    overlay = draw_prediction_overlay(image, record["target_index"], pred)
    rng = random.Random(idx)
    click = random_point_in_cell(pred, image.size[0], rng)
    points = generate_mouse_trajectory(click, seed=idx, image_size=image.size[0])
    vis = draw_trajectory(overlay, points)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    vis.save(output)

    print(json.dumps({
        "prompt": record["prompt"],
        "target_index": record["target_index"],
        "predicted_index": pred,
        "mode": args.mode,
        "probabilities": [round(float(x), 4) for x in probs],
        "saved": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
