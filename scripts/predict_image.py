from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from src.multimodal_captcha.baseline import color_grounding_predict
from src.multimodal_captcha.dataset import encode_text
from src.multimodal_captcha.generator import draw_prediction_overlay
from src.multimodal_captcha.model import MultimodalGridLocator, build_model_from_checkpoint, predict_index
from src.multimodal_captcha.template_matcher import template_grounding_predict
from src.multimodal_captcha.trajectory import cell_center, generate_mouse_trajectory
from src.multimodal_captcha.visualize import draw_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a target cell for one input image and prompt.")
    parser.add_argument("--image", required=True, help="Path to a 3x3 CAPTCHA image.")
    parser.add_argument("--prompt", required=True, help="Chinese instruction, such as 请点击黄色自行车.")
    parser.add_argument("--mode", choices=["template", "baseline"], default="template")
    parser.add_argument("--checkpoint", default=None, help="Optional neural model checkpoint.")
    parser.add_argument("--output", default="outputs/custom_prediction.png")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    if image.width != image.height:
        raise ValueError("Input image must be square, such as 192x192 or 288x288.")
    if image.width % 3 != 0:
        raise ValueError("Input image width must be divisible by 3.")

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        vocab = checkpoint["vocab"]
        object_vocab = checkpoint.get("object_vocab", {})
        model = build_model_from_checkpoint(checkpoint, len(vocab), len(object_vocab))
        model.load_state_dict(checkpoint["model"])
        arr = torch.tensor(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0, dtype=torch.float32)
        text = encode_text(args.prompt, vocab)
        pred, probs = predict_index(model, arr, text)
    elif args.mode == "template":
        pred, probs = template_grounding_predict(image, args.prompt)
    else:
        pred, probs = color_grounding_predict(image, args.prompt)

    click = cell_center(pred, image.width)
    overlay = draw_prediction_overlay(image, target_index=pred, predicted_index=pred)
    trajectory = generate_mouse_trajectory(click, seed=pred)
    vis = draw_trajectory(overlay, trajectory)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    vis.save(output)

    result = {
        "prompt": args.prompt,
        "mode": args.mode,
        "predicted_index": pred,
        "click": click,
        "probabilities": [round(float(p), 4) for p in probs],
        "saved": str(output),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
