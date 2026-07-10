from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.multimodal_captcha.action_checkpoint import build_action_model_from_checkpoint
from src.multimodal_captcha.action_sequence import cell_logits_to_actions, cell_logits_to_topk_actions
from src.multimodal_captcha.dataset import encode_text
from src.multimodal_captcha.model import ActionCellSelector
from src.multimodal_captcha.trajectory import generate_mouse_trajectory, random_point_in_cell
from src.multimodal_captcha.visualize import draw_trajectory


def load_action_model(checkpoint_path: str | Path, device: str) -> tuple[ActionCellSelector, dict[str, int], dict]:
    model, vocab, _, config = build_action_model_from_checkpoint(checkpoint_path, device)
    return model, vocab, config


def draw_cell_boxes(image: Image.Image, indices: list[int], color: tuple[int, int, int]) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    cell = out.size[0] // 3
    for index in indices:
        row, col = divmod(index, 3)
        x0, y0 = col * cell + 4, row * cell + 4
        x1, y1 = x0 + cell - 8, y0 + cell - 8
        draw.rectangle([x0, y0, x1, y1], outline=color, width=5)
    return out


def actions_to_clicked_cells(actions: list[dict]) -> list[int]:
    cells = []
    pending_cell = None
    for action in actions:
        if action["type"] == "move_to_cell":
            pending_cell = int(action["cell"])
        elif action["type"] == "click" and pending_cell is not None:
            cells.append(pending_cell)
            pending_cell = None
        elif action["type"] == "done":
            break
    return cells


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict click-all action sequence for one 3x3 CAPTCHA image.")
    parser.add_argument("--image", required=True, help="Path to a 3x3 CAPTCHA image.")
    parser.add_argument("--prompt", required=True, help="Chinese instruction, such as 请点击所有汽车.")
    parser.add_argument("--checkpoint", required=True, help="Action cell selector checkpoint.")
    parser.add_argument("--output", default="outputs/action_prediction.png")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--decode-policy", choices=["threshold", "topk_count"], default="threshold")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but CUDA is not available.")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    image = Image.open(args.image).convert("RGB")
    if image.width != image.height:
        raise ValueError("Input image must be square, such as 192x192 or 288x288.")
    if image.width % 3 != 0:
        raise ValueError("Input image width must be divisible by 3.")

    model, vocab, config = load_action_model(args.checkpoint, device)
    arr = torch.tensor(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0, dtype=torch.float32)
    text = encode_text(args.prompt, vocab)
    with torch.no_grad():
        if args.decode_policy == "topk_count":
            if getattr(model, "count_head", None) is None:
                raise ValueError("--decode-policy topk_count requires a checkpoint with use_count_head=true.")
            logits, _, count_logits = model(arr.unsqueeze(0).to(device), text.unsqueeze(0).to(device), return_aux=True)
        else:
            logits = model(arr.unsqueeze(0).to(device), text.unsqueeze(0).to(device))
            count_logits = None
    probs = torch.sigmoid(logits[0].cpu()).numpy()
    if args.decode_policy == "topk_count":
        actions = cell_logits_to_topk_actions(logits, count_logits, min_count=1, max_count=int(config.get("max_count", 4)))[0]
    else:
        actions = cell_logits_to_actions(logits, threshold=args.threshold)[0]
    predicted_indices = actions_to_clicked_cells(actions)

    vis = draw_cell_boxes(image, predicted_indices, (220, 54, 46))
    vis = draw_action_trajectory(vis, predicted_indices, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    vis.save(output)

    result = {
        "prompt": args.prompt,
        "predicted_indices": predicted_indices,
        "actions": actions,
        "cell_probabilities": [round(float(value), 4) for value in probs.tolist()],
        "threshold": args.threshold,
        "decode_policy": args.decode_policy,
        "saved": str(output),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
