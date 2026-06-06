from __future__ import annotations

import argparse
import base64
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

from src.multimodal_captcha.baseline import color_grounding_predict
from src.multimodal_captcha.dataset import CaptchaDataset, build_vocab
from src.multimodal_captcha.generator import draw_prediction_overlay, generate_dataset
from src.multimodal_captcha.model import MultimodalGridLocator, build_model_from_checkpoint, predict_index
from src.multimodal_captcha.template_matcher import template_grounding_predict
from src.multimodal_captcha.trajectory import generate_mouse_trajectory, random_point_in_cell
from src.multimodal_captcha.visualize import draw_trajectory


def image_to_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static HTML demo without Streamlit.")
    parser.add_argument("--data-dir", default="data/synthetic_captcha")
    parser.add_argument("--output", default="outputs/demo.html")
    parser.add_argument("--num-examples", type=int, default=6)
    parser.add_argument("--mode", choices=["template", "baseline", "model"], default="template")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "manifest.jsonl").exists():
        generate_dataset(data_dir, 600, seed=41, image_size=192, difficulty="medium")

    model = None
    if args.mode == "model" and args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        vocab = checkpoint["vocab"]
        object_vocab = checkpoint.get("object_vocab", {})
        model = build_model_from_checkpoint(checkpoint, len(vocab), len(object_vocab))
        model.load_state_dict(checkpoint["model"])
    else:
        vocab = build_vocab(data_dir / "manifest.jsonl")
    dataset = CaptchaDataset(data_dir, split="val", vocab=vocab)
    rng = random.Random(13)
    indices = rng.sample(range(len(dataset)), min(args.num_examples, len(dataset)))

    output = Path(args.output)
    asset_dir = output.parent / "demo_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    cards = []
    for card_idx, idx in enumerate(indices):
        record = dataset.records[idx]
        sample = dataset[idx]
        image = Image.open(data_dir / record["image"]).convert("RGB")
        if args.mode == "model" and model is not None:
            pred, probs = predict_index(model, sample["image"], sample["text"])
            model_label = "神经模型"
        elif args.mode == "baseline":
            pred, probs = color_grounding_predict(image, record["prompt"])
            model_label = "颜色基线"
        else:
            pred, probs = template_grounding_predict(image, record["prompt"])
            model_label = "模板图文模型"
        baseline_pred, _ = color_grounding_predict(image, record["prompt"])
        overlay = draw_prediction_overlay(image, int(record["target_index"]), pred)
        click = random_point_in_cell(pred, image.size[0], random.Random(idx + card_idx * 37))
        trajectory = generate_mouse_trajectory(click, seed=idx, image_size=image.size[0])
        vis = draw_trajectory(overlay, trajectory)
        image_path = asset_dir / f"example_{card_idx:02d}.png"
        vis.save(image_path)
        cards.append(
            {
                "prompt": record["prompt"],
                "target": int(record["target_index"]),
                "pred": int(pred),
                "baseline_pred": int(baseline_pred),
                "model_label": model_label,
                "correct": pred == int(record["target_index"]),
                "probs": [round(float(p), 3) for p in probs],
                "image": image_to_data_uri(image_path),
            }
        )

    html_cards = "\n".join(
        f"""
        <section class="card">
          <img src="{card['image']}" alt="prediction example">
          <div class="content">
            <h2>{card['prompt']}</h2>
            <p><strong>真实格子:</strong> {card['target']} &nbsp; <strong>{card['model_label']}预测:</strong> {card['pred']} &nbsp; <strong>颜色基线:</strong> {card['baseline_pred']}</p>
            <p><strong>{card['model_label']}是否正确:</strong> {'是' if card['correct'] else '否'}</p>
            <p class="probs">九宫格概率: {json.dumps(card['probs'], ensure_ascii=False)}</p>
          </div>
        </section>
        """
        for card in cards
    )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>多模态验证码定位 Demo</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f6f8; color: #20242a; }}
    header {{ padding: 28px 36px 16px; background: #ffffff; border-bottom: 1px solid #d8dde6; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    header p {{ margin: 0; color: #5d6675; }}
    main {{ max-width: 980px; margin: 24px auto; padding: 0 18px 36px; display: grid; gap: 16px; }}
    .card {{ display: grid; grid-template-columns: 220px 1fr; gap: 18px; background: #ffffff; border: 1px solid #d8dde6; border-radius: 8px; padding: 16px; align-items: center; }}
    .card img {{ width: 192px; height: 192px; image-rendering: pixelated; border: 1px solid #ccd3dd; }}
    h2 {{ margin: 0 0 12px; font-size: 20px; }}
    p {{ margin: 8px 0; line-height: 1.55; }}
    .probs {{ color: #5d6675; font-size: 14px; overflow-wrap: anywhere; }}
    @media (max-width: 680px) {{ .card {{ grid-template-columns: 1fr; }} .card img {{ width: 100%; height: auto; image-rendering: auto; }} }}
  </style>
</head>
<body>
  <header>
    <h1>多模态验证码定位与鼠标轨迹生成</h1>
    <p>绿色框为真实目标，红色框为当前模型预测，蓝线为生成鼠标轨迹。</p>
  </header>
  <main>
    {html_cards}
  </main>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"Saved static demo to {output}")


if __name__ == "__main__":
    main()
