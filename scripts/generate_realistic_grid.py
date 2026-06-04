from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src.multimodal_captcha.generator import COLORS, DRAWERS, CellItem, OBJECTS


def add_paper_texture(image: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(image).astype(np.int16)
    noise = np.random.default_rng(rng.randrange(1_000_000)).normal(0, 4, arr.shape).astype(np.int16)
    vignette = np.zeros(arr.shape[:2], dtype=np.float32)
    h, w = vignette.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt(((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2)
    vignette = (1.0 - dist * 0.10)[..., None]
    arr = np.clip((arr + noise) * vignette, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def draw_soft_object(base: Image.Image, item: CellItem, box: tuple[int, int, int, int], rng: random.Random) -> None:
    scale = 3
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    layer = Image.new("RGBA", (base.width * scale, base.height * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    scaled_box = tuple(int(v * scale) for v in box)
    shadow_box = (
        scaled_box[0] + int(3 * scale),
        scaled_box[1] + int(4 * scale),
        scaled_box[2] + int(3 * scale),
        scaled_box[3] + int(4 * scale),
    )
    DRAWERS[item.object_name](draw, shadow_box, (0, 0, 0, 95))
    shadow = layer.filter(ImageFilter.GaussianBlur(2.2 * scale))

    obj_layer = Image.new("RGBA", (base.width * scale, base.height * scale), (0, 0, 0, 0))
    obj_draw = ImageDraw.Draw(obj_layer)
    color = COLORS[item.color_name]
    jittered = tuple(max(0, min(255, c + rng.randint(-10, 10))) for c in color)
    DRAWERS[item.object_name](obj_draw, scaled_box, jittered)

    highlight = Image.new("RGBA", (base.width * scale, base.height * scale), (0, 0, 0, 0))
    hi_draw = ImageDraw.Draw(highlight)
    hi_draw.ellipse(
        [
            int((x0 + w * 0.18) * scale),
            int((y0 + h * 0.12) * scale),
            int((x0 + w * 0.55) * scale),
            int((y0 + h * 0.38) * scale),
        ],
        fill=(255, 255, 255, 35),
    )

    combined = Image.alpha_composite(shadow, obj_layer)
    combined = Image.alpha_composite(combined, highlight)
    combined = combined.resize(base.size, Image.Resampling.LANCZOS)
    base.alpha_composite(combined)


def make_realistic_grid(seed: int = 17, image_size: int = 288) -> tuple[Image.Image, dict]:
    rng = random.Random(seed)
    cell = image_size // 3
    image = Image.new("RGBA", (image_size, image_size), (239, 240, 236, 255))
    draw = ImageDraw.Draw(image)

    colors = [color for color in rng.sample(list(COLORS), 3) for _ in range(3)]
    rng.shuffle(colors)
    objects = rng.sample(OBJECTS, 9)
    items = [CellItem(color, obj) for color, obj in zip(colors, objects)]

    for row in range(3):
        for col in range(3):
            x0, y0 = col * cell, row * cell
            x1, y1 = x0 + cell, y0 + cell
            fill = tuple(max(0, min(255, v + rng.randint(-5, 5))) for v in (248, 248, 244))
            draw.rounded_rectangle([x0 + 3, y0 + 3, x1 - 3, y1 - 3], radius=7, fill=fill + (255,))
            draw.rounded_rectangle([x0 + 3, y0 + 3, x1 - 3, y1 - 3], radius=7, outline=(188, 193, 199, 255), width=2)

    for row in range(3):
        for col in range(3):
            item = items[row * 3 + col]
            x0, y0 = col * cell, row * cell
            x1, y1 = x0 + cell, y0 + cell
            pad = int(cell * 0.15)
            draw_soft_object(image, item, (x0 + pad, y0 + pad, x1 - pad, y1 - pad), rng)

    target_idx = rng.randrange(9)
    target = items[target_idx]
    prompt = f"请点击{target.color_name}{target.object_name}"
    rgb = add_paper_texture(image.convert("RGB"), rng)
    metadata = {
        "prompt": prompt,
        "target_index": target_idx,
        "items": [item.__dict__ for item in items],
        "image_size": image_size,
        "style": "semi_realistic",
    }
    return rgb, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one semi-realistic 3x3 CAPTCHA grid.")
    parser.add_argument("--output", default="outputs/realistic_grid.png")
    parser.add_argument("--metadata", default="outputs/realistic_grid_meta.json")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--image-size", type=int, default=288)
    args = parser.parse_args()

    image, metadata = make_realistic_grid(seed=args.seed, image_size=args.image_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    Path(args.metadata).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"image": str(output), **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

