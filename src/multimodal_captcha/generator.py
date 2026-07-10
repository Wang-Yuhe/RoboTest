from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.multimodal_captcha.action_sequence import target_indices_to_actions


COLORS = {
    "红色": (220, 54, 46),
    "蓝色": (54, 112, 220),
    "绿色": (51, 160, 88),
    "黄色": (232, 184, 42),
    "紫色": (139, 84, 201),
    "橙色": (234, 125, 42),
    "粉色": (222, 93, 156),
    "青色": (40, 170, 180),
    "黑色": (55, 59, 68),
}

OBJECTS = ["消防栓", "汽车", "自行车", "路灯", "树", "信号灯", "房子", "星星", "旗帜"]


@dataclass(frozen=True)
class CellItem:
    color_name: str
    object_name: str


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _draw_hydrant(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx = (x0 + x1) // 2
    draw.rounded_rectangle([cx - w * 0.15, y0 + h * 0.28, cx + w * 0.15, y1 - h * 0.15], radius=6, fill=color)
    draw.ellipse([cx - w * 0.18, y0 + h * 0.12, cx + w * 0.18, y0 + h * 0.42], fill=color)
    draw.rounded_rectangle([x0 + w * 0.18, y0 + h * 0.38, x1 - w * 0.18, y0 + h * 0.53], radius=6, fill=color)
    draw.rectangle([cx - w * 0.28, y1 - h * 0.15, cx + w * 0.28, y1 - h * 0.06], fill=color)


def _draw_car(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    draw.rounded_rectangle([x0 + w * 0.12, y0 + h * 0.48, x1 - w * 0.12, y0 + h * 0.74], radius=8, fill=color)
    draw.polygon([(x0 + w * 0.30, y0 + h * 0.48), (x0 + w * 0.43, y0 + h * 0.28), (x0 + w * 0.68, y0 + h * 0.28), (x0 + w * 0.82, y0 + h * 0.48)], fill=color)
    for cx in (x0 + w * 0.30, x1 - w * 0.30):
        draw.ellipse([cx - w * 0.09, y0 + h * 0.68, cx + w * 0.09, y0 + h * 0.86], fill=(35, 35, 35))


def _draw_bike(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    r = w * 0.13
    p1 = (x0 + w * 0.28, y0 + h * 0.72)
    p2 = (x1 - w * 0.28, y0 + h * 0.72)
    seat = (x0 + w * 0.48, y0 + h * 0.48)
    bar = (x0 + w * 0.66, y0 + h * 0.46)
    for p in (p1, p2):
        draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], outline=color, width=5)
    draw.line([p1, seat, p2, (x0 + w * 0.40, y0 + h * 0.70), p1], fill=color, width=5)
    draw.line([seat, (x0 + w * 0.50, y0 + h * 0.38)], fill=color, width=5)
    draw.line([p2, bar, (x0 + w * 0.77, y0 + h * 0.42)], fill=color, width=5)


def _draw_lamp(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx = (x0 + x1) // 2
    draw.line([(cx, y0 + h * 0.20), (cx, y1 - h * 0.10)], fill=color, width=7)
    draw.arc([cx - w * 0.05, y0 + h * 0.18, cx + w * 0.42, y0 + h * 0.55], 180, 270, fill=color, width=7)
    draw.ellipse([cx + w * 0.30, y0 + h * 0.42, cx + w * 0.52, y0 + h * 0.58], fill=color)
    draw.rectangle([cx - w * 0.18, y1 - h * 0.12, cx + w * 0.18, y1 - h * 0.06], fill=color)


def _draw_tree(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx = (x0 + x1) // 2
    draw.rectangle([cx - w * 0.06, y0 + h * 0.52, cx + w * 0.06, y1 - h * 0.10], fill=(119, 84, 53))
    draw.ellipse([x0 + w * 0.20, y0 + h * 0.16, x1 - w * 0.20, y0 + h * 0.62], fill=color)
    draw.ellipse([x0 + w * 0.10, y0 + h * 0.34, x0 + w * 0.55, y0 + h * 0.75], fill=color)
    draw.ellipse([x0 + w * 0.45, y0 + h * 0.34, x1 - w * 0.10, y0 + h * 0.75], fill=color)


def _draw_traffic_light(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx = (x0 + x1) // 2
    draw.rounded_rectangle([cx - w * 0.16, y0 + h * 0.12, cx + w * 0.16, y0 + h * 0.76], radius=8, fill=(42, 42, 42))
    for i, light in enumerate([(220, 54, 46), (232, 184, 42), (51, 160, 88)]):
        y = y0 + h * (0.25 + i * 0.18)
        fill = color if i == 0 else light
        draw.ellipse([cx - w * 0.08, y - w * 0.08, cx + w * 0.08, y + w * 0.08], fill=fill)
    draw.line([(cx, y0 + h * 0.76), (cx, y1 - h * 0.06)], fill=(42, 42, 42), width=5)


def _draw_house(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    draw.rectangle([x0 + w * 0.22, y0 + h * 0.45, x1 - w * 0.22, y1 - h * 0.12], fill=color)
    draw.polygon(
        [
            (x0 + w * 0.14, y0 + h * 0.48),
            ((x0 + x1) / 2, y0 + h * 0.14),
            (x1 - w * 0.14, y0 + h * 0.48),
        ],
        fill=color,
    )
    draw.rectangle([x0 + w * 0.45, y0 + h * 0.62, x0 + w * 0.57, y1 - h * 0.12], fill=(245, 246, 248))


def _draw_star(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    radius = min(x1 - x0, y1 - y0) * 0.38
    points = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        r = radius if i % 2 == 0 else radius * 0.45
        points.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))
    draw.polygon(points, fill=color)


def _draw_flag(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    pole_x = x0 + w * 0.30
    draw.line([(pole_x, y0 + h * 0.12), (pole_x, y1 - h * 0.10)], fill=(42, 42, 42), width=5)
    draw.polygon(
        [
            (pole_x, y0 + h * 0.16),
            (x1 - w * 0.18, y0 + h * 0.25),
            (pole_x, y0 + h * 0.46),
        ],
        fill=color,
    )
    draw.rectangle([pole_x - w * 0.18, y1 - h * 0.12, pole_x + w * 0.18, y1 - h * 0.06], fill=(42, 42, 42))


DRAWERS = {
    "消防栓": _draw_hydrant,
    "汽车": _draw_car,
    "自行车": _draw_bike,
    "路灯": _draw_lamp,
    "树": _draw_tree,
    "信号灯": _draw_traffic_light,
    "房子": _draw_house,
    "星星": _draw_star,
    "旗帜": _draw_flag,
}


def random_item(rng: random.Random) -> CellItem:
    return CellItem(rng.choice(list(COLORS)), rng.choice(OBJECTS))


def make_items(rng: random.Random, difficulty: str) -> list[CellItem]:
    if difficulty == "easy":
        color_names = rng.sample(list(COLORS), 9)
        return [CellItem(color_name, rng.choice(OBJECTS)) for color_name in color_names]
    if difficulty == "medium":
        color_names = [color for color in rng.sample(list(COLORS), 3) for _ in range(3)]
        rng.shuffle(color_names)
        object_names = rng.sample(OBJECTS, 9)
        return [CellItem(color_name, object_name) for color_name, object_name in zip(color_names, object_names)]
    if difficulty == "hard":
        object_names = rng.sample(OBJECTS, 9)
        return [CellItem(rng.choice(list(COLORS)), object_name) for object_name in object_names]
    raise ValueError(f"Unknown difficulty: {difficulty}")


def make_sample(
    rng: random.Random,
    image_size: int = 288,
    debug_labels: bool = False,
    difficulty: str = "medium",
) -> tuple[Image.Image, dict]:
    grid = 3
    cell = image_size // grid
    image = Image.new("RGB", (image_size, image_size), (245, 246, 248))
    draw = ImageDraw.Draw(image)
    font = _font(14)

    items = make_items(rng, difficulty)
    for row in range(grid):
        for col in range(grid):
            item = items[row * grid + col]
            x0, y0 = col * cell, row * cell
            x1, y1 = x0 + cell, y0 + cell
            draw.rectangle([x0, y0, x1, y1], fill=(250, 251, 252), outline=(205, 210, 218), width=2)
            pad = int(cell * 0.12)
            DRAWERS[item.object_name](draw, (x0 + pad, y0 + pad, x1 - pad, y1 - pad), COLORS[item.color_name])
            if debug_labels:
                label = f"{item.color_name}{item.object_name}"
                draw.text((x0 + 6, y1 - 22), label, fill=(72, 78, 88), font=font)

    target_idx = rng.randrange(9)
    target = items[target_idx]
    row, col = divmod(target_idx, grid)
    click = (col * cell + cell // 2, row * cell + cell // 2)
    prompt = f"请点击{target.object_name}" if difficulty == "hard" else f"请点击{target.color_name}{target.object_name}"
    metadata = {
        "prompt": prompt,
        "target_index": target_idx,
        "click": click,
        "items": [item.__dict__ for item in items],
        "image_size": image_size,
        "difficulty": difficulty,
    }
    return image, metadata


def generate_dataset(
    output_dir: str | Path,
    num_samples: int,
    seed: int = 7,
    image_size: int = 288,
    debug_labels: bool = False,
    difficulty: str = "medium",
) -> Path:
    output = Path(output_dir)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records = []

    for idx in range(num_samples):
        image, metadata = make_sample(rng, image_size=image_size, debug_labels=debug_labels, difficulty=difficulty)
        image_name = f"sample_{idx:05d}.png"
        image.save(image_dir / image_name)
        records.append({"image": f"images/{image_name}", **metadata})

    manifest = output / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return manifest


def make_action_sample(
    rng: random.Random,
    image_size: int = 288,
    debug_labels: bool = False,
    min_targets: int = 2,
    max_targets: int = 4,
) -> tuple[Image.Image, dict]:
    grid = 3
    cell = image_size // grid
    image = Image.new("RGB", (image_size, image_size), (245, 246, 248))
    draw = ImageDraw.Draw(image)
    font = _font(14)

    target_object = rng.choice(OBJECTS)
    target_count = rng.randint(min_targets, max_targets)
    target_indices = sorted(rng.sample(range(9), target_count))
    distractors = [name for name in OBJECTS if name != target_object]
    rng.shuffle(distractors)

    items = []
    distractor_idx = 0
    for idx in range(9):
        if idx in target_indices:
            items.append(CellItem(rng.choice(list(COLORS)), target_object))
        else:
            object_name = distractors[distractor_idx % len(distractors)]
            distractor_idx += 1
            items.append(CellItem(rng.choice(list(COLORS)), object_name))

    for row in range(grid):
        for col in range(grid):
            item = items[row * grid + col]
            x0, y0 = col * cell, row * cell
            x1, y1 = x0 + cell, y0 + cell
            draw.rectangle([x0, y0, x1, y1], fill=(250, 251, 252), outline=(205, 210, 218), width=2)
            pad = int(cell * 0.12)
            DRAWERS[item.object_name](draw, (x0 + pad, y0 + pad, x1 - pad, y1 - pad), COLORS[item.color_name])
            if debug_labels:
                label = f"{item.color_name}{item.object_name}"
                draw.text((x0 + 6, y1 - 22), label, fill=(72, 78, 88), font=font)

    actions = target_indices_to_actions(target_indices)
    prompt = f"请点击所有{target_object}"
    metadata = {
        "prompt": prompt,
        "target_object": target_object,
        "target_indices": target_indices,
        "target_index": target_indices[0],
        "click": None,
        "actions": actions,
        "items": [item.__dict__ for item in items],
        "image_size": image_size,
        "difficulty": "click_all",
    }
    return image, metadata


def generate_action_dataset(
    output_dir: str | Path,
    num_samples: int,
    seed: int = 7,
    image_size: int = 288,
    debug_labels: bool = False,
    min_targets: int = 2,
    max_targets: int = 4,
) -> Path:
    output = Path(output_dir)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records = []

    for idx in range(num_samples):
        image, metadata = make_action_sample(
            rng,
            image_size=image_size,
            debug_labels=debug_labels,
            min_targets=min_targets,
            max_targets=max_targets,
        )
        image_name = f"action_{idx:05d}.png"
        image.save(image_dir / image_name)
        records.append({"image": f"images/{image_name}", **metadata})

    manifest = output / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return manifest


def draw_prediction_overlay(image: Image.Image, target_index: int, predicted_index: int | None = None) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    cell = out.size[0] // 3
    for idx, color in [(target_index, (51, 160, 88)), (predicted_index, (220, 54, 46))]:
        if idx is None:
            continue
        row, col = divmod(idx, 3)
        x0, y0 = col * cell + 4, row * cell + 4
        x1, y1 = x0 + cell - 8, y0 + cell - 8
        draw.rectangle([x0, y0, x1, y1], outline=color, width=5)
    return out
