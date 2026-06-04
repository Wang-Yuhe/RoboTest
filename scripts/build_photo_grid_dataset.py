from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageOps


CLASS_MAP = {
    "fire_hydrant": "消防栓",
    "bicycle": "自行车",
    "car": "汽车",
    "traffic_light": "信号灯",
    "tree": "树",
    "house": "房子",
    "flag": "旗帜",
    "street_light": "路灯",
    "star": "星星",
    "flower": "花",
}

COLORLESS_PROMPT = "请点击{object_name}"


def list_images(path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in exts)


def square_crop(image: Image.Image, size: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((size, size), Image.Resampling.LANCZOS)


def make_grid(
    class_dirs: dict[str, list[Path]],
    rng: random.Random,
    image_size: int,
) -> tuple[Image.Image, dict]:
    available_classes = list(class_dirs)
    if len(available_classes) >= 9:
        classes = rng.sample(available_classes, 9)
    else:
        classes = [rng.choice(available_classes) for _ in range(9)]
        if len(available_classes) > 1:
            while len(set(classes)) < min(3, len(available_classes)):
                classes = [rng.choice(available_classes) for _ in range(9)]
    cell = image_size // 3
    grid = Image.new("RGB", (image_size, image_size), (238, 240, 243))
    items = []

    for idx, class_key in enumerate(classes):
        src = rng.choice(class_dirs[class_key])
        crop = square_crop(Image.open(src), cell)
        row, col = divmod(idx, 3)
        x, y = col * cell, row * cell
        grid.paste(crop, (x, y))
        items.append({"class_key": class_key, "object_name": CLASS_MAP[class_key], "source": str(src)})

    counts = {class_key: classes.count(class_key) for class_key in set(classes)}
    unique_indices = [idx for idx, class_key in enumerate(classes) if counts[class_key] == 1]
    target_idx = rng.choice(unique_indices) if unique_indices else rng.randrange(9)
    target = items[target_idx]
    prompt = COLORLESS_PROMPT.format(object_name=target["object_name"])

    metadata = {
        "prompt": prompt,
        "target_index": target_idx,
        "click": [target_idx % 3 * cell + cell // 2, target_idx // 3 * cell + cell // 2],
        "items": items,
        "image_size": image_size,
        "difficulty": "photo_hard",
    }
    return grid, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 3x3 photo-grid CAPTCHA data from local class folders.")
    parser.add_argument("--photo-root", required=True, help="Directory containing class subfolders, e.g. car/, bicycle/.")
    parser.add_argument("--output-dir", default="data/photo_grid")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--image-size", type=int, default=288)
    args = parser.parse_args()

    photo_root = Path(args.photo_root)
    class_dirs = {}
    for class_key in CLASS_MAP:
        images = list_images(photo_root / class_key)
        if images:
            class_dirs[class_key] = images

    if len(class_dirs) < 2:
        available = ", ".join(sorted(class_dirs)) or "none"
        needed = ", ".join(CLASS_MAP)
        raise SystemExit(
            "Need at least 2 class folders with images.\n"
            f"Available: {available}\n"
            f"Supported folder names: {needed}"
        )
    if len(class_dirs) < 9:
        print(
            f"Warning: only {len(class_dirs)} classes have images. "
            "Grids will reuse classes across cells; prompts remain class-based."
        )

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    records = []

    for idx in range(args.num_samples):
        image, metadata = make_grid(class_dirs, rng, args.image_size)
        image_name = f"photo_grid_{idx:05d}.jpg"
        image.save(image_dir / image_name, quality=92)
        records.append({"image": f"images/{image_name}", **metadata})

    manifest = output_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps({"output_dir": str(output_dir), "samples": len(records), "classes": sorted(class_dirs)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
