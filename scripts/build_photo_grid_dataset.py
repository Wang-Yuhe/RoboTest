from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from src.multimodal_captcha.classes import CLASS_MAP

COLORLESS_PROMPT = "请点击{object_name}"


def load_class_map(photo_root: Path) -> dict[str, str]:
    mapping = dict(CLASS_MAP)
    metadata_path = photo_root / "class_names.json"
    if metadata_path.exists():
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        for folder, info in data.items():
            if isinstance(info, dict):
                mapping[folder] = info.get("object_name") or info.get("openimages_label") or folder
            else:
                mapping[folder] = str(info)
    for child in photo_root.iterdir() if photo_root.exists() else []:
        if child.is_dir() and child.name not in mapping:
            mapping[child.name] = child.name.replace("_", " ")
    return mapping


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


def hard_augment_cell(image: Image.Image, rng: random.Random, size: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    scale = rng.uniform(0.72, 1.0)
    crop_w = max(8, int(w * scale))
    crop_h = max(8, int(h * scale))
    left = rng.randint(0, max(0, w - crop_w))
    top = rng.randint(0, max(0, h - crop_h))
    image = image.crop((left, top, left + crop_w, top + crop_h)).resize((size, size), Image.Resampling.LANCZOS)
    if rng.random() < 0.6:
        image = image.rotate(rng.uniform(-8.0, 8.0), resample=Image.Resampling.BICUBIC, fillcolor=(238, 240, 243))
    if rng.random() < 0.85:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.78, 1.22))
    if rng.random() < 0.85:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.78, 1.25))
    if rng.random() < 0.75:
        image = ImageEnhance.Color(image).enhance(rng.uniform(0.75, 1.30))
    if rng.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.9)))
    arr = np.asarray(image, dtype=np.float32)
    if rng.random() < 0.45:
        arr = np.clip(arr + np.random.normal(0, rng.uniform(2.0, 7.0), arr.shape), 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def make_grid(
    class_dirs: dict[str, list[Path]],
    rng: random.Random,
    image_size: int,
    hard_augment: bool = False,
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
        crop = hard_augment_cell(Image.open(src), rng, cell) if hard_augment else square_crop(Image.open(src), cell)
        row, col = divmod(idx, 3)
        x, y = col * cell, row * cell
        grid.paste(crop, (x, y))
        object_name = CLASS_MAP.get(class_key, class_key.replace("_", " "))
        items.append({"class_key": class_key, "object_name": object_name, "source": str(src)})

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
        "hard_augment": hard_augment,
    }
    return grid, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 3x3 photo-grid CAPTCHA data from local class folders.")
    parser.add_argument("--photo-root", required=True, help="Directory containing class subfolders, e.g. car/, bicycle/.")
    parser.add_argument("--output-dir", default="data/photo_grid")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--min-images-per-class", type=int, default=8)
    parser.add_argument("--max-classes", type=int, default=0, help="Limit classes after filtering. 0 means use all.")
    parser.add_argument("--hard-augment", action="store_true", help="Apply stronger photo perturbations while building each grid.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N generated samples. 0 disables progress.")
    args = parser.parse_args()

    photo_root = Path(args.photo_root)
    class_map = load_class_map(photo_root)
    class_dirs = {}
    for class_key in class_map:
        images = list_images(photo_root / class_key)
        if len(images) >= args.min_images_per_class:
            class_dirs[class_key] = images
    if args.max_classes > 0 and len(class_dirs) > args.max_classes:
        ranked = sorted(class_dirs.items(), key=lambda kv: len(kv[1]), reverse=True)[: args.max_classes]
        class_dirs = dict(ranked)

    if len(class_dirs) < 2:
        available = ", ".join(sorted(class_dirs)) or "none"
        needed = ", ".join(sorted(class_map))
        raise SystemExit(
            "Need at least 2 class folders with images.\n"
            f"Available: {available}\n"
            f"Known folder names: {needed}"
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
    start_time = time.time()
    print(
        json.dumps(
            {
                "status": "start",
                "output_dir": str(output_dir),
                "target_samples": args.num_samples,
                "usable_classes": len(class_dirs),
                "hard_augment": args.hard_augment,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for idx in range(args.num_samples):
        image, metadata = make_grid(class_dirs, rng, args.image_size, hard_augment=args.hard_augment)
        for item in metadata["items"]:
            item["object_name"] = class_map.get(item["class_key"], item["object_name"])
        target = metadata["items"][metadata["target_index"]]
        metadata["prompt"] = COLORLESS_PROMPT.format(object_name=target["object_name"])
        image_name = f"photo_grid_{idx:05d}.jpg"
        image.save(image_dir / image_name, quality=92)
        records.append({"image": f"images/{image_name}", **metadata})
        done = idx + 1
        if args.progress_every > 0 and (done == 1 or done % args.progress_every == 0 or done == args.num_samples):
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0.0
            remaining = (args.num_samples - done) / rate if rate > 0 else 0.0
            print(
                json.dumps(
                    {
                        "status": "progress",
                        "done": done,
                        "total": args.num_samples,
                        "percent": round(done / args.num_samples * 100, 2),
                        "samples_per_sec": round(rate, 2),
                        "eta_sec": round(remaining, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    manifest = output_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps({"output_dir": str(output_dir), "samples": len(records), "classes": sorted(class_dirs)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
