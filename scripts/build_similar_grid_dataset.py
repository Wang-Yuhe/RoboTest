from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from src.multimodal_captcha.classes import CLASS_MAP


PROMPT_TEMPLATE = "\u8bf7\u70b9\u51fb{object_name}"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

CLASS_ALIASES = {
    "traffic light": "traffic_light",
    "traffic-light": "traffic_light",
    "trafficlight": "traffic_light",
    "traffic sign": "traffic_sign",
    "street light": "street_light",
    "fire hydrant": "fire_hydrant",
    "hydrant": "fire_hydrant",
    "motor bike": "motorcycle",
    "motorbike": "motorcycle",
    "mobile phone": "mobile_phone",
    "cell phone": "mobile_phone",
    "coffee cup": "coffee_cup",
    "wine glass": "wine_glass",
    "computer keyboard": "computer_keyboard",
    "microwave oven": "microwave_oven",
    "washing machine": "washing_machine",
    "tablet computer": "tablet_computer",
    "tennis racket": "tennis_racket",
    "sea turtle": "sea_turtle",
    "sea lion": "sea_lion",
    "hot dog": "hot_dog",
    "french fries": "french_fries",
}

NAME_OVERRIDES = {
    "bridge": "\u6865",
    "chimney": "\u70df\u56f1",
    "crosswalk": "\u4eba\u884c\u6a2a\u9053",
    "palm": "\u68d5\u6988\u6811",
    "stair": "\u697c\u68af",
    "other": "\u5176\u4ed6",
}

SIMILAR_GROUPS = {
    "vehicles": [
        "car",
        "bus",
        "truck",
        "taxi",
        "ambulance",
        "train",
        "motorcycle",
        "bicycle",
        "airplane",
        "boat",
        "helicopter",
    ],
    "street": [
        "traffic_light",
        "traffic_sign",
        "street_light",
        "fire_hydrant",
        "bridge",
        "crosswalk",
        "tower",
        "door",
        "window",
        "fountain",
        "palm",
        "stair",
    ],
    "animals": [
        "cat",
        "dog",
        "fox",
        "lion",
        "tiger",
        "bear",
        "horse",
        "cattle",
        "sheep",
        "zebra",
        "giraffe",
        "elephant",
        "camel",
    ],
    "small_animals": [
        "bird",
        "raven",
        "owl",
        "swan",
        "bee",
        "spider",
        "frog",
        "lizard",
        "snake",
        "rabbit",
        "monkey",
        "tortoise",
        "sea_turtle",
        "sea_lion",
    ],
    "food": [
        "apple",
        "banana",
        "orange",
        "strawberry",
        "tomato",
        "sandwich",
        "hamburger",
        "pizza",
        "cake",
        "cookie",
        "muffin",
        "bagel",
        "pretzel",
        "croissant",
        "pancake",
        "waffle",
        "cheese",
        "french_fries",
        "hot_dog",
        "cucumber",
        "radish",
        "pumpkin",
    ],
    "containers": [
        "bottle",
        "mug",
        "coffee_cup",
        "wine_glass",
        "kettle",
        "teapot",
        "sink",
        "tap",
        "box",
        "backpack",
        "handbag",
        "suitcase",
    ],
    "home_appliances": [
        "chair",
        "table",
        "refrigerator",
        "oven",
        "microwave_oven",
        "washing_machine",
        "television",
        "laptop",
        "mobile_phone",
        "computer_keyboard",
        "camera",
        "clock",
        "printer",
        "remote_control",
        "tablet_computer",
    ],
    "clothes": [
        "shirt",
        "skirt",
        "shorts",
        "dress",
        "coat",
        "jeans",
        "boot",
        "hat",
        "glove",
        "scarf",
        "sunglasses",
        "belt",
        "sock",
        "necklace",
    ],
    "round_objects": [
        "ball",
        "clock",
        "tire",
        "orange",
        "tomato",
        "pumpkin",
        "apple",
        "football",
        "balloon",
        "cake",
        "muffin",
    ],
    "tools_utensils": [
        "knife",
        "fork",
        "spoon",
        "toothbrush",
        "screwdriver",
        "wrench",
        "chopsticks",
        "scissors",
        "microphone",
        "guitar",
        "violin",
        "cello",
        "tennis_racket",
        "ladder",
    ],
}


def normalize_class_key(name: str) -> str:
    text = name.strip().replace("_", " ").replace("-", " ").lower()
    text = " ".join(text.split())
    if text in CLASS_ALIASES:
        return CLASS_ALIASES[text]
    return text.replace(" ", "_")


def default_source_roots(data_root: Path) -> list[Path]:
    candidates = [
        data_root / "photo_objects",
        data_root / "google_recaptcha_v2" / "data" / "images",
    ]
    return [path for path in candidates if path.exists()]


def load_name_map(source_roots: list[Path]) -> dict[str, str]:
    names = dict(CLASS_MAP)
    names.update(NAME_OVERRIDES)
    for root in source_roots:
        metadata_path = root / "class_names.json"
        if not metadata_path.exists():
            continue
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        for folder, info in data.items():
            key = normalize_class_key(folder)
            if key in names:
                continue
            if isinstance(info, dict):
                names[key] = info.get("object_name") or info.get("openimages_label") or key
            else:
                names[key] = str(info)
    return names


def list_images(path: Path) -> list[Path]:
    return sorted(
        p
        for p in path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and not p.name.startswith("._")
    )


def collect_image_pools(source_roots: list[Path], min_images: int) -> dict[str, list[Path]]:
    pools: dict[str, list[Path]] = defaultdict(list)
    for root in source_roots:
        if not root.exists():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            images = list_images(child)
            if images:
                pools[normalize_class_key(child.name)].extend(images)
    return {key: sorted(paths) for key, paths in pools.items() if len(paths) >= min_images}


def eligible_groups(
    pools: dict[str, list[Path]],
    requested_groups: set[str] | None,
) -> dict[str, list[str]]:
    groups = {}
    for group_name, class_keys in SIMILAR_GROUPS.items():
        if requested_groups is not None and group_name not in requested_groups:
            continue
        available = [key for key in class_keys if key in pools]
        if len(available) >= 9:
            groups[group_name] = available
    return groups


def fit_cell(image: Image.Image, rng: random.Random, size: int, augment: bool) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    if augment:
        scale = rng.uniform(0.68, 1.0)
        crop_w = max(8, int(w * scale))
        crop_h = max(8, int(h * scale))
        left = rng.randint(0, max(0, w - crop_w))
        top = rng.randint(0, max(0, h - crop_h))
        image = image.crop((left, top, left + crop_w, top + crop_h))
    else:
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        image = image.crop((left, top, left + side, top + side))

    image = image.resize((size, size), Image.Resampling.LANCZOS)
    if not augment:
        return image

    if rng.random() < 0.65:
        image = image.rotate(
            rng.uniform(-8.0, 8.0),
            resample=Image.Resampling.BICUBIC,
            fillcolor=(238, 240, 243),
        )
    if rng.random() < 0.85:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.76, 1.24))
    if rng.random() < 0.85:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.76, 1.28))
    if rng.random() < 0.75:
        image = ImageEnhance.Color(image).enhance(rng.uniform(0.72, 1.32))
    if rng.random() < 0.28:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.9)))

    arr = np.asarray(image, dtype=np.float32)
    if rng.random() < 0.5:
        arr = np.clip(arr + np.random.normal(0, rng.uniform(2.0, 8.0), arr.shape), 0, 255)
    image = Image.fromarray(arr.astype(np.uint8), "RGB")

    if rng.random() < 0.45:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=rng.randint(62, 92))
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
    return image


def choose_classes(
    groups: dict[str, list[str]],
    rng: random.Random,
    target_counts: dict[str, int],
) -> tuple[str, list[str], int]:
    group_name = rng.choice(sorted(groups))
    available = groups[group_name]
    min_count = min(target_counts[key] for key in available)
    least_used = [key for key in available if target_counts[key] == min_count]
    target_key = rng.choice(least_used)
    distractors = rng.sample([key for key in available if key != target_key], 8)
    class_keys = distractors + [target_key]
    rng.shuffle(class_keys)
    return group_name, class_keys, class_keys.index(target_key)


def make_grid(
    pools: dict[str, list[Path]],
    names: dict[str, str],
    groups: dict[str, list[str]],
    rng: random.Random,
    target_counts: dict[str, int],
    image_size: int,
    augment: bool,
) -> tuple[Image.Image, dict]:
    cell = image_size // 3
    grid_size = cell * 3
    group_name, class_keys, target_index = choose_classes(groups, rng, target_counts)
    grid = Image.new("RGB", (grid_size, grid_size), (238, 240, 243))
    items = []

    for idx, class_key in enumerate(class_keys):
        source = rng.choice(pools[class_key])
        with Image.open(source) as raw:
            crop = fit_cell(raw, rng, cell, augment=augment)
        row, col = divmod(idx, 3)
        grid.paste(crop, (col * cell, row * cell))
        object_name = names.get(class_key, class_key.replace("_", " "))
        items.append(
            {
                "class_key": class_key,
                "object_name": object_name,
                "source": str(source),
                "similar_group": group_name,
            }
        )

    if augment:
        arr = np.asarray(grid, dtype=np.float32)
        if rng.random() < 0.35:
            arr = np.clip(arr + np.random.normal(0, rng.uniform(1.0, 4.0), arr.shape), 0, 255)
            grid = Image.fromarray(arr.astype(np.uint8), "RGB")

    target_counts[class_keys[target_index]] += 1
    target = items[target_index]
    prompt = PROMPT_TEMPLATE.format(object_name=target["object_name"])
    return grid, {
        "prompt": prompt,
        "target_index": target_index,
        "click": [target_index % 3 * cell + cell // 2, target_index // 3 * cell + cell // 2],
        "items": items,
        "image_size": grid_size,
        "difficulty": "similar_hard",
        "similar_group": group_name,
        "hard_augment": augment,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 3x3 CAPTCHA grids from visually similar but different classes."
    )
    parser.add_argument("--data-root", default="data", help="Root containing known datasets.")
    parser.add_argument(
        "--source-roots",
        nargs="*",
        default=None,
        help="Explicit class-folder roots. Defaults to data/photo_objects and data/google_recaptcha_v2/data/images.",
    )
    parser.add_argument("--output-dir", default="data/similar_photo_grid")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--min-images-per-class", type=int, default=8)
    parser.add_argument(
        "--groups",
        default="",
        help=f"Comma-separated similar groups to use. Available: {','.join(sorted(SIMILAR_GROUPS))}",
    )
    parser.add_argument("--no-augment", action="store_true", help="Disable crop/rotation/noise/JPEG augmentation.")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    data_root = Path(args.data_root)
    source_roots = [Path(path) for path in args.source_roots] if args.source_roots else default_source_roots(data_root)
    if not source_roots:
        raise SystemExit("No source roots found. Provide --source-roots or check --data-root.")

    requested_groups = {item.strip() for item in args.groups.split(",") if item.strip()} or None
    names = load_name_map(source_roots)
    pools = collect_image_pools(source_roots, args.min_images_per_class)
    groups = eligible_groups(pools, requested_groups)
    if not groups:
        available = ", ".join(f"{name}:{len(keys)}" for name, keys in sorted(eligible_groups(pools, None).items()))
        raise SystemExit(
            "No similar group has at least 9 available classes. "
            f"Eligible groups without filter: {available or 'none'}"
        )

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    target_counts = {key: 0 for keys in groups.values() for key in keys}
    records = []
    start = time.time()

    print(
        json.dumps(
            {
                "status": "start",
                "output_dir": str(output_dir),
                "source_roots": [str(path) for path in source_roots],
                "num_samples": args.num_samples,
                "classes": len(pools),
                "groups": {name: keys for name, keys in sorted(groups.items())},
                "augment": not args.no_augment,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for idx in range(args.num_samples):
        image, metadata = make_grid(
            pools=pools,
            names=names,
            groups=groups,
            rng=rng,
            target_counts=target_counts,
            image_size=args.image_size,
            augment=not args.no_augment,
        )
        image_name = f"similar_grid_{idx:05d}.jpg"
        image.save(image_dir / image_name, quality=90)
        records.append({"image": f"images/{image_name}", **metadata})

        done = idx + 1
        if args.progress_every > 0 and (done == 1 or done % args.progress_every == 0 or done == args.num_samples):
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (args.num_samples - done) / rate if rate > 0 else 0.0
            print(
                json.dumps(
                    {
                        "status": "progress",
                        "done": done,
                        "total": args.num_samples,
                        "percent": round(done / max(args.num_samples, 1) * 100, 2),
                        "samples_per_sec": round(rate, 2),
                        "eta_sec": round(eta, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    manifest = output_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "status": "done",
        "output_dir": str(output_dir),
        "manifest": str(manifest),
        "samples": len(records),
        "groups": {name: keys for name, keys in sorted(groups.items())},
        "target_counts": dict(sorted(target_counts.items())),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
