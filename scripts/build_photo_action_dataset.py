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

from src.multimodal_captcha.action_sequence import target_indices_to_actions
from src.multimodal_captcha.classes import CLASS_MAP


PROMPT = "请点击所有{object_name}"
SPLITS = ("train", "val", "test")


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


def parse_csv_values(text: str) -> set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


def parse_ordered_csv_values(text: str) -> list[str]:
    values = []
    seen = set()
    for item in text.split(","):
        value = item.strip()
        if value and value not in seen:
            values.append(value)
            seen.add(value)
    return values


def load_fixed_classes(include_classes: str, class_list: str | Path | None) -> list[str]:
    values = parse_ordered_csv_values(include_classes)
    if class_list:
        path = Path(class_list)
        if not path.exists():
            raise SystemExit(f"Class list file does not exist: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            if value not in values:
                values.append(value)
    return values


def source_keys(path: Path) -> set[str]:
    keys = {str(path), path.as_posix(), path.name}
    try:
        resolved = path.resolve()
        keys.update({str(resolved), resolved.as_posix()})
    except OSError:
        pass
    return keys


def load_blacklist_sources(path: str | Path | None) -> set[str]:
    if not path:
        return set()
    blacklist_path = Path(path)
    if not blacklist_path.exists():
        raise SystemExit(f"Blacklist file does not exist: {blacklist_path}")
    blocked = set()
    for line in blacklist_path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        blocked.add(value)
        blocked.add(Path(value).as_posix())
        try:
            blocked.add(str(Path(value).resolve()))
            blocked.add(Path(value).resolve().as_posix())
        except OSError:
            pass
    return blocked


def load_bad_source_labels(path: str | Path | None) -> set[str]:
    if not path:
        return set()
    labels_path = Path(path)
    if not labels_path.exists():
        raise SystemExit(f"Bad source labels file does not exist: {labels_path}")
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    raw_sources = data.get("sources") if isinstance(data, dict) else data
    if not isinstance(raw_sources, list):
        raise SystemExit(f"Bad source labels must be a list or contain a 'sources' list: {labels_path}")
    blocked = set()
    for entry in raw_sources:
        if isinstance(entry, str):
            source = entry
            exclude = True
        elif isinstance(entry, dict):
            source = entry.get("source")
            exclude = entry.get("exclude", True)
        else:
            raise SystemExit(f"Bad source label entries must be strings or objects: {labels_path}")
        if not source or not exclude:
            continue
        blocked.update(source_keys(Path(str(source))))
    return blocked


def count_bad_source_matches(class_images: dict[str, list[Path]], bad_source_keys: set[str]) -> int:
    if not bad_source_keys:
        return 0
    return sum(1 for images in class_images.values() for image in images if source_keys(image) & bad_source_keys)


def source_passes_quality(
    path: Path,
    min_source_area: int,
    max_source_aspect_ratio: float,
    blacklist_sources: set[str],
) -> bool:
    if blacklist_sources and source_keys(path) & blacklist_sources:
        return False
    try:
        with Image.open(path) as image:
            width, height = image.size
    except OSError:
        return False
    if min_source_area > 0 and width * height < min_source_area:
        return False
    if max_source_aspect_ratio > 0:
        short = max(min(width, height), 1)
        if max(width, height) / short > max_source_aspect_ratio:
            return False
    return True


def filter_source_images(
    images: list[Path],
    min_source_area: int,
    max_source_aspect_ratio: float,
    blacklist_sources: set[str],
) -> list[Path]:
    return [
        image
        for image in images
        if source_passes_quality(
            image,
            min_source_area=min_source_area,
            max_source_aspect_ratio=max_source_aspect_ratio,
            blacklist_sources=blacklist_sources,
        )
    ]


def save_source_pool_manifest(
    class_images: dict[str, list[Path]],
    path: str | Path,
    class_map: dict[str, str],
    args: argparse.Namespace,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "classes": [
            {
                "class_key": class_key,
                "object_name": class_map.get(class_key, class_key.replace("_", " ")),
                "sources": [str(source) for source in sources],
            }
            for class_key, sources in sorted(class_images.items())
        ],
        "config": {
            "seed": args.seed,
            "min_images_per_class": args.min_images_per_class,
            "include_classes": args.include_classes,
            "class_list": str(args.class_list) if args.class_list else None,
            "exclude_classes": args.exclude_classes,
            "max_classes": args.max_classes,
        },
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_source_pool_manifest(path: str | Path, photo_root: Path, class_map: dict[str, str]) -> dict[str, list[Path]]:
    pool_path = Path(path)
    if not pool_path.exists():
        raise SystemExit(f"Source pool manifest does not exist: {pool_path}")
    data = json.loads(pool_path.read_text(encoding="utf-8"))
    raw_classes = data.get("classes")
    if not isinstance(raw_classes, list):
        raise SystemExit(f"Source pool manifest must contain a 'classes' list: {pool_path}")
    class_images: dict[str, list[Path]] = {}
    for entry in raw_classes:
        if not isinstance(entry, dict):
            raise SystemExit(f"Source pool class entries must be objects: {pool_path}")
        class_key = entry.get("class_key")
        raw_sources = entry.get("sources")
        if not class_key or not isinstance(raw_sources, list):
            raise SystemExit(f"Source pool entries need class_key and sources: {pool_path}")
        if entry.get("object_name"):
            class_map[str(class_key)] = str(entry["object_name"])
        sources = []
        missing = []
        for raw_source in raw_sources:
            source = resolve_manifest_source(Path(str(raw_source)), photo_root)
            if source.exists():
                sources.append(source)
            else:
                missing.append(str(raw_source))
        if missing:
            raise SystemExit(
                f"Source pool manifest references missing files for {class_key}: "
                + ", ".join(missing[:5])
            )
        class_images[str(class_key)] = sources
    return class_images


def resolve_manifest_source(source: Path, photo_root: Path) -> Path:
    if source.is_absolute() or source.exists():
        return source
    rooted = photo_root / source
    if rooted.exists():
        return rooted
    return source


def square_crop(image: Image.Image, size: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((size, size), Image.Resampling.LANCZOS)


def hard_augment_cell(
    image: Image.Image,
    rng: random.Random,
    np_rng: np.random.Generator,
    size: int,
) -> Image.Image:
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
        arr = np.clip(arr + np_rng.normal(0, rng.uniform(2.0, 7.0), arr.shape), 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def split_class_images(
    class_images: dict[str, list[Path]],
    rng: random.Random,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, dict[str, list[Path]]]:
    split_images: dict[str, dict[str, list[Path]]] = {split: {} for split in SPLITS}
    for class_key, images in class_images.items():
        shuffled = list(images)
        rng.shuffle(shuffled)
        total = len(shuffled)
        test_count = max(1, int(total * test_ratio))
        val_count = max(1, int(total * val_ratio))
        train_count = total - val_count - test_count
        if train_count < 1:
            train_count = 1
            if val_count > test_count:
                val_count -= 1
            else:
                test_count -= 1
        split_images["train"][class_key] = shuffled[:train_count]
        split_images["val"][class_key] = shuffled[train_count : train_count + val_count]
        split_images["test"][class_key] = shuffled[train_count + val_count :]
    return split_images


def save_source_split_plan(
    split_images: dict[str, dict[str, list[Path]]],
    path: str | Path,
    args: argparse.Namespace,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "splits": {
            split: {
                class_key: [str(source) for source in sources]
                for class_key, sources in sorted(classes.items())
            }
            for split, classes in split_images.items()
        },
        "config": {
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "min_images_per_class": args.min_images_per_class,
            "min_source_area": args.min_source_area,
            "max_source_aspect_ratio": args.max_source_aspect_ratio,
            "blacklist_sources": str(args.blacklist_sources) if args.blacklist_sources else None,
        },
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_source_split_plan(
    path: str | Path,
    available_class_images: dict[str, list[Path]],
    missing_source_policy: str = "error",
) -> dict[str, dict[str, list[Path]]]:
    plan_path = Path(path)
    if not plan_path.exists():
        raise SystemExit(f"Source split plan does not exist: {plan_path}")
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    raw_splits = data.get("splits")
    if not isinstance(raw_splits, dict):
        raise SystemExit(f"Source split plan must contain a 'splits' object: {plan_path}")
    split_images: dict[str, dict[str, list[Path]]] = {split: {} for split in SPLITS}
    available_by_key = {
        class_key: {key: source for source in sources for key in source_keys(source)}
        for class_key, sources in available_class_images.items()
    }
    for split in SPLITS:
        raw_classes = raw_splits.get(split)
        if not isinstance(raw_classes, dict):
            raise SystemExit(f"Source split plan is missing split '{split}': {plan_path}")
        for class_key, raw_sources in raw_classes.items():
            if class_key not in available_by_key:
                raise SystemExit(f"Source split plan class is unavailable after filtering: {class_key}")
            if not isinstance(raw_sources, list):
                raise SystemExit(f"Source split plan entries must be lists: {class_key}/{split}")
            sources = []
            missing = []
            for raw_source in raw_sources:
                source = available_by_key[class_key].get(str(raw_source)) or available_by_key[class_key].get(Path(str(raw_source)).as_posix())
                if source is None:
                    missing.append(str(raw_source))
                else:
                    sources.append(source)
            if missing:
                if missing_source_policy == "skip":
                    pass
                else:
                    raise SystemExit(
                        f"Source split plan references unavailable sources for {class_key}/{split}: "
                        + ", ".join(missing[:5])
                    )
            split_images[split][class_key] = sources
    return split_images


def choose_sources(paths: list[Path], count: int, rng: random.Random) -> list[Path]:
    if len(paths) >= count:
        return rng.sample(paths, count)
    return [rng.choice(paths) for _ in range(count)]


def make_click_all_grid(
    split_class_images: dict[str, list[Path]],
    class_map: dict[str, str],
    rng: random.Random,
    np_rng: np.random.Generator,
    image_size: int,
    min_targets: int,
    max_targets: int,
    hard_augment: bool,
    target_class: str | None = None,
) -> tuple[Image.Image, dict]:
    class_keys = list(split_class_images)
    target_class = target_class or rng.choice(class_keys)
    if target_class not in split_class_images:
        raise ValueError(f"Target class {target_class} is not available in this split.")
    target_count = rng.randint(min_targets, max_targets)
    target_indices = sorted(rng.sample(range(9), target_count))
    distractor_classes = [class_key for class_key in class_keys if class_key != target_class]
    if not distractor_classes:
        raise ValueError("Need at least 2 usable classes for click-all photo action data.")

    cell = image_size // 3
    grid = Image.new("RGB", (image_size, image_size), (238, 240, 243))
    items: list[dict] = []
    target_sources = choose_sources(split_class_images[target_class], target_count, rng)
    target_source_idx = 0

    for idx in range(9):
        if idx in target_indices:
            class_key = target_class
            source = target_sources[target_source_idx]
            target_source_idx += 1
        else:
            class_key = rng.choice(distractor_classes)
            source = rng.choice(split_class_images[class_key])
        with Image.open(source) as src_image:
            crop = hard_augment_cell(src_image, rng, np_rng, cell) if hard_augment else square_crop(src_image, cell)
        row, col = divmod(idx, 3)
        grid.paste(crop, (col * cell, row * cell))
        items.append(
            {
                "class_key": class_key,
                "object_name": class_map.get(class_key, class_key.replace("_", " ")),
                "source": str(source),
            }
        )

    target_object = class_map.get(target_class, target_class.replace("_", " "))
    metadata = {
        "prompt": PROMPT.format(object_name=target_object),
        "target_object": target_object,
        "target_class_key": target_class,
        "target_indices": target_indices,
        "target_index": target_indices[0],
        "click": None,
        "actions": target_indices_to_actions(target_indices),
        "items": items,
        "image_size": image_size,
        "difficulty": "photo_click_all",
        "hard_augment": hard_augment,
    }
    return grid, metadata


def make_balanced_target_schedule(class_keys: list[str], sample_count: int, rng: random.Random) -> list[str]:
    if not class_keys:
        return []
    schedule = [class_keys[idx % len(class_keys)] for idx in range(sample_count)]
    rng.shuffle(schedule)
    return schedule


def build_dataset(args: argparse.Namespace) -> Path:
    photo_root = Path(args.photo_root)
    class_map = load_class_map(photo_root)
    excluded = parse_csv_values(args.exclude_classes)
    fixed_classes = load_fixed_classes(args.include_classes, args.class_list)
    name_to_class_key = {object_name: class_key for class_key, object_name in class_map.items()}
    requested_class_keys = []
    for value in fixed_classes:
        requested_class_keys.append(value if value in class_map else name_to_class_key.get(value, value))
    bad_source_keys = load_bad_source_labels(args.bad_source_labels)
    blacklist_sources = load_blacklist_sources(args.blacklist_sources) | bad_source_keys
    if args.source_pool_manifest:
        candidate_class_images = load_source_pool_manifest(args.source_pool_manifest, photo_root, class_map)
    else:
        candidate_class_images = {class_key: list_images(photo_root / class_key) for class_key in class_map}
    class_images = {}
    for class_key, images in candidate_class_images.items():
        object_name = class_map.get(class_key, class_key)
        if requested_class_keys and class_key not in requested_class_keys:
            continue
        if class_key in excluded or object_name in excluded:
            continue
        images = filter_source_images(
            images,
            min_source_area=args.min_source_area,
            max_source_aspect_ratio=args.max_source_aspect_ratio,
            blacklist_sources=blacklist_sources,
        )
        if len(images) >= args.min_images_per_class:
            class_images[class_key] = images
    if requested_class_keys:
        missing = [class_key for class_key in requested_class_keys if class_key not in class_images]
        if missing:
            raise SystemExit(
                "Requested fixed classes are unavailable after filtering/min-images checks: "
                + ", ".join(missing)
            )
        class_images = {class_key: class_images[class_key] for class_key in requested_class_keys}
    if args.max_classes > 0 and len(class_images) > args.max_classes:
        ranked = sorted(class_images.items(), key=lambda kv: len(kv[1]), reverse=True)[: args.max_classes]
        class_images = dict(ranked)
    if len(class_images) < 2:
        raise SystemExit("Need at least 2 class folders with enough images.")
    excluded_bad_sources = count_bad_source_matches(candidate_class_images, bad_source_keys)
    if args.write_source_pool_manifest:
        save_source_pool_manifest(class_images, args.write_source_pool_manifest, class_map, args)

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    if args.source_split_plan:
        split_images = load_source_split_plan(
            args.source_split_plan,
            class_images,
            missing_source_policy=args.missing_plan_source_policy,
        )
    else:
        split_images = split_class_images(class_images, rng, args.val_ratio, args.test_ratio)
    if args.write_source_split_plan:
        save_source_split_plan(split_images, args.write_source_split_plan, args)
    sample_counts = {"train": args.num_train, "val": args.num_val, "test": args.num_test}
    output = Path(args.output_dir)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    selected_classes_path = output / "selected_classes.json"
    selected_classes_path.write_text(
        json.dumps(
            {
                "classes": [
                    {
                        "class_key": class_key,
                        "object_name": class_map.get(class_key, class_key.replace("_", " ")),
                        "source_images": len(images),
                    }
                    for class_key, images in class_images.items()
                ],
                "fixed_classes": requested_class_keys,
                "exclude_classes": sorted(excluded),
                "min_images_per_class": args.min_images_per_class,
                "min_source_area": args.min_source_area,
                "max_source_aspect_ratio": args.max_source_aspect_ratio,
                "blacklist_sources": str(args.blacklist_sources) if args.blacklist_sources else None,
                "bad_source_labels": str(args.bad_source_labels) if args.bad_source_labels else None,
                "excluded_bad_sources": excluded_bad_sources,
                "source_pool_manifest": str(args.source_pool_manifest) if args.source_pool_manifest else None,
                "write_source_pool_manifest": str(args.write_source_pool_manifest) if args.write_source_pool_manifest else None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    records = []
    start = time.time()
    print(
        json.dumps(
            {
                "status": "start",
                "output_dir": str(output),
                "usable_classes": len(class_images),
                "sample_counts": sample_counts,
                "hard_augment": args.hard_augment,
                "balanced_targets": args.balanced_targets,
                "min_source_area": args.min_source_area,
                "max_source_aspect_ratio": args.max_source_aspect_ratio,
                "blacklist_sources": str(args.blacklist_sources) if args.blacklist_sources else None,
                "bad_source_labels": str(args.bad_source_labels) if args.bad_source_labels else None,
                "excluded_bad_sources": excluded_bad_sources,
                "exclude_classes": sorted(excluded),
                "fixed_classes": requested_class_keys,
                "selected_classes": str(selected_classes_path),
                "source_pool_manifest": str(args.source_pool_manifest) if args.source_pool_manifest else None,
                "write_source_pool_manifest": str(args.write_source_pool_manifest) if args.write_source_pool_manifest else None,
                "source_split_plan": str(args.source_split_plan) if args.source_split_plan else None,
                "write_source_split_plan": str(args.write_source_split_plan) if args.write_source_split_plan else None,
                "missing_plan_source_policy": args.missing_plan_source_policy,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    global_idx = 0
    for split in SPLITS:
        usable = {class_key: paths for class_key, paths in split_images[split].items() if paths}
        if len(usable) < 2:
            raise SystemExit(f"Split {split} needs at least 2 usable classes.")
        target_schedule = (
            make_balanced_target_schedule(sorted(usable), sample_counts[split], rng)
            if args.balanced_targets
            else [None] * sample_counts[split]
        )
        for local_idx in range(sample_counts[split]):
            image, metadata = make_click_all_grid(
                usable,
                class_map,
                rng,
                np_rng,
                args.image_size,
                args.min_targets,
                args.max_targets,
                args.hard_augment,
                target_class=target_schedule[local_idx],
            )
            image_name = f"{split}_photo_action_{local_idx:05d}.jpg"
            image.save(image_dir / image_name, quality=92)
            records.append({"image": f"images/{image_name}", "split": split, **metadata})
            global_idx += 1
            if args.progress_every > 0 and (global_idx == 1 or global_idx % args.progress_every == 0):
                elapsed = time.time() - start
                rate = global_idx / elapsed if elapsed > 0 else 0.0
                print(
                    json.dumps(
                        {
                            "status": "progress",
                            "done": global_idx,
                            "samples_per_sec": round(rate, 2),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    manifest = output / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "saved", "manifest": str(manifest), "samples": len(records)}, ensure_ascii=False))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build real-photo click-all action dataset with source-disjoint splits.")
    parser.add_argument("--photo-root", required=True)
    parser.add_argument("--output-dir", default="data/photo_action_click_all")
    parser.add_argument("--num-train", type=int, default=2000)
    parser.add_argument("--num-val", type=int, default=400)
    parser.add_argument("--num-test", type=int, default=400)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--min-images-per-class", type=int, default=8)
    parser.add_argument("--max-classes", type=int, default=0)
    parser.add_argument("--min-targets", type=int, default=2)
    parser.add_argument("--max-targets", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--hard-augment", action="store_true")
    parser.add_argument("--balanced-targets", action="store_true", help="Balance target classes within each split.")
    parser.add_argument("--min-source-area", type=int, default=0, help="Drop source images smaller than this pixel area. 0 disables.")
    parser.add_argument(
        "--max-source-aspect-ratio",
        type=float,
        default=0.0,
        help="Drop source images whose long/short side ratio exceeds this value. 0 disables.",
    )
    parser.add_argument("--blacklist-sources", default=None, help="UTF-8 text file with one source image path or filename per line.")
    parser.add_argument(
        "--bad-source-labels",
        default=None,
        help="JSON file with bad source labels: {'sources': [{'source': path, 'reason': text, 'exclude': true}]}",
    )
    parser.add_argument("--exclude-classes", default="", help="Comma-separated class keys or object names to exclude.")
    parser.add_argument("--include-classes", default="", help="Comma-separated class keys or object names to keep as a fixed benchmark set.")
    parser.add_argument("--class-list", default=None, help="UTF-8 text file with one fixed class key or object name per line.")
    parser.add_argument("--source-pool-manifest", default=None, help="Reuse a fixed source pool manifest written by --write-source-pool-manifest.")
    parser.add_argument("--write-source-pool-manifest", default=None, help="Write the fixed source pool manifest to this JSON path.")
    parser.add_argument("--source-split-plan", default=None, help="Reuse a source split plan written by --write-source-split-plan.")
    parser.add_argument("--write-source-split-plan", default=None, help="Write the source train/val/test split plan to this JSON path.")
    parser.add_argument(
        "--missing-plan-source-policy",
        choices=("error", "skip"),
        default="error",
        help="When reusing a split plan after filtering, either error on missing sources or skip them.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    if args.min_targets < 1 or args.max_targets < args.min_targets or args.max_targets > 9:
        raise SystemExit("--min-targets and --max-targets must define a range within [1, 9].")
    if args.min_source_area < 0:
        raise SystemExit("--min-source-area must be >= 0.")
    if args.max_source_aspect_ratio < 0:
        raise SystemExit("--max-source-aspect-ratio must be >= 0.")
    build_dataset(args)


if __name__ == "__main__":
    main()
