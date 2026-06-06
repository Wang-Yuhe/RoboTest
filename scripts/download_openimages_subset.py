from __future__ import annotations

import argparse
import json
import ssl
import sys
from pathlib import Path
from urllib.error import URLError

import requests

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.multimodal_captcha.classes import (
    BROAD_OPENIMAGES_CLASSES,
    DEFAULT_OPENIMAGES_CLASSES,
    EXTENDED_OPENIMAGES_CLASSES,
    folder_for_openimages_class,
    object_name_for_folder,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def count_existing_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.suffix.lower() in IMAGE_EXTS)


def next_image_index(path: Path, folder: str) -> int:
    if not path.exists():
        return 0
    prefix = f"{folder}_"
    max_index = -1
    for item in path.iterdir():
        if item.suffix.lower() not in IMAGE_EXTS or not item.stem.startswith(prefix):
            continue
        suffix = item.stem[len(prefix) :]
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return max_index + 1


def write_manual_folder_template(output_dir: Path, classes: list[str]) -> None:
    photo_root = output_dir / "manual_photo_objects"
    photo_root.mkdir(parents=True, exist_ok=True)
    for class_name in classes:
        folder = folder_for_openimages_class(class_name)
        (photo_root / folder).mkdir(parents=True, exist_ok=True)

    readme = photo_root / "README.txt"
    readme.write_text(
        "Open Images automatic download failed or was skipped.\n\n"
        "You can manually place real object photos into these folders, then run:\n\n"
        f"python3 scripts/build_photo_grid_dataset.py --photo-root {photo_root} --output-dir data/photo_grid --num-samples 300\n\n"
        "Each folder should contain jpg/png/webp images for that object class.\n"
        "The folder names match the project's Chinese label mapping.\n",
        encoding="utf-8",
    )
    print(f"Manual photo folder template created at: {photo_root}")


def crop_detection(image_path: Path, bbox: list[float], pad_ratio: float = 0.08):
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    width, height = image.size
    x, y, w, h = bbox
    left = max(0, int((x - pad_ratio * w) * width))
    top = max(0, int((y - pad_ratio * h) * height))
    right = min(width, int((x + w + pad_ratio * w) * width))
    bottom = min(height, int((y + h + pad_ratio * h) * height))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom))


def export_openimages_crops(dataset, photo_root: Path, classes: list[str], crops_per_class: int) -> dict[str, int]:
    counts = {}
    next_indices = {}
    for class_name in classes:
        folder = folder_for_openimages_class(class_name)
        folder_path = photo_root / folder
        counts[folder] = count_existing_images(folder_path)
        next_indices[folder] = next_image_index(folder_path, folder)
    class_to_folder = {class_name: folder_for_openimages_class(class_name) for class_name in classes}
    metadata_path = photo_root / "class_names.json"
    if metadata_path.exists():
        class_names = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        class_names = {}
    class_names.update({
        folder_for_openimages_class(class_name): {
            "openimages_label": class_name,
            "object_name": object_name_for_folder(folder_for_openimages_class(class_name), class_name),
        }
        for class_name in classes
    })
    for folder in counts:
        (photo_root / folder).mkdir(parents=True, exist_ok=True)
    (photo_root / "class_names.json").write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")

    for sample in dataset:
        if all(count >= crops_per_class for count in counts.values()):
            break
        try:
            detections = sample.get_field("ground_truth")
        except Exception:
            detections = None
        if detections is None:
            continue
        for detection in detections.detections:
            label = detection.label
            if label not in class_to_folder:
                continue
            folder = class_to_folder[label]
            if counts[folder] >= crops_per_class:
                continue
            crop = crop_detection(Path(sample.filepath), detection.bounding_box)
            if crop is None:
                continue
            out = photo_root / folder / f"{folder}_{next_indices[folder]:05d}.jpg"
            while out.exists():
                next_indices[folder] += 1
                out = photo_root / folder / f"{folder}_{next_indices[folder]:05d}.jpg"
            crop.save(out, quality=92)
            counts[folder] += 1
            next_indices[folder] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small Open Images subset with FiftyOne.")
    parser.add_argument("--output-dir", default="data/openimages_raw")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--class-preset", choices=["default", "extended", "broad"], default="broad")
    parser.add_argument("--photo-root", default="data/photo_objects", help="Where cropped object photos are exported.")
    parser.add_argument("--crops-per-class", type=int, default=80)
    parser.add_argument("--manual-template-only", action="store_true", help="Only create local folders for manually collected photos.")
    args = parser.parse_args()
    if args.classes is None:
        if args.class_preset == "broad":
            args.classes = BROAD_OPENIMAGES_CLASSES
        elif args.class_preset == "extended":
            args.classes = EXTENDED_OPENIMAGES_CLASSES
        else:
            args.classes = DEFAULT_OPENIMAGES_CLASSES

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.manual_template_only:
        write_manual_folder_template(output_dir, args.classes)
        return

    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: fiftyone\n"
            "Install it with:\n"
            "  python3 -m pip install fiftyone\n"
            "Then rerun this script."
        ) from exc

    try:
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split=args.split,
            label_types=["detections"],
            classes=args.classes,
            max_samples=args.max_samples,
            shuffle=True,
        )
    except (requests.exceptions.SSLError, ssl.SSLError, URLError, OSError, TypeError) as exc:
        write_manual_folder_template(output_dir, args.classes)
        raise SystemExit(
            "\nOpen Images download/load failed.\n"
            "If this is an SSL error, it is usually a network/proxy issue.\n"
            "If this is a TypeError from FiftyOne, upgrade/downgrade FiftyOne or use the current script version without dataset_dir.\n\n"
            "Recommended options:\n"
            "1. Retry under a stable proxy/VPN, then rerun this script.\n"
            "2. Rerun after pulling the latest script changes.\n"
            "3. Download photos manually into the created folder template.\n"
            "4. Use your own photos with scripts/build_photo_grid_dataset.py.\n\n"
            f"Original error: {exc}\n"
        ) from exc

    summary = {
        "dataset_name": dataset.name,
        "output_dir": str(output_dir),
        "split": args.split,
        "classes": args.classes,
        "samples": len(dataset),
    }
    crop_counts = export_openimages_crops(dataset, Path(args.photo_root), args.classes, args.crops_per_class)
    summary["photo_root"] = args.photo_root
    summary["crop_counts"] = crop_counts
    (output_dir / "download_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Downloaded images are managed by FiftyOne under the dataset directory.")
    print(f"Cropped object photos exported to {args.photo_root}")


if __name__ == "__main__":
    main()
