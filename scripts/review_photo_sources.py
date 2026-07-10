from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_photo_action_dataset import list_images, load_fixed_classes


def classify_source(path: Path, min_source_area: int, max_source_aspect_ratio: float) -> tuple[str | None, tuple[int, int] | None]:
    try:
        with Image.open(path) as image:
            width, height = image.size
    except OSError:
        return "unreadable_image", None
    if min_source_area > 0 and width * height < min_source_area:
        return "small_area", (width, height)
    if max_source_aspect_ratio > 0:
        short = max(min(width, height), 1)
        if max(width, height) / short > max_source_aspect_ratio:
            return "extreme_aspect_ratio", (width, height)
    return None, (width, height)


def make_thumbnail(path: Path, size: int, bad_reason: str | None) -> Image.Image:
    tile = Image.new("RGB", (size, size), (238, 240, 243))
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((size, size), Image.Resampling.LANCZOS)
            left = (size - image.width) // 2
            top = (size - image.height) // 2
            tile.paste(image, (left, top))
    except OSError:
        pass
    if bad_reason:
        draw = ImageDraw.Draw(tile)
        draw.rectangle((1, 1, size - 2, size - 2), outline=(220, 40, 40), width=3)
    return tile


def write_contact_sheet(class_key: str, images: list[Path], output: Path, thumbnail_size: int, reasons: dict[Path, str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not images:
        Image.new("RGB", (thumbnail_size, thumbnail_size), (238, 240, 243)).save(output)
        return
    columns = min(8, max(1, math.ceil(math.sqrt(len(images)))))
    rows = math.ceil(len(images) / columns)
    label_h = 18
    sheet = Image.new("RGB", (columns * thumbnail_size, rows * (thumbnail_size + label_h)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for idx, path in enumerate(images):
        row, col = divmod(idx, columns)
        x = col * thumbnail_size
        y = row * (thumbnail_size + label_h)
        reason = reasons.get(path)
        sheet.paste(make_thumbnail(path, thumbnail_size, reason), (x, y))
        label = reason or path.stem[:10]
        draw.text((x + 2, y + thumbnail_size + 2), label[:16], fill=(30, 30, 30))
    sheet.save(output, quality=92)


def review_sources(args: argparse.Namespace) -> Path:
    photo_root = Path(args.photo_root)
    output_dir = Path(args.output_dir)
    classes = load_fixed_classes(args.classes, args.class_list)
    if not classes:
        raise SystemExit("Provide --classes or --class-list.")

    all_labels = []
    summary = {"photo_root": str(photo_root), "classes": {}}
    contact_dir = output_dir / "contact_sheets"
    for class_key in classes:
        images = list_images(photo_root / class_key)
        if args.max_per_class > 0:
            images = images[: args.max_per_class]
        reasons: dict[Path, str] = {}
        sizes = {}
        for image in images:
            reason, size = classify_source(image, args.min_source_area, args.max_source_aspect_ratio)
            if size:
                sizes[str(image)] = {"width": size[0], "height": size[1]}
            if reason:
                reasons[image] = reason
                all_labels.append(
                    {
                        "source": str(image),
                        "class_key": class_key,
                        "reason": reason,
                        "exclude": True,
                    }
                )
        write_contact_sheet(
            class_key=class_key,
            images=images,
            output=contact_dir / f"{class_key}.jpg",
            thumbnail_size=args.thumbnail_size,
            reasons=reasons,
        )
        summary["classes"][class_key] = {
            "reviewed_sources": len(images),
            "bad_sources": len(reasons),
            "contact_sheet": str(contact_dir / f"{class_key}.jpg"),
            "sizes": sizes,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "bad_source_labels.json"
    labels_path.write_text(json.dumps({"sources": all_labels}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "saved", "bad_source_labels": str(labels_path), "bad_sources": len(all_labels)}, ensure_ascii=False))
    return labels_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create weak-class source review sheets and reason-coded bad source labels.")
    parser.add_argument("--photo-root", required=True)
    parser.add_argument("--classes", default="", help="Comma-separated class keys to review.")
    parser.add_argument("--class-list", default=None, help="UTF-8 file with one class key per line.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-per-class", type=int, default=160)
    parser.add_argument("--thumbnail-size", type=int, default=96)
    parser.add_argument("--min-source-area", type=int, default=5000)
    parser.add_argument("--max-source-aspect-ratio", type=float, default=4.0)
    args = parser.parse_args()
    if args.max_per_class < 0:
        raise SystemExit("--max-per-class must be >= 0.")
    if args.thumbnail_size < 16:
        raise SystemExit("--thumbnail-size must be >= 16.")
    if args.min_source_area < 0:
        raise SystemExit("--min-source-area must be >= 0.")
    if args.max_source_aspect_ratio < 0:
        raise SystemExit("--max-source-aspect-ratio must be >= 0.")
    review_sources(args)


if __name__ == "__main__":
    main()
