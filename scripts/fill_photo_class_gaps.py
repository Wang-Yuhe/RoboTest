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

from scripts.download_openimages_subset import count_existing_images, export_openimages_crops
from src.multimodal_captcha.classes import (
    BROAD_OPENIMAGES_CLASSES,
    DEFAULT_OPENIMAGES_CLASSES,
    EXTENDED_OPENIMAGES_CLASSES,
    folder_for_openimages_class,
    object_name_for_folder,
)


def classes_for_preset(preset: str) -> list[str]:
    if preset == "broad":
        return BROAD_OPENIMAGES_CLASSES
    if preset == "extended":
        return EXTENDED_OPENIMAGES_CLASSES
    return DEFAULT_OPENIMAGES_CLASSES


def coverage_rows(photo_root: Path, labels: list[str], min_images: int) -> list[dict]:
    rows = []
    for label in labels:
        folder = folder_for_openimages_class(label)
        count = count_existing_images(photo_root / folder)
        rows.append(
            {
                "openimages_label": label,
                "class_key": folder,
                "object_name": object_name_for_folder(folder, label),
                "images": count,
                "usable": count >= min_images,
            }
        )
    rows.sort(key=lambda row: (-row["images"], row["class_key"]))
    return rows


def write_report(path: Path, rows: list[dict], min_images: int) -> None:
    usable = sum(1 for row in rows if row["usable"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"min_images": min_images, "usable_classes": usable, "classes": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Top up real-photo object classes until enough classes reach a target count.")
    parser.add_argument("--photo-root", default="data/photo_objects")
    parser.add_argument("--output-dir", default="data/openimages_gap_fill_raw")
    parser.add_argument("--class-preset", choices=["default", "extended", "broad"], default="broad")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--target-classes", type=int, default=100)
    parser.add_argument("--min-images", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--candidate-buffer", type=int, default=35)
    parser.add_argument("--report-output", default="outputs/photo_gap_fill_report.json")
    args = parser.parse_args()

    photo_root = Path(args.photo_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = classes_for_preset(args.class_preset)

    rows = coverage_rows(photo_root, labels, args.min_images)
    usable = sum(1 for row in rows if row["usable"])
    print(f"Initial usable classes: {usable} / {len(rows)}")

    foz = None

    for pass_idx in range(args.passes):
        if usable >= args.target_classes:
            break
        if foz is None:
            try:
                import fiftyone.zoo as foz
            except ImportError as exc:
                raise SystemExit(
                    "Missing dependency: fiftyone\n"
                    "Install it with:\n"
                    "  python3 -m pip install fiftyone\n"
                    "Then rerun this script."
                ) from exc

        needed = args.target_classes - usable
        underfilled = [row for row in rows if not row["usable"] and row["images"] > 0]
        empty = [row for row in rows if not row["usable"] and row["images"] == 0]
        candidates = (underfilled + empty)[: min(len(underfilled) + len(empty), needed + args.candidate_buffer)]
        request_labels = [row["openimages_label"] for row in candidates]
        if not request_labels:
            break

        print(
            f"Pass {pass_idx + 1}/{args.passes}: requesting {len(request_labels)} classes "
            f"from Open Images {args.split}, max_samples={args.max_samples}"
        )
        try:
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split=args.split,
                label_types=["detections"],
                classes=request_labels,
                max_samples=args.max_samples,
                shuffle=True,
            )
        except (requests.exceptions.SSLError, ssl.SSLError, URLError, OSError, TypeError) as exc:
            raise SystemExit(f"Open Images download/load failed: {exc}") from exc

        before = {row["class_key"]: row["images"] for row in rows}
        export_openimages_crops(dataset, photo_root, request_labels, args.min_images)
        rows = coverage_rows(photo_root, labels, args.min_images)
        usable = sum(1 for row in rows if row["usable"])
        gained = sum(max(0, row["images"] - before.get(row["class_key"], 0)) for row in rows)
        print(f"After pass {pass_idx + 1}: usable classes {usable} / {len(rows)}, new crops {gained}")
        if gained == 0:
            print("No new crops found in this pass; try a larger --max-samples or another split.")
            break

    write_report(Path(args.report_output), rows, args.min_images)
    print(f"Saved gap-fill report to {args.report_output}")
    print(f"Final usable classes: {usable} / {len(rows)}")


if __name__ == "__main__":
    main()
