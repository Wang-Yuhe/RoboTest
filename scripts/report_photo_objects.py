from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.multimodal_captcha.classes import CLASS_MAP


def count_images(path: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sum(1 for item in path.rglob("*") if item.suffix.lower() in exts)


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
    if photo_root.exists():
        for child in photo_root.iterdir():
            if child.is_dir() and child.name not in mapping:
                mapping[child.name] = child.name.replace("_", " ")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Report real-photo object class coverage.")
    parser.add_argument("--photo-root", default="data/photo_objects")
    parser.add_argument("--min-images", type=int, default=20)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    root = Path(args.photo_root)
    rows = []
    class_map = load_class_map(root)
    for class_key, zh_name in class_map.items():
        n = count_images(root / class_key)
        if n > 0 or (root / class_key).exists():
            rows.append({"class_key": class_key, "object_name": zh_name, "images": n, "usable": n >= args.min_images})

    rows.sort(key=lambda x: (-x["images"], x["class_key"]))
    usable = [row for row in rows if row["usable"]]
    print(f"Photo root: {root}")
    print(f"Usable classes (>={args.min_images} images): {len(usable)} / {len(rows)}")
    for row in rows:
        mark = "OK" if row["usable"] else "--"
        print(f"{mark} {row['class_key']:<16} {row['object_name']:<8} {row['images']:>5}")

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved JSON report to {args.json_output}")


if __name__ == "__main__":
    main()
