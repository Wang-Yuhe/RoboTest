import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


def make_review_photo_root(root: Path) -> None:
    for class_name, color in {
        "dress": (220, 80, 120),
        "boat": (50, 110, 220),
    }.items():
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for idx in range(3):
            Image.new("RGB", (80, 80), color).save(class_dir / f"{class_name}_good_{idx}.jpg")
    Image.new("RGB", (20, 20), (10, 10, 10)).save(root / "dress" / "dress_tiny.jpg")
    Image.new("RGB", (400, 20), (20, 20, 20)).save(root / "boat" / "boat_wide.jpg")


class SourceReviewScriptTests(unittest.TestCase):
    def test_review_photo_sources_writes_contact_sheets_and_bad_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo_root = root / "photos"
            output_dir = root / "review"
            make_review_photo_root(photo_root)

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/review_photo_sources.py",
                    "--photo-root",
                    str(photo_root),
                    "--classes",
                    "dress,boat",
                    "--output-dir",
                    str(output_dir),
                    "--max-per-class",
                    "10",
                    "--thumbnail-size",
                    "48",
                    "--min-source-area",
                    "2000",
                    "--max-source-aspect-ratio",
                    "4.0",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            labels = json.loads((output_dir / "bad_source_labels.json").read_text(encoding="utf-8"))
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

            self.assertTrue((output_dir / "contact_sheets" / "dress.jpg").exists())
            self.assertTrue((output_dir / "contact_sheets" / "boat.jpg").exists())
            bad_names = {Path(entry["source"]).name: entry["reason"] for entry in labels["sources"]}
            self.assertEqual(bad_names["dress_tiny.jpg"], "small_area")
            self.assertEqual(bad_names["boat_wide.jpg"], "extreme_aspect_ratio")
            self.assertEqual(summary["classes"]["dress"]["bad_sources"], 1)
            self.assertEqual(summary["classes"]["boat"]["bad_sources"], 1)


if __name__ == "__main__":
    unittest.main()
