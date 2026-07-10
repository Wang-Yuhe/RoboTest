import json
import inspect
import random
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.build_photo_action_dataset import hard_augment_cell
from src.multimodal_captcha.dataset import ActionSequenceDataset, build_vocab


def make_photo_root(root: Path) -> None:
    colors = {
        "car": (220, 54, 46),
        "bicycle": (54, 112, 220),
        "traffic_light": (51, 160, 88),
    }
    for class_name, color in colors.items():
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for idx in range(12):
            image = Image.new("RGB", (80, 80), tuple(min(255, channel + idx) for channel in color))
            image.save(class_dir / f"{class_name}_{idx:03d}.jpg")


def make_quality_photo_root(root: Path) -> None:
    classes = {
        "car": (220, 54, 46),
        "bicycle": (54, 112, 220),
        "traffic_light": (51, 160, 88),
        "person": (120, 120, 120),
    }
    for class_name, color in classes.items():
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for idx in range(8):
            image = Image.new("RGB", (96, 80), tuple(min(255, channel + idx) for channel in color))
            image.save(class_dir / f"{class_name}_good_{idx:03d}.jpg")
    Image.new("RGB", (20, 20), (10, 10, 10)).save(root / "car" / "car_tiny.jpg")
    Image.new("RGB", (400, 20), (20, 20, 20)).save(root / "car" / "car_wide.jpg")
    Image.new("RGB", (96, 80), (30, 30, 30)).save(root / "car" / "car_blacklisted.jpg")


def load_manifest(output_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]


class PhotoActionDatasetTests(unittest.TestCase):
    def test_hard_augmentation_uses_explicit_numpy_rng(self):
        self.assertIn("np_rng", inspect.signature(hard_augment_cell).parameters)
        image = Image.new("RGB", (80, 80), (120, 80, 40))

        first = hard_augment_cell(image, random.Random(7), np.random.default_rng(7), 64)
        second = hard_augment_cell(image, random.Random(7), np.random.default_rng(7), 64)

        self.assertEqual(first.tobytes(), second.tobytes())

    def test_build_photo_action_dataset_keeps_source_files_disjoint_by_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo_root = root / "photos"
            output_dir = root / "photo_action"
            make_photo_root(photo_root)

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_photo_action_dataset.py",
                    "--photo-root",
                    str(photo_root),
                    "--output-dir",
                    str(output_dir),
                    "--num-train",
                    "8",
                    "--num-val",
                    "4",
                    "--num-test",
                    "4",
                    "--min-images-per-class",
                    "6",
                    "--min-targets",
                    "2",
                    "--max-targets",
                    "3",
                    "--image-size",
                    "192",
                    "--seed",
                    "13",
                    "--progress-every",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            records = [json.loads(line) for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(records), 16)
        by_split = {"train": set(), "val": set(), "test": set()}
        for record in records:
            self.assertIn(record["split"], by_split)
            self.assertIn("请点击所有", record["prompt"])
            self.assertGreaterEqual(len(record["target_indices"]), 2)
            self.assertEqual(record["actions"][-1]["type"], "done")
            for item in record["items"]:
                by_split[record["split"]].add(item["source"])
            target_object = record["target_object"]
            for index in record["target_indices"]:
                self.assertEqual(record["items"][index]["object_name"], target_object)

        self.assertTrue(by_split["train"].isdisjoint(by_split["val"]))
        self.assertTrue(by_split["train"].isdisjoint(by_split["test"]))
        self.assertTrue(by_split["val"].isdisjoint(by_split["test"]))

    def test_action_sequence_dataset_uses_manifest_split_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo_root = root / "photos"
            output_dir = root / "photo_action"
            make_photo_root(photo_root)
            subprocess.run(
                [
                    sys.executable,
                    "scripts/build_photo_action_dataset.py",
                    "--photo-root",
                    str(photo_root),
                    "--output-dir",
                    str(output_dir),
                    "--num-train",
                    "5",
                    "--num-val",
                    "3",
                    "--num-test",
                    "2",
                    "--min-images-per-class",
                    "6",
                    "--seed",
                    "21",
                    "--progress-every",
                    "0",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            vocab = build_vocab(output_dir / "manifest.jsonl")
            train_set = ActionSequenceDataset(output_dir, split="train", vocab=vocab)
            val_set = ActionSequenceDataset(output_dir, split="val", vocab=vocab)
            test_set = ActionSequenceDataset(output_dir, split="test", vocab=vocab)

        self.assertEqual(len(train_set), 5)
        self.assertEqual(len(val_set), 3)
        self.assertEqual(len(test_set), 2)

    def test_build_photo_action_dataset_can_balance_target_classes_by_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo_root = root / "photos"
            output_dir = root / "photo_action"
            make_photo_root(photo_root)

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_photo_action_dataset.py",
                    "--photo-root",
                    str(photo_root),
                    "--output-dir",
                    str(output_dir),
                    "--num-train",
                    "9",
                    "--num-val",
                    "6",
                    "--num-test",
                    "6",
                    "--min-images-per-class",
                    "6",
                    "--seed",
                    "29",
                    "--balanced-targets",
                    "--progress-every",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            records = [json.loads(line) for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]

        for split in ("train", "val", "test"):
            counts = Counter(record["target_class_key"] for record in records if record["split"] == split)
            self.assertEqual(len(counts), 3)
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_build_photo_action_dataset_filters_low_quality_sources_and_excluded_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo_root = root / "photos"
            output_dir = root / "photo_action"
            blacklist = root / "blacklist.txt"
            make_quality_photo_root(photo_root)
            blacklist.write_text(str(photo_root / "car" / "car_blacklisted.jpg") + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_photo_action_dataset.py",
                    "--photo-root",
                    str(photo_root),
                    "--output-dir",
                    str(output_dir),
                    "--num-train",
                    "9",
                    "--num-val",
                    "6",
                    "--num-test",
                    "6",
                    "--min-images-per-class",
                    "6",
                    "--min-source-area",
                    "2000",
                    "--max-source-aspect-ratio",
                    "4.0",
                    "--blacklist-sources",
                    str(blacklist),
                    "--exclude-classes",
                    "person",
                    "--seed",
                    "41",
                    "--balanced-targets",
                    "--progress-every",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            records = [json.loads(line) for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]

        used_sources = {Path(item["source"]).name for record in records for item in record["items"]}
        used_classes = {item["class_key"] for record in records for item in record["items"]}
        target_classes = {record["target_class_key"] for record in records}
        self.assertNotIn("car_tiny.jpg", used_sources)
        self.assertNotIn("car_wide.jpg", used_sources)
        self.assertNotIn("car_blacklisted.jpg", used_sources)
        self.assertNotIn("person", used_classes)
        self.assertNotIn("person", target_classes)

    def test_build_photo_action_dataset_can_use_fixed_class_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo_root = root / "photos"
            output_dir = root / "photo_action"
            class_list = root / "classes.txt"
            make_photo_root(photo_root)
            class_list.write_text("# fixed benchmark classes\ncar\nbicycle\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_photo_action_dataset.py",
                    "--photo-root",
                    str(photo_root),
                    "--output-dir",
                    str(output_dir),
                    "--num-train",
                    "8",
                    "--num-val",
                    "4",
                    "--num-test",
                    "4",
                    "--min-images-per-class",
                    "6",
                    "--class-list",
                    str(class_list),
                    "--balanced-targets",
                    "--seed",
                    "43",
                    "--progress-every",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            records = [json.loads(line) for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
            selected = json.loads((output_dir / "selected_classes.json").read_text(encoding="utf-8"))

        used_classes = {item["class_key"] for record in records for item in record["items"]}
        target_classes = {record["target_class_key"] for record in records}
        self.assertEqual(used_classes, {"car", "bicycle"})
        self.assertEqual(target_classes, {"car", "bicycle"})
        self.assertEqual([item["class_key"] for item in selected["classes"]], ["car", "bicycle"])

    def test_build_photo_action_dataset_fails_when_fixed_class_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo_root = root / "photos"
            output_dir = root / "photo_action"
            make_photo_root(photo_root)

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_photo_action_dataset.py",
                    "--photo-root",
                    str(photo_root),
                    "--output-dir",
                    str(output_dir),
                    "--num-train",
                    "4",
                    "--num-val",
                    "2",
                    "--num-test",
                    "2",
                    "--min-images-per-class",
                    "6",
                    "--include-classes",
                    "car,missing_class",
                    "--progress-every",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Requested fixed classes are unavailable", result.stderr + result.stdout)

    def test_build_photo_action_dataset_writes_and_reuses_source_split_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo_root = root / "photos"
            first_output = root / "first"
            second_output = root / "second"
            split_plan = root / "source_split_plan.json"
            make_photo_root(photo_root)

            first = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_photo_action_dataset.py",
                    "--photo-root",
                    str(photo_root),
                    "--output-dir",
                    str(first_output),
                    "--num-train",
                    "9",
                    "--num-val",
                    "6",
                    "--num-test",
                    "6",
                    "--min-images-per-class",
                    "6",
                    "--include-classes",
                    "car,bicycle,traffic_light",
                    "--write-source-split-plan",
                    str(split_plan),
                    "--balanced-targets",
                    "--seed",
                    "47",
                    "--progress-every",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(first.returncode, 0, msg=first.stderr + first.stdout)
            self.assertTrue(split_plan.exists())

            second = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_photo_action_dataset.py",
                    "--photo-root",
                    str(photo_root),
                    "--output-dir",
                    str(second_output),
                    "--num-train",
                    "9",
                    "--num-val",
                    "6",
                    "--num-test",
                    "6",
                    "--min-images-per-class",
                    "6",
                    "--include-classes",
                    "car,bicycle,traffic_light",
                    "--source-split-plan",
                    str(split_plan),
                    "--balanced-targets",
                    "--seed",
                    "99",
                    "--progress-every",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(second.returncode, 0, msg=second.stderr + second.stdout)

            plan = json.loads(split_plan.read_text(encoding="utf-8"))
            records = [json.loads(line) for line in (second_output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]

        allowed = {
            split: {class_key: set(paths) for class_key, paths in classes.items()}
            for split, classes in plan["splits"].items()
        }
        for record in records:
            for item in record["items"]:
                self.assertIn(item["source"], allowed[record["split"]][item["class_key"]])

    def test_build_photo_action_dataset_writes_and_reuses_source_pool_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo_root = root / "photos"
            first_output = root / "first"
            second_output = root / "second"
            source_pool = root / "source_pool.json"
            script = Path.cwd() / "scripts" / "build_photo_action_dataset.py"
            make_photo_root(photo_root)

            first = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--photo-root",
                    str(photo_root),
                    "--output-dir",
                    str(first_output),
                    "--num-train",
                    "9",
                    "--num-val",
                    "6",
                    "--num-test",
                    "6",
                    "--min-images-per-class",
                    "6",
                    "--include-classes",
                    "car,bicycle,traffic_light",
                    "--write-source-pool-manifest",
                    str(source_pool),
                    "--balanced-targets",
                    "--seed",
                    "53",
                    "--progress-every",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(first.returncode, 0, msg=first.stderr + first.stdout)
            self.assertTrue(source_pool.exists())

            extra_class = photo_root / "airplane"
            extra_class.mkdir()
            for idx in range(12):
                Image.new("RGB", (80, 80), (200, 200, 20)).save(extra_class / f"airplane_{idx:03d}.jpg")

            second = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--photo-root",
                    str(photo_root),
                    "--output-dir",
                    str(second_output),
                    "--source-pool-manifest",
                    str(source_pool),
                    "--num-train",
                    "9",
                    "--num-val",
                    "6",
                    "--num-test",
                    "6",
                    "--balanced-targets",
                    "--seed",
                    "59",
                    "--progress-every",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(second.returncode, 0, msg=second.stderr + second.stdout)
            records = load_manifest(second_output)

        used_classes = {item["class_key"] for record in records for item in record["items"]}
        self.assertEqual(used_classes, {"car", "bicycle", "traffic_light"})
        self.assertNotIn("airplane", used_classes)

    def test_source_pool_manifest_reuses_project_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo_root = root / "photos"
            first_output = root / "first"
            second_output = root / "second"
            source_pool = root / "source_pool.json"
            script = Path(__file__).resolve().parents[1] / "scripts" / "build_photo_action_dataset.py"
            make_photo_root(photo_root)

            first = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--photo-root",
                    str(photo_root),
                    "--output-dir",
                    str(first_output),
                    "--num-train",
                    "9",
                    "--num-val",
                    "6",
                    "--num-test",
                    "6",
                    "--min-images-per-class",
                    "6",
                    "--include-classes",
                    "car,bicycle,traffic_light",
                    "--write-source-pool-manifest",
                    str(source_pool),
                    "--balanced-targets",
                    "--seed",
                    "71",
                    "--progress-every",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(first.returncode, 0, msg=first.stderr + first.stdout)
            pool = json.loads(source_pool.read_text(encoding="utf-8"))
            for class_entry in pool["classes"]:
                class_entry["sources"] = [
                    str(Path(source).relative_to(root))
                    for source in class_entry["sources"]
                ]
            source_pool.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")

            second = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--photo-root",
                    str(photo_root),
                    "--output-dir",
                    str(second_output),
                    "--source-pool-manifest",
                    str(source_pool),
                    "--num-train",
                    "9",
                    "--num-val",
                    "6",
                    "--num-test",
                    "6",
                    "--balanced-targets",
                    "--seed",
                    "73",
                    "--progress-every",
                    "0",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(second.returncode, 0, msg=second.stderr + second.stdout)
            records = load_manifest(second_output)

        self.assertEqual(len(records), 21)

    def test_bad_source_labels_can_skip_sources_while_reusing_split_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo_root = root / "photos"
            dirty_output = root / "dirty"
            clean_output = root / "clean"
            source_pool = root / "source_pool.json"
            split_plan = root / "split_plan.json"
            bad_labels = root / "bad_source_labels.json"
            make_photo_root(photo_root)
            bad_source = photo_root / "car" / "car_000.jpg"
            bad_labels.write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "source": str(bad_source),
                                "class_key": "car",
                                "reason": "ambiguous_crop",
                                "exclude": True,
                            }
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            dirty = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_photo_action_dataset.py",
                    "--photo-root",
                    str(photo_root),
                    "--output-dir",
                    str(dirty_output),
                    "--num-train",
                    "9",
                    "--num-val",
                    "6",
                    "--num-test",
                    "6",
                    "--min-images-per-class",
                    "6",
                    "--include-classes",
                    "car,bicycle,traffic_light",
                    "--write-source-pool-manifest",
                    str(source_pool),
                    "--write-source-split-plan",
                    str(split_plan),
                    "--balanced-targets",
                    "--seed",
                    "61",
                    "--progress-every",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(dirty.returncode, 0, msg=dirty.stderr + dirty.stdout)

            clean = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_photo_action_dataset.py",
                    "--photo-root",
                    str(photo_root),
                    "--output-dir",
                    str(clean_output),
                    "--source-pool-manifest",
                    str(source_pool),
                    "--source-split-plan",
                    str(split_plan),
                    "--missing-plan-source-policy",
                    "skip",
                    "--bad-source-labels",
                    str(bad_labels),
                    "--num-train",
                    "9",
                    "--num-val",
                    "6",
                    "--num-test",
                    "6",
                    "--balanced-targets",
                    "--seed",
                    "67",
                    "--progress-every",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(clean.returncode, 0, msg=clean.stderr + clean.stdout)
            clean_records = load_manifest(clean_output)
            selected = json.loads((clean_output / "selected_classes.json").read_text(encoding="utf-8"))

        used_sources = {item["source"] for record in clean_records for item in record["items"]}
        self.assertNotIn(str(bad_source), used_sources)
        self.assertEqual(selected["bad_source_labels"], str(bad_labels))
        self.assertEqual(selected["excluded_bad_sources"], 1)


if __name__ == "__main__":
    unittest.main()
