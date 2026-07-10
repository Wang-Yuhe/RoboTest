import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.clip_zero_shot_action_eval import feature_tensor


class ClipZeroShotActionEvalTests(unittest.TestCase):
    def test_feature_tensor_accepts_pooler_output_objects(self):
        class Output:
            def __init__(self):
                self.pooler_output = "features"

        self.assertEqual(feature_tensor(Output()), "features")

    def test_clip_eval_can_run_with_mock_scores_without_transformers_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            image_dir = data_dir / "images"
            output = root / "clip_eval.json"
            image_dir.mkdir(parents=True)
            image = Image.new("RGB", (90, 90), (240, 240, 240))
            image.save(image_dir / "sample.jpg")
            record = {
                "image": "images/sample.jpg",
                "split": "test",
                "prompt": "click all cars",
                "target_object": "car",
                "target_class_key": "car",
                "target_indices": [0, 4],
            }
            (data_dir / "manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/clip_zero_shot_action_eval.py",
                    "--data-dir",
                    str(data_dir),
                    "--split",
                    "test",
                    "--output",
                    str(output),
                    "--mock-scores",
                    "--threshold",
                    "0.5",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["samples"], 1)
        self.assertEqual(payload["mode"], "mock")
        self.assertEqual(payload["oracle_topk"]["cell_precision"], 1.0)
        self.assertEqual(payload["oracle_topk"]["cell_recall"], 1.0)
        self.assertEqual(payload["oracle_topk"]["cell_exact_match"], 1.0)
        self.assertEqual(payload["fixed_topk"]["k"], 3)
        self.assertEqual(payload["fixed_topk"]["cell_precision"], 2 / 3)
        self.assertEqual(payload["fixed_topk"]["cell_recall"], 1.0)
        self.assertEqual(payload["threshold_policy"]["threshold"], 0.5)
        self.assertEqual(payload["threshold_policy"]["cell_exact_match"], 1.0)

    def test_clip_eval_auto_threshold_uses_calibration_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            image_dir = data_dir / "images"
            output = root / "clip_eval.json"
            image_dir.mkdir(parents=True)
            image = Image.new("RGB", (90, 90), (240, 240, 240))
            image.save(image_dir / "sample.jpg")
            records = [
                {
                    "image": "images/sample.jpg",
                    "split": "val",
                    "prompt": "click all cars",
                    "target_object": "car",
                    "target_class_key": "car",
                    "target_indices": [0, 4],
                },
                {
                    "image": "images/sample.jpg",
                    "split": "test",
                    "prompt": "click all cars",
                    "target_object": "car",
                    "target_class_key": "car",
                    "target_indices": [1, 3],
                },
            ]
            (data_dir / "manifest.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/clip_zero_shot_action_eval.py",
                    "--data-dir",
                    str(data_dir),
                    "--split",
                    "test",
                    "--output",
                    str(output),
                    "--mock-scores",
                    "--threshold",
                    "auto",
                    "--calibration-split",
                    "val",
                    "--threshold-grid",
                    "0.3,0.5,0.7",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["samples"], 1)
        self.assertEqual(payload["threshold_policy"]["threshold_source"], "auto_val")
        self.assertEqual(payload["threshold_policy"]["threshold"], 0.5)
        self.assertEqual(payload["threshold_policy"]["cell_exact_match"], 1.0)
        self.assertEqual(payload["threshold_policy"]["calibration_samples"], 1)
        self.assertEqual(len(payload["threshold_policy"]["candidates"]), 3)


if __name__ == "__main__":
    unittest.main()
