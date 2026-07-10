import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.screenspot_pro_clip_adapter import (
    candidate_grid_boxes,
    evaluate_results,
    feature_tensor,
    point_in_bbox,
)


class ScreenSpotProClipAdapterTests(unittest.TestCase):
    def test_feature_tensor_accepts_pooler_output_objects(self):
        class Output:
            def __init__(self):
                self.pooler_output = "features"

        self.assertEqual(feature_tensor(Output()), "features")

    def test_candidate_grid_boxes_cover_image_with_centers(self):
        boxes = candidate_grid_boxes(width=100, height=80, rows=2, cols=2)

        self.assertEqual(boxes[0]["bbox"], [0, 0, 50, 40])
        self.assertEqual(boxes[0]["center_pixel"], [25.0, 20.0])
        self.assertEqual(boxes[-1]["bbox"], [50, 40, 100, 80])
        self.assertEqual(boxes[-1]["center_norm"], [0.75, 0.75])

    def test_point_in_bbox_uses_pixel_coordinates(self):
        self.assertTrue(point_in_bbox([25, 20], [20, 10, 30, 30]))
        self.assertFalse(point_in_bbox([31, 20], [20, 10, 30, 30]))

    def test_evaluate_results_matches_official_positive_point_rule(self):
        results = [
            {"correctness": "correct", "ui_type": "text"},
            {"correctness": "wrong", "ui_type": "icon"},
            {"correctness": "wrong_format", "ui_type": "text"},
        ]

        report = evaluate_results(results)

        self.assertEqual(report["metrics"]["overall"]["num_total"], 3)
        self.assertEqual(report["metrics"]["overall"]["num_correct_action"], 1)
        self.assertEqual(report["metrics"]["overall"]["wrong_format_num"], 1)
        self.assertEqual(report["metrics"]["overall"]["action_acc"], 1 / 3)
        self.assertEqual(report["metrics"]["overall"]["text_acc"], 1 / 2)
        self.assertEqual(report["metrics"]["overall"]["icon_acc"], 0.0)

    def test_cli_writes_official_style_log_with_mock_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "imgs"
            test_dir = root / "test"
            output = root / "screenspot_clip_log.json"
            image_dir.mkdir()
            test_dir.mkdir()
            Image.new("RGB", (90, 90), "white").save(image_dir / "sample.png")
            sample = {
                "id": "case-1",
                "img_filename": "sample.png",
                "img_size": [90, 90],
                "bbox": [30, 30, 60, 60],
                "instruction": "click the center button",
                "instruction_cn": "点击中间按钮",
                "group": "Office",
                "platform": "Windows",
                "application": "Excel",
                "ui_type": "icon",
            }
            (test_dir / "office.json").write_text(json.dumps([sample]), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/screenspot_pro_clip_adapter.py",
                    "--screenspot-imgs",
                    str(image_dir),
                    "--screenspot-test",
                    str(test_dir),
                    "--task",
                    "office",
                    "--output",
                    str(output),
                    "--mock-oracle",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["details"][0]["correctness"], "correct")
        self.assertEqual(payload["details"][0]["pred"], [45.0, 45.0])
        self.assertEqual(payload["metrics"]["overall"]["action_acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
