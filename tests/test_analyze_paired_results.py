import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class AnalyzePairedResultsTests(unittest.TestCase):
    def test_analyze_paired_results_maps_classes_thresholds_and_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirty_eval = root / "dirty_eval" / "test"
            clean_eval = root / "clean_eval" / "test"
            clean_data = root / "clean_data"
            output_dir = root / "analysis"
            dirty_eval.mkdir(parents=True)
            clean_eval.mkdir(parents=True)
            clean_data.mkdir()
            (clean_data / "selected_classes.json").write_text(
                json.dumps(
                    {
                        "classes": [
                            {"class_key": "tent", "object_name": "tent", "source_images": 75},
                            {"class_key": "person", "object_name": "person", "source_images": 66},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            dirty_metrics = {
                "threshold": 0.5,
                "cell_exact_match": 0.4,
                "cell_precision": 0.8,
                "cell_recall": 0.7,
                "click_order_accuracy": 0.4,
                "threshold_candidates": [
                    {"threshold": 0.3, "cell_exact_match": 0.3, "cell_precision": 0.7, "cell_recall": 0.8},
                    {"threshold": 0.5, "cell_exact_match": 0.4, "cell_precision": 0.8, "cell_recall": 0.7},
                ],
                "per_class": {
                    "tent": {"total": 25, "cell_exact_match": 0.2, "cell_recall": 0.6},
                    "person": {"total": 25, "cell_exact_match": 0.1, "cell_recall": 0.5},
                },
                "failures": [
                    {"target_object": "tent", "target_indices": [0, 1], "predicted_indices": [0]},
                    {"target_object": "person", "target_indices": [2], "predicted_indices": [2, 3]},
                    {"target_object": "person", "target_indices": [4], "predicted_indices": [5]},
                ],
            }
            clean_metrics = {
                "threshold": 0.3,
                "cell_exact_match": 0.45,
                "cell_precision": 0.82,
                "cell_recall": 0.76,
                "click_order_accuracy": 0.45,
                "threshold_candidates": [
                    {"threshold": 0.3, "cell_exact_match": 0.45, "cell_precision": 0.82, "cell_recall": 0.76},
                    {"threshold": 0.5, "cell_exact_match": 0.42, "cell_precision": 0.9, "cell_recall": 0.7},
                ],
                "per_class": {
                    "tent": {"total": 25, "cell_exact_match": 0.32, "cell_recall": 0.7},
                    "person": {"total": 25, "cell_exact_match": 0.16, "cell_recall": 0.55},
                },
                "failures": [
                    {"target_object": "tent", "target_indices": [0], "predicted_indices": [0, 1]},
                ],
            }
            (dirty_eval / "metrics.json").write_text(json.dumps(dirty_metrics), encoding="utf-8")
            (clean_eval / "metrics.json").write_text(json.dumps(clean_metrics), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/analyze_paired_results.py",
                    "--dirty-metrics",
                    str(dirty_eval / "metrics.json"),
                    "--clean-metrics",
                    str(clean_eval / "metrics.json"),
                    "--selected-classes",
                    str(clean_data / "selected_classes.json"),
                    "--output-dir",
                    str(output_dir),
                    "--focus-classes",
                    "tent,person",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            summary = json.loads((output_dir / "paired_analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["focus_classes"]["tent"]["object_name"], "tent")
            self.assertEqual(summary["focus_classes"]["tent"]["dirty_cell_exact_match"], 0.2)
            self.assertEqual(summary["focus_classes"]["tent"]["clean_cell_exact_match"], 0.32)
            self.assertAlmostEqual(summary["focus_classes"]["tent"]["delta_cell_exact_match"], 0.12)
            self.assertEqual(summary["thresholds"]["dirty"]["best_exact_match"]["threshold"], 0.5)
            self.assertEqual(summary["thresholds"]["clean"]["best_exact_match"]["threshold"], 0.3)
            self.assertEqual(summary["failure_summary"]["dirty"]["error_types"]["missing_targets"], 1)
            self.assertEqual(summary["failure_summary"]["dirty"]["error_types"]["extra_targets"], 1)
            self.assertEqual(summary["failure_summary"]["dirty"]["error_types"]["mixed_or_wrong"], 1)
            self.assertTrue((output_dir / "paired_analysis.md").exists())


if __name__ == "__main__":
    unittest.main()
