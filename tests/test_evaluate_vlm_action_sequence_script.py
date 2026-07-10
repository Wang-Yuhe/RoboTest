import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts import evaluate_vlm_action_sequence
from src.multimodal_captcha.generator import generate_action_dataset


class EvaluateVlmActionSequenceScriptTests(unittest.TestCase):
    def test_resume_config_rejects_changed_model_or_dataset_signature(self):
        self.assertTrue(hasattr(evaluate_vlm_action_sequence, "build_run_signature"))
        self.assertTrue(hasattr(evaluate_vlm_action_sequence, "validate_resume_config"))
        records = [{"image": "images/a.jpg", "prompt": "请点击所有汽车", "target_indices": [1, 3]}]
        expected = evaluate_vlm_action_sequence.build_run_signature(
            data_dir=Path("data/example"),
            split="test",
            provider="qwen",
            model="qwen3-vl-flash",
            base_url="https://example.test/v1",
            records=records,
        )

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "run_config.json"
            config_path.write_text(json.dumps(expected), encoding="utf-8")
            evaluate_vlm_action_sequence.validate_resume_config(config_path, expected)
            changed = {**expected, "model": "qwen3-vl-plus"}

            with self.assertRaisesRegex(ValueError, "resume configuration mismatch"):
                evaluate_vlm_action_sequence.validate_resume_config(config_path, changed)

    def test_evaluate_vlm_action_sequence_writes_metrics_predictions_and_failures_with_mock_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            generate_action_dataset(data_dir, num_samples=12, seed=41, image_size=192)
            output_dir = root / "vlm_eval"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_vlm_action_sequence.py",
                    "--data-dir",
                    str(data_dir),
                    "--split",
                    "val",
                    "--output-dir",
                    str(output_dir),
                    "--mock-oracle",
                    "--max-samples",
                    "3",
                    "--max-failures",
                    "2",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            metrics = json.loads((output_dir / "val" / "metrics.json").read_text(encoding="utf-8"))
            predictions = (output_dir / "val" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            run_config = json.loads((output_dir / "val" / "run_config.json").read_text(encoding="utf-8"))

        self.assertEqual(metrics["provider"], "mock_oracle")
        self.assertEqual(metrics["total"], 3)
        self.assertEqual(metrics["cell_exact_match"], 1.0)
        self.assertEqual(len(predictions), 3)
        self.assertEqual(run_config["provider"], "mock_oracle")
        self.assertIn("records_sha256", run_config)

    def test_evaluate_vlm_action_sequence_requires_api_key_without_mock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            generate_action_dataset(data_dir, num_samples=8, seed=43, image_size=192)

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_vlm_action_sequence.py",
                    "--data-dir",
                    str(data_dir),
                    "--split",
                    "val",
                    "--output-dir",
                    str(root / "vlm_eval"),
                    "--api-key",
                    "",
                    "--max-samples",
                    "1",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DASHSCOPE_API_KEY", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
