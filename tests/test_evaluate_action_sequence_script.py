import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from src.multimodal_captcha.dataset import build_object_vocab, build_vocab
from src.multimodal_captcha.generator import generate_action_dataset
from src.multimodal_captcha.model import ActionCellSelector
from scripts.evaluate_action_sequence import DEFAULT_THRESHOLD_GRID, parse_threshold_grid, score_predictions
from src.multimodal_captcha.action_sequence import actions_to_tokens, target_indices_to_actions


class EvaluateActionSequenceScriptTests(unittest.TestCase):
    def test_default_threshold_grid_includes_low_recall_friendly_values(self):
        grid = parse_threshold_grid(DEFAULT_THRESHOLD_GRID)

        self.assertIn(0.03, grid)
        self.assertIn(0.05, grid)
        self.assertIn(0.07, grid)
        self.assertIn(0.1, grid)
        self.assertLess(grid[0], 0.1)

    def test_score_predictions_can_decode_with_predicted_count_topk(self):
        logits = torch.tensor([[0.2, 3.0, -1.0, 2.0, 0.1, -0.2, 4.0, 0.0, -2.0]])
        targets = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        count_logits = torch.tensor([[-2.0, -1.0, 0.0, 5.0, 1.0]])
        records = [{"target_object": "car", "target_indices": [1, 3, 6]}]
        action_targets = [actions_to_tokens(target_indices_to_actions([1, 3, 6]))]

        metrics = score_predictions(
            logits,
            targets,
            action_targets,
            records,
            threshold=0.95,
            decode_policy="topk_count",
            count_logits=count_logits,
            max_count=4,
        )

        self.assertEqual(metrics["cell_exact_match"], 1.0)
        self.assertEqual(metrics["cell_precision"], 1.0)
        self.assertEqual(metrics["cell_recall"], 1.0)
        self.assertEqual(metrics["count_accuracy"], 1.0)
        self.assertEqual(metrics["predicted_count_histogram"], {"3": 1})

    def test_evaluate_action_sequence_writes_metrics_and_failure_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            manifest = generate_action_dataset(data_dir, num_samples=16, seed=17, image_size=192)
            vocab = build_vocab(manifest)
            object_vocab = build_object_vocab(manifest)
            model = ActionCellSelector(vocab_size=len(vocab), object_vocab_size=len(object_vocab))
            checkpoint = root / "action_model.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "vocab": vocab,
                    "object_vocab": object_vocab,
                    "model_config": {
                        "architecture": "action_cell_selector",
                        "vocab_size": len(vocab),
                        "object_vocab_size": len(object_vocab),
                        "max_action_len": 10,
                        "hidden_dim": 96,
                        "image_size": 64,
                        "base_channels": 24,
                        "model_size": "small",
                    },
                },
                checkpoint,
            )
            output_dir = root / "eval"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_action_sequence.py",
                    "--data-dir",
                    str(data_dir),
                    "--checkpoint",
                    str(checkpoint),
                    "--split",
                    "val",
                    "--output-dir",
                    str(output_dir),
                    "--threshold",
                    "0.5",
                    "--max-failures",
                    "2",
                    "--max-samples",
                    "2",
                    "--device",
                    "cpu",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            metrics_path = output_dir / "val" / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            failures = list((output_dir / "val" / "failures").glob("*.png"))

        self.assertEqual(metrics["split"], "val")
        self.assertEqual(metrics["total"], 2)
        self.assertEqual(metrics["threshold"], 0.5)
        self.assertEqual(metrics["threshold_source"], "fixed")
        self.assertIn("cell_exact_match", metrics)
        self.assertIn("click_order_accuracy", metrics)
        self.assertIn("cell_precision", metrics)
        self.assertIn("cell_recall", metrics)
        self.assertIn("per_class", metrics)
        self.assertLessEqual(len(failures), 2)

    def test_evaluate_action_sequence_can_select_threshold_on_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            manifest = generate_action_dataset(data_dir, num_samples=16, seed=23, image_size=192)
            vocab = build_vocab(manifest)
            object_vocab = build_object_vocab(manifest)
            model = ActionCellSelector(vocab_size=len(vocab), object_vocab_size=len(object_vocab))
            checkpoint = root / "action_model.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "vocab": vocab,
                    "object_vocab": object_vocab,
                    "model_config": {
                        "architecture": "action_cell_selector",
                        "vocab_size": len(vocab),
                        "object_vocab_size": len(object_vocab),
                        "max_action_len": 10,
                        "hidden_dim": 96,
                        "image_size": 64,
                        "base_channels": 24,
                        "model_size": "small",
                    },
                },
                checkpoint,
            )
            output_dir = root / "eval"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_action_sequence.py",
                    "--data-dir",
                    str(data_dir),
                    "--checkpoint",
                    str(checkpoint),
                    "--split",
                    "test",
                    "--output-dir",
                    str(output_dir),
                    "--threshold",
                    "auto",
                    "--threshold-grid",
                    "0.3,0.5,0.7",
                    "--max-failures",
                    "0",
                    "--device",
                    "cpu",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            metrics = json.loads((output_dir / "test" / "metrics.json").read_text(encoding="utf-8"))

        self.assertIn(metrics["threshold"], [0.3, 0.5, 0.7])
        self.assertEqual(metrics["threshold_source"], "auto_val")
        self.assertIn("threshold_candidates", metrics)

    def test_evaluate_action_sequence_can_use_count_head_decode_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            manifest = generate_action_dataset(data_dir, num_samples=16, seed=29, image_size=192)
            vocab = build_vocab(manifest)
            object_vocab = build_object_vocab(manifest)
            model = ActionCellSelector(vocab_size=len(vocab), object_vocab_size=len(object_vocab), use_count_head=True)
            checkpoint = root / "action_model.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "vocab": vocab,
                    "object_vocab": object_vocab,
                    "model_config": {
                        "architecture": "action_cell_selector",
                        "vocab_size": len(vocab),
                        "object_vocab_size": len(object_vocab),
                        "max_action_len": 10,
                        "hidden_dim": 96,
                        "image_size": 64,
                        "base_channels": 24,
                        "model_size": "small",
                        "use_count_head": True,
                        "max_count": 4,
                    },
                },
                checkpoint,
            )
            output_dir = root / "eval"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_action_sequence.py",
                    "--data-dir",
                    str(data_dir),
                    "--checkpoint",
                    str(checkpoint),
                    "--split",
                    "val",
                    "--output-dir",
                    str(output_dir),
                    "--decode-policy",
                    "topk_count",
                    "--max-failures",
                    "0",
                    "--device",
                    "cpu",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            metrics = json.loads((output_dir / "val" / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(metrics["decode_policy"], "topk_count")
        self.assertIn("cell_exact_match", metrics)

    def test_evaluate_action_sequence_fails_clearly_when_split_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            manifest = generate_action_dataset(data_dir, num_samples=8, seed=31, image_size=192)
            records = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines()]
            with Path(manifest).open("w", encoding="utf-8") as f:
                for record in records:
                    record["split"] = "train"
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            vocab = build_vocab(manifest)
            object_vocab = build_object_vocab(manifest)
            model = ActionCellSelector(vocab_size=len(vocab), object_vocab_size=len(object_vocab))
            checkpoint = root / "action_model.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "vocab": vocab,
                    "object_vocab": object_vocab,
                    "model_config": {
                        "architecture": "action_cell_selector",
                        "vocab_size": len(vocab),
                        "object_vocab_size": len(object_vocab),
                        "max_action_len": 10,
                        "hidden_dim": 96,
                        "image_size": 64,
                        "base_channels": 24,
                        "model_size": "small",
                    },
                },
                checkpoint,
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_action_sequence.py",
                    "--data-dir",
                    str(data_dir),
                    "--checkpoint",
                    str(checkpoint),
                    "--split",
                    "test",
                    "--output-dir",
                    str(root / "eval"),
                    "--device",
                    "cpu",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No records found for split 'test'", result.stderr + result.stdout)

    def test_evaluate_action_sequence_fails_clearly_when_auto_threshold_val_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            manifest = generate_action_dataset(data_dir, num_samples=8, seed=37, image_size=192)
            records = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines()]
            with Path(manifest).open("w", encoding="utf-8") as f:
                for idx, record in enumerate(records):
                    record["split"] = "test" if idx < 4 else "train"
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            vocab = build_vocab(manifest)
            object_vocab = build_object_vocab(manifest)
            model = ActionCellSelector(vocab_size=len(vocab), object_vocab_size=len(object_vocab))
            checkpoint = root / "action_model.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "vocab": vocab,
                    "object_vocab": object_vocab,
                    "model_config": {
                        "architecture": "action_cell_selector",
                        "vocab_size": len(vocab),
                        "object_vocab_size": len(object_vocab),
                        "max_action_len": 10,
                        "hidden_dim": 96,
                        "image_size": 64,
                        "base_channels": 24,
                        "model_size": "small",
                    },
                },
                checkpoint,
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_action_sequence.py",
                    "--data-dir",
                    str(data_dir),
                    "--checkpoint",
                    str(checkpoint),
                    "--split",
                    "test",
                    "--threshold",
                    "auto",
                    "--output-dir",
                    str(root / "eval"),
                    "--device",
                    "cpu",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No records found for split 'val'", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
