import subprocess
import sys
import tempfile
import unittest
import random
from pathlib import Path

import numpy as np
import torch

from scripts import train_action_sequence
from scripts.train_action_sequence import resolve_encoder_train_mode, should_update_best_checkpoint


class ActionTrainingScriptTests(unittest.TestCase):
    def test_should_update_best_checkpoint_uses_loss_tiebreaker(self):
        self.assertTrue(should_update_best_checkpoint(score=0.2, loss=5.0, best_score=0.1, best_loss=1.0))
        self.assertTrue(should_update_best_checkpoint(score=0.0, loss=2.0, best_score=0.0, best_loss=3.0))
        self.assertFalse(should_update_best_checkpoint(score=0.0, loss=4.0, best_score=0.0, best_loss=3.0))

    def test_resolve_encoder_train_mode_prefers_frozen_for_pretrained_resnet18(self):
        self.assertEqual(resolve_encoder_train_mode("auto", "resnet18", True), "frozen")
        self.assertEqual(resolve_encoder_train_mode("auto", "resnet18", False), "full")
        self.assertEqual(resolve_encoder_train_mode("auto", "custom", False), "full")
        self.assertEqual(resolve_encoder_train_mode("last_block", "resnet18", True), "last_block")

    def test_resolve_encoder_train_mode_prefers_frozen_for_pretrained_clip(self):
        self.assertEqual(resolve_encoder_train_mode("auto", "clip_vit_b32", True), "frozen")
        self.assertEqual(resolve_encoder_train_mode("auto", "clip_vit_b32", False), "full")

    def test_seed_everything_covers_python_numpy_and_torch(self):
        self.assertTrue(hasattr(train_action_sequence, "seed_everything"))

        train_action_sequence.seed_everything(23)
        first = (random.random(), float(np.random.random()), float(torch.rand(1).item()))
        train_action_sequence.seed_everything(23)
        second = (random.random(), float(np.random.random()), float(torch.rand(1).item()))

        self.assertEqual(first, second)

    def test_train_action_sequence_script_creates_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "action_model.pt"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/train_action_sequence.py",
                    "--data-dir",
                    str(root / "data"),
                    "--output",
                    str(output),
                    "--num-samples",
                    "24",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "8",
                    "--max-action-len",
                    "10",
                    "--image-encoder",
                    "custom",
                    "--no-augment",
                    "--progress-every",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            self.assertTrue(output.exists())
            checkpoint = torch.load(output, map_location="cpu", weights_only=True)
            self.assertTrue(output.with_suffix(".action_vocab.json").exists())
            self.assertTrue(output.with_suffix(".vocab.json").exists())
            self.assertTrue(output.with_suffix(".object_vocab.json").exists())

        self.assertIn("model", checkpoint)
        self.assertIn("vocab", checkpoint)
        self.assertIn("object_vocab", checkpoint)
        self.assertGreater(len(checkpoint["object_vocab"]), 0)
        self.assertEqual(checkpoint["model_config"]["max_action_len"], 10)
        self.assertEqual(checkpoint["model_config"]["architecture"], "action_cell_selector")
        self.assertEqual(checkpoint["model_config"]["image_encoder"], "custom")
        self.assertFalse(checkpoint["model_config"]["pretrained"])
        self.assertEqual(checkpoint["training_args"]["epochs"], 1)
        self.assertEqual(checkpoint["training_args"]["batch_size"], 8)
        self.assertEqual(checkpoint["training_args"]["image_encoder"], "custom")
        self.assertTrue(checkpoint["training_args"]["no_augment"])
        self.assertIn("best_cell_exact_match", checkpoint)

    def test_train_action_sequence_script_can_limit_smoke_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "action_model.pt"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/train_action_sequence.py",
                    "--data-dir",
                    str(root / "data"),
                    "--output",
                    str(output),
                    "--num-samples",
                    "40",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--max-train-samples",
                    "8",
                    "--max-val-samples",
                    "4",
                    "--no-augment",
                    "--progress-every",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            checkpoint = torch.load(output, map_location="cpu", weights_only=True)

        self.assertEqual(checkpoint["training_args"]["max_train_samples"], 8)
        self.assertEqual(checkpoint["training_args"]["max_val_samples"], 4)

    def test_train_action_sequence_script_can_enable_count_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "action_model.pt"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/train_action_sequence.py",
                    "--data-dir",
                    str(root / "data"),
                    "--output",
                    str(output),
                    "--num-samples",
                    "24",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "8",
                    "--use-count-head",
                    "--count-loss-weight",
                    "0.3",
                    "--no-augment",
                    "--progress-every",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            checkpoint = torch.load(output, map_location="cpu", weights_only=True)

        self.assertTrue(checkpoint["model_config"]["use_count_head"])
        self.assertEqual(checkpoint["model_config"]["max_count"], 4)
        self.assertEqual(checkpoint["training_args"]["count_loss_weight"], 0.3)


if __name__ == "__main__":
    unittest.main()
