import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.multimodal_captcha import action_checkpoint
from src.multimodal_captcha.action_checkpoint import load_action_checkpoint


class ActionCheckpointTests(unittest.TestCase):
    def test_load_action_checkpoint_rejects_missing_vocab(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model.pt"
            torch.save(
                {
                    "model": {},
                    "object_vocab": {},
                    "model_config": {"architecture": "action_cell_selector"},
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(ValueError, "missing required checkpoint key: vocab"):
                load_action_checkpoint(checkpoint_path, "cpu")

    def test_load_action_checkpoint_rejects_wrong_architecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model.pt"
            torch.save(
                {
                    "model": {},
                    "vocab": {},
                    "object_vocab": {},
                    "model_config": {"architecture": "single_target"},
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(ValueError, "action_cell_selector"):
                load_action_checkpoint(checkpoint_path, "cpu")

    def test_load_action_checkpoint_accepts_valid_action_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model.pt"
            torch.save(
                {
                    "model": {},
                    "vocab": {"<pad>": 0, "<unk>": 1},
                    "object_vocab": {"car": 0},
                    "model_config": {"architecture": "action_cell_selector"},
                },
                checkpoint_path,
            )

            checkpoint, vocab, object_vocab, config = load_action_checkpoint(checkpoint_path, "cpu")

        self.assertIn("model", checkpoint)
        self.assertEqual(vocab["<pad>"], 0)
        self.assertEqual(object_vocab["car"], 0)
        self.assertEqual(config["architecture"], "action_cell_selector")

    def test_build_action_model_does_not_reload_pretrained_weights(self):
        self.assertTrue(
            hasattr(action_checkpoint, "build_action_model_from_checkpoint"),
            "checkpoint module should expose a shared offline-safe model loader",
        )
        captured = {}

        class FakeActionCellSelector:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def to(self, device):
                captured["device"] = device
                return self

            def load_state_dict(self, state_dict):
                captured["state_dict"] = state_dict

            def eval(self):
                captured["evaluated"] = True
                return self

        checkpoint = {
            "model": {"weight": torch.tensor([1.0])},
            "vocab": {"<pad>": 0, "<unk>": 1},
            "object_vocab": {"car": 0},
            "model_config": {
                "architecture": "action_cell_selector",
                "image_encoder": "resnet18",
                "pretrained": True,
                "encoder_train_mode": "frozen",
            },
        }
        with patch.object(action_checkpoint, "load_action_checkpoint", return_value=(
            checkpoint,
            checkpoint["vocab"],
            checkpoint["object_vocab"],
            checkpoint["model_config"],
        )), patch.object(action_checkpoint, "ActionCellSelector", FakeActionCellSelector):
            _, _, _, config = action_checkpoint.build_action_model_from_checkpoint("model.pt", "cpu")

        self.assertEqual(config["pretrained"], True)
        self.assertEqual(captured["image_encoder"], "resnet18")
        self.assertFalse(captured["pretrained"])
        self.assertEqual(captured["device"], "cpu")
        self.assertTrue(captured["evaluated"])


if __name__ == "__main__":
    unittest.main()
