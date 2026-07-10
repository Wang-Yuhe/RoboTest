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


class PredictActionSequenceScriptTests(unittest.TestCase):
    def test_predict_action_sequence_outputs_json_and_visualization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            manifest = generate_action_dataset(data_dir, num_samples=4, seed=11, image_size=192)
            first = json.loads(Path(manifest).read_text(encoding="utf-8").splitlines()[0])
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
            output = root / "prediction.png"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/predict_action_sequence.py",
                    "--image",
                    str(data_dir / first["image"]),
                    "--prompt",
                    first["prompt"],
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            payload = json.loads(result.stdout)
            output_exists = output.exists()

        self.assertTrue(output_exists)
        self.assertEqual(payload["prompt"], first["prompt"])
        self.assertIn("predicted_indices", payload)
        self.assertIn("actions", payload)
        self.assertEqual(payload["saved"], str(output))

    def test_predict_action_sequence_loads_count_head_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            manifest = generate_action_dataset(data_dir, num_samples=4, seed=13, image_size=192)
            first = json.loads(Path(manifest).read_text(encoding="utf-8").splitlines()[0])
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
            output = root / "prediction.png"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/predict_action_sequence.py",
                    "--image",
                    str(data_dir / first["image"]),
                    "--prompt",
                    first["prompt"],
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(output),
                    "--decode-policy",
                    "topk_count",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            payload = json.loads(result.stdout)
            output_exists = output.exists()

        self.assertIn("predicted_indices", payload)
        self.assertEqual(payload["decode_policy"], "topk_count")
        self.assertTrue(output_exists)


if __name__ == "__main__":
    unittest.main()
