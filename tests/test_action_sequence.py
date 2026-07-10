import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.multimodal_captcha.action_sequence import (
    actions_to_tokens,
    cell_logits_to_topk_actions,
    clicked_cells_from_tokens,
    compute_action_metrics,
    target_indices_to_actions,
    tokens_to_actions,
)
from src.multimodal_captcha.generator import generate_action_dataset


class ActionSequenceTests(unittest.TestCase):
    def test_target_indices_become_click_all_action_sequence(self):
        actions = target_indices_to_actions([1, 4, 7])

        self.assertEqual(
            actions,
            [
                {"type": "move_to_cell", "cell": 1},
                {"type": "click"},
                {"type": "move_to_cell", "cell": 4},
                {"type": "click"},
                {"type": "move_to_cell", "cell": 7},
                {"type": "click"},
                {"type": "done"},
            ],
        )
        self.assertEqual(tokens_to_actions(actions_to_tokens(actions)), actions)

    def test_generate_action_dataset_creates_click_all_same_class_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = generate_action_dataset(tmp, num_samples=8, seed=3, image_size=192)
            records = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(records), 8)
        for record in records:
            self.assertIn("请点击所有", record["prompt"])
            self.assertGreaterEqual(len(record["target_indices"]), 2)
            target_object = record["target_object"]
            for index in record["target_indices"]:
                self.assertEqual(record["items"][index]["object_name"], target_object)
            self.assertEqual(record["actions"], target_indices_to_actions(record["target_indices"]))
            self.assertTrue(Path(tmp, record["image"]).suffix == ".png")

    def test_generated_action_dataset_images_are_valid_grid_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = generate_action_dataset(tmp, num_samples=2, seed=9, image_size=192)
            first = json.loads(Path(manifest).read_text(encoding="utf-8").splitlines()[0])
            with Image.open(Path(tmp) / first["image"]) as image:
                size = image.size

        self.assertEqual(size, (192, 192))

    def test_action_metrics_track_exact_and_clicked_cell_order(self):
        expected = actions_to_tokens(target_indices_to_actions([0, 3]))
        correct = actions_to_tokens(target_indices_to_actions([0, 3]))
        wrong_order = actions_to_tokens(
            [
                {"type": "move_to_cell", "cell": 3},
                {"type": "click"},
                {"type": "move_to_cell", "cell": 0},
                {"type": "click"},
                {"type": "done"},
            ]
        )

        self.assertEqual(clicked_cells_from_tokens(correct), [0, 3])
        self.assertEqual(clicked_cells_from_tokens(wrong_order), [3, 0])
        self.assertEqual(
            compute_action_metrics([correct, wrong_order], [expected, expected]),
            {"exact_match": 0.5, "click_order_accuracy": 0.5},
        )

    def test_cell_logits_to_topk_actions_uses_predicted_count(self):
        logits = torch.tensor(
            [
                [0.1, 2.0, -1.0, 1.5, 0.0, -0.5, 3.0, -2.0, 0.3],
                [4.0, -1.0, 2.5, 0.1, 1.0, 3.0, -0.2, 0.0, 2.0],
            ]
        )
        count_logits = torch.tensor(
            [
                [-4.0, -2.0, 3.0, 0.5, -1.0],
                [-4.0, -2.0, 0.1, 0.2, 4.0],
            ]
        )

        actions = cell_logits_to_topk_actions(logits, count_logits, min_count=1, max_count=4)

        self.assertEqual(actions[0], target_indices_to_actions([1, 6]))
        self.assertEqual(actions[1], target_indices_to_actions([0, 2, 5, 8]))


if __name__ == "__main__":
    unittest.main()
