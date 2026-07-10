import tempfile
import unittest
from pathlib import Path

import torch

from src.multimodal_captcha.action_sequence import (
    ACTION_VOCAB_SIZE,
    DONE_TOKEN,
    IGNORE_INDEX,
    cell_logits_to_actions,
    target_indices_to_actions,
)
from src.multimodal_captcha.dataset import ActionSequenceDataset, build_object_vocab, build_vocab
from src.multimodal_captcha.generator import generate_action_dataset
from src.multimodal_captcha.model import ActionCellSelector, ActionSequenceLocator


class ActionTrainingComponentTests(unittest.TestCase):
    def test_action_sequence_dataset_returns_image_text_and_action_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = generate_action_dataset(tmp, num_samples=12, seed=5, image_size=192)
            vocab = build_vocab(manifest)
            object_vocab = build_object_vocab(manifest)
            dataset = ActionSequenceDataset(tmp, split="train", vocab=vocab, object_vocab=object_vocab, max_action_len=10)
            sample = dataset[0]

        self.assertEqual(sample["image"].shape, (3, 192, 192))
        self.assertEqual(sample["text"].shape[0], 20)
        self.assertEqual(sample["action_targets"].shape[0], 10)
        self.assertIn(DONE_TOKEN, sample["action_targets"].tolist())
        self.assertEqual(sample["action_targets"].tolist()[-1], IGNORE_INDEX)
        self.assertEqual(sample["cell_targets"].shape, (9,))
        self.assertEqual(int(sample["cell_targets"].sum().item()), len(dataset.records[0]["target_indices"]))
        self.assertEqual(sample["object_ids"].shape, (9,))
        for index in dataset.records[0]["target_indices"]:
            self.assertEqual(float(sample["cell_targets"][index].item()), 1.0)
            self.assertGreaterEqual(int(sample["object_ids"][index].item()), 0)

    def test_action_sequence_locator_outputs_logits_per_action_step(self):
        model = ActionSequenceLocator(vocab_size=16, max_action_len=10)
        image = torch.rand(2, 3, 192, 192)
        text = torch.randint(0, 16, (2, 20))

        logits = model(image, text)

        self.assertEqual(logits.shape, (2, 10, ACTION_VOCAB_SIZE))

    def test_action_cell_selector_outputs_one_logit_per_grid_cell(self):
        model = ActionCellSelector(vocab_size=16, object_vocab_size=9)
        image = torch.rand(2, 3, 192, 192)
        text = torch.randint(0, 16, (2, 20))

        logits, object_logits = model(image, text, return_aux=True)

        self.assertEqual(logits.shape, (2, 9))
        self.assertEqual(object_logits.shape, (2, 9, 9))

    def test_action_cell_selector_can_return_count_logits(self):
        model = ActionCellSelector(vocab_size=16, object_vocab_size=9, use_count_head=True, max_count=4)
        image = torch.rand(2, 3, 192, 192)
        text = torch.randint(0, 16, (2, 20))

        logits, object_logits, count_logits = model(image, text, return_aux=True)

        self.assertEqual(logits.shape, (2, 9))
        self.assertEqual(object_logits.shape, (2, 9, 9))
        self.assertEqual(count_logits.shape, (2, 5))

    def test_action_cell_selector_can_use_resnet18_cell_encoder(self):
        model = ActionCellSelector(vocab_size=16, object_vocab_size=9, image_encoder="resnet18", pretrained=False)
        image = torch.rand(2, 3, 192, 192)
        text = torch.randint(0, 16, (2, 20))

        logits, object_logits = model(image, text, return_aux=True)

        self.assertEqual(logits.shape, (2, 9))
        self.assertEqual(object_logits.shape, (2, 9, 9))

    def test_action_cell_selector_can_use_clip_vision_cell_encoder(self):
        model = ActionCellSelector(
            vocab_size=16,
            object_vocab_size=9,
            image_encoder="clip_vit_b32",
            pretrained=False,
            encoder_train_mode="frozen",
        )
        image = torch.rand(1, 3, 192, 192)
        text = torch.randint(0, 16, (1, 20))

        logits, object_logits = model(image, text, return_aux=True)

        self.assertEqual(logits.shape, (1, 9))
        self.assertEqual(object_logits.shape, (1, 9, 9))
        self.assertFalse(any(parameter.requires_grad for parameter in model.encoder.image_encoder.parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in model.encoder.image_projection.parameters()))

    def test_action_cell_selector_can_freeze_resnet18_backbone(self):
        model = ActionCellSelector(
            vocab_size=16,
            object_vocab_size=9,
            image_encoder="resnet18",
            pretrained=False,
            encoder_train_mode="frozen",
        )

        self.assertFalse(any(parameter.requires_grad for parameter in model.encoder.image_encoder.parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in model.encoder.image_projection.parameters()))

        model.train()

        self.assertFalse(model.encoder.image_encoder.training)

    def test_action_cell_selector_can_train_only_resnet18_last_block(self):
        model = ActionCellSelector(
            vocab_size=16,
            object_vocab_size=9,
            image_encoder="resnet18",
            pretrained=False,
            encoder_train_mode="last_block",
        )
        image_encoder_children = list(model.encoder.image_encoder.children())

        self.assertFalse(any(parameter.requires_grad for module in image_encoder_children[:7] for parameter in module.parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in image_encoder_children[7].parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in model.encoder.image_projection.parameters()))

    def test_cell_logits_convert_to_click_all_actions(self):
        logits = torch.tensor([[-4.0, 3.0, -2.0, 2.5, -1.0, -3.0, 4.0, -2.0, -5.0]])

        actions = cell_logits_to_actions(logits, threshold=0.5)

        self.assertEqual(actions[0], target_indices_to_actions([1, 3, 6]))


if __name__ == "__main__":
    unittest.main()
