import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.multimodal_captcha import streamlit_action_demo
from src.multimodal_captcha.streamlit_action_demo import (
    build_turnstile_widget_html,
    load_action_demo_records,
    predict_action_demo,
    resolve_first_existing_path,
    sync_action_prompt_state,
)


class FixedActionModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor([logits], dtype=torch.float32))

    def forward(self, image, text):
        return self.logits.to(image.device)


class StreamlitActionDemoTests(unittest.TestCase):
    def test_resolve_first_existing_path_returns_first_existing_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "missing.pt"
            second = root / "model.pt"
            second.write_text("checkpoint", encoding="utf-8")

            resolved = resolve_first_existing_path([first, second])

        self.assertEqual(resolved, second)

    def test_resolve_first_existing_path_returns_none_when_no_candidate_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            resolved = resolve_first_existing_path([root / "a.pt", root / "b.pt"])

        self.assertIsNone(resolved)

    def test_load_action_demo_records_filters_split_and_limits_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            records = [
                {"split": "train", "image": "images/a.jpg", "prompt": "train", "target_indices": [0]},
                {"split": "test", "image": "images/b.jpg", "prompt": "test1", "target_indices": [1]},
                {"split": "test", "image": "images/c.jpg", "prompt": "test2", "target_indices": [2]},
            ]
            (root / "manifest.jsonl").write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
                encoding="utf-8",
            )

            loaded = load_action_demo_records(root, split="test", limit=1)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["prompt"], "test1")

    def test_predict_action_demo_decodes_cells_and_marks_correctness(self):
        image = Image.new("RGB", (192, 192), "white")
        model = FixedActionModel([-2.0, 3.0, -2.0, 2.5, -2.0, -2.0, -2.0, -2.0, -2.0])
        vocab = {"<pad>": 0, "<unk>": 1, "请": 2, "点": 3, "车": 4}

        result = predict_action_demo(
            model=model,
            vocab=vocab,
            config={},
            image=image,
            prompt="请点车",
            target_indices=[1, 3],
            threshold=0.5,
            device="cpu",
            seed=7,
        )

        self.assertEqual(result.predicted_indices, [1, 3])
        self.assertTrue(result.correct)
        self.assertEqual(len(result.cell_probabilities), 9)
        self.assertEqual(result.visualization.size, image.size)

    def test_build_turnstile_widget_html_uses_cloudflare_test_key(self):
        html = build_turnstile_widget_html()

        self.assertIn("https://challenges.cloudflare.com/turnstile/v0/api.js", html)
        self.assertIn("1x00000000000000000000AA", html)

    def test_sync_action_prompt_state_preserves_manual_prompt_until_sample_changes(self):
        state = {}

        first_prompt = sync_action_prompt_state(state, record_key="test:a.jpg", default_prompt="请点击所有汽车")
        self.assertEqual(first_prompt, "请点击所有汽车")

        state["action_prompt"] = "请点击所有公交车"
        same_record_prompt = sync_action_prompt_state(state, record_key="test:a.jpg", default_prompt="请点击所有汽车")
        self.assertEqual(same_record_prompt, "请点击所有公交车")

        next_record_prompt = sync_action_prompt_state(state, record_key="test:b.jpg", default_prompt="请点击所有帐篷")
        self.assertEqual(next_record_prompt, "请点击所有帐篷")

    def test_paid_request_runs_only_on_explicit_trigger_and_reuses_matching_cache(self):
        self.assertTrue(hasattr(streamlit_action_demo, "explicit_cached_request"))
        state = {}
        calls = []

        def request():
            calls.append("called")
            return {"cells": [1, 3]}

        first = streamlit_action_demo.explicit_cached_request(
            state, "vlm_cache", ("sample-a", "prompt-a"), False, request
        )
        triggered = streamlit_action_demo.explicit_cached_request(
            state, "vlm_cache", ("sample-a", "prompt-a"), True, request
        )
        cached = streamlit_action_demo.explicit_cached_request(
            state, "vlm_cache", ("sample-a", "prompt-a"), False, request
        )
        stale = streamlit_action_demo.explicit_cached_request(
            state, "vlm_cache", ("sample-b", "prompt-a"), False, request
        )

        self.assertIsNone(first)
        self.assertEqual(triggered, {"cells": [1, 3]})
        self.assertEqual(cached, triggered)
        self.assertIsNone(stale)
        self.assertEqual(calls, ["called"])


if __name__ == "__main__":
    unittest.main()
