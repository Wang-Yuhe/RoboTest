import json
import tempfile
import unittest
from pathlib import Path

from src.multimodal_captcha.prompt_rewriter import (
    DeepSeekPromptRewriter,
    PromptRewriteResult,
    load_supported_objects,
    merge_rewrite_results,
    rule_based_rewrite_prompt,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.text or f"HTTP {self.status_code}")


class PromptRewriterTests(unittest.TestCase):
    def test_load_supported_objects_prefers_selected_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "selected_classes.json").write_text(
                json.dumps(
                    {
                        "classes": [
                            {"class_key": "guitar", "object_name": "吉他"},
                            {"class_key": "car", "object_name": "汽车"},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            objects = load_supported_objects(root)

        self.assertEqual(objects, ["吉他", "汽车"])

    def test_rule_based_rewrite_maps_music_affordance_to_available_instruments(self):
        result = rule_based_rewrite_prompt(
            "请点击可以演奏音乐的物体",
            supported_objects=["汽车", "吉他", "大提琴", "耳机"],
        )

        self.assertEqual(result.provider, "rule")
        self.assertEqual(result.target_objects, ["吉他", "大提琴"])
        self.assertEqual(result.rewritten_prompts, ["请点击所有吉他", "请点击所有大提琴"])

    def test_deepseek_rewriter_filters_unsupported_objects_and_returns_standard_prompts(self):
        calls = []

        def fake_post(url, headers, json, timeout):
            calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
            return FakeResponse(
                payload={
                    "choices": [
                        {
                            "message": {
                                "content": json_module.dumps(
                                    {
                                        "target_objects": ["吉他", "钢琴", "大提琴"],
                                        "reason": "音乐演奏物体",
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ]
                }
            )

        json_module = json
        rewriter = DeepSeekPromptRewriter(api_key="secret", post=fake_post)

        result = rewriter.rewrite(
            "请点击可以演奏音乐的物体",
            supported_objects=["汽车", "吉他", "大提琴"],
        )

        self.assertEqual(result.provider, "deepseek")
        self.assertEqual(result.target_objects, ["吉他", "大提琴"])
        self.assertEqual(result.rewritten_prompts, ["请点击所有吉他", "请点击所有大提琴"])
        self.assertEqual(calls[0]["url"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(calls[0]["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(calls[0]["json"]["response_format"], {"type": "json_object"})

    def test_deepseek_rewriter_falls_back_to_rules_without_api_key(self):
        rewriter = DeepSeekPromptRewriter(api_key=None)

        result = rewriter.rewrite(
            "请点击可以演奏音乐的物体",
            supported_objects=["汽车", "吉他", "大提琴"],
        )

        self.assertEqual(result.provider, "rule")
        self.assertEqual(result.target_objects, ["吉他", "大提琴"])

    def test_merge_rewrite_results_unions_cells_and_keeps_prompt_trace(self):
        merged = merge_rewrite_results(
            original_prompt="请点击可以演奏音乐的物体",
            rewrite=PromptRewriteResult(
                original_prompt="请点击可以演奏音乐的物体",
                target_objects=["吉他", "大提琴"],
                rewritten_prompts=["请点击所有吉他", "请点击所有大提琴"],
                provider="deepseek",
                reason="",
            ),
            per_prompt_predictions=[
                {"prompt": "请点击所有吉他", "predicted_indices": [3, 1]},
                {"prompt": "请点击所有大提琴", "predicted_indices": [1, 8]},
            ],
        )

        self.assertEqual(merged["prompt"], "请点击可以演奏音乐的物体")
        self.assertEqual(merged["predicted_indices"], [1, 3, 8])
        self.assertEqual(merged["rewritten_prompts"], ["请点击所有吉他", "请点击所有大提琴"])


if __name__ == "__main__":
    unittest.main()
