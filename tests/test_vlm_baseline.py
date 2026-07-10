import json
import tempfile
import unittest
from pathlib import Path
from requests.exceptions import ConnectionError

from PIL import Image

from src.multimodal_captcha.vlm_baseline import (
    QwenVlmBaseline,
    parse_cells_from_response,
    score_vlm_predictions,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.text)


class VlmBaselineTests(unittest.TestCase):
    def test_parse_cells_from_plain_json_and_filters_invalid_values(self):
        cells = parse_cells_from_response('{"cells":[3, 1, 9, -1, 3, "2"]}')

        self.assertEqual(cells, [1, 2, 3])

    def test_parse_cells_from_markdown_json_block(self):
        cells = parse_cells_from_response("```json\n{\"cells\":[4, 8]}\n```")

        self.assertEqual(cells, [4, 8])

    def test_qwen_baseline_sends_openai_compatible_vision_request(self):
        calls = []

        def fake_post(url, headers, json, timeout):
            calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
            return FakeResponse({"choices": [{"message": {"content": "{\"cells\":[0,2]}"}}]})

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "grid.jpg"
            Image.new("RGB", (192, 192), "white").save(image_path)
            baseline = QwenVlmBaseline(api_key="secret", model="qwen3-vl-flash", post=fake_post)

            prediction = baseline.predict(image_path, "请点击所有汽车")

        self.assertEqual(prediction.predicted_indices, [0, 2])
        self.assertEqual(calls[0]["url"], "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        self.assertEqual(calls[0]["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(calls[0]["json"]["model"], "qwen3-vl-flash")
        content = calls[0]["json"]["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertEqual(content[1]["type"], "text")
        self.assertIn("九宫格编号", content[1]["text"])

    def test_qwen_baseline_retries_transient_connection_errors(self):
        calls = []

        def flaky_post(url, headers, json, timeout):
            calls.append(url)
            if len(calls) == 1:
                raise ConnectionError("temporary reset")
            return FakeResponse({"choices": [{"message": {"content": "{\"cells\":[5]}"}}]})

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "grid.jpg"
            Image.new("RGB", (192, 192), "white").save(image_path)
            baseline = QwenVlmBaseline(api_key="secret", post=flaky_post, retry_sleep=0.0)

            prediction = baseline.predict(image_path, "请点击所有汽车")

        self.assertEqual(prediction.predicted_indices, [5])
        self.assertEqual(len(calls), 2)

    def test_score_vlm_predictions_matches_click_all_metrics(self):
        metrics = score_vlm_predictions(
            [
                {"target_indices": [1, 3], "predicted_indices": [1, 3], "target_object": "汽车"},
                {"target_indices": [2], "predicted_indices": [2, 4], "target_object": "汽车"},
            ]
        )

        self.assertEqual(metrics["total"], 2)
        self.assertEqual(metrics["cell_exact_match"], 0.5)
        self.assertAlmostEqual(metrics["cell_precision"], 3 / 4)
        self.assertEqual(metrics["cell_recall"], 1.0)
        self.assertEqual(metrics["per_class"]["汽车"]["total"], 2)


if __name__ == "__main__":
    unittest.main()
