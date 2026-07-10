from __future__ import annotations

import base64
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import requests


QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen3-vl-flash"


@dataclass(frozen=True)
class VlmPrediction:
    provider: str
    model: str
    prompt: str
    predicted_indices: list[int]
    raw_response: str


def image_to_data_uri(image_path: str | Path) -> str:
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1)
    else:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    return json.loads(stripped)


def parse_cells_from_response(text: str) -> list[int]:
    payload = extract_json_object(text)
    values = payload.get("cells", [])
    cells = []
    seen = set()
    for value in values:
        try:
            cell = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= cell <= 8 and cell not in seen:
            seen.add(cell)
            cells.append(cell)
    return sorted(cells)


class QwenVlmBaseline:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = QWEN_BASE_URL,
        model: str = QWEN_MODEL,
        timeout: int = 60,
        max_retries: int = 3,
        retry_sleep: float = 1.0,
        post: Callable[..., Any] = requests.post,
    ):
        self.api_key = api_key if api_key is not None else os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.retry_sleep = max(0.0, float(retry_sleep))
        self.post = post

    def predict(self, image_path: str | Path, prompt: str) -> VlmPrediction:
        if not self.api_key:
            raise RuntimeError("Qwen VLM baseline requires DASHSCOPE_API_KEY or --api-key.")
        request_payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_to_data_uri(image_path)}},
                        {"type": "text", "text": build_grid_instruction(prompt)},
                    ],
                }
            ],
            "temperature": 0,
        }
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_payload,
                    timeout=self.timeout,
                )
                break
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_sleep * attempt)
        else:
            raise RuntimeError(f"Qwen VLM request failed: {last_error}")
        response.raise_for_status()
        payload = response.json()
        raw = payload["choices"][0]["message"]["content"]
        return VlmPrediction(
            provider="qwen",
            model=self.model,
            prompt=prompt,
            predicted_indices=parse_cells_from_response(raw),
            raw_response=raw,
        )


def build_grid_instruction(prompt: str) -> str:
    return (
        "你是九宫格视觉定位模型。请根据图片和中文指令判断需要点击哪些格子。\n"
        "九宫格编号固定为：\n"
        "0 1 2\n"
        "3 4 5\n"
        "6 7 8\n\n"
        f"指令：{prompt}\n\n"
        "只输出 JSON，不要解释，格式必须是：{\"cells\":[0,1]}"
    )


def empty_class_stat() -> dict[str, int]:
    return {
        "total": 0,
        "cell_exact": 0,
        "true_positive": 0,
        "predicted_positive": 0,
        "target_positive": 0,
    }


def finalize_binary_metrics(stats: dict[str, int]) -> dict[str, float | int]:
    total = max(int(stats["total"]), 1)
    return {
        "total": int(stats["total"]),
        "cell_exact_match": stats["cell_exact"] / total,
        "cell_precision": stats["true_positive"] / max(int(stats["predicted_positive"]), 1),
        "cell_recall": stats["true_positive"] / max(int(stats["target_positive"]), 1),
    }


def score_vlm_predictions(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    exact = 0
    true_positive = 0
    predicted_positive = 0
    target_positive = 0
    count_correct = 0
    per_class_raw: dict[str, dict[str, int]] = defaultdict(empty_class_stat)
    for row in rows:
        target = set(int(value) for value in row.get("target_indices", []))
        predicted = set(int(value) for value in row.get("predicted_indices", []))
        label = str(row.get("target_object") or row.get("target_class_key") or "unknown")
        exact_match = predicted == target
        tp = len(predicted & target)
        exact += int(exact_match)
        true_positive += tp
        predicted_positive += len(predicted)
        target_positive += len(target)
        count_correct += int(len(predicted) == len(target))
        stats = per_class_raw[label]
        stats["total"] += 1
        stats["cell_exact"] += int(exact_match)
        stats["true_positive"] += tp
        stats["predicted_positive"] += len(predicted)
        stats["target_positive"] += len(target)
    total = max(len(rows), 1)
    return {
        "total": len(rows),
        "cell_exact_match": exact / total,
        "cell_precision": true_positive / max(predicted_positive, 1),
        "cell_recall": true_positive / max(target_positive, 1),
        "click_order_accuracy": exact / total,
        "count_accuracy": count_correct / total,
        "per_class": {
            key: finalize_binary_metrics(value)
            for key, value in sorted(per_class_raw.items(), key=lambda item: (-item[1]["total"], item[0]))
        },
    }
