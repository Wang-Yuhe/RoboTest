from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import requests


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


@dataclass(frozen=True)
class PromptRewriteResult:
    original_prompt: str
    target_objects: list[str]
    rewritten_prompts: list[str]
    provider: str
    reason: str = ""


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        value = str(value).strip()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def standard_prompt(object_name: str) -> str:
    return f"请点击所有{object_name}"


def load_supported_objects(data_dir: str | Path) -> list[str]:
    root = Path(data_dir)
    selected_classes = root / "selected_classes.json"
    if selected_classes.exists():
        payload = json.loads(selected_classes.read_text(encoding="utf-8"))
        return unique_in_order(item["object_name"] for item in payload.get("classes", []) if item.get("object_name"))

    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Cannot load supported objects; missing {selected_classes} and {manifest}.")
    objects = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for item in record.get("items", []):
                name = item.get("object_name")
                if name:
                    objects.append(name)
    return unique_in_order(objects)


def filter_supported_objects(candidates: Iterable[str], supported_objects: Iterable[str]) -> list[str]:
    supported = set(supported_objects)
    return unique_in_order(candidate for candidate in candidates if str(candidate).strip() in supported)


def rule_based_rewrite_prompt(prompt: str, supported_objects: list[str]) -> PromptRewriteResult:
    groups = [
        (("演奏", "音乐", "乐器"), ["吉他", "大提琴", "钢琴", "小提琴", "麦克风"]),
        (("交通", "行驶", "乘坐", "车辆"), ["汽车", "公交车", "卡车", "摩托车", "自行车", "火车", "飞机", "船"]),
        (("水果", "吃", "食物"), ["苹果", "香蕉", "橙子", "蛋糕", "披萨", "贝果", "松饼"]),
        (("穿", "衣服", "服装"), ["裙子", "衬衫", "短裤", "外套", "帽子", "靴子", "手套"]),
        (("动物", "活的"), ["猫", "狗", "鸟", "马", "牛", "羊", "大象", "斑马", "长颈鹿", "熊"]),
    ]
    matched: list[str] = []
    for keywords, object_names in groups:
        if any(keyword in prompt for keyword in keywords):
            matched.extend(object_names)
    if not matched:
        matched = [name for name in supported_objects if name in prompt]
    targets = filter_supported_objects(matched, supported_objects)
    return PromptRewriteResult(
        original_prompt=prompt,
        target_objects=targets,
        rewritten_prompts=[standard_prompt(name) for name in targets],
        provider="rule",
        reason="rule_fallback",
    )


class DeepSeekPromptRewriter:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEEPSEEK_BASE_URL,
        model: str = DEEPSEEK_MODEL,
        timeout: int = 30,
        post: Callable[..., Any] = requests.post,
    ):
        self.api_key = api_key if api_key is not None else os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.post = post

    def rewrite(self, prompt: str, supported_objects: list[str]) -> PromptRewriteResult:
        if not self.api_key:
            return rule_based_rewrite_prompt(prompt, supported_objects)

        system = (
            "你是 RoboTest 的 prompt 标准化器。"
            "你只能从给定的 supported_objects 中选择目标类别。"
            "不要输出列表之外的类别。"
            "如果用户描述的是功能、用途或上位概念，请映射到 supported_objects 中最合理的具体物体。"
            "只输出 JSON，格式为 {\"target_objects\":[...],\"reason\":\"...\"}。"
        )
        user = {
            "prompt": prompt,
            "supported_objects": supported_objects,
            "output_contract": {
                "target_objects": "list of object names copied exactly from supported_objects",
                "reason": "short Chinese reason",
            },
        }
        response = self.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        targets = filter_supported_objects(parsed.get("target_objects", []), supported_objects)
        if not targets:
            fallback = rule_based_rewrite_prompt(prompt, supported_objects)
            return PromptRewriteResult(
                original_prompt=prompt,
                target_objects=fallback.target_objects,
                rewritten_prompts=fallback.rewritten_prompts,
                provider="rule",
                reason="deepseek_empty_supported_targets",
            )
        return PromptRewriteResult(
            original_prompt=prompt,
            target_objects=targets,
            rewritten_prompts=[standard_prompt(name) for name in targets],
            provider="deepseek",
            reason=str(parsed.get("reason", "")),
        )


def rewrite_prompt(
    prompt: str,
    supported_objects: list[str],
    provider: str = "deepseek",
    api_key: str | None = None,
    base_url: str = DEEPSEEK_BASE_URL,
    model: str = DEEPSEEK_MODEL,
) -> PromptRewriteResult:
    if provider == "rule":
        return rule_based_rewrite_prompt(prompt, supported_objects)
    if provider == "deepseek":
        return DeepSeekPromptRewriter(api_key=api_key, base_url=base_url, model=model).rewrite(prompt, supported_objects)
    raise ValueError(f"Unknown prompt rewrite provider: {provider}")


def merge_rewrite_results(
    original_prompt: str,
    rewrite: PromptRewriteResult,
    per_prompt_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    cells = []
    for prediction in per_prompt_predictions:
        cells.extend(int(index) for index in prediction.get("predicted_indices", []))
    return {
        "prompt": original_prompt,
        "provider": rewrite.provider,
        "target_objects": rewrite.target_objects,
        "rewritten_prompts": rewrite.rewritten_prompts,
        "predicted_indices": sorted(set(cells)),
        "per_prompt_predictions": per_prompt_predictions,
        "reason": rewrite.reason,
    }
