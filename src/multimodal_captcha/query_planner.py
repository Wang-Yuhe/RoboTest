from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PIL import Image

from src.multimodal_captcha.attribute_scorer import (
    COLOR_ALIASES,
    score_cell_colors,
    score_cell_positions,
    score_cell_sizes,
    select_cells_from_scores,
)
from src.multimodal_captcha.prompt_rewriter import rule_based_rewrite_prompt, standard_prompt


@dataclass(frozen=True)
class QueryPlan:
    original_prompt: str
    mode: str
    objects: list[str]
    color: str | None = None
    position: str | None = None
    size: str | None = None
    rewritten_prompts: list[str] | None = None


def detect_color(prompt: str) -> str | None:
    for keyword, color in COLOR_ALIASES.items():
        if keyword in prompt:
            return color
    return None


def detect_position(prompt: str) -> str | None:
    for keyword in ("左边", "最左", "左侧", "右边", "最右", "右侧", "上方", "上面", "最上", "下方", "下面", "最下", "中间", "中心"):
        if keyword in prompt:
            return keyword
    return None


def detect_size(prompt: str) -> str | None:
    if "最大" in prompt or "大的" in prompt or "较大" in prompt:
        return "最大"
    if "最小" in prompt or "小的" in prompt or "较小" in prompt:
        return "最小"
    return None


def detect_objects(prompt: str, supported_objects: list[str]) -> list[str]:
    direct = [name for name in supported_objects if name and name in prompt]
    if direct:
        return direct
    rewrite = rule_based_rewrite_prompt(prompt, supported_objects)
    return rewrite.target_objects


def plan_prompt(prompt: str, supported_objects: list[str]) -> QueryPlan:
    color = detect_color(prompt)
    position = detect_position(prompt)
    size = detect_size(prompt)
    objects = detect_objects(prompt, supported_objects)
    has_attrs = any([color, position, size])
    if objects and has_attrs:
        mode = "object_plus_attributes"
    elif objects:
        mode = "object_only"
    elif color:
        mode = "color_only"
    elif position:
        mode = "position_only"
    elif size:
        mode = "size_only"
    else:
        mode = "unsupported"
    return QueryPlan(
        original_prompt=prompt,
        mode=mode,
        objects=objects,
        color=color,
        position=position,
        size=size,
        rewritten_prompts=[standard_prompt(name) for name in objects],
    )


def union_object_candidates(objects: list[str], object_predictor: Callable[[str], list[int]] | None) -> list[int]:
    if object_predictor is None:
        return []
    cells = []
    for object_name in objects:
        cells.extend(int(index) for index in object_predictor(standard_prompt(object_name)))
    return sorted(set(cells))


def apply_attribute_filters(
    image: Image.Image,
    plan: QueryPlan,
    candidates: list[int] | None = None,
) -> tuple[list[int], dict[str, list[float]]]:
    allowed = list(candidates) if candidates else list(range(9))
    traces: dict[str, list[float]] = {}
    if plan.color:
        scores = score_cell_colors(image, plan.color)
        traces["color_scores"] = scores
        allowed = select_cells_from_scores(scores, threshold=0.30, candidates=allowed)
    if plan.position:
        scores = score_cell_positions(plan.position)
        traces["position_scores"] = scores
        allowed = select_cells_from_scores(scores, threshold=0.90, candidates=allowed, fallback_top_k=3)
    if plan.size:
        scores = score_cell_sizes(image)
        traces["size_scores"] = scores
        if plan.size == "最小":
            inverted = [1.0 - score for score in scores]
            traces["size_scores_inverted"] = inverted
            allowed = select_cells_from_scores(inverted, threshold=0.70, candidates=allowed)
        else:
            allowed = select_cells_from_scores(scores, threshold=0.70, candidates=allowed)
    return sorted(set(allowed)), traces


def execute_query_plan(
    image: Image.Image,
    plan: QueryPlan,
    object_predictor: Callable[[str], list[int]] | None = None,
) -> dict:
    object_candidates = union_object_candidates(plan.objects, object_predictor)
    if plan.mode == "object_only":
        selected = object_candidates
        traces: dict[str, list[float]] = {}
    elif plan.mode in {"object_plus_attributes", "color_only", "position_only", "size_only"}:
        base_candidates = object_candidates if object_candidates else None
        selected, traces = apply_attribute_filters(image, plan, base_candidates)
    else:
        selected, traces = [], {}
    return {
        "mode": plan.mode,
        "objects": plan.objects,
        "color": plan.color,
        "position": plan.position,
        "size": plan.size,
        "rewritten_prompts": plan.rewritten_prompts or [],
        "object_candidate_cells": object_candidates,
        "selected_cells": selected,
        **traces,
    }
