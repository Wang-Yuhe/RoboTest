from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image


COLOR_ALIASES = {
    "红": "红色",
    "红色": "红色",
    "蓝": "蓝色",
    "蓝色": "蓝色",
    "绿": "绿色",
    "绿色": "绿色",
    "黄": "黄色",
    "黄色": "黄色",
    "橙": "橙色",
    "橙色": "橙色",
    "紫": "紫色",
    "紫色": "紫色",
    "粉": "粉色",
    "粉色": "粉色",
    "黑": "黑色",
    "黑色": "黑色",
    "白": "白色",
    "白色": "白色",
    "灰": "灰色",
    "灰色": "灰色",
    "棕": "棕色",
    "棕色": "棕色",
}


def normalize_color(color: str) -> str:
    color = str(color).strip()
    return COLOR_ALIASES.get(color, color)


def cell_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    cell_w = width // 3
    cell_h = height // 3
    return [
        (col * cell_w, row * cell_h, (col + 1) * cell_w, (row + 1) * cell_h)
        for row in range(3)
        for col in range(3)
    ]


def center_crop_array(image: Image.Image, box: tuple[int, int, int, int], fraction: float = 0.70) -> np.ndarray:
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    pad_x = int(w * (1.0 - fraction) / 2.0)
    pad_y = int(h * (1.0 - fraction) / 2.0)
    crop = image.crop((x0 + pad_x, y0 + pad_y, x1 - pad_x, y1 - pad_y)).convert("HSV")
    return np.asarray(crop, dtype=np.float32)


def color_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    color = normalize_color(color)
    h = hsv[..., 0] / 255.0
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    saturated = s > 0.25
    visible = v > 0.15
    if color == "红色":
        return ((h <= 0.04) | (h >= 0.94)) & saturated & visible
    if color == "橙色":
        return (h > 0.04) & (h <= 0.10) & saturated & visible
    if color == "黄色":
        return (h > 0.10) & (h <= 0.18) & saturated & visible
    if color == "绿色":
        return (h > 0.20) & (h <= 0.45) & saturated & visible
    if color == "蓝色":
        return (h > 0.52) & (h <= 0.72) & saturated & visible
    if color == "紫色":
        return (h > 0.72) & (h <= 0.84) & saturated & visible
    if color == "粉色":
        return ((h <= 0.04) | (h >= 0.86)) & (s > 0.18) & (v > 0.45)
    if color == "黑色":
        return v < 0.22
    if color == "白色":
        return (s < 0.18) & (v > 0.78)
    if color == "灰色":
        return (s < 0.18) & (v >= 0.22) & (v <= 0.78)
    if color == "棕色":
        return (h > 0.04) & (h <= 0.13) & (s > 0.25) & (v > 0.20) & (v < 0.75)
    return np.zeros(h.shape, dtype=bool)


def score_cell_colors(image: Image.Image, color: str, center_fraction: float = 0.70) -> list[float]:
    boxes = cell_boxes(image.width, image.height)
    scores = []
    for box in boxes:
        hsv = center_crop_array(image, box, fraction=center_fraction)
        mask = color_mask(hsv, color)
        scores.append(float(mask.mean()))
    return scores


def score_cell_positions(position: str) -> list[float]:
    position = str(position)
    scores = [0.0] * 9
    if "左" in position:
        for idx in (0, 3, 6):
            scores[idx] = 1.0
    elif "右" in position:
        for idx in (2, 5, 8):
            scores[idx] = 1.0
    elif "上" in position:
        for idx in (0, 1, 2):
            scores[idx] = 1.0
    elif "下" in position:
        for idx in (6, 7, 8):
            scores[idx] = 1.0
    elif "中" in position or "中心" in position:
        scores[4] = 1.0
    return scores


def score_cell_sizes(image: Image.Image, center_fraction: float = 0.82) -> list[float]:
    boxes = cell_boxes(image.width, image.height)
    scores = []
    for box in boxes:
        crop = center_crop_array(image, box, fraction=center_fraction)
        s = crop[..., 1] / 255.0
        v = crop[..., 2] / 255.0
        foreground = (s > 0.18) & (v > 0.12)
        scores.append(float(foreground.mean()))
    max_score = max(scores) if scores else 0.0
    if max_score > 0:
        return [score / max_score for score in scores]
    return scores


def select_cells_from_scores(
    scores: Iterable[float],
    threshold: float = 0.30,
    candidates: Iterable[int] | None = None,
    fallback_top_k: int = 1,
) -> list[int]:
    values = [float(score) for score in scores]
    allowed = set(int(idx) for idx in candidates) if candidates is not None else set(range(len(values)))
    selected = [idx for idx, score in enumerate(values) if idx in allowed and score >= threshold]
    if selected:
        return selected
    ranked = sorted((idx for idx in allowed if 0 <= idx < len(values)), key=lambda idx: values[idx], reverse=True)
    return ranked[:fallback_top_k]
