from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from src.multimodal_captcha.baseline import parse_color
from src.multimodal_captcha.generator import COLORS, DRAWERS, OBJECTS


def parse_object(prompt: str) -> str | None:
    for object_name in OBJECTS:
        if object_name in prompt:
            return object_name
    return None


def _foreground_mask(arr: np.ndarray) -> np.ndarray:
    bg = np.asarray([250, 251, 252], dtype=np.float32)
    grid = np.asarray([205, 210, 218], dtype=np.float32)
    dist_bg = np.linalg.norm(arr.astype(np.float32) - bg, axis=-1)
    dist_grid = np.linalg.norm(arr.astype(np.float32) - grid, axis=-1)
    return (dist_bg > 28.0) & (dist_grid > 28.0)


def _object_template(object_name: str, cell_size: int) -> np.ndarray:
    image = Image.new("RGB", (cell_size, cell_size), (250, 251, 252))
    draw = ImageDraw.Draw(image)
    pad = int(cell_size * 0.12)
    DRAWERS[object_name](draw, (pad, pad, cell_size - pad, cell_size - pad), (70, 70, 70))
    return _foreground_mask(np.asarray(image))


def _object_scores(crops: list[np.ndarray], object_name: str) -> np.ndarray:
    template = _object_template(object_name, crops[0].shape[0])
    scores = []
    for crop in crops:
        mask = _foreground_mask(crop)
        intersection = np.logical_and(mask, template).sum()
        union = np.logical_or(mask, template).sum()
        scores.append(float(intersection / max(union, 1)))
    return np.asarray(scores, dtype=np.float32)


def _color_scores(crops: list[np.ndarray], color_name: str | None) -> np.ndarray:
    if color_name is None:
        return np.zeros(len(crops), dtype=np.float32)

    target = np.asarray(COLORS[color_name], dtype=np.float32)
    scores = []
    for crop in crops:
        pixel_dist = np.linalg.norm(crop.reshape(-1, 3).astype(np.float32) - target, axis=1)
        close_ratio = float((pixel_dist < 18.0).mean())
        fallback = -float(np.percentile(pixel_dist, 1)) / 255.0
        scores.append(close_ratio * 5.0 + fallback)
    return np.asarray(scores, dtype=np.float32)


def template_grounding_predict(image: Image.Image, prompt: str) -> tuple[int, np.ndarray]:
    arr = np.asarray(image.convert("RGB"))
    height, width, _ = arr.shape
    cell_h, cell_w = height // 3, width // 3
    crops = []
    for row in range(3):
        for col in range(3):
            crop = arr[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w]
            crops.append(crop[4:-4, 4:-4])

    object_name = parse_object(prompt)
    color_name = parse_color(prompt)
    scores = _color_scores(crops, color_name)
    if object_name is not None:
        scores = scores + _object_scores(crops, object_name) * 8.0

    exp = np.exp(scores - scores.max())
    probs = exp / exp.sum()
    return int(probs.argmax()), probs

