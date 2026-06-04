from __future__ import annotations

import numpy as np
from PIL import Image

from src.multimodal_captcha.generator import COLORS


def parse_color(prompt: str) -> str | None:
    for color_name in COLORS:
        if color_name in prompt:
            return color_name
    return None


def cell_mean_colors(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width, _ = arr.shape
    cell_h, cell_w = height // 3, width // 3
    means = []
    for row in range(3):
        for col in range(3):
            crop = arr[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w]
            means.append(crop.mean(axis=(0, 1)))
    return np.asarray(means)


def color_grounding_predict(image: Image.Image, prompt: str) -> tuple[int, np.ndarray]:
    color_name = parse_color(prompt)
    if color_name is None:
        probs = np.ones(9, dtype=np.float32) / 9
        return 0, probs

    target = np.asarray(COLORS[color_name], dtype=np.float32)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width, _ = arr.shape
    cell_h, cell_w = height // 3, width // 3
    scores = []
    for row in range(3):
        for col in range(3):
            crop = arr[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w]
            pixel_dist = np.linalg.norm(crop.reshape(-1, 3) - target, axis=1)
            close_ratio = float((pixel_dist < 18.0).mean())
            fallback = -float(np.percentile(pixel_dist, 1)) / 255.0
            scores.append(close_ratio * 20.0 + fallback)
    scores = np.asarray(scores, dtype=np.float32)
    exp = np.exp(scores - scores.max())
    probs = exp / exp.sum()
    return int(probs.argmax()), probs
