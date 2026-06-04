from __future__ import annotations

import math
import random


def generate_mouse_trajectory(
    target: tuple[int, int],
    start: tuple[int, int] = (12, 12),
    steps: int = 36,
    seed: int = 0,
) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    sx, sy = start
    tx, ty = target
    dx, dy = tx - sx, ty - sy
    normal = (-dy, dx)
    norm = math.hypot(*normal) or 1.0
    normal = (normal[0] / norm, normal[1] / norm)
    bend = rng.uniform(-26, 26)
    points = []
    for i in range(steps):
        t = i / (steps - 1)
        ease = 3 * t * t - 2 * t * t * t
        curve = math.sin(math.pi * t) * bend
        jitter = (1 - abs(2 * t - 1)) * rng.uniform(-1.8, 1.8)
        x = sx + dx * ease + normal[0] * (curve + jitter)
        y = sy + dy * ease + normal[1] * (curve + jitter)
        points.append((x, y))
    return points


def cell_center(index: int, image_size: int = 288) -> tuple[int, int]:
    cell = image_size // 3
    row, col = divmod(index, 3)
    return col * cell + cell // 2, row * cell + cell // 2

