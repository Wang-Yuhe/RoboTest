from __future__ import annotations

import math
import random


def random_start_point(image_size: int, rng: random.Random) -> tuple[int, int]:
    margin = max(20, image_size // 10)
    side = rng.choice(["top", "left", "right", "bottom", "inside"])
    if side == "top":
        return rng.randint(0, image_size), rng.randint(-margin, margin)
    if side == "left":
        return rng.randint(-margin, margin), rng.randint(0, image_size)
    if side == "right":
        return rng.randint(image_size - margin, image_size + margin), rng.randint(0, image_size)
    if side == "bottom":
        return rng.randint(0, image_size), rng.randint(image_size - margin, image_size + margin)
    return rng.randint(0, image_size), rng.randint(0, image_size)


def random_point_in_cell(index: int, image_size: int, rng: random.Random, padding_ratio: float = 0.22) -> tuple[int, int]:
    cell = image_size // 3
    row, col = divmod(index, 3)
    pad = int(cell * padding_ratio)
    x0, y0 = col * cell + pad, row * cell + pad
    x1, y1 = (col + 1) * cell - pad, (row + 1) * cell - pad
    return rng.randint(x0, max(x0, x1)), rng.randint(y0, max(y0, y1))


def generate_mouse_trajectory(
    target: tuple[int, int],
    start: tuple[int, int] | None = None,
    steps: int = 0,
    seed: int = 0,
    image_size: int | None = None,
) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    if image_size is None:
        image_size = max(target) * 2 if max(target) > 0 else 288
    if start is None:
        start = random_start_point(image_size, rng)
    if steps <= 0:
        distance = math.hypot(target[0] - start[0], target[1] - start[1])
        steps = max(24, min(72, int(distance / 7) + rng.randint(10, 22)))
    sx, sy = start
    tx, ty = target
    dx, dy = tx - sx, ty - sy
    normal = (-dy, dx)
    norm = math.hypot(*normal) or 1.0
    normal = (normal[0] / norm, normal[1] / norm)
    bend = rng.uniform(-0.16, 0.16) * max(math.hypot(dx, dy), 1.0)
    hesitation_at = rng.choice([None, rng.uniform(0.35, 0.75)])
    points = []
    for i in range(steps):
        t = i / (steps - 1)
        ease = 1 - (1 - t) ** rng.uniform(2.2, 3.4)
        if hesitation_at is not None and abs(t - hesitation_at) < 0.045:
            ease -= rng.uniform(0.008, 0.025)
            ease = max(0.0, min(1.0, ease))
        curve = math.sin(math.pi * t) * bend
        jitter = (1 - abs(2 * t - 1)) * rng.uniform(-2.4, 2.4)
        x = sx + dx * ease + normal[0] * (curve + jitter)
        y = sy + dy * ease + normal[1] * (curve + jitter)
        points.append((x, y))
    for _ in range(rng.randint(1, 3)):
        points.append((tx + rng.uniform(-1.5, 1.5), ty + rng.uniform(-1.5, 1.5)))
    return points


def cell_center(index: int, image_size: int = 288) -> tuple[int, int]:
    cell = image_size // 3
    row, col = divmod(index, 3)
    return col * cell + cell // 2, row * cell + cell // 2
