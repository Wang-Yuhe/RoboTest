from __future__ import annotations

from PIL import Image, ImageDraw


def draw_trajectory(image: Image.Image, points: list[tuple[float, float]], color: tuple[int, int, int] = (38, 95, 185)) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    if len(points) > 1:
        draw.line(points, fill=color, width=3)
    for i, (x, y) in enumerate(points):
        r = 2 if i < len(points) - 1 else 6
        fill = color if i < len(points) - 1 else (220, 54, 46)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)
    return out

