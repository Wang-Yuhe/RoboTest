from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, MutableMapping

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.multimodal_captcha.action_checkpoint import build_action_model_from_checkpoint
from src.multimodal_captcha.action_sequence import cell_logits_to_actions, cell_logits_to_topk_actions
from src.multimodal_captcha.dataset import encode_text
from src.multimodal_captcha.model import ActionCellSelector
from src.multimodal_captcha.trajectory import generate_mouse_trajectory, random_point_in_cell
from src.multimodal_captcha.visualize import draw_trajectory


DEFAULT_ACTION_DATA_DIR = Path("data/photo_action_click_all_clean80_paired_10k_20260707_2211")
DEFAULT_ACTION_CHECKPOINT = Path(
    "outputs/overnight_paired_20260707_2211/checkpoints/"
    "action_resnet18_frozen_clean80_paired_20260707_2211.pt"
)
TURNSTILE_TEST_SITE_KEY = "1x00000000000000000000AA"
ACTION_PROMPT_KEY = "action_prompt"
ACTION_PROMPT_RECORD_KEY = "action_prompt_record_key"


@dataclass(frozen=True)
class ActionDemoPrediction:
    prompt: str
    predicted_indices: list[int]
    target_indices: list[int]
    actions: list[dict]
    cell_probabilities: list[float]
    correct: bool | None
    visualization: Image.Image


def resolve_first_existing_path(candidates: Iterable[str | Path]) -> Path | None:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return None


def load_action_demo_records(data_dir: str | Path, split: str = "test", limit: int | None = None) -> list[dict]:
    root = Path(data_dir)
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Action demo manifest not found: {manifest}")

    records = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split", split) == split:
                records.append(record)
                if limit is not None and len(records) >= limit:
                    break
    if not records:
        raise ValueError(f"No records found for split={split!r} in {manifest}")
    return records


def load_action_demo_model(checkpoint_path: str | Path, device: str) -> tuple[ActionCellSelector, dict[str, int], dict]:
    model, vocab, _, config = build_action_model_from_checkpoint(checkpoint_path, device)
    return model, vocab, config


def sync_action_prompt_state(
    state: MutableMapping[str, Any],
    record_key: str,
    default_prompt: str,
    prompt_key: str = ACTION_PROMPT_KEY,
    record_key_field: str = ACTION_PROMPT_RECORD_KEY,
) -> str:
    if state.get(record_key_field) != record_key:
        state[prompt_key] = default_prompt
        state[record_key_field] = record_key
    return str(state.get(prompt_key, default_prompt))


def explicit_cached_request(
    state: MutableMapping[str, Any],
    cache_field: str,
    request_key: Any,
    trigger: bool,
    request: Callable[[], Any],
) -> Any | None:
    cached = state.get(cache_field)
    if trigger:
        result = request()
        state[cache_field] = {"request_key": request_key, "result": result}
        return result
    if isinstance(cached, dict) and cached.get("request_key") == request_key:
        return cached.get("result")
    return None


def actions_to_clicked_cells(actions: list[dict]) -> list[int]:
    cells = []
    pending_cell = None
    for action in actions:
        action_type = action.get("type")
        if action_type == "move_to_cell":
            pending_cell = int(action["cell"])
        elif action_type == "click" and pending_cell is not None:
            cells.append(pending_cell)
            pending_cell = None
        elif action_type == "done":
            break
    return cells


def image_to_model_tensor(image: Image.Image) -> torch.Tensor:
    if image.width != image.height:
        raise ValueError("Action demo image must be square.")
    if image.width % 3 != 0:
        raise ValueError("Action demo image width must be divisible by 3.")
    arr = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.tensor(arr, dtype=torch.float32)


def draw_action_demo_overlay(
    image: Image.Image,
    predicted_indices: Iterable[int],
    target_indices: Iterable[int] | None = None,
    seed: int = 0,
) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    cell = out.size[0] // 3

    def draw_cells(indices: Iterable[int], color: tuple[int, int, int], width: int) -> None:
        for index in indices:
            row, col = divmod(int(index), 3)
            x0, y0 = col * cell + 4, row * cell + 4
            x1, y1 = x0 + cell - 8, y0 + cell - 8
            draw.rectangle([x0, y0, x1, y1], outline=color, width=width)

    if target_indices is not None:
        draw_cells(target_indices, (40, 160, 80), 4)
    predicted = list(predicted_indices)
    draw_cells(predicted, (220, 54, 46), 5)

    rng = random.Random(seed)
    current = None
    for step, index in enumerate(predicted):
        target = random_point_in_cell(int(index), image.size[0], rng)
        points = generate_mouse_trajectory(target, start=current, seed=seed + step, image_size=image.size[0])
        current = (int(points[-1][0]), int(points[-1][1]))
        out = draw_trajectory(out, points)
    return out


def predict_action_demo(
    model: torch.nn.Module,
    vocab: dict[str, int],
    config: dict,
    image: Image.Image,
    prompt: str,
    target_indices: Iterable[int] | None = None,
    threshold: float = 0.5,
    decode_policy: str = "threshold",
    device: str = "cpu",
    seed: int = 0,
) -> ActionDemoPrediction:
    image_tensor = image_to_model_tensor(image).unsqueeze(0).to(device)
    text_tensor = encode_text(prompt, vocab).unsqueeze(0).to(device)
    with torch.no_grad():
        if decode_policy == "topk_count":
            if getattr(model, "count_head", None) is None:
                raise ValueError("topk_count decode policy requires a checkpoint with use_count_head=true.")
            logits, _, count_logits = model(image_tensor, text_tensor, return_aux=True)
            actions = cell_logits_to_topk_actions(
                logits,
                count_logits,
                min_count=1,
                max_count=int(config.get("max_count", 4)),
            )[0]
        elif decode_policy == "threshold":
            logits = model(image_tensor, text_tensor)
            actions = cell_logits_to_actions(logits, threshold=threshold)[0]
        else:
            raise ValueError(f"Unknown decode policy: {decode_policy}")

    probs = torch.sigmoid(logits[0].detach().cpu()).tolist()
    predicted_indices = actions_to_clicked_cells(actions)
    targets = [int(index) for index in target_indices] if target_indices is not None else []
    correct = sorted(predicted_indices) == sorted(targets) if target_indices is not None else None
    visualization = draw_action_demo_overlay(image, predicted_indices, targets, seed=seed)
    return ActionDemoPrediction(
        prompt=prompt,
        predicted_indices=predicted_indices,
        target_indices=targets,
        actions=actions,
        cell_probabilities=[round(float(value), 4) for value in probs],
        correct=correct,
        visualization=visualization,
    )


def build_turnstile_widget_html(site_key: str = TURNSTILE_TEST_SITE_KEY) -> str:
    return f"""
<script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer></script>
<div class="cf-turnstile" data-sitekey="{site_key}"></div>
"""
