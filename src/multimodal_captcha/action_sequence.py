from __future__ import annotations

from typing import Iterable

import torch


PAD_TOKEN = 0
CLICK_TOKEN = 1
DONE_TOKEN = 2
MOVE_TOKEN_OFFSET = 3
ACTION_VOCAB_SIZE = MOVE_TOKEN_OFFSET + 9
IGNORE_INDEX = -100


def move_token(cell: int) -> int:
    if not 0 <= cell < 9:
        raise ValueError(f"cell must be in [0, 8], got {cell}")
    return MOVE_TOKEN_OFFSET + cell


def target_indices_to_actions(target_indices: Iterable[int]) -> list[dict]:
    actions: list[dict] = []
    for cell in sorted(int(index) for index in target_indices):
        if not 0 <= cell < 9:
            raise ValueError(f"target index must be in [0, 8], got {cell}")
        actions.append({"type": "move_to_cell", "cell": cell})
        actions.append({"type": "click"})
    actions.append({"type": "done"})
    return actions


def actions_to_tokens(actions: list[dict]) -> list[int]:
    tokens = []
    for action in actions:
        action_type = action.get("type")
        if action_type == "move_to_cell":
            tokens.append(move_token(int(action["cell"])))
        elif action_type == "click":
            tokens.append(CLICK_TOKEN)
        elif action_type == "done":
            tokens.append(DONE_TOKEN)
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    return tokens


def tokens_to_actions(tokens: Iterable[int]) -> list[dict]:
    actions = []
    for token in tokens:
        token = int(token)
        if token == PAD_TOKEN:
            continue
        if token == CLICK_TOKEN:
            actions.append({"type": "click"})
        elif token == DONE_TOKEN:
            actions.append({"type": "done"})
            break
        elif MOVE_TOKEN_OFFSET <= token < ACTION_VOCAB_SIZE:
            actions.append({"type": "move_to_cell", "cell": token - MOVE_TOKEN_OFFSET})
        else:
            raise ValueError(f"Unknown action token: {token}")
    return actions


def encode_action_targets(actions: list[dict], max_len: int = 10) -> list[int]:
    tokens = actions_to_tokens(actions)
    if len(tokens) > max_len:
        raise ValueError(f"Action sequence length {len(tokens)} exceeds max_len={max_len}")
    return tokens + [IGNORE_INDEX] * (max_len - len(tokens))


def normalize_tokens(tokens: Iterable[int]) -> list[int]:
    normalized = []
    for token in tokens:
        token = int(token)
        if token in (PAD_TOKEN, IGNORE_INDEX):
            continue
        normalized.append(token)
        if token == DONE_TOKEN:
            break
    return normalized


def clicked_cells_from_tokens(tokens: Iterable[int]) -> list[int]:
    cells = []
    pending_cell: int | None = None
    for token in normalize_tokens(tokens):
        if MOVE_TOKEN_OFFSET <= token < ACTION_VOCAB_SIZE:
            pending_cell = token - MOVE_TOKEN_OFFSET
        elif token == CLICK_TOKEN and pending_cell is not None:
            cells.append(pending_cell)
            pending_cell = None
        elif token == DONE_TOKEN:
            break
    return cells


def compute_action_metrics(predictions: list[Iterable[int]], targets: list[Iterable[int]]) -> dict[str, float]:
    total = max(len(targets), 1)
    exact = 0
    click_order = 0
    for pred, target in zip(predictions, targets):
        pred_tokens = normalize_tokens(pred)
        target_tokens = normalize_tokens(target)
        exact += int(pred_tokens == target_tokens)
        click_order += int(clicked_cells_from_tokens(pred_tokens) == clicked_cells_from_tokens(target_tokens))
    return {
        "exact_match": exact / total,
        "click_order_accuracy": click_order / total,
    }


def target_indices_to_cell_targets(target_indices: Iterable[int]) -> list[float]:
    targets = [0.0] * 9
    for index in target_indices:
        index = int(index)
        if not 0 <= index < 9:
            raise ValueError(f"target index must be in [0, 8], got {index}")
        targets[index] = 1.0
    return targets


def cell_logits_to_actions(logits: torch.Tensor, threshold: float = 0.5) -> list[list[dict]]:
    probs = torch.sigmoid(logits.detach().cpu())
    all_actions = []
    for row in probs:
        indices = [idx for idx, value in enumerate(row.tolist()) if value >= threshold]
        if not indices:
            indices = [int(row.argmax().item())]
        all_actions.append(target_indices_to_actions(indices))
    return all_actions


def cell_logits_to_topk_actions(
    logits: torch.Tensor,
    count_logits: torch.Tensor,
    min_count: int = 1,
    max_count: int = 4,
) -> list[list[dict]]:
    scores = logits.detach().cpu()
    counts = count_logits.detach().cpu().argmax(dim=1)
    all_actions = []
    for row, count in zip(scores, counts):
        k = max(min_count, min(int(count.item()), max_count, row.numel()))
        indices = torch.topk(row, k=k).indices.tolist()
        all_actions.append(target_indices_to_actions(indices))
    return all_actions
