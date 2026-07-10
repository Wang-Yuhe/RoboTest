from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.multimodal_captcha.model import ActionCellSelector


REQUIRED_ACTION_CHECKPOINT_KEYS = ("model", "vocab", "object_vocab", "model_config")
ACTION_CHECKPOINT_ARCHITECTURE = "action_cell_selector"


def load_action_checkpoint(checkpoint_path: str | Path, device: str) -> tuple[dict[str, Any], dict[str, int], dict[str, int], dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    for key in REQUIRED_ACTION_CHECKPOINT_KEYS:
        if key not in checkpoint:
            raise ValueError(f"Invalid action checkpoint: missing required checkpoint key: {key}")
    config = checkpoint["model_config"]
    if not isinstance(config, dict):
        raise ValueError("Invalid action checkpoint: model_config must be a dictionary.")
    architecture = config.get("architecture")
    if architecture != ACTION_CHECKPOINT_ARCHITECTURE:
        raise ValueError(
            "Invalid action checkpoint: expected architecture "
            f"{ACTION_CHECKPOINT_ARCHITECTURE!r}, got {architecture!r}."
        )
    vocab = checkpoint["vocab"]
    object_vocab = checkpoint["object_vocab"]
    if not isinstance(vocab, dict):
        raise ValueError("Invalid action checkpoint: vocab must be a dictionary.")
    if not isinstance(object_vocab, dict):
        raise ValueError("Invalid action checkpoint: object_vocab must be a dictionary.")
    return checkpoint, vocab, object_vocab, config


def build_action_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str,
) -> tuple[ActionCellSelector, dict[str, int], dict[str, int], dict]:
    checkpoint, vocab, object_vocab, config = load_action_checkpoint(checkpoint_path, device)
    model = ActionCellSelector(
        vocab_size=int(config.get("vocab_size", len(vocab))),
        embed_dim=int(config.get("embed_dim", 96)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        object_vocab_size=int(config.get("object_vocab_size", len(object_vocab))),
        image_size=int(config.get("image_size", 64)),
        base_channels=int(config.get("base_channels", 24)),
        use_interactions=bool(config.get("use_interactions", True)),
        image_encoder=str(config.get("image_encoder", "custom")),
        # The checkpoint already contains encoder weights. Reloading pretrained
        # initialization here is redundant and can trigger a network download.
        pretrained=False,
        encoder_train_mode=str(config.get("encoder_train_mode", "full")),
        use_count_head=bool(config.get("use_count_head", False)),
        max_count=int(config.get("max_count", 4)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, vocab, object_vocab, config
