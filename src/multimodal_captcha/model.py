from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x), inplace=True)


class MultimodalGridLocator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 96,
        hidden_dim: int = 96,
        object_vocab_size: int = 0,
        image_size: int = 64,
        base_channels: int = 24,
        use_transformer: bool = False,
        use_interactions: bool = False,
    ):
        super().__init__()
        self.object_vocab_size = object_vocab_size
        self.image_size = image_size
        self.use_transformer = use_transformer
        self.use_interactions = use_interactions
        self.text_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.text_encoder = nn.GRU(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 6
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            ResidualBlock(c1),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            ResidualBlock(c2),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            ResidualBlock(c3),
            nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.image_projection = nn.Sequential(
            nn.Linear(c4, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        feat_dim = hidden_dim * 2
        self.position_embedding = nn.Parameter(torch.zeros(1, 9, feat_dim))
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=feat_dim,
                nhead=4,
                dim_feedforward=feat_dim * 2,
                dropout=0.15,
                batch_first=True,
                activation="gelu",
            )
            self.cell_context = nn.TransformerEncoder(encoder_layer, num_layers=2)
        else:
            self.cell_context = None
        fusion_dim = hidden_dim * 8 if use_interactions else hidden_dim * 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.object_head = nn.Linear(hidden_dim * 2, object_vocab_size) if object_vocab_size > 0 else None

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        emb = self.text_embedding(text)
        _, h = self.text_encoder(emb)
        return torch.cat([h[-2], h[-1]], dim=-1)

    def encode_cells(self, image: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = image.shape
        cell_h, cell_w = height // 3, width // 3
        cells = image.reshape(batch, channels, 3, cell_h, 3, cell_w)
        cells = cells.permute(0, 2, 4, 1, 3, 5).reshape(batch * 9, channels, cell_h, cell_w)
        cells = F.interpolate(cells, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        image_feat = self.image_encoder(cells).flatten(1)
        return self.image_projection(image_feat).reshape(batch, 9, -1)

    def forward(self, image: torch.Tensor, text: torch.Tensor, return_aux: bool = False):
        text_feat = self.encode_text(text)
        image_feat = self.encode_cells(image)
        image_feat = image_feat + self.position_embedding
        if self.cell_context is not None:
            image_feat = self.cell_context(image_feat)
        text_grid = text_feat.unsqueeze(1).expand(-1, 9, -1)
        if self.use_interactions:
            fused = torch.cat([image_feat, text_grid, image_feat * text_grid, torch.abs(image_feat - text_grid)], dim=-1)
        else:
            fused = torch.cat([image_feat, text_grid], dim=-1)
        grid_logits = self.fusion(fused).squeeze(-1)
        if not return_aux:
            return grid_logits
        object_logits = self.object_head(image_feat) if self.object_head is not None else None
        return grid_logits, object_logits


class LegacyMultimodalGridLocator(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 96, object_vocab_size: int = 0):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.text_encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.image_projection = nn.Sequential(
            nn.Linear(128, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        self.object_head = nn.Linear(hidden_dim * 2, object_vocab_size) if object_vocab_size > 0 else None

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        emb = self.text_embedding(text)
        _, h = self.text_encoder(emb)
        return torch.cat([h[-2], h[-1]], dim=-1)

    def encode_cells(self, image: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = image.shape
        cell_h, cell_w = height // 3, width // 3
        cells = image.reshape(batch, channels, 3, cell_h, 3, cell_w)
        cells = cells.permute(0, 2, 4, 1, 3, 5).reshape(batch * 9, channels, cell_h, cell_w)
        cells = F.interpolate(cells, size=(64, 64), mode="bilinear", align_corners=False)
        image_feat = self.image_encoder(cells).flatten(1)
        return self.image_projection(image_feat).reshape(batch, 9, -1)

    def forward(self, image: torch.Tensor, text: torch.Tensor, return_aux: bool = False):
        text_feat = self.encode_text(text)
        image_feat = self.encode_cells(image)
        text_grid = text_feat.unsqueeze(1).expand(-1, 9, -1)
        fused = torch.cat([image_feat, text_grid], dim=-1)
        grid_logits = self.fusion(fused).squeeze(-1)
        if not return_aux:
            return grid_logits
        object_logits = self.object_head(image_feat) if self.object_head is not None else None
        return grid_logits, object_logits


class LegacyResidualGridLocator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 96,
        hidden_dim: int = 96,
        object_vocab_size: int = 0,
        image_size: int = 64,
        base_channels: int = 24,
    ):
        super().__init__()
        self.image_size = image_size
        self.text_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.text_encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 6
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            ResidualBlock(c1),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            ResidualBlock(c2),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            ResidualBlock(c3),
            nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.image_projection = nn.Sequential(
            nn.Linear(c4, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.object_head = nn.Linear(hidden_dim * 2, object_vocab_size) if object_vocab_size > 0 else None

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        emb = self.text_embedding(text)
        _, h = self.text_encoder(emb)
        return torch.cat([h[-2], h[-1]], dim=-1)

    def encode_cells(self, image: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = image.shape
        cell_h, cell_w = height // 3, width // 3
        cells = image.reshape(batch, channels, 3, cell_h, 3, cell_w)
        cells = cells.permute(0, 2, 4, 1, 3, 5).reshape(batch * 9, channels, cell_h, cell_w)
        cells = F.interpolate(cells, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        image_feat = self.image_encoder(cells).flatten(1)
        return self.image_projection(image_feat).reshape(batch, 9, -1)

    def forward(self, image: torch.Tensor, text: torch.Tensor, return_aux: bool = False):
        text_feat = self.encode_text(text)
        image_feat = self.encode_cells(image)
        text_grid = text_feat.unsqueeze(1).expand(-1, 9, -1)
        fused = torch.cat([image_feat, text_grid], dim=-1)
        grid_logits = self.fusion(fused).squeeze(-1)
        if not return_aux:
            return grid_logits
        object_logits = self.object_head(image_feat) if self.object_head is not None else None
        return grid_logits, object_logits


def build_model_from_checkpoint(checkpoint: dict, vocab_size: int, object_vocab_size: int) -> nn.Module:
    config = checkpoint.get("model_config", {})
    state = checkpoint.get("model", {})
    is_legacy = "hidden_dim" not in config and "image_encoder.4.weight" in state
    if is_legacy:
        embed_dim = state["text_embedding.weight"].shape[1]
        hidden_dim = state["text_encoder.weight_hh_l0"].shape[1]
        return LegacyMultimodalGridLocator(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            object_vocab_size=object_vocab_size,
        )
    is_residual_legacy = "hidden_dim" in config and "fusion.3.weight" in state and "fusion.6.weight" not in state
    if is_residual_legacy:
        return LegacyResidualGridLocator(
            vocab_size=config.get("vocab_size", vocab_size),
            object_vocab_size=config.get("object_vocab_size", object_vocab_size),
            hidden_dim=config.get("hidden_dim", 96),
            image_size=config.get("image_size", 64),
            base_channels=config.get("base_channels", 24),
        )
    return MultimodalGridLocator(
        vocab_size=config.get("vocab_size", vocab_size),
        object_vocab_size=config.get("object_vocab_size", object_vocab_size),
        hidden_dim=config.get("hidden_dim", 96),
        image_size=config.get("image_size", 64),
        base_channels=config.get("base_channels", 24),
        use_transformer=config.get("use_transformer", False),
        use_interactions=config.get("use_interactions", False),
    )


def predict_index(model: nn.Module, image: torch.Tensor, text: torch.Tensor, device: str = "cpu") -> tuple[int, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device), text.unsqueeze(0).to(device))[0]
        probs = torch.softmax(logits.cpu(), dim=0)
        return int(probs.argmax().item()), probs
