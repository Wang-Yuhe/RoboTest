from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.multimodal_captcha.action_sequence import ACTION_VOCAB_SIZE


class ClipVisionPooler(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        try:
            from transformers import CLIPVisionConfig, CLIPVisionModel
        except ImportError as exc:
            raise RuntimeError("image_encoder='clip_vit_b32' requires transformers. Install transformers first.") from exc
        if pretrained:
            self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            self.model = CLIPVisionModel(CLIPVisionConfig())
        self.output_dim = int(self.model.config.hidden_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values=image)
        return output.pooler_output


def build_cell_image_encoder(
    image_encoder: str,
    base_channels: int,
    hidden_dim: int,
    pretrained: bool,
) -> tuple[nn.Module, int]:
    if image_encoder == "custom":
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 6
        return (
            nn.Sequential(
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
            ),
            c4,
        )
    if image_encoder == "resnet18":
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:
            raise RuntimeError("image_encoder='resnet18' requires torchvision. Install torchvision first.") from exc
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        return nn.Sequential(*list(model.children())[:-1]), 512
    if image_encoder == "clip_vit_b32":
        model = ClipVisionPooler(pretrained=pretrained)
        return model, model.output_dim
    raise ValueError(f"Unknown image_encoder: {image_encoder}")


def configure_cell_image_encoder_trainability(image_encoder: nn.Module, image_encoder_name: str, train_mode: str) -> None:
    if train_mode == "full":
        for parameter in image_encoder.parameters():
            parameter.requires_grad = True
        return
    if train_mode == "frozen":
        for parameter in image_encoder.parameters():
            parameter.requires_grad = False
        return
    if train_mode == "last_block":
        if image_encoder_name != "resnet18":
            raise ValueError("--encoder-train-mode last_block is only supported for image_encoder='resnet18'.")
        children = list(image_encoder.children())
        for parameter in image_encoder.parameters():
            parameter.requires_grad = False
        for parameter in children[7].parameters():
            parameter.requires_grad = True
        return
    raise ValueError(f"Unknown encoder train mode: {train_mode}")


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
        image_encoder: str = "custom",
        pretrained: bool = False,
        encoder_train_mode: str = "full",
        use_count_head: bool = False,
        max_count: int = 4,
    ):
        super().__init__()
        self.object_vocab_size = object_vocab_size
        self.image_size = image_size
        self.use_transformer = use_transformer
        self.use_interactions = use_interactions
        self.image_encoder_name = image_encoder
        self.encoder_train_mode = encoder_train_mode
        self.text_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.text_encoder = nn.GRU(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.image_encoder, encoder_dim = build_cell_image_encoder(image_encoder, base_channels, hidden_dim, pretrained)
        configure_cell_image_encoder_trainability(self.image_encoder, image_encoder, encoder_train_mode)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.image_projection = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim * 2),
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

    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder_train_mode == "frozen":
            self.image_encoder.eval()
        return self

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
        if self.image_encoder_name == "clip_vit_b32":
            cells = F.interpolate(cells, size=(224, 224), mode="bilinear", align_corners=False)
            clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=cells.device).view(1, 3, 1, 1)
            clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=cells.device).view(1, 3, 1, 1)
            cells = (cells - clip_mean) / clip_std
        elif self.image_encoder_name == "resnet18":
            cells = (cells - self.image_mean) / self.image_std
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


class ActionSequenceLocator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_action_len: int = 10,
        embed_dim: int = 96,
        hidden_dim: int = 96,
        image_size: int = 64,
        base_channels: int = 24,
    ):
        super().__init__()
        self.max_action_len = max_action_len
        self.encoder = MultimodalGridLocator(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            image_size=image_size,
            base_channels=base_channels,
        )
        context_dim = hidden_dim * 4
        self.step_embedding = nn.Parameter(torch.zeros(1, max_action_len, context_dim))
        self.action_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, ACTION_VOCAB_SIZE),
        )

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        text_feat = self.encoder.encode_text(text)
        cell_feat = self.encoder.encode_cells(image)
        image_feat = cell_feat.mean(dim=1)
        context = torch.cat([image_feat, text_feat], dim=-1)
        stepped = context.unsqueeze(1) + self.step_embedding
        return self.action_head(stepped)


class ActionCellSelector(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 96,
        hidden_dim: int = 96,
        object_vocab_size: int = 0,
        image_size: int = 64,
        base_channels: int = 24,
        use_interactions: bool = True,
        image_encoder: str = "custom",
        pretrained: bool = False,
        encoder_train_mode: str = "full",
        use_count_head: bool = False,
        max_count: int = 4,
    ):
        super().__init__()
        self.encoder = MultimodalGridLocator(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            object_vocab_size=object_vocab_size,
            image_size=image_size,
            base_channels=base_channels,
            image_encoder=image_encoder,
            pretrained=pretrained,
            encoder_train_mode=encoder_train_mode,
        )
        self.use_interactions = use_interactions
        self.use_count_head = use_count_head
        self.max_count = max_count
        fusion_dim = hidden_dim * 8 if use_interactions else hidden_dim * 4
        self.selector = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.count_head = (
            nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim * 2, max_count + 1),
            )
            if use_count_head
            else None
        )

    def forward(self, image: torch.Tensor, text: torch.Tensor, return_aux: bool = False):
        text_feat = self.encoder.encode_text(text)
        cell_feat = self.encoder.encode_cells(image) + self.encoder.position_embedding
        text_grid = text_feat.unsqueeze(1).expand(-1, 9, -1)
        if self.use_interactions:
            fused = torch.cat([cell_feat, text_grid, cell_feat * text_grid, torch.abs(cell_feat - text_grid)], dim=-1)
        else:
            fused = torch.cat([cell_feat, text_grid], dim=-1)
        cell_logits = self.selector(fused).squeeze(-1)
        if not return_aux:
            return cell_logits
        object_logits = self.encoder.object_head(cell_feat) if self.encoder.object_head is not None else None
        if self.count_head is None:
            return cell_logits, object_logits
        global_cell_feat = cell_feat.mean(dim=1)
        count_logits = self.count_head(torch.cat([global_cell_feat, text_feat], dim=-1))
        return cell_logits, object_logits, count_logits


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
