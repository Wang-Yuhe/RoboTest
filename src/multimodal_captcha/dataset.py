from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL import ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from src.multimodal_captcha.action_sequence import encode_action_targets, target_indices_to_cell_targets


PAD = "<pad>"
UNK = "<unk>"


def tokenize(text: str) -> list[str]:
    return list(text.strip())


def build_vocab(manifest_path: str | Path) -> dict[str, int]:
    counter: Counter[str] = Counter()
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            counter.update(tokenize(record["prompt"]))
    vocab = {PAD: 0, UNK: 1}
    for token, _ in counter.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def build_object_vocab(manifest_path: str | Path) -> dict[str, int]:
    objects = set()
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            for item in record.get("items", []):
                object_name = item.get("object_name")
                if object_name:
                    objects.add(object_name)
    return {name: idx for idx, name in enumerate(sorted(objects))}


def encode_text(text: str, vocab: dict[str, int], max_len: int = 20) -> torch.Tensor:
    ids = [vocab.get(token, vocab[UNK]) for token in tokenize(text)[:max_len]]
    ids += [vocab[PAD]] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class CaptchaDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        vocab: dict[str, int] | None = None,
        object_vocab: dict[str, int] | None = None,
        train_ratio: float = 0.85,
        augment: bool = False,
    ):
        self.root = Path(root)
        manifest = self.root / "manifest.jsonl"
        with manifest.open("r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]

        cut = int(len(records) * train_ratio)
        self.records = records[:cut] if split == "train" else records[cut:]
        self.vocab = vocab or build_vocab(manifest)
        self.object_vocab = object_vocab or build_object_vocab(manifest)
        self.augment = augment and split == "train"

    def __len__(self) -> int:
        return len(self.records)

    def augment_image(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.8:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.82, 1.18))
        if random.random() < 0.8:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.82, 1.22))
        if random.random() < 0.7:
            image = ImageEnhance.Color(image).enhance(random.uniform(0.80, 1.25))
        if random.random() < 0.25:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
        arr = np.asarray(image, dtype=np.float32)
        if random.random() < 0.55:
            noise = np.random.normal(0.0, random.uniform(2.0, 8.0), arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0, 255)
        return Image.fromarray(arr.astype(np.uint8), "RGB")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        record = self.records[idx]
        image = Image.open(self.root / record["image"]).convert("RGB")
        if self.augment:
            image = self.augment_image(image)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        object_ids = []
        for item in record.get("items", []):
            object_ids.append(self.object_vocab.get(item.get("object_name", ""), -100))
        object_ids = (object_ids + [-100] * 9)[:9]
        target_index = int(record["target_index"])
        target_object = object_ids[target_index] if 0 <= target_index < len(object_ids) else -100

        return {
            "image": torch.tensor(arr, dtype=torch.float32),
            "text": encode_text(record["prompt"], self.vocab),
            "target": torch.tensor(target_index, dtype=torch.long),
            "object_ids": torch.tensor(object_ids, dtype=torch.long),
            "target_object": torch.tensor(target_object, dtype=torch.long),
            "prompt": record["prompt"],
        }


class ActionSequenceDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        vocab: dict[str, int] | None = None,
        object_vocab: dict[str, int] | None = None,
        train_ratio: float = 0.85,
        max_action_len: int = 10,
        augment: bool = False,
    ):
        self.root = Path(root)
        manifest = self.root / "manifest.jsonl"
        with manifest.open("r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]

        if records and "split" in records[0]:
            self.records = [record for record in records if record.get("split") == split]
        else:
            cut = int(len(records) * train_ratio)
            self.records = records[:cut] if split == "train" else records[cut:]
        self.vocab = vocab or build_vocab(manifest)
        self.object_vocab = object_vocab or build_object_vocab(manifest)
        self.max_action_len = max_action_len
        self.augment = augment and split == "train"

    def __len__(self) -> int:
        return len(self.records)

    def augment_image(self, image: Image.Image) -> Image.Image:
        return CaptchaDataset.augment_image(self, image)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        record = self.records[idx]
        image = Image.open(self.root / record["image"]).convert("RGB")
        if self.augment:
            image = self.augment_image(image)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        object_ids = []
        for item in record.get("items", []):
            object_ids.append(self.object_vocab.get(item.get("object_name", ""), -100))
        object_ids = (object_ids + [-100] * 9)[:9]

        return {
            "image": torch.tensor(arr, dtype=torch.float32),
            "text": encode_text(record["prompt"], self.vocab),
            "action_targets": torch.tensor(
                encode_action_targets(record["actions"], self.max_action_len),
                dtype=torch.long,
            ),
            "cell_targets": torch.tensor(target_indices_to_cell_targets(record["target_indices"]), dtype=torch.float32),
            "object_ids": torch.tensor(object_ids, dtype=torch.long),
            "prompt": record["prompt"],
        }
