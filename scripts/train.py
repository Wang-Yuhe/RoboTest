from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.multimodal_captcha.dataset import CaptchaDataset, build_object_vocab, build_vocab
from src.multimodal_captcha.generator import generate_dataset
from src.multimodal_captcha.model import MultimodalGridLocator


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device), batch["text"].to(device))
            pred = logits.argmax(dim=1).cpu()
            correct += int((pred == batch["target"]).sum())
            total += len(pred)
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a multimodal grid locator.")
    parser.add_argument("--data-dir", default="data/synthetic_captcha")
    parser.add_argument("--output", default="outputs/model.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--patience", type=int, default=0, help="Early-stop patience. 0 disables early stopping.")
    parser.add_argument("--model-size", choices=["small", "base"], default="small")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    data_dir = Path(args.data_dir)
    if not (data_dir / "manifest.jsonl").exists():
        print("Dataset not found. Generating 600 samples first...", flush=True)
        generate_dataset(data_dir, 600, seed=args.seed, image_size=192)

    vocab = build_vocab(data_dir / "manifest.jsonl")
    object_vocab = build_object_vocab(data_dir / "manifest.jsonl")
    train_set = CaptchaDataset(data_dir, split="train", vocab=vocab, object_vocab=object_vocab, augment=not args.no_augment)
    val_set = CaptchaDataset(data_dir, split="val", vocab=vocab, object_vocab=object_vocab)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"hidden_dim": 96, "image_size": 64, "base_channels": 24}
    if args.model_size == "base":
        model_kwargs = {"hidden_dim": 128, "image_size": 96, "base_channels": 32}
    model = MultimodalGridLocator(vocab_size=len(vocab), object_vocab_size=len(object_vocab), **model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    loss_fn = nn.CrossEntropyLoss()
    object_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    best_acc = -1.0
    best_state = None
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            logits, object_logits = model(batch["image"].to(device), batch["text"].to(device), return_aux=True)
            loss = loss_fn(logits, batch["target"].to(device))
            if object_logits is not None:
                object_loss = object_loss_fn(
                    object_logits.reshape(-1, object_logits.shape[-1]),
                    batch["object_ids"].to(device).reshape(-1),
                )
                loss = loss + args.aux_weight * object_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running += float(loss.item()) * len(batch["target"])
        train_loss = running / len(train_set)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"epoch={epoch} loss={train_loss:.4f} val_acc={val_acc:.3f}", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if args.patience > 0 and bad_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs. best_val_acc={best_acc:.3f}", flush=True)
                break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model": model.state_dict(),
            "vocab": vocab,
            "object_vocab": object_vocab,
            "best_val_acc": best_acc,
            "model_config": {
                "vocab_size": len(vocab),
                "object_vocab_size": len(object_vocab),
                **model_kwargs,
                "model_size": args.model_size,
            },
        },
        output,
    )
    (output.parent / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    (output.parent / "object_vocab.json").write_text(json.dumps(object_vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved checkpoint to {output}", flush=True)


if __name__ == "__main__":
    main()
