from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.multimodal_captcha.dataset import CaptchaDataset, build_object_vocab, build_vocab
from src.multimodal_captcha.generator import generate_dataset
from src.multimodal_captcha.model import MultimodalGridLocator


def emit_log(record: dict, log_file=None) -> None:
    text = json.dumps(record, ensure_ascii=False)
    print(text, flush=True)
    if log_file is not None:
        log_file.write(text + "\n")
        log_file.flush()


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
    parser.add_argument("--model-size", choices=["small", "base", "attn"], default="small")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Training device.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress-every", type=int, default=20, help="Print training progress every N batches. 0 disables batch progress.")
    parser.add_argument("--log-output", default=None, help="JSONL metric log path. Defaults to <checkpoint>.log.jsonl.")
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

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but CUDA is not available.")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    model_kwargs = {"hidden_dim": 96, "image_size": 64, "base_channels": 24}
    if args.model_size == "base":
        model_kwargs = {"hidden_dim": 128, "image_size": 96, "base_channels": 32}
    elif args.model_size == "attn":
        model_kwargs = {
            "hidden_dim": 128,
            "image_size": 96,
            "base_channels": 32,
            "use_transformer": True,
            "use_interactions": True,
        }
    model = MultimodalGridLocator(vocab_size=len(vocab), object_vocab_size=len(object_vocab), **model_kwargs).to(device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_output) if args.log_output else output.with_suffix(".log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        emit_log(
            {
                "status": "start",
                "data_dir": str(data_dir),
                "output": str(output),
                "log_output": str(log_path),
                "train_samples": len(train_set),
                "val_samples": len(val_set),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "model_size": args.model_size,
                "device": device,
            },
            log_file,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
        loss_fn = nn.CrossEntropyLoss()
        object_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        best_acc = -1.0
        best_state = None
        bad_epochs = 0
        total_batches = len(train_loader)
        stopped_epoch = args.epochs
        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            epoch_start = time.time()
            for batch_idx, batch in enumerate(train_loader, start=1):
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
                if args.progress_every > 0 and (
                    batch_idx == 1 or batch_idx % args.progress_every == 0 or batch_idx == total_batches
                ):
                    elapsed = time.time() - epoch_start
                    batches_per_sec = batch_idx / elapsed if elapsed > 0 else 0.0
                    remaining = (total_batches - batch_idx) / batches_per_sec if batches_per_sec > 0 else 0.0
                    emit_log(
                        {
                            "status": "train_progress",
                            "epoch": epoch,
                            "epochs": args.epochs,
                            "batch": batch_idx,
                            "batches": total_batches,
                            "percent": round(batch_idx / max(total_batches, 1) * 100, 2),
                            "loss": round(float(loss.item()), 4),
                            "batches_per_sec": round(batches_per_sec, 2),
                            "eta_sec": round(remaining, 1),
                        },
                        log_file,
                    )
            train_loss = running / len(train_set)
            emit_log({"status": "evaluating", "epoch": epoch}, log_file)
            val_acc = evaluate(model, val_loader, device)
            scheduler.step()
            current_best = max(best_acc, val_acc)
            emit_log(
                {
                    "status": "epoch_done",
                    "epoch": epoch,
                    "loss": round(train_loss, 4),
                    "val_acc": round(val_acc, 4),
                    "best_val_acc": round(current_best, 4),
                    "lr": round(float(scheduler.get_last_lr()[0]), 8),
                },
                log_file,
            )
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if args.patience > 0 and bad_epochs >= args.patience:
                    stopped_epoch = epoch
                    emit_log(
                        {"status": "early_stop", "epoch": epoch, "best_val_acc": round(best_acc, 4)},
                        log_file,
                    )
                    break

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
        emit_log(
            {
                "status": "saved",
                "checkpoint": str(output),
                "log_output": str(log_path),
                "best_val_acc": round(best_acc, 4),
                "stopped_epoch": stopped_epoch,
            },
            log_file,
        )


if __name__ == "__main__":
    main()
