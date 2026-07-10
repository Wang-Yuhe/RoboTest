from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.multimodal_captcha.action_sequence import (
    ACTION_VOCAB_SIZE,
    cell_logits_to_actions,
    actions_to_tokens,
    compute_action_metrics,
)
from src.multimodal_captcha.dataset import ActionSequenceDataset, build_object_vocab, build_vocab
from src.multimodal_captcha.generator import generate_action_dataset
from src.multimodal_captcha.model import ActionCellSelector


def emit_log(record: dict, log_file=None) -> None:
    text = json.dumps(record, ensure_ascii=False)
    print(text, flush=True)
    if log_file is not None:
        log_file.write(text + "\n")
        log_file.flush()


def resolve_encoder_train_mode(requested_mode: str, image_encoder: str, pretrained: bool) -> str:
    if requested_mode != "auto":
        return requested_mode
    if image_encoder in {"resnet18", "clip_vit_b32"} and pretrained:
        return "frozen"
    return "full"


def should_update_best_checkpoint(score: float, loss: float, best_score: float, best_loss: float) -> bool:
    if score > best_score:
        return True
    return score == best_score and loss < best_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    exact = 0
    true_positive = 0
    predicted_positive = 0
    target_positive = 0
    action_predictions = []
    action_targets = []
    total = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device), batch["text"].to(device))
            pred = (torch.sigmoid(logits).cpu() >= 0.5).float()
            target = batch["cell_targets"].cpu()
            exact += int((pred == target).all(dim=1).sum())
            true_positive += int(((pred == 1) & (target == 1)).sum())
            predicted_positive += int((pred == 1).sum())
            target_positive += int((target == 1).sum())
            total += target.shape[0]
            action_predictions.extend([actions_to_tokens(actions) for actions in cell_logits_to_actions(logits)])
            action_targets.extend(batch["action_targets"].tolist())
    metrics = compute_action_metrics(action_predictions, action_targets)
    metrics["cell_exact_match"] = exact / max(total, 1)
    metrics["cell_precision"] = true_positive / max(predicted_positive, 1)
    metrics["cell_recall"] = true_positive / max(target_positive, 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an action-sequence model for click-all CAPTCHA tasks.")
    parser.add_argument("--data-dir", default="data/action_captcha")
    parser.add_argument("--output", default="outputs/action_model.pt")
    parser.add_argument("--num-samples", type=int, default=2000, help="Generated when data-dir has no manifest.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--max-action-len", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--min-targets", type=int, default=2)
    parser.add_argument("--max-targets", type=int, default=4)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--patience", type=int, default=0, help="Early-stop patience. 0 disables early stopping.")
    parser.add_argument("--model-size", choices=["small", "base"], default="small")
    parser.add_argument("--image-encoder", choices=["custom", "resnet18", "clip_vit_b32"], default="custom")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights when supported by image encoder.")
    parser.add_argument(
        "--encoder-train-mode",
        choices=["auto", "full", "frozen", "last_block"],
        default="auto",
        help="Which image encoder parameters to train. auto uses frozen for pretrained resnet18/clip_vit_b32 and full otherwise.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--log-output", default=None, help="JSONL metric log path. Defaults to <checkpoint>.log.jsonl.")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Limit train records for smoke tests. 0 uses all.")
    parser.add_argument("--max-val-samples", type=int, default=0, help="Limit val records for smoke tests. 0 uses all.")
    parser.add_argument("--use-count-head", action="store_true", help="Predict click count and train a top-k count head.")
    parser.add_argument("--max-count", type=int, default=4, help="Maximum click-count class for --use-count-head.")
    parser.add_argument("--count-loss-weight", type=float, default=0.2, help="Loss weight for the count head.")
    args = parser.parse_args()
    requested_encoder_train_mode = args.encoder_train_mode
    args.encoder_train_mode = resolve_encoder_train_mode(args.encoder_train_mode, args.image_encoder, args.pretrained)

    seed_everything(args.seed)
    torch.set_num_threads(2)
    data_dir = Path(args.data_dir)
    if not (data_dir / "manifest.jsonl").exists():
        generate_action_dataset(
            data_dir,
            num_samples=args.num_samples,
            seed=args.seed,
            image_size=args.image_size,
            min_targets=args.min_targets,
            max_targets=args.max_targets,
        )

    vocab = build_vocab(data_dir / "manifest.jsonl")
    object_vocab = build_object_vocab(data_dir / "manifest.jsonl")
    train_set = ActionSequenceDataset(
        data_dir,
        split="train",
        vocab=vocab,
        object_vocab=object_vocab,
        max_action_len=args.max_action_len,
        augment=not args.no_augment,
    )
    val_set = ActionSequenceDataset(
        data_dir,
        split="val",
        vocab=vocab,
        object_vocab=object_vocab,
        max_action_len=args.max_action_len,
    )
    if args.max_train_samples > 0:
        train_set.records = train_set.records[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_set.records = val_set.records[: args.max_val_samples]
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
    model = ActionCellSelector(
        vocab_size=len(vocab),
        object_vocab_size=len(object_vocab),
        image_encoder=args.image_encoder,
        pretrained=args.pretrained,
        encoder_train_mode=args.encoder_train_mode,
        use_count_head=args.use_count_head,
        max_count=args.max_count,
        **model_kwargs,
    ).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters remain after applying --encoder-train-mode.")

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
                "image_encoder": args.image_encoder,
                "pretrained": args.pretrained,
                "encoder_train_mode": args.encoder_train_mode,
                "requested_encoder_train_mode": requested_encoder_train_mode,
                "use_count_head": args.use_count_head,
                "max_count": args.max_count,
                "count_loss_weight": args.count_loss_weight,
                "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
                "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "device": device,
                "action_vocab_size": ACTION_VOCAB_SIZE,
                "object_vocab_size": len(object_vocab),
                "max_action_len": args.max_action_len,
            },
            log_file,
        )
        optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
        loss_fn = nn.BCEWithLogitsLoss()
        object_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        count_loss_fn = nn.CrossEntropyLoss()
        best_score = -1.0
        best_loss = float("inf")
        best_state = None
        bad_epochs = 0
        total_batches = len(train_loader)
        stopped_epoch = args.epochs

        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            token_count = 0
            epoch_start = time.time()
            for batch_idx, batch in enumerate(train_loader, start=1):
                targets = batch["cell_targets"].to(device)
                model_output = model(batch["image"].to(device), batch["text"].to(device), return_aux=True)
                if args.use_count_head:
                    logits, object_logits, count_logits = model_output
                else:
                    logits, object_logits = model_output
                    count_logits = None
                loss = loss_fn(logits, targets)
                if object_logits is not None:
                    object_loss = object_loss_fn(
                        object_logits.reshape(-1, object_logits.shape[-1]),
                        batch["object_ids"].to(device).reshape(-1),
                    )
                    loss = loss + args.aux_weight * object_loss
                if count_logits is not None:
                    target_counts = targets.sum(dim=1).long().clamp(0, args.max_count)
                    loss = loss + args.count_loss_weight * count_loss_fn(count_logits, target_counts)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                examples = int(targets.shape[0])
                running += float(loss.item()) * examples
                token_count += examples
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

            train_loss = running / max(token_count, 1)
            emit_log({"status": "evaluating", "epoch": epoch}, log_file)
            metrics = evaluate(model, val_loader, device)
            scheduler.step()
            score = metrics["cell_exact_match"]
            current_best = max(best_score, score)
            emit_log(
                {
                    "status": "epoch_done",
                    "epoch": epoch,
                    "loss": round(train_loss, 4),
                    "exact_match": round(metrics["exact_match"], 4),
                    "click_order_accuracy": round(metrics["click_order_accuracy"], 4),
                    "cell_exact_match": round(metrics["cell_exact_match"], 4),
                    "cell_precision": round(metrics["cell_precision"], 4),
                    "cell_recall": round(metrics["cell_recall"], 4),
                    "best_cell_exact_match": round(current_best, 4),
                    "lr": round(float(scheduler.get_last_lr()[0]), 8),
                },
                log_file,
            )
            if should_update_best_checkpoint(score, train_loss, best_score, best_loss):
                best_score = score
                best_loss = train_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if args.patience > 0 and bad_epochs >= args.patience:
                    stopped_epoch = epoch
                    emit_log({"status": "early_stop", "epoch": epoch, "best_cell_exact_match": round(best_score, 4)}, log_file)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save(
            {
                "model": model.state_dict(),
                "vocab": vocab,
                "object_vocab": object_vocab,
                "best_cell_exact_match": best_score,
                "training_args": vars(args),
                "model_config": {
                    "architecture": "action_cell_selector",
                    "vocab_size": len(vocab),
                    "object_vocab_size": len(object_vocab),
                    "max_action_len": args.max_action_len,
                    **model_kwargs,
                    "model_size": args.model_size,
                    "image_encoder": args.image_encoder,
                    "pretrained": args.pretrained,
                    "encoder_train_mode": args.encoder_train_mode,
                    "requested_encoder_train_mode": requested_encoder_train_mode,
                    "use_count_head": args.use_count_head,
                    "max_count": args.max_count,
                },
            },
            output,
        )
        output.with_suffix(".action_vocab.json").write_text(
            json.dumps(
                {
                    "pad": 0,
                    "click": 1,
                    "done": 2,
                    **{f"move_to_cell_{idx}": idx + 3 for idx in range(9)},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        output.with_suffix(".vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
        output.with_suffix(".object_vocab.json").write_text(
            json.dumps(object_vocab, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        emit_log(
            {
                "status": "saved",
                "checkpoint": str(output),
                "log_output": str(log_path),
                "best_cell_exact_match": round(best_score, 4),
                "stopped_epoch": stopped_epoch,
            },
            log_file,
        )


if __name__ == "__main__":
    main()
