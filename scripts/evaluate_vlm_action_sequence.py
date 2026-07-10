from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from src.multimodal_captcha.streamlit_action_demo import draw_action_demo_overlay, load_action_demo_records
from src.multimodal_captcha.vlm_baseline import QWEN_BASE_URL, QWEN_MODEL, QwenVlmBaseline, score_vlm_predictions


def class_name(record: dict) -> str:
    return str(record.get("target_object") or record.get("target_class_key") or "unknown")


def predict_record(record: dict, data_dir: Path, baseline: QwenVlmBaseline | None, mock_oracle: bool) -> dict:
    target_indices = [int(value) for value in record.get("target_indices", [])]
    if mock_oracle:
        return {
            "provider": "mock_oracle",
            "model": "mock_oracle",
            "image": record["image"],
            "prompt": record["prompt"],
            "target_object": class_name(record),
            "target_indices": target_indices,
            "predicted_indices": target_indices,
            "raw_response": json.dumps({"cells": target_indices}, ensure_ascii=False),
        }
    if baseline is None:
        raise RuntimeError("baseline is required unless --mock-oracle is set.")
    prediction = baseline.predict(data_dir / record["image"], record["prompt"])
    return {
        "provider": prediction.provider,
        "model": prediction.model,
        "image": record["image"],
        "prompt": record["prompt"],
        "target_object": class_name(record),
        "target_indices": target_indices,
        "predicted_indices": prediction.predicted_indices,
        "raw_response": prediction.raw_response,
    }


def load_existing_predictions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row.get("image"))] = row
    return rows


def build_run_signature(
    data_dir: Path,
    split: str,
    provider: str,
    model: str,
    base_url: str | None,
    records: list[dict],
) -> dict:
    fingerprint_rows = [
        {
            "image": record.get("image"),
            "prompt": record.get("prompt"),
            "target_object": record.get("target_object"),
            "target_class_key": record.get("target_class_key"),
            "target_indices": record.get("target_indices", []),
        }
        for record in records
    ]
    payload = json.dumps(fingerprint_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "data_dir": str(data_dir.resolve()),
        "split": split,
        "records": len(records),
        "records_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def validate_resume_config(path: Path, expected: dict) -> None:
    if not path.exists():
        raise ValueError(f"resume configuration mismatch: missing {path}")
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        changed = sorted(key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key))
        raise ValueError("resume configuration mismatch: changed fields: " + ", ".join(changed))


def write_failures(data_dir: Path, predictions: list[dict], failure_dir: Path, max_failures: int) -> list[dict]:
    failure_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    if max_failures <= 0:
        return failures
    for idx, row in enumerate(predictions):
        target = sorted(int(value) for value in row.get("target_indices", []))
        predicted = sorted(int(value) for value in row.get("predicted_indices", []))
        if target == predicted:
            continue
        image = Image.open(data_dir / row["image"]).convert("RGB")
        vis = draw_action_demo_overlay(image, predicted, target, seed=idx)
        output = failure_dir / f"failure_{len(failures):03d}.png"
        vis.save(output)
        failures.append(
            {
                "image": str(output),
                "source_image": row["image"],
                "prompt": row["prompt"],
                "target_object": row["target_object"],
                "target_indices": target,
                "predicted_indices": predicted,
            }
        )
        if len(failures) >= max_failures:
            break
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a VLM API baseline on RoboTest click-all action data.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", default="outputs/vlm_action_eval")
    parser.add_argument("--provider", choices=["qwen"], default="qwen")
    parser.add_argument("--model", default=QWEN_MODEL)
    parser.add_argument("--base-url", default=QWEN_BASE_URL)
    parser.add_argument("--api-key", default=None, help="Defaults to DASHSCOPE_API_KEY.")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-samples", type=int, default=0, help="0 evaluates the full split.")
    parser.add_argument("--max-failures", type=int, default=24)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true", help="Reuse existing predictions.jsonl rows by image path.")
    parser.add_argument("--mock-oracle", action="store_true", help="Use target cells as predictions for tests.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    run_config_path = output_dir / "run_config.json"

    records = load_action_demo_records(data_dir, split=args.split, limit=None)
    if args.max_samples > 0:
        records = records[: args.max_samples]
    if not records:
        raise SystemExit(f"No records found for split '{args.split}' in {data_dir}.")

    baseline = None
    provider = "mock_oracle" if args.mock_oracle else args.provider
    model = "mock_oracle" if args.mock_oracle else args.model
    if not args.mock_oracle:
        baseline = QwenVlmBaseline(api_key=args.api_key, base_url=args.base_url, model=args.model, timeout=args.timeout)
        if not baseline.api_key:
            raise SystemExit("Qwen VLM evaluation requires DASHSCOPE_API_KEY or --api-key. Use --mock-oracle for smoke tests.")

    run_config = build_run_signature(
        data_dir=data_dir,
        split=args.split,
        provider=provider,
        model=model,
        base_url=None if args.mock_oracle else args.base_url,
        records=records,
    )
    if args.resume and (predictions_path.exists() or run_config_path.exists()):
        try:
            validate_resume_config(run_config_path, run_config)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        run_config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    existing = load_existing_predictions(predictions_path) if args.resume else {}
    predictions = []
    start = time.time()
    with predictions_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for idx, record in enumerate(records, start=1):
            key = str(record["image"])
            if key in existing:
                row = existing[key]
            else:
                row = predict_record(record, data_dir=data_dir, baseline=baseline, mock_oracle=args.mock_oracle)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            predictions.append(row)
            if args.progress_every > 0 and (idx == 1 or idx % args.progress_every == 0 or idx == len(records)):
                elapsed = max(time.time() - start, 1e-6)
                print(
                    json.dumps(
                        {
                            "status": "progress",
                            "done": idx,
                            "total": len(records),
                            "examples_per_sec": round(idx / elapsed, 3),
                            "provider": provider,
                            "model": model,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    metrics = {
        "provider": provider,
        "model": model,
        "split": args.split,
        "data_dir": str(data_dir),
        "run_config": str(run_config_path),
        **score_vlm_predictions(predictions),
    }
    failures = write_failures(data_dir, predictions, output_dir / "failures", max_failures=args.max_failures)
    metrics["failures_saved"] = len(failures)
    metrics["failures"] = failures

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: value for key, value in metrics.items() if key not in {"per_class", "failures"}}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
