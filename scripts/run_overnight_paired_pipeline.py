from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


WEAK_CLASSES = "dress,tent,boat,person,toy"


@dataclass(frozen=True)
class PipelinePaths:
    timestamp: str
    run_dir: Path
    logs_dir: Path
    checkpoints_dir: Path
    source_review_dir: Path
    bad_source_labels: Path
    source_pool: Path
    source_split_plan: Path
    dirty_data_dir: Path
    clean_data_dir: Path
    dirty_checkpoint: Path
    clean_checkpoint: Path
    dirty_eval_dir: Path
    clean_eval_dir: Path
    dirty_metrics: Path
    clean_metrics: Path
    summary_json: Path
    summary_md: Path
    events_jsonl: Path


def build_paths(timestamp: str) -> PipelinePaths:
    run_dir = Path("outputs") / f"overnight_paired_{timestamp}"
    return PipelinePaths(
        timestamp=timestamp,
        run_dir=run_dir,
        logs_dir=run_dir / "logs",
        checkpoints_dir=run_dir / "checkpoints",
        source_review_dir=Path("outputs") / f"source_review_weak_paired_{timestamp}",
        bad_source_labels=Path("outputs") / f"source_review_weak_paired_{timestamp}" / "bad_source_labels.json",
        source_pool=Path("outputs") / f"paired80_source_pool_{timestamp}.json",
        source_split_plan=Path("outputs") / f"paired80_source_split_plan_{timestamp}.json",
        dirty_data_dir=Path("data") / f"photo_action_click_all_dirty80_paired_10k_{timestamp}",
        clean_data_dir=Path("data") / f"photo_action_click_all_clean80_paired_10k_{timestamp}",
        dirty_checkpoint=run_dir / "checkpoints" / f"action_resnet18_frozen_dirty80_paired_{timestamp}.pt",
        clean_checkpoint=run_dir / "checkpoints" / f"action_resnet18_frozen_clean80_paired_{timestamp}.pt",
        dirty_eval_dir=run_dir / "dirty_eval",
        clean_eval_dir=run_dir / "clean_eval",
        dirty_metrics=run_dir / "dirty_eval" / "test" / "metrics.json",
        clean_metrics=run_dir / "clean_eval" / "test" / "metrics.json",
        summary_json=run_dir / "summary.json",
        summary_md=run_dir / "summary.md",
        events_jsonl=run_dir / "pipeline_events.jsonl",
    )


def command_to_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) if " " not in part else f'"{part}"' for part in command)


def replace_batch_size(command: list[str], value: str) -> list[str]:
    updated = list(command)
    if "--batch-size" not in updated:
        return updated + ["--batch-size", value]
    idx = updated.index("--batch-size")
    updated[idx + 1] = value
    return updated


def append_event(paths: PipelinePaths, event: dict) -> None:
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **event}
    with paths.events_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_nightly_log(paths: PipelinePaths, title: str, lines: list[str]) -> None:
    log_path = Path("NIGHTLY_OPTIMIZATION_LOG.md")
    body = [f"\n## {time.strftime('%Y-%m-%d %H:%M')} - {title}\n"]
    body.extend(lines)
    body.append("")
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(body))


def manifest_count(data_dir: Path) -> int:
    manifest = data_dir / "manifest.jsonl"
    if not manifest.exists():
        return 0
    with manifest.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def metrics_exist(metrics_path: Path) -> bool:
    return metrics_path.exists() and metrics_path.stat().st_size > 0


def command_failed_with_oom(stdout_path: Path, stderr_path: Path) -> bool:
    text = ""
    for path in (stdout_path, stderr_path):
        if path.exists():
            text += path.read_text(encoding="utf-8", errors="replace").lower()
    return "out of memory" in text or "cuda out of memory" in text


def run_command(paths: PipelinePaths, name: str, command: list[str], skip_if: bool = False) -> None:
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = paths.logs_dir / f"{name}.stdout.log"
    stderr_path = paths.logs_dir / f"{name}.stderr.log"
    if skip_if:
        append_event(paths, {"step": name, "status": "skipped", "command": command_to_text(command)})
        return
    append_event(paths, {"step": name, "status": "started", "command": command_to_text(command)})
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, text=True)
    append_event(
        paths,
        {
            "step": name,
            "status": "finished" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        },
    )
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {name}. See {stdout_path} and {stderr_path}.")


def run_train_with_oom_retry(paths: PipelinePaths, name: str, command: list[str], checkpoint: Path, batch_size: str) -> None:
    if checkpoint.exists():
        run_command(paths, name, command, skip_if=True)
        return
    try:
        run_command(paths, name, command)
    except RuntimeError:
        stdout_path = paths.logs_dir / f"{name}.stdout.log"
        stderr_path = paths.logs_dir / f"{name}.stderr.log"
        if batch_size == "16" or not command_failed_with_oom(stdout_path, stderr_path):
            raise
        retry_name = f"{name}_retry_bs16"
        retry_command = replace_batch_size(command, "16")
        append_event(paths, {"step": name, "status": "oom_retry", "retry_command": command_to_text(retry_command)})
        run_command(paths, retry_name, retry_command)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(metrics: dict, key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if isinstance(value, int | float) else None


def per_class(metrics: dict, class_key: str) -> dict:
    raw = metrics.get("per_class", {})
    if not isinstance(raw, dict):
        return {}
    return raw.get(class_key, {})


def load_class_name_map(data_dir: Path) -> dict[str, str]:
    path = data_dir / "selected_classes.json"
    if not path.exists():
        return {}
    payload = load_json(path)
    return {
        str(item["class_key"]): str(item["object_name"])
        for item in payload.get("classes", [])
        if item.get("class_key") and item.get("object_name")
    }


def build_focus_class_summary(
    dirty: dict,
    clean: dict,
    class_name_map: dict[str, str],
    class_keys: list[str],
) -> dict[str, dict]:
    focus = {}
    for class_key in class_keys:
        object_name = class_name_map.get(class_key, class_key)
        dirty_class = per_class(dirty, object_name)
        clean_class = per_class(clean, object_name)
        dirty_exact = dirty_class.get("cell_exact_match")
        clean_exact = clean_class.get("cell_exact_match")
        focus[class_key] = {
            "object_name": object_name,
            "dirty_cell_exact_match": dirty_exact,
            "clean_cell_exact_match": clean_exact,
            "delta_cell_exact_match": (
                clean_exact - dirty_exact
                if isinstance(clean_exact, int | float) and isinstance(dirty_exact, int | float)
                else None
            ),
            "dirty_cell_recall": dirty_class.get("cell_recall"),
            "clean_cell_recall": clean_class.get("cell_recall"),
        }
    return focus


def expected_manifest_samples(args: argparse.Namespace) -> int:
    return int(args.num_train) + int(args.num_val) + int(args.num_test)


def create_summary(paths: PipelinePaths) -> None:
    dirty = load_json(paths.dirty_metrics)
    clean = load_json(paths.clean_metrics)
    class_name_map = load_class_name_map(paths.dirty_data_dir)
    class_name_map.update(load_class_name_map(paths.clean_data_dir))
    focus = build_focus_class_summary(dirty, clean, class_name_map, WEAK_CLASSES.split(","))
    summary = {
        "timestamp": paths.timestamp,
        "dirty_data_dir": str(paths.dirty_data_dir),
        "clean_data_dir": str(paths.clean_data_dir),
        "source_review_dir": str(paths.source_review_dir),
        "source_pool": str(paths.source_pool),
        "source_split_plan": str(paths.source_split_plan),
        "dirty_checkpoint": str(paths.dirty_checkpoint),
        "clean_checkpoint": str(paths.clean_checkpoint),
        "dirty_metrics": str(paths.dirty_metrics),
        "clean_metrics": str(paths.clean_metrics),
        "global": {
            key: {
                "dirty": metric_value(dirty, key),
                "clean": metric_value(clean, key),
                "delta": (
                    metric_value(clean, key) - metric_value(dirty, key)
                    if metric_value(dirty, key) is not None and metric_value(clean, key) is not None
                    else None
                ),
            }
            for key in ("cell_exact_match", "cell_precision", "cell_recall", "click_order_accuracy")
        },
        "focus_classes": focus,
        "remaining_risks": [
            "bad_source_labels.json is automatic quality filtering only; contact sheets still need manual review.",
            "A clean dataset can have fewer usable sources in affected classes because missing split-plan sources are skipped.",
            "If training stopped early, compare logs before making final model claims.",
        ],
    }
    paths.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# Overnight Paired Dirty/Clean Summary - {paths.timestamp}",
        "",
        "## Artifacts",
        f"- Source review: `{paths.source_review_dir}`",
        f"- Dirty data: `{paths.dirty_data_dir}`",
        f"- Clean data: `{paths.clean_data_dir}`",
        f"- Dirty metrics: `{paths.dirty_metrics}`",
        f"- Clean metrics: `{paths.clean_metrics}`",
        "",
        "## Global Metrics",
        "| Metric | Dirty | Clean | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, values in summary["global"].items():
        lines.append(
            f"| `{key}` | {format_float(values['dirty'])} | {format_float(values['clean'])} | {format_float(values['delta'])} |"
        )
    lines.extend(["", "## Focus Classes", "| Class | Dirty Exact | Clean Exact | Delta Exact | Dirty Recall | Clean Recall |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for class_key, values in focus.items():
        lines.append(
            f"| `{class_key}` ({values['object_name']}) | {format_float(values['dirty_cell_exact_match'])} | "
            f"{format_float(values['clean_cell_exact_match'])} | {format_float(values['delta_cell_exact_match'])} | "
            f"{format_float(values['dirty_cell_recall'])} | {format_float(values['clean_cell_recall'])} |"
        )
    lines.extend(["", "## Remaining Risks"])
    lines.extend(f"- {risk}" for risk in summary["remaining_risks"])
    paths.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def build_commands(paths: PipelinePaths, args: argparse.Namespace) -> dict[str, list[str]]:
    py = sys.executable
    return {
        "preflight_unittest": [py, "-m", "unittest", "discover", "-s", "tests", "-v"],
        "preflight_compile": [py, "-m", "compileall", "src", "scripts"],
        "source_review": [
            py,
            "scripts/review_photo_sources.py",
            "--photo-root",
            args.photo_root,
            "--classes",
            WEAK_CLASSES,
            "--output-dir",
            str(paths.source_review_dir),
            "--max-per-class",
            "240",
            "--thumbnail-size",
            "96",
            "--min-source-area",
            "5000",
            "--max-source-aspect-ratio",
            "4.0",
        ],
        "build_dirty": [
            py,
            "scripts/build_photo_action_dataset.py",
            "--photo-root",
            args.photo_root,
            "--output-dir",
            str(paths.dirty_data_dir),
            "--num-train",
            str(args.num_train),
            "--num-val",
            str(args.num_val),
            "--num-test",
            str(args.num_test),
            "--min-images-per-class",
            "40",
            "--class-list",
            args.class_list,
            "--image-size",
            "192",
            "--min-targets",
            "2",
            "--max-targets",
            "4",
            "--hard-augment",
            "--balanced-targets",
            "--write-source-pool-manifest",
            str(paths.source_pool),
            "--write-source-split-plan",
            str(paths.source_split_plan),
            "--progress-every",
            "1000",
        ],
        "build_clean": [
            py,
            "scripts/build_photo_action_dataset.py",
            "--photo-root",
            args.photo_root,
            "--output-dir",
            str(paths.clean_data_dir),
            "--source-pool-manifest",
            str(paths.source_pool),
            "--source-split-plan",
            str(paths.source_split_plan),
            "--missing-plan-source-policy",
            "skip",
            "--bad-source-labels",
            str(paths.bad_source_labels),
            "--num-train",
            str(args.num_train),
            "--num-val",
            str(args.num_val),
            "--num-test",
            str(args.num_test),
            "--image-size",
            "192",
            "--min-targets",
            "2",
            "--max-targets",
            "4",
            "--hard-augment",
            "--balanced-targets",
            "--progress-every",
            "1000",
        ],
        "train_dirty": train_command(py, paths.dirty_data_dir, paths.dirty_checkpoint, args),
        "train_clean": train_command(py, paths.clean_data_dir, paths.clean_checkpoint, args),
        "eval_dirty": eval_command(py, paths.dirty_data_dir, paths.dirty_checkpoint, paths.dirty_eval_dir, args),
        "eval_clean": eval_command(py, paths.clean_data_dir, paths.clean_checkpoint, paths.clean_eval_dir, args),
    }


def train_command(py: str, data_dir: Path, output: Path, args: argparse.Namespace) -> list[str]:
    return [
        py,
        "scripts/train_action_sequence.py",
        "--data-dir",
        str(data_dir),
        "--output",
        str(output),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        "0.0003",
        "--aux-weight",
        "0.7",
        "--model-size",
        "base",
        "--image-encoder",
        "resnet18",
        "--pretrained",
        "--encoder-train-mode",
        "frozen",
        "--device",
        args.device,
        "--patience",
        "6",
        "--progress-every",
        "20",
    ]


def eval_command(py: str, data_dir: Path, checkpoint: Path, output_dir: Path, args: argparse.Namespace) -> list[str]:
    return [
        py,
        "scripts/evaluate_action_sequence.py",
        "--data-dir",
        str(data_dir),
        "--checkpoint",
        str(checkpoint),
        "--split",
        "test",
        "--threshold",
        "auto",
        "--output-dir",
        str(output_dir),
        "--max-failures",
        "48",
        "--device",
        args.device,
        "--progress-every",
        "500",
    ]


def run_pipeline(args: argparse.Namespace) -> PipelinePaths:
    paths = build_paths(args.timestamp)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    commands = build_commands(paths, args)
    append_nightly_log(
        paths,
        f"Overnight Paired Pipeline Started {args.timestamp}",
        [
            "- Modified files:",
            "  - scripts/run_overnight_paired_pipeline.py",
            "  - tests/test_overnight_paired_pipeline.py",
            "  - NIGHTLY_OPTIMIZATION_LOG.md",
            "- Design intent:",
            "  - Run a resumable paired dirty/clean frozen ResNet18 workflow overnight.",
            "- Key commands:",
            "  ```bash",
            *[f"  {command_to_text(command)}" for command in commands.values()],
            "  ```",
            f"- Output directory: `{paths.run_dir}`",
            "- Verification result:",
            "  - pending; see pipeline summary and event logs.",
        ],
    )
    append_event(paths, {"step": "pipeline", "status": "started", "timestamp": args.timestamp})

    try:
        run_command(paths, "preflight_unittest", commands["preflight_unittest"])
        run_command(paths, "preflight_compile", commands["preflight_compile"])
        run_command(paths, "source_review", commands["source_review"], skip_if=paths.bad_source_labels.exists())
        expected_samples = expected_manifest_samples(args)
        dirty_ready = manifest_count(paths.dirty_data_dir) == expected_samples and paths.source_pool.exists() and paths.source_split_plan.exists()
        run_command(paths, "build_dirty", commands["build_dirty"], skip_if=dirty_ready)
        run_command(paths, "build_clean", commands["build_clean"], skip_if=manifest_count(paths.clean_data_dir) == expected_samples)
        run_train_with_oom_retry(paths, "train_dirty", commands["train_dirty"], paths.dirty_checkpoint, str(args.batch_size))
        run_command(paths, "eval_dirty", commands["eval_dirty"], skip_if=metrics_exist(paths.dirty_metrics))
        run_train_with_oom_retry(paths, "train_clean", commands["train_clean"], paths.clean_checkpoint, str(args.batch_size))
        run_command(paths, "eval_clean", commands["eval_clean"], skip_if=metrics_exist(paths.clean_metrics))
        create_summary(paths)
        append_event(paths, {"step": "pipeline", "status": "completed", "summary": str(paths.summary_md)})
        append_nightly_log(
            paths,
            f"Overnight Paired Pipeline Finished {args.timestamp}",
            [
                "- Verification result:",
                f"  - Summary: `{paths.summary_md}`",
                f"  - Dirty metrics: `{paths.dirty_metrics}`",
                f"  - Clean metrics: `{paths.clean_metrics}`",
                "- Remaining risks:",
                "  - Review contact sheets before treating automatic bad-source labels as final human labels.",
            ],
        )
    except Exception as exc:
        write_failure_summary(paths, exc)
        append_event(paths, {"step": "pipeline", "status": "failed", "error": str(exc)})
        append_nightly_log(
            paths,
            f"Overnight Paired Pipeline Failed {args.timestamp}",
            [
                f"- Error: `{exc}`",
                f"- Event log: `{paths.events_jsonl}`",
                f"- Partial summary: `{paths.summary_md}`",
            ],
        )
        raise
    return paths


def write_failure_summary(paths: PipelinePaths, exc: Exception) -> None:
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": paths.timestamp,
        "status": "failed",
        "error": str(exc),
        "event_log": str(paths.events_jsonl),
        "dirty_metrics": str(paths.dirty_metrics) if paths.dirty_metrics.exists() else None,
        "clean_metrics": str(paths.clean_metrics) if paths.clean_metrics.exists() else None,
    }
    paths.summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.summary_md.write_text(
        "\n".join(
            [
                f"# Overnight Paired Dirty/Clean Summary - {paths.timestamp}",
                "",
                "Status: failed",
                "",
                f"Error: `{exc}`",
                "",
                f"Event log: `{paths.events_jsonl}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paired dirty/clean action-sequence workflow overnight.")
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--photo-root", default="data/photo_objects")
    parser.add_argument("--class-list", default="outputs/failure_analysis_20260707_1853/baseline_80_classes.txt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--num-train", type=int, default=10000)
    parser.add_argument("--num-val", type=int, default=2000)
    parser.add_argument("--num-test", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
