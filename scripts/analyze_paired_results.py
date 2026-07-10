from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("cell_exact_match", "cell_precision", "cell_recall", "click_order_accuracy")


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def selected_class_maps(path: str | Path) -> tuple[dict[str, str], dict[str, int]]:
    data = load_json(path)
    key_to_name = {}
    source_counts = {}
    for item in data.get("classes", []):
        class_key = item.get("class_key")
        object_name = item.get("object_name")
        if class_key and object_name:
            key_to_name[class_key] = object_name
            source_counts[class_key] = item.get("source_images")
    return key_to_name, source_counts


def delta(clean: float | None, dirty: float | None) -> float | None:
    if isinstance(clean, int | float) and isinstance(dirty, int | float):
        return clean - dirty
    return None


def class_row(class_key: str, object_name: str, dirty_class: dict, clean_class: dict, source_count: int | None) -> dict:
    row = {
        "class_key": class_key,
        "object_name": object_name,
        "source_images": source_count,
        "dirty_total": dirty_class.get("total"),
        "clean_total": clean_class.get("total"),
    }
    for metric in METRICS:
        dirty_value = dirty_class.get(metric)
        clean_value = clean_class.get(metric)
        row[f"dirty_{metric}"] = dirty_value
        row[f"clean_{metric}"] = clean_value
        row[f"delta_{metric}"] = delta(clean_value, dirty_value)
    return row


def best_threshold(metrics: dict, key: str = "cell_exact_match") -> dict | None:
    candidates = metrics.get("threshold_candidates") or []
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item.get(key, 0.0),
            item.get("click_order_accuracy", 0.0),
            item.get("cell_precision", 0.0),
        ),
    )


def classify_failure(failure: dict) -> str:
    target = set(map(int, failure.get("target_indices", [])))
    predicted = set(map(int, failure.get("predicted_indices", [])))
    missing = target - predicted
    extra = predicted - target
    if missing and not extra:
        return "missing_targets"
    if extra and not missing:
        return "extra_targets"
    return "mixed_or_wrong"


def summarize_failures(metrics: dict) -> dict:
    by_object: dict[str, dict[str, int]] = {}
    error_types = {"missing_targets": 0, "extra_targets": 0, "mixed_or_wrong": 0}
    for failure in metrics.get("failures", []):
        error_type = classify_failure(failure)
        error_types[error_type] += 1
        target_object = str(failure.get("target_object") or "unknown")
        row = by_object.setdefault(
            target_object,
            {"total": 0, "missing_targets": 0, "extra_targets": 0, "mixed_or_wrong": 0},
        )
        row["total"] += 1
        row[error_type] += 1
    return {
        "total_saved": len(metrics.get("failures", [])),
        "error_types": error_types,
        "by_object": dict(sorted(by_object.items(), key=lambda item: (-item[1]["total"], item[0]))),
    }


def analyze(args: argparse.Namespace) -> dict:
    dirty = load_json(args.dirty_metrics)
    clean = load_json(args.clean_metrics)
    key_to_name, source_counts = selected_class_maps(args.selected_classes)
    dirty_per_class = dirty.get("per_class", {})
    clean_per_class = clean.get("per_class", {})

    rows = []
    for class_key, object_name in key_to_name.items():
        rows.append(
            class_row(
                class_key,
                object_name,
                dirty_per_class.get(object_name, {}),
                clean_per_class.get(object_name, {}),
                source_counts.get(class_key),
            )
        )
    rows.sort(key=lambda item: (item["delta_cell_exact_match"] is None, item["delta_cell_exact_match"] or 0), reverse=True)

    focus = {}
    for class_key in [value.strip() for value in args.focus_classes.split(",") if value.strip()]:
        object_name = key_to_name.get(class_key, class_key)
        focus[class_key] = class_row(
            class_key,
            object_name,
            dirty_per_class.get(object_name, {}),
            clean_per_class.get(object_name, {}),
            source_counts.get(class_key),
        )

    global_metrics = {}
    for metric in METRICS:
        dirty_value = dirty.get(metric)
        clean_value = clean.get(metric)
        global_metrics[metric] = {
            "dirty": dirty_value,
            "clean": clean_value,
            "delta": delta(clean_value, dirty_value),
        }

    return {
        "global": global_metrics,
        "thresholds": {
            "dirty": {
                "selected_threshold": dirty.get("threshold"),
                "best_exact_match": best_threshold(dirty),
            },
            "clean": {
                "selected_threshold": clean.get("threshold"),
                "best_exact_match": best_threshold(clean),
            },
        },
        "failure_summary": {
            "dirty": summarize_failures(dirty),
            "clean": summarize_failures(clean),
        },
        "focus_classes": focus,
        "per_class": rows,
        "best_improved_classes": rows[:10],
        "most_regressed_classes": list(reversed(rows[-10:])),
    }


def fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int | float):
        return f"{value:.4f}"
    return str(value)


def write_outputs(summary: dict, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Paired Dirty/Clean Analysis",
        "",
        "## Global Metrics",
        "| Metric | Dirty | Clean | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric, values in summary["global"].items():
        lines.append(f"| `{metric}` | {fmt(values['dirty'])} | {fmt(values['clean'])} | {fmt(values['delta'])} |")

    lines.extend(["", "## Thresholds", "| Split | Selected | Best Exact Threshold | Best Exact | Precision | Recall |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for split in ("dirty", "clean"):
        best = summary["thresholds"][split]["best_exact_match"] or {}
        lines.append(
            f"| `{split}` | {fmt(summary['thresholds'][split]['selected_threshold'])} | "
            f"{fmt(best.get('threshold'))} | {fmt(best.get('cell_exact_match'))} | "
            f"{fmt(best.get('cell_precision'))} | {fmt(best.get('cell_recall'))} |"
        )

    lines.extend(["", "## Failure Summary", "| Split | Saved | Missing | Extra | Mixed/Wrong |", "| --- | ---: | ---: | ---: | ---: |"])
    for split in ("dirty", "clean"):
        failure_summary = summary["failure_summary"][split]
        error_types = failure_summary["error_types"]
        lines.append(
            f"| `{split}` | {failure_summary['total_saved']} | {error_types['missing_targets']} | "
            f"{error_types['extra_targets']} | {error_types['mixed_or_wrong']} |"
        )

    lines.extend(["", "## Focus Classes", "| Class | Object | Dirty Exact | Clean Exact | Delta Exact | Dirty Recall | Clean Recall |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for class_key, values in summary["focus_classes"].items():
        lines.append(
            f"| `{class_key}` | {values['object_name']} | {fmt(values['dirty_cell_exact_match'])} | "
            f"{fmt(values['clean_cell_exact_match'])} | {fmt(values['delta_cell_exact_match'])} | "
            f"{fmt(values['dirty_cell_recall'])} | {fmt(values['clean_cell_recall'])} |"
        )

    lines.extend(["", "## Best Improved Classes", "| Class | Object | Delta Exact | Dirty Exact | Clean Exact |", "| --- | --- | ---: | ---: | ---: |"])
    for row in summary["best_improved_classes"]:
        lines.append(
            f"| `{row['class_key']}` | {row['object_name']} | {fmt(row['delta_cell_exact_match'])} | "
            f"{fmt(row['dirty_cell_exact_match'])} | {fmt(row['clean_cell_exact_match'])} |"
        )

    lines.extend(["", "## Most Regressed Classes", "| Class | Object | Delta Exact | Dirty Exact | Clean Exact |", "| --- | --- | ---: | ---: | ---: |"])
    for row in summary["most_regressed_classes"]:
        lines.append(
            f"| `{row['class_key']}` | {row['object_name']} | {fmt(row['delta_cell_exact_match'])} | "
            f"{fmt(row['dirty_cell_exact_match'])} | {fmt(row['clean_cell_exact_match'])} |"
        )

    (output / "paired_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired dirty/clean action-sequence results.")
    parser.add_argument("--dirty-metrics", required=True)
    parser.add_argument("--clean-metrics", required=True)
    parser.add_argument("--selected-classes", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--focus-classes", default="dress,tent,boat,person,toy")
    args = parser.parse_args()
    write_outputs(analyze(args), args.output_dir)


if __name__ == "__main__":
    main()
