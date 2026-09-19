#!/usr/bin/env python3
"""Recompute benchmark metrics from predictions already on disk.

Every variant's records.json is the expensive artifact; the metrics derived
from it are cheap. When the scoring changes — a new metric, a ground-truth
correction, a fixed denominator — this replays the existing predictions
instead of re-running hours of OCR.

    python test/rescore_benchmark.py
    python test/rescore_benchmark.py --dataset panoramic --sort recall_covered
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tomllib
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))

from benchmark_deid import (  # noqa: E402
    MARKDOWN_HEADERS,
    REFERENCE_ELLIPSE,
    SUMMARY_METRIC_KEYS,
    _markdown_table,
)
from test_accuracy import (  # noqa: E402
    EllipseParams,
    build_image_index,
    build_image_size_index,
    evaluate,
    filter_blank_gt,
    filter_degenerate_gt,
    load_json,
    parse_ground_truth,
    parse_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "thesis/thesis_data_out/ocr_benchmarks",
    )
    parser.add_argument("--dataset", nargs="*", default=None)
    parser.add_argument("--margin-px", type=float, default=5.0)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--coverage-threshold", type=float, default=0.5)
    parser.add_argument("--keep-blank-gt", action="store_true")
    parser.add_argument("--sort", default="recall_covered")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=REPO_ROOT / "setups/benchmark_matrix.toml",
        help="Used to locate a dataset's images when its sweep has not finished.",
    )
    return parser.parse_args()


def _matrix_datasets(matrix_path: Path) -> Dict[str, Dict[str, Any]]:
    if not matrix_path.exists():
        return {}
    matrix = tomllib.loads(matrix_path.read_text(encoding="utf-8"))
    entries = matrix.get("dataset") or []
    return {str(entry.get("name", "default")): entry for entry in entries}


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


REPORT_COLUMNS = [
    ("recall_covered", "recall"),
    ("gt_full_coverage_rate_union", "fully cov"),
    ("image_level_recall", "img recall"),
    ("uncovered", "uncov"),
    ("total_pred", "preds"),
    ("clean_image_fp_rate", "cleanFP"),
    ("central_over_redaction_ref", "anatFP"),
    ("redacted_area_fraction_global", "area%"),
]


def _report_section(dataset: str, total_gt: int, rows: List[Dict[str, Any]]) -> str:
    """One markdown table per dataset, ordered by the headline recall."""
    labels = [label for _, label in REPORT_COLUMNS]
    lines = [
        f"## {dataset} ({total_gt} findable GT boxes)",
        "",
        "| variant | " + " | ".join(labels) + " |",
        "|---" * (len(labels) + 1) + "|",
    ]
    for row in rows:
        cells = []
        for key, _ in REPORT_COLUMNS:
            value = row.get(key, 0)
            if key == "redacted_area_fraction_global":
                cells.append(f"{value * 100:.2f}")
            elif isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append(f"| `{row['variant']}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    from_matrix = _matrix_datasets(args.matrix)
    report: List[str] = []

    # A dataset's summary.json only appears once its sweep finishes, so fall
    # back to the matrix for one that is still running (or was interrupted) --
    # its per-variant records are already on disk and perfectly scoreable.
    dataset_dirs = sorted(
        d for d in args.results_root.iterdir()
        if d.is_dir() and any(d.glob("*/deidentification/records.json"))
    )
    for dataset_dir in dataset_dirs:
        dataset = dataset_dir.name
        if args.dataset and dataset not in args.dataset:
            continue
        summary_path = dataset_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            images_dir = Path(summary["images_dir"])
            ground_truth_path = Path(summary["ground_truth"])
        elif dataset in from_matrix:
            entry = from_matrix[dataset]
            images_dir = _resolve(str(entry.get("images_dir") or entry["crops_dir"]))
            ground_truth_path = _resolve(str(entry["ground_truth"]))
            summary = {
                "dataset": dataset,
                "images_dir": str(images_dir),
                "ground_truth": str(ground_truth_path),
                "partial": True,
            }
            print(f"(no summary.json for {dataset}; using the matrix — sweep still running?)")
        else:
            print(f"(skipping {dataset}: no summary.json and no matching [[dataset]])")
            continue

        ground_truth, degenerate = filter_degenerate_gt(
            parse_ground_truth(load_json(ground_truth_path)),
            float(summary.get("min_gt_side_px", 1.0)),
        )
        blank: List[Any] = []
        if not args.keep_blank_gt:
            ground_truth, blank = filter_blank_gt(
                ground_truth, build_image_index(images_dir)
            )
        image_sizes = build_image_size_index(images_dir)
        annotated_only = bool(from_matrix.get(dataset, {}).get("annotated_only"))
        annotated = set(ground_truth)
        if annotated_only:
            # Mirror the sweep: images absent from the ground truth are
            # unannotated, not verified text-free (see the matrix comment).
            # Predictions on them must be dropped too, or evaluate() pulls them
            # back into the image universe via the prediction keys.
            image_sizes = {k: v for k, v in image_sizes.items() if k in annotated}
        total_gt = sum(len(boxes) for boxes in ground_truth.values())
        print(
            f"=== {dataset}: {total_gt} GT boxes "
            f"(dropped {len(degenerate)} degenerate, {len(blank)} blank)"
        )

        by_variant = {row["variant"]: row for row in summary.get("rows", [])}
        rows: List[Dict[str, Any]] = []
        for records_path in sorted(dataset_dir.glob("*/deidentification/records.json")):
            variant = records_path.parent.parent.name
            records = load_json(records_path)
            predictions = parse_predictions(records, min_confidence=0.0)
            if annotated_only:
                predictions = {
                    image: boxes
                    for image, boxes in predictions.items()
                    if image in annotated
                }
            overrides = by_variant.get(variant, {}).get("overrides", {})
            ellipse = EllipseParams(
                axis_x_ratio=float(overrides.get("ellipse_axis_x_ratio", 0.45)),
                axis_y_ratio=float(overrides.get("ellipse_axis_y_ratio", 0.35)),
                proximity_px=float(overrides.get("ellipse_proximity_px", 0.0)),
            )
            common = dict(
                iou_threshold=args.iou_threshold,
                coverage_threshold=args.coverage_threshold,
                image_sizes=image_sizes,
            )
            metrics = evaluate(
                predictions, ground_truth, args.margin_px, ellipse=ellipse, **common
            )
            reference = evaluate(
                predictions,
                ground_truth,
                args.margin_px,
                ellipse=REFERENCE_ELLIPSE,
                **common,
            )
            metrics["central_fp_ref"] = reference["central_fp"]
            metrics["central_over_redaction_ref"] = reference["central_over_redaction"]
            metrics["degenerate_gt_dropped"] = len(degenerate)
            metrics["blank_gt_dropped"] = len(blank)
            (records_path.parent.parent / "metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )

            row: Dict[str, Any] = {
                "dataset": dataset,
                "variant": variant,
                "name": variant,
                "status": "ok",
                "overrides": overrides,
            }
            row.update({key: metrics[key] for key in SUMMARY_METRIC_KEYS})
            for carried in ("mean_time_s", "wall_time_s", "images", "engine_calls"):
                if carried in by_variant.get(variant, {}):
                    row[carried] = by_variant[variant][carried]
            rows.append(row)

        rows.sort(key=lambda r: -r.get(args.sort, 0.0))
        summary["rows"] = rows
        summary["rescored"] = True
        summary["blank_gt_dropped"] = len(blank)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (dataset_dir / "summary.md").write_text(_markdown_table(rows), encoding="utf-8")
        fields = ["dataset", "variant", "status", *SUMMARY_METRIC_KEYS, "mean_time_s", "images", "overrides"]
        with (dataset_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                csv_row = dict(row)
                csv_row["overrides"] = json.dumps(row.get("overrides", {}))
                writer.writerow(csv_row)

        columns = REPORT_COLUMNS
        report.append(_report_section(dataset, total_gt, rows))
        header = f"{'variant':<30}" + "".join(f"{label:>11}" for _, label in columns)
        print(header)
        print("-" * len(header))
        for row in rows:
            cells = ""
            for key, _ in columns:
                value = row.get(key, 0)
                if key == "redacted_area_fraction_global":
                    cells += f"{value * 100:>11.2f}"
                elif isinstance(value, float):
                    cells += f"{value:>11.3f}"
                else:
                    cells += f"{value:>11}"
            print(f"{row['variant']:<30}{cells}")
        print()

    if report:
        out = args.results_root / "RESULTS.md"
        out.write_text(
            "# OCR configuration benchmark\n\n"
            "Generated by `test/rescore_benchmark.py` from each variant's saved\n"
            "predictions. See CLAUDE.md for what each metric means, which to quote,\n"
            "and the ground-truth caveats behind the box counts.\n\n"
            + "\n".join(report),
            encoding="utf-8",
        )
        print(f"Combined report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
