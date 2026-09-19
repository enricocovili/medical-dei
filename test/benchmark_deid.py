#!/usr/bin/env python3
"""Sweep deidentification configurations and compare accuracy metrics.

Runs the deidentification stage over a matrix of config variants (defined in
a TOML file), evaluates each against the text ground truth, and emits a
comparison table (JSON + CSV + markdown) for the thesis.

Ground truth is annotated directly on the images named by each dataset's
`images_dir`, so the sweep needs neither a GPU nor a SAM3/postprocess rerun:
the postprocess records the deidentification stage consumes are synthesised
from a directory listing. Do NOT point this at re-generated crops — the
postprocess stage rotates and crops, which invalidates the annotations.

Usage:
    python test/benchmark_deid.py --matrix setups/benchmark_matrix.toml
    python test/benchmark_deid.py --datasets panoramic --only baseline_easyocr --limit 12
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
import tomllib
import traceback
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))
sys.path.insert(0, str(REPO_ROOT / "test"))

import app as pipeline_app  # noqa: E402
from test_accuracy import (  # noqa: E402
    EllipseParams,
    build_image_size_index,
    evaluate,
    filter_degenerate_gt,
    load_json,
    parse_ground_truth,
    parse_predictions,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# The reference central ellipse, so `central_fp` stays comparable across
# variants that change the ellipse themselves.
REFERENCE_ELLIPSE = EllipseParams(axis_x_ratio=0.45, axis_y_ratio=0.35, proximity_px=0.0)

SUMMARY_METRIC_KEYS = [
    # Original keys — thesis tables reference these, do not reorder or rename.
    "gt_coverage_score",
    "gt_full_coverage_rate",
    "fully_covered",
    "total_gt",
    "uncovered",
    "outside_gt_fp",
    "total_pred",
    "outside_gt_fp_rate",
    # Coverage recall: the right recall notion for redaction.
    "recall_covered",
    "detected_gt",
    # Literature-comparable detection metrics (they penalise the intentional
    # over-redaction, so expect them to read low).
    "recall_iou",
    "precision_iou",
    "f1_iou",
    "tp",
    "fp",
    "fn",
    # Case-level safety: the PHI-leak number.
    "image_level_recall",
    "images_fully_safe",
    "images_with_gt",
    # Over-redaction.
    "clean_image_fp_rate",
    "clean_images_with_fp",
    "clean_images",
    "central_fp_ref",
    "central_over_redaction_ref",
    "redacted_area_fraction_global",
    "redacted_area_fraction_clean",
]

MARKDOWN_HEADERS = [
    "variant",
    "status",
    "recall_covered",
    "image_level_recall",
    "gt_coverage_score",
    "recall_iou",
    "gt_full_coverage_rate",
    "uncovered",
    "clean_image_fp_rate",
    "redacted_area_fraction_global",
    "mean_time_s",
]

# Every engine's tunables share a field-name prefix. Deriving the cache key from
# the prefix means a new engine's parameters are picked up automatically; an
# unregistered engine raises rather than silently reusing another engine's
# cached instance and reporting its numbers under a new name.
ENGINE_FIELD_PREFIX = {
    "easyocr": "easyocr_",
    "paddleocr": "paddleocr_",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark deidentification config variants against ground truth."
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=REPO_ROOT / "setups/benchmark_matrix.toml",
        help="TOML file with [base], [[dataset]] and [[variant]] tables.",
    )
    parser.add_argument(
        "--only", nargs="*", default=None, help="Run only these variant names."
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None, help="Run only these dataset names."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N images per dataset (CPU smoke runs).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=None,
        help="Override the matrix [base] iou_threshold.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Also save redacted images per variant (large output).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _hashable(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def _engine_cache_key(config: Any) -> tuple:
    name = config.ocr_engine
    prefix = ENGINE_FIELD_PREFIX.get(name)
    if prefix is None:
        raise ValueError(
            f"No cache-key prefix registered for ocr_engine '{name}'. Add it to "
            "ENGINE_FIELD_PREFIX in test/benchmark_deid.py, otherwise variants "
            "will silently reuse another engine's cached instance."
        )
    return (
        name,
        *sorted(
            (field.name, _hashable(getattr(config, field.name)))
            for field in dataclass_fields(config)
            if field.name.startswith(prefix)
        ),
    )


def _records_from_images_dir(images_dir: Path, limit: int | None) -> List[Dict[str, Any]]:
    """Synthesise postprocess records from a directory listing.

    run_deidentification_stage reads exactly two fields per record: `name` and
    `crop_relpath` (resolved against postprocess_crops_dir). Everything else in
    the postprocess schema is only consumed by the report stage, which the
    benchmark never runs — so nothing needs to be written to disk.
    """
    paths = sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit:
        paths = paths[:limit]
    records = []
    for path in paths:
        relpath = path.relative_to(images_dir).as_posix()
        records.append({"name": relpath, "crop_relpath": relpath})
    return records


def _datasets(matrix: Dict[str, Any], base: Dict[str, Any]) -> List[Dict[str, Any]]:
    """[[dataset]] tables when present, else one dataset built from [base].

    The [base] fallback keeps single-dataset matrices from before the
    [[dataset]] section working unchanged.
    """
    entries = matrix.get("dataset")
    if isinstance(entries, list) and entries:
        return entries
    return [{"name": "default", **base}]


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _markdown_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(MARKDOWN_HEADERS) + " |",
        "| " + " | ".join("---" for _ in MARKDOWN_HEADERS) + " |",
    ]
    for row in rows:
        cells = []
        for header in MARKDOWN_HEADERS:
            value = row.get(header, "")
            cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _write_dataset_summary(
    out_dir: Path, summary: Dict[str, Any], rows: List[Dict[str, Any]]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(_markdown_table(rows), encoding="utf-8")

    fields = [
        "dataset",
        "variant",
        "status",
        *SUMMARY_METRIC_KEYS,
        "mean_time_s",
        "wall_time_s",
        "images",
        "overrides",
    ]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["overrides"] = json.dumps(row.get("overrides", {}))
            writer.writerow(csv_row)


def run_variant(
    variant: Dict[str, Any],
    dataset: Dict[str, Any],
    base_section: Dict[str, Any],
    artifacts_root: Path,
    crops_dir: Path,
    postprocess_records: List[Dict[str, Any]] | None,
    postprocess_records_json: Path,
    ground_truth: Dict[str, List],
    image_sizes: Dict[str, tuple],
    margin_px: float,
    iou_threshold: float,
    engine_cache: Dict[tuple, Any],
    logger: Any,
    save_images: bool,
) -> tuple[Dict[str, Any], Dict[str, Any] | None]:
    name = str(variant["name"])
    overrides = {key: value for key, value in variant.items() if key != "name"}

    section: Dict[str, object] = dict(base_section)
    section.update(overrides)
    config = pipeline_app.build_config_from_dict(section)

    variant_dir = artifacts_root / name
    artifacts = pipeline_app.StageArtifacts(
        sam3_masks_dir=variant_dir / "sam3" / "masks",
        sam3_records_json=variant_dir / "sam3" / "records.json",
        postprocess_crops_dir=crops_dir,
        postprocess_records_json=postprocess_records_json,
        deid_images_dir=variant_dir / "deidentification" / "images",
        deid_records_json=variant_dir / "deidentification" / "records.json",
    )

    row: Dict[str, Any] = {
        "dataset": str(dataset.get("name", "default")),
        "name": name,
        "variant": name,
        "overrides": overrides,
    }
    start = time.perf_counter()
    try:
        key = _engine_cache_key(config)
        engine = engine_cache.get(key)
        if engine is None:
            engine = pipeline_app.build_ocr_engine(logger, config)
            engine_cache[key] = engine
        deidentifier = pipeline_app.build_deidentifier(logger, config, engine=engine)
        records = pipeline_app.run_deidentification_stage(
            artifacts,
            logger,
            deidentifier,
            postprocess_records=postprocess_records,
            save_artifacts=save_images,
            record_raw_detections=config.record_raw_detections,
        )
    except Exception as exc:  # noqa: BLE001 — keep sweeping on variant failure
        traceback.print_exc()
        row["status"] = f"failed: {type(exc).__name__}"
        return row, None

    wall_time = time.perf_counter() - start
    predictions = parse_predictions(records, min_confidence=0.0)
    variant_ellipse = EllipseParams(
        axis_x_ratio=config.ellipse_axis_x_ratio,
        axis_y_ratio=config.ellipse_axis_y_ratio,
        proximity_px=config.ellipse_proximity_px,
    )
    metrics = evaluate(
        predictions,
        ground_truth,
        margin_px,
        iou_threshold=iou_threshold,
        image_sizes=image_sizes,
        ellipse=variant_ellipse,
    )
    # Variants that move the ellipse would otherwise each measure central_fp
    # against their own geometry, making the ablation column meaningless.
    reference = evaluate(
        predictions,
        ground_truth,
        margin_px,
        iou_threshold=iou_threshold,
        image_sizes=image_sizes,
        ellipse=REFERENCE_ELLIPSE,
    )
    metrics["central_fp_ref"] = reference["central_fp"]
    metrics["central_over_redaction_ref"] = reference["central_over_redaction"]

    stage_times = [float(record.get("deidentification_time", 0.0)) for record in records]
    row["status"] = "ok"
    for metric_key in SUMMARY_METRIC_KEYS:
        row[metric_key] = metrics[metric_key]
    row["mean_time_s"] = sum(stage_times) / len(stage_times) if stage_times else 0.0
    row["wall_time_s"] = wall_time
    row["images"] = len(records)

    # Persist the full metrics next to the predictions so thesis figures can be
    # regenerated without re-running OCR.
    metrics_path = variant_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        f"    recall_covered={metrics['recall_covered']:.4f} "
        f"recall_iou={metrics['recall_iou']:.4f} "
        f"image_level_recall={metrics['image_level_recall']:.4f} "
        f"coverage={metrics['gt_coverage_score']:.4f} "
        f"clean_fp={metrics['clean_image_fp_rate']:.4f} "
        f"area={metrics['redacted_area_fraction_global']:.4f} "
        f"mean_time={row['mean_time_s']:.2f}s"
    )
    return row, metrics


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = pipeline_app.PipelineEventLogger(logging.getLogger("benchmark"))

    with args.matrix.open("rb") as file_handle:
        matrix = tomllib.load(file_handle)

    base = matrix.get("base")
    if not isinstance(base, dict):
        raise ValueError(f"Missing [base] section in {args.matrix}")
    variants = matrix.get("variant")
    if not isinstance(variants, list) or not variants:
        raise ValueError(f"Missing [[variant]] entries in {args.matrix}")
    for variant in variants:
        if not isinstance(variant, dict) or "name" not in variant:
            raise ValueError("Each [[variant]] must have a 'name' key")

    if args.only:
        wanted = set(args.only)
        variants = [v for v in variants if v.get("name") in wanted]
        if not variants:
            raise ValueError(f"No variants match --only {args.only}")

    datasets = _datasets(matrix, base)
    if args.datasets:
        wanted = set(args.datasets)
        datasets = [d for d in datasets if str(d.get("name", "default")) in wanted]
        if not datasets:
            raise ValueError(f"No datasets match --datasets {args.datasets}")

    out_root = _resolve(str(base["out_root"]))
    margin_px = float(base.get("margin_px", 5.0))
    iou_threshold = float(
        args.iou_threshold
        if args.iou_threshold is not None
        else base.get("iou_threshold", 0.5)
    )
    min_gt_side_px = float(base.get("min_gt_side_px", 1.0))
    base_section = pipeline_app._load_pipeline_section()

    engine_cache: Dict[tuple, Any] = {}
    all_rows: List[Dict[str, Any]] = []

    for dataset in datasets:
        dataset_name = str(dataset.get("name", "default"))
        ground_truth_path = _resolve(str(dataset["ground_truth"]))
        ground_truth, dropped = filter_degenerate_gt(
            parse_ground_truth(load_json(ground_truth_path)), min_gt_side_px
        )
        for image, rect in dropped:
            logging.warning(
                "dropped degenerate GT box in %s: %.3fx%.3f px (annotation artefact)",
                image,
                rect[2] - rect[0],
                rect[3] - rect[1],
            )

        if dataset.get("images_dir"):
            crops_dir = _resolve(str(dataset["images_dir"]))
            postprocess_records = _records_from_images_dir(crops_dir, args.limit)
            # Never read: records are passed in memory.
            postprocess_records_json = crops_dir / "__synthesised__.json"
        else:
            crops_dir = _resolve(str(dataset["crops_dir"]))
            postprocess_records_json = _resolve(str(dataset["postprocess_records"]))
            postprocess_records = None

        image_sizes = build_image_size_index(crops_dir)
        artifacts_root = out_root / dataset_name

        print(f"\n########## Dataset: {dataset_name} ##########")
        print(f"  images      : {crops_dir}")
        print(f"  ground truth: {ground_truth_path}")
        print(
            f"  GT boxes    : {sum(len(v) for v in ground_truth.values())} "
            f"across {sum(1 for v in ground_truth.values() if v)} images "
            f"({len(image_sizes)} images on disk)"
        )

        rows: List[Dict[str, Any]] = []
        details: List[Dict[str, Any]] = []
        for variant in variants:
            print(f"=== [{dataset_name}] Variant: {variant['name']} ===")
            row, metrics = run_variant(
                variant,
                dataset,
                base_section,
                artifacts_root,
                crops_dir,
                postprocess_records,
                postprocess_records_json,
                ground_truth,
                image_sizes,
                margin_px,
                iou_threshold,
                engine_cache,
                logger,
                args.save_images,
            )
            rows.append(row)
            all_rows.append(row)
            if metrics is not None:
                details.append({"name": row["name"], "metrics": metrics})

        summary = {
            "matrix": str(args.matrix),
            "dataset": dataset_name,
            "ground_truth": str(ground_truth_path),
            "images_dir": str(crops_dir),
            "margin_px": margin_px,
            "iou_threshold": iou_threshold,
            "min_gt_side_px": min_gt_side_px,
            "degenerate_gt_dropped": len(dropped),
            "image_limit": args.limit,
            "git_commit": _git_commit(),
            "rows": rows,
        }
        _write_dataset_summary(artifacts_root, summary, rows)
        (artifacts_root / "summary_per_image.json").write_text(
            json.dumps(details, indent=2), encoding="utf-8"
        )

    out_root.mkdir(parents=True, exist_ok=True)
    combined_fields = [
        "dataset",
        "variant",
        "status",
        *SUMMARY_METRIC_KEYS,
        "mean_time_s",
        "wall_time_s",
        "images",
        "overrides",
    ]
    with (out_root / "summary_all.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=combined_fields, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            csv_row = dict(row)
            csv_row["overrides"] = json.dumps(row.get("overrides", {}))
            writer.writerow(csv_row)

    print(f"\nSummaries written under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
