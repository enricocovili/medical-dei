#!/usr/bin/env python3
"""Sweep deidentification configurations and compare accuracy metrics.

Runs the deidentification stage over a matrix of config variants (defined in
a TOML file), evaluates each against the text ground truth, and emits a
comparison table (JSON + CSV + markdown) for the thesis.

Upstream sam3/postprocess artifacts are treated as frozen inputs: ground-truth
boxes live in crop coordinates, so the postprocess crops/records must be the
same ones the ground truth was annotated on.

Usage (inside the container):
    python test/benchmark_deid.py --matrix setups/benchmark_matrix.toml
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import tomllib
import traceback
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))
sys.path.insert(0, str(REPO_ROOT / "test"))

import app as pipeline_app  # noqa: E402
from test_accuracy import (  # noqa: E402
    evaluate,
    load_json,
    parse_ground_truth,
    parse_predictions,
)

SUMMARY_METRIC_KEYS = [
    "gt_coverage_score",
    "gt_full_coverage_rate",
    "fully_covered",
    "total_gt",
    "uncovered",
    "outside_gt_fp",
    "total_pred",
    "outside_gt_fp_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark deidentification config variants against ground truth."
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=REPO_ROOT / "setups/benchmark_matrix.toml",
        help="TOML file with [base] settings and [[variant]] config overrides.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only the variants with these names.",
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


def _engine_cache_key(config: Any) -> tuple:
    if config.ocr_engine == "easyocr":
        return (
            "easyocr",
            tuple(config.easyocr_langs),
            config.easyocr_gpu,
            config.easyocr_text_threshold,
            config.easyocr_low_text,
            config.easyocr_link_threshold,
            config.easyocr_canvas_size,
            config.easyocr_mag_ratio,
        )
    return (
        "paddleocr",
        config.paddleocr_device,
        config.paddleocr_det_model,
        config.paddleocr_rec_model,
    )


def _markdown_table(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "variant",
        "status",
        "gt_coverage_score",
        "gt_full_coverage_rate",
        "uncovered",
        "outside_gt_fp_rate",
        "outside_gt_fp",
        "total_pred",
        "mean_time_s",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


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

    crops_dir = _resolve(str(base["crops_dir"]))
    postprocess_records = _resolve(str(base["postprocess_records"]))
    ground_truth_path = _resolve(str(base["ground_truth"]))
    out_root = _resolve(str(base["out_root"]))
    margin_px = float(base.get("margin_px", 5.0))

    ground_truth = parse_ground_truth(load_json(ground_truth_path))
    base_section = pipeline_app._load_pipeline_section()

    if args.only:
        variants = [v for v in variants if v.get("name") in set(args.only)]
        if not variants:
            raise ValueError(f"No variants match --only {args.only}")

    engine_cache: dict[tuple, Any] = {}
    rows: List[Dict[str, Any]] = []
    results_detail: List[Dict[str, Any]] = []

    for variant in variants:
        if not isinstance(variant, dict) or "name" not in variant:
            raise ValueError("Each [[variant]] must have a 'name' key")
        name = str(variant["name"])
        overrides = {key: value for key, value in variant.items() if key != "name"}

        section: dict[str, object] = dict(base_section)
        section.update(overrides)
        config = pipeline_app.build_config_from_dict(section)

        variant_dir = out_root / name
        artifacts = pipeline_app.StageArtifacts(
            sam3_masks_dir=variant_dir / "sam3" / "masks",
            sam3_records_json=variant_dir / "sam3" / "records.json",
            postprocess_crops_dir=crops_dir,
            postprocess_records_json=postprocess_records,
            deid_images_dir=variant_dir / "deidentification" / "images",
            deid_records_json=variant_dir / "deidentification" / "records.json",
        )

        print(f"=== Variant: {name} ===")
        row: Dict[str, Any] = {"name": name, "variant": name, "overrides": overrides}
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
                save_artifacts=args.save_images,
                record_raw_detections=config.record_raw_detections,
            )
        except Exception as exc:  # noqa: BLE001 — keep sweeping on variant failure
            traceback.print_exc()
            row["status"] = f"failed: {type(exc).__name__}"
            rows.append(row)
            continue

        wall_time = time.perf_counter() - start
        predictions = parse_predictions(records, min_confidence=0.0)
        metrics = evaluate(predictions, ground_truth, margin_px)

        stage_times = [float(r.get("deidentification_time", 0.0)) for r in records]
        mean_time = sum(stage_times) / len(stage_times) if stage_times else 0.0

        row["status"] = "ok"
        for key_name in SUMMARY_METRIC_KEYS:
            row[key_name] = metrics[key_name]
        row["mean_time_s"] = mean_time
        row["wall_time_s"] = wall_time
        row["images"] = len(records)
        rows.append(row)
        results_detail.append({"name": name, "metrics": metrics})

        print(
            f"    coverage={metrics['gt_coverage_score']:.4f} "
            f"fully_covered={metrics['fully_covered']}/{metrics['total_gt']} "
            f"fp_rate={metrics['outside_gt_fp_rate']:.4f} "
            f"mean_time={mean_time:.2f}s"
        )

    out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "matrix": str(args.matrix),
        "ground_truth": str(ground_truth_path),
        "crops_dir": str(crops_dir),
        "postprocess_records": str(postprocess_records),
        "margin_px": margin_px,
        "rows": rows,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out_root / "summary_per_image.json").write_text(
        json.dumps(results_detail, indent=2), encoding="utf-8"
    )

    csv_fields = [
        "variant",
        "status",
        *SUMMARY_METRIC_KEYS,
        "mean_time_s",
        "wall_time_s",
        "images",
        "overrides",
    ]
    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["overrides"] = json.dumps(row.get("overrides", {}))
            writer.writerow(csv_row)

    (out_root / "summary.md").write_text(_markdown_table(rows), encoding="utf-8")

    print(f"\nSummary written to {out_root}/summary.{{json,csv,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
