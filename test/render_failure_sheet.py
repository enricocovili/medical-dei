#!/usr/bin/env python3
"""Contact sheet of the ground-truth boxes a configuration failed to cover.

A full-image overlay per scan is the right artifact for checking one case, but
there are ~200 of them and the interesting content is a handful of text strips.
This crops just the boxes that were missed (plus context) and tiles them, so
the whole failure set can be reviewed on one page.

Green = ground truth, red = whatever the pipeline predicted nearby, and the
caption gives the image, the box size and how much of it was covered.

    python test/render_failure_sheet.py --variant onnxtr_clahe_recall_max
    python test/render_failure_sheet.py --variant onnxtr_clahe_recall_max \
        --dataset teleradiography --max-coverage 0.999
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))

from test_accuracy import (  # noqa: E402
    build_image_index,
    filter_blank_gt,
    filter_degenerate_gt,
    gt_union_coverage_ratio,
    load_json,
    parse_ground_truth,
    parse_predictions,
)

DATASETS = {
    "panoramic": (
        "imgs/sam3_processed_panoramic/imgs",
        "imgs/sam3_processed_panoramic/test_dataset_text_groundtruth.json",
    ),
    "teleradiography": (
        "imgs/teleradiography_with_text",
        "imgs/teleradiography_with_text/groundtruth.json",
    ),
}

GREEN, RED, BG, TEXT = (0, 220, 0), (0, 0, 255), (32, 32, 32), (0, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="onnxtr_clahe_recall_max")
    parser.add_argument("--dataset", nargs="*", default=list(DATASETS))
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "thesis/thesis_data_out/ocr_benchmarks",
    )
    parser.add_argument(
        "--max-coverage",
        type=float,
        default=0.999999,
        help="Include GT boxes covered at or below this (default: anything not fully covered).",
    )
    parser.add_argument("--margin-px", type=float, default=5.0)
    parser.add_argument("--context-px", type=int, default=24)
    parser.add_argument("--tile-width", type=int, default=760)
    parser.add_argument("--per-sheet", type=int, default=14)
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "thesis/thesis_data_out/overlays"
    )
    return parser.parse_args()


def crop_with_boxes(
    image: np.ndarray, gt_box, preds, context: int, width: int
) -> np.ndarray:
    height, image_width = image.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in gt_box)
    cx1, cy1 = max(0, x1 - context), max(0, y1 - context)
    cx2, cy2 = min(image_width, x2 + context), min(height, y2 + context)
    crop = image[cy1:cy2, cx1:cx2].copy()
    if crop.size == 0:
        return np.zeros((10, width, 3), dtype=np.uint8)
    for px1, py1, px2, py2 in preds:
        cv2.rectangle(
            crop,
            (int(px1) - cx1, int(py1) - cy1),
            (int(px2) - cx1, int(py2) - cy1),
            RED,
            2,
        )
    cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), GREEN, 2)
    scale = width / crop.shape[1]
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def main() -> int:
    args = parse_args()
    total = 0
    for dataset in args.dataset:
        images_dir, gt_path = DATASETS[dataset]
        index = build_image_index(REPO_ROOT / images_dir)
        ground_truth, _ = filter_degenerate_gt(
            parse_ground_truth(load_json(REPO_ROOT / gt_path))
        )
        ground_truth, _ = filter_blank_gt(ground_truth, index)

        records = args.results_root / dataset / args.variant / "deidentification/records.json"
        if not records.exists():
            print(f"(no predictions for {dataset}/{args.variant})")
            continue
        predictions = parse_predictions(load_json(records), min_confidence=0.0)

        tiles = []
        for name in sorted(ground_truth):
            boxes = ground_truth[name]
            if not boxes or name not in index:
                continue
            image = None
            preds = predictions.get(name, [])
            for i, gt_box in enumerate(boxes):
                coverage = gt_union_coverage_ratio(gt_box, preds, args.margin_px)
                if coverage > args.max_coverage:
                    continue
                if image is None:
                    image = cv2.imread(str(index[name]), cv2.IMREAD_COLOR)
                    if image is None:
                        break
                tile = crop_with_boxes(
                    image, gt_box, preds, args.context_px, args.tile_width
                )
                caption = (
                    f"{name} #{i}  {int(gt_box[2]-gt_box[0])}x{int(gt_box[3]-gt_box[1])}px"
                    f"  covered {coverage:.0%}"
                )
                tile = cv2.copyMakeBorder(tile, 22, 6, 0, 0, cv2.BORDER_CONSTANT, value=BG)
                cv2.putText(tile, caption, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT, 1)
                tiles.append(tile)

        if not tiles:
            print(f"{dataset}: nothing below the coverage threshold — no sheet written")
            continue
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for sheet_index in range(0, len(tiles), args.per_sheet):
            chunk = tiles[sheet_index : sheet_index + args.per_sheet]
            widest = max(t.shape[1] for t in chunk)
            padded = [
                cv2.copyMakeBorder(t, 0, 0, 0, widest - t.shape[1], cv2.BORDER_CONSTANT, value=BG)
                for t in chunk
            ]
            out = args.out_dir / (
                f"{dataset}_{args.variant}_misses_{sheet_index // args.per_sheet + 1}.png"
            )
            cv2.imwrite(str(out), np.vstack(padded))
            print(f"  wrote {out}  ({len(chunk)} boxes)")
        print(f"{dataset}: {len(tiles)} boxes below {args.max_coverage:.0%} coverage")
        total += len(tiles)
    print(f"\n{total} missed boxes rendered for '{args.variant}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
