#!/usr/bin/env python3
"""Which variant finds the text the others miss.

Standalone recall is the wrong way to choose an ensemble member. What matters
is *marginal* recall: the GT boxes a variant covers that the current best does
not. A mediocre engine whose errors are decorrelated from the leader adds more
than a strong engine that fails on exactly the same images.

Reads the records.json files a benchmark sweep already wrote, so it costs no
OCR time.

    python test/analyze_engine_overlap.py --dataset panoramic
    python test/analyze_engine_overlap.py --dataset panoramic --variants baseline_easyocr rapidocr onnxtr
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

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

DATASET_GROUND_TRUTH = {
    "panoramic": "imgs/sam3_processed_panoramic/test_dataset_text_groundtruth.json",
    "teleradiography": "imgs/teleradiography_with_text/groundtruth.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="panoramic", choices=list(DATASET_GROUND_TRUTH))
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "thesis/thesis_data_out/ocr_benchmarks",
    )
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Defaults to the dataset's image folder; used to drop blank GT boxes.",
    )
    parser.add_argument("--keep-blank-gt", action="store_true")
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.999999,
        help="Coverage at which a GT box counts as found (default: fully covered).",
    )
    parser.add_argument("--margin-px", type=float, default=5.0)
    return parser.parse_args()


def found_set(records_path: Path, ground_truth, margin_px: float, threshold: float) -> set:
    """The GT boxes this variant covers, as (image, box index) keys."""
    predictions = parse_predictions(load_json(records_path), min_confidence=0.0)
    found = set()
    for image, gt_boxes in ground_truth.items():
        preds = predictions.get(image, [])
        for index, gt_box in enumerate(gt_boxes):
            if gt_union_coverage_ratio(gt_box, preds, margin_px) >= threshold:
                found.add((image, index))
    return found


def main() -> int:
    args = parse_args()
    gt_path = args.ground_truth or REPO_ROOT / DATASET_GROUND_TRUTH[args.dataset]
    ground_truth, _ = filter_degenerate_gt(parse_ground_truth(load_json(gt_path)))
    # Match the benchmark's filtering, or the denominators disagree.
    if not args.keep_blank_gt:
        images_dir = args.images_dir or gt_path.parent / (
            "imgs" if args.dataset == "panoramic" else "."
        )
        ground_truth, blank = filter_blank_gt(
            ground_truth, build_image_index(images_dir)
        )
        if blank:
            print(f"(dropped {len(blank)} blank GT boxes over uniform pixels)")
    total_gt = sum(len(boxes) for boxes in ground_truth.values())

    dataset_root = args.results_root / args.dataset
    candidates = sorted(
        path.parent.parent
        for path in dataset_root.glob("*/deidentification/records.json")
    )
    if args.variants:
        wanted = set(args.variants)
        candidates = [path for path in candidates if path.name in wanted]
    if not candidates:
        raise SystemExit(f"No variant records found under {dataset_root}")

    found = {
        path.name: found_set(
            path / "deidentification" / "records.json",
            ground_truth,
            args.margin_px,
            args.coverage_threshold,
        )
        for path in candidates
    }

    print(f"dataset: {args.dataset}  |  {total_gt} GT boxes  |  coverage >= {args.coverage_threshold}")
    print(f"\n{'variant':<34}{'found':>7}{'recall':>9}{'unique':>8}")
    print("-" * 58)
    for name in sorted(found, key=lambda n: -len(found[n])):
        others = set().union(*(v for k, v in found.items() if k != name)) if len(found) > 1 else set()
        unique = len(found[name] - others)
        print(f"{name:<34}{len(found[name]):>7}{len(found[name]) / total_gt:>9.3f}{unique:>8}")
    print("\n'unique' = GT boxes only this variant finds. A variant with a low")
    print("recall but a high unique count is still a good ensemble member.")

    if len(found) > 1:
        best = max(found, key=lambda n: len(found[n]))
        print(f"\nMarginal recall over the best single variant ({best}, {len(found[best])} found):")
        print(f"{'variant':<34}{'adds':>6}{'union':>7}{'union recall':>14}")
        print("-" * 61)
        for name in sorted(found, key=lambda n: -len(found[n] - found[best])):
            if name == best:
                continue
            adds = found[name] - found[best]
            union = found[best] | found[name]
            print(f"{name:<34}{len(adds):>6}{len(union):>7}{len(union) / total_gt:>14.3f}")

        print("\nPairwise union recall (top pairs):")
        pairs = sorted(
            combinations(found, 2), key=lambda p: -len(found[p[0]] | found[p[1]])
        )[:8]
        for a, b in pairs:
            union = found[a] | found[b]
            print(f"  {a} + {b}: {len(union)}/{total_gt} = {len(union) / total_gt:.3f}")

        everything = set().union(*found.values())
        print(
            f"\nUnion of ALL {len(found)} variants: {len(everything)}/{total_gt} = "
            f"{len(everything) / total_gt:.3f}  (the attainable ceiling for this set)"
        )
        missed = total_gt - len(everything)
        if missed:
            print(f"{missed} GT boxes no variant found:")
            for image, boxes in ground_truth.items():
                for index in range(len(boxes)):
                    if (image, index) not in everything:
                        box = boxes[index]
                        print(
                            f"  {image} #{index}  "
                            f"{box[2] - box[0]:.0f}x{box[3] - box[1]:.0f} px"
                        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
