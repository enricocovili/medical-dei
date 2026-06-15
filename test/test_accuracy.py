#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2

Rect = Tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare OCR detections against text ground-truth."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("imgs/output/deidentification/records.json"),
        help="Path to predicted deidentification records.json (or legacy labels.json).",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("imgs/test_dataset_text_groundtruth.json"),
        help="Path to test_dataset_text_groundtruth.json.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Ignore predictions with confidence lower than this value (legacy format).",
    )
    parser.add_argument(
        "--margin-px",
        type=float,
        default=5.0,
        help="Pixel tolerance margin used in coverage/outside-GT calculations.",
    )
    parser.add_argument(
        "--per-image",
        action="store_true",
        help="Print per-image coverage and outside-GT false positive details.",
    )
    parser.add_argument(
        "--save-overlay-dir",
        type=Path,
        default=None,
        help="If set, save annotated images (GT green, predictions red) into this folder.",
    )
    parser.add_argument(
        "--cropped-image-dir",
        type=Path,
        default=None,
        help="the dir to get the cropped image from before overlaying the GT and predictions. If not set, the script will look for the original images in the same folder as the ground-truth JSON.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_rect(points: Sequence[Sequence[float]]) -> Rect:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def xywh_to_rect(box: Sequence[float]) -> Rect:
    x, y, w, h = [float(v) for v in box]
    return x, y, x + w, y + h


def parse_box_payload(payload: Any) -> Rect | None:
    if not isinstance(payload, list):
        return None
    if len(payload) == 4 and all(isinstance(v, (int, float)) for v in payload):
        x, y, w, h = [float(v) for v in payload]
        if w <= 0 or h <= 0:
            return None
        return xywh_to_rect(payload)
    if len(payload) >= 2 and all(
        isinstance(point, (list, tuple)) and len(point) >= 2 for point in payload
    ):
        return normalize_rect(payload)
    return None


def rect_area(rect: Rect) -> float:
    x1, y1, x2, y2 = rect
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: Rect, b: Rect) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    return rect_area((inter_x1, inter_y1, inter_x2, inter_y2))


def expand_rect(rect: Rect, margin_px: float) -> Rect:
    x1, y1, x2, y2 = rect
    return x1 - margin_px, y1 - margin_px, x2 + margin_px, y2 + margin_px


def parse_predictions(raw: Any, min_confidence: float) -> Dict[str, List[Rect]]:
    grouped: Dict[str, List[Rect]] = {}

    if isinstance(raw, dict):
        for image_name, detections in raw.items():
            if not isinstance(detections, list):
                continue
            image_rects: List[Rect] = []
            for payload in detections:
                if not isinstance(payload, dict):
                    continue
                confidence = float(payload.get("confidence", 0.0))
                if confidence < min_confidence:
                    continue
                rect = parse_box_payload(payload.get("boxes", []))
                if rect is not None:
                    image_rects.append(rect)
            grouped[Path(image_name).stem] = image_rects
        return grouped

    if isinstance(raw, list):
        for record in raw:
            if not isinstance(record, dict):
                continue
            image_name = record.get("name")
            if not isinstance(image_name, str) or not image_name.strip():
                continue
            key = Path(image_name).stem
            image_rects = grouped.setdefault(key, [])
            boxes = record.get("boxes", [])
            if not isinstance(boxes, list):
                continue
            for box in boxes:
                rect = parse_box_payload(box)
                if rect is not None:
                    image_rects.append(rect)
        return grouped

    raise ValueError("Predictions JSON must be either a dict (legacy) or a list (records).")


def parse_ground_truth(raw: Dict) -> Dict[str, List[Rect]]:
    grouped: Dict[str, List[Rect]] = {}
    for image_name, payload in raw.items():
        shapes = payload.get("shapes", []) if isinstance(payload, dict) else []
        rects: List[Rect] = []
        for shape in shapes:
            points = shape.get("points", []) if isinstance(shape, dict) else []
            if len(points) < 2:
                continue
            rects.append(normalize_rect(points))
        grouped[image_name] = rects
    return grouped


def gt_best_coverage_ratio(
    gt_box: Rect, pred_boxes: List[Rect], margin_px: float
) -> float:
    gt_area = rect_area(gt_box)
    if gt_area <= 0:
        return 0.0
    best = 0.0
    for pred_box in pred_boxes:
        inter = intersection_area(gt_box, expand_rect(pred_box, margin_px))
        coverage = min(1.0, inter / gt_area)
        if coverage > best:
            best = coverage
    return best


def pred_is_outside_ground_truth(
    pred_box: Rect, gt_boxes: List[Rect], margin_px: float
) -> bool:
    px1, py1, px2, py2 = pred_box
    for gt_box in gt_boxes:
        gx1, gy1, gx2, gy2 = expand_rect(gt_box, margin_px)
        touches_or_overlaps = px1 <= gx2 and px2 >= gx1 and py1 <= gy2 and py2 >= gy1
        if touches_or_overlaps:
            return False
    return True


def build_image_index(search_root: Path) -> Dict[str, Path]:
    supported_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    index: Dict[str, Path] = {}
    if not search_root.exists():
        return index
    for path in sorted(search_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in supported_ext:
            continue
        index.setdefault(path.stem, path)
    return index


def draw_rect(img, rect: Rect, color: Tuple[int, int, int], thickness: int = 2) -> None:
    x1, y1, x2, y2 = rect
    h, w = img.shape[:2]
    p1 = (max(0, min(w - 1, int(round(x1)))), max(0, min(h - 1, int(round(y1)))))
    p2 = (max(0, min(w - 1, int(round(x2)))), max(0, min(h - 1, int(round(y2)))))
    cv2.rectangle(img, p1, p2, color, thickness)


def save_overlay_image(
    image_path: Path, out_path: Path, preds: List[Rect], gts: List[Rect]
) -> bool:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return False
    for gt_box in gts:
        draw_rect(img, gt_box, color=(0, 255, 0), thickness=2)
    for pred_box in preds:
        draw_rect(img, pred_box, color=(0, 0, 255), thickness=2)
    return bool(cv2.imwrite(str(out_path), img))


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def main() -> None:
    args = parse_args()
    raw_predictions = load_json(args.predictions)
    raw_ground_truth = load_json(args.ground_truth)
    if not isinstance(raw_predictions, (dict, list)):
        raise ValueError("Predictions JSON must be either a dict or a list.")
    if not isinstance(raw_ground_truth, dict):
        raise ValueError("Ground-truth JSON must be a dict keyed by image name.")

    predictions = parse_predictions(raw_predictions, min_confidence=args.min_confidence)
    ground_truth = parse_ground_truth(raw_ground_truth)

    all_images = sorted(set(ground_truth.keys()) | set(predictions.keys()))

    total_gt_boxes = 0
    fully_covered_gt_boxes = 0
    uncovered_gt_boxes = 0
    gt_coverage_sum = 0.0

    total_pred_boxes = 0
    outside_gt_fp_boxes = 0

    images_with_not_fully_covered: List[str] = []
    not_fully_covered_count_by_image: Dict[str, int] = {}

    image_index: Dict[str, Path] = {}
    overlays_saved = 0
    overlays_missing_source = 0
    overlays_write_fail = 0
    if args.save_overlay_dir is not None:
        args.save_overlay_dir.mkdir(parents=True, exist_ok=True)
        image_index = build_image_index(args.cropped_image_dir)

    print("=== Dataset structure ===")
    if isinstance(raw_predictions, dict):
        print(
            f"predictions: dict[{len(raw_predictions)}] -> "
            "list[{bbox: list[4 points], text: str, confidence: float}]"
        )
    else:
        print(
            f"predictions: list[{len(raw_predictions)}] -> "
            "{name: str, deidentification_boxes: list[[x, y, w, h]]}"
        )
    print(
        f"ground-truth: dict[{len(raw_ground_truth)}] -> "
        "{shapes: list[{label: str, points: [[x1,y1],[x2,y2]], shape_type: str}]}"
    )
    print()

    for image in all_images:
        preds = predictions.get(image, [])
        gts = ground_truth.get(image, [])

        image_coverage_sum = 0.0
        image_fully_covered = 0
        image_uncovered = 0

        for gt_box in gts:
            coverage = gt_best_coverage_ratio(gt_box, preds, args.margin_px)
            image_coverage_sum += coverage
            if coverage >= 0.999999:
                image_fully_covered += 1
            if coverage <= 0.0:
                image_uncovered += 1

        image_outside_fp = sum(
            1
            for pred_box in preds
            if pred_is_outside_ground_truth(pred_box, gts, args.margin_px)
        )

        if args.per_image:
            image_gt_count = len(gts)
            image_mean_cov = safe_div(image_coverage_sum, image_gt_count)
            print(
                f"{image}: gt={image_gt_count} pred={len(preds)} "
                f"mean_gt_coverage={image_mean_cov:.4f} "
                f"fully_covered_gt={image_fully_covered}/{image_gt_count} "
                f"outside_gt_fp={image_outside_fp}"
            )

        total_gt_boxes += len(gts)
        fully_covered_gt_boxes += image_fully_covered
        uncovered_gt_boxes += image_uncovered
        gt_coverage_sum += image_coverage_sum

        total_pred_boxes += len(preds)
        outside_gt_fp_boxes += image_outside_fp

        image_not_fully_covered = len(gts) - image_fully_covered
        if image_not_fully_covered > 0:
            images_with_not_fully_covered.append(image)
            not_fully_covered_count_by_image[image] = image_not_fully_covered

        if args.save_overlay_dir is not None:
            image_path = image_index.get(image)
            if image_path is None:
                overlays_missing_source += 1
            else:
                out_name = f"{image_path.stem}_overlay{image_path.suffix.lower()}"
                out_path = args.save_overlay_dir / out_name
                if save_overlay_image(image_path, out_path, preds, gts):
                    print(f"Saved overlay for {image} to {out_path}")
                    overlays_saved += 1
                else:
                    overlays_write_fail += 1

    gt_coverage_score = safe_div(gt_coverage_sum, total_gt_boxes)
    gt_full_coverage_rate = safe_div(fully_covered_gt_boxes, total_gt_boxes)
    outside_gt_fp_rate = safe_div(outside_gt_fp_boxes, total_pred_boxes)

    print("=== Ground-truth coverage metrics ===")
    print(f"Margin tolerance: {args.margin_px:.1f}px")
    print(f"Total GT boxes: {total_gt_boxes}")
    print(f"GT coverage score (mean covered area): {gt_coverage_score:.4f}")
    print(
        f"Fully covered GT boxes: {fully_covered_gt_boxes}/{total_gt_boxes} "
        f"({gt_full_coverage_rate:.4f})"
    )
    print(f"Uncovered GT boxes: {uncovered_gt_boxes}")
    print()
    print("=== False positives outside GT ===")
    print(
        f"Outside-GT predictions: {outside_gt_fp_boxes}/{total_pred_boxes} "
        f"({outside_gt_fp_rate:.4f})"
    )
    print()
    print("=== Images with not fully covered GT boxes ===")
    if images_with_not_fully_covered:
        for image in images_with_not_fully_covered:
            image_gt_count = len(ground_truth.get(image, []))
            image_not_fully_covered = not_fully_covered_count_by_image[image]
            print(
                f"{image}: {image_not_fully_covered}/{image_gt_count} not fully covered"
            )
    else:
        print("None")

    if args.save_overlay_dir is not None:
        print()
        print("=== Overlay export ===")
        print(f"Output folder: {args.save_overlay_dir}")
        print(f"Saved overlays: {overlays_saved}")
        print(f"Missing source images: {overlays_missing_source}")
        print(f"Write failures: {overlays_write_fail}")


if __name__ == "__main__":
    main()
