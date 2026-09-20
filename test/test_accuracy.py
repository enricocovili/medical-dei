#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2

Rect = Tuple[float, float, float, float]

# JPEG start-of-frame markers carrying the image dimensions (excludes 0xC4 DHT,
# 0xC8 JPG and 0xCC DAC, which are not frame headers).
_JPEG_SOF_MARKERS = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare OCR detections against text ground-truth."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "thesis/thesis_data_out/ocr_benchmarks/panoramic/"
            "baseline_easyocr/deidentification/records.json"
        ),
        help="Path to predicted deidentification records.json (or legacy labels.json).",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("imgs/sam3_processed_panoramic/test_dataset_text_groundtruth.json"),
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
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for the one-to-one detection metrics.",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.5,
        help="Fraction of a GT box that must be covered for it to count as detected.",
    )
    parser.add_argument(
        "--keep-blank-gt",
        action="store_true",
        help=(
            "Score against GT boxes whose pixels are uniform (already-blanked "
            "regions no OCR can find). Off by default; requires --images-dir."
        ),
    )
    parser.add_argument(
        "--min-gt-side-px",
        type=float,
        default=1.0,
        help="Drop GT rectangles thinner than this on either axis (annotation slips).",
    )
    parser.add_argument(
        "--ellipse-axis-x",
        type=float,
        default=0.45,
        help="Reference central-ellipse x half-axis ratio, for the central_fp metric.",
    )
    parser.add_argument(
        "--ellipse-axis-y",
        type=float,
        default=0.35,
        help="Reference central-ellipse y half-axis ratio, for the central_fp metric.",
    )
    parser.add_argument(
        "--ellipse-proximity-px",
        type=float,
        default=0.0,
        help="Reference central-ellipse proximity padding, for the central_fp metric.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write the full metrics dict (including per_image) to this JSON file.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help=(
            "Directory holding the images the ground truth was annotated on. Enables "
            "the central-ellipse and redacted-area metrics, and makes images with no "
            "ground-truth entry count as clean instead of vanishing from the totals."
        ),
    )
    parser.add_argument(
        "--cropped-image-dir",
        type=Path,
        default=None,
        help="Deprecated alias for --images-dir.",
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
                # test/mede_easyocr.py (the legacy producer) writes "bbox";
                # accept both so the legacy path is not silently empty.
                raw_box = payload.get("bbox")
                if raw_box is None:
                    raw_box = payload.get("boxes", [])
                rect = parse_box_payload(raw_box)
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


def gt_union_coverage_ratio(
    gt_box: Rect, pred_boxes: List[Rect], margin_px: float
) -> float:
    """Fraction of a GT box covered by the UNION of all predictions.

    gt_best_coverage_ratio takes the max over a single prediction, which
    understates redaction: a detector that splits one text line into two
    adjacent boxes covering it completely scores ~0.7 there, even though every
    PHI pixel is blacked out. For anonymization the union is the correct notion,
    so the safety metrics (recall_covered, image_level_recall) use this one.
    gt_best_coverage_ratio is kept for continuity of the original metric.
    """
    gt_area = rect_area(gt_box)
    if gt_area <= 0:
        return 0.0
    gx1, gy1, gx2, gy2 = gt_box
    clipped: List[Rect] = []
    for pred_box in pred_boxes:
        px1, py1, px2, py2 = expand_rect(pred_box, margin_px)
        rect = (max(px1, gx1), max(py1, gy1), min(px2, gx2), min(py2, gy2))
        if rect[2] > rect[0] and rect[3] > rect[1]:
            clipped.append(rect)
    return min(1.0, union_area(clipped) / gt_area)


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


# ---------------------------------------------------------------------------
# Geometry helpers for the recall-first metrics.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EllipseParams:
    """Mirror of the deidentifier's central keep-out zone.

    Defaults match setups/pipeline_config.toml. This duplicates
    Deidentifier._touches_center_ellipse on purpose: test_accuracy.py is
    standalone (no pipeline import), which is what lets benchmark_deid.py
    import it before any pipeline module is on sys.path. test_ellipse_parity
    asserts the two implementations agree.
    """

    axis_x_ratio: float = 0.45
    axis_y_ratio: float = 0.35
    proximity_px: float = 0.0


def touches_center_ellipse(
    rect: Rect, width: int, height: int, ellipse: EllipseParams
) -> bool:
    """Port of Deidentifier._touches_center_ellipse (deidentifier_component.py)."""
    if width <= 0 or height <= 0:
        return False
    cx, cy = width / 2.0, height / 2.0
    axis_x = max(1.0, width * ellipse.axis_x_ratio + ellipse.proximity_px)
    axis_y = max(1.0, height * ellipse.axis_y_ratio + ellipse.proximity_px)
    x1, y1, x2, y2 = rect
    nearest_x = min(max(cx, x1), x2)
    nearest_y = min(max(cy, y1), y2)
    return ((nearest_x - cx) / axis_x) ** 2 + ((nearest_y - cy) / axis_y) ** 2 <= 1.0


def iou(a: Rect, b: Rect) -> float:
    inter = intersection_area(a, b)
    union = rect_area(a) + rect_area(b) - inter
    return inter / union if union > 0 else 0.0


def greedy_match(
    gt_boxes: List[Rect], pred_boxes: List[Rect], iou_threshold: float
) -> List[Tuple[int, int, float]]:
    """One-to-one matching, highest-IoU-first (the COCO/ICDAR convention).

    Hungarian is unnecessary here: per-image GT and prediction counts are single
    to low-double digits, where greedy-by-score differs from optimal only under
    pathological overlap.
    """
    scored = sorted(
        (
            (iou(gt, pred), gt_index, pred_index)
            for gt_index, gt in enumerate(gt_boxes)
            for pred_index, pred in enumerate(pred_boxes)
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    matches: List[Tuple[int, int, float]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    for score, gt_index, pred_index in scored:
        if score < iou_threshold or score <= 0.0:
            break
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append((gt_index, pred_index, score))
    return matches


def union_area(rects: Sequence[Rect], clip: Rect | None = None) -> float:
    """Exact area of the union of axis-aligned rectangles (sweep line).

    Redaction boxes overlap after merging and padding, so summing their areas
    would over-count the over-redaction cost.
    """
    boxes: List[Rect] = []
    for x1, y1, x2, y2 in rects:
        if clip is not None:
            cx1, cy1, cx2, cy2 = clip
            x1, y1 = max(x1, cx1), max(y1, cy1)
            x2, y2 = min(x2, cx2), min(y2, cy2)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    if not boxes:
        return 0.0

    xs = sorted({box[0] for box in boxes} | {box[2] for box in boxes})
    total = 0.0
    for left, right in zip(xs, xs[1:]):
        strip_width = right - left
        if strip_width <= 0:
            continue
        spans = sorted(
            (box[1], box[3]) for box in boxes if box[0] <= left and box[2] >= right
        )
        covered = 0.0
        span_start = span_end = None
        for start, end in spans:
            if span_end is None or start > span_end:
                if span_end is not None:
                    covered += span_end - span_start
                span_start, span_end = start, end
            elif end > span_end:
                span_end = end
        if span_end is not None:
            covered += span_end - span_start
        total += strip_width * covered
    return total


def read_image_size(path: Path) -> Tuple[int, int] | None:
    """(width, height) from the PNG/JPEG header, without decoding pixels.

    The dataset contains 6343x2713 panoramics; fully decoding every image once
    per benchmark variant just to learn its size is pure waste.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(32)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                width, height = struct.unpack(">II", head[16:24])
                return int(width), int(height)
            if head[:2] == b"\xff\xd8":
                handle.seek(2)
                data = handle.read()
                index = 0
                while index < len(data) - 9:
                    if data[index] != 0xFF:
                        index += 1
                        continue
                    marker = data[index + 1]
                    if marker in _JPEG_SOF_MARKERS:
                        height, width = struct.unpack(">HH", data[index + 5 : index + 9])
                        return int(width), int(height)
                    if marker == 0xD8 or marker == 0xD9 or 0xD0 <= marker <= 0xD7:
                        index += 2
                        continue
                    index += 2 + struct.unpack(">H", data[index + 2 : index + 4])[0]
    except (OSError, struct.error):
        pass
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    return int(image.shape[1]), int(image.shape[0])


def build_image_size_index(image_dir: Path | None) -> Dict[str, Tuple[int, int]]:
    """stem -> (width, height).

    Also defines the *image universe* for the clean-image false-positive rate, so
    images present on disk but absent from the ground truth still count as clean
    rather than silently vanishing from the denominator.
    """
    if image_dir is None:
        return {}
    sizes: Dict[str, Tuple[int, int]] = {}
    for stem, path in build_image_index(image_dir).items():
        size = read_image_size(path)
        if size is not None:
            sizes[stem] = size
    return sizes


def filter_blank_gt(
    ground_truth: Dict[str, List[Rect]],
    image_index: Dict[str, Path],
    min_std: float = 0.0,
) -> Tuple[Dict[str, List[Rect]], List[Tuple[str, Rect]]]:
    """Drop GT rectangles whose pixels are a single uniform value.

    Some regions were annotated as text and then blanked before the images were
    handed over: 8 of the 81 panoramic boxes are solid 255 or solid 0, five of
    them in one image. No OCR engine can find text in a constant-valued region,
    so scoring against them measures nothing and silently caps recall at 0.901.

    This is not circular reasoning of the "the model missed it, so ignore it"
    kind: the test is a property of the pixels alone and never looks at any
    prediction. The PHI in those regions is already gone, which is precisely
    why they are blank.

    The ground-truth file is never modified; dropped shapes are returned so the
    caller can report them.
    """
    kept: Dict[str, List[Rect]] = {}
    dropped: List[Tuple[str, Rect]] = []
    for image, rects in ground_truth.items():
        path = image_index.get(image)
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path is not None else None
        keep: List[Rect] = []
        for rect in rects:
            if gray is None:
                keep.append(rect)
                continue
            height, width = gray.shape[:2]
            x1 = max(0, int(rect[0]))
            y1 = max(0, int(rect[1]))
            x2 = min(width, int(rect[2]) + 1)
            y2 = min(height, int(rect[3]) + 1)
            crop = gray[y1:y2, x1:x2]
            if crop.size and float(crop.std()) <= min_std:
                dropped.append((image, rect))
            else:
                keep.append(rect)
        kept[image] = keep
    return kept, dropped


def filter_degenerate_gt(
    ground_truth: Dict[str, List[Rect]], min_side_px: float = 1.0
) -> Tuple[Dict[str, List[Rect]], List[Tuple[str, Rect]]]:
    """Drop ground-truth rectangles thinner than min_side_px on either axis.

    These are annotation slips (a click-drag that never moved on one axis), not
    text: no OCR engine can detect a 0.03px-tall box, and its coverage ratio is
    decided by float rounding on a sub-pixel intersection. The ground-truth file
    itself is never modified; dropped shapes are returned so the caller can log
    them.
    """
    kept: Dict[str, List[Rect]] = {}
    dropped: List[Tuple[str, Rect]] = []
    for image, rects in ground_truth.items():
        keep: List[Rect] = []
        for rect in rects:
            if (rect[2] - rect[0]) < min_side_px or (rect[3] - rect[1]) < min_side_px:
                dropped.append((image, rect))
            else:
                keep.append(rect)
        kept[image] = keep
    return kept, dropped


def evaluate(
    predictions: Dict[str, List[Rect]],
    ground_truth: Dict[str, List[Rect]],
    margin_px: float,
    *,
    iou_threshold: float = 0.5,
    coverage_threshold: float = 0.5,
    image_sizes: Dict[str, Tuple[int, int]] | None = None,
    ellipse: EllipseParams | None = None,
) -> Dict:
    """Score predicted redaction boxes against the text ground truth.

    The original coverage keys are preserved verbatim; everything else is
    additive. Two families of metric are reported side by side because they
    answer different questions:

    * coverage / full-coverage — did the redaction rectangle actually cover the
      PHI pixels? This is what matters for anonymization.
    * coverage recall — a GT box counts as detected when some prediction covers
      at least coverage_threshold of it. This is the right recall notion for
      redaction, where covering *more* than the text is the goal.
    * IoU recall / precision / F1 — standard detection metrics, reported for
      comparability with published text-detection work. Expect them to look bad
      here by construction: redaction boxes are intentionally larger than the
      text they hide (merge + padding), and IoU penalises precisely that. A GT
      box covered 100% by a box twice its size scores IoU ~0.3 and is counted as
      a miss at the usual 0.5 threshold.

    Passing image_sizes additionally enables the geometry-dependent metrics
    (central-ellipse false positives, redacted pixel area) and widens the image
    universe to every image on disk, so text-free images still count toward the
    clean-image false-positive rate.
    """
    image_sizes = image_sizes or {}
    all_images = sorted(
        set(ground_truth.keys()) | set(predictions.keys()) | set(image_sizes.keys())
    )

    total_gt_boxes = 0
    fully_covered_gt_boxes = 0
    uncovered_gt_boxes = 0
    gt_coverage_sum = 0.0
    gt_union_coverage_sum = 0.0
    fully_covered_union_boxes = 0
    total_pred_boxes = 0
    outside_gt_fp_boxes = 0
    per_image: List[Dict] = []
    per_region_coverage: List[float] = []

    true_positives = 0
    detected_gt_boxes = 0
    images_with_gt = 0
    images_fully_safe = 0
    leak_images: List[str] = []
    matched_iou_sum = 0.0

    clean_images = 0
    clean_images_with_fp = 0
    clean_fp_boxes = 0

    central_fp = 0
    central_over_redaction = 0

    redacted_area_sum = 0.0
    image_area_sum = 0.0
    redacted_fraction_values: List[float] = []
    clean_redacted_area_sum = 0.0
    clean_image_area_sum = 0.0
    images_without_size = 0

    for image in all_images:
        preds = predictions.get(image, [])
        gts = ground_truth.get(image, [])

        image_coverage_sum = 0.0
        image_union_coverage_sum = 0.0
        image_fully_covered = 0
        image_fully_covered_union = 0
        image_detected = 0
        image_uncovered = 0
        for gt_box in gts:
            coverage = gt_best_coverage_ratio(gt_box, preds, margin_px)
            union_coverage = gt_union_coverage_ratio(gt_box, preds, margin_px)
            per_region_coverage.append(coverage)
            image_coverage_sum += coverage
            image_union_coverage_sum += union_coverage
            if coverage >= 0.999999:
                image_fully_covered += 1
            # The safety metrics use union coverage: what matters is whether
            # every PHI pixel ended up under some redaction box, not whether a
            # single box did all the work.
            if union_coverage >= 0.999999:
                image_fully_covered_union += 1
            if union_coverage >= coverage_threshold:
                image_detected += 1
            if union_coverage <= 0.0:
                image_uncovered += 1

        image_outside_fp = sum(
            1
            for pred_box in preds
            if pred_is_outside_ground_truth(pred_box, gts, margin_px)
        )

        # Standard one-to-one detection metrics at the IoU threshold.
        matches = greedy_match(gts, preds, iou_threshold)
        image_tp = len(matches)
        matched_iou_sum += sum(match[2] for match in matches)

        # Case-level safety: one uncovered box leaks that patient's identity,
        # so an image only counts as safe when *every* GT box is fully covered.
        if gts:
            images_with_gt += 1
            if image_fully_covered_union == len(gts):
                images_fully_safe += 1
            else:
                leak_images.append(image)
        else:
            clean_images += 1
            if preds:
                clean_images_with_fp += 1
                clean_fp_boxes += len(preds)

        size = image_sizes.get(image)
        image_central_fp = 0
        image_central_over_redaction = 0
        image_redacted_area = 0.0
        image_area = 0.0
        if size is None:
            images_without_size += 1
        else:
            width, height = size
            image_area = float(width * height)
            image_redacted_area = union_area(preds, clip=(0.0, 0.0, float(width), float(height)))
            redacted_area_sum += image_redacted_area
            image_area_sum += image_area
            if image_area > 0:
                redacted_fraction_values.append(image_redacted_area / image_area)
            if not gts:
                clean_redacted_area_sum += image_redacted_area
                clean_image_area_sum += image_area
            if ellipse is not None:
                for pred_box in preds:
                    if touches_center_ellipse(pred_box, width, height, ellipse):
                        image_central_fp += 1
                        # A GT box may legitimately sit inside the ellipse, so
                        # only a central prediction that matches no GT counts as
                        # anatomy being over-redacted.
                        if pred_is_outside_ground_truth(pred_box, gts, margin_px):
                            image_central_over_redaction += 1
        central_fp += image_central_fp
        central_over_redaction += image_central_over_redaction

        per_image.append(
            {
                "image": image,
                "gt_count": len(gts),
                "pred_count": len(preds),
                "mean_gt_coverage": safe_div(image_coverage_sum, len(gts)),
                "mean_gt_union_coverage": safe_div(image_union_coverage_sum, len(gts)),
                "fully_covered": image_fully_covered,
                "fully_covered_union": image_fully_covered_union,
                "detected": image_detected,
                "uncovered": image_uncovered,
                "outside_gt_fp": image_outside_fp,
                "tp": image_tp,
                "fp": len(preds) - image_tp,
                "fn": len(gts) - image_tp,
                "fully_safe": bool(gts) and image_fully_covered_union == len(gts),
                "central_fp": image_central_fp,
                "central_over_redaction": image_central_over_redaction,
                "redacted_area": image_redacted_area,
                "image_area": image_area,
                "redacted_area_fraction": safe_div(image_redacted_area, image_area),
            }
        )

        total_gt_boxes += len(gts)
        fully_covered_gt_boxes += image_fully_covered
        fully_covered_union_boxes += image_fully_covered_union
        gt_union_coverage_sum += image_union_coverage_sum
        detected_gt_boxes += image_detected
        uncovered_gt_boxes += image_uncovered
        gt_coverage_sum += image_coverage_sum
        total_pred_boxes += len(preds)
        outside_gt_fp_boxes += image_outside_fp
        true_positives += image_tp

    false_positives = total_pred_boxes - true_positives
    false_negatives = total_gt_boxes - true_positives
    recall_iou = safe_div(true_positives, total_gt_boxes)
    precision_iou = safe_div(true_positives, total_pred_boxes)

    return {
        # --- original keys, unchanged semantics ---
        "margin_px": margin_px,
        "total_gt": total_gt_boxes,
        "gt_coverage_score": safe_div(gt_coverage_sum, total_gt_boxes),
        "fully_covered": fully_covered_gt_boxes,
        "gt_full_coverage_rate": safe_div(fully_covered_gt_boxes, total_gt_boxes),
        "uncovered": uncovered_gt_boxes,
        "total_pred": total_pred_boxes,
        "outside_gt_fp": outside_gt_fp_boxes,
        "outside_gt_fp_rate": safe_div(outside_gt_fp_boxes, total_pred_boxes),
        "per_image": per_image,
        # --- coverage recall: the right recall notion for redaction ---
        # Union-based: a GT box counts as covered when the predictions TOGETHER
        # cover it, which is what actually determines whether PHI is hidden.
        "gt_union_coverage_score": safe_div(gt_union_coverage_sum, total_gt_boxes),
        "fully_covered_union": fully_covered_union_boxes,
        "gt_full_coverage_rate_union": safe_div(
            fully_covered_union_boxes, total_gt_boxes
        ),
        "coverage_threshold": coverage_threshold,
        "detected_gt": detected_gt_boxes,
        "recall_covered": safe_div(detected_gt_boxes, total_gt_boxes),
        # --- standard detection metrics (literature-comparable, but see the
        #     docstring: they penalise the intentional over-redaction) ---
        "iou_threshold": iou_threshold,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "recall_iou": recall_iou,
        "precision_iou": precision_iou,
        "f1_iou": safe_div(2.0 * recall_iou * precision_iou, recall_iou + precision_iou),
        "mean_matched_iou": safe_div(matched_iou_sum, true_positives),
        # --- case-level safety: the PHI-leak number ---
        "images_with_gt": images_with_gt,
        "images_fully_safe": images_fully_safe,
        "image_level_recall": safe_div(images_fully_safe, images_with_gt),
        "leak_images": sorted(leak_images),
        # --- over-redaction on text-free images ---
        "clean_images": clean_images,
        "clean_images_with_fp": clean_images_with_fp,
        "clean_fp_boxes": clean_fp_boxes,
        "clean_image_fp_rate": safe_div(clean_images_with_fp, clean_images),
        "clean_fp_boxes_per_image": safe_div(clean_fp_boxes, clean_images),
        # --- over-redaction on anatomy ---
        "central_fp": central_fp,
        "central_over_redaction": central_over_redaction,
        # --- over-redaction cost in pixels ---
        "redacted_area_fraction": safe_div(
            sum(redacted_fraction_values), len(redacted_fraction_values)
        ),
        "redacted_area_fraction_global": safe_div(redacted_area_sum, image_area_sum),
        "redacted_area_fraction_clean": safe_div(
            clean_redacted_area_sum, clean_image_area_sum
        ),
        "images_without_size": images_without_size,
        # --- raw per-GT-box coverage, for the distribution figure ---
        "per_region_coverage": per_region_coverage,
    }


def main() -> None:
    args = parse_args()
    raw_predictions = load_json(args.predictions)
    raw_ground_truth = load_json(args.ground_truth)
    if not isinstance(raw_predictions, (dict, list)):
        raise ValueError("Predictions JSON must be either a dict or a list.")
    if not isinstance(raw_ground_truth, dict):
        raise ValueError("Ground-truth JSON must be a dict keyed by image name.")

    predictions = parse_predictions(raw_predictions, min_confidence=args.min_confidence)
    ground_truth, dropped_gt = filter_degenerate_gt(
        parse_ground_truth(raw_ground_truth), args.min_gt_side_px
    )
    for image, rect in dropped_gt:
        print(
            f"WARNING: dropped degenerate GT box in {image}: "
            f"{rect[2] - rect[0]:.3f}x{rect[3] - rect[1]:.3f} px "
            f"(< {args.min_gt_side_px} px on one side; annotation artefact, not text). "
            "The ground-truth file was NOT modified."
        )

    images_dir = args.images_dir or args.cropped_image_dir
    image_sizes = build_image_size_index(images_dir)

    blank_gt: List[Tuple[str, Rect]] = []
    if images_dir is not None and not args.keep_blank_gt:
        ground_truth, blank_gt = filter_blank_gt(
            ground_truth, build_image_index(images_dir)
        )
        for image, rect in blank_gt:
            print(
                f"WARNING: dropped blank GT box in {image}: "
                f"{rect[2] - rect[0]:.0f}x{rect[3] - rect[1]:.0f} px of uniform "
                "pixels — the region was annotated as text and then blanked, so "
                "no OCR can find it. Pass --keep-blank-gt to score against it "
                "anyway. The ground-truth file was NOT modified."
            )

    images_with_not_fully_covered: List[str] = []
    not_fully_covered_count_by_image: Dict[str, int] = {}

    image_index: Dict[str, Path] = {}
    overlays_saved = 0
    overlays_missing_source = 0
    overlays_write_fail = 0
    if args.save_overlay_dir is not None:
        args.save_overlay_dir.mkdir(parents=True, exist_ok=True)
        # Honour the documented fallback instead of crashing in build_image_index.
        overlay_root = images_dir or args.ground_truth.parent
        image_index = build_image_index(overlay_root)

    print("=== Dataset structure ===")
    if isinstance(raw_predictions, dict):
        print(
            f"predictions: dict[{len(raw_predictions)}] -> "
            "list[{bbox|boxes: [x,y,w,h] or points, text: str, confidence: float}]"
        )
    else:
        print(
            f"predictions: list[{len(raw_predictions)}] -> "
            "{name: str, boxes: list[[x, y, w, h]]}"
        )
    print(
        f"ground-truth: dict[{len(raw_ground_truth)}] -> "
        "{shapes: list[{label: str, points: [[x1,y1],[x2,y2]], shape_type: str}]}"
    )
    print()

    metrics = evaluate(
        predictions,
        ground_truth,
        args.margin_px,
        iou_threshold=args.iou_threshold,
        coverage_threshold=args.coverage_threshold,
        image_sizes=image_sizes,
        ellipse=EllipseParams(
            axis_x_ratio=args.ellipse_axis_x,
            axis_y_ratio=args.ellipse_axis_y,
            proximity_px=args.ellipse_proximity_px,
        ),
    )
    metrics["degenerate_gt_dropped"] = len(dropped_gt)
    metrics["blank_gt_dropped"] = len(blank_gt)

    for image_stats in metrics["per_image"]:
        image = image_stats["image"]

        if args.per_image:
            print(
                f"{image}: gt={image_stats['gt_count']} pred={image_stats['pred_count']} "
                f"mean_gt_coverage={image_stats['mean_gt_coverage']:.4f} "
                f"fully_covered_gt={image_stats['fully_covered']}/{image_stats['gt_count']} "
                f"outside_gt_fp={image_stats['outside_gt_fp']}"
            )

        image_not_fully_covered = (
            image_stats["gt_count"] - image_stats["fully_covered"]
        )
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
                preds = predictions.get(image, [])
                gts = ground_truth.get(image, [])
                if save_overlay_image(image_path, out_path, preds, gts):
                    print(f"Saved overlay for {image} to {out_path}")
                    overlays_saved += 1
                else:
                    overlays_write_fail += 1

    print("=== Ground-truth coverage metrics ===")
    print(f"Margin tolerance: {args.margin_px:.1f}px")
    print(f"Total GT boxes: {metrics['total_gt']}")
    print(f"GT coverage score (mean covered area): {metrics['gt_coverage_score']:.4f}")
    print(
        f"Fully covered GT boxes: {metrics['fully_covered']}/{metrics['total_gt']} "
        f"({metrics['gt_full_coverage_rate']:.4f})"
    )
    print(f"Uncovered GT boxes: {metrics['uncovered']}")
    print()
    print("=== False positives outside GT ===")
    print(
        f"Outside-GT predictions: {metrics['outside_gt_fp']}/{metrics['total_pred']} "
        f"({metrics['outside_gt_fp_rate']:.4f})"
    )
    print()
    print(
        f"=== Coverage recall (GT covered >= {metrics['coverage_threshold']:.2f}) ==="
    )
    print(
        f"Recall (covered): {metrics['recall_covered']:.4f}  "
        f"({metrics['detected_gt']}/{metrics['total_gt']})"
    )
    print(
        f"Fully covered   : {metrics['gt_full_coverage_rate_union']:.4f}  "
        f"({metrics['fully_covered_union']}/{metrics['total_gt']})"
    )
    print(f"Mean coverage   : {metrics['gt_union_coverage_score']:.4f}")
    print("(union of all predictions; the single-box variant is reported above)")
    print()
    print(
        f"=== Detection metrics (one-to-one, IoU >= {metrics['iou_threshold']:.2f}) ==="
    )
    print(
        "NOTE: IoU penalises intentional over-redaction. A GT box covered 100% by"
    )
    print(
        "      a redaction box twice its size scores ~0.3 and counts as a miss."
    )
    print(
        f"Recall    : {metrics['recall_iou']:.4f}  "
        f"({metrics['tp']}/{metrics['tp'] + metrics['fn']})"
    )
    print(
        f"Precision : {metrics['precision_iou']:.4f}  "
        f"({metrics['tp']}/{metrics['tp'] + metrics['fp']})"
    )
    print(f"F1        : {metrics['f1_iou']:.4f}")
    print(f"Mean matched IoU: {metrics['mean_matched_iou']:.4f}")
    print()
    print("=== Case-level safety (PHI leakage) ===")
    print(f"Images with GT text    : {metrics['images_with_gt']}")
    print(
        f"Images fully protected : {metrics['images_fully_safe']} "
        f"(image_level_recall = {metrics['image_level_recall']:.4f})"
    )
    leaks = metrics["leak_images"]
    shown = ", ".join(leaks[:8]) + (" ..." if len(leaks) > 8 else "")
    print(f"Leaking images         : {shown if leaks else 'none'}")
    print()
    print("=== Over-redaction cost ===")
    print(f"Clean images (no GT text): {metrics['clean_images']}")
    print(
        f"  with >= 1 prediction   : {metrics['clean_images_with_fp']} "
        f"(clean_image_fp_rate = {metrics['clean_image_fp_rate']:.4f})"
    )
    print(f"  predictions per clean  : {metrics['clean_fp_boxes_per_image']:.2f}")
    if images_dir is None:
        print(
            "  (pass --images-dir to enable the central-ellipse and "
            "redacted-area metrics)"
        )
    else:
        print(
            f"Predictions on the central ellipse: {metrics['central_fp']} "
            f"(of which outside GT: {metrics['central_over_redaction']})"
        )
        print(
            f"Redacted pixel area: {metrics['redacted_area_fraction_global'] * 100:.2f}% "
            f"of image area (global), "
            f"{metrics['redacted_area_fraction_clean'] * 100:.2f}% on clean images"
        )
        if metrics["images_without_size"]:
            print(f"Images with unknown size: {metrics['images_without_size']}")
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

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\nFull metrics written to {args.json_out}")


if __name__ == "__main__":
    main()
