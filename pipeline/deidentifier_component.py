from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

try:
    from .ocr_engines import OcrDetection, OcrEngine
    from .ocr_strategies import FullImageStrategy, ResolutionStrategy
    from .text_classifiers import RedactAllClassifier, RedactionClassifier
except ImportError:
    from ocr_engines import OcrDetection, OcrEngine
    from ocr_strategies import FullImageStrategy, ResolutionStrategy
    from text_classifiers import RedactAllClassifier, RedactionClassifier

type Rect = tuple[float, float, float, float]

REDACTION_MODES = {"outline", "fill"}


@dataclass(frozen=True, slots=True)
class DeidentifierParams:
    merge_distance_px: int = 10
    max_box_area_px: int | None = 120000
    # Relative twin of max_box_area_px, as a fraction of the image area. The cap
    # exists to reject "the engine returned the whole image", which is inherently
    # relative: an absolute 120000 px2 is generous on a panoramic and smaller
    # than 13 real text regions on a teleradiograph. None disables it.
    max_box_area_ratio: float | None = None
    # False disables the central keep-out entirely. Previously the only way to
    # turn it off was to set the ratios to ~0.001, which still excluded anything
    # overlapping the exact image centre.
    ellipse_enabled: bool = True
    center_ellipse_axes_ratio: tuple[float, float] = (0.35, 0.25)
    ellipse_proximity_px: float = 0.0
    padding_px: int = 3
    min_confidence: float = 0.0
    # fill is the real anonymization; outline only draws debug rectangles and
    # leaves the text readable underneath, so it must never be the default.
    redaction_mode: str = "fill"


@dataclass(frozen=True, slots=True)
class DeidentificationResult:
    image: np.ndarray
    detections: list[dict[str, Any]] = field(default_factory=list)
    skipped_detections: list[dict[str, Any]] = field(default_factory=list)
    raw_detections: list[dict[str, Any]] = field(default_factory=list)

    @property
    def boxes(self) -> list[list[int]]:
        return [list(det["box"]) for det in self.detections]


def _detection_payload(detection: OcrDetection) -> dict[str, Any]:
    x1, y1, x2, y2 = _bbox_to_rect(detection.quad)
    return {
        "box": [int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))],
        "text": detection.text,
        "confidence": detection.confidence,
    }


def _bbox_to_rect(bbox: list[list[int]]) -> Rect:
    xs = [float(point[0]) for point in bbox]
    ys = [float(point[1]) for point in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def _rect_to_bbox(rect: Rect) -> list[list[int]]:
    x1, y1, x2, y2 = rect
    return [
        [int(round(x1)), int(round(y1))],
        [int(round(x2)), int(round(y1))],
        [int(round(x2)), int(round(y2))],
        [int(round(x1)), int(round(y2))],
    ]


def _rect_area(rect: Rect) -> float:
    x1, y1, x2, y2 = rect
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class Deidentifier:
    def __init__(
        self,
        engine: OcrEngine,
        params: DeidentifierParams | None = None,
        *,
        strategy: ResolutionStrategy | None = None,
        classifier: RedactionClassifier | None = None,
        preprocess: Callable[[np.ndarray], np.ndarray] | None = None,
        preprocess_variants: list[Callable[[np.ndarray], np.ndarray] | None] | None = None,
        dual_pass_invert: bool = False,
    ) -> None:
        self._engine = engine
        self._params = params or DeidentifierParams()
        self._strategy = strategy or FullImageStrategy()
        self._classifier = classifier or RedactAllClassifier()
        self._preprocess = preprocess
        # Each variant is an independent preprocessing chain; the engine runs
        # once per variant and the detections are unioned. This is the cheapest
        # recall lever available — one model, N passes — and it generalises the
        # old dual_pass_invert flag, which was a hardcoded union over exactly
        # {identity, invert}. Every chain must be coordinate-preserving, so all
        # the boxes land in the same space (see ocr_preprocess).
        self._preprocess_variants = preprocess_variants
        self._dual_pass_invert = dual_pass_invert
        if self._params.merge_distance_px < 0:
            raise ValueError("merge_distance_px must be >= 0")
        if (
            self._params.max_box_area_px is not None
            and self._params.max_box_area_px <= 0
        ):
            raise ValueError("max_box_area_px must be > 0 when provided")
        axis_x, axis_y = self._params.center_ellipse_axes_ratio
        if axis_x <= 0 or axis_x > 1 or axis_y <= 0 or axis_y > 1:
            raise ValueError("center_ellipse_axes_ratio values must be in (0, 1]")
        if self._params.ellipse_proximity_px < 0:
            raise ValueError("ellipse_proximity_px must be >= 0")
        if self._params.padding_px < 0:
            raise ValueError("padding_px must be >= 0")
        if (
            self._params.max_box_area_ratio is not None
            and not 0.0 < self._params.max_box_area_ratio <= 1.0
        ):
            raise ValueError("max_box_area_ratio must be in (0, 1] or None")
        if not 0.0 <= self._params.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if self._params.redaction_mode not in REDACTION_MODES:
            raise ValueError(
                f"redaction_mode must be one of {sorted(REDACTION_MODES)}"
            )

    def deidentify(self, image_rgb: np.ndarray, image_name: str) -> np.ndarray:
        return self.run(image_rgb, image_name).image

    def deidentify_with_boxes(
        self, image_rgb: np.ndarray, image_name: str
    ) -> tuple[np.ndarray, list[list[int]]]:
        result = self.run(image_rgb, image_name)
        return result.image, result.boxes

    def run(self, image_rgb: np.ndarray, image_name: str) -> DeidentificationResult:
        fill_mode = self._params.redaction_mode == "fill"
        if image_rgb.ndim == 2:
            read_target = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
            out = image_rgb.copy()
            draw_color: tuple[int, int, int] | int = 0
        else:
            read_target = image_rgb
            out = image_rgb.copy()
            draw_color = (0, 0, 0) if fill_mode else (0, 0, 255)

        if self._preprocess is not None:
            read_target = self._preprocess(read_target)

        # read_target fixes the coordinate space for everything downstream
        # (the ellipse test below measures against its shape), so every variant
        # must preserve width and height.
        variants: list[np.ndarray] = []
        if self._preprocess_variants:
            for variant in self._preprocess_variants:
                variants.append(
                    read_target if variant is None else variant(read_target)
                )
        else:
            variants.append(read_target)
        if self._dual_pass_invert:
            variants.extend(cv2.bitwise_not(variant) for variant in list(variants))

        raw: list[OcrDetection] = []
        for variant_image in variants:
            raw.extend(self._strategy.detect(self._engine, variant_image))

        confident = [
            det for det in raw if det.confidence >= self._params.min_confidence
        ]
        kept: list[OcrDetection] = []
        skipped: list[OcrDetection] = []
        for det in confident:
            if self._classifier.should_redact(det):
                kept.append(det)
            else:
                skipped.append(det)

        filtered_single = self._filter_detections(kept, read_target.shape)
        filtered = self._merge_close_detections(filtered_single)

        height, width = out.shape[:2]
        thickness = cv2.FILLED if fill_mode else 3
        detections: list[dict[str, Any]] = []
        for det in filtered:
            xs = [int(point[0]) for point in det.quad]
            ys = [int(point[1]) for point in det.quad]
            left = max(0, min(width, min(xs) - self._params.padding_px))
            top = max(0, min(height, min(ys) - self._params.padding_px))
            right = max(0, min(width, max(xs) + self._params.padding_px))
            bottom = max(0, min(height, max(ys) + self._params.padding_px))
            if right > left and bottom > top:
                cv2.rectangle(
                    out, (left, top), (right, bottom), draw_color, thickness=thickness
                )
                detections.append(
                    {
                        "box": [left, top, right - left, bottom - top],
                        "text": det.text,
                        "confidence": det.confidence,
                    }
                )
        return DeidentificationResult(
            image=out,
            detections=detections,
            skipped_detections=[_detection_payload(det) for det in skipped],
            raw_detections=[_detection_payload(det) for det in raw],
        )

    def _touches_center_ellipse(self, rect: Rect, image_shape: tuple[int, ...]) -> bool:
        if not self._params.ellipse_enabled:
            return False
        image_h, image_w = image_shape[:2]
        cx, cy = image_w / 2.0, image_h / 2.0
        axis_x = max(
            1.0,
            (image_w * self._params.center_ellipse_axes_ratio[0])
            + self._params.ellipse_proximity_px,
        )
        axis_y = max(
            1.0,
            (image_h * self._params.center_ellipse_axes_ratio[1])
            + self._params.ellipse_proximity_px,
        )
        x1, y1, x2, y2 = rect
        nearest_x = min(max(cx, x1), x2)
        nearest_y = min(max(cy, y1), y2)
        ellipse_equation = ((nearest_x - cx) / axis_x) ** 2 + (
            (nearest_y - cy) / axis_y
        ) ** 2
        return ellipse_equation <= 1.0

    def _filter_detections(
        self, detections: list[OcrDetection], image_shape: tuple[int, ...]
    ) -> list[OcrDetection]:
        filtered: list[OcrDetection] = []
        image_h, image_w = image_shape[:2]
        area_limit: float | None = self._params.max_box_area_px
        if self._params.max_box_area_ratio is not None:
            relative_limit = self._params.max_box_area_ratio * image_h * image_w
            area_limit = (
                relative_limit if area_limit is None else min(area_limit, relative_limit)
            )
        for det in detections:
            rect = _bbox_to_rect(det.quad)
            if area_limit is not None and _rect_area(rect) > area_limit:
                continue
            if self._touches_center_ellipse(rect, image_shape):
                continue
            filtered.append(det)
        return filtered

    @staticmethod
    def _rects_are_close(rect_a: Rect, rect_b: Rect, distance_px: float) -> bool:
        ax1, ay1, ax2, ay2 = rect_a
        bx1, by1, bx2, by2 = rect_b
        return not (
            ax2 + distance_px < bx1
            or bx2 + distance_px < ax1
            or ay2 + distance_px < by1
            or by2 + distance_px < ay1
        )

    def _merge_close_detections(
        self, detections: list[OcrDetection]
    ) -> list[OcrDetection]:
        if not detections:
            return []

        rects = [_bbox_to_rect(det.quad) for det in detections]
        parent = list(range(len(rects)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(i: int, j: int) -> None:
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                parent[root_j] = root_i

        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                if self._rects_are_close(rects[i], rects[j], self._params.merge_distance_px):
                    union(i, j)

        groups: dict[int, list[int]] = {}
        for index in range(len(rects)):
            root = find(index)
            if root not in groups:
                groups[root] = []
            groups[root].append(index)

        merged: list[OcrDetection] = []
        for indices in groups.values():
            x1 = min(rects[i][0] for i in indices)
            y1 = min(rects[i][1] for i in indices)
            x2 = max(rects[i][2] for i in indices)
            y2 = max(rects[i][3] for i in indices)
            merged_text = " | ".join(
                text for i in indices if (text := detections[i].text.strip())
            )
            merged_confidence = max(float(detections[i].confidence) for i in indices)
            merged.append(
                OcrDetection(
                    quad=_rect_to_bbox((x1, y1, x2, y2)),
                    text=merged_text,
                    confidence=merged_confidence,
                )
            )

        merged.sort(key=lambda det: (det.quad[0][1], det.quad[0][0]))
        return merged


# Backward-compatible alias (pre-engine-abstraction name).
EasyOcrDeidentifier = Deidentifier
