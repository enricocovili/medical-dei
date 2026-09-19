from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

try:
    from .ocr_engines import OcrDetection, OcrEngine
except ImportError:
    from ocr_engines import OcrDetection, OcrEngine

RESOLUTION_STRATEGIES = {"full", "tiled"}

type _Rect = tuple[float, float, float, float]


class ResolutionStrategy(Protocol):
    def detect(self, engine: OcrEngine, image: np.ndarray) -> list[OcrDetection]: ...


class FullImageStrategy:
    def detect(self, engine: OcrEngine, image: np.ndarray) -> list[OcrDetection]:
        return engine.detect(image)


def _quad_to_rect(quad: list[list[int]]) -> _Rect:
    xs = [float(point[0]) for point in quad]
    ys = [float(point[1]) for point in quad]
    return min(xs), min(ys), max(xs), max(ys)


def _rect_to_quad(rect: _Rect) -> list[list[int]]:
    x1, y1, x2, y2 = rect
    return [
        [int(round(x1)), int(round(y1))],
        [int(round(x2)), int(round(y1))],
        [int(round(x2)), int(round(y2))],
        [int(round(x1)), int(round(y2))],
    ]


def _containment_ratio(rect_a: _Rect, rect_b: _Rect) -> float:
    ax1, ay1, ax2, ay2 = rect_a
    bx1, by1, bx2, by2 = rect_b
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    smaller = min(area_a, area_b)
    if smaller <= 0:
        return 0.0
    return inter / smaller


def _tile_starts(image_extent: int, tile_size: int, stride: int) -> list[int]:
    if image_extent <= tile_size:
        return [0]
    starts = list(range(0, image_extent - tile_size, stride))
    starts.append(image_extent - tile_size)
    return starts


class TiledStrategy:
    """Runs OCR on overlapping tiles so large images are never downscaled
    by the engine, then merges duplicate detections across tile seams.

    Seam duplicates are usually one full box plus a truncated partial box,
    so dedupe uses containment ratio (intersection / smaller area) rather
    than plain IoU."""

    def __init__(
        self,
        tile_size_px: int = 1600,
        tile_overlap_px: int = 200,
        dedupe_containment: float = 0.5,
    ) -> None:
        if tile_size_px <= 0:
            raise ValueError("tile_size_px must be > 0")
        if tile_overlap_px < 0 or tile_overlap_px >= tile_size_px:
            raise ValueError("tile_overlap_px must be in [0, tile_size_px)")
        if not 0.0 < dedupe_containment <= 1.0:
            raise ValueError("dedupe_containment must be in (0, 1]")
        self._tile_size = tile_size_px
        self._overlap = tile_overlap_px
        self._dedupe_containment = dedupe_containment

    def detect(self, engine: OcrEngine, image: np.ndarray) -> list[OcrDetection]:
        height, width = image.shape[:2]
        stride = self._tile_size - self._overlap
        detections: list[OcrDetection] = []
        for y0 in _tile_starts(height, self._tile_size, stride):
            for x0 in _tile_starts(width, self._tile_size, stride):
                tile = image[y0 : y0 + self._tile_size, x0 : x0 + self._tile_size]
                for det in engine.detect(tile):
                    shifted = [[point[0] + x0, point[1] + y0] for point in det.quad]
                    detections.append(
                        OcrDetection(
                            quad=shifted, text=det.text, confidence=det.confidence
                        )
                    )
        return self._dedupe(detections)

    def _dedupe(self, detections: list[OcrDetection]) -> list[OcrDetection]:
        items: list[tuple[_Rect, str, float]] = [
            (_quad_to_rect(det.quad), det.text, det.confidence) for det in detections
        ]
        changed = True
        while changed:
            changed = False
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    rect_i, text_i, conf_i = items[i]
                    rect_j, text_j, conf_j = items[j]
                    if (
                        _containment_ratio(rect_i, rect_j)
                        < self._dedupe_containment
                    ):
                        continue
                    union_rect = (
                        min(rect_i[0], rect_j[0]),
                        min(rect_i[1], rect_j[1]),
                        max(rect_i[2], rect_j[2]),
                        max(rect_i[3], rect_j[3]),
                    )
                    longer_text = text_i if len(text_i) >= len(text_j) else text_j
                    items[i] = (union_rect, longer_text, max(conf_i, conf_j))
                    items.pop(j)
                    changed = True
                    break
                if changed:
                    break
        return [
            OcrDetection(quad=_rect_to_quad(rect), text=text, confidence=conf)
            for rect, text, conf in items
        ]


UPSCALE_INTERPOLATIONS = {
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
    "linear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
}


class UpscaleStrategy:
    """Wraps another strategy: enlarges the OCR input, then maps every detection
    coordinate back into the original array's space.

    Upscaling belongs here rather than in ocr_preprocess.apply_chain because the
    strategy layer already owns coordinate translation (see TiledStrategy). The
    Deidentifier therefore never sees the enlarged array, so the centre-ellipse
    test, max_box_area_px and the drawing clamp all keep working in original
    pixels. As a preprocess step it would instead make the area cap 4x too
    permissive at 2x and clamp enlarged coordinates into the original frame,
    collapsing every box into the top-left quadrant.

    Composes with tiling: UpscaleStrategy(TiledStrategy(1600, 200), 2.0) gives
    1600px tiles over a 2x image, i.e. 800 native pixels at double detail.
    """

    def __init__(
        self,
        inner: ResolutionStrategy,
        factor: float,
        interpolation: str = "cubic",
    ) -> None:
        if factor < 1.0:
            raise ValueError("upscale factor must be >= 1.0")
        if interpolation not in UPSCALE_INTERPOLATIONS:
            raise ValueError(
                f"Invalid upscale interpolation '{interpolation}'. "
                f"Valid values: {sorted(UPSCALE_INTERPOLATIONS)}"
            )
        self._inner = inner
        self._factor = float(factor)
        self._interpolation = UPSCALE_INTERPOLATIONS[interpolation]

    def detect(self, engine: OcrEngine, image: np.ndarray) -> list[OcrDetection]:
        if self._factor == 1.0:
            return self._inner.detect(engine, image)
        enlarged = cv2.resize(
            image,
            None,
            fx=self._factor,
            fy=self._factor,
            interpolation=self._interpolation,
        )
        inverse = 1.0 / self._factor
        return [
            OcrDetection(
                quad=[
                    [int(round(point[0] * inverse)), int(round(point[1] * inverse))]
                    for point in det.quad
                ],
                text=det.text,
                confidence=det.confidence,
            )
            for det in self._inner.detect(engine, enlarged)
        ]
