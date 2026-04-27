from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

type Detection = tuple[list[list[int]], str, float]
type Rect = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class DeidentifierParams:
    merge_distance_px: int = 10
    max_box_area_px: int | None = 120000
    center_ellipse_axes_ratio: tuple[float, float] = (0.35, 0.25)
    ellipse_proximity_px: float = 0.0
    padding_px: int = 3


class EasyOcrDeidentifier:
    def __init__(self, reader: Any, params: DeidentifierParams | None = None) -> None:
        self._reader = reader
        self._params = params or DeidentifierParams()
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

    def deidentify(self, image_rgb: np.ndarray, image_name: str) -> np.ndarray:
        deidentified, _ = self.deidentify_with_boxes(image_rgb, image_name)
        return deidentified

    def deidentify_with_boxes(
        self, image_rgb: np.ndarray, image_name: str
    ) -> tuple[np.ndarray, list[list[int]]]:
        if image_rgb.ndim == 2:
            read_target = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
            out = image_rgb.copy()
            fill_color: tuple[int, int, int] | int = 0
        else:
            read_target = image_rgb
            out = image_rgb.copy()
            fill_color = (0, 0, 255)

        results_normal = self._reader.readtext(
            read_target, 
            text_threshold=0.5,
            canvas_size=4000,
            mag_ratio=1.0,
        )
        # results_shrunk = self._reader.readtext(
        #     read_target, mag_ratio=0.1, text_threshold=0.5
        # )
        results_shrunk = []
        all_results = self._parse_detections(results_normal) + self._parse_detections(
            results_shrunk
        )

        filtered_single = self._filter_detections(all_results, read_target.shape)
        filtered = self._merge_close_detections(filtered_single)

        height, width = out.shape[:2]
        drawn_boxes: list[list[int]] = []
        for bbox, _, _ in filtered:
            xs = [int(point[0]) for point in bbox]
            ys = [int(point[1]) for point in bbox]
            left = max(0, min(width, min(xs) - self._params.padding_px))
            top = max(0, min(height, min(ys) - self._params.padding_px))
            right = max(0, min(width, max(xs) + self._params.padding_px))
            bottom = max(0, min(height, max(ys) + self._params.padding_px))
            if right > left and bottom > top:
                cv2.rectangle(out, (left, top), (right, bottom), fill_color, thickness=3)
                drawn_boxes.append([left, top, right - left, bottom - top])
        return out, drawn_boxes

    def _parse_detections(self, raw_results: Any) -> list[Detection]:
        parsed: list[Detection] = []
        for item in raw_results:
            if not isinstance(item, (tuple, list)) or len(item) < 3:
                continue
            bbox, text, prob = item[0], item[1], item[2]
            if not isinstance(bbox, (tuple, list)) or len(bbox) < 4:
                continue
            bbox_int = [[int(round(point[0])), int(round(point[1]))] for point in bbox]
            parsed.append((bbox_int, str(text), float(prob)))
        return parsed

    @staticmethod
    def _bbox_to_rect(bbox: list[list[int]]) -> Rect:
        xs = [float(point[0]) for point in bbox]
        ys = [float(point[1]) for point in bbox]
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def _rect_to_bbox(rect: Rect) -> list[list[int]]:
        x1, y1, x2, y2 = rect
        return [
            [int(round(x1)), int(round(y1))],
            [int(round(x2)), int(round(y1))],
            [int(round(x2)), int(round(y2))],
            [int(round(x1)), int(round(y2))],
        ]

    @staticmethod
    def _rect_area(rect: Rect) -> float:
        x1, y1, x2, y2 = rect
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _touches_center_ellipse(self, rect: Rect, image_shape: tuple[int, ...]) -> bool:
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
        self, detections: list[Detection], image_shape: tuple[int, ...]
    ) -> list[Detection]:
        filtered: list[Detection] = []
        for bbox, text, prob in detections:
            rect = self._bbox_to_rect(bbox)
            if (
                self._params.max_box_area_px is not None
                and self._rect_area(rect) > self._params.max_box_area_px
            ):
                continue
            if self._touches_center_ellipse(rect, image_shape):
                continue
            filtered.append((bbox, text, prob))
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

    def _merge_close_detections(self, detections: list[Detection]) -> list[Detection]:
        if not detections:
            return []

        rects = [self._bbox_to_rect(bbox) for bbox, _, _ in detections]
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

        merged: list[Detection] = []
        for indices in groups.values():
            x1 = min(rects[i][0] for i in indices)
            y1 = min(rects[i][1] for i in indices)
            x2 = max(rects[i][2] for i in indices)
            y2 = max(rects[i][3] for i in indices)
            merged_text = " | ".join(
                text for i in indices if (text := detections[i][1].strip())
            )
            merged_confidence = max(float(detections[i][2]) for i in indices)
            merged.append(
                (
                    self._rect_to_bbox((x1, y1, x2, y2)),
                    merged_text,
                    merged_confidence,
                )
            )

        merged.sort(key=lambda det: (det[0][0][1], det[0][0][0]))
        return merged
