from __future__ import annotations

import cv2
import numpy as np

try:
    from .models import AlarmInfo, BoundingBox, MaskTransformResult, SegmentationResult
except ImportError:
    from models import AlarmInfo, BoundingBox, MaskTransformResult, SegmentationResult


class MaskPostprocessor:
    def __init__(self, kernel_size: int, iterations: int, large_bb_area_ratio: float):
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if iterations <= 0:
            raise ValueError("iterations must be > 0")
        if large_bb_area_ratio <= 0.0 or large_bb_area_ratio > 1.0:
            raise ValueError("large_bb_area_ratio must be in (0, 1]")

        self._kernel_size = kernel_size
        self._iterations = iterations
        self._large_bb_area_ratio = large_bb_area_ratio

    def transform(
        self, image_rgb: np.ndarray, segmentation: SegmentationResult
    ) -> MaskTransformResult:
        combined_mask = self._combine_masks(image_rgb.shape[:2], segmentation)
        rotated_image, rotated_mask, rotation_angle = self._rotate_using_mask(
            image_rgb, combined_mask
        )
        cleaned_mask = self._clean_mask(rotated_mask)
        extracted_boxes = self._extract_boxes(cleaned_mask)
        boxes = extracted_boxes if extracted_boxes else list(segmentation.sam_boxes)

        image_area = int(image_rgb.shape[0] * image_rgb.shape[1])
        large_area_threshold = image_area * self._large_bb_area_ratio
        large_boxes = [box for box in boxes if box.area >= large_area_threshold]
        alarm_triggered = len(large_boxes) > 1
        alarm = AlarmInfo(
            triggered=alarm_triggered,
            motivation="more than 1 big BB found" if alarm_triggered else "",
        )

        if alarm.triggered:
            pass

        cut_image = self._crop_to_mask(rotated_image, cleaned_mask)
        cut_size = (int(cut_image.shape[1]), int(cut_image.shape[0]))

        return MaskTransformResult(
            bounding_boxes=boxes,
            alarm=alarm,
            cut_image=cut_image,
            cut_size=cut_size,
            rotation_angle=rotation_angle,
        )

    def _combine_masks(
        self, image_shape: tuple[int, int], segmentation: SegmentationResult
    ) -> np.ndarray:
        height, width = image_shape
        if segmentation.masks.size == 0:
            fallback = np.zeros((height, width), dtype=np.uint8)
            for box in segmentation.sam_boxes:
                x1 = int(max(0, min(width, box.x)))
                y1 = int(max(0, min(height, box.y)))
                x2 = int(max(0, min(width, box.x + box.w)))
                y2 = int(max(0, min(height, box.y + box.h)))
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(fallback, (x1, y1), (x2, y2), 255, thickness=-1)
            return fallback

        combined = np.max(segmentation.masks, axis=0)
        return np.where(combined > 0, 255, 0).astype(np.uint8)

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((self._kernel_size, self._kernel_size), dtype=np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=self._iterations)
        return cv2.dilate(eroded, kernel, iterations=self._iterations)

    def _extract_boxes(self, mask: np.ndarray) -> list[BoundingBox]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[BoundingBox] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 0 and h > 0:
                boxes.append(BoundingBox(x=int(x), y=int(y), w=int(w), h=int(h)))
        boxes.sort(key=lambda box: box.area, reverse=True)
        return boxes

    def _rotate_using_mask(
        self, image_rgb: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        rect = self._find_largest_fitting_rectangle(mask)
        if rect is None:
            return image_rgb.copy(), mask.copy(), 0.0

        center = rect[0]
        angle = self._rotation_to_zero(rect)

        rows, cols = image_rgb.shape[:2]
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(
            image_rgb,
            matrix,
            (cols, rows),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        rotated_mask = cv2.warpAffine(
            mask,
            matrix,
            (cols, rows),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        rotated_mask = np.where(rotated_mask > 0, 255, 0).astype(np.uint8)
        return rotated_image, rotated_mask, float(angle)

    def _find_largest_fitting_rectangle(
        self, mask: np.ndarray
    ) -> tuple[tuple[float, float], tuple[float, float], float] | None:
        mask_binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        best_rect: tuple[tuple[float, float], tuple[float, float], float] | None = None
        best_area = 0.0
        for contour in contours:
            if cv2.contourArea(contour) <= 0:
                continue
            candidate = cv2.minAreaRect(contour)
            fitted = self._fit_rectangle_inside_mask(mask_binary, candidate)
            if fitted is None:
                continue
            _, (width, height), _ = fitted
            area = float(width * height)
            if area > best_area:
                best_area = area
                best_rect = fitted
        return best_rect

    def _fit_rectangle_inside_mask(
        self,
        mask_binary: np.ndarray,
        rect: tuple[tuple[float, float], tuple[float, float], float],
    ) -> tuple[tuple[float, float], tuple[float, float], float] | None:
        center, (width, height), angle = rect
        if width <= 0 or height <= 0:
            return None

        low, high = 0.0, 1.0
        best: tuple[tuple[float, float], tuple[float, float], float] | None = None
        for _ in range(24):
            scale = (low + high) / 2.0
            candidate = (center, (width * scale, height * scale), angle)
            if self._rectangle_inside_mask(mask_binary, candidate):
                best = candidate
                low = scale
            else:
                high = scale
        return best

    def _rectangle_inside_mask(
        self,
        mask_binary: np.ndarray,
        rect: tuple[tuple[float, float], tuple[float, float], float],
    ) -> bool:
        rect_mask = np.zeros(mask_binary.shape, dtype=np.uint8)
        corners = cv2.boxPoints(rect)
        polygon = np.round(corners).astype(np.int32)
        cv2.fillConvexPoly(rect_mask, polygon, 255)
        if cv2.countNonZero(rect_mask) == 0:
            return False
        outside = cv2.bitwise_and(rect_mask, cv2.bitwise_not(mask_binary))
        return cv2.countNonZero(outside) == 0

    def _rotation_to_zero(
        self, rect: tuple[tuple[float, float], tuple[float, float], float]
    ) -> float:
        points = cv2.boxPoints(rect)
        edge_vectors = [points[(index + 1) % 4] - points[index] for index in range(4)]
        longest_edge = max(
            edge_vectors,
            key=lambda edge: float(edge[0] * edge[0] + edge[1] * edge[1]),
        )
        orientation = float(np.degrees(np.arctan2(longest_edge[1], longest_edge[0])))

        while orientation <= -90.0:
            orientation += 180.0
        while orientation > 90.0:
            orientation -= 180.0

        correction = orientation
        if correction > 45.0:
            correction -= 90.0
        elif correction < -45.0:
            correction += 90.0

        return float(correction)

    def _crop_to_mask(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        non_zero = cv2.findNonZero(mask)
        if non_zero is None:
            return image_rgb.copy()
        x, y, w, h = cv2.boundingRect(non_zero)
        if w <= 0 or h <= 0:
            return image_rgb.copy()
        return image_rgb[y : y + h, x : x + w].copy()
