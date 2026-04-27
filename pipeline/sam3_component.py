from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

try:
    from .models import BoundingBox, SegmentationResult
except ImportError:
    from models import BoundingBox, SegmentationResult


class Sam3ImageSegmenter:
    def __init__(self, model: Any) -> None:
        from sam3.model.sam3_image_processor import Sam3Processor

        self._processor = Sam3Processor(model)

    def infer(self, image: Image.Image, prompt: str) -> SegmentationResult:
        state = self._processor.set_image(image)
        output = self._processor.set_text_prompt(state=state, prompt=prompt)
        masks = self._to_mask_stack(output.get("masks"), image.size)
        sam_boxes = self._extract_sam_boxes(output.get("boxes"), image.size)
        if not sam_boxes:
            sam_boxes = self._extract_boxes_from_masks(masks)
        return SegmentationResult(masks=masks, sam_boxes=sam_boxes)

    def _to_mask_stack(
        self, raw_masks: Any, image_size: tuple[int, int]
    ) -> np.ndarray:
        width, height = image_size
        if raw_masks is None:
            return np.zeros((0, height, width), dtype=np.uint8)

        masks = self._to_numpy(raw_masks)
        if masks is None:
            return np.zeros((0, height, width), dtype=np.uint8)

        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0, :, :]
        elif masks.ndim == 2:
            masks = masks[np.newaxis, :, :]
        elif masks.ndim == 3 and masks.shape[0] == height and masks.shape[1] == width:
            masks = np.transpose(masks, (2, 0, 1))
        elif masks.ndim != 3:
            raise ValueError(f"Unexpected SAM3 mask shape: {masks.shape}")

        if masks.shape[-2:] != (height, width):
            raise ValueError(
                f"Mask/image size mismatch. masks={masks.shape}, image={(height, width)}"
            )
        return np.where(masks > 0, 255, 0).astype(np.uint8)

    def _extract_sam_boxes(
        self, raw_boxes: Any, image_size: tuple[int, int]
    ) -> list[BoundingBox]:
        boxes_np = self._to_numpy(raw_boxes)
        if boxes_np is None or boxes_np.size == 0:
            return []
        boxes = np.array(boxes_np, dtype=float).reshape(-1, 4)
        width, height = image_size
        parsed: list[BoundingBox] = []
        for x1, y1, x2, y2 in boxes:
            left = max(0, min(width, int(round(min(x1, x2)))))
            top = max(0, min(height, int(round(min(y1, y2)))))
            right = max(0, min(width, int(round(max(x1, x2)))))
            bottom = max(0, min(height, int(round(max(y1, y2)))))
            w = right - left
            h = bottom - top
            if w > 0 and h > 0:
                parsed.append(BoundingBox(x=left, y=top, w=w, h=h))
        parsed.sort(key=lambda box: box.area, reverse=True)
        return parsed

    def _extract_boxes_from_masks(self, masks: np.ndarray) -> list[BoundingBox]:
        boxes: list[BoundingBox] = []
        for mask in masks:
            ys, xs = np.where(mask > 0)
            if ys.size == 0 or xs.size == 0:
                continue
            left = int(xs.min())
            top = int(ys.min())
            right = int(xs.max()) + 1
            bottom = int(ys.max()) + 1
            boxes.append(
                BoundingBox(
                    x=left,
                    y=top,
                    w=right - left,
                    h=bottom - top,
                )
            )
        boxes.sort(key=lambda box: box.area, reverse=True)
        return boxes

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        if hasattr(value, "cpu"):
            return value.cpu().numpy()
        return np.asarray(value)
