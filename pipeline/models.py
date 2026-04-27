from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

type BoxTuple = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return int(self.w * self.h)

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]


@dataclass(frozen=True, slots=True)
class AlarmInfo:
    triggered: bool
    motivation: str


@dataclass(frozen=True, slots=True)
class StageTimes:
    sam3_inference: float
    erosion_diffusion: float
    deidentification: float


@dataclass(frozen=True, slots=True)
class ImageMetrics:
    times: StageTimes
    original_size: tuple[int, int]
    cut_size: tuple[int, int]
    rotation_angle: float


@dataclass(frozen=True, slots=True)
class ImageEntry:
    name: str
    alarm: AlarmInfo
    bounding_boxes: list[BoundingBox]
    deidentification_boxes: list[BoundingBox]
    metrics: ImageMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "alarm": {
                "triggered": self.alarm.triggered,
                "motivation": self.alarm.motivation,
            },
            "bounding_boxes": [box.as_list() for box in self.bounding_boxes],
            "deidentification_boxes": [
                box.as_list() for box in self.deidentification_boxes
            ],
            "metrics": {
                "times": {
                    "sam3_inference": self.metrics.times.sam3_inference,
                    "erosion_diffusion": self.metrics.times.erosion_diffusion,
                    "deidentification": self.metrics.times.deidentification,
                },
                "original_size": list(self.metrics.original_size),
                "cut_size": list(self.metrics.cut_size),
                "rotation_angle": self.metrics.rotation_angle,
            },
        }


@dataclass(frozen=True, slots=True)
class LoadedImage:
    name: str
    path: Path
    pil_image: Image.Image
    rgb_image: np.ndarray
    size: tuple[int, int]


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    masks: np.ndarray
    sam_boxes: list[BoundingBox]


@dataclass(frozen=True, slots=True)
class MaskTransformResult:
    bounding_boxes: list[BoundingBox]
    alarm: AlarmInfo
    cut_image: np.ndarray
    cut_size: tuple[int, int]
    rotation_angle: float
