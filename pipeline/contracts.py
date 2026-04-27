from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
from PIL import Image

try:
    from .models import (
        ImageEntry,
        LoadedImage,
        MaskTransformResult,
        SegmentationResult,
    )
except ImportError:
    from models import (
        ImageEntry,
        LoadedImage,
        MaskTransformResult,
        SegmentationResult,
    )


class ImageSource(Protocol):
    def iter_images(self) -> Iterable[LoadedImage]:
        ...


class Segmenter(Protocol):
    def infer(self, image: Image.Image, prompt: str) -> SegmentationResult:
        ...


class MaskTransformer(Protocol):
    def transform(
        self, image_rgb: np.ndarray, segmentation: SegmentationResult
    ) -> MaskTransformResult:
        ...


class Deidentifier(Protocol):
    def deidentify(self, image_rgb: np.ndarray, image_name: str) -> np.ndarray:
        ...


class ReportWriter(Protocol):
    def write(self, output_path: Path, entries: list[ImageEntry]) -> None:
        ...
