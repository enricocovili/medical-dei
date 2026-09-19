from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
from PIL import Image

try:
    from .deidentifier_component import DeidentificationResult
    from .models import (
        ImageEntry,
        LoadedImage,
        MaskTransformResult,
        SegmentationResult,
    )
    from .ocr_engines import OcrDetection
except ImportError:
    from deidentifier_component import DeidentificationResult
    from models import (
        ImageEntry,
        LoadedImage,
        MaskTransformResult,
        SegmentationResult,
    )
    from ocr_engines import OcrDetection


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
    """The redaction stage.

    run() is the real entry point — it returns the redacted image together with
    the boxes and the raw detections, which the deidentification stage records.
    deidentify() is the convenience wrapper that keeps only the image.
    """

    def run(self, image_rgb: np.ndarray, image_name: str) -> DeidentificationResult:
        ...

    def deidentify(self, image_rgb: np.ndarray, image_name: str) -> np.ndarray:
        ...


class OcrEngine(Protocol):
    """A text-detection backend.

    This is the extension point for a new OCR model: a name and a detect().
    Quad coordinates must be in the pixel space of the array passed in, because
    the resolution strategies translate them (tile offsets, upscale factors)
    and the deidentifier measures its filters against that same array.

    Implementations must accept both 2-D grayscale and 3-D RGB input, since the
    preprocess chain emits 2-D whenever it contains "grayscale"
    (ocr_engines.to_rgb handles this). Engines with no meaningful confidence
    should report 1.0: the confidence filter, the merge's max(), and the
    benchmark all assume a float in [0, 1].
    """

    name: str

    def detect(self, image: np.ndarray) -> list[OcrDetection]:
        ...


class ReportWriter(Protocol):
    def write(self, output_path: Path, entries: list[ImageEntry]) -> None:
        ...
