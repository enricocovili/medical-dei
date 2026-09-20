"""Specialised text-detection backends.

These live apart from ocr_engines.py so that importing the pipeline on a machine
without onnxruntime or torch-based OCR still works: app.build_ocr_engine imports
this module lazily, inside the per-engine builder.

Why detectors rather than a VLM for the box geometry: redaction blacks out
pixels, so a box that is 20px off leaks PHI. Published reviews of VLM grounding
report coordinates that drift between runs, and a 2025 study of large multimodal
models for burned-in PHI found they beat EasyOCR on transcription while *not*
consistently improving PHI detection. Detectors own the geometry here; a VLM is
a recall booster on top (see ocr_engines_vlm.py).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

try:
    from .ocr_engines import OcrDetection, parse_quad_results, rect_to_quad, to_rgb
except ImportError:
    from ocr_engines import OcrDetection, parse_quad_results, rect_to_quad, to_rgb

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RapidOCR — the PaddleOCR PP-OCRv5/v6 DB detector over ONNX Runtime.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RapidOcrParams:
    """Detection thresholds, which the paddleocr backend hard-codes away.

    These are the cheapest recall lever in the repo. Defaults here are
    deliberately looser than the library's, because over-redaction is
    acceptable and a missed text region is a PHI leak:
      text_score     library 0.5  -> 0.3
      box_thresh     library 0.5  -> 0.3   (box acceptance)
      thresh         library 0.3  -> 0.2   (pixel-level binarisation)
      unclip_ratio   library 1.6  -> 2.0   (inflates every detected polygon)
      limit_side_len library ~736 -> 2560  (736 would crush a 3000px panoramic)
    """

    text_score: float = 0.3
    box_thresh: float = 0.3
    thresh: float = 0.2
    unclip_ratio: float = 2.0
    limit_side_len: int = 2560
    use_rec: bool = True


class RapidOcrEngine:
    name = "rapidocr"

    def __init__(self, ocr: Any, params: RapidOcrParams) -> None:
        self._ocr = ocr
        self._params = params

    def detect(self, image: np.ndarray) -> list[OcrDetection]:
        bgr = cv2.cvtColor(to_rgb(image), cv2.COLOR_RGB2BGR)
        result = self._ocr(bgr)
        if result is None:
            return []

        # rapidocr v3 returns an object with .boxes (N,4,2) / .txts / .scores.
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            # rapidocr_onnxruntime v1/v2 returned (list[[quad, text, score]], elapse).
            legacy = result[0] if isinstance(result, tuple) else result
            return parse_quad_results(legacy or [])

        texts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        detections: list[OcrDetection] = []
        for index, quad in enumerate(boxes):
            points = [
                [int(round(float(point[0]))), int(round(float(point[1])))]
                for point in quad[:4]
            ]
            if len(points) < 4:
                continue
            text = str(texts[index]) if texts is not None and index < len(texts) else ""
            score = (
                float(scores[index])
                if scores is not None and index < len(scores)
                else 1.0
            )
            detections.append(OcrDetection(quad=points, text=text, confidence=score))
        return detections


def _onnx_providers() -> list[str]:
    try:
        import onnxruntime

        return list(onnxruntime.get_available_providers())
    except Exception:  # noqa: BLE001 — any probe failure means "assume CPU"
        return []


def build_rapidocr_engine(device: str, params: RapidOcrParams) -> RapidOcrEngine:
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise ValueError(
            "rapidocr is not installed; install the 'rapidocr' (CPU) or "
            "'rapidocr-gpu' optional dependency group"
        ) from exc

    providers = _onnx_providers()
    use_cuda = device == "gpu" and "CUDAExecutionProvider" in providers
    if device == "gpu" and not use_cuda:
        # The official onnxruntime-gpu wheels carry no sm_120 kernels, so on a
        # Blackwell card the CUDA provider is simply absent and everything
        # silently runs on CPU. Say so, or a slow run looks like a slow model.
        _logger.warning(
            "rapidocr_device is 'gpu' but onnxruntime exposes no "
            "CUDAExecutionProvider (available: %s) — using CPU",
            providers or "none",
        )

    config = {
        "Global.text_score": params.text_score,
        "Det.box_thresh": params.box_thresh,
        "Det.thresh": params.thresh,
        "Det.unclip_ratio": params.unclip_ratio,
        "Det.limit_side_len": params.limit_side_len,
        "Det.engine_cfg.use_cuda": use_cuda,
        "Rec.engine_cfg.use_cuda": use_cuda,
    }
    try:
        ocr = RapidOCR(params=config)
    except (TypeError, KeyError, ValueError) as exc:
        _logger.warning(
            "RapidOCR rejected the tuned detection parameters (%s); falling back "
            "to library defaults, which are tighter and will cost recall",
            exc,
        )
        ocr = RapidOCR()
    _logger.info("RapidOCR ready (cuda=%s, providers=%s)", use_cuda, providers or "none")
    return RapidOcrEngine(ocr=ocr, params=params)


# ---------------------------------------------------------------------------
# Surya — a detector with a different lineage from CRAFT and DB, which is what
# makes it worth having in the ensemble: it fails on different images.
# ---------------------------------------------------------------------------


class SuryaDetEngine:
    """Text-line detection only; every detection carries text="".

    Recognition would cost a second model and buy nothing: the text field feeds
    only the short_text classifier (which treats "" as "redact", see
    text_classifiers) and human inspection of records.json.
    """

    name = "surya"

    def __init__(self, detector: Any, min_confidence: float = 0.0) -> None:
        self._detector = detector
        self._min_confidence = min_confidence

    def detect(self, image: np.ndarray) -> list[OcrDetection]:
        from PIL import Image

        pil_image = Image.fromarray(to_rgb(image))
        pages = self._detector([pil_image])
        if not pages:
            return []

        detections: list[OcrDetection] = []
        for box in getattr(pages[0], "bboxes", []) or []:
            confidence = float(getattr(box, "confidence", None) or 1.0)
            if confidence < self._min_confidence:
                continue
            polygon = getattr(box, "polygon", None)
            if polygon is not None and len(polygon) >= 4:
                quad = [
                    [int(round(float(point[0]))), int(round(float(point[1])))]
                    for point in polygon[:4]
                ]
            else:
                x1, y1, x2, y2 = (float(value) for value in box.bbox)
                quad = rect_to_quad(x1, y1, x2, y2)
            detections.append(OcrDetection(quad=quad, text="", confidence=confidence))
        return detections


def build_surya_engine(device: str, min_confidence: float = 0.0) -> SuryaDetEngine:
    try:
        from surya.detection import DetectionPredictor
    except ImportError as exc:
        raise ValueError(
            "surya-ocr is not installed; install the 'surya' optional dependency group"
        ) from exc

    import torch

    if device == "cuda" and not torch.cuda.is_available():
        _logger.warning("surya_device is 'cuda' but CUDA is unavailable — using CPU")
        device = "cpu"
    try:
        detector = DetectionPredictor(device=device)
    except TypeError:
        # Older releases pick the device from the environment instead.
        detector = DetectionPredictor()
    return SuryaDetEngine(detector=detector, min_confidence=min_confidence)


# ---------------------------------------------------------------------------
# OnnxTR — DBNet with a ResNet-50 backbone, over the same ONNX Runtime as
# RapidOCR but with a different backbone and training set.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OnnxTrParams:
    arch: str = "db_resnet50"
    # Library defaults are bin_thresh 0.3 / box_thresh 0.1 / unclip_ratio 1.5.
    # Loosened here for the same reason as RapidOcrParams: over-redaction is
    # cheap, a missed text region is a PHI leak.
    bin_thresh: float = 0.2
    box_thresh: float = 0.05
    unclip_ratio: float = 2.0


class OnnxTrEngine:
    """Detection only; every detection carries text="".

    Coordinates come back RELATIVE to the image (0-1), unlike every other
    backend here, so they must be scaled by the width and height of the array
    that was passed in. Getting this wrong collapses every box into the
    top-left corner, where the area filter then silently discards it — the
    failure looks exactly like "the engine found nothing".
    """

    name = "onnxtr"

    def __init__(self, predictor: Any) -> None:
        self._predictor = predictor

    def detect(self, image: np.ndarray) -> list[OcrDetection]:
        rgb = to_rgb(image)
        height, width = rgb.shape[:2]
        pages = self._predictor([rgb])
        detections: list[OcrDetection] = []
        for page in pages:
            # Newer releases return a bare ndarray per page; older ones a dict
            # keyed by class name.
            rows = page["words"] if isinstance(page, dict) else page
            for row in np.asarray(rows):
                row = np.asarray(row)
                if row.ndim == 2:  # (N, 2) polygon, still relative
                    quad = [
                        [int(round(float(px) * width)), int(round(float(py) * height))]
                        for px, py in row[:4]
                    ]
                    confidence = 1.0
                    if len(quad) < 4:
                        continue
                else:
                    if row.shape[0] < 4:
                        continue
                    x1, y1, x2, y2 = (float(value) for value in row[:4])
                    confidence = float(row[4]) if row.shape[0] > 4 else 1.0
                    quad = rect_to_quad(
                        x1 * width, y1 * height, x2 * width, y2 * height
                    )
                detections.append(
                    OcrDetection(quad=quad, text="", confidence=confidence)
                )
        return detections


def build_onnxtr_engine(device: str, params: OnnxTrParams) -> OnnxTrEngine:
    try:
        from onnxtr.models import detection_predictor
    except ImportError as exc:
        raise ValueError(
            "onnxtr is not installed; install the 'onnxtr' optional dependency group"
        ) from exc

    providers = _onnx_providers()
    if device == "gpu" and "CUDAExecutionProvider" not in providers:
        _logger.warning(
            "onnxtr_device is 'gpu' but onnxruntime exposes no "
            "CUDAExecutionProvider (available: %s) — using CPU",
            providers or "none",
        )

    predictor = detection_predictor(arch=params.arch, assume_straight_pages=True)
    postprocessor = getattr(getattr(predictor, "model", None), "postprocessor", None)
    if postprocessor is not None:
        postprocessor.bin_thresh = params.bin_thresh
        postprocessor.box_thresh = params.box_thresh
        if hasattr(postprocessor, "unclip_ratio"):
            postprocessor.unclip_ratio = params.unclip_ratio
    else:
        _logger.warning(
            "onnxtr postprocessor not reachable; detection thresholds left at "
            "library defaults, which are tighter and will cost recall"
        )
    return OnnxTrEngine(predictor=predictor)
