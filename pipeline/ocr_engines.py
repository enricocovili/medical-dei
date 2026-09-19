from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OcrDetection:
    quad: list[list[int]]  # 4 corner points [x, y], possibly rotated
    text: str
    confidence: float


class OcrEngine(Protocol):
    name: str

    def detect(self, image: np.ndarray) -> list[OcrDetection]: ...


def parse_quad_results(raw_results: Any) -> list[OcrDetection]:
    parsed: list[OcrDetection] = []
    for item in raw_results:
        if not isinstance(item, (tuple, list)) or len(item) < 3:
            continue
        bbox, text, prob = item[0], item[1], item[2]
        if not isinstance(bbox, (tuple, list)) or len(bbox) < 4:
            continue
        quad = [[int(round(point[0])), int(round(point[1]))] for point in bbox]
        parsed.append(OcrDetection(quad=quad, text=str(text), confidence=float(prob)))
    return parsed


def rect_to_quad(x1: float, y1: float, x2: float, y2: float) -> list[list[int]]:
    return [
        [int(round(x1)), int(round(y1))],
        [int(round(x2)), int(round(y1))],
        [int(round(x2)), int(round(y2))],
        [int(round(x1)), int(round(y2))],
    ]


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Normalise an engine's input to a contiguous 3-channel RGB array.

    Engines receive whatever the preprocess chain emitted, which is 2-D whenever
    the chain contains "grayscale"; and TiledStrategy hands out non-contiguous
    slices, which several backends mishandle.
    """
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return np.ascontiguousarray(image)


@dataclass(frozen=True, slots=True)
class EnsembleMember:
    engine: OcrEngine
    # Per-member floor. A single global ocr_min_confidence cannot work across an
    # ensemble: EasyOCR reports a recognition softmax, a DB detector a box
    # score, a detection-only engine a constant — they are not commensurable,
    # and one threshold would mute whichever member scores conservatively.
    min_confidence: float = 0.0


class EnsembleOcrEngine:
    """Union of several engines' detections.

    Recall-first: different architectures fail on different images, so the union
    finds more text than any member alone. The Deidentifier's union-find merge
    then collapses the near-duplicates that members agree on.

    A member that raises is logged and skipped rather than failing the whole
    run, unless fail_mode is "raise".
    """

    def __init__(self, members: list[EnsembleMember], *, fail_mode: str = "warn") -> None:
        if not members:
            raise ValueError("ensemble requires at least one member engine")
        if fail_mode not in {"warn", "raise"}:
            raise ValueError("ensemble fail_mode must be 'warn' or 'raise'")
        self._members = members
        self._fail_mode = fail_mode
        self.name = "ensemble[" + "+".join(m.engine.name for m in members) + "]"

    def detect(self, image: np.ndarray) -> list[OcrDetection]:
        detections: list[OcrDetection] = []
        for member in self._members:
            try:
                found = member.engine.detect(image)
            except Exception as exc:  # noqa: BLE001 — one bad member must not lose the rest
                if self._fail_mode == "raise":
                    raise
                _logger.warning(
                    "ensemble member '%s' failed: %s: %s",
                    member.engine.name,
                    type(exc).__name__,
                    exc,
                )
                continue
            kept = [det for det in found if det.confidence >= member.min_confidence]
            _logger.debug(
                "ensemble member '%s': %d detections (%d above its floor)",
                member.engine.name,
                len(found),
                len(kept),
            )
            detections.extend(kept)
        return detections


@dataclass(frozen=True, slots=True)
class EasyOcrParams:
    text_threshold: float = 0.5
    low_text: float = 0.4
    link_threshold: float = 0.4
    canvas_size: int = 4000
    mag_ratio: float = 1.0


class EasyOcrEngine:
    name = "easyocr"

    def __init__(self, reader: Any, params: EasyOcrParams | None = None) -> None:
        self._reader = reader
        self._params = params or EasyOcrParams()

    def detect(self, image: np.ndarray) -> list[OcrDetection]:
        results = self._reader.readtext(
            image,
            text_threshold=self._params.text_threshold,
            low_text=self._params.low_text,
            link_threshold=self._params.link_threshold,
            canvas_size=self._params.canvas_size,
            mag_ratio=self._params.mag_ratio,
        )
        return parse_quad_results(results)


def build_easyocr_engine(
    langs: list[str], gpu: bool, params: EasyOcrParams | None = None
) -> EasyOcrEngine:
    import easyocr
    import torch

    if gpu and not torch.cuda.is_available():
        _logger.warning("easyocr_gpu is enabled but CUDA is unavailable — using CPU")
        gpu = False
    reader = easyocr.Reader(langs, gpu=gpu)
    return EasyOcrEngine(reader=reader, params=params)


def _paddle_result_field(result: Any, key: str) -> Any:
    try:
        value = result[key]
    except (KeyError, TypeError, IndexError):
        value = None
    if value is None:
        json_payload = getattr(result, "json", None)
        if isinstance(json_payload, dict):
            inner = json_payload.get("res", json_payload)
            if isinstance(inner, dict):
                value = inner.get(key)
    return value


class PaddleOcrEngine:
    name = "paddleocr"

    def __init__(self, ocr: Any) -> None:
        self._ocr = ocr

    def detect(self, image: np.ndarray) -> list[OcrDetection]:
        if image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = self._ocr.predict(bgr)
        detections: list[OcrDetection] = []
        for result in results:
            polys = _paddle_result_field(result, "rec_polys")
            if polys is None:
                polys = _paddle_result_field(result, "dt_polys")
            texts = _paddle_result_field(result, "rec_texts") or []
            scores = _paddle_result_field(result, "rec_scores") or []
            if polys is None:
                continue
            for poly, text, score in zip(polys, texts, scores):
                quad = [
                    [int(round(float(point[0]))), int(round(float(point[1])))]
                    for point in poly
                ]
                if len(quad) < 4:
                    continue
                detections.append(
                    OcrDetection(quad=quad, text=str(text), confidence=float(score))
                )
        return detections


def _paddle_gpu_available() -> bool:
    try:
        import paddle

        return bool(
            paddle.device.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > 0
        )
    except Exception:  # noqa: BLE001 — any probe failure means "no usable GPU"
        return False


def build_paddle_engine(
    device: str, det_model: str, rec_model: str
) -> PaddleOcrEngine:
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise ValueError(
            "paddleocr is not installed; install the 'paddle' optional dependency "
            "group (pyproject) or build the Docker image with INSTALL_PADDLE=1"
        ) from exc

    if device == "gpu" and not _paddle_gpu_available():
        _logger.warning(
            "paddleocr_device is 'gpu' but no usable GPU (paddlepaddle CPU build "
            "or no CUDA device) — using CPU"
        )
        device = "cpu"

    # enable_mkldnn=False: paddlepaddle 3.x CPU inference crashes with
    # "ConvertPirAttribute2RuntimeAttribute not support" when oneDNN is on.
    ocr = PaddleOCR(
        text_detection_model_name=det_model,
        text_recognition_model_name=rec_model,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=device,
        enable_mkldnn=False,
    )
    return PaddleOcrEngine(ocr=ocr)
