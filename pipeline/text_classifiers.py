from __future__ import annotations

import re
from typing import Protocol

try:
    from .ocr_engines import OcrDetection
except ImportError:
    from ocr_engines import OcrDetection

TEXT_FILTERS = {"none", "short_text"}

# Keep letters (including accented, for Italian text) and digits.
_NON_ALNUM_RE = re.compile(r"[^0-9A-Za-zÀ-ÖØ-öø-ÿ]+")


class RedactionClassifier(Protocol):
    def should_redact(self, detection: OcrDetection) -> bool: ...


class RedactAllClassifier:
    def should_redact(self, detection: OcrDetection) -> bool:
        return True


class ShortTextSkipClassifier:
    """Skips detections that are too short to be personal information,
    e.g. L/R laterality markers on radiographs."""

    def __init__(self, max_skip_chars: int = 2) -> None:
        if max_skip_chars < 0:
            raise ValueError("max_skip_chars must be >= 0")
        self._max_skip_chars = max_skip_chars

    def should_redact(self, detection: OcrDetection) -> bool:
        if not detection.text.strip():
            # Detection-only engines (Surya, OnnxTR, RapidOCR with recognition
            # off) report text="". Without this carve-out every one of their
            # boxes would be skipped, which the benchmark would report as a
            # precision win rather than as total recall failure. Recall first:
            # redact content we cannot read.
            return True
        normalized = _NON_ALNUM_RE.sub("", detection.text)
        return len(normalized) > self._max_skip_chars


def build_classifier(
    text_filter: str, short_text_max_chars: int
) -> RedactionClassifier:
    if text_filter == "none":
        return RedactAllClassifier()
    if text_filter == "short_text":
        return ShortTextSkipClassifier(max_skip_chars=short_text_max_chars)
    raise ValueError(
        f"Invalid text_filter '{text_filter}'. Valid values: {sorted(TEXT_FILTERS)}"
    )
