"""Vision-language OCR over any OpenAI-compatible endpoint.

One transport class serves the hosted API and every locally-served model
(`vllm serve ...`, SGLang, ...); only the prompt and the response parser differ.
That boundary is not a stylistic choice — it is forced:

  * DeepSeek-OCR-2 pins torch==2.6.0 / transformers==4.46.3 / flash-attn==2.7.3.
    torch 2.6 predates sm_120, so it cannot run in-process on a Blackwell card
    at all, and installing it would drag SAM3's torch>=2.7 backwards.
  * GLM-OCR wants transformers>=5.3.0, which is incompatible with DeepSeek's
    4.46.3, so those two can never share an environment either.

Running each model behind HTTP dissolves all of it, and costs nothing: `openai`
is already a core dependency, the pipeline image gains no packages, and the
model stays resident across benchmark variants instead of reloading per run.

Caveat that shapes how these are used: published reviews report VLM grounding
coordinates drifting between runs, and redaction blacks out pixels, so a box
20px off leaks PHI. Treat a VLM as a recall booster inside an ensemble (with
vlm_box_dilate_px absorbing the drift), not as the sole source of geometry.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

try:
    from .ocr_engines import OcrDetection, rect_to_quad, to_rgb
except ImportError:
    from ocr_engines import OcrDetection, rect_to_quad, to_rgb

_logger = logging.getLogger(__name__)

VLM_RESPONSE_FORMATS = {"json", "deepseek_grounding"}
# How to read the numbers a model returns. Kept orthogonal to the response
# format so a new model can be characterised by flipping a config key instead of
# editing a parser: send the same crop at two sizes, and if the numbers do not
# change they are normalised.
VLM_COORD_SPACES = {"pixel", "normalized_999", "normalized_1"}

PROMPT_JSON = (
    "Sei un sistema di anonimizzazione di immagini mediche. "
    "Nell'immagine, individua OGNI porzione di testo identificativo. "
    "Restituisci SOLO un oggetto JSON con questa forma esatta:\n"
    '{"detections": [{"text": "<testo>", "bbox": [x, y, w, h]}]}\n'
    "dove x,y sono l'angolo in alto a sinistra e w,h larghezza e altezza in PIXEL, "
    "con origine (0,0) in alto a sinistra dell'immagine. "
    "Non aggiungere spiegazioni, solo il JSON."
)

PROMPT_GROUNDING = "<image>\n<|grounding|>Locate every piece of text in the image."

DEFAULT_PROMPTS = {
    "json": PROMPT_JSON,
    "deepseek_grounding": PROMPT_GROUNDING,
}

# <|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2], ...]<|/det|>
_GROUNDING_RE = re.compile(
    r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>\s*(\[\[.*?\]\])\s*<\|/det\|>", re.S
)


def extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a chat reply that may be fenced or chatty."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {"detections": []}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"detections": []}


def detections_to_boxes(payload: dict[str, Any]) -> tuple[list[list[int]], list[str]]:
    """[x, y, w, h] boxes and their texts from the JSON response shape."""
    boxes: list[list[int]] = []
    texts: list[str] = []
    for detection in payload.get("detections", []):
        if not isinstance(detection, dict):
            continue
        bbox = detection.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x, y, width, height = (int(round(float(value))) for value in bbox)
        except (TypeError, ValueError):
            continue
        if width <= 0 or height <= 0:
            continue
        boxes.append([x, y, width, height])
        texts.append(str(detection.get("text", "")))
    return boxes, texts


def _scale(value: float, extent: int, coord_space: str) -> float:
    if coord_space == "normalized_999":
        return value / 999.0 * extent
    if coord_space == "normalized_1":
        return value * extent
    return value


def parse_json_detections(
    content: str, width: int, height: int, coord_space: str
) -> list[OcrDetection]:
    boxes, texts = detections_to_boxes(extract_json(content))
    detections = []
    for (x, y, box_width, box_height), text in zip(boxes, texts):
        x1 = _scale(x, width, coord_space)
        y1 = _scale(y, height, coord_space)
        x2 = _scale(x + box_width, width, coord_space)
        y2 = _scale(y + box_height, height, coord_space)
        detections.append(
            OcrDetection(quad=rect_to_quad(x1, y1, x2, y2), text=text, confidence=1.0)
        )
    return detections


def parse_grounding_detections(
    content: str, width: int, height: int, coord_space: str
) -> list[OcrDetection]:
    """DeepSeek-OCR-style <|ref|>/<|det|> tags; coordinates normalised to 0-999."""
    detections: list[OcrDetection] = []
    for label, raw in _GROUNDING_RE.findall(content):
        try:
            groups = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for group in groups:
            if not isinstance(group, (list, tuple)) or len(group) != 4:
                continue
            try:
                x1, y1, x2, y2 = (float(value) for value in group)
            except (TypeError, ValueError):
                continue
            detections.append(
                OcrDetection(
                    quad=rect_to_quad(
                        _scale(x1, width, coord_space),
                        _scale(y1, height, coord_space),
                        _scale(x2, width, coord_space),
                        _scale(y2, height, coord_space),
                    ),
                    text=label.strip(),
                    confidence=1.0,
                )
            )
    return detections


VLM_PARSERS: dict[str, Callable[[str, int, int, str], list[OcrDetection]]] = {
    "json": parse_json_detections,
    "deepseek_grounding": parse_grounding_detections,
}

DEFAULT_COORD_SPACES = {"json": "pixel", "deepseek_grounding": "normalized_999"}


@dataclass(frozen=True, slots=True)
class VlmParams:
    model: str = "gpt-5.4-mini"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    prompt: str | None = None
    response_format: str = "json"
    coord_space: str | None = None
    # Extra slack on every box, on top of the global deid_padding_px, to absorb
    # grounding drift. Geometric detectors do not need it; VLMs do.
    box_dilate_px: int = 24
    max_side_px: int = 2048
    timeout_s: float = 120.0
    confidence: float = 1.0


class VlmEngine:
    name = "vlm"

    def __init__(self, client: Any, params: VlmParams) -> None:
        self._client = client
        self._params = params
        self._prompt = params.prompt or DEFAULT_PROMPTS[params.response_format]
        self._parser = VLM_PARSERS[params.response_format]
        self._coord_space = params.coord_space or DEFAULT_COORD_SPACES[
            params.response_format
        ]

    def detect(self, image: np.ndarray) -> list[OcrDetection]:
        rgb = to_rgb(image)
        original_h, original_w = rgb.shape[:2]
        scale = min(1.0, self._params.max_side_px / max(original_h, original_w))
        sent = (
            cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else rgb
        )
        height, width = sent.shape[:2]

        ok, buffer = cv2.imencode(".png", cv2.cvtColor(sent, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError("failed to PNG-encode the image for the VLM request")
        data_url = "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode(
            "ascii"
        )

        response = self._client.chat.completions.create(
            model=self._params.model,
            timeout=self._params.timeout_s,
            temperature=0,  # benchmark runs must be reproducible
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        content = response.choices[0].message.content or ""
        detections = self._parser(content, width, height, self._coord_space)

        # Map back into the coordinate space of the array we were handed, so the
        # engine composes with TiledStrategy and UpscaleStrategy.
        inverse = 1.0 / scale if scale > 0 else 1.0
        pad = self._params.box_dilate_px
        restored: list[OcrDetection] = []
        for detection in detections:
            xs = [point[0] * inverse for point in detection.quad]
            ys = [point[1] * inverse for point in detection.quad]
            restored.append(
                OcrDetection(
                    quad=rect_to_quad(
                        min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad
                    ),
                    text=detection.text,
                    confidence=self._params.confidence,
                )
            )
        return restored


def build_vlm_engine(params: VlmParams) -> VlmEngine:
    if params.response_format not in VLM_PARSERS:
        raise ValueError(
            f"Unknown vlm_response_format '{params.response_format}'. "
            f"Valid values: {sorted(VLM_PARSERS)}"
        )
    if params.coord_space is not None and params.coord_space not in VLM_COORD_SPACES:
        raise ValueError(
            f"Unknown vlm_coord_space '{params.coord_space}'. "
            f"Valid values: {sorted(VLM_COORD_SPACES)}"
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ValueError("the 'openai' package is required for the vlm engine") from exc

    api_key = os.environ.get(params.api_key_env) if params.api_key_env else None
    if params.base_url and not api_key:
        # Locally served models ignore the key but the client insists on one.
        api_key = "EMPTY"
    if not api_key:
        raise ValueError(
            f"No API key found in ${params.api_key_env}; set it, or point "
            "vlm_base_url at a locally served OpenAI-compatible endpoint"
        )
    client = OpenAI(api_key=api_key, base_url=params.base_url or None)
    _logger.info(
        "VLM engine ready (model=%s, endpoint=%s, format=%s, coords=%s)",
        params.model,
        params.base_url or "hosted OpenAI",
        params.response_format,
        params.coord_space or DEFAULT_COORD_SPACES[params.response_format],
    )
    return VlmEngine(client=client, params=params)
