from __future__ import annotations

import argparse
import json
import logging
import time
import tomllib
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .deidentifier_component import (
        REDACTION_MODES,
        Deidentifier,
        DeidentifierParams,
    )
    from .image_source import LocalImageSource
    from .logging_component import PipelineEventLogger
    from .mask_component import MaskPostprocessor
    from .models import (
        AlarmInfo,
        BoundingBox,
        ImageEntry,
        ImageMetrics,
        SegmentationResult,
        StageTimes,
    )
    from .ocr_engines import (
        EasyOcrParams,
        OcrEngine,
        build_easyocr_engine,
        build_paddle_engine,
    )
    from .ocr_preprocess import PreprocessParams, apply_chain, validate_steps
    from .ocr_strategies import (
        RESOLUTION_STRATEGIES,
        UPSCALE_INTERPOLATIONS,
        FullImageStrategy,
        ResolutionStrategy,
        TiledStrategy,
        UpscaleStrategy,
    )
    from .report_writer import JsonReportWriter
    from .sam3_component import Sam3ImageSegmenter
    from .text_classifiers import TEXT_FILTERS, build_classifier
except ImportError:
    from deidentifier_component import (
        REDACTION_MODES,
        Deidentifier,
        DeidentifierParams,
    )
    from image_source import LocalImageSource
    from logging_component import PipelineEventLogger
    from mask_component import MaskPostprocessor
    from models import (
        AlarmInfo,
        BoundingBox,
        ImageEntry,
        ImageMetrics,
        SegmentationResult,
        StageTimes,
    )
    from ocr_engines import (
        EasyOcrParams,
        OcrEngine,
        build_easyocr_engine,
        build_paddle_engine,
    )
    from ocr_preprocess import PreprocessParams, apply_chain, validate_steps
    from ocr_strategies import (
        RESOLUTION_STRATEGIES,
        UPSCALE_INTERPOLATIONS,
        FullImageStrategy,
        ResolutionStrategy,
        TiledStrategy,
        UpscaleStrategy,
    )
    from report_writer import JsonReportWriter
    from sam3_component import Sam3ImageSegmenter
    from text_classifiers import TEXT_FILTERS, build_classifier


CONFIG_FILE = Path("setups/pipeline_config.toml")
CONFIG_SECTION = "pipeline"
RUN_MODES = {"full", "sam3", "postprocess", "deidentification", "report"}


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    run_mode: str
    input_path: Path
    output_json: Path
    artifacts_dir: Path
    save_artifacts: bool
    prompt: str
    fallback_prompt: str | None
    kernel_size: int
    iterations: int
    large_bb_area_ratio: float
    easyocr_langs: list[str]
    easyocr_gpu: bool
    save_deidentified_dir: Path | None
    merge_distance_px: int
    max_box_area_px: int | None
    max_box_area_ratio: float | None
    ellipse_enabled: bool
    ellipse_axis_x_ratio: float
    ellipse_axis_y_ratio: float
    ellipse_proximity_px: float
    deid_padding_px: int
    ocr_engine: str
    ocr_min_confidence: float
    easyocr_text_threshold: float
    easyocr_low_text: float
    easyocr_link_threshold: float
    easyocr_canvas_size: int
    easyocr_mag_ratio: float
    paddleocr_device: str
    paddleocr_det_model: str
    paddleocr_rec_model: str
    preprocess_steps: list[str]
    preprocess_variants: list[list[str]]
    clahe_clip_limit: float
    clahe_tile_grid_size: int
    clahe_tile_px: int
    morph_kernel_px: int
    stretch_low_pct: float
    stretch_high_pct: float
    preprocess_gamma: float
    unsharp_sigma: float
    unsharp_amount: float
    ocr_dual_pass_invert: bool
    ocr_upscale_factor: float
    ocr_upscale_interpolation: str
    resolution_strategy: str
    tile_size_px: int
    tile_overlap_px: int
    tile_dedupe_iou: float
    text_filter: str
    short_text_max_chars: int
    redaction_mode: str
    record_raw_detections: bool


@dataclass(frozen=True, slots=True)
class StageArtifacts:
    sam3_masks_dir: Path
    sam3_records_json: Path
    postprocess_crops_dir: Path
    postprocess_records_json: Path
    deid_images_dir: Path
    deid_records_json: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run modular stage-based CV pipeline.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_pipeline_section() -> dict[str, object]:
    config_path = _repo_root() / CONFIG_FILE
    with config_path.open("rb") as file_handle:
        raw = tomllib.load(file_handle)
    section = raw.get(CONFIG_SECTION)
    if not isinstance(section, dict):
        raise ValueError(
            f"Missing or invalid [{CONFIG_SECTION}] section in {config_path}"
        )
    return section


def _require_str(config: dict[str, object], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config key '{key}' must be a non-empty string")
    return value.strip()


def _require_int(config: dict[str, object], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Config key '{key}' must be an integer")
    return int(value)


def _require_float(config: dict[str, object], key: str) -> float:
    value = config.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Config key '{key}' must be a float")
    return float(value)


def _require_bool(config: dict[str, object], key: str) -> bool:
    value = config.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Config key '{key}' must be a boolean")
    return bool(value)


def _resolve_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (_repo_root() / candidate).resolve()


def _optional_str(config: dict[str, object], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Config key '{key}' must be a string when provided")
    stripped = value.strip()
    return stripped if stripped else None


def _optional_int(config: dict[str, object], key: str, default: int) -> int:
    value = config.get(key)
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Config key '{key}' must be an integer when provided")
    return int(value)


def _optional_float(config: dict[str, object], key: str, default: float) -> float:
    value = config.get(key)
    if value is None:
        return default
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Config key '{key}' must be a float when provided")
    return float(value)


def _optional_bool(config: dict[str, object], key: str, default: bool) -> bool:
    value = config.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"Config key '{key}' must be a boolean when provided")
    return bool(value)


def _optional_choice(
    config: dict[str, object], key: str, default: str, choices: set[str]
) -> str:
    value = config.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"Config key '{key}' must be a string when provided")
    normalized = value.strip().lower()
    if normalized not in choices:
        raise ValueError(
            f"Config key '{key}' must be one of {sorted(choices)}, got '{value}'"
        )
    return normalized


def _optional_str_list(config: dict[str, object], key: str) -> list[str]:
    value = config.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Config key '{key}' must be a list of strings when provided")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"All values in '{key}' must be non-empty strings")
        items.append(item.strip().lower())
    return items


def _optional_path(config: dict[str, object], key: str) -> Path | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Config key '{key}' must be a string when provided")
    value = value.strip()
    if not value:
        return None
    return _resolve_path(value)


def build_config() -> PipelineConfig:
    return build_config_from_dict(_load_pipeline_section())


def build_config_from_dict(config: dict[str, object]) -> PipelineConfig:
    run_mode_raw = config.get("run_mode", "full")
    if not isinstance(run_mode_raw, str):
        raise ValueError("Config key 'run_mode' must be a string")
    run_mode = run_mode_raw.strip().lower()
    if run_mode not in RUN_MODES:
        raise ValueError(
            f"Invalid run_mode '{run_mode}'. Valid values: {sorted(RUN_MODES)}"
        )

    artifacts_raw = config.get("artifacts_dir", "pipeline_artifacts")
    if not isinstance(artifacts_raw, str) or not artifacts_raw.strip():
        raise ValueError("Config key 'artifacts_dir' must be a non-empty string")

    langs_raw = config.get("easyocr_langs")
    if not isinstance(langs_raw, list) or not langs_raw:
        raise ValueError("Config key 'easyocr_langs' must be a non-empty list")
    langs: list[str] = []
    for language in langs_raw:
        if not isinstance(language, str) or not language.strip():
            raise ValueError("All values in 'easyocr_langs' must be non-empty strings")
        langs.append(language.strip())

    max_box_area_raw = config.get("max_box_area_px")
    if max_box_area_raw is None:
        max_box_area_px = None
    elif isinstance(max_box_area_raw, int):
        max_box_area_px = None if max_box_area_raw <= 0 else int(max_box_area_raw)
    else:
        raise ValueError("Config key 'max_box_area_px' must be an integer or null")

    save_artifacts_raw = config.get("save_artifacts", True)
    if not isinstance(save_artifacts_raw, bool):
        raise ValueError("Config key 'save_artifacts' must be a boolean")

    preprocess_steps = _optional_str_list(config, "preprocess_steps")
    validate_steps(preprocess_steps)

    # A list of independent preprocessing chains; the engine runs once per chain
    # and the detections are unioned. Empty means "just preprocess_steps".
    variants_raw = config.get("preprocess_variants")
    preprocess_variants: list[list[str]] = []
    if variants_raw is not None:
        if not isinstance(variants_raw, list):
            raise ValueError(
                "Config key 'preprocess_variants' must be a list of step lists"
            )
        for entry in variants_raw:
            if not isinstance(entry, list):
                raise ValueError(
                    "Each entry of 'preprocess_variants' must be a list of step names"
                )
            steps = [str(step).strip().lower() for step in entry]
            validate_steps(steps)
            preprocess_variants.append(steps)

    area_ratio_raw = config.get("max_box_area_ratio")
    if area_ratio_raw is None:
        max_box_area_ratio = None
    elif isinstance(area_ratio_raw, (int, float)) and not isinstance(
        area_ratio_raw, bool
    ):
        max_box_area_ratio = None if area_ratio_raw <= 0 else float(area_ratio_raw)
        if max_box_area_ratio is not None and max_box_area_ratio > 1.0:
            raise ValueError("Config key 'max_box_area_ratio' must be <= 1.0")
    else:
        raise ValueError("Config key 'max_box_area_ratio' must be a number or null")

    return PipelineConfig(
        run_mode=run_mode,
        input_path=_resolve_path(_require_str(config, "input_path")),
        output_json=_resolve_path(_require_str(config, "output_json")),
        artifacts_dir=_resolve_path(artifacts_raw.strip()),
        save_artifacts=save_artifacts_raw,
        prompt=_require_str(config, "prompt"),
        fallback_prompt=_optional_str(config, "fallback_prompt"),
        kernel_size=_require_int(config, "kernel_size"),
        iterations=_require_int(config, "iterations"),
        large_bb_area_ratio=_require_float(config, "large_bb_area_ratio"),
        easyocr_langs=langs,
        easyocr_gpu=_require_bool(config, "easyocr_gpu"),
        save_deidentified_dir=_optional_path(config, "save_deidentified_dir"),
        merge_distance_px=_require_int(config, "merge_distance_px"),
        max_box_area_px=max_box_area_px,
        max_box_area_ratio=max_box_area_ratio,
        ellipse_enabled=_optional_bool(config, "ellipse_enabled", True),
        ellipse_axis_x_ratio=_require_float(config, "ellipse_axis_x_ratio"),
        ellipse_axis_y_ratio=_require_float(config, "ellipse_axis_y_ratio"),
        ellipse_proximity_px=_require_float(config, "ellipse_proximity_px"),
        deid_padding_px=_require_int(config, "deid_padding_px"),
        ocr_engine=_optional_choice(
            config, "ocr_engine", "easyocr", {"easyocr", "paddleocr"}
        ),
        ocr_min_confidence=_optional_float(config, "ocr_min_confidence", 0.0),
        easyocr_text_threshold=_optional_float(config, "easyocr_text_threshold", 0.5),
        easyocr_low_text=_optional_float(config, "easyocr_low_text", 0.4),
        easyocr_link_threshold=_optional_float(config, "easyocr_link_threshold", 0.4),
        easyocr_canvas_size=_optional_int(config, "easyocr_canvas_size", 4000),
        easyocr_mag_ratio=_optional_float(config, "easyocr_mag_ratio", 1.0),
        paddleocr_device=_optional_str(config, "paddleocr_device") or "gpu",
        paddleocr_det_model=_optional_str(config, "paddleocr_det_model")
        or "PP-OCRv5_mobile_det",
        paddleocr_rec_model=_optional_str(config, "paddleocr_rec_model")
        or "PP-OCRv5_mobile_rec",
        preprocess_steps=preprocess_steps,
        preprocess_variants=preprocess_variants,
        clahe_clip_limit=_optional_float(config, "clahe_clip_limit", 2.0),
        clahe_tile_grid_size=_optional_int(config, "clahe_tile_grid_size", 8),
        clahe_tile_px=_optional_int(config, "clahe_tile_px", 128),
        morph_kernel_px=_optional_int(config, "morph_kernel_px", 21),
        stretch_low_pct=_optional_float(config, "stretch_low_pct", 1.0),
        stretch_high_pct=_optional_float(config, "stretch_high_pct", 99.0),
        preprocess_gamma=_optional_float(config, "preprocess_gamma", 0.5),
        unsharp_sigma=_optional_float(config, "unsharp_sigma", 2.0),
        unsharp_amount=_optional_float(config, "unsharp_amount", 1.5),
        ocr_dual_pass_invert=_optional_bool(config, "ocr_dual_pass_invert", False),
        ocr_upscale_factor=_optional_float(config, "ocr_upscale_factor", 1.0),
        ocr_upscale_interpolation=_optional_choice(
            config, "ocr_upscale_interpolation", "cubic", set(UPSCALE_INTERPOLATIONS)
        ),
        resolution_strategy=_optional_choice(
            config, "resolution_strategy", "full", RESOLUTION_STRATEGIES
        ),
        tile_size_px=_optional_int(config, "tile_size_px", 1600),
        tile_overlap_px=_optional_int(config, "tile_overlap_px", 200),
        tile_dedupe_iou=_optional_float(config, "tile_dedupe_iou", 0.5),
        text_filter=_optional_choice(config, "text_filter", "none", TEXT_FILTERS),
        short_text_max_chars=_optional_int(config, "short_text_max_chars", 2),
        redaction_mode=_optional_choice(
            config, "redaction_mode", "fill", REDACTION_MODES
        ),
        record_raw_detections=_optional_bool(config, "record_raw_detections", False),
    )


def build_artifacts(config: PipelineConfig) -> StageArtifacts:
    base = config.artifacts_dir
    deid_images_dir = (
        config.save_deidentified_dir
        if config.save_deidentified_dir is not None
        else base / "deidentification" / "images"
    )
    return StageArtifacts(
        sam3_masks_dir=base / "sam3" / "masks",
        sam3_records_json=base / "sam3" / "records.json",
        postprocess_crops_dir=base / "postprocess" / "crops",
        postprocess_records_json=base / "postprocess" / "records.json",
        deid_images_dir=deid_images_dir,
        deid_records_json=base / "deidentification" / "records.json",
    )


def build_segmenter(logger: PipelineEventLogger) -> Sam3ImageSegmenter:
    logger.loading_model("SAM3")
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model()
    logger.model_loaded("SAM3")
    return Sam3ImageSegmenter(model=model)


def build_postprocessor(config: PipelineConfig) -> MaskPostprocessor:
    return MaskPostprocessor(
        kernel_size=config.kernel_size,
        iterations=config.iterations,
        large_bb_area_ratio=config.large_bb_area_ratio,
    )


def build_ocr_engine(logger: PipelineEventLogger, config: PipelineConfig) -> OcrEngine:
    if config.ocr_engine == "easyocr":
        logger.loading_model("EasyOCR")
        engine: OcrEngine = build_easyocr_engine(
            config.easyocr_langs,
            config.easyocr_gpu,
            EasyOcrParams(
                text_threshold=config.easyocr_text_threshold,
                low_text=config.easyocr_low_text,
                link_threshold=config.easyocr_link_threshold,
                canvas_size=config.easyocr_canvas_size,
                mag_ratio=config.easyocr_mag_ratio,
            ),
        )
        logger.model_loaded("EasyOCR")
        return engine

    logger.loading_model("PaddleOCR")
    engine = build_paddle_engine(
        device=config.paddleocr_device,
        det_model=config.paddleocr_det_model,
        rec_model=config.paddleocr_rec_model,
    )
    logger.model_loaded("PaddleOCR")
    return engine


def build_preprocess_params(config: PipelineConfig) -> PreprocessParams:
    return PreprocessParams(
        clahe_clip_limit=config.clahe_clip_limit,
        clahe_tile_grid_size=config.clahe_tile_grid_size,
        clahe_tile_px=config.clahe_tile_px,
        morph_kernel_px=config.morph_kernel_px,
        stretch_low_pct=config.stretch_low_pct,
        stretch_high_pct=config.stretch_high_pct,
        gamma=config.preprocess_gamma,
        unsharp_sigma=config.unsharp_sigma,
        unsharp_amount=config.unsharp_amount,
    )


def build_deidentifier(
    logger: PipelineEventLogger,
    config: PipelineConfig,
    engine: OcrEngine | None = None,
) -> Deidentifier:
    if engine is None:
        engine = build_ocr_engine(logger, config)
    params = DeidentifierParams(
        merge_distance_px=config.merge_distance_px,
        max_box_area_px=config.max_box_area_px,
        max_box_area_ratio=config.max_box_area_ratio,
        ellipse_enabled=config.ellipse_enabled,
        center_ellipse_axes_ratio=(
            config.ellipse_axis_x_ratio,
            config.ellipse_axis_y_ratio,
        ),
        ellipse_proximity_px=config.ellipse_proximity_px,
        padding_px=config.deid_padding_px,
        min_confidence=config.ocr_min_confidence,
        redaction_mode=config.redaction_mode,
    )
    strategy: ResolutionStrategy = FullImageStrategy()
    if config.resolution_strategy == "tiled":
        strategy = TiledStrategy(
            tile_size_px=config.tile_size_px,
            tile_overlap_px=config.tile_overlap_px,
            dedupe_containment=config.tile_dedupe_iou,
        )
    if config.ocr_upscale_factor > 1.0:
        # Wraps whatever strategy we just built, so upscale composes with
        # tiling and the enlarged array never escapes into the Deidentifier.
        strategy = UpscaleStrategy(
            strategy,
            factor=config.ocr_upscale_factor,
            interpolation=config.ocr_upscale_interpolation,
        )
    classifier = build_classifier(config.text_filter, config.short_text_max_chars)

    preprocess_params = build_preprocess_params(config)
    preprocess = None
    if config.preprocess_steps:
        preprocess = partial(
            apply_chain, steps=list(config.preprocess_steps), params=preprocess_params
        )
    preprocess_variants = None
    if config.preprocess_variants:
        preprocess_variants = [
            partial(apply_chain, steps=list(steps), params=preprocess_params)
            for steps in config.preprocess_variants
        ]
    return Deidentifier(
        engine=engine,
        params=params,
        strategy=strategy,
        classifier=classifier,
        preprocess=preprocess,
        preprocess_variants=preprocess_variants,
        dual_pass_invert=config.ocr_dual_pass_invert,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)


def _load_records(path: Path, required: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            raise ValueError(f"Missing records file: {path}")
        return []
    with path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if not isinstance(payload, list):
        raise ValueError(f"Records file must contain a list: {path}")
    records: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"Record must be a dict in: {path}")
        records.append(item)
    return records


def _image_relpath_png(image_name: str) -> Path:
    return Path(image_name).with_suffix(".png")


def _resolve_input_image_path(input_path: Path, image_name: str) -> Path:
    if input_path.is_file():
        return input_path
    return input_path / image_name


def _load_image_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.ndim == 2:
        return image_rgb
    if image_rgb.shape[2] == 3:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if image_rgb.shape[2] == 4:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2BGRA)
    raise ValueError(f"Unsupported image shape for save: {image_rgb.shape}")


def _save_image(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), _to_bgr(image_rgb))
    if not ok:
        raise ValueError(f"Failed to save image: {path}")


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    ok = cv2.imwrite(str(path), mask_u8)
    if not ok:
        raise ValueError(f"Failed to save mask: {path}")


def _combine_segmentation_mask(
    image_shape: tuple[int, int], segmentation: SegmentationResult
) -> np.ndarray:
    height, width = image_shape
    if segmentation.masks.size > 0:
        return np.where(np.max(segmentation.masks, axis=0) > 0, 255, 0).astype(np.uint8)

    fallback = np.zeros((height, width), dtype=np.uint8)
    for box in segmentation.sam_boxes:
        x1 = int(max(0, min(width, box.x)))
        y1 = int(max(0, min(height, box.y)))
        x2 = int(max(0, min(width, box.x + box.w)))
        y2 = int(max(0, min(height, box.y + box.h)))
        if x2 > x1 and y2 > y1:
            cv2.rectangle(fallback, (x1, y1), (x2, y2), 255, thickness=-1)
    return fallback


def _deserialize_boxes(raw_boxes: Any) -> list[BoundingBox]:
    if not isinstance(raw_boxes, list):
        return []
    boxes: list[BoundingBox] = []
    for item in raw_boxes:
        if (
            isinstance(item, list)
            and len(item) == 4
            and all(isinstance(value, (int, float)) for value in item)
        ):
            x, y, w, h = [int(round(float(value))) for value in item]
            if w > 0 and h > 0:
                boxes.append(BoundingBox(x=x, y=y, w=w, h=h))
    boxes.sort(key=lambda box: box.area, reverse=True)
    return boxes


def run_sam3_stage(
    config: PipelineConfig,
    artifacts: StageArtifacts,
    logger: PipelineEventLogger,
    segmenter: Sam3ImageSegmenter,
    *,
    save_artifacts: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    logger.stage_started("sam3")
    source = LocalImageSource(config.input_path)
    records: list[dict[str, Any]] = []
    masks_in_memory: dict[str, np.ndarray] = {}
    for loaded_image in source.iter_images():
        logger.processing_image(loaded_image.name, stage="sam3")
        start = time.perf_counter()
        segmentation = segmenter.infer(loaded_image.pil_image, config.prompt)
        if not segmentation.sam_boxes and config.fallback_prompt:
            logger.fallback_prompt_attempted(loaded_image.name, config.fallback_prompt)
            segmentation = segmenter.infer(loaded_image.pil_image, config.fallback_prompt)
        inference_time = time.perf_counter() - start

        mask = _combine_segmentation_mask(
            image_shape=(loaded_image.rgb_image.shape[0], loaded_image.rgb_image.shape[1]),
            segmentation=segmentation,
        )
        mask_relpath = _image_relpath_png(loaded_image.name)
        if save_artifacts:
            _save_mask(artifacts.sam3_masks_dir / mask_relpath, mask)
            record: dict[str, Any] = {
                "name": loaded_image.name,
                "original_size": [int(loaded_image.size[0]), int(loaded_image.size[1])],
                "sam3_inference_time": float(inference_time),
                "sam_boxes": [box.as_list() for box in segmentation.sam_boxes],
                "mask_relpath": mask_relpath.as_posix(),
            }
        else:
            masks_in_memory[loaded_image.name] = mask
            record = {
                "name": loaded_image.name,
                "original_size": [int(loaded_image.size[0]), int(loaded_image.size[1])],
                "sam3_inference_time": float(inference_time),
                "sam_boxes": [box.as_list() for box in segmentation.sam_boxes],
            }
        records.append(record)
    _write_json(artifacts.sam3_records_json, records)
    logger.stage_completed("sam3", len(records))
    return records, masks_in_memory


def run_postprocess_stage(
    config: PipelineConfig,
    artifacts: StageArtifacts,
    logger: PipelineEventLogger,
    postprocessor: MaskPostprocessor,
    sam3_records: list[dict[str, Any]] | None = None,
    *,
    in_memory_masks: dict[str, np.ndarray] | None = None,
    save_artifacts: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    logger.stage_started("postprocess")
    records_in = (
        sam3_records
        if sam3_records is not None
        else _load_records(artifacts.sam3_records_json, required=True)
    )
    out_records: list[dict[str, Any]] = []
    crops_in_memory: dict[str, np.ndarray] = {}
    for record in records_in:
        name = _require_str(record, "name")
        logger.processing_image(name, stage="postprocess")
        image_path = _resolve_input_image_path(config.input_path, name)
        image_rgb = _load_image_rgb(image_path)

        if in_memory_masks is not None and name in in_memory_masks:
            mask = in_memory_masks[name]
        else:
            mask_relpath_raw = _require_str(record, "mask_relpath")
            mask_path = artifacts.sam3_masks_dir / mask_relpath_raw
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not read SAM3 mask: {mask_path}")

        segmentation = SegmentationResult(
            masks=np.where(mask > 0, 255, 0).astype(np.uint8)[np.newaxis, :, :],
            sam_boxes=_deserialize_boxes(record.get("sam_boxes", [])),
        )

        start = time.perf_counter()
        transformed = postprocessor.transform(image_rgb, segmentation)
        stage_time = time.perf_counter() - start

        crop_relpath = _image_relpath_png(name)
        if save_artifacts:
            _save_image(artifacts.postprocess_crops_dir / crop_relpath, transformed.cut_image)
            out_record: dict[str, Any] = {
                "name": name,
                "original_size": [int(image_rgb.shape[1]), int(image_rgb.shape[0])],
                "sam3_inference_time": float(record.get("sam3_inference_time", 0.0)),
                "erosion_diffusion_time": float(stage_time),
                "alarm": {
                    "triggered": bool(transformed.alarm.triggered),
                    "motivation": transformed.alarm.motivation,
                },
                "bounding_boxes": [box.as_list() for box in transformed.bounding_boxes],
                "cut_size": [int(transformed.cut_size[0]), int(transformed.cut_size[1])],
                "rotation_angle": float(transformed.rotation_angle),
                "crop_relpath": crop_relpath.as_posix(),
            }
        else:
            crops_in_memory[name] = transformed.cut_image
            out_record = {
                "name": name,
                "original_size": [int(image_rgb.shape[1]), int(image_rgb.shape[0])],
                "sam3_inference_time": float(record.get("sam3_inference_time", 0.0)),
                "erosion_diffusion_time": float(stage_time),
                "alarm": {
                    "triggered": bool(transformed.alarm.triggered),
                    "motivation": transformed.alarm.motivation,
                },
                "bounding_boxes": [box.as_list() for box in transformed.bounding_boxes],
                "cut_size": [int(transformed.cut_size[0]), int(transformed.cut_size[1])],
                "rotation_angle": float(transformed.rotation_angle),
            }
        out_records.append(out_record)

        if transformed.alarm.triggered:
            logger.alarm_triggered(name, transformed.alarm.motivation)

    _write_json(artifacts.postprocess_records_json, out_records)
    logger.stage_completed("postprocess", len(out_records))
    return out_records, crops_in_memory


def run_deidentification_stage(
    artifacts: StageArtifacts,
    logger: PipelineEventLogger,
    deidentifier: Deidentifier,
    postprocess_records: list[dict[str, Any]] | None = None,
    *,
    in_memory_crops: dict[str, np.ndarray] | None = None,
    save_artifacts: bool = True,
    record_raw_detections: bool = False,
) -> list[dict[str, Any]]:
    logger.stage_started("deidentification")
    records_in = (
        postprocess_records
        if postprocess_records is not None
        else _load_records(artifacts.postprocess_records_json, required=True)
    )
    out_records: list[dict[str, Any]] = []
    for record in records_in:
        name = _require_str(record, "name")
        logger.processing_image(name, stage="deidentification")

        if in_memory_crops is not None and name in in_memory_crops:
            crop_rgb = in_memory_crops[name]
        else:
            crop_relpath = _require_str(record, "crop_relpath")
            crop_path = artifacts.postprocess_crops_dir / crop_relpath
            crop_rgb = _load_image_rgb(crop_path)

        start = time.perf_counter()
        result = deidentifier.run(crop_rgb, name)
        stage_time = time.perf_counter() - start

        if save_artifacts:
            deid_relpath = _image_relpath_png(name)
            _save_image(artifacts.deid_images_dir / deid_relpath, result.image)

        out_record: dict[str, Any] = {
            "name": name,
            "deidentification_time": float(stage_time),
            "boxes": result.boxes,
            "detections": result.detections,
        }
        if record_raw_detections:
            out_record["raw_detections"] = result.raw_detections
            out_record["skipped_detections"] = result.skipped_detections
        out_records.append(out_record)

    _write_json(artifacts.deid_records_json, out_records)
    logger.stage_completed("deidentification", len(out_records))
    return out_records


def run_report_stage(
    config: PipelineConfig,
    artifacts: StageArtifacts,
    logger: PipelineEventLogger,
    sam3_records: list[dict[str, Any]] | None = None,
    postprocess_records: list[dict[str, Any]] | None = None,
    deid_records: list[dict[str, Any]] | None = None,
) -> list[ImageEntry]:
    sam3 = (
        sam3_records
        if sam3_records is not None
        else _load_records(artifacts.sam3_records_json, required=False)
    )
    post = (
        postprocess_records
        if postprocess_records is not None
        else _load_records(artifacts.postprocess_records_json, required=True)
    )
    deid = (
        deid_records
        if deid_records is not None
        else _load_records(artifacts.deid_records_json, required=False)
    )

    sam3_by_name = {
        _require_str(record, "name"): record for record in sam3 if isinstance(record, dict)
    }
    deid_by_name = {
        _require_str(record, "name"): record for record in deid if isinstance(record, dict)
    }

    entries: list[ImageEntry] = []
    for record in post:
        name = _require_str(record, "name")
        alarm_data = record.get("alarm")
        if not isinstance(alarm_data, dict):
            alarm_data = {"triggered": False, "motivation": ""}

        original_size_raw = record.get("original_size")
        cut_size_raw = record.get("cut_size")
        if not (
            isinstance(original_size_raw, list)
            and len(original_size_raw) == 2
            and isinstance(cut_size_raw, list)
            and len(cut_size_raw) == 2
        ):
            raise ValueError(f"Invalid size metadata in postprocess record for {name}")

        sam_time = float(
            record.get(
                "sam3_inference_time",
                sam3_by_name.get(name, {}).get("sam3_inference_time", 0.0),
            )
        )
        erosion_time = float(record.get("erosion_diffusion_time", 0.0))
        deid_record = deid_by_name.get(name, {})
        deid_time = float(deid_record.get("deidentification_time", 0.0))

        entry = ImageEntry(
            name=name,
            alarm=AlarmInfo(
                triggered=bool(alarm_data.get("triggered", False)),
                motivation=str(alarm_data.get("motivation", "")),
            ),
            bounding_boxes=_deserialize_boxes(record.get("bounding_boxes", [])),
            deidentification_boxes=_deserialize_boxes(
                deid_record.get("boxes", deid_record.get("deidentification_boxes", []))
            ),
            metrics=ImageMetrics(
                times=StageTimes(
                    sam3_inference=sam_time,
                    erosion_diffusion=erosion_time,
                    deidentification=deid_time,
                ),
                original_size=(
                    int(float(original_size_raw[0])),
                    int(float(original_size_raw[1])),
                ),
                cut_size=(int(float(cut_size_raw[0])), int(float(cut_size_raw[1]))),
                rotation_angle=float(record.get("rotation_angle", 0.0)),
            ),
        )
        entries.append(entry)

    logger.stage_started("report")
    writer = JsonReportWriter()
    writer.write(config.output_json, entries)
    logger.report_written(str(config.output_json), len(entries))
    return entries


def run(config: PipelineConfig, logger: PipelineEventLogger | None = None) -> int:
    """Run the pipeline for an in-memory config (no argparse / TOML reload).

    Reusable entrypoint for embedding the pipeline (e.g. the ToothFairy4M runner
    adapter). `main()` is the CLI wrapper around this.
    """
    if logger is None:
        logger = PipelineEventLogger(logging.getLogger("pipeline"))
    artifacts = build_artifacts(config)

    logger.pipeline_started(config.run_mode)

    if config.run_mode == "sam3":
        segmenter = build_segmenter(logger)
        run_sam3_stage(config, artifacts, logger, segmenter)
        return 0

    if config.run_mode == "postprocess":
        postprocessor = build_postprocessor(config)
        run_postprocess_stage(config, artifacts, logger, postprocessor)
        return 0

    if config.run_mode == "deidentification":
        deidentifier = build_deidentifier(logger, config)
        run_deidentification_stage(
            artifacts,
            logger,
            deidentifier,
            record_raw_detections=config.record_raw_detections,
        )
        return 0

    if config.run_mode == "report":
        run_report_stage(config, artifacts, logger)
        return 0

    save_arts = config.save_artifacts
    segmenter = build_segmenter(logger)
    postprocessor = build_postprocessor(config)
    deidentifier = build_deidentifier(logger, config)
    sam3_records, in_mem_masks = run_sam3_stage(
        config, artifacts, logger, segmenter, save_artifacts=save_arts
    )
    post_records, in_mem_crops = run_postprocess_stage(
        config,
        artifacts,
        logger,
        postprocessor,
        sam3_records=sam3_records,
        in_memory_masks=in_mem_masks,
        save_artifacts=save_arts,
    )
    deid_records = run_deidentification_stage(
        artifacts,
        logger,
        deidentifier,
        postprocess_records=post_records,
        in_memory_crops=in_mem_crops,
        save_artifacts=save_arts,
        record_raw_detections=config.record_raw_detections,
    )
    run_report_stage(
        config,
        artifacts,
        logger,
        sam3_records=sam3_records,
        postprocess_records=post_records,
        deid_records=deid_records,
    )
    return 0


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    config = build_config()
    logger = PipelineEventLogger(logging.getLogger("pipeline"))
    return run(config, logger)


if __name__ == "__main__":
    raise SystemExit(main())
