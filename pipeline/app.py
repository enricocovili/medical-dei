from __future__ import annotations

import argparse
import json
import logging
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .deidentifier_component import DeidentifierParams, EasyOcrDeidentifier
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
    from .report_writer import JsonReportWriter
    from .sam3_component import Sam3ImageSegmenter
except ImportError:
    from deidentifier_component import DeidentifierParams, EasyOcrDeidentifier
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
    from report_writer import JsonReportWriter
    from sam3_component import Sam3ImageSegmenter


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
    ellipse_axis_x_ratio: float
    ellipse_axis_y_ratio: float
    ellipse_proximity_px: float
    deid_padding_px: int


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
    config = _load_pipeline_section()

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
        ellipse_axis_x_ratio=_require_float(config, "ellipse_axis_x_ratio"),
        ellipse_axis_y_ratio=_require_float(config, "ellipse_axis_y_ratio"),
        ellipse_proximity_px=_require_float(config, "ellipse_proximity_px"),
        deid_padding_px=_require_int(config, "deid_padding_px"),
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


def build_deidentifier(
    logger: PipelineEventLogger, config: PipelineConfig
) -> EasyOcrDeidentifier:
    logger.loading_model("EasyOCR")
    import easyocr

    reader = easyocr.Reader(config.easyocr_langs, gpu=config.easyocr_gpu)
    logger.model_loaded("EasyOCR")
    params = DeidentifierParams(
        merge_distance_px=config.merge_distance_px,
        max_box_area_px=config.max_box_area_px,
        center_ellipse_axes_ratio=(
            config.ellipse_axis_x_ratio,
            config.ellipse_axis_y_ratio,
        ),
        ellipse_proximity_px=config.ellipse_proximity_px,
        padding_px=config.deid_padding_px,
    )
    return EasyOcrDeidentifier(reader=reader, params=params)


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
    deidentifier: EasyOcrDeidentifier,
    postprocess_records: list[dict[str, Any]] | None = None,
    *,
    in_memory_crops: dict[str, np.ndarray] | None = None,
    save_artifacts: bool = True,
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
        deidentified, deidentification_boxes = deidentifier.deidentify_with_boxes(
            crop_rgb, name
        )
        stage_time = time.perf_counter() - start

        if save_artifacts:
            deid_relpath = _image_relpath_png(name)
            _save_image(artifacts.deid_images_dir / deid_relpath, deidentified)

        out_records.append(
            {
                "name": name,
                "deidentification_time": float(stage_time),
                "boxes": deidentification_boxes,
            }
        )

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
                deid_record.get("deidentification_boxes", [])
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
        run_deidentification_stage(artifacts, logger, deidentifier)
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
