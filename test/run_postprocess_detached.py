#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.mask_component import MaskPostprocessor
from pipeline.models import BoundingBox, SegmentationResult

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone postprocessing runner for SAM3-like folders containing "
            "images, *_mask.png and *_labels.json files."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root folder that contains SAM3 outputs (recursive).",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("pipeline_artifacts"),
        help="Artifacts root where postprocess outputs will be written.",
    )
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--large-bb-area-ratio", type=float, default=0.1)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def iter_masks(input_root: Path) -> list[Path]:
    masks = [path for path in input_root.rglob("*_mask.png") if path.is_file()]
    masks.sort(key=lambda path: path.relative_to(input_root).as_posix())
    return masks


def parse_label_boxes(mask_path: Path) -> list[BoundingBox]:
    base = mask_path.stem.removesuffix("_mask")
    labels_path = mask_path.with_name(f"{base}_labels.json")
    if not labels_path.exists():
        return []

    with labels_path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    raw_boxes = payload.get("boxes", []) if isinstance(payload, dict) else []
    boxes: list[BoundingBox] = []
    for item in raw_boxes:
        if (
            isinstance(item, list)
            and len(item) == 4
            and all(isinstance(value, (int, float)) for value in item)
        ):
            x1, y1, x2, y2 = [int(round(float(value))) for value in item]
            left, top = min(x1, x2), min(y1, y2)
            right, bottom = max(x1, x2), max(y1, y2)
            width, height = right - left, bottom - top
            if width > 0 and height > 0:
                boxes.append(BoundingBox(x=left, y=top, w=width, h=height))
    boxes.sort(key=lambda box: box.area, reverse=True)
    return boxes


def _candidate_image_paths(mask_path: Path, input_root: Path) -> list[Path]:
    base = mask_path.stem.removesuffix("_mask")
    rel = mask_path.relative_to(input_root)
    candidates: list[Path] = []

    for ext in IMAGE_EXTENSIONS:
        candidates.append(mask_path.with_name(f"{base}{ext}"))

    labels_path = mask_path.with_name(f"{base}_labels.json")
    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        if isinstance(payload, dict):
            image_file = payload.get("image_file")
            if isinstance(image_file, str) and image_file.strip():
                image_name = image_file.strip()
                candidates.append(mask_path.parent / image_name)
                candidates.append(input_root / image_name)

    parts = list(rel.parts)
    if "masks" in parts:
        masks_idx = parts.index("masks")
        tail = Path(*parts[masks_idx + 1 :])
        tail_base = tail.with_name(base)
        for ext in IMAGE_EXTENSIONS:
            candidates.append(input_root / "imgs" / tail_base.with_suffix(ext))

    for ext in IMAGE_EXTENSIONS:
        candidates.append(input_root / rel.with_name(base).with_suffix(ext))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def resolve_image_path(mask_path: Path, input_root: Path) -> Path:
    for candidate in _candidate_image_paths(mask_path, input_root):
        if candidate.exists() and candidate.is_file():
            return candidate
    raise ValueError(f"Could not find source image for mask: {mask_path}")


def to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.ndim == 2:
        return image_rgb
    if image_rgb.shape[2] == 3:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if image_rgb.shape[2] == 4:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2BGRA)
    raise ValueError(f"Unsupported image shape: {image_rgb.shape}")


def save_image(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), to_bgr(image_rgb))
    if not ok:
        raise ValueError(f"Failed to save image: {path}")


def relative_name(mask_path: Path, input_root: Path, source_image_path: Path) -> str:
    mask_rel = mask_path.relative_to(input_root)
    base = mask_rel.with_name(mask_rel.stem.removesuffix("_mask"))
    return base.with_suffix(source_image_path.suffix.lower()).as_posix()


def run() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    input_root = args.input_root.resolve()
    artifacts_root = args.artifacts_root.resolve()
    if not input_root.exists() or not input_root.is_dir():
        raise ValueError(f"Invalid input root: {input_root}")

    masks = iter_masks(input_root)
    if not masks:
        raise ValueError(f"No *_mask.png files found under: {input_root}")

    postprocessor = MaskPostprocessor(
        kernel_size=args.kernel_size,
        iterations=args.iterations,
        large_bb_area_ratio=args.large_bb_area_ratio,
    )

    crops_root = artifacts_root / "postprocess" / "crops"
    records_path = artifacts_root / "postprocess" / "records.json"
    failures_path = artifacts_root / "postprocess" / "failures.json"

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for mask_path in masks:
        try:
            image_path = resolve_image_path(mask_path, input_root)
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError(f"Could not read image: {image_path}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not read mask: {mask_path}")
            if mask.shape[:2] != image_rgb.shape[:2]:
                raise ValueError(
                    f"Mask/image size mismatch for {mask_path}: "
                    f"mask={mask.shape[:2]}, image={image_rgb.shape[:2]}"
                )

            sam_boxes = parse_label_boxes(mask_path)
            segmentation = SegmentationResult(
                masks=np.where(mask > 0, 255, 0).astype(np.uint8)[np.newaxis, :, :],
                sam_boxes=sam_boxes,
            )

            start = time.perf_counter()
            transformed = postprocessor.transform(image_rgb=image_rgb, segmentation=segmentation)
            elapsed = time.perf_counter() - start

            name = relative_name(mask_path, input_root, image_path)
            crop_relpath = Path(name).with_suffix(".png")
            save_image(crops_root / crop_relpath, transformed.cut_image)

            record = {
                "name": name,
                "original_size": [int(image_rgb.shape[1]), int(image_rgb.shape[0])],
                "sam3_inference_time": 0.0,
                "erosion_diffusion_time": float(elapsed),
                "alarm": {
                    "triggered": bool(transformed.alarm.triggered),
                    "motivation": transformed.alarm.motivation,
                },
                "bounding_boxes": [box.as_list() for box in transformed.bounding_boxes],
                "cut_size": [int(transformed.cut_size[0]), int(transformed.cut_size[1])],
                "rotation_angle": float(transformed.rotation_angle),
                "crop_relpath": crop_relpath.as_posix(),
                "mask_relpath": mask_path.relative_to(input_root).as_posix(),
                "source_image_relpath": image_path.relative_to(input_root).as_posix()
                if image_path.is_relative_to(input_root)
                else str(image_path),
            }
            records.append(record)
            LOGGER.info("Postprocessed: %s", name)
        except Exception as exc:  # noqa: BLE001 - per-file failure isolation for testing
            failures.append(
                {
                    "mask": str(mask_path),
                    "error": str(exc),
                }
            )
            LOGGER.error("Failed on %s: %s", mask_path, exc)

    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8") as file_handle:
        json.dump(records, file_handle, indent=2)
    with failures_path.open("w", encoding="utf-8") as file_handle:
        json.dump(failures, file_handle, indent=2)

    LOGGER.info(
        "Done. processed=%d failed=%d records=%s",
        len(records),
        len(failures),
        records_path,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run())
