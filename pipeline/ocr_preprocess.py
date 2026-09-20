from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PREPROCESS_STEPS = (
    "grayscale",
    "clahe",
    "invert",
    "tophat",
    "blackhat",
    "percentile_stretch",
    "gamma",
    "unsharp",
)


@dataclass(frozen=True, slots=True)
class PreprocessParams:
    """Tunables for apply_chain.

    Every step here must be coordinate-preserving: Deidentifier.run feeds the
    chain's output straight to the engine AND uses its .shape for the
    centre-ellipse test, so a resize would silently corrupt every box. Scaling
    lives in ocr_strategies.UpscaleStrategy instead.
    """

    clahe_clip_limit: float = 2.0
    # Fixed NxN grid, used only when clahe_tile_px <= 0.
    clahe_tile_grid_size: int = 8
    # Preferred: target tile edge in pixels, from which the grid is derived per
    # image. The dataset spans 512x248 to 6343x2713, so a fixed 8x8 grid means
    # 64px tiles on one image and 793px tiles on another — the latter is
    # effectively global equalisation against a 14px-tall median text box.
    clahe_tile_px: int = 128
    # Morphological top-hat/black-hat kernel. Should exceed the text stroke
    # thickness so the opening erases the text and the residual keeps it.
    morph_kernel_px: int = 21
    # Percentile contrast stretch bounds.
    stretch_low_pct: float = 1.0
    stretch_high_pct: float = 99.0
    gamma: float = 0.5
    unsharp_sigma: float = 2.0
    unsharp_amount: float = 1.5


def _clahe_grid(image: np.ndarray, params: PreprocessParams) -> tuple[int, int]:
    if params.clahe_tile_px <= 0:
        size = max(1, params.clahe_tile_grid_size)
        return size, size
    height, width = image.shape[:2]
    return (
        max(2, round(width / params.clahe_tile_px)),
        max(2, round(height / params.clahe_tile_px)),
    )


def _apply_clahe(image: np.ndarray, params: PreprocessParams) -> np.ndarray:
    grid_x, grid_y = _clahe_grid(image, params)
    clahe = cv2.createCLAHE(clipLimit=params.clahe_clip_limit, tileGridSize=(grid_x, grid_y))
    if image.ndim == 2:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _morph_residual(image: np.ndarray, params: PreprocessParams, op: int) -> np.ndarray:
    """Top-hat / black-hat: isolate thin bright (or dark) structures.

    Burned-in text is an additive overlay of near-uniform thin strokes on a
    slowly-varying anatomical background, which is exactly what this separates —
    and unlike CLAHE it does not amplify anatomy noise along with the text.
    """
    size = max(3, params.morph_kernel_px | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.morphologyEx(image, op, kernel)


def _percentile_stretch(image: np.ndarray, params: PreprocessParams) -> np.ndarray:
    """Linear contrast stretch between two intensity percentiles.

    X-ray margins, where most burned-in text sits, often occupy a narrow slice
    of the 0-255 range; stretching it is a cheap way to make faint text legible.
    """
    low_pct = min(params.stretch_low_pct, params.stretch_high_pct)
    high_pct = max(params.stretch_low_pct, params.stretch_high_pct)
    low, high = np.percentile(image, [low_pct, high_pct])
    if high <= low:
        return image
    scaled = (image.astype(np.float32) - low) * (255.0 / (high - low))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _gamma(image: np.ndarray, params: PreprocessParams) -> np.ndarray:
    if params.gamma <= 0:
        raise ValueError("gamma must be > 0")
    table = np.array(
        [((i / 255.0) ** params.gamma) * 255.0 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(image, table)


def _unsharp(image: np.ndarray, params: PreprocessParams) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), params.unsharp_sigma)
    return cv2.addWeighted(image, 1.0 + params.unsharp_amount, blurred, -params.unsharp_amount, 0)


def apply_chain(
    image: np.ndarray,
    steps: list[str],
    *,
    params: PreprocessParams | None = None,
    clahe_clip_limit: float | None = None,
    clahe_tile_grid_size: int | None = None,
) -> np.ndarray:
    """Apply an ordered list of coordinate-preserving preprocessing steps.

    clahe_clip_limit / clahe_tile_grid_size are kept as keyword overrides for
    backward compatibility with callers that predate PreprocessParams.
    """
    effective = params or PreprocessParams()
    if clahe_clip_limit is not None or clahe_tile_grid_size is not None:
        effective = replace_params(
            effective,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid_size=clahe_tile_grid_size,
        )

    out = image
    for step in steps:
        if step == "grayscale":
            if out.ndim == 3:
                out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        elif step == "clahe":
            out = _apply_clahe(out, effective)
        elif step == "invert":
            out = cv2.bitwise_not(out)
        elif step == "tophat":
            out = _morph_residual(out, effective, cv2.MORPH_TOPHAT)
        elif step == "blackhat":
            out = _morph_residual(out, effective, cv2.MORPH_BLACKHAT)
        elif step == "percentile_stretch":
            out = _percentile_stretch(out, effective)
        elif step == "gamma":
            out = _gamma(out, effective)
        elif step == "unsharp":
            out = _unsharp(out, effective)
        else:
            raise ValueError(
                f"Invalid preprocess step '{step}'. Valid steps: {PREPROCESS_STEPS}"
            )
    return out


def replace_params(
    params: PreprocessParams,
    *,
    clahe_clip_limit: float | None = None,
    clahe_tile_grid_size: int | None = None,
) -> PreprocessParams:
    import dataclasses

    changes: dict[str, object] = {}
    if clahe_clip_limit is not None:
        changes["clahe_clip_limit"] = clahe_clip_limit
    if clahe_tile_grid_size is not None:
        # An explicit fixed grid disables the resolution-relative tiling.
        changes["clahe_tile_grid_size"] = clahe_tile_grid_size
        changes["clahe_tile_px"] = 0
    return dataclasses.replace(params, **changes) if changes else params


def validate_steps(steps: list[str]) -> None:
    for step in steps:
        if step not in PREPROCESS_STEPS:
            raise ValueError(
                f"Invalid preprocess step '{step}'. Valid steps: {PREPROCESS_STEPS}"
            )
