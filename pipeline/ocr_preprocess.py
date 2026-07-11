from __future__ import annotations

import cv2
import numpy as np

PREPROCESS_STEPS = ("grayscale", "clahe", "invert")


def apply_chain(
    image: np.ndarray,
    steps: list[str],
    *,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: int = 8,
) -> np.ndarray:
    out = image
    for step in steps:
        if step == "grayscale":
            if out.ndim == 3:
                out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        elif step == "clahe":
            clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit,
                tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size),
            )
            if out.ndim == 2:
                out = clahe.apply(out)
            else:
                lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        elif step == "invert":
            out = cv2.bitwise_not(out)
        else:
            raise ValueError(
                f"Invalid preprocess step '{step}'. Valid steps: {PREPROCESS_STEPS}"
            )
    return out


def validate_steps(steps: list[str]) -> None:
    for step in steps:
        if step not in PREPROCESS_STEPS:
            raise ValueError(
                f"Invalid preprocess step '{step}'. Valid steps: {PREPROCESS_STEPS}"
            )
