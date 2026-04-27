from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

try:
    from .models import LoadedImage
except ImportError:
    from models import LoadedImage


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class LocalImageSource:
    def __init__(self, input_path: Path) -> None:
        self._input_path = input_path

    def iter_images(self) -> Iterable[LoadedImage]:
        files, base_dir = self._resolve_files()
        for image_path in files:
            pil_image = Image.open(image_path).convert("RGB")
            rgb = np.asarray(pil_image).copy()
            name = (
                image_path.relative_to(base_dir).as_posix()
                if base_dir is not None
                else image_path.name
            )
            yield LoadedImage(
                name=name,
                path=image_path,
                pil_image=pil_image,
                rgb_image=rgb,
                size=(pil_image.width, pil_image.height),
            )

    def _resolve_files(self) -> tuple[list[Path], Path | None]:
        if self._input_path.is_file():
            if self._input_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported image format: {self._input_path}")
            return [self._input_path], None

        if not self._input_path.is_dir():
            raise ValueError(f"Input path does not exist: {self._input_path}")

        files: list[Path] = []
        for path in sorted(self._input_path.rglob("*")):
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS:
                files.append(path)

        if not files:
            raise ValueError(f"No supported images found in: {self._input_path}")
        return files, self._input_path
