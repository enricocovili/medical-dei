from __future__ import annotations

import json
from pathlib import Path

try:
    from .models import ImageEntry
except ImportError:
    from models import ImageEntry


class JsonReportWriter:
    def write(self, output_path: Path, entries: list[ImageEntry]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [entry.to_dict() for entry in entries]
        with output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2)
