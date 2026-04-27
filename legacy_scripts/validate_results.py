#!/usr/bin/env python3
import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Paths:
    scans_dir: Path
    processed_dir: Path
    results_csv: Path


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_EXTS


def _iter_images_recursive(root: Path):
    # Collect then sort by relative path to ensure a true alphabetical order
    # across the whole tree (os.walk order is platform/filesystem dependent).
    files: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        dirpath_p = Path(dirpath)
        for name in filenames:
            p = dirpath_p / name
            if p.is_file() and _is_image_file(p):
                files.append(p)

    files.sort(key=lambda p: p.relative_to(root).as_posix())
    yield from files


def _load_existing_results(results_csv: Path) -> dict[str, str]:
    """Returns relpath -> status for the most recent decision per relpath."""
    if not results_csv.exists():
        return {}

    decisions: dict[str, str] = {}
    try:
        with results_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel = (row.get("relpath") or "").strip()
                status = (row.get("status") or "").strip()
                if rel:
                    decisions[rel] = status
    except Exception:
        # If the file is malformed, don't block validation.
        return {}

    return decisions


def _ensure_results_header(results_csv: Path):
    if results_csv.exists() and results_csv.stat().st_size > 0:
        return

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with results_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "relpath", "status"])


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _make_missing_placeholder(height: int, width: int, text: str) -> np.ndarray:
    height = max(1, int(height))
    width = max(1, int(width))
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        img,
        text,
        (20, max(40, height // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("img is None")
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _stack_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_bgr = _to_bgr(left)
    right_bgr = _to_bgr(right)

    h_left = left_bgr.shape[0]
    h_right = right_bgr.shape[0]

    # Resize right to match left height (keep aspect)
    if h_right != h_left:
        aspect = right_bgr.shape[1] / max(1, right_bgr.shape[0])
        new_w = max(1, int(round(h_left * aspect)))
        right_bgr = cv2.resize(right_bgr, (new_w, h_left), interpolation=cv2.INTER_AREA)

    # Separator (red line)
    sep = np.zeros((h_left, 10, 3), dtype=np.uint8)
    sep[:] = (0, 0, 255)

    combined = np.hstack([left_bgr, sep, right_bgr])
    return combined


def _fit_to_screen(img: np.ndarray, max_w: int = 1500, max_h: int = 900) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
    if scale >= 1.0:
        return img
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _overlay_instructions(img: np.ndarray, relpath: str, extra: str = "") -> np.ndarray:
    out = img.copy()
    lines = [
        f"{relpath}",
        "Keys: y=OK  n=BAD  s=SKIP  q=QUIT",
    ]
    if extra:
        lines.append(extra)

    y = 30
    for line in lines:
        cv2.putText(
            out,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 30

    return out


def _key_to_status(key: int) -> str | None:
    # cv2.waitKey returns platform-dependent codes; normalize to lower-case ascii when possible.
    if key == -1:
        return None
    key &= 0xFF
    c = chr(key).lower()
    if c == "y":
        return "ok"
    if c == "n":
        return "bad"
    if c == "s":
        return "skip"
    if c == "q":
        return "quit"
    return None


def validate(paths: Paths) -> int:
    if not paths.scans_dir.is_dir():
        print(f"Error: scans folder not found: {paths.scans_dir}", file=sys.stderr)
        return 2

    if not paths.processed_dir.exists():
        print(
            f"Error: processed folder not found: {paths.processed_dir}", file=sys.stderr
        )
        return 2

    existing = _load_existing_results(paths.results_csv)
    _ensure_results_header(paths.results_csv)

    total = 0
    reviewed = 0
    ok = 0
    bad = 0
    skipped = 0
    missing = 0

    window = "Validate: Scan (left) vs Processed (right)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    with paths.results_csv.open("a", newline="") as f:
        writer = csv.writer(f)

        for scan_file in _iter_images_recursive(paths.scans_dir):
            rel = scan_file.relative_to(paths.scans_dir).as_posix()
            total += 1

            if rel in existing and existing[rel] in {"ok", "bad"}:
                continue  # already validated

            processed_file = paths.processed_dir / rel

            scan_img = cv2.imread(str(scan_file), cv2.IMREAD_COLOR)
            if scan_img is None:
                continue

            proc_img = None
            extra = ""
            if processed_file.exists():
                proc_img = cv2.imread(str(processed_file), cv2.IMREAD_COLOR)

            if proc_img is None:
                missing += 1
                proc_img = _make_missing_placeholder(
                    height=scan_img.shape[0],
                    width=max(400, scan_img.shape[1]),
                    text="MISSING/UNREADABLE",
                )
                extra = f"Processed missing: {processed_file.as_posix()}"

            combined = _stack_side_by_side(scan_img, proc_img)
            combined = _overlay_instructions(combined, rel, extra=extra)
            display = _fit_to_screen(combined)

            cv2.imshow(window, display)

            while True:
                key = cv2.waitKey(0)
                status = _key_to_status(key)
                if status is None:
                    continue
                if status == "quit":
                    cv2.destroyAllWindows()
                    print(
                        f"Stopped. total_seen={total} reviewed={reviewed}",
                        file=sys.stderr,
                    )
                    return 0

                reviewed += 1
                if status == "ok":
                    ok += 1
                elif status == "bad":
                    bad += 1
                else:
                    skipped += 1

                writer.writerow([_now_utc_iso(), rel, status])
                f.flush()
                existing[rel] = status
                break

    cv2.destroyAllWindows()
    print(
        f"Done. total_seen={total} reviewed={reviewed} ok={ok} bad={bad} skip={skipped} missing_processed={missing}",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str]) -> int:
    root_dir = Path(__file__).resolve().parent
    scans_dir = root_dir / "imgs" / "scans"
    processed_dir = root_dir / "imgs" / "processed"
    results_csv = root_dir / "validation_results.csv"

    paths = Paths(
        scans_dir=scans_dir, processed_dir=processed_dir, results_csv=results_csv
    )
    return validate(paths)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
