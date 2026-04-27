#!/usr/bin/env python3
"""Visualize predicted boxes and scores from *_labels.json files.

This script scans an input folder for images whose filenames match
`*_patient_<digits>.<ext>`, loads the corresponding label file
`<stem>_labels.json`, and renders the detected bounding box(es)
with score labels.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PATIENT_REGEX = re.compile(r".*_patient_\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read images named like *_patient_<digits> and matching "
            "<stem>_labels.json files, then visualize boxes and scores."
        )
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Folder containing images and their corresponding *_labels.json files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan input folder recursively.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Only draw detections with score >= this threshold (default: 0.0).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional output folder where annotated images are written.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open matplotlib windows (useful when only saving files).",
    )
    return parser.parse_args()


def list_candidate_images(folder: Path, recursive: bool) -> list[Path]:
    iterator: Iterable[Path] = folder.rglob("*") if recursive else folder.glob("*")
    candidates = []
    for path in iterator:
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not PATIENT_REGEX.match(path.stem):
            continue
        candidates.append(path)
    return sorted(candidates)


def load_labels(label_path: Path) -> tuple[list[list[float]], list[float], str]:
    with label_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    boxes = data.get("boxes", [])
    scores = data.get("scores", [])
    prompt = data.get("prompt", "")

    if not isinstance(boxes, list) or not isinstance(scores, list):
        raise ValueError("Invalid labels format: 'boxes' and 'scores' must be lists.")

    return boxes, scores, prompt


def draw_boxes(
    image_bgr, boxes: list[list[float]], scores: list[float], min_score: float
):
    annotated = image_bgr.copy()
    drawn = 0

    for idx, box in enumerate(boxes):
        if not isinstance(box, list) or len(box) != 4:
            continue

        score = float(scores[idx]) if idx < len(scores) else float("nan")
        if score == score and score < min_score:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 3)

        label_text = f"score: {score:.4f}" if score == score else "score: n/a"
        text_y = max(20, y1 - 10)
        cv2.putText(
            annotated,
            label_text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 50, 50),
            2,
            lineType=cv2.LINE_AA,
        )
        drawn += 1

    return annotated, drawn


def visualize_image(
    image_path: Path,
    boxes: list[list[float]],
    scores: list[float],
    prompt: str,
    min_score: float,
):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    annotated_bgr, drawn = draw_boxes(image_bgr, boxes, scores, min_score)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    return annotated_bgr, annotated_rgb, drawn


def main() -> int:
    args = parse_args()
    input_folder = args.input_folder

    if not input_folder.exists() or not input_folder.is_dir():
        print(
            f"Error: input folder does not exist or is not a directory: {input_folder}"
        )
        return 1

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    images = list_candidate_images(input_folder, recursive=args.recursive)
    if not images:
        print("No matching images found (expected pattern: *_patient_<digits>.<ext>).")
        return 0

    processed = 0
    for image_path in images:
        label_path = image_path.with_name(f"{image_path.stem}_labels.json")
        if not label_path.exists():
            print(f"Skipping {image_path.name}: missing {label_path.name}")
            continue

        try:
            boxes, scores, prompt = load_labels(label_path)
            annotated_bgr, annotated_rgb, drawn = visualize_image(
                image_path=image_path,
                boxes=boxes,
                scores=scores,
                prompt=prompt,
                min_score=args.min_score,
            )
        except Exception as exc:
            print(f"Skipping {image_path.name}: {exc}")
            continue

        print(f"{image_path.name}: drew {drawn} box(es)")
        processed += 1

        if args.save_dir is not None:
            out_path = args.save_dir / f"{image_path.stem}_annotated{image_path.suffix}"
            cv2.imwrite(str(out_path), annotated_bgr)

        if not args.no_show:
            title = image_path.name
            if prompt:
                title = f"{title} | {prompt}"
            fig = plt.figure(figsize=(13, 6))
            plt.imshow(annotated_rgb)
            plt.title(title)
            plt.axis("off")
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.001)
            fig.waitforbuttonpress()
            plt.close(fig)

    print(f"Completed. Visualized {processed} image(s).")
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        exit(0)
