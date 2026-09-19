#!/usr/bin/env python3
"""Characterise a VLM's grounding output before writing a parser for it.

GLM-OCR's bounding-box format is documented nowhere — not on the model card,
not in the GitHub README, not in the Z.AI docs. DeepSeek-OCR-2's 0-999
normalisation is documented but unverified here. Guessing either would produce
boxes that are silently wrong, which for redaction means leaked PHI.

So this script answers two questions with evidence instead:

  1. WHAT does the model emit?  It dumps message.content verbatim and unparsed
     for each prompt, which is the whole answer.

  2. WHAT COORDINATE SPACE?  It sends the same crop at two different sizes. If
     the numbers come back identical the coordinates are normalised; if they
     scale with the image they are pixels. Two requests, no ambiguity.

It also renders each candidate interpretation (pixel / 0-999 / 0-1) onto the
image so the right one can be confirmed by eye rather than by argument.

Nothing is written to the pipeline config: once the space is known, set
vlm_coord_space in setups/pipeline_config.toml.

    # local model
    vllm serve zai-org/GLM-OCR --trust-remote-code --port 8000
    python test/probe_vlm_grounding.py --base-url http://localhost:8000/v1 \
        --model zai-org/GLM-OCR --image imgs/sam3_processed_panoramic/imgs/x.jpg

    # hosted
    export OPENAI_API_KEY=...
    python test/probe_vlm_grounding.py --model gpt-5.4-mini --image ...
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from ocr_engines_vlm import (  # noqa: E402
    PROMPT_GROUNDING,
    PROMPT_JSON,
    VLM_PARSERS,
)

PROMPTS = {
    "json": PROMPT_JSON,
    "grounding": PROMPT_GROUNDING,
    "plain": (
        "List every region of text in this image. For each, give the text and "
        "its bounding box. State explicitly what coordinate system you used."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None, help="Omit for the hosted API.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--prompts", nargs="*", default=list(PROMPTS), choices=list(PROMPTS)
    )
    parser.add_argument(
        "--sizes",
        nargs=2,
        type=int,
        default=[1024, 2048],
        help="Two longest-side sizes; identical numbers back means normalised.",
    )
    parser.add_argument(
        "--render-dir",
        type=Path,
        default=None,
        help="Draw each coordinate interpretation onto the image here.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def encode(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise SystemExit("failed to encode the image")
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def resized(image: np.ndarray, longest_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = longest_side / max(height, width)
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def ask(client, model: str, image: np.ndarray, prompt: str, timeout: float) -> str:
    response = client.chat.completions.create(
        model=model,
        timeout=timeout,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": encode(image)}},
                ],
            }
        ],
    )
    return response.choices[0].message.content or ""


def numbers_in(content: str) -> list[float]:
    import re

    return [float(match) for match in re.findall(r"-?\d+\.?\d*", content)][:40]


def render(image: np.ndarray, content: str, prompt_key: str, out_dir: Path) -> None:
    """Draw each candidate interpretation, so the right one is obvious by eye."""
    height, width = image.shape[:2]
    parser = VLM_PARSERS.get("deepseek_grounding" if prompt_key == "grounding" else "json")
    out_dir.mkdir(parents=True, exist_ok=True)
    for space in ("pixel", "normalized_999", "normalized_1"):
        try:
            detections = parser(content, width, height, space)
        except Exception as exc:  # noqa: BLE001 — a wrong guess is expected to fail
            print(f"      {space:16s} parse failed: {type(exc).__name__}: {exc}")
            continue
        canvas = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        for detection in detections:
            xs = [point[0] for point in detection.quad]
            ys = [point[1] for point in detection.quad]
            cv2.rectangle(
                canvas, (min(xs), min(ys)), (max(xs), max(ys)), (0, 0, 255), 3
            )
        path = out_dir / f"{prompt_key}_{space}.png"
        cv2.imwrite(str(path), canvas)
        print(f"      {space:16s} {len(detections):3d} boxes -> {path}")


def main() -> int:
    args = parse_args()
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("the 'openai' package is required")

    api_key = os.environ.get(args.api_key_env) or ("EMPTY" if args.base_url else None)
    if not api_key:
        raise SystemExit(f"set ${args.api_key_env}, or pass --base-url for a local model")
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot read {args.image}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(f"image: {args.image} ({image.shape[1]}x{image.shape[0]})")
    print(f"model: {args.model} @ {args.base_url or 'hosted OpenAI'}\n")

    small, large = sorted(args.sizes)
    for prompt_key in args.prompts:
        prompt = PROMPTS[prompt_key]
        print(f"{'=' * 70}\n=== prompt: {prompt_key}\n{'=' * 70}")

        content_large = ask(client, args.model, resized(image, large), prompt, args.timeout)
        print(f"--- RAW RESPONSE at {large}px (verbatim, unparsed) ---")
        print(content_large[:3000] or "<empty>")

        content_small = ask(client, args.model, resized(image, small), prompt, args.timeout)
        numbers_large = numbers_in(content_large)
        numbers_small = numbers_in(content_small)
        print(f"\n--- coordinate space, {small}px vs {large}px ---")
        if not numbers_large or not numbers_small:
            verdict = "no numbers returned — the prompt or the model is wrong"
        elif numbers_large == numbers_small:
            biggest = max(abs(value) for value in numbers_large)
            verdict = (
                "NORMALISED (identical at both sizes); "
                + ("looks like 0-1" if biggest <= 1.5 else "looks like 0-999")
            )
        else:
            verdict = "PIXELS (the numbers scale with the image)"
        print(f"  {small}px first numbers: {numbers_small[:8]}")
        print(f"  {large}px first numbers: {numbers_large[:8]}")
        print(f"  VERDICT: {verdict}")

        if args.render_dir is not None:
            print("\n--- rendered interpretations (open these and look) ---")
            render(resized(image, large), content_large, prompt_key, args.render_dir)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
