#!/usr/bin/env python3
"""Sfoca i dati sensibili negli overlay dei risultati (Capitolo 4).

Gli overlay in thesis/images/overlay_examples/ mostrano in verde la ground-truth
e in rosso le predizioni della pipeline, ma il testo identificativo del paziente
resta leggibile. Questo script rileva ogni porzione di testo con EasyOCR e la
sfoca, preservando i rettangoli gia disegnati.

Uso:
    .venv/bin/python test/blur_overlay_pii.py
"""

from __future__ import annotations

from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
OVERLAY_DIR = ROOT / "thesis/images/overlay_examples"
FILES = ["pano_2134.png", "pano_5460.png", "telerad_2011.png", "telerad_2043.png"]

BLUR_PAD = 10


def blur_region(img, rect, pad: int = BLUR_PAD) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = rect
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad)
    y2 = min(h, int(y2) + pad)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    k = max(31, (max(roi.shape[:2]) // 2) | 1)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def main() -> None:
    import easyocr

    reader = easyocr.Reader(["en", "it"], gpu=False, verbose=False)
    for name in FILES:
        path = OVERLAY_DIR / name
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] {path}")
            continue
        n = 0
        for poly, _txt, _conf in reader.readtext(img):
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            blur_region(img, (min(xs), min(ys), max(xs), max(ys)))
            n += 1
        cv2.imwrite(str(path), img)
        print(f"[ok] {name}: sfocate {n} regioni di testo")


if __name__ == "__main__":
    main()
