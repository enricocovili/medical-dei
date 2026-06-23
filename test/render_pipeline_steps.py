#!/usr/bin/env python3
"""Prepara le immagini illustrative dei singoli stadi della pipeline.

Esempio: una teleradiografia (paziente 2011) attraverso gli stadi:
  1. immagine grezza in ingresso
  2. maschera SAM3 (segmentazione)
  3. ritaglio dopo il post-processing
  4. risultato de-identificato

Il testo identificativo viene sfocato con EasyOCR su tutte le immagini che lo
contengono; la maschera binaria SAM3 viene copiata invariata.

Uso:
    .venv/bin/python test/render_pipeline_steps.py
"""

from __future__ import annotations

from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
TELE = ROOT / "imgs/telerad_tests"
RES3 = TELE / "teleradiography_results_3"
STEM = "teleradiography_patient_2081"
OUT_DIR = ROOT / "thesis/images/pipeline_steps"

# (file sorgente, nome output, sfocare il testo?)
STEPS = [
    (ROOT / "imgs/bkps/scans/teleradiography" / f"{STEM}.jpg", "step1_input.png", True),
    (RES3 / "sam3/masks" / f"{STEM}.png", "step2_sam3_mask.png", False),
    (RES3 / "postprocess/crops" / f"{STEM}.png", "step3_crop.png", True),
    (RES3 / "deidentification/images" / f"{STEM}.png", "step4_deid.png", True),
]

BLUR_PAD = 6


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
    k = max(21, (max(roi.shape[:2]) // 2) | 1)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def main() -> None:
    import easyocr

    reader = easyocr.Reader(["en", "it"], gpu=False, verbose=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for src, out_name, do_blur in STEPS:
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] {src}")
            continue
        n = 0
        if do_blur:
            for poly, _txt, _conf in reader.readtext(img):
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                blur_region(img, (min(xs), min(ys), max(xs), max(ys)))
                n += 1
        if out_name == "step1_input.png":
            # contorno nero per esaltare i bordi sullo sfondo bianco
            h, w = img.shape[:2]
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 0), thickness=3)
        cv2.imwrite(str(OUT_DIR / out_name), img)
        print(f"[ok] {out_name}: {'sfocate ' + str(n) + ' regioni' if do_blur else 'copiata'}")


if __name__ == "__main__":
    main()
