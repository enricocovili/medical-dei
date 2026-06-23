#!/usr/bin/env python3
"""Render overlay di esempio per i tentativi con VLM.

Per ogni immagine selezionata disegna in verde le regioni di ground-truth
(testo identificativo annotato manualmente) e in rosso le bounding box
restituite dal VLM. Le regioni di ground-truth vengono SFOCATE prima del
disegno per non esporre dati personali del paziente nella tesi.

Uso:
    .venv/bin/python test/render_vlm_overlays.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_accuracy import (  # noqa: E402
    build_image_index,
    draw_rect,
    parse_ground_truth,
    parse_predictions,
)

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "imgs/sam3_processed_panoramic/imgs"
GT_PATH = ROOT / "imgs/sam3_processed_panoramic/test_dataset_text_groundtruth.json"
RECORDS = ROOT / "imgs/output/vlm/gpt-5.4-mini.json"
OUT_DIR = ROOT / "thesis/images/vlm_examples"

# immagini scelte per illustrare il disallineamento delle bbox del VLM
SELECTED = [
    "panoramic_patient_2014",
    "panoramic_patient_2134",
    "panoramic_patient_5460",
    "panoramic_patient_4835",
]

BLUR_PAD = 12  # padding (px) attorno alla GT da sfocare


# semiassi ellisse centrale (frazione di mezza larghezza/altezza): il testo
# dentro il corpo della scansione NON va sfocato, come nella pipeline
ELLIPSE_RX = 0.78
ELLIPSE_RY = 0.72


def in_central_ellipse(rect, w: int, h: int) -> bool:
    cx = (rect[0] + rect[2]) / 2
    cy = (rect[1] + rect[3]) / 2
    nx = (cx - w / 2) / (ELLIPSE_RX * w / 2)
    ny = (cy - h / 2) / (ELLIPSE_RY * h / 2)
    return nx * nx + ny * ny <= 1.0


def ocr_text_boxes(reader, img):
    """Tutte le regioni di testo rilevate da EasyOCR, (x1,y1,x2,y2).

    Per la privacy si sfoca OGNI testo: i pannelli identificativi possono
    trovarsi anche sul corpo della scansione (es. nelle foto di schermo).
    """
    boxes = []
    for poly, _txt, _conf in reader.readtext(img):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes


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
    k = max(31, (max(roi.shape[:2]) // 2) | 1)  # kernel dispari, forte
    img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def main() -> None:
    gt = parse_ground_truth(json.loads(GT_PATH.read_text(encoding="utf-8")))
    preds = parse_predictions(
        json.loads(RECORDS.read_text(encoding="utf-8")), min_confidence=0.0
    )
    index = build_image_index(IMAGES_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import easyocr  # sfoca OGNI testo rilevato, non solo la GT

    reader = easyocr.Reader(["en", "it"], gpu=False, verbose=False)

    for stem in SELECTED:
        path = index.get(stem)
        if path is None:
            print(f"[skip] immagine mancante: {stem}")
            continue
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] lettura fallita: {path}")
            continue
        gts = gt.get(stem, [])
        vlm = preds.get(stem, [])
        # sfoca ogni testo (OCR) + GT + box VLM prima di disegnare
        for box in ocr_text_boxes(reader, img):
            blur_region(img, box)
        for box in gts:
            blur_region(img, box)
        for box in vlm:
            blur_region(img, box)
        for box in gts:
            draw_rect(img, box, color=(0, 255, 0), thickness=3)
        for box in vlm:
            draw_rect(img, box, color=(0, 0, 255), thickness=3)
        out = OUT_DIR / f"{stem}_vlm.png"
        cv2.imwrite(str(out), img)
        print(f"[ok] {out}  (GT={len(gts)}, VLM={len(vlm)})")


if __name__ == "__main__":
    main()
