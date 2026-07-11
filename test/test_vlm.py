#!/usr/bin/env python3
"""Valuta modelli Visione-Linguaggio (VLM) sulla localizzazione del testo.

Per ogni modello e immagine il VLM riceve la scansione e deve restituire il
testo identificativo rilevato con la relativa bounding box. Le predizioni
vengono salvate nello stesso formato di `deidentification/records.json` e
confrontate con la ground-truth riusando le metriche di `test_accuracy.py`.

La ground-truth annota solo la posizione del testo (label generica "text"),
quindi lo script quantifica la LOCALIZZAZIONE (copertura GT, falsi positivi).
Il testo riconosciuto viene salvato nei record per l'ispezione qualitativa.

Richiede il pacchetto `openai` e la variabile d'ambiente OPENAI_API_KEY:
    uv add openai
    export OPENAI_API_KEY=...
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# riuso delle metriche di test_accuracy.py (stessa cartella)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_accuracy import (  # noqa: E402
    gt_best_coverage_ratio,
    parse_ground_truth,
    parse_predictions,
    pred_is_outside_ground_truth,
    safe_div,
)

DEFAULT_MODELS = ["gpt-5.4-mini"]

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

PROMPT = (
    "Sei un sistema di anonimizzazione di immagini mediche. "
    "Nell'immagine, individua OGNI porzione di testo identificativo"
    "Restituisci SOLO un oggetto JSON con questa forma esatta:\n"
    '{"detections": [{"text": "<testo>", "bbox": [x, y, w, h]}]}\n'
    "dove x,y sono l'angolo in alto a sinistra e w,h larghezza e altezza in PIXEL, "
    "con origine (0,0) in alto a sinistra dell'immagine. "
    "Non aggiungere spiegazioni, solo il JSON."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valutazione VLM per la localizzazione del testo."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Cartella con le immagini (ricerca ricorsiva).",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="JSON di ground-truth (formato LabelMe).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("imgs/output/vlm"),
        help="Cartella di output per le predizioni e il riepilogo.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Modelli da testare (default: {DEFAULT_MODELS}).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Limita il numero di immagini (0 = tutte).",
    )
    parser.add_argument(
        "--margin-px",
        type=float,
        default=5.0,
        help="Tolleranza in pixel per copertura/falsi positivi.",
    )
    parser.add_argument(
        "--prompt", type=str, default=PROMPT, help="Prompt inviato al VLM."
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Riusa i record già salvati invece di richiamare l'API.",
    )
    return parser.parse_args()


def index_images(root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXT:
            index.setdefault(path.stem, path)
    return index


def encode_image(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = MIME.get(path.suffix.lower(), "image/png")
    return f"data:{mime};base64,{data}"


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {"detections": []}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"detections": []}


def detections_to_boxes(payload: Dict[str, Any]) -> tuple[list[list[int]], list[str]]:
    boxes: list[list[int]] = []
    texts: list[str] = []
    for det in payload.get("detections", []):
        if not isinstance(det, dict):
            continue
        bbox = det.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x, y, w, h = (int(round(float(v))) for v in bbox)
        except (TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        boxes.append([x, y, w, h])
        texts.append(str(det.get("text", "")))
    return boxes, texts


def query_model(
    client: Any, model: str, image_path: Path, prompt: str
) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image(image_path)},
                    },
                ],
            }
        ],
    )
    content = response.choices[0].message.content or ""
    return extract_json(content)


def run_model(
    client: Any, model: str, images: Dict[str, Path], prompt: str, out_path: Path
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for i, (stem, path) in enumerate(images.items(), 1):
        t0 = time.perf_counter()
        try:
            payload = query_model(client, model, path, prompt)
            boxes, texts = detections_to_boxes(payload)
            error = ""
        except Exception as exc:  # noqa: BLE001 - vogliamo proseguire sugli altri
            boxes, texts, error = [], [], str(exc)
        records.append(
            {
                "name": path.name,
                "boxes": boxes,
                "texts": texts,
                "vlm_time": time.perf_counter() - t0,
                "error": error,
            }
        )
        flag = "ERR" if error else f"{len(boxes)} box"
        print(f"[{model}] {i}/{len(images)} {stem}: {flag}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return records


def score(
    records: List[Dict[str, Any]], ground_truth: Dict[str, List], margin_px: float
) -> Dict[str, float]:
    predictions = parse_predictions(records, min_confidence=0.0)
    images = sorted(set(ground_truth) | set(predictions))

    total_gt = fully_covered = uncovered = 0
    coverage_sum = 0.0
    total_pred = outside_fp = 0

    for image in images:
        preds = predictions.get(image, [])
        gts = ground_truth.get(image, [])
        for gt_box in gts:
            cov = gt_best_coverage_ratio(gt_box, preds, margin_px)
            coverage_sum += cov
            if cov >= 0.999999:
                fully_covered += 1
            if cov <= 0.0:
                uncovered += 1
        total_gt += len(gts)
        total_pred += len(preds)
        outside_fp += sum(
            1 for p in preds if pred_is_outside_ground_truth(p, gts, margin_px)
        )

    return {
        "gt_boxes": total_gt,
        "pred_boxes": total_pred,
        "coverage_score": safe_div(coverage_sum, total_gt),
        "fully_covered_rate": safe_div(fully_covered, total_gt),
        "uncovered_boxes": uncovered,
        "outside_gt_fp_rate": safe_div(outside_fp, total_pred),
    }


def main() -> None:
    args = parse_args()
    ground_truth = parse_ground_truth(
        json.loads(args.ground_truth.read_text(encoding="utf-8"))
    )

    images = index_images(args.images_dir)
    gt_stems = set(ground_truth)
    if gt_stems:
        images = {s: p for s, p in images.items() if s in gt_stems}
    if args.max_images > 0:
        images = dict(list(images.items())[: args.max_images])
    if not images:
        raise SystemExit("Nessuna immagine trovata che corrisponda alla ground-truth.")
    print(f"Immagini da valutare: {len(images)}; modelli: {args.models}")

    client = None
    if not args.reuse:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit(
                "Pacchetto 'openai' mancante. Esegui: uv add openai"
            ) from exc
        client = OpenAI()

    summary: Dict[str, Dict[str, float]] = {}
    for model in args.models:
        out_path = args.out_dir / f"{model.replace('/', '_')}.json"
        if args.reuse and out_path.exists():
            records = json.loads(out_path.read_text(encoding="utf-8"))
            print(f"[{model}] riuso {out_path}")
        else:
            records = run_model(client, model, images, args.prompt, out_path)
        summary[model] = score(records, ground_truth, args.margin_px)

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== Confronto VLM (localizzazione) ===")
    header = f"{'modello':<16}{'cov':>8}{'full':>8}{'uncov':>8}{'fp_out':>8}{'gt':>6}{'pred':>7}"
    print(header)
    print("-" * len(header))
    for model, m in summary.items():
        print(
            f"{model:<16}"
            f"{m['coverage_score']:>8.3f}"
            f"{m['fully_covered_rate']:>8.3f}"
            f"{m['uncovered_boxes']:>8d}"
            f"{m['outside_gt_fp_rate']:>8.3f}"
            f"{m['gt_boxes']:>6d}"
            f"{m['pred_boxes']:>7d}"
        )
    print(f"\nRiepilogo salvato in {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
