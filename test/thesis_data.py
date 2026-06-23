#!/usr/bin/env python3
"""Produce every number/figure still missing from the thesis.

Run on a GPU machine inside the repo (uv/venv active). Reuses the pipeline's
own stage functions (pipeline/app.py) and the evaluation logic
(test/test_accuracy.py); nothing in the core pipeline is modified.

It fills the placeholder ("--") tables and figures of the thesis:

  tab:prestazioni       -> task "bench"        (per-stage runtime)
  tab:ablation_ellipse  -> task "ablation"     (ellipse on/off)
  tab:sensitivity       -> task "sensitivity"  (param sweeps, panoramics)
  tab:paddle            -> task "paddle"       (PaddleOCR comparison)
  fig:distributions     -> task "dist"         (coverage + |theta| histograms)

Usage:
  python test/thesis_data.py all                 # everything
  python test/thesis_data.py ablation sensitivity
  python test/thesis_data.py bench --bench-limit 40

Outputs:
  - human-readable summary + LaTeX snippets under test/thesis_data_out/
  - histogram PNGs under thesis/images/distributions/
  - prints the ready-to-paste numbers to stdout
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2

# --- make pipeline/ and test/ importable regardless of CWD -------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))
sys.path.insert(0, str(REPO_ROOT / "test"))

import app as pipeline_app  # noqa: E402  (pipeline/app.py)
from app import (  # noqa: E402
    PipelineConfig,
    build_artifacts,
    build_deidentifier,
    run,
)
import test_accuracy as ta  # noqa: E402

# =============================================================================
# CONFIGURATION  --  edit paths here if the dataset lives elsewhere
# =============================================================================

OUT_DIR = REPO_ROOT / "test" / "thesis_data_out"
FIG_DIR = REPO_ROOT / "thesis" / "images" / "distributions"
MARGIN_PX = 5.0  # same tolerance used in the thesis / test_accuracy

# Reference de-identification parameters (mirror setups/pipeline_config.toml).
REF = dict(
    prompt="rectangular panoramic scan",
    fallback_prompt="",
    kernel_size=9,
    iterations=5,
    large_bb_area_ratio=0.1,
    easyocr_langs=["en", "it"],
    easyocr_gpu=True,
    merge_distance_px=10,
    max_box_area_px=120000,
    ellipse_axis_x_ratio=0.45,
    ellipse_axis_y_ratio=0.35,
    ellipse_proximity_px=0.0,
    deid_padding_px=10,
)

# Per-modality dataset layout.
#   crops_dir : post-processed images the GT was annotated on (de-id input)
#   gt_json   : LabelMe-style text ground truth
#   raw_dir   : raw uncropped scans (only used by the "bench" full run)
DATASETS: dict[str, dict[str, Path]] = {
    "panoramic": {
        "crops_dir": REPO_ROOT / "imgs" / "sam3_processed_panoramic" / "imgs",
        "gt_json": REPO_ROOT
        / "imgs"
        / "sam3_processed_panoramic"
        / "test_dataset_text_groundtruth.json",
        "raw_dir": REPO_ROOT / "imgs" / "bkps" / "scans" / "panoramic",
    },
    "teleradiography": {
        "crops_dir": REPO_ROOT
        / "imgs"
        / "telerad_tests"
        / "teleradiography_results_3"
        / "postprocess"
        / "crops",
        "gt_json": REPO_ROOT
        / "imgs"
        / "telerad_tests"
        / "teleradiography_results_3"
        / "groundtruth.json",
        "raw_dir": REPO_ROOT
        / "imgs"
        / "telerad_tests"
        / "teleradiography_with_text",
    },
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_LOG = logging.getLogger("thesis_data")
_DIMS_CACHE: dict[str, tuple[int, int]] = {}


# =============================================================================
# Pipeline / metric helpers
# =============================================================================

def _make_config(artifacts_dir: Path, **overrides: Any) -> PipelineConfig:
    """Build a valid in-memory PipelineConfig from REF + overrides."""
    params = {**REF, **overrides}
    return PipelineConfig(
        run_mode=params.get("run_mode", "deidentification"),
        input_path=params.get("input_path", artifacts_dir),
        output_json=params.get("output_json", artifacts_dir / "results.json"),
        artifacts_dir=artifacts_dir,
        save_artifacts=params.get("save_artifacts", False),
        prompt=params["prompt"],
        fallback_prompt=params["fallback_prompt"] or None,
        kernel_size=params["kernel_size"],
        iterations=params["iterations"],
        large_bb_area_ratio=params["large_bb_area_ratio"],
        easyocr_langs=params["easyocr_langs"],
        easyocr_gpu=params["easyocr_gpu"],
        save_deidentified_dir=None,
        merge_distance_px=params["merge_distance_px"],
        max_box_area_px=params["max_box_area_px"],
        ellipse_axis_x_ratio=params["ellipse_axis_x_ratio"],
        ellipse_axis_y_ratio=params["ellipse_axis_y_ratio"],
        ellipse_proximity_px=params["ellipse_proximity_px"],
        deid_padding_px=params["deid_padding_px"],
    )


def _list_crops(crops_dir: Path, gt_keys: set[str] | None) -> list[Path]:
    paths = [
        p
        for p in sorted(crops_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    if gt_keys is not None:
        paths = [p for p in paths if p.stem in gt_keys]
    return paths


def _img_dims(path: Path) -> tuple[int, int]:
    """Return (width, height) of an image, cached by stem."""
    if path.stem in _DIMS_CACHE:
        return _DIMS_CACHE[path.stem]
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        dims = (0, 0)
    else:
        dims = (img.shape[1], img.shape[0])
    _DIMS_CACHE[path.stem] = dims
    return dims


def run_deid(crops: list[Path], **deid_overrides: Any) -> list[dict[str, Any]]:
    """Run only the de-identification stage over `crops`, return records.

    Each record: {"name": <file>, "boxes": [[x, y, w, h], ...]}.
    """
    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    config = _make_config(work, **deid_overrides)
    artifacts = build_artifacts(config)
    artifacts = dataclasses.replace(
        artifacts, postprocess_crops_dir=crops[0].parent if crops else work
    )
    logger = pipeline_app.PipelineEventLogger(_LOG)
    deidentifier = build_deidentifier(logger, config)
    fake_records = [
        {"name": p.name, "crop_relpath": p.name} for p in crops
    ]
    # crops may live in nested dirs; point crop dir at each file's parent set.
    # run_deidentification_stage resolves crop via postprocess_crops_dir / crop_relpath,
    # so when crops share one dir this is correct; otherwise fall back per-file below.
    out = pipeline_app.run_deidentification_stage(
        artifacts,
        logger,
        deidentifier,
        postprocess_records=fake_records,
        in_memory_crops=_load_crops_in_memory(crops),
        save_artifacts=False,
    )
    return out


def _load_crops_in_memory(crops: list[Path]) -> dict[str, Any]:
    """Pre-load crops so de-id works regardless of nested directory layout."""
    loaded: dict[str, Any] = {}
    for p in crops:
        rgb = pipeline_app._load_image_rgb(p)
        loaded[p.name] = rgb
        _DIMS_CACHE[p.stem] = (rgb.shape[1], rgb.shape[0])
    return loaded


def _touches_ref_ellipse(rect: ta.Rect, w: int, h: int) -> bool:
    """Replicate EasyOcrDeidentifier._touches_center_ellipse for the ref ellipse."""
    if w <= 0 or h <= 0:
        return False
    cx, cy = w / 2.0, h / 2.0
    ax = max(1.0, w * REF["ellipse_axis_x_ratio"] + REF["ellipse_proximity_px"])
    ay = max(1.0, h * REF["ellipse_axis_y_ratio"] + REF["ellipse_proximity_px"])
    x1, y1, x2, y2 = rect
    nx = min(max(cx, x1), x2)
    ny = min(max(cy, y1), y2)
    return ((nx - cx) / ax) ** 2 + ((ny - cy) / ay) ** 2 <= 1.0


def evaluate(
    records: list[dict[str, Any]],
    gt_raw: dict[str, Any],
    *,
    crops_by_stem: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Compute coverage / FP metrics for a set of prediction records."""
    preds = ta.parse_predictions(records, min_confidence=0.0)
    gts = ta.parse_ground_truth(gt_raw)
    images = sorted(set(gts) | set(preds))

    gt_total = fully = uncovered = 0
    cov_sum = 0.0
    pred_total = outside_fp = central_fp = 0
    per_region_cov: list[float] = []

    for image in images:
        p = preds.get(image, [])
        g = gts.get(image, [])
        for gbox in g:
            cov = ta.gt_best_coverage_ratio(gbox, p, MARGIN_PX)
            per_region_cov.append(cov)
            cov_sum += cov
            if cov >= 0.999999:
                fully += 1
            if cov <= 0.0:
                uncovered += 1
        gt_total += len(g)
        pred_total += len(p)
        outside_fp += sum(
            1 for pb in p if ta.pred_is_outside_ground_truth(pb, g, MARGIN_PX)
        )
        if crops_by_stem is not None and image in crops_by_stem:
            w, h = _img_dims(crops_by_stem[image])
            central_fp += sum(1 for pb in p if _touches_ref_ellipse(pb, w, h))

    return {
        "gt_total": gt_total,
        "pred_total": pred_total,
        "cov_mean": cov_sum / gt_total if gt_total else 0.0,
        "fully_covered": fully,
        "fully_rate": fully / gt_total if gt_total else 0.0,
        "uncovered": uncovered,
        "outside_fp": outside_fp,
        "outside_rate": outside_fp / pred_total if pred_total else 0.0,
        "central_fp": central_fp,
        "per_region_cov": per_region_cov,
    }


def _gt_keys(gt_json: Path) -> set[str]:
    return set(json.loads(gt_json.read_text(encoding="utf-8")).keys())


def _crops_by_stem(crops: list[Path]) -> dict[str, Path]:
    return {p.stem: p for p in crops}


# =============================================================================
# Tasks
# =============================================================================

def task_ablation(gt_only: bool = True) -> dict[str, Any]:
    """tab:ablation_ellipse -- ellipse on (ref) vs off, per modality."""
    print("\n### ABLATION: central-ellipse exclusion (tab:ablation_ellipse)")
    rows: dict[str, Any] = {}
    for name, ds in DATASETS.items():
        gt_raw = json.loads(ds["gt_json"].read_text(encoding="utf-8"))
        keys = set(gt_raw) if gt_only else None
        crops = _list_crops(ds["crops_dir"], keys)
        if not crops:
            print(f"  [skip] {name}: no crops at {ds['crops_dir']}")
            continue
        cbs = _crops_by_stem(crops)
        # ON = reference ratios; OFF = tiny ellipse so nothing is excluded.
        rec_on = run_deid(crops)
        rec_off = run_deid(
            crops, ellipse_axis_x_ratio=0.001, ellipse_axis_y_ratio=0.001
        )
        m_on = evaluate(rec_on, gt_raw, crops_by_stem=cbs)
        m_off = evaluate(rec_off, gt_raw, crops_by_stem=cbs)
        rows[name] = {"on": m_on, "off": m_off}
        print(
            f"  {name:16s} ON : cov={m_on['cov_mean']:.3f} central_fp={m_on['central_fp']}"
        )
        print(
            f"  {name:16s} OFF: cov={m_off['cov_mean']:.3f} central_fp={m_off['central_fp']}"
        )
    _write_latex_ablation(rows)
    return rows


def task_sensitivity(gt_only: bool = True) -> dict[str, Any]:
    """tab:sensitivity -- per-parameter sweeps on panoramics."""
    print("\n### SENSITIVITY: filtering parameters, panoramics (tab:sensitivity)")
    ds = DATASETS["panoramic"]
    gt_raw = json.loads(ds["gt_json"].read_text(encoding="utf-8"))
    keys = set(gt_raw) if gt_only else None
    crops = _list_crops(ds["crops_dir"], keys)
    if not crops:
        print(f"  [skip] no panoramic crops at {ds['crops_dir']}")
        return {}
    sweeps = {
        "merge_distance_px": [5, 10, 20],
        "max_box_area_px": [60000, 120000, None],
        "deid_padding_px": [0, 10, 20],
    }
    results: dict[str, Any] = {}
    for param, values in sweeps.items():
        results[param] = {}
        for v in values:
            rec = run_deid(crops, **{param: v})
            m = evaluate(rec, gt_raw)
            results[param][str(v)] = m
            print(
                f"  {param}={str(v):>8s}  cov={m['cov_mean']:.3f}  "
                f"outside={m['outside_rate']:.3f}"
            )
    _write_latex_sensitivity(results)
    return results


def task_paddle(gt_only: bool = True) -> dict[str, Any]:
    """tab:paddle -- PaddleOCR localization on the 98 panoramics."""
    print("\n### PADDLEOCR comparison (tab:paddle)")
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] PaddleOCR not importable: {exc}")
        print("        install with: uv pip install paddleocr paddlepaddle-gpu")
        return {}
    ds = DATASETS["panoramic"]
    gt_raw = json.loads(ds["gt_json"].read_text(encoding="utf-8"))
    keys = set(gt_raw) if gt_only else None
    crops = _list_crops(ds["crops_dir"], keys)
    if not crops:
        print(f"  [skip] no panoramic crops at {ds['crops_dir']}")
        return {}
    ocr = PaddleOCR(use_angle_cls=False, lang="it", show_log=False)
    records: list[dict[str, Any]] = []
    for p in crops:
        boxes = _paddle_boxes(ocr, str(p))
        records.append({"name": p.name, "boxes": boxes})
    (OUT_DIR / "paddle_records.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    m = evaluate(records, gt_raw)
    print(
        f"  PaddleOCR: cov={m['cov_mean']:.3f} fully={m['fully_covered']}/{m['gt_total']} "
        f"uncovered={m['uncovered']} outside={m['outside_rate']:.3f}"
    )
    _write_latex_paddle(m)
    return m


def _paddle_boxes(ocr: Any, image_path: str) -> list[list[int]]:
    """Extract [x, y, w, h] boxes from a PaddleOCR result (version-tolerant)."""
    boxes: list[list[int]] = []
    try:
        result = ocr.ocr(image_path, cls=False)
    except TypeError:
        result = ocr.ocr(image_path)
    if not result:
        return boxes
    # result is typically list[ list[ (quad, (text, conf)) ] ]
    lines = result[0] if len(result) == 1 and isinstance(result[0], list) else result
    for entry in lines or []:
        try:
            quad = entry[0]
            xs = [float(pt[0]) for pt in quad]
            ys = [float(pt[1]) for pt in quad]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x, max(ys) - y
            if w > 0 and h > 0:
                boxes.append([int(x), int(y), int(w), int(h)])
        except (TypeError, IndexError, ValueError):
            continue
    return boxes


def task_bench(bench_limit: int = 30) -> dict[str, Any]:
    """tab:prestazioni -- per-stage runtime on a clean full run."""
    print(f"\n### BENCHMARK: per-stage runtime, up to {bench_limit} imgs/modality")
    summary: dict[str, Any] = {}
    for name, ds in DATASETS.items():
        raw_dir = ds["raw_dir"]
        raws = _list_crops(raw_dir, None)[:bench_limit]
        if not raws:
            print(f"  [skip] {name}: no raw scans at {raw_dir}")
            continue
        work = OUT_DIR / f"_bench_{name}"
        # stage input dir = a temp dir with the sampled raws (symlinks)
        in_dir = work / "input"
        in_dir.mkdir(parents=True, exist_ok=True)
        for p in raws:
            link = in_dir / p.name
            if not link.exists():
                try:
                    link.symlink_to(p)
                except OSError:
                    link.write_bytes(p.read_bytes())
        out_json = work / "results.json"
        config = _make_config(
            work,
            run_mode="full",
            input_path=in_dir,
            output_json=out_json,
            save_artifacts=False,
        )
        logger = pipeline_app.PipelineEventLogger(_LOG)
        t0 = time.perf_counter()
        run(config, logger)
        wall = time.perf_counter() - t0
        entries = json.loads(out_json.read_text(encoding="utf-8"))
        sam = [e["metrics"]["times"]["sam3_inference"] for e in entries]
        post = [e["metrics"]["times"]["erosion_diffusion"] for e in entries]
        deid = [e["metrics"]["times"]["deidentification"] for e in entries]
        n = len(entries)
        row = {
            "n": n,
            "sam3": statistics.mean(sam) if sam else 0.0,
            "post": statistics.mean(post) if post else 0.0,
            "deid": statistics.mean(deid) if deid else 0.0,
            "wall_per_img": wall / n if n else 0.0,
            "results_json": str(out_json),
        }
        row["total"] = row["sam3"] + row["post"] + row["deid"]
        summary[name] = row
        print(
            f"  {name:16s} n={n} sam3={row['sam3']:.3f}s post={row['post']:.4f}s "
            f"deid={row['deid']:.3f}s total={row['total']:.3f}s"
        )
    _write_latex_bench(summary)
    return summary


def task_dist(report_jsons: dict[str, Path] | None = None) -> None:
    """fig:distributions -- coverage-per-region and |theta| histograms."""
    print("\n### DISTRIBUTIONS: histograms (fig:distributions)")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] matplotlib unavailable: {exc}")
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- coverage per region (reference de-id on annotated crops) ---
    cov_by_mod: dict[str, list[float]] = {}
    for name, ds in DATASETS.items():
        if not ds["gt_json"].exists():
            continue
        gt_raw = json.loads(ds["gt_json"].read_text(encoding="utf-8"))
        crops = _list_crops(ds["crops_dir"], set(gt_raw))
        if not crops:
            continue
        rec = run_deid(crops)
        cov_by_mod[name] = evaluate(rec, gt_raw)["per_region_cov"]

    if cov_by_mod:
        plt.figure(figsize=(5, 3.2))
        for name, vals in cov_by_mod.items():
            plt.hist(vals, bins=[i / 10 for i in range(11)], alpha=0.6, label=name)
        plt.xlabel("copertura per regione")
        plt.ylabel("numero di regioni")
        plt.legend()
        plt.tight_layout()
        out = FIG_DIR / "coverage_hist.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  wrote {out}")

    # --- |theta| from report jsons (default: bench outputs) ---
    if report_jsons is None:
        report_jsons = {
            name: OUT_DIR / f"_bench_{name}" / "results.json" for name in DATASETS
        }
    theta_by_mod: dict[str, list[float]] = {}
    for name, jpath in report_jsons.items():
        if not Path(jpath).exists():
            print(f"  [theta skip] {name}: run 'bench' first or pass a report json")
            continue
        entries = json.loads(Path(jpath).read_text(encoding="utf-8"))
        theta_by_mod[name] = [
            abs(float(e["metrics"]["rotation_angle"])) for e in entries
        ]

    if theta_by_mod:
        plt.figure(figsize=(5, 3.2))
        maxdeg = max((max(v) for v in theta_by_mod.values() if v), default=5)
        bins = [i for i in range(int(maxdeg) + 2)]
        for name, vals in theta_by_mod.items():
            plt.hist(vals, bins=bins, alpha=0.6, label=name)
        plt.xlabel(r"$|\theta|$ (gradi)")
        plt.ylabel("numero di scansioni")
        plt.legend()
        plt.tight_layout()
        out = FIG_DIR / "theta_hist.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  wrote {out}")


# =============================================================================
# LaTeX snippet writers
# =============================================================================

def _f(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace(".", "{,}")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"  -> {path}")


def _write_latex_ablation(rows: dict[str, Any]) -> None:
    pano = rows.get("panoramic")
    tele = rows.get("teleradiography")
    def cell(m, k):
        return "--" if m is None else (_f(m["cov_mean"]) if k == "cov" else str(m["central_fp"]))
    lines = [
        "% auto-generated: tab:ablation_ellipse",
        "Con ellisse (riferimento) & "
        f"{cell(pano and pano['on'],'cov')} & {cell(pano and pano['on'],'fp')} & "
        f"{cell(tele and tele['on'],'cov')} & {cell(tele and tele['on'],'fp')} \\\\",
        "Senza ellisse             & "
        f"{cell(pano and pano['off'],'cov')} & {cell(pano and pano['off'],'fp')} & "
        f"{cell(tele and tele['off'],'cov')} & {cell(tele and tele['off'],'fp')} \\\\",
    ]
    _write(OUT_DIR / "tab_ablation_ellipse.tex", "\n".join(lines) + "\n")


def _write_latex_sensitivity(results: dict[str, Any]) -> None:
    def cells(param, v):
        m = results.get(param, {}).get(str(v))
        if m is None:
            return "-- & --"
        return f"{_f(m['cov_mean'])} & {_f(m['outside_rate'])}"
    lines = ["% auto-generated: tab:sensitivity (full 4-column rows)"]
    lines.append(
        "\\multirow{3}{*}{Distanza di fusione (px)} & 5             & "
        f"{cells('merge_distance_px',5)} \\\\"
    )
    lines.append(f"                                          & 10 (rif.)     & {cells('merge_distance_px',10)} \\\\")
    lines.append(f"                                          & 20            & {cells('merge_distance_px',20)} \\\\")
    lines.append("\\midrule")
    lines.append(
        "\\multirow{3}{*}{Area massima (px$^2$)}    & 60\\,000       & "
        f"{cells('max_box_area_px',60000)} \\\\"
    )
    lines.append(f"                                          & 120\\,000 (rif.) & {cells('max_box_area_px',120000)} \\\\")
    lines.append(f"                                          & nessun limite & {cells('max_box_area_px',None)} \\\\")
    lines.append("\\midrule")
    lines.append(
        "\\multirow{3}{*}{Margine oscuramento (px)} & 0             & "
        f"{cells('deid_padding_px',0)} \\\\"
    )
    lines.append(f"                                          & 10 (rif.)     & {cells('deid_padding_px',10)} \\\\")
    lines.append(f"                                          & 20            & {cells('deid_padding_px',20)} \\\\")
    _write(OUT_DIR / "tab_sensitivity.tex", "\n".join(lines) + "\n")


def _write_latex_paddle(m: dict[str, Any]) -> None:
    # Full 4-column rows: EasyOCR (ref) | PaddleOCR (measured) | GPT-5.4 mini (ref).
    p_full = f"{m['fully_covered']}/{m['gt_total']} \\;({_f(m['fully_rate'])})"
    lines = [
        "% auto-generated: tab:paddle (full 4-column rows)",
        f"Copertura media GT            & 0{{,}}887           & {_f(m['cov_mean'])} & 0{{,}}319 \\\\",
        f"Regioni completamente coperte & 68/81 \\;(0{{,}}840) & {p_full} & 6/81 \\;(0{{,}}074) \\\\",
        f"Regioni non coperte           & 7                 & {m['uncovered']} & 36 \\\\",
        f"Predizioni esterne alla GT    & 0{{,}}622           & {_f(m['outside_rate'])} & 0{{,}}829 \\\\",
    ]
    _write(OUT_DIR / "tab_paddle.tex", "\n".join(lines) + "\n")


def _write_latex_bench(summary: dict[str, Any]) -> None:
    def col(name, key):
        r = summary.get(name)
        return "--" if r is None else _f(r[key], 3)
    lines = [
        "% auto-generated: tab:prestazioni",
        f"Segmentazione (SAM3)         & {col('panoramic','sam3')} & {col('teleradiography','sam3')} \\\\",
        f"Post-processing              & {col('panoramic','post')} & {col('teleradiography','post')} \\\\",
        f"De-identificazione (EasyOCR) & {col('panoramic','deid')} & {col('teleradiography','deid')} \\\\",
        "\\midrule",
        f"\\textbf{{Totale per immagine}} & {col('panoramic','total')} & {col('teleradiography','total')} \\\\",
    ]
    _write(OUT_DIR / "tab_prestazioni.tex", "\n".join(lines) + "\n")


# =============================================================================
# Entry point
# =============================================================================

TASKS = {
    "ablation": task_ablation,
    "sensitivity": task_sensitivity,
    "paddle": task_paddle,
    "bench": task_bench,
    "dist": task_dist,
}
ALL_ORDER = ["bench", "ablation", "sensitivity", "paddle", "dist"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tasks", nargs="+", choices=[*TASKS, "all"], help="tasks to run"
    )
    parser.add_argument("--bench-limit", type=int, default=30)
    parser.add_argument(
        "--all-crops",
        action="store_true",
        help="evaluate every crop, not only GT-annotated ones",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt_only = not args.all_crops

    chosen = ALL_ORDER if "all" in args.tasks else args.tasks
    for t in chosen:
        if t == "bench":
            task_bench(args.bench_limit)
        elif t == "ablation":
            task_ablation(gt_only)
        elif t == "sensitivity":
            task_sensitivity(gt_only)
        elif t == "paddle":
            task_paddle(gt_only)
        elif t == "dist":
            task_dist()

    print(f"\nDone. LaTeX snippets + outputs in: {OUT_DIR}")
    print(f"Histograms (if dist ran) in: {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
