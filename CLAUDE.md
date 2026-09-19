# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Bachelor's thesis: anonymization and alignment correction pipeline for orthopanoramic and teleradiography medical images (University of Ferrara dataset).

## Commands

```bash
# Install dependencies (uses uv)
uv sync                                   # core: easyocr only
uv sync --extra rapidocr --extra onnxtr   # + the ONNX detectors

# Run the pipeline
python pipeline/app.py
python pipeline/app.py --verbose

# Accuracy of one prediction file
python test/test_accuracy.py --per-image \
  --predictions thesis/thesis_data_out/ocr_benchmarks/panoramic/<variant>/deidentification/records.json \
  --ground-truth imgs/sam3_processed_panoramic/test_dataset_text_groundtruth.json \
  --images-dir imgs/sam3_processed_panoramic/imgs

# Compare OCR configurations (both modalities, one command)
python test/benchmark_deid.py --matrix setups/benchmark_matrix.toml
python test/benchmark_deid.py --datasets panoramic --only baseline_easyocr --limit 12

# Invariant tests (no test framework installed; plain script, non-zero on failure)
python test/test_invariants.py
```

Python 3.12+ required (set in `.python-version`). Virtual environment at `.venv/`.

## Architecture

Four-stage CV pipeline, each stage independently runnable. Entrypoint: `pipeline/app.py`. Config: `setups/pipeline_config.toml` (`[pipeline]` section).

### Stage flow

```
SAM3 (segmentation) → Postprocess (erosion/rotation/crop) → Deidentification (OCR + blackout) → Report (JSON)
```

Inside the deidentification stage:

```
preprocess chain(s) → resolution strategy → OCR engine → confidence filter
  → text classifier → area/ellipse filter → union-find merge → pad → redact
```

Each of those four extension points is pluggable and configured from the TOML;
`Deidentifier` never touches engine internals.

**Stage dependency**: each stage reads the previous stage's `records.json` from `artifacts_dir`. When `save_artifacts = false` in full mode, stages pass data in-memory instead.

### Key files

| File | Role |
|---|---|
| `pipeline/app.py` | Entrypoint, config loading, stage orchestration |
| `pipeline/models.py` | Frozen dataclasses: `BoundingBox`, `ImageEntry`, `SegmentationResult`, `MaskTransformResult`, `LoadedImage` |
| `pipeline/contracts.py` | Protocols: `ImageSource`, `Segmenter`, `MaskTransformer`, `Deidentifier`, `ReportWriter` |
| `pipeline/sam3_component.py` | `Sam3ImageSegmenter` — wraps SAM3 model with text prompt |
| `pipeline/mask_component.py` | `MaskPostprocessor` — erosion/dilation, minAreaRect rotation, crop |
| `pipeline/deidentifier_component.py` | `Deidentifier` — orchestrates preprocess → detect → filter → merge → redact |
| `pipeline/ocr_engines.py` | `OcrEngine` protocol, `OcrDetection`, EasyOCR + PaddleOCR, `EnsembleOcrEngine` |
| `pipeline/ocr_engines_det.py` | RapidOCR (PP-OCRv6 ONNX), OnnxTR (DBNet/ResNet-50), Surya — lazily imported |
| `pipeline/ocr_engines_vlm.py` | `VlmEngine` over any OpenAI-compatible endpoint, plus grounding parsers |
| `pipeline/ocr_strategies.py` | `FullImageStrategy`, `TiledStrategy`, `UpscaleStrategy` |
| `pipeline/ocr_preprocess.py` | Coordinate-preserving preprocessing chain |
| `pipeline/text_classifiers.py` | Which detections get redacted |
| `pipeline/image_source.py` | `LocalImageSource` — yields `LoadedImage` from a dir or single file |
| `pipeline/report_writer.py` | `JsonReportWriter` — serializes `list[ImageEntry]` to JSON |
| `pipeline/logging_component.py` | `PipelineEventLogger` — structured log events |

### Config keys (`setups/pipeline_config.toml`)

- `run_mode`: `full | sam3 | postprocess | deidentification | report`
- `input_path`: source images (dir or single file); relative paths resolve from repo root
- `artifacts_dir`: base for all intermediate outputs
- `save_artifacts`: `false` skips writing intermediate images (in-memory pass-through in full mode)
- `prompt` / `fallback_prompt`: text prompts for SAM3 segmentation
- `ocr_engine`: `easyocr | paddleocr | rapidocr | onnxtr | surya | vlm | ensemble`
- `easyocr_langs`, `easyocr_gpu`: OCR language list and GPU flag
- `rapidocr_*`: PP-OCRv6 DB detector thresholds (`box_thresh`, `thresh`, `unclip_ratio`, `limit_side_len`) — the cheapest recall levers in the repo
- `onnxtr_*`: DBNet/ResNet-50 thresholds
- `vlm_*`: model, `vlm_base_url` (empty = hosted OpenAI), `vlm_response_format`, `vlm_coord_space`, `vlm_box_dilate_px`
- `ensemble_engines`, `ensemble_min_confidences`: union of several engines, with per-member confidence floors
- `preprocess_steps`: ordered subset of `grayscale, clahe, invert, tophat, blackhat, percentile_stretch, gamma, unsharp`
- `preprocess_variants`: several independent chains; the engine runs once per chain and detections are unioned
- `ocr_upscale_factor` / `ocr_upscale_interpolation`: enlarge the OCR input, boxes mapped back
- `resolution_strategy`: `full | tiled`
- `ellipse_enabled`, `ellipse_axis_x_ratio`, `ellipse_axis_y_ratio`, `ellipse_proximity_px`: central exclusion zone (see the warning below)
- `merge_distance_px`: union-find distance for merging nearby OCR detections
- `max_box_area_px` / `max_box_area_ratio`: discard oversized detections, absolute or as a fraction of image area (0/null = no limit)
- `deid_padding_px`: padding added around each blacked-out box
- `redaction_mode`: `fill` (real anonymization) | `outline` (debug only — draws rectangles and leaves the PHI readable)

### The geometric filters cap recall before OCR runs

Measured against the ground truth, with the reference settings:

| Filter | Panoramic GT lost | Teleradiography GT lost |
|---|---|---|
| central ellipse `0.45 / 0.35` | 8 / 81 (9.9%) | 16 / 136 (11.8%) |
| `max_box_area_px = 120000` | 0 / 81 | 13 / 136 (9.6%) |

The ellipse half-axes are fractions of the **full** width and height, so
`0.45/0.35` spans 5%–95% horizontally and 15%–85% vertically — nearly the whole
image, not a central core. A perfect OCR model therefore cannot exceed 0.901
box recall on panoramic or 0.816 on teleradiography until these change. Use
`ellipse_enabled = false` to disable the zone outright, and prefer
`max_box_area_ratio` over the absolute cap. `setups/benchmark_matrix.toml`
contains the ablation.

### Alarm system

`MaskPostprocessor` triggers `AlarmInfo(triggered=True)` when more than one bounding box exceeds `large_bb_area_ratio` of the image area — signals the image may contain multiple scan regions or unusual content.

### Artifact paths (relative to `artifacts_dir`)

| Stage | Files |
|---|---|
| SAM3 | `sam3/masks/<image>.png`, `sam3/records.json` |
| Postprocess | `postprocess/crops/<image>.png`, `postprocess/records.json` |
| Deidentification | `deidentification/images/<image>.png` (or `save_deidentified_dir`), `deidentification/records.json` |
| Report | `output_json` (configured path) |

### Accuracy testing (`test/test_accuracy.py`)

Standalone script — no pipeline dependency, which is what lets `benchmark_deid.py`
import it before `pipeline/` is on `sys.path`. Compares a `records.json` against a
LabelMe-style ground truth. `--save-overlay-dir` renders annotations (green = GT,
red = prediction); `--images-dir` enables the geometry-dependent metrics and makes
images absent from the ground truth count as text-free.

Metrics, and which to quote:

- **`recall_covered`, `gt_full_coverage_rate_union`** — fraction of GT boxes the
  predictions **together** cover. This is the headline recall for redaction.
- **`image_level_recall`** — fraction of images where *every* GT box is fully
  covered. This is the PHI-leak number, and what the literature calls case-level
  recall.
- `recall_iou` / `precision_iou` / `f1_iou` — standard detection metrics, kept for
  comparison with published work. **They read low here by construction**: a GT box
  covered 100% by a redaction box twice its size scores IoU ~0.3 and counts as a
  miss at the usual 0.5 threshold. Redaction deliberately over-covers; IoU
  penalises exactly that.
- `clean_image_fp_rate` — over-redaction on images with no text at all (72 of the
  98 panoramics), undiluted by the text-bearing ones.
- `central_fp` / `central_over_redaction` — predictions landing on anatomy.
- `redacted_area_fraction_global` — the pixel cost of over-redaction, i.e. the
  other side of every recall gain.
- `gt_coverage_score` / `gt_full_coverage_rate` — the original max-over-a-single-
  prediction variants, kept for continuity. They understate redaction when a
  detector splits one text line into two adjacent boxes.

### Benchmarking (`test/benchmark_deid.py`)

Sweeps `[[variant]]` config overrides across the `[[dataset]]` entries of
`setups/benchmark_matrix.toml` and writes `summary.{json,csv,md}`, a combined
`summary_all.csv`, and a per-variant `metrics.json` (so thesis figures can be
regenerated without re-running OCR).

Ground truth is annotated directly on the `images_dir` images, so the sweep needs
**neither a GPU nor a SAM3/postprocess rerun** — the postprocess records are
synthesised from a directory listing. Do not point it at regenerated crops: the
postprocess stage rotates and crops, which invalidates the annotations.

Detections are memoised on the exact pixels handed to the engine, so variants that
change only post-filtering replay in milliseconds instead of re-running OCR.
