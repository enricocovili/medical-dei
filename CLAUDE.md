# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Bachelor's thesis: anonymization and alignment correction pipeline for orthopanoramic and teleradiography medical images (University of Ferrara dataset).

## Commands

```bash
# Run the pipeline
python pipeline/app.py
python pipeline/app.py --verbose

# Run accuracy test
python test/test_accuracy.py
python test/test_accuracy.py --predictions imgs/output/deidentification/records.json \
  --ground-truth imgs/test_dataset_text_groundtruth.json --per-image
python test/test_accuracy.py --save-overlay-dir overlays/ --cropped-image-dir imgs/output/postprocess/crops/

# Install dependencies (uses uv)
uv sync
```

Python 3.12+ required (set in `.python-version`). Virtual environment at `.venv/`.

## Architecture

Four-stage CV pipeline, each stage independently runnable. Entrypoint: `pipeline/app.py`. Config: `setups/pipeline_config.toml` (`[pipeline]` section).

### Stage flow

```
SAM3 (segmentation) → Postprocess (erosion/rotation/crop) → Deidentification (OCR + blackout) → Report (JSON)
```

**Stage dependency**: each stage reads the previous stage's `records.json` from `artifacts_dir`. When `save_artifacts = false` in full mode, stages pass data in-memory instead.

### Key files

| File | Role |
|---|---|
| `pipeline/app.py` | Entrypoint, config loading, stage orchestration |
| `pipeline/models.py` | Frozen dataclasses: `BoundingBox`, `ImageEntry`, `SegmentationResult`, `MaskTransformResult`, `LoadedImage` |
| `pipeline/contracts.py` | Protocols: `ImageSource`, `Segmenter`, `MaskTransformer`, `Deidentifier`, `ReportWriter` |
| `pipeline/sam3_component.py` | `Sam3ImageSegmenter` — wraps SAM3 model with text prompt |
| `pipeline/mask_component.py` | `MaskPostprocessor` — erosion/dilation, minAreaRect rotation, crop |
| `pipeline/deidentifier_component.py` | `EasyOcrDeidentifier` — OCR text detection, ellipse exclusion, union-find merge |
| `pipeline/image_source.py` | `LocalImageSource` — yields `LoadedImage` from a dir or single file |
| `pipeline/report_writer.py` | `JsonReportWriter` — serializes `list[ImageEntry]` to JSON |
| `pipeline/logging_component.py` | `PipelineEventLogger` — structured log events |

### Config keys (`setups/pipeline_config.toml`)

- `run_mode`: `full | sam3 | postprocess | deidentification | report`
- `input_path`: source images (dir or single file); relative paths resolve from repo root
- `artifacts_dir`: base for all intermediate outputs
- `save_artifacts`: `false` skips writing intermediate images (in-memory pass-through in full mode)
- `prompt` / `fallback_prompt`: text prompts for SAM3 segmentation
- `easyocr_langs`, `easyocr_gpu`: OCR language list and GPU flag
- `ellipse_axis_x_ratio`, `ellipse_axis_y_ratio`, `ellipse_proximity_px`: central ellipse exclusion zone (text inside the scan body is not blacked out)
- `merge_distance_px`: union-find distance for merging nearby OCR detections
- `max_box_area_px`: discard OCR detections larger than this (0/null = no limit)
- `deid_padding_px`: padding added around each blacked-out box

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

Standalone script — no pipeline dependency. Compares `deidentification/records.json` predictions against a ground-truth JSON (LabelMe format). Reports GT coverage score, fully-covered rate, and outside-GT false positives. Supports `--save-overlay-dir` to render annotated images (green = GT, red = prediction).
