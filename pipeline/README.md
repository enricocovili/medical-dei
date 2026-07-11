# Pipeline Paths Summary

Single entrypoint: `pipeline/app.py`  
Only CLI flag: `--verbose`  
Execution mode is selected in `setups/pipeline_config.toml` with `run_mode`.

## Run modes

- `full`: executes all stages in order
- `sam3`: executes only SAM3 inference
- `postprocess`: executes only erosion/diffusion + rotation + crop
- `deidentification`: executes only deidentification
- `report`: executes only final JSON generation

## Path resolution

All paths from `setups/pipeline_config.toml` are resolved from repository root when relative (`_resolve_path` in `pipeline/app.py`).

## Intermediate folders and files

Assuming:
- `artifacts_dir = ".../artifacts"`
- optional `save_deidentified_dir` (if empty, fallback is `artifacts/deidentification/images`)

the pipeline writes:

| Stage | Output | Path |
|---|---|---|
| SAM3 | Per-image masks | `artifacts/sam3/masks/<relative-image-path>.png` |
| SAM3 | SAM3 stage records | `artifacts/sam3/records.json` |
| Postprocess | Rotated/cropped images | `artifacts/postprocess/crops/<relative-image-path>.png` |
| Postprocess | Postprocess stage records | `artifacts/postprocess/records.json` |
| Deidentification | Deidentified images | `<save_deidentified_dir or artifacts/deidentification/images>/<relative-image-path>.png` |
| Deidentification | Deidentification records | `artifacts/deidentification/records.json` |
| Report | Final output JSON | `output_json` |

## Stage dependency chain

1. `sam3` produces masks + `sam3/records.json`
2. `postprocess` consumes `sam3` artifacts and produces crops + `postprocess/records.json`
3. `deidentification` consumes postprocess crops/records and produces deidentified images + `deidentification/records.json`
4. `report` consumes records and writes final `output_json`

Example: to run only postprocessing, set `run_mode = "postprocess"` and ensure `sam3/records.json` + `sam3/masks/...` already exist.

## Deidentification (OCR) stage

The OCR stage is modular; every piece is selected in `setups/pipeline_config.toml`:

- **Engine** (`ocr_engine`): `easyocr` (default) or `paddleocr` (PP-OCRv5, needs the
  `paddle` optional dependency group / `INSTALL_PADDLE=1` Docker build arg).
  Engine adapters live in `pipeline/ocr_engines.py`.
- **Preprocessing** (`preprocess_steps`, `pipeline/ocr_preprocess.py`): ordered chain of
  `grayscale` / `clahe` / `invert`, applied to the OCR input only — the saved output
  image is never altered. `ocr_dual_pass_invert` additionally OCRs the inverted image
  and unions detections.
- **Resolution strategy** (`resolution_strategy`, `pipeline/ocr_strategies.py`): `full`
  feeds the whole image to the engine (EasyOCR downscales anything larger than
  `easyocr_canvas_size`); `tiled` splits into overlapping tiles so large panoramics are
  never downscaled, then merges duplicate detections across tile seams.
- **Filtering**: `ocr_min_confidence` drops low-confidence detections;
  `text_filter = "short_text"` skips detections whose text is `short_text_max_chars` or
  fewer alphanumeric characters (e.g. L/R laterality markers). Classifiers live in
  `pipeline/text_classifiers.py` behind a swappable protocol.
- **Redaction** (`redaction_mode`): `outline` draws debug rectangles; `fill` paints the
  boxes black (real anonymization).

`deidentification/records.json` stores per image: `boxes` (redacted rectangles,
`[x, y, w, h]`) and `detections` (box + recognized text + confidence). With
`record_raw_detections = true` it also stores `raw_detections` (pre-filter) and
`skipped_detections` (declined by the text filter) for auditing.

## Benchmarking OCR configurations

`test/benchmark_deid.py` sweeps config variants (defined in
`setups/benchmark_matrix.toml`) over frozen postprocess artifacts, evaluates each
against the text ground truth with `test/test_accuracy.py` metrics, and writes
`summary.{json,csv,md}`:

```
docker compose run --rm pipeline python test/benchmark_deid.py \
  --matrix setups/benchmark_matrix.toml
```

Ground-truth boxes are annotated in crop coordinates: never re-run sam3/postprocess
between benchmark runs, or the ground truth no longer lines up.
