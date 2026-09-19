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

- **Engine** (`ocr_engine`), selected from the `OCR_ENGINES` registry in
  `pipeline/app.py`:

  | value | what it is | install |
  |---|---|---|
  | `easyocr` | CRAFT detector + CRNN recogniser (default) | core |
  | `paddleocr` | PP-OCRv5 via paddlepaddle | `--extra paddle` |
  | `rapidocr` | the same PP-OCR DB detector over ONNX Runtime | `--extra rapidocr` |
  | `onnxtr` | DBNet / ResNet-50 over ONNX Runtime | `--extra onnxtr` |
  | `surya` | 650M torch detector — needs its own venv, see below | — |
  | `vlm` | any OpenAI-compatible endpoint (hosted, vLLM, SGLang) | core |
  | `ensemble` | union of several of the above | — |

  A new backend needs only `name` and `detect(np.ndarray) -> list[OcrDetection]`
  (`pipeline/contracts.py:OcrEngine`), plus an entry in `OCR_ENGINES` and a
  `<engine>_*` config prefix. Quad coordinates must be in the pixel space of the
  array passed in.

  `rapidocr` is preferred over `paddleocr` on GPU hosts: it reaches the same DB
  detector without paddlepaddle, which ships no sm_120 wheels and so cannot use
  a Blackwell card at all. It also exposes the DB thresholds
  (`rapidocr_box_thresh`, `rapidocr_thresh`, `rapidocr_unclip_ratio`,
  `rapidocr_limit_side_len`) that the paddle adapter hard-codes — `unclip_ratio`
  inflates every detected polygon, which is the cheapest recall lever available.

  Surya is implemented but not packaged: `surya-ocr` pins `pillow<11` and
  `opencv-python-headless==4.11.0.86` exactly, which conflicts with this
  project's core dependencies. Install it into a separate venv
  (`INSTALL_SURYA=1` builds one at `/opt/venv-surya`).

  VLM backends run behind HTTP rather than in-process, because their pins are
  mutually unsatisfiable — DeepSeek-OCR-2 wants `torch==2.6.0` (which predates
  sm_120 and would drag SAM3 backwards) and GLM-OCR wants
  `transformers>=5.3.0`. See `docker-compose.yml` for one service per model,
  and **characterise the output format with `test/probe_vlm_grounding.py`
  before trusting the boxes**: grounding coordinates are known to drift, and
  for redaction a box 20px off leaks PHI.

- **Preprocessing** (`preprocess_steps`, `pipeline/ocr_preprocess.py`): an ordered
  chain of `grayscale`, `clahe`, `invert`, `tophat`, `blackhat`,
  `percentile_stretch`, `gamma`, `unsharp`, applied to the OCR input only — the
  saved output image is never altered. Every step must preserve the image
  dimensions, because its output both feeds the engine and fixes the shape the
  centre-ellipse test measures against.

  `tophat`/`blackhat` target this dataset's failure mode directly: burned-in text
  is a thin additive overlay on a slowly-varying anatomical background, which a
  morphological residual separates far more cleanly than CLAHE, since CLAHE
  amplifies anatomy noise along with the text.

  CLAHE tiles are sized in pixels (`clahe_tile_px`) and the grid derived per
  image. The old fixed 8x8 grid meant 64px tiles on the smallest image and 793px
  on the largest — effectively global equalisation against a 14px-tall median
  text box. Set `clahe_tile_px = 0` to restore the fixed grid.

  `preprocess_variants` runs several chains and unions the detections: one model,
  N passes, which is much cheaper per unit of recall than adding another model.
  `ocr_dual_pass_invert` is the special case `[[], ["invert"]]`.

- **Resolution strategy** (`resolution_strategy`, `pipeline/ocr_strategies.py`): `full`
  feeds the whole image to the engine (EasyOCR downscales anything larger than
  `easyocr_canvas_size`); `tiled` splits into overlapping tiles so large panoramics are
  never downscaled, then merges duplicate detections across tile seams.
  `ocr_upscale_factor` wraps either one, enlarging the input and mapping the boxes
  back — so the enlarged array never reaches the deidentifier and the ellipse test,
  area cap and drawing clamp all stay in original pixels. With `easyocr`, raise
  `easyocr_canvas_size` alongside it or the engine downscales the gain straight
  back and the experiment returns a null result for the wrong reason.

- **Filtering**: `ocr_min_confidence` drops low-confidence detections. For an
  `ensemble`, leave it at 0 and use `ensemble_min_confidences` instead: the members'
  scores are a recognition softmax, a detector box score and a constant, which are
  not comparable, so one global threshold mutes whichever member scores
  conservatively. `text_filter = "short_text"` skips detections whose text is
  `short_text_max_chars` or fewer alphanumeric characters (e.g. L/R laterality
  markers) — but never skips empty text, since detection-only engines report
  `text=""` and skipping those would discard every box they find.

  `ellipse_enabled` and `max_box_area_ratio` control the geometric filters. Both
  cost real recall at their reference settings (8/81 panoramic and 16+13/136
  teleradiography GT boxes respectively); see CLAUDE.md.

- **Redaction** (`redaction_mode`): `fill` paints the boxes black (real
  anonymization, and the default); `outline` only draws debug rectangles and
  leaves the text readable underneath.

`deidentification/records.json` stores per image: `boxes` (redacted rectangles,
`[x, y, w, h]`) and `detections` (box + recognized text + confidence). With
`record_raw_detections = true` it also stores `raw_detections` (pre-filter) and
`skipped_detections` (declined by the text filter) for auditing.

## Benchmarking OCR configurations

`test/benchmark_deid.py` sweeps `[[variant]]` config overrides across the
`[[dataset]]` entries of `setups/benchmark_matrix.toml`, scores each with the
`test/test_accuracy.py` metrics, and writes `summary.{json,csv,md}` per dataset
plus a combined `summary_all.csv` and a per-variant `metrics.json`.

```
# everything, both modalities
docker compose run --rm pipeline python test/benchmark_deid.py

# one dataset, one variant, first 12 images — a CPU smoke run
python test/benchmark_deid.py --datasets panoramic --only baseline_easyocr --limit 12
```

The ground truth is annotated directly on each dataset's `images_dir`, so the
sweep needs **neither a GPU nor a SAM3/postprocess rerun**: the postprocess
records are synthesised from a directory listing. Do not point it at
regenerated crops — the postprocess stage rotates and crops, which invalidates
the annotations.

Detections are memoised on the exact pixels handed to the engine. Most variants
change only post-filtering and feed byte-identical arrays, so they replay in
milliseconds rather than re-running OCR; a variant that changes preprocessing,
tiling or the engine misses the cache, which is correct.

`test/analyze_engine_overlap.py` reads the records a sweep already wrote and
reports, per variant, how many ground-truth boxes it finds that the others do
not. That is the number to choose an ensemble member on: decorrelated errors
matter more than standalone recall.
