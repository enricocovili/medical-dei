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
