#!/usr/bin/env python3
"""Self-checking tests for the invariants the OCR-engine work depends on.

No test framework is installed, so this is a plain script: it prints one line
per check and exits non-zero on the first failure.

    python test/test_invariants.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))
sys.path.insert(0, str(REPO_ROOT / "test"))

import numpy as np  # noqa: E402

import test_accuracy as ta  # noqa: E402
from deidentifier_component import Deidentifier, DeidentifierParams  # noqa: E402
from ocr_engines import OcrDetection  # noqa: E402
from ocr_preprocess import PREPROCESS_STEPS, apply_chain  # noqa: E402
from ocr_strategies import FullImageStrategy, TiledStrategy, UpscaleStrategy  # noqa: E402
from text_classifiers import ShortTextSkipClassifier  # noqa: E402

_CHECKS = 0


def check(condition: bool, label: str) -> None:
    global _CHECKS
    _CHECKS += 1
    if not condition:
        print(f"FAIL: {label}")
        raise SystemExit(1)
    print(f"  ok: {label}")


class _StubEngine:
    """Returns one fixed quad, in the coordinate space it is handed."""

    name = "stub"

    def __init__(self, quads: list[list[list[int]]]) -> None:
        self._quads = quads
        self.seen_shapes: list[tuple[int, ...]] = []

    def detect(self, image: np.ndarray) -> list[OcrDetection]:
        self.seen_shapes.append(image.shape)
        return [OcrDetection(quad=quad, text="X", confidence=1.0) for quad in self._quads]


def test_ellipse_parity() -> None:
    """test_accuracy ports the deidentifier's ellipse test; they must agree.

    The port exists so test_accuracy stays standalone (no pipeline import),
    which is what lets benchmark_deid import it before pipeline/ is on sys.path.
    """
    params = DeidentifierParams(
        ellipse_enabled=True, center_ellipse_axes_ratio=(0.45, 0.35), ellipse_proximity_px=0.0
    )
    deidentifier = Deidentifier(engine=_StubEngine([]), params=params)
    reference = ta.EllipseParams(axis_x_ratio=0.45, axis_y_ratio=0.35, proximity_px=0.0)

    mismatches = 0
    for width, height in [(2609, 1572), (512, 248), (6343, 2713), (1659, 1659)]:
        for x1 in range(0, width, max(1, width // 11)):
            for y1 in range(0, height, max(1, height // 11)):
                rect = (float(x1), float(y1), float(x1 + 90), float(y1 + 14))
                mine = ta.touches_center_ellipse(rect, width, height, reference)
                theirs = deidentifier._touches_center_ellipse(rect, (height, width))
                mismatches += int(mine != theirs)
    check(mismatches == 0, f"ellipse implementations agree ({mismatches} mismatches)")


def test_ellipse_disable() -> None:
    params = DeidentifierParams(ellipse_enabled=False, center_ellipse_axes_ratio=(0.45, 0.35))
    deidentifier = Deidentifier(engine=_StubEngine([]), params=params)
    centred = (1000.0, 700.0, 1100.0, 720.0)
    check(
        not deidentifier._touches_center_ellipse(centred, (1572, 2609)),
        "ellipse_enabled=False disables the keep-out entirely",
    )


def test_empty_text_is_redacted() -> None:
    """Detection-only engines report text=''. Without the carve-out every one of
    their boxes would be skipped, which the benchmark would read as a precision
    win rather than as total recall failure."""
    classifier = ShortTextSkipClassifier(max_skip_chars=2)
    blank = OcrDetection(quad=[[0, 0], [9, 0], [9, 9], [0, 9]], text="", confidence=1.0)
    short = OcrDetection(quad=[[0, 0], [9, 0], [9, 9], [0, 9]], text="L", confidence=1.0)
    long = OcrDetection(quad=[[0, 0], [9, 0], [9, 9], [0, 9]], text="Rossi", confidence=1.0)
    check(classifier.should_redact(blank), "short_text filter redacts empty text")
    check(not classifier.should_redact(short), "short_text filter still skips 'L'")
    check(classifier.should_redact(long), "short_text filter redacts a real name")


def test_upscale_maps_coordinates_back() -> None:
    """UpscaleStrategy must return boxes in the ORIGINAL image's space; if it
    did not, the ellipse test, the area cap and the drawing clamp would all be
    silently wrong."""
    image = np.zeros((400, 900, 3), dtype=np.uint8)
    engine = _StubEngine([[[100, 50], [300, 50], [300, 90], [100, 90]]])
    strategy = UpscaleStrategy(FullImageStrategy(), factor=2.0)
    detections = strategy.detect(engine, image)
    check(engine.seen_shapes == [(800, 1800, 3)], "engine sees the enlarged array")
    check(
        detections[0].quad == [[50, 25], [150, 25], [150, 45], [50, 45]],
        "detections are mapped back to original coordinates",
    )
    identity = UpscaleStrategy(FullImageStrategy(), factor=1.0)
    check(
        identity.detect(_StubEngine([[[7, 8], [9, 8], [9, 10], [7, 10]]]), image)[0].quad
        == [[7, 8], [9, 8], [9, 10], [7, 10]],
        "factor 1.0 is a pass-through",
    )


def test_upscale_composes_with_tiling() -> None:
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    engine = _StubEngine([])
    UpscaleStrategy(TiledStrategy(tile_size_px=200, tile_overlap_px=50), 2.0).detect(
        engine, image
    )
    check(
        all(shape[0] <= 200 and shape[1] <= 200 for shape in engine.seen_shapes)
        and len(engine.seen_shapes) > 1,
        "upscale + tiling yields multiple tiles of the configured size",
    )


def test_preprocess_is_coordinate_preserving() -> None:
    """apply_chain feeds the engine AND fixes the shape used by the ellipse
    test, so no step may change the image dimensions."""
    image = np.random.default_rng(0).integers(0, 255, (317, 811, 3)).astype(np.uint8)
    for step in PREPROCESS_STEPS:
        out = apply_chain(image, ["grayscale", step])
        check(
            out.shape[:2] == image.shape[:2] and out.dtype == np.uint8,
            f"preprocess step '{step}' preserves shape and dtype",
        )


def test_clahe_grid_scales_with_resolution() -> None:
    from ocr_preprocess import PreprocessParams, _clahe_grid

    params = PreprocessParams(clahe_tile_px=128)
    small = _clahe_grid(np.zeros((248, 512), dtype=np.uint8), params)
    large = _clahe_grid(np.zeros((2713, 6343), dtype=np.uint8), params)
    check(large[0] > small[0] and large[1] > small[1], "CLAHE grid scales with resolution")
    fixed = _clahe_grid(np.zeros((2713, 6343), dtype=np.uint8), PreprocessParams(clahe_tile_px=0))
    check(fixed == (8, 8), "clahe_tile_px = 0 restores the fixed grid")


def test_preprocess_variants_union() -> None:
    """Each variant chain triggers its own detection pass and the results are
    unioned (the union-find merge then collapses duplicates)."""
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    engine = _StubEngine([[[10, 10], [60, 10], [60, 30], [10, 30]]])
    deidentifier = Deidentifier(
        engine=engine,
        params=DeidentifierParams(ellipse_enabled=False, padding_px=0, merge_distance_px=0),
        preprocess_variants=[None, lambda img: img],
    )
    result = deidentifier.run(image, "stub")
    check(len(engine.seen_shapes) == 2, "one detection pass per preprocess variant")
    check(len(result.raw_detections) == 2, "variant detections are unioned before merging")
    check(len(result.detections) == 1, "identical boxes merge back into one")


def test_max_box_area_ratio() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    big = [[[0, 0], [60, 0], [60, 60], [0, 60]]]  # 3600 px2 of a 10000 px2 image
    kept = Deidentifier(
        engine=_StubEngine(big),
        params=DeidentifierParams(
            ellipse_enabled=False, max_box_area_px=None, max_box_area_ratio=0.5
        ),
    ).run(image, "stub")
    dropped = Deidentifier(
        engine=_StubEngine(big),
        params=DeidentifierParams(
            ellipse_enabled=False, max_box_area_px=None, max_box_area_ratio=0.25
        ),
    ).run(image, "stub")
    check(len(kept.detections) == 1, "box under max_box_area_ratio is kept")
    check(len(dropped.detections) == 0, "box over max_box_area_ratio is dropped")


def test_union_area() -> None:
    check(abs(ta.union_area([(0, 0, 10, 10)]) - 100.0) < 1e-9, "union_area of one rect")
    check(
        abs(ta.union_area([(0, 0, 10, 10), (5, 0, 15, 10)]) - 150.0) < 1e-9,
        "union_area does not double-count the overlap",
    )
    check(
        abs(ta.union_area([(0, 0, 10, 10), (0, 0, 10, 10)]) - 100.0) < 1e-9,
        "union_area of duplicates",
    )
    check(
        abs(ta.union_area([(-5, -5, 5, 5)], clip=(0, 0, 10, 10)) - 25.0) < 1e-9,
        "union_area honours the clip rect",
    )


def test_union_coverage() -> None:
    """A detector that splits one text line into two adjacent boxes still hides
    every PHI pixel; max-over-single-prediction coverage says otherwise."""
    gt_box = (0.0, 0.0, 100.0, 10.0)
    halves = [(0.0, 0.0, 50.0, 10.0), (50.0, 0.0, 100.0, 10.0)]
    single = ta.gt_best_coverage_ratio(gt_box, halves, 0.0)
    union = ta.gt_union_coverage_ratio(gt_box, halves, 0.0)
    check(abs(single - 0.5) < 1e-9, "max-over-single coverage sees only half")
    check(abs(union - 1.0) < 1e-9, "union coverage sees the whole box")

    metrics = ta.evaluate({"i": halves}, {"i": [gt_box]}, margin_px=0.0)
    check(metrics["gt_full_coverage_rate_union"] == 1.0, "union metric counts it covered")
    check(metrics["image_level_recall"] == 1.0, "image counts as fully protected")


def test_degenerate_gt_filter() -> None:
    gt = {"a": [(0.0, 0.0, 27.0, 0.029), (0.0, 0.0, 90.0, 14.0)]}
    kept, dropped = ta.filter_degenerate_gt(gt, min_side_px=1.0)
    check(len(kept["a"]) == 1 and len(dropped) == 1, "sub-pixel GT box is filtered out")


def test_vlm_grounding_parser() -> None:
    """DeepSeek-style grounding: <|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>
    with coordinates normalised to 0-999. Getting the scale wrong puts every
    redaction in the wrong place, which for this pipeline means leaked PHI."""
    from ocr_engines_vlm import parse_grounding_detections, parse_json_detections

    content = "<|ref|>text<|/ref|><|det|>[[100, 50, 400, 150]]<|/det|>"
    dets = parse_grounding_detections(content, 2000, 1000, "normalized_999")
    check(len(dets) == 1, "grounding tags are parsed")
    xs = [pt[0] for pt in dets[0].quad]
    ys = [pt[1] for pt in dets[0].quad]
    check(
        abs(min(xs) - round(100 / 999 * 2000)) <= 1
        and abs(max(xs) - round(400 / 999 * 2000)) <= 1,
        "0-999 coordinates are scaled to image width",
    )
    check(
        abs(min(ys) - round(50 / 999 * 1000)) <= 1
        and abs(max(ys) - round(150 / 999 * 1000)) <= 1,
        "0-999 coordinates are scaled to image height",
    )
    check(dets[0].text == "text", "the ref label is kept as the text")

    # Same numbers read as pixels must NOT be rescaled.
    pixel = parse_grounding_detections(content, 2000, 1000, "pixel")
    check(
        min(pt[0] for pt in pixel[0].quad) == 100
        and max(pt[0] for pt in pixel[0].quad) == 400,
        "pixel coordinate space passes numbers through unchanged",
    )

    multi = parse_grounding_detections(
        "<|ref|>a<|/ref|><|det|>[[0,0,10,10],[20,20,30,30]]<|/det|>", 999, 999, "normalized_999"
    )
    check(len(multi) == 2, "several boxes in one det block are all parsed")

    check(parse_grounding_detections("no tags here", 100, 100, "pixel") == [],
          "a reply with no grounding tags yields no detections")
    check(parse_grounding_detections(
        "<|ref|>a<|/ref|><|det|>[[not json]]<|/det|>", 100, 100, "pixel") == [],
        "malformed coordinates are skipped, not raised")

    payload = '```json\n{"detections": [{"text": "Rossi", "bbox": [10, 20, 30, 40]}]}\n```'
    js = parse_json_detections(payload, 100, 100, "pixel")
    check(len(js) == 1 and js[0].text == "Rossi", "fenced JSON replies are parsed")
    check(
        min(pt[0] for pt in js[0].quad) == 10
        and max(pt[0] for pt in js[0].quad) == 40,
        "JSON bbox is [x, y, w, h], so x2 = x + w",
    )
    check(parse_json_detections("garbage", 100, 100, "pixel") == [],
          "an unparseable JSON reply yields no detections")


def test_vlm_engine_maps_back_and_dilates() -> None:
    """The engine must return boxes in the space of the array it was handed,
    even after downscaling for the API, and must apply its own dilation on top
    of the global padding to absorb grounding drift."""
    from ocr_engines_vlm import VlmEngine, VlmParams

    class _Resp:
        def __init__(self, text):
            self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]

    class _Client:
        def __init__(self, text):
            self.sent = {}
            outer = self

            class _Completions:
                def create(self, **kwargs):
                    outer.sent = kwargs
                    return _Resp(text)

            self.chat = type("Chat", (), {"completions": _Completions()})()

    # 4000px wide image, capped to 1000px for the request -> scale 0.25.
    client = _Client('{"detections": [{"text": "X", "bbox": [100, 50, 100, 50]}]}')
    engine = VlmEngine(client, VlmParams(max_side_px=1000, box_dilate_px=5, coord_space="pixel"))
    image = np.zeros((2000, 4000, 3), dtype=np.uint8)
    dets = engine.detect(image)
    xs = [pt[0] for pt in dets[0].quad]
    # 100px in the 1000px-wide request -> 400px in the original, minus dilation.
    check(min(xs) == 400 - 5, "boxes are mapped back through the API downscale")
    check(max(xs) == 800 + 5, "dilation widens the box on both sides")
    check(client.sent.get("temperature") == 0, "requests are deterministic")


def test_blank_gt_filter(tmp_dir: Path = Path("/tmp")) -> None:
    """A GT box over uniform pixels is unfindable by any OCR, so scoring against
    it measures nothing. The test looks only at the image, never at a
    prediction, so it is not "the model missed it, therefore ignore it"."""
    import tempfile

    import cv2

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "img.png"
        image = np.zeros((100, 200), dtype=np.uint8)
        image[10:20, 10:60] = 255  # a bright patch: real content
        cv2.imwrite(str(path), image)

        ground_truth = {
            "img": [
                (10.0, 10.0, 60.0, 20.0),  # over the patch -> kept
                (100.0, 50.0, 160.0, 70.0),  # over flat black -> dropped
            ]
        }
        kept, dropped = ta.filter_blank_gt(ground_truth, {"img": path})
        check(len(kept["img"]) == 1, "GT over real content is kept")
        check(len(dropped) == 1, "GT over uniform pixels is dropped")
        check(
            ta.filter_blank_gt(ground_truth, {})[0]["img"] == ground_truth["img"],
            "no image available -> nothing is dropped",
        )


def test_coverage_vs_iou() -> None:
    """A fully covered GT box inside a much larger redaction box is a perfect
    anonymization result but a sub-threshold IoU. The coverage metric must see
    it; recall_iou is expected not to."""
    gt = {"img": [(100.0, 100.0, 200.0, 115.0)]}
    preds = {"img": [(80.0, 70.0, 260.0, 160.0)]}
    metrics = ta.evaluate(preds, gt, margin_px=5.0, iou_threshold=0.5)
    check(metrics["gt_full_coverage_rate"] == 1.0, "coverage sees the fully covered box")
    check(metrics["recall_covered"] == 1.0, "recall_covered sees the fully covered box")
    check(metrics["recall_iou"] == 0.0, "recall_iou penalises the over-redaction")


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        print(f"{test.__name__}:")
        test()
    print(f"\nAll {_CHECKS} checks passed across {len(tests)} tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
