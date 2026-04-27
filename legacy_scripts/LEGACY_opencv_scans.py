import cv2
import numpy as np
import argparse
from pathlib import Path
import sys


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def get_binary_map(img, block_size=201, c_val=2):
    """Converts image to grayscale and applies adaptive thresholding."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # We use THRESH_BINARY_INV so the rectangle is white (foreground)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c_val,
    )


def find_largest_rotated_rect(binary_img):
    """Finds the largest contour and returns its minAreaRect properties."""
    contours, _ = cv2.findContours(
        binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    return cv2.minAreaRect(largest_contour)


def rotate_and_straighten(img, rect):
    """Applies Affine Transform to level the rotated rectangle."""
    center, size, angle = rect
    width, height = size

    # Angle Correction: Ensures the rectangle stays in a logical orientation
    if width < height:
        angle += 90
        width, height = height, width

    # Get the 2x3 Rotation Matrix
    # The math: R = [[cosθ, sinθ, (1-cosθ)cx - sinθcy], [-sinθ, cosθ, sinθcx + (1-cosθ)cy]]
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rows, cols = img.shape[:2]
    straightened = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_CUBIC)

    return straightened, (width, height)


def visualize(original, final_crop):
    # 6. Prepare for Visualization (The "hstack" fix)
    # Ensure both are BGR so they can be stacked
    if len(original.shape) == 2:
        original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        original_bgr = original.copy()

    if len(final_crop.shape) == 2:
        crop_bgr = cv2.cvtColor(final_crop, cv2.COLOR_GRAY2BGR)
    else:
        crop_bgr = final_crop.copy()

    # Resize crop to match original height while keeping aspect ratio
    h_orig, w_orig = original_bgr.shape[:2]
    aspect_ratio = crop_bgr.shape[1] / crop_bgr.shape[0]
    new_width = int(h_orig * aspect_ratio)
    crop_resized = cv2.resize(
        crop_bgr, (new_width, h_orig), interpolation=cv2.INTER_CUBIC
    )

    # 7. Create Separator and Combine
    separator = np.zeros((h_orig, 10, 3), dtype=np.uint8)
    separator[:] = [0, 0, 255]  # Red line

    combined = np.hstack([original_bgr, separator, crop_resized])

    # 8. Adaptive Display Scaling
    # Scale down if it's too big for the monitor, but keep aspect ratio
    screen_res = (1500, 844)  # Target max resolution
    scale_width = screen_res[0] / combined.shape[1]
    scale_height = screen_res[1] / combined.shape[0]
    scale = min(scale_width, scale_height, 1.0)  # Don't upscale

    display_w = int(combined.shape[1] * scale)
    display_h = int(combined.shape[0] * scale)

    display = cv2.resize(combined, (display_w, display_h))

    # Show Results
    cv2.imshow("Comparison: Original vs Straightened Crop", display)

    cv2.moveWindow(
        "Comparison: Original vs Straightened Crop", 200, 400
    )  # Position window

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_pipeline(image_path):
    """Orchestrates the image processing steps with a robust display."""
    # 1. Load
    original = cv2.imread(image_path)
    if original is None:
        print("Error: Image not found")
        return None

    # 2. Binarize (assuming get_binary_map is defined as before)
    binary = get_binary_map(original, block_size=201, c_val=2)

    # 3. Analyze Geometry (assuming find_largest_rotated_rect is defined)
    rect = find_largest_rotated_rect(binary)
    if rect is None:
        print("Error: No rectangle detected")
        return None

    # 4. Transform (assuming rotate_and_straighten returns result and final_size)
    # Ensure rotate_and_straighten uses cv2.getRotationMatrix2D(rect[0], ...)
    result, final_size = rotate_and_straighten(original, rect)

    # 5. Crop
    # rect[0] is the (x, y) center. Since we rotated around THIS point,
    # it remains the center in the 'result' image.
    crop_w = max(1, int(round(final_size[0])))
    crop_h = max(1, int(round(final_size[1])))
    final_crop = cv2.getRectSubPix(result, (crop_w, crop_h), rect[0])

    return final_crop


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_EXTS


def _iter_input_images(input_path: Path) -> tuple[list[Path], Path | None]:
    """Returns (files, input_root_dir_if_any)."""
    if input_path.is_file():
        return [input_path], None
    if not input_path.is_dir():
        return [], None

    files = [
        p for p in sorted(input_path.iterdir()) if p.is_file() and _is_image_file(p)
    ]
    return files, input_path


def _resolve_output_path(
    *,
    input_file: Path,
    input_root: Path | None,
    output_path: Path,
) -> Path:
    """Resolves output path for a given input file.

    - If input_root is None (single input file):
        - output_path as existing dir OR dir-like path => write into that dir with same name
        - otherwise treat output_path as a file
    - If input_root is a dir (batch):
        - output_path must be a dir (created if missing); preserve relative file name
    """
    if input_root is None:
        if output_path.exists() and output_path.is_dir():
            return output_path / input_file.name
        # Heuristic: if user passes something ending with a path separator, treat as directory
        if str(output_path).endswith(("/", "\\")):
            return output_path / input_file.name
        # If user didn't specify a known image extension, treat -o as an output folder.
        if output_path.suffix.lower() not in _IMAGE_EXTS:
            return output_path / input_file.name
        return output_path

    # Batch mode
    rel = input_file.relative_to(input_root)
    return output_path / rel


# Execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Straighten and crop scanned images using OpenCV"
    )
    parser.add_argument("-i", "--input", required=True, help="Input file or folder")
    parser.add_argument("-o", "--output", required=True, help="Output file or folder")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    input_files, input_root = _iter_input_images(input_path)
    if not input_files:
        print(
            "Error: input path is not a file/folder with supported images",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if input_root is not None:
        # Batch mode: output must be a directory (we create it if missing)
        if output_path.exists() and output_path.is_file():
            print(
                "Error: when input is a folder, output must be a folder",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if output_path.suffix.lower() in _IMAGE_EXTS:
            print(
                "Error: when input is a folder, output must be a folder (not a file path)",
                file=sys.stderr,
            )
            raise SystemExit(2)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        # Single-file mode: if output is a directory (or looks like one), ensure it exists
        output_is_dir = (
            (output_path.exists() and output_path.is_dir())
            or str(output_path).endswith(("/", "\\"))
            or (output_path.suffix.lower() not in _IMAGE_EXTS)
        )
        if output_is_dir:
            output_path.mkdir(parents=True, exist_ok=True)

    failures = 0
    for input_file in input_files:
        crop = run_pipeline(str(input_file))
        if crop is None:
            failures += 1
            continue

        out_file = _resolve_output_path(
            input_file=input_file, input_root=input_root, output_path=output_path
        )
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if out_file.suffix == "":
            # Default to PNG if user didn't provide an extension
            out_file = out_file.with_suffix(".png")

        ok = cv2.imwrite(str(out_file), crop)
        if not ok:
            print(f"Error: failed writing output: {out_file}", file=sys.stderr)
            failures += 1

    if failures:
        raise SystemExit(1)
