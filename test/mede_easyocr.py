import pydicom
from pydicom.encaps import encapsulate
import cv2
import numpy as np
import os
import easyocr
import json
from glob import glob
import nibabel as nib
from PIL import Image
from pathlib import Path
import logging
from collections import defaultdict
from typing import Optional
import time


class TextRemoval:
    """
    Class for performing text removal on images using EasyOCR (CRAFT Text Detection).

    Attributes:
        output_path (str): Path to save the output images.
        verbose (bool): If True, enables verbose logging.
        reader (easyocr.Reader): The deep learning OCR model initialized in memory.
        interactive (bool): If True, enables interactive refinement of the output images.

    Methods:
        predict: Apply text removal algorithm to an image.
        __call__: Apply text removal to a directory of images.
    """

    def __init__(
        self,
        output_path: str = None,
        verbose: bool = False,
        langs: list = ["en", "it"],
        interactive: bool = False,
        save_labels: bool = True,
    ) -> None:
        self.output_path = (
            output_path if output_path is not None else "./text_removed_images"
        )
        self.interactive = interactive
        if verbose:
            logging.getLogger().setLevel(logging.INFO)
            logging.info(f"Saving text removed images to {self.output_path}")

        os.makedirs(self.output_path, exist_ok=True)

        self.save_labels = save_labels
        if save_labels:
            self.labels = {}
            self.labels_path = os.path.join(self.output_path, "labels.json")
            if verbose:
                logging.info(f"Saving detected text labels to {self.labels_path}")

        # Initialize the EasyOCR reader once to keep the model in memory.
        # It will automatically use a GPU if CUDA is available.
        self.reader = easyocr.Reader(langs, gpu=True)

    def predict(
        self, img: np.array, img_orig: np.array = None, img_name: str = None
    ) -> np.array:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        target_img = img_orig if img_orig is not None else img.copy()

        # --- PASS 1: Normal Scale ---
        results_normal = self.reader.readtext(img)

        # --- PASS 2: Downscaled for detection of big characters / numbers ---
        results_shrunk = self.reader.readtext(img, mag_ratio=0.1, text_threshold=0.5)

        # Combine results
        all_results = results_normal + results_shrunk

        logging.info(
            f"Detected {len(all_results)} text regions in {img_name if img_name else 'the image'}."
        )

        # Save detected text labels if enabled
        if self.save_labels:
            if img_name is None:
                raise ValueError(
                    "img_name must be provided when save_labels is enabled."
                )

            self.labels[img_name] = []
            for bbox, text, prob in all_results:
                self.labels[img_name].append(
                    {
                        "bbox": [[int(point[0]), int(point[1])] for point in bbox],
                        "text": str(text),
                        "confidence": float(prob),
                    }
                )
            with open(self.labels_path, "w") as f:
                json.dump(self.labels, f, indent=4)

        # Draw rectangles
        for bbox, text, prob in all_results:
            xs = [int(point[0]) for point in bbox]
            ys = [int(point[1]) for point in bbox]

            left, right = min(xs), max(xs)
            top, bottom = min(ys), max(ys)

            padding = 3
            target_img = cv2.rectangle(
                target_img,
                (max(0, left - padding), max(0, top - padding)),
                (right + padding, bottom + padding),
                (0, 0, 255),
                5,
            )

        return target_img

    def refine_image(
        self, img: np.array, window_name: str = "Interactive Refinement"
    ) -> np.array:
        """
        Opens an interactive OpenCV window to let the user manually draw white rectangles
        over remaining text/artifacts.

        Args:
            img (np.array): The image to refine.
            window_name (str): The name of the OpenCV window.

        Returns:
            np.array: The manually refined image.
        """
        # Create copies so we can reset if the user makes a mistake
        original_clone = img.copy()
        display_img = img.copy()

        # State variables for mouse tracking
        drawing = False
        ix, iy = -1, -1

        def draw_rectangle(event, x, y, flags, param):
            nonlocal ix, iy, drawing, img, display_img

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix, iy = x, y

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    # Draw on a temporary copy so we see the box expanding as we drag
                    display_img = img.copy()
                    cv2.rectangle(display_img, (ix, iy), (x, y), (255, 255, 255), -1)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                # Commit the rectangle to the actual image
                cv2.rectangle(img, (ix, iy), (x, y), (255, 255, 255), -1)
                display_img = img.copy()

        # Set up OpenCV window and attach mouse listener
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # WINDOW_NORMAL allows resizing
        cv2.setMouseCallback(window_name, draw_rectangle)

        print(f"\n--- Refinement Mode: {window_name} ---")
        print("1. Click and drag to draw white boxes over artifacts.")
        print("2. Press 'r' to RESET the image if you make a mistake.")
        print("3. Press 'Enter' or 'Space' to SAVE and continue to the next image.")

        while True:
            cv2.imshow(window_name, display_img)
            key = cv2.waitKey(1) & 0xFF

            # Press Enter (13) or Space (32) to confirm and exit
            if key in [13, 32]:
                break
            # Press 'r' to reset
            elif key == ord("r"):
                print("Image reset.")
                img = original_clone.copy()
                display_img = original_clone.copy()

        cv2.destroyWindow(window_name)
        return img

    def __call__(self, directory: str) -> None:
        """
        Apply text removal to a directory of images.

        Args:
            directory (str): Path to the directory containing the images.

        Returns:
            None
        """
        if os.path.isdir(directory):
            files = glob(os.path.join(directory, "**", "*"), recursive=True)
        else:
            files = [directory]

        for filepath in files:
            # Skip directories
            if os.path.isdir(filepath):
                continue

            img_orig = None
            file_ending = filepath.split(".")[-1].lower()

            match file_ending:
                case "png" | "jpg":
                    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                    img_was_greyscale = len(img.shape) == 2 or (
                        len(img.shape) == 3 and img.shape[2] == 1
                    )
                    base_fn = filepath[:-4]
                case "jpeg":
                    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                    img_was_greyscale = len(img.shape) == 2 or (
                        len(img.shape) == 3 and img.shape[2] == 1
                    )
                    base_fn = filepath[:-5]
                case "dcm":
                    dcm = pydicom.dcmread(filepath, force=True)
                    img_orig = pydicom.pixel_array(dcm)
                    img_was_greyscale = len(img_orig.shape) == 2 or (
                        len(img_orig.shape) == 3 and img_orig.shape[2] == 1
                    )
                    img = np.array(
                        Image.fromarray(img_orig).convert(
                            "L" if img_was_greyscale else "RGB"
                        )
                    )
                    base_fn = filepath[:-4]
                case "nii":
                    nifti = nib.load(filepath)
                    nii_data = nifti.get_fdata().squeeze()
                    img_was_greyscale = len(nii_data.shape) == 2 or (
                        len(nii_data.shape) == 3 and nii_data.shape[2] == 1
                    )
                    img = np.array(
                        Image.fromarray(nii_data).convert(
                            "L" if img_was_greyscale else "RGB"
                        )
                    )
                    base_fn = filepath[:-4]
                case "gz":
                    nifti = nib.load(filepath)
                    nii_data = nifti.get_fdata().squeeze()
                    img_was_greyscale = len(nii_data.shape) == 2 or (
                        len(nii_data.shape) == 3 and nii_data.shape[2] == 1
                    )
                    img = np.array(
                        Image.fromarray(nii_data).convert(
                            "L" if img_was_greyscale else "RGB"
                        )
                    )
                    base_fn = filepath[:-7]
                case _:
                    continue

            img = self.predict(
                img=img,
                img_orig=img_orig
                if "img_orig" in locals() and img_orig is not None
                else None,
                img_name=Path(filepath).name,
            )

            # Optional manual refinement step for any remaining text/artifacts
            if self.interactive:
                img = self.refine_image(
                    img, window_name=f"Refining: {Path(base_fn).name}"
                )

            # Convert back to greyscale if the input was greyscale
            if img_was_greyscale and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            _output_path = os.path.join(
                self.output_path, f"{Path(base_fn).name}_text_removed"
            )

            match file_ending:
                case "png":
                    cv2.imwrite(f"{_output_path}.png", img)
                case "jpg" | "jpeg":
                    cv2.imwrite(f"{_output_path}.jpg", img)
                case "dcm":
                    # Check if the pixel data is compressed
                    if (
                        hasattr(dcm.file_meta, "TransferSyntaxUID")
                        and dcm.file_meta.TransferSyntaxUID.is_compressed
                    ):
                        # Re-encapsulate the pixel data if compression is required
                        dcm.PixelData = encapsulate([img.tobytes()])
                        dcm.file_meta.TransferSyntaxUID = (
                            pydicom.uid.ExplicitVRLittleEndian
                        )
                    else:
                        dcm.PixelData = img.tobytes()
                    dcm.save_as(f"{_output_path}.dcm")
                case "nii":
                    nifti = nib.Nifti1Image(img, nifti.affine)
                    nib.save(nifti, f"{_output_path}.nii")
                case "gz":
                    nifti = nib.Nifti1Image(img, nifti.affine)
                    nib.save(nifti, f"{_output_path}.nii.gz")


class MergedFilteredTextRemoval(TextRemoval):
    """
    Text removal with post-processing on OCR detections:
    1) merge close boxes into one enclosing rectangle
    2) remove boxes that are too large
    3) remove boxes that touch a center ellipse
    """

    def __init__(
        self,
        output_path: str = None,
        verbose: bool = False,
        langs: list = ["en", "it"],
        interactive: bool = False,
        save_labels: bool = True,
        merge_distance_px: int = 10,
        max_box_area_px: Optional[int] = 120000,
        center_ellipse_axes_ratio: tuple[float, float] = (0.35, 0.25),
        ellipse_proximity_px: float = 0.0,
    ) -> None:
        super().__init__(
            output_path=output_path,
            verbose=verbose,
            langs=langs,
            interactive=interactive,
            save_labels=save_labels,
        )
        if merge_distance_px < 0:
            raise ValueError("merge_distance_px must be >= 0.")
        if max_box_area_px is not None and max_box_area_px <= 0:
            raise ValueError("max_box_area_px must be > 0 when provided.")
        if (
            center_ellipse_axes_ratio[0] <= 0
            or center_ellipse_axes_ratio[1] <= 0
            or center_ellipse_axes_ratio[0] > 1
            or center_ellipse_axes_ratio[1] > 1
        ):
            raise ValueError("center_ellipse_axes_ratio values must be in (0, 1].")
        if ellipse_proximity_px < 0:
            raise ValueError("ellipse_proximity_px must be >= 0.")

        self.merge_distance_px = merge_distance_px
        self.max_box_area_px = max_box_area_px
        self.center_ellipse_axes_ratio = center_ellipse_axes_ratio
        self.ellipse_proximity_px = ellipse_proximity_px

    @staticmethod
    def _bbox_to_rect(bbox: list[list[float]]) -> tuple[float, float, float, float]:
        xs = [float(point[0]) for point in bbox]
        ys = [float(point[1]) for point in bbox]
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def _rect_to_bbox(rect: tuple[float, float, float, float]) -> list[list[int]]:
        x1, y1, x2, y2 = rect
        return [
            [int(round(x1)), int(round(y1))],
            [int(round(x2)), int(round(y1))],
            [int(round(x2)), int(round(y2))],
            [int(round(x1)), int(round(y2))],
        ]

    @staticmethod
    def _rect_area(rect: tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = rect
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @staticmethod
    def _rects_are_close(
        rect_a: tuple[float, float, float, float],
        rect_b: tuple[float, float, float, float],
        distance_px: float,
    ) -> bool:
        ax1, ay1, ax2, ay2 = rect_a
        bx1, by1, bx2, by2 = rect_b
        return not (
            ax2 + distance_px < bx1
            or bx2 + distance_px < ax1
            or ay2 + distance_px < by1
            or by2 + distance_px < ay1
        )

    def _merge_close_detections(self, detections: list[tuple]) -> list[tuple]:
        if not detections:
            return []

        rects = [self._bbox_to_rect(bbox) for bbox, _, _ in detections]
        parent = list(range(len(rects)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                if self._rects_are_close(rects[i], rects[j], self.merge_distance_px):
                    union(i, j)

        groups: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(rects)):
            groups[find(idx)].append(idx)

        merged: list[tuple] = []
        for group_indices in groups.values():
            x1 = min(rects[i][0] for i in group_indices)
            y1 = min(rects[i][1] for i in group_indices)
            x2 = max(rects[i][2] for i in group_indices)
            y2 = max(rects[i][3] for i in group_indices)

            texts = [
                str(detections[i][1])
                for i in group_indices
                if str(detections[i][1]).strip()
            ]
            merged_text = " | ".join(texts)
            merged_confidence = max(float(detections[i][2]) for i in group_indices)
            merged.append(
                (self._rect_to_bbox((x1, y1, x2, y2)), merged_text, merged_confidence)
            )

        merged.sort(key=lambda det: (det[0][0][1], det[0][0][0]))
        return merged

    def _touches_center_ellipse(
        self, rect: tuple[float, float, float, float], img_shape: tuple[int, ...]
    ) -> bool:
        img_h, img_w = img_shape[:2]
        cx, cy = img_w / 2.0, img_h / 2.0
        axis_x = max(
            1.0, (img_w * self.center_ellipse_axes_ratio[0]) + self.ellipse_proximity_px
        )
        axis_y = max(
            1.0, (img_h * self.center_ellipse_axes_ratio[1]) + self.ellipse_proximity_px
        )

        x1, y1, x2, y2 = rect
        nearest_x = min(max(cx, x1), x2)
        nearest_y = min(max(cy, y1), y2)
        ellipse_eq = ((nearest_x - cx) / axis_x) ** 2 + ((nearest_y - cy) / axis_y) ** 2
        return ellipse_eq <= 1.0

    def _filter_detections(
        self, detections: list[tuple], img_shape: tuple[int, ...]
    ) -> list[tuple]:
        filtered: list[tuple] = []
        for bbox, text, prob in detections:
            rect = self._bbox_to_rect(bbox)
            if (
                self.max_box_area_px is not None
                and self._rect_area(rect) > self.max_box_area_px
            ):
                continue
            if self._touches_center_ellipse(rect, img_shape):
                continue
            filtered.append((bbox, text, prob))
        return filtered

    def predict(
        self, img: np.array, img_orig: np.array = None, img_name: str = None
    ) -> np.array:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        target_img = img_orig if img_orig is not None else img.copy()

        start = time.time()

        results_normal = self.reader.readtext(img)
        results_shrunk = self.reader.readtext(img, mag_ratio=0.1, text_threshold=0.5)
        all_results = results_normal + results_shrunk

        filtered_single_results = self._filter_detections(all_results, img.shape)
        filtered_results = self._merge_close_detections(filtered_single_results)

        end = time.time()

        logging.info(
            "Detected %d raw regions, kept %d after single-box filtering, merged to %d in %s. Processing %s.",
            len(all_results),
            len(filtered_single_results),
            len(filtered_results),
            img_name if img_name else "the image",
            f"{end - start:.2f}s",
        )

        if self.save_labels:
            if img_name is None:
                raise ValueError(
                    "img_name must be provided when save_labels is enabled."
                )

            self.labels[img_name] = []
            for bbox, text, prob in filtered_results:
                self.labels[img_name].append(
                    {
                        "bbox": [[int(point[0]), int(point[1])] for point in bbox],
                        "text": str(text),
                        "confidence": float(prob),
                    }
                )
            with open(self.labels_path, "w", encoding="utf-8") as f:
                json.dump(self.labels, f, indent=4)

        for bbox, text, prob in filtered_results:
            xs = [int(point[0]) for point in bbox]
            ys = [int(point[1]) for point in bbox]
            left, right = min(xs), max(xs)
            top, bottom = min(ys), max(ys)

            padding = 3
            target_img = cv2.rectangle(
                target_img,
                (max(0, left - padding), max(0, top - padding)),
                (right + padding, bottom + padding),
                (0, 0, 255),
                5,
            )

        return target_img


if __name__ == "__main__":
    # Example usage:
    text_removal = MergedFilteredTextRemoval(
        output_path="./imgs/mede-deidentify",
        verbose=True,
        interactive=False,
        save_labels=True,
        merge_distance_px=10,
        max_box_area_px=120000,
    )
    text_removal("./imgs/output/postprocess/images")
