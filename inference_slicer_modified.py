# Implementation of batched sliced inference by LinasKo from https://github.com/roboflow/supervision/pull/1239
# Modified by National Archives of Finland
# Original license for this file in supervision library:
# 
# MIT License
# Copyright (c) 2022 Roboflow
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import annotations

import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections
from supervision.detection.utils.boxes import move_boxes, move_oriented_boxes
from supervision.detection.utils.iou_and_nms import OverlapFilter, OverlapMetric
from supervision.detection.utils.masks import move_masks
from supervision.draw.base import ImageType
from supervision.utils.image import crop_image, get_image_resolution_wh
from supervision.utils.internal import SupervisionWarnings
from supervision.utils.iterables import create_batches

from supervision.detection.utils.iou_and_nms import mask_non_max_suppression, box_non_max_suppression

def move_detections(
    detections: Detections,
    offset: np.ndarray,
    resolution_wh: tuple[int, int] | None = None,
) -> Detections:
    """
    Args:
        detections (sv.Detections): Detections object to be moved.
        offset (np.ndarray): An array of shape `(2,)` containing offset values in format
            is `[dx, dy]`.
        resolution_wh (Tuple[int, int]): The width and height of the desired mask
            resolution. Required for segmentation detections.

    Returns:
        (sv.Detections) repositioned Detections object.
    """
    detections.xyxy = move_boxes(xyxy=detections.xyxy, offset=offset)
    if ORIENTED_BOX_COORDINATES in detections.data:
        detections.data[ORIENTED_BOX_COORDINATES] = move_oriented_boxes(
            xyxyxyxy=detections.data[ORIENTED_BOX_COORDINATES], offset=offset
        )
    if detections.mask is not None:
        if resolution_wh is None:
            raise ValueError(
                "Resolution width and height are required for moving segmentation "
                "detections. This should be the same as (width, height) of image shape."
            )
        detections.mask = move_masks(
            masks=detections.mask, offset=offset, resolution_wh=resolution_wh
        )
    return detections

def with_nms(
    self,
    threshold: float = 0.5,
    class_agnostic: bool = False,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> Detections:
    """
    Performs non-max suppression on detection set. If the detections result
    from a segmentation model, the IoU mask is applied. Otherwise, box IoU is used.

    Args:
        threshold (float): The intersection-over-union threshold
            to use for non-maximum suppression. I'm the lower the value the more
            restrictive the NMS becomes. Defaults to 0.5.
        class_agnostic (bool): Whether to perform class-agnostic
            non-maximum suppression. If True, the class_id of each detection
            will be ignored. Defaults to False.
        overlap_metric (OverlapMetric): Metric used to compute the degree of
            overlap between pairs of masks or boxes (e.g., IoU, IoS).

    Returns:
        Detections: A new Detections object containing the subset of detections
            after non-maximum suppression.

    Raises:
        AssertionError: If `confidence` is None and class_agnostic is False.
            If `class_id` is None and class_agnostic is False.
    """
    if len(self) == 0:
        return self

    assert self.confidence is not None, (
        "Detections confidence must be given for NMS to be executed."
    )

    if class_agnostic:
        predictions = np.hstack((self.xyxy, self.confidence.reshape(-1, 1)))
    else:
        assert self.class_id is not None, (
            "Detections class_id must be given for NMS to be executed. If you"
            " intended to perform class agnostic NMS set class_agnostic=True."
        )
        predictions = np.hstack(
            (
                self.xyxy,
                self.confidence.reshape(-1, 1),
                self.class_id.reshape(-1, 1),
            )
        )

    if self.mask is not None:
        indices = mask_non_max_suppression(
            predictions=predictions,
            masks=self.mask,
            iou_threshold=threshold,
            overlap_metric=overlap_metric,
            mask_dimension=128
        )
    else:
        indices = box_non_max_suppression(
            predictions=predictions,
            iou_threshold=threshold,
            overlap_metric=overlap_metric,
        )

    return self[indices]


class InferenceSlicer:
    """
    Perform tiled inference on large images by slicing them into overlapping patches.

    This class divides an input image into overlapping slices of configurable size
    and overlap, runs inference on each slice through a user-provided callback, and
    merges the resulting detections. The slicing process allows efficient processing
    of large images with limited resources while preserving detection accuracy via
    configurable overlap and post-processing of overlaps. Uses multi-threading for
    parallel slice inference.

    Args:
        callback (Callable[[ImageType], Detections]): Inference function that takes
            a sliced image and returns a `Detections` object.
        slice_wh (int or tuple[int, int]): Size of each slice `(width, height)`.
            If int, both width and height are set to this value.
        overlap_wh (int or tuple[int, int]): Overlap size `(width, height)` between
            slices. If int, both width and height are set to this value.
        overlap_filter (OverlapFilter or str): Strategy to merge overlapping
            detections (`NON_MAX_SUPPRESSION`, `NON_MAX_MERGE`, or `NONE`).
        iou_threshold (float): IOU threshold used in merging overlap filtering.
        overlap_metric (OverlapMetric or str): Metric to compute overlap
            (`IOU` or `IOS`).
        thread_workers (int): Number of threads for concurrent slice inference.
        batch_size (int): Number of images per batch for batched inference

    Raises:
        ValueError: If `slice_wh` or `overlap_wh` are invalid or inconsistent.

    Example:
        ```python
        import cv2
        import supervision as sv
        from rfdetr import RFDETRMedium

        model = RFDETRMedium()

        def callback(tile):
            return model.predict(tile)

        slicer = sv.InferenceSlicer(callback, slice_wh=640, overlap_wh=100)

        image = cv2.imread("example.png")
        detections = slicer(image)
        ```

        ```python
        import supervision as sv
        from PIL import Image
        from ultralytics import YOLO

        model = YOLO("yolo11m.pt")

        def callback(tile):
            results = model(tile)[0]
            return sv.Detections.from_ultralytics(results)

        slicer = sv.InferenceSlicer(callback, slice_wh=640, overlap_wh=100)

        image = Image.open("example.png")
        detections = slicer(image)
        ```
    """

    def __init__(
        self,
        callback: Union[
            Callable[[np.ndarray], Detections],
            Callable[[List[np.ndarray]], List[Detections]],
        ],
        slice_wh: int | tuple[int, int] = 640,
        overlap_wh: int | tuple[int, int] = 100,
        overlap_filter: OverlapFilter | str = OverlapFilter.NON_MAX_SUPPRESSION,
        iou_threshold: float = 0.5,
        overlap_metric: OverlapMetric | str = OverlapMetric.IOU,
        thread_workers: int = 1,
        batch_size: int = 1,
    ):
        slice_wh_norm = self._normalize_slice_wh(slice_wh)
        overlap_wh_norm = self._normalize_overlap_wh(overlap_wh)

        self.batch_size = batch_size
        if self.batch_size < 1:
            raise ValueError("batch_size should be greater than 0")

        self._validate_overlap(slice_wh=slice_wh_norm, overlap_wh=overlap_wh_norm)

        self.slice_wh = slice_wh_norm
        self.overlap_wh = overlap_wh_norm
        self.iou_threshold = iou_threshold
        self.overlap_metric = OverlapMetric.from_value(overlap_metric)
        self.overlap_filter = OverlapFilter.from_value(overlap_filter)
        self.callback = callback
        self.thread_workers = thread_workers

    def __call__(self, image: ImageType) -> Detections:
        """
        Perform tiled inference on the full image and return merged detections.

        Args:
            image (ImageType): The full image to run inference on.

        Returns:
            Detections: Merged detections across all slices.
        """
        detections_list: list[Detections] = []
        resolution_wh = get_image_resolution_wh(image)

        offsets = self._generate_offset(
            resolution_wh=resolution_wh,
            slice_wh=self.slice_wh,
            overlap_wh=self.overlap_wh,
        )

        batched_offsets_generator = create_batches(offsets, self.batch_size)

        with ThreadPoolExecutor(max_workers=self.thread_workers) as executor:
            futures = []
            for offset_batch in batched_offsets_generator:
                if self.batch_size == 1:
                    future = executor.submit(
                        self._callback_image_single, image, offset_batch[0]
                    )
                else:
                    future = executor.submit(
                        self._callback_image_batch, image, offset_batch
                    )
                futures.append(future)

            for future in as_completed(futures):
                if self.batch_size == 1:
                    detections_list.append(future.result())
                else:
                    detections_list.extend(future.result())

        merged = Detections.merge(detections_list=detections_list)
        if self.overlap_filter == OverlapFilter.NONE:
            return merged
        if self.overlap_filter == OverlapFilter.NON_MAX_SUPPRESSION:
            return with_nms(merged,
                threshold=self.iou_threshold,
                overlap_metric=self.overlap_metric,
            )
        if self.overlap_filter == OverlapFilter.NON_MAX_MERGE:
            return merged.with_nmm(
                threshold=self.iou_threshold,
                overlap_metric=self.overlap_metric,
            )

        warnings.warn(
            f"Invalid overlap filter strategy: {self.overlap_filter}",
            category=SupervisionWarnings,
        )
        return merged


    def _callback_image_batch(
        self, image: np.ndarray, offsets_batch: List[np.ndarray]
    ) -> List[Detections]:
        """
        Run the callback on a batch of images.

        Args:
            image (np.ndarray): The input image on which inference needs to run
            offsets_batch (List[np.ndarray]): List of N arrays of shape `(4,)`,
                containing coordinates of the slices.

        Returns:
            List[Detections]: Detections found in each slice
        """
        assert isinstance(offsets_batch, list)

        slices = [crop_image(image=image, xyxy=offset) for offset in offsets_batch]
        detections_in_slices = self.callback(slices)
        if not isinstance(detections_in_slices, list):
            raise ValueError(
                f"Callback should return a list of Detections objects when "
                f"max_batch_size is greater than 1. "
                f"Instead it returned: {type(detections_in_slices)}"
            )
        
        resolution_wh = get_image_resolution_wh(image)

        detections_with_offset = [
            move_detections(detections=detections, offset=offset[:2], resolution_wh=resolution_wh)
            for detections, offset in zip(detections_in_slices, offsets_batch)
        ]

        return detections_with_offset

    def _callback_image_single(self, image: ImageType, offset: np.ndarray) -> Detections:
        """
        Run detection callback on a sliced portion of the image and adjust coordinates.

        Args:
            image (ImageType): The full image.
            offset (numpy.ndarray): Coordinates `(x_min, y_min, x_max, y_max)` defining
                the slice region.

        Returns:
            Detections: Detections adjusted to the full image coordinate system.
        """
        image_slice: ImageType = crop_image(image=image, xyxy=offset)
        detections = self.callback(image_slice)
        resolution_wh = get_image_resolution_wh(image)

        detections = move_detections(
            detections=detections,
            offset=offset[:2],
            resolution_wh=resolution_wh,
        )
        return detections

    @staticmethod
    def _normalize_slice_wh(
        slice_wh: int | tuple[int, int],
    ) -> tuple[int, int]:
        if isinstance(slice_wh, int):
            if slice_wh <= 0:
                raise ValueError(
                    f"`slice_wh` must be a positive integer. Received: {slice_wh}"
                )
            return slice_wh, slice_wh

        if isinstance(slice_wh, tuple) and len(slice_wh) == 2:
            width, height = slice_wh
            if width <= 0 or height <= 0:
                raise ValueError(
                    f"`slice_wh` values must be positive. Received: {slice_wh}"
                )
            return width, height

        raise ValueError(
            "`slice_wh` must be an int or a tuple of two positive integers "
            "(slice_w, slice_h). "
            f"Received: {slice_wh}"
        )

    @staticmethod
    def _normalize_overlap_wh(
        overlap_wh: int | tuple[int, int],
    ) -> tuple[int, int]:
        if isinstance(overlap_wh, int):
            if overlap_wh < 0:
                raise ValueError(
                    "`overlap_wh` must be a non negative integer. "
                    f"Received: {overlap_wh}"
                )
            return overlap_wh, overlap_wh

        if isinstance(overlap_wh, tuple) and len(overlap_wh) == 2:
            overlap_w, overlap_h = overlap_wh
            if overlap_w < 0 or overlap_h < 0:
                raise ValueError(
                    f"`overlap_wh` values must be non negative. Received: {overlap_wh}"
                )
            return overlap_w, overlap_h

        raise ValueError(
            "`overlap_wh` must be an int or a tuple of two non negative integers "
            "(overlap_w, overlap_h). "
            f"Received: {overlap_wh}"
        )

    @staticmethod
    def _generate_offset(
        resolution_wh: tuple[int, int],
        slice_wh: tuple[int, int],
        overlap_wh: tuple[int, int],
    ) -> np.ndarray:
        """
        Generate bounding boxes defining the coordinates of image slices with overlap.

        Args:
            resolution_wh (tuple[int, int]): Image resolution `(width, height)`.
            slice_wh (tuple[int, int]): Size of each slice `(width, height)`.
            overlap_wh (tuple[int, int]): Overlap size between slices `(width, height)`.

        Returns:
            numpy.ndarray: Array of shape `(num_slices, 4)` with each row as
                `(x_min, y_min, x_max, y_max)` coordinates for a slice.
        """
        slice_width, slice_height = slice_wh
        image_width, image_height = resolution_wh
        overlap_width, overlap_height = overlap_wh

        stride_x = slice_width - overlap_width
        stride_y = slice_height - overlap_height

        def _compute_axis_starts(
            image_size: int,
            slice_size: int,
            stride: int,
        ) -> list[int]:
            if image_size <= slice_size:
                return [0]

            if stride == slice_size:
                return np.arange(0, image_size, stride).tolist()

            last_start = image_size - slice_size
            starts = np.arange(0, last_start, stride).tolist()
            if not starts or starts[-1] != last_start:
                starts.append(last_start)
            return starts

        x_starts = _compute_axis_starts(
            image_size=image_width,
            slice_size=slice_width,
            stride=stride_x,
        )
        y_starts = _compute_axis_starts(
            image_size=image_height,
            slice_size=slice_height,
            stride=stride_y,
        )

        x_min, y_min = np.meshgrid(x_starts, y_starts)
        x_max = np.clip(x_min + slice_width, 0, image_width)
        y_max = np.clip(y_min + slice_height, 0, image_height)

        offsets = np.stack(
            [x_min, y_min, x_max, y_max],
            axis=-1,
        ).reshape(-1, 4)

        return offsets

    @staticmethod
    def _validate_overlap(
        slice_wh: tuple[int, int],
        overlap_wh: tuple[int, int],
    ) -> None:
        overlap_w, overlap_h = overlap_wh
        slice_w, slice_h = slice_wh

        if overlap_w < 0 or overlap_h < 0:
            raise ValueError(
                "Overlap values must be greater than or equal to 0. "
                f"Received: {overlap_wh}"
            )

        if overlap_w >= slice_w or overlap_h >= slice_h:
            raise ValueError(
                "`overlap_wh` must be smaller than `slice_wh` in both dimensions "
                f"to keep a positive stride. Received overlap_wh={overlap_wh}, "
                f"slice_wh={slice_wh}."
            )
