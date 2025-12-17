from image_processing import load_with_torchvision, preprocess_resize_torch_transform, preprocess_resize_smallerof_wh_torch_transform
from rfdetr import RFDETRSegPreview
import supervision as sv
import numpy as np
import cv2
from shapely.validation import make_valid
from shapely.geometry import Polygon, LineString
from shapely.prepared import prep
from collections import defaultdict
import torch
from PIL import Image
from supervision.detection.utils.iou_and_nms import OverlapFilter, OverlapMetric
from inference_slicer_modified import InferenceSlicer

def poly_features(poly_coords, step=1.0, fast_mode=True):
    """Calculates approximate polygon mean thickness using the bounding box if fast_mode=True, or by intersecting a grid with the polygon if fast_mode=False.

    Args:
        poly_coords (array-like): Sequence of (x, y) coordinate pairs defining
                                  the polygon exterior.
        step (float): Spacing between sampling lines in both x and y directions.
                      Smaller values increase accuracy but also computation time.

    Returns:
        dict or None: Dictionary with key:
            - "thickness" (float): Estimated mean polygon thickness.
        Returns None if the polygon is empty or has zero area.
    """

    g = Polygon(poly_coords)

    if g.is_empty:
        return None

    minx, miny, maxx, maxy = g.bounds

    if fast_mode:
        ax = maxx - minx
        ay = maxy - miny
    else:
        pad = step * 2

        def total_len(geom):
            t = geom.geom_type
            if t == "LineString": return geom.length
            if t == "MultiLineString": return sum(x.length for x in geom.geoms)
            if t == "GeometryCollection":
                return sum(x.length for x in geom.geoms if x.geom_type in ("LineString", "MultiLineString"))
            return 0.0  # points etc.

        xs = np.arange(np.floor(minx/step)*step + 0.5*step, np.ceil(maxx/step)*step, step)
        ys = np.arange(np.floor(miny/step)*step + 0.5*step, np.ceil(maxy/step)*step, step)

        g = prep(g)  # prepare geometry for faster intersection calculations

        vx = []
        for x in xs:
            try:  # this can fail if the intersection is undefined for some reason, in that case we say the length is 0.0 (no intersection)
                vx.append(total_len(g.intersection(LineString([(x, miny-pad), (x, maxy+pad)]))))
            except:
                vx.append(0.0)
        
        hy = []
        for y in ys:
            try:  # this can fail if the intersection is undefined for some reason, in that case we say the length is 0.0 (no intersection)
                hy.append(total_len(g.intersection(LineString([(minx-pad, y), (maxx+pad, y)]))))
            except:
                hy.append(0.0)

        ax = float(np.mean([v for v in vx if v > 0])) if np.any(np.array(vx) > 0) else 0.0
        ay = float(np.mean([v for v in hy if v > 0])) if np.any(np.array(hy) > 0) else 0.0

    return {"thickness": min(ax, ay)}

def filter_slivers(polygons, confs=None, threshold = 10.0):
    """
    Drops outliers that are extremely thin.

    Args:
        polygons (list): List of polygon coordinate arrays to be filtered.
        threshold (float): Thickness threshold. Polygons with estimated
                           thickness less than or equal to this value are
                           considered outliers and dropped. The value is
                           clamped to be non-negative.
        confs (list or None): Optional list of confidence values corresponding
                              one-to-one with `polygons`. If provided, the
                              confidences of retained polygons are returned
                              alongside the polygons.

    Returns:
        list or tuple:
            - If `confs` is None:
                list of polygon coordinate arrays that were retained.
            - If `confs` is provided:
                (kept_polygons, kept_confs), where `kept_polygons` is the list
                of retained polygon coordinate arrays and `kept_confs` is the
                corresponding list of confidence values.
    """
    feats = []
    for i, p in enumerate(polygons):
        f = poly_features(p)
        f["idx"] = i
        feats.append(f)

    if not feats:
        return [], [] if confs is not None else []

    th_lo = threshold
    th_lo = max(0.0, float(th_lo))  # ensure threshold >= 0, e.g. in case it is based on IQR

    keep_idxs = []
    for f in feats:
        if f is None:
            continue
        sliver = f["thickness"] <= th_lo

        if not sliver:
            keep_idxs.append(f["idx"])

    kept_polys = [polygons[i] for i in keep_idxs]
    if confs is None:
        return kept_polys
    kept_confs = [confs[i] for i in keep_idxs]
    return kept_polys, kept_confs

def interval_iou(a1, a2, b1, b2):
    """
    Intersection over Union (IoU) for 1D intervals
    """
    inter_len = max(0, min(a2, b2) - max(a1, b1))
    union_len = max(a2, b2) - min(a1, b1)
    if union_len <= 0:
        return 0.0
    return inter_len / union_len

def iou_x(A, B):
    """
    IoU of two polygons along X, based on their bounding boxes.
    """

    if not A or not B or A.is_empty or B.is_empty:
        return 0.5

    ax_min, ay_min, ax_max, ay_max = A.bounds
    bx_min, by_min, bx_max, by_max = B.bounds

    # Overlap along axes
    iou_x = interval_iou(ax_min, ax_max, bx_min, bx_max)  # alignment in x -> vertical stacking

    return iou_x

def iou_y(A, B):
    """
    IoU of two polygons along Y, based on their bounding boxes.
    """

    if not A or not B or A.is_empty or B.is_empty:
        return 0.5

    ax_min, ay_min, ax_max, ay_max = A.bounds
    bx_min, by_min, bx_max, by_max = B.bounds

    # Overlap along axes
    iou_y = interval_iou(ay_min, ay_max, by_min, by_max)  # alignment in y -> side-by-side

    return iou_y

def validate_polygon(polygon):
    """"Function for testing and correcting the validity of polygons."""
    if len(polygon) > 2:
        polygon = Polygon(polygon)
        if not polygon.is_valid:
            polygon = make_valid(polygon)
        return polygon
    else:
        return None

def merge_polygons(polygons, polygon_iou, overlap_threshold, use_verticality=True):
    """Merges overlapping polygons using connected components.
    
    Args:
        polygons (list): List of polygon coordinate arrays
        polygon_iou (float): IoU threshold value for merge
        overlap_threshold (float): Threshold value for polygon area overlap
        
    Returns:
        merged_polygons: List of polygon coordinate arrays
        polygon_mapping: List where polygon_mapping[i] indicates which output 
                         polygon the i-th input polygon was merged into (or i if unchanged)
    """
    n = len(polygons)    
    if n == 0:
        return [], []
    
    # Validate all polygons 
    validated = [validate_polygon(p) for p in polygons]
    
    # Build adjacency graph of overlapping polygons
    parent = list(range(n))  
    
    def find(x):
        # Finds recursively the parent for polygon x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        # Updates parent list for merged polygons
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Check all pairs for overlap
    for i in range(n):
        poly1 = validated[i]
        if not poly1:
            continue
            
        for j in range(i + 1, n):
            poly2 = validated[j]
            if not poly2 or not poly1.intersects(poly2):
                continue
            # Calculate IoU
            intersect = poly1.intersection(poly2)
            union_geom = poly1.union(poly2)
            iou = intersect.area / union_geom.area

            if use_verticality:
                # prioritize horizontal part of overlap
                iou = iou * iou_y(poly1, poly2)
            
            # Check merge criteria
            should_merge = iou > polygon_iou
            # If IoU threshold is not met, checks overlap threshold
            if not should_merge and overlap_threshold:
                overlap_ratio = intersect.area / min(poly1.area, poly2.area)

                if use_verticality:
                    # prioritize horizontal part of overlap
                    overlap_ratio = overlap_ratio * iou_y(poly1, poly2)

                should_merge = overlap_ratio > overlap_threshold
            # If merge criteria is fulfilled, parent list is updated
            if should_merge:
                union(i, j)
    
    # Group polygons by connected component
    components = defaultdict(list)
    for i in range(n):
        if validated[i]: 
            root = find(i)
            components[root].append(i)
    
    # Merge each component
    merged_polygons = []
    polygon_mapping = [-1] * n  # Maps input index to output index
    
    for root, indices in components.items():
        if len(indices) == 1:
            # No merge needed
            idx = indices[0]
            merged_polygons.append(polygons[idx])
            polygon_mapping[idx] = len(merged_polygons) - 1
        else:
            # Merge all polygons in component
            merged = validated[indices[0]]
            for idx in indices[1:]:
                merged = merged.union(validated[idx])
            
            # Extract polygon(s) from result
            output_idx = len(merged_polygons)
            if merged.geom_type == 'Polygon':
                merged_polygons.append(np.array(merged.exterior.coords).astype(np.int32))
                for idx in indices:
                    polygon_mapping[idx] = output_idx
            elif merged.geom_type in ['MultiPolygon', 'GeometryCollection']:
                # Split into separate polygons
                for geom in merged.geoms:
                    if geom.geom_type == 'Polygon':
                        merged_polygons.append(np.array(geom.exterior.coords).astype(np.int32))
                # All source polygons map to first output polygon of this component
                for idx in indices:
                    polygon_mapping[idx] = output_idx
    
    return merged_polygons, polygon_mapping

def calculate_confidences(indices_list, confidence_values):
    """
    Calculate confidence values based on indices.
    
    Args:
        indices_list: List containing either single integers or lists of integers
        confidence_values: List of confidence value strings
        
    Returns:
        List of confidence values (as floats) corresponding to each element in indices_list
    """
    result = []
    
    for item in indices_list:
        if isinstance(item, list):
            # If item is a list, calculate mean of confidences at those indices
            confidences = [float(confidence_values[idx]) for idx in item]
            mean_confidence = sum(confidences) / len(confidences)
            result.append(mean_confidence)
        else:
            # If item is a single index, get confidence at that index
            result.append(float(confidence_values[item]))
    
    return result

def calculate_polygon_area(vertices):
    """Calculate area using Shoelace formula with array shifting."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.sum(x[:-1] * y[1:]) - np.sum(y[:-1] * x[1:]) + 
                        x[-1] * y[0] - y[-1] * x[0])

def mask_to_polygon_cv2(mask, original_shape):
    """
    Convert mask to polygon using OpenCV contours.

    Args:
        mask: numpy array of bool or uint8 (0-255)
    Returns:
        list of polygons, where each polygon is array of (x,y) coordinates
        numpy array of polygon area percentages of the whole image
    """
    # Ensure mask is uint8
    if mask.dtype == bool:
        mask_uint8 = mask.astype(np.uint8) * 255
    else:
        mask_uint8 = mask.astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert contours to list of polygons
    polygons = [contour.squeeze() for contour in contours if len(contour) >= 3]

    # calculate scales
    orig_height, orig_width = original_shape
    mask_height, mask_width = mask.shape[:2]
    scale_x = orig_width / mask_width
    scale_y = orig_height / mask_height
    
    # Scale polygons to original coordinates
    scaled_polygons = []
    area_percentages = []
    mask_area = mask_height*mask_width
    for poly in polygons:
        area = calculate_polygon_area(poly)
        area_percentage = area / (mask_area)
        area_percentages.append(area_percentage)
        if len(poly.shape) == 1:  # Single point, shape (2,)
            scaled_poly = np.round(poly * np.array([scale_x, scale_y])).astype(int)
        else:  # Multiple points, shape (N, 2)
            scaled_poly = np.round(poly * np.array([scale_x, scale_y])).astype(int)
        scaled_polygons.append(scaled_poly)
    return scaled_polygons, np.array(area_percentages)

def load_rfdetr_model(model_path, device="cuda", batch_size=1):
    """
    Load and optimize an RFDETR segmentation model for inference.

    Args:
        model_path: Path to the pretrained model weights file.

    Returns:
        RFDETRSegPreview: Optimized model ready for inference.
    """
    model = RFDETRSegPreview(pretrain_weights=model_path, device=device)
    model.optimize_for_inference(batch_size=batch_size)
    return model

def process_polygons(poly_mask, poly_confs, image_shape, percentage_threshold, overlap_threshold, iou_threshold, use_verticality=True):
    """
    Get polygons from segmentation masks, merge overlapping polygons and drop outlier polygons that are too thin.

    Args:
        poly_masks: segmentation masks from model output
        poly_confs: confidence values for polygon predictions
        image_shape: image height and width
        overlap_threshold: threshold value for polygon overlap 
        iou_threshold: threshold value for polygon IoU

    Returns:
        merged_polygons: coordinates of merged and non-merged polygons
        merged_confs: confidence values for merged and non-merged polygons
        merged_max_mins: max-min values for merged and non-merged polygons
        
    """
    all_polygons = []
    new_confs = []
    area_percentages = np.array([])
    for mask, conf in zip(poly_mask, poly_confs):
        # get polygons from mask. Upscales back to original shape
        polygons, area_percentage = mask_to_polygon_cv2(mask=mask, original_shape=image_shape) #this can output multiple polygons from one mask
        area_percentages = np.concatenate([area_percentages, area_percentage])
        #take into account if there are multiple polygons inside one mask
        all_polygons += (polygons)
        new_confs += [conf] * len(polygons)

    # Filter polygons by area percentages
    filtered_polygons = []
    filtered_confs = []

    for idx in np.where(area_percentages > percentage_threshold)[0]:
        filtered_polygons.append(all_polygons[idx])
        filtered_confs.append(new_confs[idx])

    # Merge polygons that overlap too much
    merged_polygons, merged_indices = merge_polygons(filtered_polygons, iou_threshold, overlap_threshold, use_verticality=use_verticality)

    merged_confs = calculate_confidences(indices_list=merged_indices, confidence_values=filtered_confs)

    # drop polygons that are too thin (slivers)
    if use_verticality:
        merged_polygons, merged_confs = filter_slivers(
            merged_polygons, confs=merged_confs, threshold=0.01*min(image_shape[0], image_shape[1]),
        )
    
    merged_max_mins = []
    for poly in merged_polygons:
        xmax, ymax = np.max(poly,axis=0)
        xmin, ymin = np.min(poly,axis=0)
        merged_max_mins.append((xmin,ymin,xmax,ymax))

    return merged_polygons, merged_confs, merged_max_mins

def predict_polygons(model, 
                     image_path, 
                     max_size=768, 
                     confidence_threshold = 0.15, 
                     line_percentage_threshold=7e-05,
                     region_percentage_threshold=7e-05,
                     line_iou=0.3,
                     region_iou=0.3,
                     line_overlap_threshold=0.5,
                     region_overlap_threshold=0.5,
                     tile_size=0,
                     tile_overlap=128,
                     tile_iou_threshold=0.85,
                     tile_batch_size=1
                     ):
    """
    Predict and extract line and region polygons from an image using a segmentation model.

    Args:
        model: Loaded RFDETR segmentation model.
        image_path: Path to the input image file.
        max_size: Maximum dimension size for image preprocessing. Default is 768.
        confidence_threshold: Minimum confidence score for detections. Default is 0.15.
        line_percentage_threshold: Threshold value for filtering out small line polygons
        region_percentage_threshold: Threshold value for filtering out small region polygons
        line_iou: Threshold value for merging overlapping lines based on IoU
        region_iou: Threshold value for merging overlapping regions based on IoU
        line_overlap_threshold: Threshold value for merging overlapping lines based on overlap
        region_overlap_threshold: Threshold value for merging overlapping regions based on overlap
        tile_size: Size of tiles in document pixels, for Slicing Aided Hyper Inference (SAHI). Default is 0, which disables tiling.
        tile_overlap: Overlap of SAHI tiles in pixels.
        tile_iou_threshold: IoU threshold for suppressing duplicate detections across SAHI tiles (separate from line_iou and region_iou which are applied later).

    Returns:
        tuple: A 7-element tuple containing:
            - line_polygons (list): List of polygon coordinates for detected text lines.
            - new_line_confs (list): Confidence scores for each line polygon.
            - line_max_mins (list): Bounding box coordinates (xmin, ymin, xmax, ymax) for each line.
            - region_polygons (list): List of polygon coordinates for detected regions.
            - new_region_confs (list): Confidence scores for each region polygon.
            - region_max_mins (list): Bounding box coordinates (xmin, ymin, xmax, ymax) for each region.
            - image_shape (tuple): Original image dimensions (height, width).
    """
    if tile_size > 0:  # process image as tiles
        # load with Pillow, because InferenceSlicer expects PIL images
        image = Image.open(image_path).convert('RGB')
        # downscale for time savings, only resizes smaller dimension of (width, height) for arbitrarily long documents
        preprocessed_image = preprocess_resize_smallerof_wh_torch_transform(image, max_size=max_size)

        if tile_batch_size < 2:
            def tiling_callback(tile):
                return model.predict(tile, threshold=confidence_threshold)
        else:
            def tiling_callback(tiles):
                original_tile_count = len(tiles)
                while len(tiles) < tile_batch_size:  # pad using first image as dummy until batch size is filled
                    tiles.append(tiles[0])
                return model.predict(tiles, threshold=confidence_threshold)[:original_tile_count]
        
        # Predict in slices. InferenceSlicer  ensures that slices do not exceed the boundaries of the original image.
        # As a result, the final slices in the row and column dimensions might be smaller than the specified slice dimensions,
        # if the image's width or height is not a multiple of the slice's width or height minus the overlap.
        slicer = InferenceSlicer(tiling_callback, slice_wh=tile_size, overlap_wh=tile_overlap, iou_threshold=tile_iou_threshold, overlap_metric=OverlapMetric.IOU, overlap_filter=OverlapFilter.NON_MAX_SUPPRESSION, batch_size=tile_batch_size)
        detections = slicer(preprocessed_image)
    else:  # process as whole image without tiling
        # load with torchvision
        image = load_with_torchvision(image_path)
        # downscale for VRAM savings, resizes the larger dimension of (width, height)
        preprocessed_image = preprocess_resize_torch_transform(image, max_size=max_size)
        #predict
        detections = model.predict(preprocessed_image, threshold=confidence_threshold)

    # filter out text line predictions
    line_mask = detections.mask[detections.class_id >= 2]
    line_confs = detections.confidence[detections.class_id >= 2]
    # filter region coordinates
    region_mask = detections.mask[detections.class_id == 1]
    region_confs = detections.confidence[detections.class_id == 1]

    # extract H, W
    if tile_size > 0:  # loaded with Pillow
        w, h = image.size
        image_shape = (h, w)
    else:  # loaded with torchvision
        image_shape = (image.shape[0], image.shape[1])
    
    
                         
    # Post-processing for line and region segmentation results
    merged_line_polygons, merged_line_confs, merged_line_max_mins = process_polygons(line_mask, 
                                                                                     line_confs, 
                                                                                     image_shape, 
                                                                                     line_percentage_threshold, 
                                                                                     line_overlap_threshold, 
                                                                                     line_iou, use_verticality=True)

    merged_region_polygons, merged_region_confs, merged_region_max_mins = process_polygons(region_mask, 
                                                                                           region_confs, 
                                                                                           image_shape, 
                                                                                           region_percentage_threshold, 
                                                                                           region_overlap_threshold, 
                                                                                           region_iou, use_verticality=False)
                         
    return merged_line_polygons, merged_line_confs, merged_line_max_mins, merged_region_polygons, merged_region_confs, merged_region_max_mins, image_shape
