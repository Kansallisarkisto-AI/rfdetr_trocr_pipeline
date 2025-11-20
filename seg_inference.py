from image_processing import load_with_torchvision, preprocess_resize_torch_transform
from rfdetr import RFDETRSegPreview
import numpy as np
import cv2
from shapely.validation import make_valid
from shapely.geometry import Polygon
from pydantic import BaseModel

class MergeInput(BaseModel):
    line_iou: int = 0.3
    line_overlap_threshold: int = 0.5

def validate_polygon(polygon):
    """"Function for testing and correcting the validity of polygons."""
    if len(polygon) > 2:
        polygon = Polygon(polygon)
        if not polygon.is_valid:
            polygon = make_valid(polygon)
        return polygon
    else:
        return None

def merge_polygons(polygons, data):
    """Merges overlapping polygons using connected components.
    
    Args:
        polygons (list): List of polygon coordinate arrays
        data (object): A configuration object containing merge threshold parameters:
                    - line_iou (float): The IoU threshold above which polygons are merged.
                    - line_overlap_threshold (float, optional): If provided, polygons are also
                        merged when their intersection area exceeds this fraction of the smaller
                        polygon's area.
        
    Returns:
        merged_polygons: List of polygon coordinate arrays
        polygon_mapping: List where polygon_mapping[i] indicates which output 
                         polygon the i-th input polygon was merged into (or i if unchanged)
    """
    n = len(polygons)
    polygon_iou = data.line_iou
    overlap_threshold = data.line_overlap_threshold
    
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
            
            # Check merge criteria
            should_merge = iou > polygon_iou
            # If IoU threshold is not met, checks overlap threshold
            if not should_merge and overlap_threshold:
                overlap_ratio = intersect.area / min(poly1.area, poly2.area)
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

def load_rfdetr_model(model_path):
    """
    Load and optimize an RFDETR segmentation model for inference.

    Args:
        model_path: Path to the pretrained model weights file.

    Returns:
        RFDETRSegPreview: Optimized model ready for inference.
    """
    model = RFDETRSegPreview(pretrain_weights=model_path)
    model.optimize_for_inference()
    return model

def predict_polygons(model, image_path, max_size=768, confidence_threshold = 0.15, percentage_threshold=7e-05):
    """
    Predict and extract line and region polygons from an image using a segmentation model.

    Args:
        model: Loaded RFDETR segmentation model.
        image_path: Path to the input image file.
        max_size: Maximum dimension size for image preprocessing. Default is 768.
        confidence_threshold: Minimum confidence score for detections. Default is 0.15.

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
    #load image and downscale for vram savings
    image = load_with_torchvision(image_path)
    preprocessed_image = preprocess_resize_torch_transform(image, max_size=max_size)

    #predict
    detections = model.predict(preprocessed_image, threshold=confidence_threshold)

    #filter out text line predictions
    line_mask = detections.mask[detections.class_id == 2]
    line_confs = detections.confidence[detections.class_id == 2]
    #filter region coordinates
    region_mask = detections.mask[detections.class_id == 1]
    region_confs = detections.confidence[detections.class_id == 1]

    image_shape = (image.shape[0], image.shape[1])
    line_polygons = []
    new_line_confs = []
    area_percentages = np.array([])
    for mask, conf in zip(line_mask, line_confs):
        # get polygons from mask. Upscales back to original shape
        polygons, area_percentage = mask_to_polygon_cv2(mask=mask, original_shape=image_shape) #this can output multiple polygons from one mask
        area_percentages = np.concatenate([area_percentages, area_percentage])
        #take into account if there are multiple polygons inside one mask
        line_polygons += (polygons)
        new_line_confs += [conf] * len(polygons)

    # Filter lines by area percentages
    filtered_polygons = []
    filtered_confs = []
    for idx in np.where(area_percentages>percentage_threshold)[0]:
        filtered_polygons.append(line_polygons[idx])
        filtered_confs.append(new_line_confs[idx])

    # Merge lines that overlap too much
    merge_input = MergeInput()
    merged_polygons, merged_indices = merge_polygons(filtered_polygons, merge_input)

    merged_confs = calculate_confidences(indices_list=merged_indices, confidence_values=filtered_confs)
    
    merged_line_max_mins = []
    for poly in merged_polygons:
        xmax, ymax = np.max(poly,axis=0)
        xmin, ymin = np.min(poly,axis=0)
        merged_line_max_mins.append((xmin,ymin,xmax,ymax))
    
    # transform region masks to polygons
    region_polygons = []
    region_max_mins = []
    new_region_confs = []
    for mask, conf in zip(region_mask, region_confs):
        # get polygons from mask. Upscales back to original shape
        polygons, _ = mask_to_polygon_cv2(mask=mask, original_shape=image_shape) #this can output multiple polygons from one mask
        #take into account if there are multiple polygons inside one mask
        region_polygons += (polygons)
        new_region_confs += [conf] * len(polygons)
        #calculate maxmins
        for polygon in polygons:
            xmax, ymax = np.max(polygon,axis=0)
            xmin, ymin = np.min(polygon,axis=0)
            region_max_mins.append((xmin,ymin,xmax,ymax))
    return merged_polygons, merged_confs, merged_line_max_mins, region_polygons, new_region_confs, region_max_mins, image_shape
