import glob
import os
from shapely.geometry import Polygon
from reading_order import OrderPolygons
from shapely.validation import make_valid
from reading_order_graph import GraphBasedOrdering
graph_order_poly = GraphBasedOrdering()
order_poly = OrderPolygons()

def load_image_paths(input_folder, extensions=None):
    """
    Load image files from a folder.
    
    Args:
        input_folder: Path to the folder containing images
        extensions: List of file extensions to include (default: common image formats)
    
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'tif', 'webp']
    
    images = []
    for ext in extensions:
        # Add both lowercase and uppercase versions
        images.extend(glob.glob(os.path.join(input_folder, f'*.{ext}')))
        images.extend(glob.glob(os.path.join(input_folder, f'*.{ext.upper()}')))
    
    return sorted(images)

def get_default_region(image_shape):
    """Function for creating a default region if no regions are detected."""
    w, h = image_shape
    region = {'coords': [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], 
            'max_min': [w, 0.0, h, 0.0], 
            #'class': '1',  tämä lienee turha. ei käytetä get_line_regions tai order_regions_lines funktioissa
            'name': "paragraph", 
            'conf': 0.0,
            'id': '0', 
            'img_shape': (h, w)}
    return [region]

def get_dist(line_polygon, regions):
    """Function for finding the closest region to the text line."""
    dist, reg_id = 1000000, None
    line_polygon = validate_polygon(line_polygon)
    if line_polygon:
        for region in regions:
            # Calculates dictance between line and regions polygons
            region_polygon = validate_polygon(region['coords'])
            if region_polygon:
                line_reg_dist = line_polygon.distance(region_polygon)
                if line_reg_dist < dist:
                    dist = line_reg_dist
                    reg_id = region['id']
    return reg_id

def validate_polygon(polygon):
    """"Function for testing and correcting the validity of polygons."""
    if len(polygon) > 2:
        polygon = Polygon(polygon)
        if not polygon.is_valid:
            polygon = make_valid(polygon)
        return polygon
    else:
        return None

def get_iou(poly1, poly2):
    """Function for calculating Intersection over Union (IoU) values."""
    # If the polygons don't intersect, IoU is 0
    iou = 0
    poly1 = validate_polygon(poly1)
    poly2 = validate_polygon(poly2)
    if poly1 and poly2:
        if poly1.intersects(poly2):
            # Calculates intersection of the 2 polygons
            intersect = poly1.intersection(poly2).area
            # Calculates union of the 2 polygons
            uni = poly1.union(poly2)
            # Calculates intersection over union
            iou = intersect / uni.area
    return iou

def get_line_regions(lines, regions):
    """Function for connecting each text line to one region.
    
    Docstring generated with Claude
    Connect each text line to a region based on intersection or distance.

    Args:
        lines (dict): Dictionary containing text line information with keys:
            - 'coords' (list): List of line polygons (coordinates). E.g. [[x1,y1], ..., [xn,yn]]
            - 'max_min' (list): List of bounding box coordinates for each line
            - 'confs' (list): List of confidence scores for each line
        regions (list): List of region dictionaries, each containing:
            - 'coords': Region polygon coordinates. E.g. [[x1,y1], ..., [xn,yn]]
            - 'id': Region identifier

    Returns:
        list: List of dictionaries, each representing a line with keys:
            - 'polygon': Line polygon coordinates
            - 'reg_id': ID of the region the line belongs to
            - 'max_min': Bounding box coordinates [x_min, y_min, x_max, y_max]
            - 'conf': Confidence score for the line

    """
    lines_list = []
    for i in range(len(lines['coords'])):
        iou, reg_id, conf = 0, '', 0.0
        max_min = [0.0, 0.0, 0.0, 0.0]
        polygon = lines['coords'][i]
        for region in regions:
            line_reg_iou = get_iou(polygon, region['coords']) 
            if line_reg_iou > iou:
                iou = line_reg_iou
                reg_id = region['id']
        # If line polygon does not intersect with any region, a distance metric is used for defining 
        # the region that the line belongs to
        if iou == 0:
            reg_id = get_dist(polygon, regions)

        if (len(lines['max_min']) - 1) >= i:
            max_min = lines['max_min'][i]
       
        if (len(lines['confs']) - 1) >= i:
            conf = lines['confs'][i]

        new_line = {'polygon': polygon, 'reg_id': reg_id, 'max_min': max_min, 'conf': conf}
        lines_list.append(new_line)
    return lines_list

def order_regions_lines(lines, regions, new_order = False):
    """Function for ordering line predictions inside each region.
    
    Docstring generated with Claude
    Order text lines within each region and order the regions themselves.

    Args:
        lines (list): List of line dictionaries, each containing:
            - 'reg_id': ID of the region the line belongs to
            - 'max_min': Bounding box coordinates for the line
            - 'conf': Confidence score for the line
            - 'polygon': Line polygon coordinates
        regions (list): List of region dictionaries, each containing:
            - 'id': Region identifier
            - 'coords': Region polygon coordinates
            - 'name': Region name
            - 'conf': Region confidence score
            - 'max_min': Bounding box coordinates for the region
            - 'img_shape': Shape of the source image

    Returns:
        list: List of ordered region dictionaries, each containing:
            - 'region_coords': Region polygon coordinates
            - 'region_name': Region name
            - 'lines': Ordered list of line polygons within the region
            - 'line_confs': Ordered list of line confidence scores
            - 'region_conf': Region confidence score
            - 'img_shape': Shape of the source image
        
        Note: Only regions containing at least one line are included in the output.
        Both lines within regions and regions themselves are ordered by their spatial positions.
    """
    regions_with_rows = []
    region_max_mins = []
    for i, region in enumerate(regions):
        line_max_mins = []
        line_confs = []
        line_polygons = []
        for line in lines:
            if line['reg_id'] == region['id']:
                line_max_mins.append(line['max_min'])
                line_confs.append(line['conf'])
                line_polygons.append(line['polygon'])
        if line_polygons:
            # If one or more lines are connected to a region, line order inside the region is defined
            # and the predicted text lines are joined in the same python dict
            if new_order:
                line_order = graph_order_poly.order(line_max_mins)
            else:
                line_order = order_poly.order(line_max_mins)
            line_polygons = [line_polygons[i] for i in line_order]
            line_confs = [line_confs[i] for i in line_order]
            new_region = {'region_coords': region['coords'], 
                        'region_name': region['name'], 
                        'lines': line_polygons, 
                        'line_confs': line_confs,
                        'region_conf': region['conf'],
                        'img_shape': region['img_shape']}
            region_max_mins.append(region['max_min'])
            regions_with_rows.append(new_region)
        else:
            continue
    # Creates an ordering of the detected regions based on their polygon coordinates
    if new_order:
        region_order = graph_order_poly.order(region_max_mins)
    else:
        region_order = order_poly.order(region_max_mins)
    regions_with_rows = [regions_with_rows[i] for i in region_order]
    return regions_with_rows

def flatten_lines(segment_predictions):
    """
    Merge all text lines from multiple regions into flat lists.

    Args:
        segment_predictions (list): List of ordered region dictionaries, each containing:
            - 'region_coords': Region polygon coordinates
            - 'region_name': Region name
            - 'lines': Ordered list of line polygons within the region
            - 'line_confs': Ordered list of line confidence scores
            - 'region_conf': Region confidence score
            - 'img_shape': Shape of the source image

    Returns:
        tuple: A tuple containing:
            - img_lines (list): Flattened list of all line polygons from all regions
            - img_line_confs (list): Flattened list of all line confidence scores from all regions
            - n_lines (list): List of integers indicating the number of lines per region
    """
    img_lines = []
    img_line_confs = []
    n_lines = []
    for region in segment_predictions:
        img_lines += region['lines']
        img_line_confs += region['line_confs']
        n_lines.append(len(region['lines']))
    return img_lines, img_line_confs, n_lines


def process_text_predictions(text_predictions, segment_predictions, n_lines):
    lines_dicts = text_predictions['text_lines']
    regions = []
    for ind, region in enumerate(segment_predictions):
        n_region_lines = n_lines[ind]
        region_lines_dicts = lines_dicts[:n_region_lines]
        lines_dicts = lines_dicts[n_region_lines:]
        # Combine page data and region specific data
        region_lines = {'img_name': text_predictions['img_name'], 
                        'height': text_predictions['height'], 
                        'width': text_predictions['width'],
                        'page_conf_mean':  text_predictions['page_conf_mean'],
                        'page_conf_median': text_predictions['page_conf_median'],
                        'page_conf_25': text_predictions['page_conf_25'],
                        'page_conf_75': text_predictions['page_conf_75'],
                        'n_long_rowtext': text_predictions['n_long_rowtext'],
                        'language': text_predictions['language'],
                        'region_conf': region['region_conf'],
                        'region_coords': region['region_coords'],
                        'region_name': region['region_name'],
                        'text_lines': region_lines_dicts
                        }
        regions.append(region_lines)
    return regions
