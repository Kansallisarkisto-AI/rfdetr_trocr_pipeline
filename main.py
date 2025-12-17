from seg_inference import load_rfdetr_model, predict_polygons
import argparse
from utils import load_image_paths, get_default_region, get_line_regions, order_regions_lines, flatten_lines, process_text_predictions
from tqdm import tqdm
from trocr import get_text_preds, load_trocr_model
from pydantic import BaseModel
from xml_koodit import get_xml
from pathlib import Path
import os
import time
import torch
import torch.multiprocessing as mp


class TextPreditionInput(BaseModel):
    image_path: str
    line_threshold: int

class XmlInput(BaseModel):
    image_path: str
    page_xml: bool
    alto_xml: bool
    xml_path: str
    region_segment_model_name: str
    line_segment_model_name: str
    text_recognition_model_name: str

class TrOCRInput(BaseModel):
    image_path: str
    image_name: str
    image_lines: list
    line_confs: list
    height: int
    width: int
    line_threshold: int

def parse_args():
    parser = argparse.ArgumentParser(description="Load and run inference model")
    parser.add_argument(
        "--detection_model_path",
        type=str,
        default='/path/to/rfdetr/model.pth',
        help="Path to the detection model file"
    )
    parser.add_argument(
        "--recognition_model_path",
        type=str,
        default = '/path/to/trocr/model/folder/',
        help="Path to the recognition model folder"
    )
    parser.add_argument(
        "--processor_path",
        type=str,
        default = '/path/to/trocr/processor/folder/',
        help="Path to the processor folder"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Image input folder"
    )
    parser.add_argument(
        "--region_model_name",
        type=str,
        default = 'rfdetr_text_seg_model_202510',
        help="region model name"
    )
    parser.add_argument(
        "--line_model_name",
        type=str,
        default = 'rfdetr_text_seg_model_202510',
        help="line  model name"
    )
    parser.add_argument(
        "--text_rec_model_name",
        type=str,
        default = '202509_tf32',
        help="Text rec model name"
    )
    parser.add_argument(
        "--line_threshold",
        type=int,
        default=8,
        help="Batch size for text_rec"
    )
    parser.add_argument(
        "--page_xml",
        type=bool,
        default=False,
        help="Whether to save as page xml"
    )
    parser.add_argument(
        "--alto_xml",
        type=bool,
        default=True,
        help="Whether to save as alto xml"
    )
    parser.add_argument(
        "--xml_folder",
        type=str,
        default=None,
        help="Where to save xmls. If None, saves to img_folder under alto or page folders"
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.15,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--new_ordering",
        type=bool,
        default=False,
        help="Whether to use new ordering for the lines"
    )
    parser.add_argument(
        "--line_percentage_threshold",
        type=float,
        default=7e-05,
        help="Threshold value for filtering out small line polygons"
    )
    parser.add_argument(
        "--region_percentage_threshold",
        type=float,
        default=7e-05,
        help="Threshold value for filtering out small region polygons"
    )
    parser.add_argument(
        "--line_iou",
        type=float,
        default=0.3,
        help="Threshold value for merging overlapping lines"
    )
    parser.add_argument(
        "--region_iou",
        type=float,
        default=0.3,
        help="Threshold value for merging overlapping regions"
    )
    parser.add_argument(
        "--line_overlap_threshold",
        type=float,
        default=0.5,
        help="Threshold value for merging overlapping lines"
    )
    parser.add_argument(
        "--region_overlap_threshold",
        type=float,
        default=0.5,
        help="Threshold value for merging overlapping regions"
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=0,
        help="Tile size for Slicing Aided Hyper Inference (SAHI), in pixels. 384 is usually a good value for RF-DETR segmentation models. 0 to disable."
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=128,
        help="Tile overlap for SAHI, in pixels."
    )
    parser.add_argument(
        "--tiles_across",
        type=int,
        default=2,
        help="The number of SAHI tiles across the shorter dimension of the image."
    )
    parser.add_argument(
        "--tile_iou_threshold",
        type=float,
        default=0.85,
        help="IoU threshold for suppressing overlapping detections _when SAHI is used_. 0.85 for general material, 0.5 is sometimes better for tables."
    )
    parser.add_argument(
        "--tile_batch_size",
        type=int,
        default=6,
        help="Batch size for SAHI. 6 is a reasonable value."
    )
    parser.add_argument(
        "--multi_gpu",
        type=bool,
        default=False,
        help="Whether to use all GPUs on system instead of just the first one. Requires a patched version of the RF-DETR library that accepts a device argument with the rank specified."
    )
        
    args = parser.parse_args()
    return args

def get_text_predictions(input_data, segment_predictions, recognition_model, processor):
    """Collects text prediction data into dicts based on detected text regions.

    Args:
        input_data: Pydantic data model containing:
                - 'image_path': Path to the input image
                - 'line_threshold': Batch size for text recognition

        segment_predictions: List of ordered region dictionaries, each containing:
                - 'region_coords': Region polygon coordinates
                - 'region_name': Region name
                - 'lines': Ordered list of line polygons within the region
                - 'line_confs': Ordered list of line confidence scores
                - 'region_conf': Region confidence score
                - 'img_shape': Shape of the source image

        'recognition_model': Pytorch TrOCR model

        'processor': TrOCR processor

    """
    img_lines, img_line_confs, n_lines = flatten_lines(segment_predictions)
    height, width = segment_predictions[0]['img_shape']
    image_name = Path(input_data.image_path).name
    payload = TrOCRInput(image_path=input_data.image_path,
                        image_name=image_name,
                        image_lines=img_lines,
                        line_confs=img_line_confs,
                        height=height,
                        width=width, 
                        line_threshold=input_data.line_threshold)
    # Process all lines of an image
    text_predictions = get_text_preds(payload, recognition_model, processor)
    if text_predictions:
        preds = process_text_predictions(text_predictions, segment_predictions, n_lines)
        return preds
    else:
        return None

def process_all_images(images, detection_model, recognition_model, processor, args, rank=0):
    """
    Process a collection of images through detection, recognition, and XML output pipeline.

    This function performs end-to-end OCR processing on a batch of images by:
    1. Detecting text lines and regions using a detection model
    2. Organizing detected lines into regions and ordering them
    3. Recognizing text content using a recognition model
    4. Generating XML output (PAGE or ALTO format) with the recognized text

    Args:
        images: Iterable of image file paths to process
        detection_model: Model for detecting text lines and regions in images
        recognition_model: Model for recognizing text content from detected lines
        processor: Processor for preparing data for the recognition model
        args: Argument object containing configuration parameters including:
            - line_threshold: Threshold for line detection
            - page_xml: Flag for PAGE XML output
            - alto_xml: Flag for ALTO XML output
            - region_model_name: Name of the region segmentation model
            - line_model_name: Name of the line segmentation model
            - text_rec_model_name: Name of the text recognition model

    Returns:
        None. Outputs are written as XML files to the same directory as input images.

    Note:
        Progress is displayed via tqdm progress bar during processing.
    """
    bar = tqdm(
        images,
        desc=f"GPU {rank}",
        position=rank,
        leave=True,
        dynamic_ncols=True,
    )

    for image_path in bar:
        start_time = time.time()
        line_polygons, line_confs, line_max_mins, region_polygons, region_confs, region_max_mins, image_shape = predict_polygons(
                                    detection_model, 
                                    image_path, 
                                    max_size = args.tile_size * args.tiles_across - args.tile_overlap if args.tile_size else 768, 
                                    confidence_threshold = args.confidence_threshold,
                                    line_percentage_threshold = args.line_percentage_threshold,
                                    region_percentage_threshold = args.region_percentage_threshold,
                                    line_iou = args.line_iou,
                                    region_iou = args.region_iou,
                                    line_overlap_threshold = args.line_overlap_threshold,
                                    region_overlap_threshold = args.region_overlap_threshold,
                                    tile_size = args.tile_size,
                                    tile_overlap = args.tile_overlap,
                                    tile_iou_threshold = args.tile_iou_threshold,
                                    tile_batch_size=args.tile_batch_size)
        
        predict_polygons_time = time.time() - start_time

        start_time = time.time()
        line_preds = {'coords':line_polygons,
                      'max_min': line_max_mins,
                      'confs':line_confs
                      }
        if len (region_polygons) > 0:
            region_preds = []
            for num, (region_polygon, region_conf, region_max_min) in enumerate(zip(region_polygons, region_confs, region_max_mins)):
                region_preds.append({'coords': region_polygon,
                                    'id': str(num),
                                    'max_min': region_max_min,
                                    'name': 'paragraph',
                                    'img_shape': image_shape,
                                    'conf': region_conf})
        else:
            region_preds = get_default_region(image_shape=image_shape)
        lines_connected_to_regions = get_line_regions(lines=line_preds, regions=region_preds)
        ordered_lines = order_regions_lines(lines=lines_connected_to_regions, regions=region_preds, new_order = args.new_ordering)
        if ordered_lines:
            input_data = TextPreditionInput(image_path = image_path,
                                            line_threshold = args.line_threshold)
            text_predictions = get_text_predictions(input_data, ordered_lines, recognition_model, processor)
            if text_predictions:
                xml_input = XmlInput(image_path = image_path,
                                    page_xml = args.page_xml,
                                    alto_xml = args.alto_xml,
                                    xml_path = os.path.dirname(image_path) if not args.xml_folder else args.xml_folder,
                                    region_segment_model_name=args.region_model_name,
                                    line_segment_model_name=args.line_model_name,
                                    text_recognition_model_name=args.text_rec_model_name)
                get_xml(text_predictions, xml_input)
        
        get_text_predictions_time = time.time() - start_time
        bar.set_postfix_str(f"  predict_polygons: {predict_polygons_time:.2f} s, get_text_predictions: {get_text_predictions_time:.2f} s")
        
def main(args):
    print("Loading rfdetr model")
    detection_model = load_rfdetr_model(args.detection_model_path, batch_size=args.tile_batch_size if args.tile_size else 1)
    print('Loading TrOCR model')
    recognition_model, processor = load_trocr_model(args.recognition_model_path, args.processor_path)

    print('Find images in folder', str(args.input_folder))
    images = load_image_paths(args.input_folder)
    print('Found ', str(len(images)))

    print('Starting HTR')
    process_all_images(images, detection_model, recognition_model, processor, args)
    print('Processing Finished')

def worker(rank, world_size, args):
    # Pin this process to one GPU
    torch.cuda.set_device(rank)
    device_string = f"cuda:{rank}"
    #device = torch.device(f"cuda:{rank}")

    print(f"[GPU {rank}] loading models...")
    detection_model = load_rfdetr_model(args.detection_model_path, device=device_string, batch_size=args.tile_batch_size if args.tile_size else 1)
    recognition_model, processor = load_trocr_model(args.recognition_model_path, args.processor_path, device=device_string)

    images = load_image_paths(args.input_folder)

    # select the range of images for this worker, images are split across world_size workers
    my_images = images[rank::world_size]
    print(f"[GPU {rank}] got {len(my_images)} images")

    process_all_images(my_images, detection_model, recognition_model, processor, args, rank=rank)


if __name__ == "__main__":
    args = parse_args()

    if args.multi_gpu:
        ngpu = torch.cuda.device_count()
        print(f"{ngpu} GPUs detected")
    else:
        ngpu = 1
        print("Using only first GPU.")
    
    if ngpu <= 1:
        main(args)
    else:
        mp.set_start_method("spawn", force=True)  # important for CUDA
        mp.spawn(worker, args=(ngpu, args), nprocs=ngpu, join=True)
