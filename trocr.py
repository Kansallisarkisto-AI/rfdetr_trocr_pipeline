from image_processing import load_with_torchvision
import numpy as np
import cv2
import math
from langdetect import detect
import torch
from transformers.models.vit.modeling_vit import ViTPatchEmbeddings, ViTEmbeddings
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_HEIGHT = 192
IMG_WIDTH = 1024

def load_trocr_model(model_path, processor_path, device=None):
    """Load a TrOCR model with custom image size support.
    
    Load a TrOCR model with custom image size support and positional encoding interpolation.

    This function applies patches to ViTPatchEmbeddings and ViTEmbeddings to enable
    models to handle custom image sizes by interpolating positional encodings.

    Args:
        model_path: Path to the pretrained TrOCR model directory or checkpoint.
        processor_path: Path to the TrOCR processor directory or checkpoint.

    Returns:
        tuple: A 2-element tuple containing:
            - processor (TrOCRProcessor): Configured image processor with custom dimensions.
            - model (VisionEncoderDecoderModel): Loaded TrOCR model on the specified device.
    """
    global DEVICE  # ugly, but necessary until we refactor this into a class
    if device:
        DEVICE = device
    # Store original
    original_embeddings_forward = ViTEmbeddings.forward
    
    # Always apply patches for models saved with custom image sizes
    def universal_patch_forward(self, *args, **kwargs):
        pixel_values = args[0] if args else kwargs['pixel_values']
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings
    
    def universal_embeddings_forward(self, *args, **kwargs):
        kwargs['interpolate_pos_encoding'] = True
        return original_embeddings_forward(self, *args, **kwargs)
    
    # Apply patches
    ViTPatchEmbeddings.forward = universal_patch_forward
    ViTEmbeddings.forward = universal_embeddings_forward
    
    # Load model and processor
    processor = TrOCRProcessor.from_pretrained(processor_path,
                                               use_fast=True,
                                               do_resize=True, 
                                               size={'height': IMG_HEIGHT,'width': IMG_WIDTH})
     
    model = VisionEncoderDecoderModel.from_pretrained(
                                                    model_path,
                                                    torch_dtype=torch.float16
                                                ).to(DEVICE)
    
    return model, processor

def crop_line(image, polygon):
    """Crops predicted text line based on the polygon coordinates
    and returns binarised text line image.

    Crop a text line from an image based on polygon coordinates and return a binarized image.

    Args:
        image: Input image array.
        polygon: List of coordinate pairs defining the text line polygon.

    Returns:
        numpy.ndarray: Cropped and binarized text line image with white background.
    """
    polygon = np.array([[int(lst[0]), int(lst[1])] for lst in polygon], dtype=np.int32)
    rect = cv2.boundingRect(polygon)
    cropped_image = image[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    mask = np.zeros([cropped_image.shape[0], cropped_image.shape[1]], dtype=np.uint8)
    cv2.drawContours(mask, [polygon- np.array([[rect[0],rect[1]]])], -1, (255, 255, 255), -1, cv2.LINE_AA)
    res = cv2.bitwise_and(cropped_image, cropped_image, mask = mask)
    wbg = np.ones_like(cropped_image, np.uint8)*255
    cv2.bitwise_not(wbg,wbg, mask=mask)
    # Overlap the resulted cropped image on the white background
    dst = wbg+res
    return dst

def crop_lines(polygons, image):
    """Returns a list of line images cropped following the 
    detected polygon coordinates.
    
    Crop multiple text lines from an image based on polygon coordinates.

    Args:
        polygons: List of polygons, each containing coordinate pairs.
        image: Input image array.

    Returns:
        list: List of cropped text line images.
    """
    cropped_lines = []
    for polygon in polygons:
        cropped_line = crop_line(image, polygon)
        cropped_lines.append(cropped_line)
    return cropped_lines

def get_scores(lgscores):
    """Get exponent of log scores.
    
    Convert log scores to probability scores by computing their exponent.

    Args:
        lgscores: List of log scores.

    Returns:
        list: List of exponentiated scores.
        
    """
    scores = []
    for lgscore in lgscores:
        score = math.exp(lgscore)
        scores.append(score)
    return scores

def predict_text(cropped_lines, recognition_model, processor):
    """Functions for predicting text content from the cropped line images.

    Predict text content from cropped line images using a recognition model.

    Args:
        cropped_lines: List of cropped text line images.
        recognition_model: Pre-trained text recognition model.
        processor: Image processor for the recognition model.

    Returns:
        tuple: A 2-element tuple containing:
            - scores (list): Confidence scores for each prediction.
            - generated_text (list): Predicted text strings for each line.
    """
    pixel_values = processor(cropped_lines, return_tensors="pt").pixel_values
    generated_dict = recognition_model.generate(pixel_values.to(DEVICE, dtype=torch.float16), max_new_tokens=128, return_dict_in_generate=True, output_scores=True)
    generated_ids, lgscores = generated_dict['sequences'], generated_dict['sequences_scores']
    scores = get_scores(lgscores.tolist())
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    #torch.cuda.empty_cache()
    return scores, generated_text

def detect_language(text_lines):
    """
    Detect the language of text from multiple text lines.

    Args:
        text_lines: List of text strings to analyze.

    Returns:
        str: Detected language code, or "unknown" if detection fails.
    """
    combined_text = " ".join(text_lines)
    try:
        return detect(combined_text)
    except:
        return "unknown"

def get_text_lines(cropped_lines, line_threshold, recognition_model, processor):
    """
    Process text lines in batches and predict text content with confidence scores.

    Args:
        cropped_lines: List of cropped text line images.
        line_threshold: Maximum number of lines to process in a single batch.
        recognition_model: Pre-trained text recognition model.
        processor: Image processor for the recognition model.

    Returns:
        tuple: A 2-element tuple containing:
            - scores (list): Confidence scores for all predictions.
            - generated_text (list): Predicted text strings for all lines.
    """
    scores, generated_text = [], []
    if len(cropped_lines) <= line_threshold:
        scores, generated_text = predict_text(cropped_lines, recognition_model, processor)
    else:
        n = math.ceil(len(cropped_lines) / line_threshold)
        for i in range(n):
            start = int(i * line_threshold)
            end = int(min(start + line_threshold, len(cropped_lines)))
            sc, gt = predict_text(cropped_lines[start:end], recognition_model, processor)
            scores += sc
            generated_text += gt
    return scores, generated_text

def get_res_dict(polygons, generated_text, height, width, image_name, line_confs, scores):
    """Combines the results in a dictionary form.
    
    Combine OCR results into a structured dictionary with metadata and statistics.

    Args:
        polygons: List of polygon coordinates for each text line.
        generated_text: List of predicted text strings.
        height: Image height in pixels.
        width: Image width in pixels.
        image_name: Name of the processed image file.
        line_confs: List of confidence scores for line detection.
        scores: List of confidence scores for text recognition.

    Returns:
        dict: Dictionary with the following structure:
            - img_name (str): Name of the processed image file.
            - height (int): Image height in pixels.
            - width (int): Image width in pixels.
            - page_conf_mean (float): Mean confidence score across all text lines.
            - page_conf_median (float): Median confidence score across all text lines.
            - page_conf_25 (float): 25th percentile of confidence scores.
            - page_conf_75 (float): 75th percentile of confidence scores.
            - n_long_rowtext (int): Number of text lines exceeding 100 characters.
            - language (str): Detected language code.
            - text_lines (list): List of dictionaries, each containing:
                - polygon (list): Polygon coordinates for the text line.
                - text (str): Predicted text content.
                - conf (float): Confidence score for line detection.
                - text_conf (float): Confidence score for text recognition.
                - row_length (int): Character count of the text.
    """
    line_dicts = []
    for i in range(len(generated_text)):
        row_length = len(generated_text[i])
        line_dict = {'polygon': polygons[i], 'text': generated_text[i], 'conf': line_confs[i], 'text_conf':scores[i], 'row_length': row_length}
        line_dicts.append(line_dict)
    page_conf_mean = np.mean(scores) if scores else 0
    page_conf_median = np.median(scores) if scores else 0
    page_conf_25 = np.percentile(scores, 25) if scores else 0
    page_conf_75 = np.percentile(scores, 75) if scores else 0
    n_long_rowtext = sum(1 for d in line_dicts if d['row_length'] > 100) # the limit for the length of the  row text
    language = detect_language(generated_text)
    lines_dict = {'img_name': image_name, 
                  'height': height, 
                  'width': width, 
                  'page_conf_mean': page_conf_mean, 
                  'page_conf_median': page_conf_median, 
                  'page_conf_25': page_conf_25, 
                  'page_conf_75':page_conf_75, 
                  'n_long_rowtext': n_long_rowtext, 
                  'language': language, 
                  'text_lines': line_dicts}
    return lines_dict

def get_text_preds(data, recognition_model, processor):
    """
    Process an image to extract and recognize text from detected text lines.

    Args:
        data: Object containing input parameters including:
            - image_path: Path to the input image.
            - image_lines: List of polygon coordinates for text lines.
            - height: Image height.
            - width: Image width.
            - line_threshold: Maximum batch size for text recognition.
            - image_name: Name of the image file.
            - line_confs: Confidence scores for line detection.
        recognition_model: Pre-trained text recognition model.
        processor: Image processor for the recognition model.

    Returns:
        dict: Dictionary with the following structure:
            - img_name (str): Name of the processed image file.
            - height (int): Image height in pixels.
            - width (int): Image width in pixels.
            - page_conf_mean (float): Mean confidence score across all text lines.
            - page_conf_median (float): Median confidence score across all text lines.
            - page_conf_25 (float): 25th percentile of confidence scores.
            - page_conf_75 (float): 75th percentile of confidence scores.
            - n_long_rowtext (int): Number of text lines exceeding 100 characters.
            - language (str): Detected language code.
            - text_lines (list): List of dictionaries, each containing:
                - polygon (list): Polygon coordinates for the text line.
                - text (str): Predicted text content.
                - conf (float): Confidence score for line detection.
                - text_conf (float): Confidence score for text recognition.
                - row_length (int): Character count of the text.
    """
    # Load image file
    image = load_with_torchvision(data.image_path)
    # Crop line images
    cropped_lines = crop_lines(data.image_lines, image)
    # Get text predictions
    scores, generated_text = get_text_lines(cropped_lines, data.line_threshold, recognition_model, processor)
    # Get results in dictionary form
    res_dict = get_res_dict(data.image_lines, generated_text, data.height, data.width, data.image_name, data.line_confs, scores)
    return res_dict
