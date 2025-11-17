from torchvision.io import read_image, ImageReadMode
import numpy as np
from torchvision.transforms import v2 as transforms_v2
import torch
import cv2

def load_with_torchvision(img_path):
    """
    Load an image using torchvision and convert to numpy array.

    Args:
        img_path (str or Path): Path to the image file

    Returns:
        numpy.ndarray: Image array in RGB format with shape (H, W, C)
    """
    # Read as tensor
    img_tensor = read_image(str(img_path), mode= ImageReadMode.RGB)
    # Convert to numpy: (C, H, W) -> (H, W, C)
    img_np = img_tensor.permute(1, 2, 0).numpy()
    return img_np

def preprocess_resize_torch_transform(image, max_size=1024, normalize=True):
    """
    Resize using torchvision.transforms.v2 (most concise, PyTorch only).

    Args:
        image: torch.Tensor (C, H, W) or PIL Image
        max_size: maximum size for the longer dimension
        normalize: whether to normalize to [0, 1] range

    Returns:
        torch.Tensor (C, H, W) or PIL Image (same type as input)
    """
    # Convert to tensor if numpy
    input_type = type(image)
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
        if image.ndim == 3 and image.shape[2] in [1, 3]:
            image = image.permute(2, 0, 1)

    c, h, w = image.shape if isinstance(image, torch.Tensor) else (None, *image.size[::-1])

    # Build transform pipeline
    transform_list = []

    # Add resize if needed
    if h > max_size or w > max_size:
        transform_list.append(transforms_v2.Resize(size=None, max_size=max_size, antialias=True))

    # Add normalization
    if normalize:
        transform_list.append(transforms_v2.ToDtype(torch.float32, scale=True))

    # Apply transforms
    if transform_list:
        transform = transforms_v2.Compose(transform_list)
        resized = transform(image)
    else:
        resized = image

    return resized

def upscale_mask_opencv(mask, bbox, upscaled_bbox_shape):
    """Upscale using OpenCV resize with nearest neighbor."""
    x1, y1, x2, y2 = map(int, bbox)
    cropped_mask = mask[y1:y2, x1:x2]
    mask_uint8 = cropped_mask.astype(np.uint8)
    upscaled = cv2.resize(mask_uint8, 
                         upscaled_bbox_shape, 
                         interpolation=cv2.INTER_NEAREST)

    return upscaled * 255

def upscale_bbox(bbox, original_shape, mask_shape):
    """
    Upscale bounding box coordinates from mask resolution to original image resolution.

    Parameters:
    -----------
    bbox : np.ndarray or list
        Bounding box coordinates in format [x_min, y_min, x_max, y_max]
        in the mask's coordinate system
    original_shape : tuple
        Original image shape (H, W) or (H, W, C) - e.g., (4545, 5527, 3)
    mask_shape : tuple
        Mask shape (H, W) - e.g., (631, 768)

    Returns:
    --------
    np.ndarray
        Upscaled bounding box as integer coordinates [x_min, y_min, x_max, y_max]
    """

    # Ensure bbox is a numpy array
    bbox = np.array(bbox)

    # Extract height and width from shapes
    original_h, original_w = original_shape[0], original_shape[1]
    mask_h, mask_w = mask_shape[0], mask_shape[1]

    # Calculate scale factors
    scale_x = original_w / mask_w  # Width scaling
    scale_y = original_h / mask_h  # Height scaling

    # Unpack bbox coordinates
    x_min, y_min, x_max, y_max = bbox

    # Scale coordinates
    x_min_scaled = x_min * scale_x
    y_min_scaled = y_min * scale_y
    x_max_scaled = x_max * scale_x
    y_max_scaled = y_max * scale_y

    # limit to range 0 to original width/height
    if x_min_scaled < 0:
        x_min_scaled = 0
    if y_min_scaled < 0:
        y_min_scaled = 0
    if x_max_scaled > original_w:
        x_max_scaled = original_w
    if y_max_scaled > original_h:
        y_max_scaled = original_h

    # Convert to integers (rounding to nearest)
    bbox_scaled = np.array([
        x_min_scaled,
        y_min_scaled,
        x_max_scaled,
        y_max_scaled
    ]).astype(np.int32)

    return bbox_scaled

def crop_line(image, mask, upscaledbbox):
    """Crops predicted text line based on the polygon coordinates
    and returns binarised text line image."""
    x1,y1,x2,y2 = upscaledbbox
    cropped_image = image[y1:y2,x1:x2,:]
    res = cv2.bitwise_and(cropped_image, cropped_image, mask = mask)
    wbg = np.ones_like(cropped_image, np.uint8)*255
    cv2.bitwise_not(wbg,wbg, mask=mask)
    # Overlap the resulted cropped image on the white background
    dst = wbg+res
    return dst