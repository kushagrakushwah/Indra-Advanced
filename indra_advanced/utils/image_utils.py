"""
Utility functions for document and image processing.
Handles PIL Images, OpenCV arrays, PDFs, resizing, and normalization.
"""

from typing import Union, List, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import io


def load_image(image_input: Union[str, Path, bytes, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Load an image from path, bytes, PIL Image, or numpy array into an RGB numpy array.
    """
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {image_input}")
        pil_img = Image.open(p).convert("RGB")
        return np.array(pil_img)

    elif isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        return np.array(pil_img)

    elif isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"))

    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            # Grayscale -> RGB
            import cv2
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif len(image_input.shape) == 3 and image_input.shape[2] == 4:
            # RGBA -> RGB
            import cv2
            return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        return image_input

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def crop_bbox(image: np.ndarray, bbox: List[float]) -> np.ndarray:
    """Crop image with bounding box [xmin, ymin, xmax, ymax]."""
    h, w = image.shape[:2]
    xmin = max(0, int(bbox[0]))
    ymin = max(0, int(bbox[1]))
    xmax = min(w, int(bbox[2]))
    ymax = min(h, int(bbox[3]))
    return image[ymin:ymax, xmin:xmax]


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area
