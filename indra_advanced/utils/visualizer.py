"""
Visualizer utility for drawing table cell bounding boxes, OCR spans,
and extracted entity highlights onto document images.
"""

from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class DocumentVisualizer:
    """Visualizes document analysis outputs (cell structures, OCR text, extracted entities)."""

    @staticmethod
    def draw_table_cells(
        image: np.ndarray,
        cell_bboxes: List[List[float]],
        outline_color: Tuple[int, int, int] = (0, 120, 255),
        line_width: int = 2,
    ) -> np.ndarray:
        """Draw bounding boxes of table cells predicted by SLANet/SLANeXt."""
        pil_img = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_img)

        for bbox in cell_bboxes:
            if len(bbox) >= 4:
                draw.rectangle(
                    [bbox[0], bbox[1], bbox[2], bbox[3]],
                    outline=outline_color,
                    width=line_width,
                )

        return np.array(pil_img)

    @staticmethod
    def draw_entities(
        image: np.ndarray,
        entities: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Draw colored entity tags when spatial coordinate spans are available."""
        pil_img = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_img)

        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()

        for ent in entities:
            bbox = ent.get("bbox")
            if bbox and len(bbox) >= 4:
                draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=(255, 60, 60), width=2)
                draw.text((bbox[0], max(0, bbox[1] - 14)), f"{ent.get('label')}", fill=(255, 60, 60), font=font)

        return np.array(pil_img)
