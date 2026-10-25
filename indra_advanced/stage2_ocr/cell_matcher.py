"""
Stage 2: Cell-to-Text Spatial Alignment Engine.
Maps OCR text snippets (from PP-OCRv4) to SLANet predicted table cells
using spatial containment, IoU overlap, and vertical reading order.
Populates the HTML table skeleton with exact cell text.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from bs4 import BeautifulSoup
from indra_advanced.utils.image_utils import compute_iou


class CellMatcher:
    """
    Spatial alignment engine: maps OCR detected words/lines into SLANet predicted cell bounding boxes.
    """

    @staticmethod
    def match_and_populate_html(
        pred_html: str,
        cell_bboxes: List[List[float]],
        ocr_boxes: List[Any],
        ocr_texts: List[str],
        ocr_scores: Optional[List[float]] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Assigns OCR texts into cell bboxes and injects text into HTML table.
        
        Args:
            pred_html: Raw HTML skeleton (e.g. <table><tr><td></td>...</tr></table>).
            cell_bboxes: Bounding boxes [xmin, ymin, xmax, ymax] for each cell in reading order.
            ocr_boxes: List of OCR text bounding polygons/boxes.
            ocr_texts: List of recognized OCR text strings.
            ocr_scores: List of OCR confidence scores.
            
        Returns:
            Tuple of (populated_html_string, cell_assignment_records).
        """
        num_cells = len(cell_bboxes)
        cell_texts: List[List[Tuple[float, float, str, float]]] = [[] for _ in range(num_cells)]

        if ocr_scores is None:
            ocr_scores = [1.0] * len(ocr_texts)

        # Standardize OCR box formats: can be 4-point polygon [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        # or [xmin, ymin, xmax, ymax]
        standard_ocr_boxes: List[List[float]] = []
        centers: List[Tuple[float, float]] = []

        for box in ocr_boxes:
            if isinstance(box, (list, tuple, np.ndarray)):
                if len(box) == 4 and isinstance(box[0], (list, tuple, np.ndarray)):
                    # 4 points
                    xs = [pt[0] for pt in box]
                    ys = [pt[1] for pt in box]
                    b = [min(xs), min(ys), max(xs), max(ys)]
                else:
                    b = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
            else:
                b = [0.0, 0.0, 0.0, 0.0]

            standard_ocr_boxes.append(b)
            center_x = (b[0] + b[2]) / 2.0
            center_y = (b[1] + b[3]) / 2.0
            centers.append((center_x, center_y))

        # Assign each OCR item to the most fitting cell bbox
        for idx, (b_ocr, (cx, cy), text, score) in enumerate(zip(standard_ocr_boxes, centers, ocr_texts, ocr_scores)):
            if not text.strip():
                continue

            best_cell_idx = -1
            best_iou = -1.0

            # 1. Primary check: Center point containment
            for c_idx, cbox in enumerate(cell_bboxes):
                xmin, ymin, xmax, ymax = cbox
                # Small margin tolerance (5% cell size)
                w_margin = (xmax - xmin) * 0.05
                h_margin = (ymax - ymin) * 0.05

                if (xmin - w_margin <= cx <= xmax + w_margin) and (ymin - h_margin <= cy <= ymax + h_margin):
                    best_cell_idx = c_idx
                    break

            # 2. Secondary check: IoU overlap if center check didn't match
            if best_cell_idx == -1:
                for c_idx, cbox in enumerate(cell_bboxes):
                    iou = compute_iou(b_ocr, cbox)
                    if iou > best_iou and iou > 0.05:
                        best_iou = iou
                        best_cell_idx = c_idx

            if best_cell_idx != -1:
                # Store (y, x, text, score) for sorting within cell
                cell_texts[best_cell_idx].append((cy, cx, text.strip(), score))

        # Sort multiple lines inside the same cell top-to-bottom, left-to-right
        final_cell_strings: List[str] = []
        cell_records: List[Dict[str, Any]] = []

        for c_idx, lines in enumerate(cell_texts):
            # Sort by y then x
            sorted_lines = sorted(lines, key=lambda x: (x[0], x[1]))
            merged_text = " ".join([item[2] for item in sorted_lines])
            final_cell_strings.append(merged_text)

            bbox = cell_bboxes[c_idx] if c_idx < len(cell_bboxes) else []
            cell_records.append({
                "cell_index": c_idx,
                "bbox": bbox,
                "text": merged_text,
                "lines": [item[2] for item in sorted_lines],
            })

        # Inject final texts into the HTML skeleton
        soup = BeautifulSoup(pred_html, "html.parser")
        tags = soup.find_all(["th", "td"])

        for idx, tag in enumerate(tags):
            if idx < len(final_cell_strings):
                tag.string = final_cell_strings[idx]

        return str(soup), cell_records
