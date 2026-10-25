"""
Unit tests for Stage 2: CellMatcher spatial alignment.
"""

from indra_advanced.stage2_ocr.cell_matcher import CellMatcher


def test_cell_matcher_center_containment():
    pred_html = "<table><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>"
    cell_bboxes = [
        [0.0, 0.0, 100.0, 50.0],
        [100.0, 0.0, 200.0, 50.0],
        [0.0, 50.0, 100.0, 100.0],
        [100.0, 50.0, 200.0, 100.0],
    ]

    # OCR text boxes inside cells
    ocr_boxes = [
        [[10.0, 10.0], [90.0, 10.0], [90.0, 40.0], [10.0, 40.0]],       # Cell 0
        [[110.0, 10.0], [190.0, 10.0], [190.0, 40.0], [110.0, 40.0]],   # Cell 1
        [[10.0, 60.0], [90.0, 60.0], [90.0, 90.0], [10.0, 90.0]],       # Cell 2
        [[110.0, 60.0], [190.0, 60.0], [190.0, 90.0], [110.0, 90.0]],   # Cell 3
    ]
    ocr_texts = ["Header A", "Header B", "Data A1", "Data B1"]

    populated_html, records = CellMatcher.match_and_populate_html(
        pred_html=pred_html,
        cell_bboxes=cell_bboxes,
        ocr_boxes=ocr_boxes,
        ocr_texts=ocr_texts,
    )

    assert "<td>Header A</td>" in populated_html
    assert "<td>Header B</td>" in populated_html
    assert "<td>Data A1</td>" in populated_html
    assert "<td>Data B1</td>" in populated_html
    assert len(records) == 4
    assert records[0]["text"] == "Header A"
