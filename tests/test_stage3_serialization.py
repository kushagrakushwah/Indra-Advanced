"""
Unit tests for Stage 3: Custom Serialization Logic.
Verifies preservation of row/column relationships ('Header: Value').
"""

import pytest
from indra_advanced.stage3_serialization.formats import (
    TableHTMLParser,
    HeaderValueSerializer,
    MarkdownTableSerializer,
    CoordinateAnnotatedSerializer,
    HierarchicalSerializer,
)
from indra_advanced.stage3_serialization.serializer import TableSerializer
from indra_advanced.config import SerializationMode


SAMPLE_HTML = """
<table>
    <tr><th>Item</th><th>Quantity</th><th>Price</th></tr>
    <tr><td>Product Alpha</td><td>2</td><td>$50.00</td></tr>
    <tr><td>Product Beta</td><td>5</td><td>$10.00</td></tr>
</table>
"""

SAMPLE_HTML_WITH_SPAN = """
<table>
    <tr><th colspan="2">System Specs</th><th>Cost</th></tr>
    <tr><td>RAM</td><td>32GB</td><td>$120</td></tr>
    <tr><td>SSD</td><td>1TB</td><td>$80</td></tr>
</table>
"""


def test_html_parser_dimensions():
    parsed = TableHTMLParser.parse(SAMPLE_HTML)
    assert parsed.num_rows == 3
    assert parsed.num_cols == 3
    assert parsed.column_headers == ["Item", "Quantity", "Price"]


def test_header_value_serializer():
    parsed = TableHTMLParser.parse(SAMPLE_HTML)
    serialized = HeaderValueSerializer.serialize(parsed)

    assert "Item: Product Alpha | Quantity: 2 | Price: $50.00" in serialized
    assert "Item: Product Beta | Quantity: 5 | Price: $10.00" in serialized


def test_markdown_table_serializer():
    parsed = TableHTMLParser.parse(SAMPLE_HTML)
    md = MarkdownTableSerializer.serialize(parsed)

    assert "| Item | Quantity | Price |" in md
    assert "| --- | --- | --- |" in md
    assert "| Product Alpha | 2 | $50.00 |" in md


def test_colspan_handling():
    parsed = TableHTMLParser.parse(SAMPLE_HTML_WITH_SPAN)
    assert parsed.num_cols == 3
    # Check that colspan cell spans cols 0 and 1
    assert parsed.rows[0][0].text == "System Specs"
    assert parsed.rows[0][1].text == "System Specs"


def test_coordinate_annotated_serializer():
    bboxes = [
        [10, 10, 50, 30], [50, 10, 90, 30], [90, 10, 140, 30],
        [10, 35, 50, 55], [50, 35, 90, 55], [90, 35, 140, 55],
        [10, 60, 50, 80], [50, 60, 90, 80], [90, 60, 140, 80],
    ]
    parsed = TableHTMLParser.parse(SAMPLE_HTML, cell_bboxes=bboxes)
    coord_text = CoordinateAnnotatedSerializer.serialize(parsed)

    assert "[bbox:" in coord_text
    assert "Item: Product Alpha [bbox:10,35,50,55]" in coord_text


def test_table_serializer_modes():
    serializer = TableSerializer()
    hv = serializer.serialize(SAMPLE_HTML, mode=SerializationMode.HEADER_VALUE)
    md = serializer.serialize(SAMPLE_HTML, mode=SerializationMode.MARKDOWN_TABLE)

    assert "Item: Product Alpha" in hv
    assert "| Product Alpha |" in md
