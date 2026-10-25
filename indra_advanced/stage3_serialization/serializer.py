"""
Stage 3: Serialization Module Interface.
Acts as the critical bridge converting populated HTML + bounding coordinates into structured linear text.
"""

from typing import Optional, List, Dict, Any
from indra_advanced.config import Stage3SerializationConfig, SerializationMode
from indra_advanced.stage3_serialization.formats import (
    ParsedTable,
    TableHTMLParser,
    HeaderValueSerializer,
    MarkdownTableSerializer,
    CoordinateAnnotatedSerializer,
    HierarchicalSerializer,
)


class TableSerializer:
    """
    Main serializer engine coordinating table parsing and linear formatting.
    Preserves row-column spatial relationships for downstream zero-shot extraction.
    """

    def __init__(self, config: Optional[Stage3SerializationConfig] = None):
        self.config = config or Stage3SerializationConfig()

    def serialize(
        self,
        html_str: str,
        cell_bboxes: Optional[List[List[float]]] = None,
        mode: Optional[SerializationMode] = None,
    ) -> str:
        """
        Serialize populated table HTML into structured linear text.
        
        Args:
            html_str: Populated HTML <table> string.
            cell_bboxes: Optional list of bounding boxes for each cell.
            mode: SerializationMode override (defaults to config.primary_mode).
            
        Returns:
            Structured linear text preserving row/column relationships.
        """
        target_mode = mode or self.config.primary_mode
        parsed = TableHTMLParser.parse(html_str, cell_bboxes)

        if target_mode == SerializationMode.HEADER_VALUE:
            return HeaderValueSerializer.serialize(parsed, delimiter=self.config.cell_delimiter)
        elif target_mode == SerializationMode.MARKDOWN_TABLE:
            return MarkdownTableSerializer.serialize(parsed)
        elif target_mode == SerializationMode.COORDINATE_ANNOTATED:
            return CoordinateAnnotatedSerializer.serialize(parsed)
        elif target_mode == SerializationMode.HIERARCHICAL:
            return HierarchicalSerializer.serialize(parsed)
        else:
            return HeaderValueSerializer.serialize(parsed)

    def parse_to_grid(self, html_str: str, cell_bboxes: Optional[List[List[float]]] = None) -> ParsedTable:
        """Parse raw HTML into resolved 2D ParsedTable grid."""
        return TableHTMLParser.parse(html_str, cell_bboxes)
