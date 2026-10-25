"""
Stage 3: Serialization Formats and Algorithms.
Converts populated HTML and cell bounding box matrices into semantically structured linear text.
Preserves row/column spatial relationships ("Header: Value" pairings).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from bs4 import BeautifulSoup, Tag
import re


@dataclass
class CellData:
    """Represents a single table cell with text and geometry."""
    row_idx: int
    col_idx: int
    text: str
    is_header: bool = False
    rowspan: int = 1
    colspan: int = 1
    bbox: Optional[List[float]] = None  # [xmin, ymin, xmax, ymax]


@dataclass
class ParsedTable:
    """Represents a fully resolved 2D grid matrix of table cells."""
    rows: List[List[Optional[CellData]]] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0
    raw_html: str = ""
    column_headers: List[str] = field(default_factory=list)
    row_headers: List[str] = field(default_factory=list)


class TableHTMLParser:
    """Parses populated HTML table string into a 2D resolved matrix with span handling."""

    @staticmethod
    def parse(html_str: str, cell_bboxes: Optional[List[List[float]]] = None) -> ParsedTable:
        """Parse HTML table, expanding rowspan and colspan accurately into a 2D grid."""
        soup = BeautifulSoup(html_str, "html.parser")
        table_tag = soup.find("table")
        if not table_tag:
            # Fallback if bare tr/td passed
            table_tag = soup

        tr_tags = table_tag.find_all("tr")
        if not tr_tags:
            return ParsedTable(raw_html=html_str)

        # Estimate max grid size
        grid: List[List[Optional[CellData]]] = []
        cell_counter = 0

        for r_idx, tr in enumerate(tr_tags):
            while len(grid) <= r_idx:
                grid.append([])

            c_idx = 0
            cells = tr.find_all(["th", "td"])
            for cell in cells:
                # Find next free slot in current row
                while c_idx < len(grid[r_idx]) and grid[r_idx][c_idx] is not None:
                    c_idx += 1

                rowspan = int(cell.get("rowspan", 1))
                colspan = int(cell.get("colspan", 1))
                is_header = cell.name == "th" or "header" in cell.get("class", [])
                text = cell.get_text(separator=" ", strip=True)

                bbox = cell_bboxes[cell_counter] if (cell_bboxes and cell_counter < len(cell_bboxes)) else None
                cell_counter += 1

                cell_obj = CellData(
                    row_idx=r_idx,
                    col_idx=c_idx,
                    text=text,
                    is_header=is_header,
                    rowspan=rowspan,
                    colspan=colspan,
                    bbox=bbox,
                )

                # Fill grid accounting for spans
                for r in range(r_idx, r_idx + rowspan):
                    while len(grid) <= r:
                        grid.append([])
                    for c in range(c_idx, c_idx + colspan):
                        while len(grid[r]) <= c:
                            grid[r].append(None)
                        grid[r][c] = cell_obj

                c_idx += colspan

        num_rows = len(grid)
        num_cols = max((len(r) for r in grid), default=0)

        # Normalize grid to exact rectangular shape
        for r in grid:
            while len(r) < num_cols:
                r.append(None)

        # Extract column headers (usually row 0 or all <th> in top rows)
        col_headers: List[str] = []
        if num_rows > 0:
            first_row = grid[0]
            col_headers = [cell.text if cell else f"Col_{i}" for i, cell in enumerate(first_row)]

        # Extract row headers (column 0 if marked header or key column)
        row_headers: List[str] = []
        if num_cols > 0:
            for r in grid:
                c0 = r[0]
                row_headers.append(c0.text if c0 else "")

        return ParsedTable(
            rows=grid,
            num_rows=num_rows,
            num_cols=num_cols,
            raw_html=html_str,
            column_headers=col_headers,
            row_headers=row_headers,
        )


class HeaderValueSerializer:
    """
    Serializes table into associative "Header: Value" statements per row.
    This is the primary serialization format shown in the architecture diagram:
    'Preserve row/column relationships (e.g. Header: Value). Critical bridge.'
    """

    @staticmethod
    def serialize(parsed_table: ParsedTable, delimiter: str = " | ") -> str:
        if parsed_table.num_rows == 0 or parsed_table.num_cols == 0:
            return ""

        headers = parsed_table.column_headers
        lines: List[str] = []

        # Determine if row 0 was headers
        start_row = 1 if any(cell and cell.is_header for cell in parsed_table.rows[0]) else 1

        for r_idx in range(start_row, parsed_table.num_rows):
            row = parsed_table.rows[r_idx]
            pairs: List[str] = []
            visited_cells = set()

            for c_idx, cell in enumerate(row):
                if cell is None or id(cell) in visited_cells:
                    continue
                visited_cells.add(id(cell))

                cell_text = cell.text.strip()
                if not cell_text:
                    continue

                header_name = headers[c_idx] if c_idx < len(headers) and headers[c_idx] else f"Field_{c_idx+1}"
                pairs.append(f"{header_name}: {cell_text}")

            if pairs:
                lines.append(delimiter.join(pairs))

        return "\n".join(lines)


class MarkdownTableSerializer:
    """Serializes table to clean GitHub-Flavored Markdown syntax."""

    @staticmethod
    def serialize(parsed_table: ParsedTable) -> str:
        if parsed_table.num_rows == 0 or parsed_table.num_cols == 0:
            return ""

        lines: List[str] = []
        headers = [h.replace("|", "\\|").replace("\n", " ") for h in parsed_table.column_headers]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for r_idx in range(1, parsed_table.num_rows):
            row = parsed_table.rows[r_idx]
            row_texts = []
            for cell in row:
                t = cell.text.replace("|", "\\|").replace("\n", " ") if cell else ""
                row_texts.append(t)
            lines.append("| " + " | ".join(row_texts) + " |")

        return "\n".join(lines)


class CoordinateAnnotatedSerializer:
    """
    Serializes text while preserving normalized bounding coordinates [x1, y1, x2, y2].
    Enables spatial reasoning and document grounding for downstream extractors.
    """

    @staticmethod
    def serialize(parsed_table: ParsedTable) -> str:
        lines: List[str] = []
        headers = parsed_table.column_headers

        for r_idx in range(1, parsed_table.num_rows):
            row = parsed_table.rows[r_idx]
            items: List[str] = []
            visited = set()
            for c_idx, cell in enumerate(row):
                if cell is None or id(cell) in visited:
                    continue
                visited.add(id(cell))
                header = headers[c_idx] if c_idx < len(headers) else f"Col_{c_idx}"
                coord_str = ""
                if cell.bbox:
                    coord_str = f" [bbox:{int(cell.bbox[0])},{int(cell.bbox[1])},{int(cell.bbox[2])},{int(cell.bbox[3])}]"
                items.append(f"{header}: {cell.text}{coord_str}")
            if items:
                lines.append(" | ".join(items))

        return "\n".join(lines)


class HierarchicalSerializer:
    """
    Handles complex multi-level headers, section rows (rows with 1 cell spanning all cols),
    and parent-child nested column hierarchies.
    """

    @staticmethod
    def serialize(parsed_table: ParsedTable) -> str:
        if parsed_table.num_rows == 0:
            return ""

        lines: List[str] = []
        headers = parsed_table.column_headers
        current_section = ""

        for r_idx in range(1, parsed_table.num_rows):
            row = parsed_table.rows[r_idx]
            non_empty = [c for c in row if c and c.text.strip()]
            if not non_empty:
                continue

            # Check if this row is a section break / header spanning whole table
            if len(non_empty) == 1 and non_empty[0].colspan >= parsed_table.num_cols - 1:
                current_section = non_empty[0].text.strip()
                lines.append(f"### Section: {current_section}")
                continue

            row_tokens = []
            if current_section:
                row_tokens.append(f"[{current_section}]")

            visited = set()
            for c_idx, cell in enumerate(row):
                if cell is None or id(cell) in visited:
                    continue
                visited.add(id(cell))
                h = headers[c_idx] if c_idx < len(headers) else f"Col_{c_idx}"
                row_tokens.append(f"{h}: {cell.text.strip()}")

            lines.append(" | ".join(row_tokens))

        return "\n".join(lines)
