"""
Synthetic Table Document Generator.
Creates synthetic document images with realistic table structures and text
for verification, benchmarks, and unit testing.
"""

from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class SyntheticTableGenerator:
    """Generates synthetic high-resolution table images for testing pipeline stages."""

    @staticmethod
    def create_invoice_table_image(
        width: int = 800,
        height: int = 500,
        title: str = "INVOICE #INV-2024-9841",
    ) -> Tuple[np.ndarray, str]:
        """
        Creates a synthetic invoice document with a structured line items table.
        Returns (image_array, expected_markdown).
        """
        img = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        try:
            font_title = ImageFont.truetype("arial.ttf", 22)
            font_bold = ImageFont.truetype("arial.ttf", 15)
            font_regular = ImageFont.truetype("arial.ttf", 14)
        except IOError:
            font_title = ImageFont.load_default()
            font_bold = ImageFont.load_default()
            font_regular = ImageFont.load_default()

        # Header section
        draw.text((40, 30), title, fill=(20, 20, 20), font=font_title)
        draw.text((40, 65), "Vendor: Acme Cloud Solutions Inc.", fill=(50, 50, 50), font=font_regular)
        draw.text((40, 85), "Date: 2024-10-15  |  Due: 2024-11-15", fill=(50, 50, 50), font=font_regular)
        draw.text((40, 105), "Customer: Global Logistics Corp", fill=(50, 50, 50), font=font_regular)

        # Table dimensions
        start_x, start_y = 40, 140
        row_height = 36
        col_widths = [220, 90, 120, 130]
        table_w = sum(col_widths)

        headers = ["Item Description", "Quantity", "Unit Price", "Total Amount"]
        rows = [
            ["Cloud Compute Cluster", "4", "$120.00", "$480.00"],
            ["Managed Storage 10TB", "2", "$45.00", "$90.00"],
            ["API Gateway Ingestion", "10", "$15.00", "$150.00"],
            ["Enterprise Support Tier", "1", "$250.00", "$250.00"],
        ]

        # Draw header row background
        draw.rectangle(
            [start_x, start_y, start_x + table_w, start_y + row_height],
            fill=(240, 240, 245),
            outline=(100, 100, 100),
            width=2,
        )

        # Draw header texts
        curr_x = start_x
        for i, header in enumerate(headers):
            draw.text((curr_x + 10, start_y + 9), header, fill=(0, 0, 0), font=font_bold)
            curr_x += col_widths[i]
            if i < len(headers) - 1:
                draw.line([(curr_x, start_y), (curr_x, start_y + row_height)], fill=(120, 120, 120), width=1)

        # Draw data rows
        curr_y = start_y + row_height
        for row in rows:
            draw.rectangle(
                [start_x, curr_y, start_x + table_w, curr_y + row_height],
                outline=(180, 180, 180),
                width=1,
            )
            curr_x = start_x
            for i, val in enumerate(row):
                draw.text((curr_x + 10, curr_y + 9), val, fill=(30, 30, 30), font=font_regular)
                curr_x += col_widths[i]
                if i < len(row) - 1:
                    draw.line([(curr_x, curr_y), (curr_x, curr_y + row_height)], fill=(200, 200, 200), width=1)
            curr_y += row_height

        # Summary footer
        curr_y += 15
        draw.text((start_x + 360, curr_y), "Subtotal: $970.00", fill=(30, 30, 30), font=font_bold)
        draw.text((start_x + 360, curr_y + 22), "Tax (10%): $97.00", fill=(30, 30, 30), font=font_regular)
        draw.text((start_x + 360, curr_y + 44), "Total Due: $1,067.00", fill=(0, 0, 0), font=font_title)

        expected_markdown = (
            "| Item Description | Quantity | Unit Price | Total Amount |\n"
            "| --- | --- | --- | --- |\n"
            "| Cloud Compute Cluster | 4 | $120.00 | $480.00 |\n"
            "| Managed Storage 10TB | 2 | $45.00 | $90.00 |\n"
            "| API Gateway Ingestion | 10 | $15.00 | $150.00 |\n"
            "| Enterprise Support Tier | 1 | $250.00 | $250.00 |"
        )

        return np.array(img), expected_markdown
