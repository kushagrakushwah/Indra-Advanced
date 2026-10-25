"""
Indra-Advanced & Indra-Original Document Intelligence Playground
Hosted on Hugging Face Spaces with ZeroGPU.

Uses official datalab-to/indra (indra-ocr) for Indra Original (~1M params)
and Indra-Advanced (~90M params: SLANet + PP-OCRv4 + Associative Serializer + GLiNER).
"""

import os
import io
import time
import json
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pypdfium2 as pdfium
import gradio as gr

# ZeroGPU decorator for dynamic A100 GPU acceleration on Hugging Face Spaces
try:
    import spaces
    GPU_DECORATOR = spaces.GPU
except Exception:
    def GPU_DECORATOR(func):
        return func

# Load Indra-Advanced pipeline
from indra_advanced.pipeline import IndraAdvancedPipeline
from indra_advanced.schemas.base_schema import ExtractionSchema

pipeline = IndraAdvancedPipeline()

# Load official Indra Original (datalab-to/indra)
try:
    import indra_ocr
    INDRA_OCR_AVAILABLE = True
    indra_converter = indra_ocr.load()
except Exception as e:
    INDRA_OCR_AVAILABLE = False
    indra_converter = None

# Preconfigured Schemas
PRECONFIGURED_SCHEMAS = {
    "Academic Book": [
        "book_title",
        "author_name",
        "qualifications",
        "designation",
        "institution",
        "location",
        "library_stamp",
        "accession_number",
        "publisher",
        "distribution_cities",
    ],
    "Invoice / Receipt": [
        "vendor_name",
        "invoice_number",
        "invoice_date",
        "total_amount",
        "tax_amount",
        "item_description",
        "quantity",
        "unit_price",
    ],
    "Financial / Bank Statement": [
        "account_holder",
        "bank_name",
        "account_number",
        "closing_balance",
        "statement_date",
        "transaction_amount",
    ],
}


def _get_page_pil(file_path: str, page_number: int = 1) -> Tuple[Image.Image, int]:
    """Load a specific page (1-indexed) as a PIL RGB image and return (image, total_pages)."""
    if not file_path or not os.path.exists(file_path):
        raise ValueError("Invalid file path provided")

    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        doc = pdfium.PdfDocument(file_path)
        total_pages = len(doc)
        idx = max(0, min(page_number - 1, total_pages - 1))
        page = doc[idx]
        pil_img = page.render(scale=2.0).to_pil().convert("RGB")
        doc.close()
        return pil_img, total_pages
    else:
        pil_img = Image.open(file_path).convert("RGB")
        return pil_img, 1


def draw_styled_annotations(
    pil_img: Image.Image,
    items: List[Dict[str, Any]],
    default_border_color: Tuple[int, int, int] = (26, 115, 232),
    default_badge_bg: Tuple[int, int, int] = (232, 240, 254),
    default_badge_text: Tuple[int, int, int] = (23, 78, 166),
) -> Image.Image:
    """
    Renders bounding boxes with pill badges matching the Indra visual inspection layout.
    items: List of dicts with keys:
      - 'box': [x1, y1, x2, y2]
      - 'label': str (e.g. 'Section-Header', 'Text', 'Page-Header', 'book_title', etc.)
      - 'is_container': bool (whether this is a macro block like Section-Header)
      - 'border_color': optional tuple
      - 'badge_bg': optional tuple
      - 'badge_text': optional tuple
    """
    img_copy = pil_img.copy()
    overlay = Image.new("RGBA", img_copy.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_solid = ImageDraw.Draw(img_copy)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for item in items:
        box = item.get("box")
        if not box or len(box) != 4:
            continue

        x1, y1, x2, y2 = box
        label = item.get("label", "Text")
        is_container = item.get("is_container", False)

        border = item.get("border_color", default_border_color)
        badge_bg = item.get("badge_bg", default_badge_bg)
        badge_txt = item.get("badge_text", default_badge_text)

        # Draw semi-transparent fill
        fill_alpha = 35 if is_container else 20
        draw_overlay.rectangle([x1, y1, x2, y2], fill=(border[0], border[1], border[2], fill_alpha))

        # Draw solid outline
        width = 2 if is_container else 1
        draw_solid.rectangle([x1, y1, x2, y2], outline=border, width=width)

        # Draw badge above box
        if label:
            badge_w = len(label) * 6 + 8
            badge_h = 13
            by1 = max(0, y1 - badge_h) if y1 > badge_h else y1
            by2 = by1 + badge_h
            draw_solid.rectangle([x1, by1, x1 + badge_w, by2], fill=badge_bg, outline=border, width=1)
            draw_solid.text((x1 + 3, by1 + 1), label, fill=badge_txt, font=font)

    result = Image.alpha_composite(img_copy.convert("RGBA"), overlay).convert("RGB")
    return result


def on_file_uploaded(file):
    """Handle document upload and update page controls."""
    if file is None:
        return (
            gr.update(visible=False),
            gr.update(visible=False, minimum=1, maximum=2, value=1),
            "No file uploaded",
            None,
        )

    file_path = file.name if hasattr(file, "name") else str(file)
    try:
        pil_img, total_pages = _get_page_pil(file_path, 1)
        is_multipage = total_pages > 1
        page_info = f"Document loaded: {total_pages} page{'s' if total_pages > 1 else ''}"
        slider_max = max(2, total_pages)
        return (
            gr.update(visible=is_multipage),
            gr.update(visible=is_multipage, minimum=1, maximum=slider_max, value=1, step=1),
            page_info,
            pil_img,
        )
    except Exception as e:
        return (
            gr.update(visible=False),
            gr.update(visible=False, minimum=1, maximum=2, value=1),
            f"Error loading file: {e}",
            None,
        )


def on_page_changed(file, page_num):
    """Update rendered preview when page slider changes."""
    if file is None:
        return None, "No file uploaded"
    file_path = file.name if hasattr(file, "name") else str(file)
    try:
        pil_img, total_pages = _get_page_pil(file_path, int(page_num))
        page_info = f"Displaying Page {int(page_num)} of {total_pages}"
        return pil_img, page_info
    except Exception as e:
        return None, f"Error rendering page: {e}"


@GPU_DECORATOR
def process_document(
    file,
    page_num: int,
    model_choice: str,
    schema_choice: str,
    custom_labels: str,
):
    """Run model inference on the selected page and generate structured outputs."""
    if file is None:
        empty_df = [["-", "No document uploaded", "-"]]
        return None, empty_df, {}, "Please upload a document or select an example above.", "", None, None, [], []

    file_path = file.name if hasattr(file, "name") else str(file)
    pil_img, total_pages = _get_page_pil(file_path, int(page_num))
    img_array = np.array(pil_img)

    # Build schema
    if schema_choice == "Custom" and custom_labels.strip():
        labels = [l.strip() for l in custom_labels.split(",") if l.strip()]
        schema = ExtractionSchema.from_labels("custom", labels)
    else:
        labels = PRECONFIGURED_SCHEMAS.get(schema_choice, PRECONFIGURED_SCHEMAS["Academic Book"])
        schema = ExtractionSchema.from_labels(schema_choice.lower().replace(" ", "_"), labels)

    indra_orig_data = None
    indra_orig_annotated = None
    indra_orig_table = []

    # 1. Process with Indra Original (datalab-to/indra)
    if model_choice in ("Indra Original (1M)", "Side-by-Side Analysis"):
        if INDRA_OCR_AVAILABLE and indra_converter is not None:
            t0 = time.perf_counter()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                pil_img.save(tmp.name)
                tmp_path = tmp.name
            try:
                pages_res = list(indra_converter.convert(tmp_path))
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if pages_res:
                    _, indra_page = pages_res[0]
                    orig_items = []

                    # 1. Add layout blocks (Section-Header, Page-Header, Table, Text)
                    if hasattr(indra_page, "blocks") and indra_page.blocks:
                        for b in indra_page.blocks:
                            lbl = b.get("label", "Block")
                            bbox = b.get("bbox", [])
                            if len(bbox) == 4:
                                is_sec = "Header" in lbl or "Table" in lbl
                                border = (26, 115, 232) if not is_sec else (21, 101, 192)
                                bg = (232, 240, 254) if not is_sec else (227, 242, 253)
                                txt_c = (23, 78, 166) if not is_sec else (13, 71, 161)
                                orig_items.append({
                                    "box": [round(c, 1) for c in bbox],
                                    "label": lbl,
                                    "is_container": is_sec,
                                    "border_color": border,
                                    "badge_bg": bg,
                                    "badge_text": txt_c,
                                })

                    # 2. Add lines
                    orig_lines = []
                    for idx, line in enumerate(indra_page.lines, 1):
                        conf_val = round(float(line.conf), 2)
                        orig_lines.append([idx, line.text, conf_val])
                        # If no macro blocks, draw line-level boxes
                        if not indra_page.blocks:
                            orig_items.append({
                                "box": [round(c, 1) for c in line.box],
                                "label": "Text",
                                "is_container": False,
                            })

                    indra_orig_table = orig_lines
                    indra_orig_annotated = draw_styled_annotations(pil_img, orig_items)
                    indra_orig_data = {
                        "latency_ms": round(elapsed_ms, 2),
                        "markdown": indra_page.markdown or "",
                        "num_lines": len(orig_lines),
                        "lines": orig_lines,
                    }
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            indra_orig_table = [["-", "indra-ocr package not available in environment", "-"]]
            indra_orig_annotated = pil_img.copy()
            indra_orig_data = {"error": "indra-ocr not available"}

    # 2. Process with Indra-Advanced (~90M)
    indra_adv_data = None
    indra_adv_annotated = None
    indra_adv_table = []

    if model_choice in ("Indra-Advanced (~90M)", "Side-by-Side Analysis"):
        t0 = time.perf_counter()
        adv_res = pipeline.process(img_array, schema=schema)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        adv_items = []
        # Build set of extracted entity texts for matching
        entity_map = {}
        for ent in adv_res.raw_entities:
            entity_map[ent.text.strip().lower()] = ent.label

        # Collect lines & match with extracted entities
        if adv_res.ocr_result and adv_res.ocr_result.boxes:
            for idx, (box, text, score) in enumerate(
                zip(adv_res.ocr_result.boxes, adv_res.ocr_result.texts, adv_res.ocr_result.scores),
                1,
            ):
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                bbox = [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]
                conf_val = round(float(score), 2)
                adv_adv_text = text.strip()
                indra_adv_table.append([idx, adv_adv_text, conf_val])

                # Check if this line is an extracted schema entity
                matched_label = None
                adv_lower = adv_adv_text.lower()
                for e_text, e_label in entity_map.items():
                    if e_text in adv_lower or adv_lower in e_text:
                        matched_label = e_label
                        break

                if matched_label:
                    # Entity badge (Emerald Green)
                    adv_items.append({
                        "box": bbox,
                        "label": matched_label,
                        "is_container": True,
                        "border_color": (16, 185, 129),
                        "badge_bg": (236, 253, 245),
                        "badge_text": (6, 95, 70),
                    })
                else:
                    # Text line badge (Blue)
                    adv_items.append({
                        "box": bbox,
                        "label": "Text",
                        "is_container": False,
                        "border_color": (37, 99, 235),
                        "badge_bg": (239, 246, 255),
                        "badge_text": (30, 64, 175),
                    })

        # Add detected table regions if any
        if adv_res.cell_bboxes:
            for cell in adv_res.cell_bboxes[:10]:
                adv_items.append({
                    "box": [round(c, 1) for c in cell],
                    "label": "Table-Cell",
                    "is_container": False,
                    "border_color": (139, 92, 246),
                    "badge_bg": (245, 243, 255),
                    "badge_text": (109, 40, 217),
                })

        indra_adv_annotated = draw_styled_annotations(pil_img, adv_items)
        indra_adv_data = {
            "latency_ms": round(elapsed_ms, 2),
            "structured_json": adv_res.structured_json,
            "markdown_table": adv_res.markdown_table,
            "serialized_text": adv_res.serialized_text,
            "timing_breakdown_ms": adv_res.timing_breakdown_ms,
            "num_lines": len(indra_adv_table),
            "lines": indra_adv_table,
        }

    # Format return outputs depending on choice
    if model_choice == "Indra Original (1M)":
        display_img = indra_orig_annotated or pil_img
        display_table = indra_orig_table
        display_json = {
            "model": "Indra Original (datalab-to/indra)",
            "parameter_size": "~1M parameters",
            "limitation": "Indra Original performs line detection and OCR only. Typed JSON schema extraction requires an external LLM.",
            "total_lines": len(indra_orig_table),
        }
        display_md = f"### Indra Original Markdown Output\n\n```markdown\n{indra_orig_data.get('markdown', '')}\n```"
        metrics_md = (
            f"**Model:** Indra Original (`datalab-to/indra`) | "
            f"**Latency:** {indra_orig_data.get('latency_ms', 'N/A')} ms | "
            f"**Parameters:** ~1M | "
            f"**Detected Lines:** {len(indra_orig_table)}"
        )
        return display_img, display_table, display_json, display_md, metrics_md, indra_adv_annotated, indra_orig_annotated, indra_adv_table, indra_orig_table

    elif model_choice == "Indra-Advanced (~90M)":
        display_img = indra_adv_annotated or pil_img
        display_table = indra_adv_table
        display_json = indra_adv_data["structured_json"]
        display_md = (
            f"### Reconstructed Table & Serialized Text\n\n"
            f"{indra_adv_data['markdown_table'] or indra_adv_data['serialized_text']}"
        )
        metrics_md = (
            f"**Model:** Indra-Advanced | "
            f"**Total Latency:** {indra_adv_data['latency_ms']} ms | "
            f"**Parameters:** ~90M | "
            f"**Detected Lines:** {len(indra_adv_table)}\n\n"
            f"**Stage Breakdown:** "
            + " | ".join(f"{k}: {v}ms" for k, v in indra_adv_data["timing_breakdown_ms"].items())
        )
        return display_img, display_table, display_json, display_md, metrics_md, indra_adv_annotated, indra_orig_annotated, indra_adv_table, indra_orig_table

    else:  # Side-by-Side Analysis
        # Active view default is Indra-Advanced
        display_img = indra_adv_annotated or pil_img
        display_table = indra_adv_table
        display_json = {
            "indra_advanced_schema_extraction": indra_adv_data["structured_json"],
            "indra_original_limitation": "Indra Original outputs raw line tokens and cannot extract schema fields without an external LLM.",
        }
        orig_md = indra_orig_data.get("markdown", "N/A") if indra_orig_data else "N/A"
        adv_table_md = indra_adv_data["markdown_table"] or indra_adv_data["serialized_text"]
        display_md = (
            f"### Side-by-Side Text & Table Comparison\n\n"
            f"#### Indra-Advanced Reconstructed Table\n{adv_table_md}\n\n"
            f"---\n\n"
            f"#### Indra Original Raw Markdown Dump\n```markdown\n{orig_md}\n```"
        )
        orig_time = indra_orig_data.get("latency_ms", "N/A") if indra_orig_data else "N/A"
        metrics_md = (
            f"| Capability | Indra Original (`datalab-to/indra`) | Indra-Advanced (~90M) |\n"
            f"| :--- | :--- | :--- |\n"
            f"| **Core Parameters** | ~1M parameters | ~77M - 90M parameters |\n"
            f"| **Inference Latency** | {orig_time} ms | {indra_adv_data['latency_ms']} ms |\n"
            f"| **Extracted Schema** | None (Raw lines dump) | Typed JSON (`{schema_choice}`) |\n"
            f"| **Table Reconstruction** | Line clustering | Associative Header-to-Value pairs |\n"
            f"| **Fine-Print / Stamp OCR** | Prone to character drops | High-precision DBNet + CRNN |\n"
            f"| **External LLM Required?** | Yes | No (Zero-shot onboard) |"
        )
        return display_img, display_table, display_json, display_md, metrics_md, indra_adv_annotated, indra_orig_annotated, indra_adv_table, indra_orig_table


# Build Gradio UI
with gr.Blocks(title="Indra Document Intelligence") as demo:
    gr.Markdown(
        """
        # Indra Document Intelligence: Original vs Advanced
        Visual inspection, multi-page book navigation, and zero-shot schema extraction.
        
        Comparing **Indra Original (`datalab-to/indra` ~1M)** with **Indra-Advanced (~90M)**.
        """
    )

    # State stores for switching visual inspection between models in Side-by-Side mode
    adv_img_state = gr.State(None)
    orig_img_state = gr.State(None)
    adv_table_state = gr.State([])
    orig_table_state = gr.State([])

    with gr.Row():
        file_input = gr.File(
            label="Upload Document (PDF Book or Image)",
            file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"],
            type="filepath",
            scale=2,
        )

        with gr.Column(scale=2):
            with gr.Group(visible=False) as page_nav_group:
                page_info_label = gr.Markdown("Page 1 of 1")
                page_slider = gr.Slider(
                    minimum=1,
                    maximum=2,
                    value=1,
                    step=1,
                    label="Page Navigation",
                )

            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=["Indra-Advanced (~90M)", "Indra Original (1M)", "Side-by-Side Analysis"],
                    value="Side-by-Side Analysis",
                    label="Active Model",
                )

                schema_dropdown = gr.Dropdown(
                    choices=["Academic Book", "Invoice / Receipt", "Financial / Bank Statement", "Custom"],
                    value="Academic Book",
                    label="Extraction Schema",
                )

            custom_labels_input = gr.Textbox(
                label="Custom Entity Labels (comma-separated)",
                placeholder="e.g. customer_name, total_usd, tax_rate, date",
                visible=False,
            )

            run_btn = gr.Button("Run Document Extraction", variant="primary")

    # Built-in Examples
    example_paths = []
    if os.path.exists("theory_of_computation.png"):
        example_paths.append(["theory_of_computation.png", "Academic Book"])
    if os.path.exists("demo_invoice.png"):
        example_paths.append(["demo_invoice.png", "Invoice / Receipt"])

    if example_paths:
        gr.Examples(
            examples=example_paths,
            inputs=[file_input, schema_dropdown],
            label="Quick Test Examples",
        )

    gr.Markdown("---")

    # Main Visual & Line Inspection View (Exactly like user's screenshot!)
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                view_toggle = gr.Radio(
                    choices=["Indra-Advanced Annotations", "Indra Original Annotations"],
                    value="Indra-Advanced Annotations",
                    label="Visual Annotation Switcher",
                )
            image_output = gr.Image(
                label="Document Visual Inspection (Bounding Boxes & Badges)",
                type="pil",
            )

        with gr.Column(scale=1):
            gr.Markdown("### Extracted Text Lines & Confidence Scores")
            lines_dataframe = gr.Dataframe(
                headers=["#", "Extracted Text", "Confidence"],
                datatype=["number", "str", "number"],
                column_count=(3, "fixed"),
                interactive=False,
                label="Line-by-Line OCR Output",
            )

    # Detailed Structured & Comparative Intelligence Below
    metrics_display = gr.Markdown("Ready to process document.")

    with gr.Tabs():
        with gr.TabItem("Structured JSON Schema (Zero-Shot)"):
            json_output = gr.JSON(label="Extracted Typed Schema Fields")

        with gr.TabItem("Table Structure & Serialized Markdown"):
            markdown_output = gr.Markdown()

    # Event handlers
    def toggle_custom_schema(schema_name):
        return gr.update(visible=(schema_name == "Custom"))

    schema_dropdown.change(
        fn=toggle_custom_schema,
        inputs=[schema_dropdown],
        outputs=[custom_labels_input],
    )

    file_input.change(
        fn=on_file_uploaded,
        inputs=[file_input],
        outputs=[page_nav_group, page_slider, page_info_label, image_output],
    )

    page_slider.change(
        fn=on_page_changed,
        inputs=[file_input, page_slider],
        outputs=[image_output, page_info_label],
    )

    def switch_annotation_view(choice, adv_img, orig_img, adv_table, orig_table):
        if choice == "Indra Original Annotations":
            img = orig_img if orig_img is not None else adv_img
            tbl = orig_table if orig_table else adv_table
            return img, tbl
        else:
            img = adv_img if adv_img is not None else orig_img
            tbl = adv_table if adv_table else orig_table
            return img, tbl

    view_toggle.change(
        fn=switch_annotation_view,
        inputs=[view_toggle, adv_img_state, orig_img_state, adv_table_state, orig_table_state],
        outputs=[image_output, lines_dataframe],
    )

    run_btn.click(
        fn=process_document,
        inputs=[
            file_input,
            page_slider,
            model_dropdown,
            schema_dropdown,
            custom_labels_input,
        ],
        outputs=[
            image_output,
            lines_dataframe,
            json_output,
            markdown_output,
            metrics_display,
            adv_img_state,
            orig_img_state,
            adv_table_state,
            orig_table_state,
        ],
    )

if __name__ == "__main__":
    demo.launch()
