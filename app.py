"""
Indra-Advanced & Indra-Original Document Intelligence Playground
Hosted on Hugging Face Spaces with ZeroGPU.

Features:
- Multi-page document navigation (entire PDF books or single images)
- Model Switcher: Indra-Advanced (~90M params), Indra Original (1M params), Side-by-Side Comparison
- Zero-Shot Typed JSON Schema Extraction (GLiNER)
- Table Structure Recognition & Associative Serialization (SLANet + PP-OCRv4)
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

# Load Indra Original if available
try:
    import indra_ocr
    INDRA_OCR_AVAILABLE = True
    indra_converter = indra_ocr.load()
except Exception:
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


def draw_bounding_boxes(
    pil_img: Image.Image,
    boxes: List[List[float]],
    labels: Optional[List[str]] = None,
    color: str = "#007acc",
    fill_alpha: int = 40,
) -> Image.Image:
    """Draw bounding boxes with transparent overlays and legible tags."""
    img_copy = pil_img.copy()
    overlay = Image.new("RGBA", img_copy.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    solid_draw = ImageDraw.Draw(img_copy)

    # Parse hex color
    hex_color = color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    fill_color = (r, g, b, fill_alpha)
    outline_color = (r, g, b, 220)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, box in enumerate(boxes):
        if len(box) == 4:
            x1, y1, x2, y2 = box
        else:
            continue

        overlay_draw.rectangle([x1, y1, x2, y2], fill=fill_color)
        solid_draw.rectangle([x1, y1, x2, y2], outline=outline_color[:3], width=2)

        if labels and i < len(labels) and labels[i]:
            tag = labels[i]
            solid_draw.rectangle([x1, max(0, y1 - 14), x1 + len(tag) * 6 + 6, y1], fill=(r, g, b))
            solid_draw.text((x1 + 3, max(0, y1 - 13)), tag, fill=(255, 255, 255), font=font)

    img_copy.paste(Image.alpha_composite(img_copy.convert("RGBA"), overlay).convert("RGB"))
    return img_copy


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
        return None, {}, "Please upload a document or select an example above.", ""

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

    results = {}
    annotated_img = pil_img.copy()
    markdown_output = ""
    json_output = {}
    metrics_md = ""

    # 1. Indra Original Inference
    indra_orig_data = None
    if model_choice in ("Indra Original (1M)", "Side-by-Side Comparison"):
        if INDRA_OCR_AVAILABLE and indra_converter is not None:
            t0 = time.perf_counter()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                pil_img.save(tmp.name)
                tmp_path = tmp.name
            try:
                _, indra_page = next(indra_converter.convert(tmp_path))
                elapsed_ms = (time.perf_counter() - t0) * 1000
                lines = []
                for idx, line in enumerate(indra_page.lines):
                    lines.append({
                        "id": idx + 1,
                        "text": line.text,
                        "conf": round(float(line.conf), 2),
                        "box": [round(c, 1) for c in line.box],
                    })
                indra_orig_data = {
                    "latency_ms": round(elapsed_ms, 2),
                    "markdown": indra_page.markdown,
                    "num_lines": len(lines),
                    "lines": lines,
                }
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            indra_orig_data = {
                "error": "indra-ocr package is not available in current environment",
                "markdown": "Indra OCR unavailable",
            }

    # 2. Indra-Advanced Inference
    indra_adv_data = None
    if model_choice in ("Indra-Advanced (~90M)", "Side-by-Side Comparison"):
        t0 = time.perf_counter()
        adv_res = pipeline.process(img_array, schema=schema)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        boxes = []
        labels_list = []
        if adv_res.ocr_result and adv_res.ocr_result.boxes:
            for box, text in zip(adv_res.ocr_result.boxes, adv_res.ocr_result.texts):
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                boxes.append([min(xs), min(ys), max(xs), max(ys)])
                labels_list.append(text[:20])

        indra_adv_data = {
            "latency_ms": round(elapsed_ms, 2),
            "structured_json": adv_res.structured_json,
            "markdown_table": adv_res.markdown_table,
            "serialized_text": adv_res.serialized_text,
            "timing_breakdown_ms": adv_res.timing_breakdown_ms,
            "num_lines": len(boxes),
            "boxes": boxes,
            "labels": labels_list,
        }

    # Render outputs according to model choice
    if model_choice == "Indra Original (1M)":
        if indra_orig_data and "lines" in indra_orig_data:
            orig_boxes = [l["box"] for l in indra_orig_data["lines"]]
            orig_labels = [f"#{l['id']}" for l in indra_orig_data["lines"]]
            annotated_img = draw_bounding_boxes(pil_img, orig_boxes, orig_labels, color="#e65100")
            markdown_output = indra_orig_data["markdown"]
            json_output = {
                "info": "Indra Original produces raw text & lines only. Schema extraction requires an external LLM.",
                "total_lines_detected": indra_orig_data["num_lines"],
                "lines": indra_orig_data["lines"][:25],
            }
            metrics_md = f"**Latency:** {indra_orig_data['latency_ms']} ms | **Parameters:** ~1M | **OCR Lines:** {indra_orig_data['num_lines']}"
        else:
            markdown_output = indra_orig_data.get("error", "Error running Indra Original")

    elif model_choice == "Indra-Advanced (~90M)":
        if indra_adv_data:
            annotated_img = draw_bounding_boxes(
                pil_img,
                indra_adv_data["boxes"],
                indra_adv_data["labels"],
                color="#00838f",
            )
            json_output = indra_adv_data["structured_json"]
            markdown_output = (
                f"### Reconstructed Table & Serialized Text\n\n"
                f"{indra_adv_data['markdown_table'] or indra_adv_data['serialized_text']}"
            )
            metrics_md = (
                f"**Total Latency:** {indra_adv_data['latency_ms']} ms | "
                f"**Parameters:** ~90M | "
                f"**Detected Lines:** {indra_adv_data['num_lines']}\n\n"
                f"**Stage Breakdown:** "
                + " | ".join(f"{k}: {v}ms" for k, v in indra_adv_data["timing_breakdown_ms"].items())
            )

    else:  # Side-by-Side Comparison
        if indra_adv_data:
            annotated_img = draw_bounding_boxes(
                pil_img,
                indra_adv_data["boxes"],
                indra_adv_data["labels"],
                color="#00838f",
            )
            json_output = {
                "indra_advanced_schema_extraction": indra_adv_data["structured_json"],
                "indra_original_limitation": "Indra Original output is unstructured raw text; cannot extract typed schema without external LLM.",
            }

            orig_md = indra_orig_data.get("markdown", "N/A") if indra_orig_data else "N/A"
            adv_table = indra_adv_data["markdown_table"] or indra_adv_data["serialized_text"]

            markdown_output = (
                f"### Side-by-Side Reconstruction\n\n"
                f"#### Indra-Advanced Output\n"
                f"{adv_table}\n\n"
                f"---\n\n"
                f"#### Indra Original Output (Raw Markdown Line Dump)\n"
                f"```markdown\n{orig_md}\n```\n"
            )

            orig_time = indra_orig_data.get("latency_ms", "N/A") if indra_orig_data else "N/A"
            metrics_md = (
                f"| Metric | Indra Original (1M) | Indra-Advanced (~90M) |\n"
                f"| :--- | :--- | :--- |\n"
                f"| **Latency** | {orig_time} ms | {indra_adv_data['latency_ms']} ms |\n"
                f"| **Extracted Schema** | None (Raw Markdown) | Typed JSON (`{schema_choice}`) |\n"
                f"| **Table Reconstruction** | Line clusters | Hierarchical & Markdown table |\n"
                f"| **Fine-Print / Stamp** | Prone to character drops | High-precision DBNet + CRNN |\n"
                f"| **External LLM Needed?** | Yes | No (Zero-shot onboard) |"
            )

    return annotated_img, json_output, markdown_output, metrics_md


# Build Gradio UI
with gr.Blocks(title="Indra-Advanced Document Intelligence") as demo:
    gr.Markdown(
        """
        # Indra-Advanced: Document & Table Intelligence
        A lightweight pipeline combining table structure recognition, OCR, associative serialization, and zero-shot schema extraction.
        
        Compare **Indra Original (1M params)** vs **Indra-Advanced (~90M params)** on academic books, invoices, and multi-page PDFs.
        """
    )

    with gr.Row():
        # Left Column: Inputs & Controls
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload Document (PDF Book or Image)",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"],
                type="filepath",
            )

            with gr.Group(visible=False) as page_nav_group:
                page_info_label = gr.Markdown("Page 1 of 1")
                page_slider = gr.Slider(
                    minimum=1,
                    maximum=2,
                    value=1,
                    step=1,
                    label="Page Navigation",
                )

            model_dropdown = gr.Dropdown(
                choices=["Indra-Advanced (~90M)", "Indra Original (1M)", "Side-by-Side Comparison"],
                value="Indra-Advanced (~90M)",
                label="Model Selection",
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
                    label="Quick Examples",
                )

        # Right Column: Visual and Structured Results
        with gr.Column(scale=2):
            metrics_display = gr.Markdown("Ready to process document.")

            with gr.Tabs():
                with gr.TabItem("Visual Inspection"):
                    image_output = gr.Image(label="Annotated Page with Detected Regions", type="pil")

                with gr.TabItem("Structured JSON"):
                    json_output = gr.JSON(label="Extracted Schema Fields")

                with gr.TabItem("Markdown & Table Output"):
                    markdown_output = gr.Markdown()

    # Dynamic interactions
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
            json_output,
            markdown_output,
            metrics_display,
        ],
    )

if __name__ == "__main__":
    demo.launch()
