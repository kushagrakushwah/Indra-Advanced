"""
FastAPI REST API & Interactive Visual Inspection Frontend for Indra-Advanced.
Provides:
  1. Side-by-side interactive document canvas with SVG bounding box overlays (like Surya/Marker/Indra UI)
  2. Numbered line-by-line inspector with confidence scores
  3. Structured zero-shot schema extraction cards
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import json
import io
from PIL import Image
import numpy as np

from indra_advanced.pipeline import IndraAdvancedPipeline
from indra_advanced.config import PipelineConfig, OCRModelTier, SerializationMode
from indra_advanced.schemas.base_schema import ExtractionSchema
from indra_advanced.utils.image_utils import load_image

app = FastAPI(
    title="Indra-Advanced Visual Inspection UI",
    description="Interactive Document Intelligence and Bounding Box Extraction Inspector",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = IndraAdvancedPipeline()


@app.get("/health")
def health():
    return {"status": "healthy", "service": "indra-advanced"}


@app.post("/api/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    schema_type: str = Form("academic_book"),
):
    """
    Process document image and return OCR bounding boxes, lines, confidences,
    and structured schema extraction for the visual frontend.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    width, height = pil_img.size
    img_array = np.array(pil_img)

    # Define schema
    if schema_type == "academic_book":
        schema = ExtractionSchema.from_labels(
            "academic_book",
            [
                "book_title",
                "author_name",
                "qualifications",
                "designation",
                "institution",
                "location",
                "library_stamp",
                "accession_number",
                "copy_status",
                "accession_date",
                "publisher",
                "publisher_imprint",
                "distribution_cities",
            ],
        )
    else:
        schema = ExtractionSchema.from_labels(
            "invoice",
            ["vendor_name", "invoice_number", "invoice_date", "total_amount", "item_description", "quantity", "unit_price"],
        )

    # Run extraction
    result = pipeline.process(img_array, schema=schema)

    # Format bounding box lines for the visual inspector
    ocr_lines = []
    if result.ocr_result and result.ocr_result.boxes:
        for idx, (box, text, score) in enumerate(zip(result.ocr_result.boxes, result.ocr_result.texts, result.ocr_result.scores)):
            # Normalize 4 points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] to [xmin, ymin, xmax, ymax]
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            ocr_lines.append({
                "id": idx + 1,
                "text": text,
                "conf": round(float(score), 2),
                "box": [min(xs), min(ys), max(xs), max(ys)],
            })

    return JSONResponse({
        "image_width": width,
        "image_height": height,
        "lines": ocr_lines,
        "structured_json": result.structured_json,
        "timing_breakdown_ms": result.timing_breakdown_ms,
        "total_elapsed_ms": result.total_elapsed_ms,
    })


@app.get("/", response_class=HTMLResponse)
def index_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Indra-Advanced Visual Document Inspector</title>
        <style>
            * { box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; background: #0f172a; color: #f8fafc; height: 100vh; display: flex; flex-direction: column; }
            header { background: #1e293b; padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #334155; }
            h1 { font-size: 18px; margin: 0; color: #38bdf8; display: flex; align-items: center; gap: 8px; }
            .badge { background: #0284c7; color: white; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }
            .main-content { display: flex; flex: 1; overflow: hidden; }
            .left-panel { flex: 1.2; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; align-items: center; background: #090d16; border-right: 1px solid #334155; }
            .canvas-wrapper { position: relative; display: inline-block; box-shadow: 0 4px 20px rgba(0,0,0,0.6); border-radius: 8px; overflow: hidden; }
            #docImg { max-width: 100%; height: auto; display: block; }
            #overlaySvg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
            .right-panel { flex: 1.2; display: flex; flex-direction: column; background: #1e293b; }
            .tab-nav { display: flex; border-bottom: 1px solid #334155; background: #182234; }
            .tab-btn { flex: 1; padding: 12px; background: none; border: none; color: #94a3b8; font-weight: 600; cursor: pointer; border-bottom: 2px solid transparent; }
            .tab-btn.active { color: #38bdf8; border-bottom: 2px solid #38bdf8; background: #1e293b; }
            .tab-content { flex: 1; overflow-y: auto; padding: 20px; }
            .line-card { background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 8px 12px; margin-bottom: 8px; display: flex; align-items: center; justify-content: space-between; transition: all 0.2s; }
            .line-card:hover, .line-card.active { border-color: #38bdf8; background: #132238; }
            .line-index { background: #334155; color: #38bdf8; width: 24px; height: 24px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; margin-right: 10px; }
            .line-text { flex: 1; font-size: 13px; color: #f1f5f9; }
            .line-conf { background: #064e3b; color: #34d399; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 600; margin-left: 10px; }
            .schema-card { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px; margin-bottom: 12px; }
            .schema-title { font-size: 11px; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.5px; margin-bottom: 4px; }
            .schema-value { font-size: 14px; color: #38bdf8; font-weight: 500; }
            .tag { display: inline-block; background: #1e293b; border: 1px solid #475569; padding: 3px 8px; border-radius: 4px; font-size: 12px; margin: 2px; color: #e2e8f0; }
            .controls { display: flex; gap: 10px; }
            input[type="file"] { display: none; }
            .btn { background: #38bdf8; color: #0f172a; padding: 8px 16px; border-radius: 6px; font-weight: 600; font-size: 13px; border: none; cursor: pointer; }
            .btn:hover { background: #7dd3fc; }
        </style>
    </head>
    <body>
        <header>
            <h1>Indra-Advanced <span class="badge">Visual Inspector</span></h1>
            <div class="controls">
                <label class="btn" for="fileInput">Upload Document</label>
                <input type="file" id="fileInput" accept="image/*" onchange="uploadImage()">
            </div>
        </header>

        <div class="main-content">
            <!-- Left: Document Image Canvas with SVG BBoxes -->
            <div class="left-panel">
                <div class="canvas-wrapper" id="canvasWrapper" style="display:none;">
                    <img id="docImg" src="" alt="Document">
                    <svg id="overlaySvg"></svg>
                </div>
                <div id="placeholderText" style="margin-top: 100px; color: #64748b; font-size: 15px;">
                    Upload a book cover, table, or invoice image to inspect extraction with bounding boxes.
                </div>
            </div>

            <!-- Right: Tabbed Inspector (Line OCR List vs Structured Schema JSON) -->
            <div class="right-panel">
                <div class="tab-nav">
                    <button class="tab-btn active" onclick="switchTab('lines')">OCR Lines & Confidences</button>
                    <button class="tab-btn" onclick="switchTab('schema')">Structured Schema JSON</button>
                </div>

                <div class="tab-content" id="linesTab">
                    <div id="linesContainer" style="color: #64748b;">No document uploaded yet.</div>
                </div>

                <div class="tab-content" id="schemaTab" style="display:none;">
                    <div id="schemaContainer" style="color: #64748b;">No document uploaded yet.</div>
                </div>
            </div>
        </div>

        <script>
            let currentData = null;

            function switchTab(tab) {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                if (tab === 'lines') {
                    document.querySelectorAll('.tab-btn')[0].classList.add('active');
                    document.getElementById('linesTab').style.display = 'block';
                    document.getElementById('schemaTab').style.display = 'none';
                } else {
                    document.querySelectorAll('.tab-btn')[1].classList.add('active');
                    document.getElementById('linesTab').style.display = 'none';
                    document.getElementById('schemaTab').style.display = 'block';
                }
            }

            async function uploadImage() {
                const file = document.getElementById('fileInput').files[0];
                if (!file) return;

                const wrapper = document.getElementById('canvasWrapper');
                const docImg = document.getElementById('docImg');
                const placeholder = document.getElementById('placeholderText');
                
                docImg.src = URL.createObjectURL(file);
                wrapper.style.display = 'inline-block';
                placeholder.style.display = 'none';

                const formData = new FormData();
                formData.append('file', file);
                formData.append('schema_type', 'academic_book');

                document.getElementById('linesContainer').innerHTML = "<div style='color:#38bdf8;'>Scanning image through all 4 stages on CPU...</div>";
                document.getElementById('schemaContainer').innerHTML = "<div style='color:#38bdf8;'>Extracting zero-shot schema fields...</div>";

                try {
                    const resp = await fetch('/api/analyze', { method: 'POST', body: formData });
                    currentData = await resp.json();
                    renderInspector(currentData);
                } catch (err) {
                    alert("Analysis error: " + err);
                }
            }

            function renderInspector(data) {
                const svg = document.getElementById('overlaySvg');
                const img = document.getElementById('docImg');
                svg.innerHTML = '';
                svg.setAttribute('viewBox', `0 0 ${data.image_width} ${data.image_height}`);

                const linesContainer = document.getElementById('linesContainer');
                linesContainer.innerHTML = '';

                data.lines.forEach(line => {
                    const [xmin, ymin, xmax, ymax] = line.box;
                    const w = xmax - xmin;
                    const h = ymax - ymin;

                    // Draw SVG rectangle with box ID
                    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    rect.setAttribute('x', xmin);
                    rect.setAttribute('y', ymin);
                    rect.setAttribute('width', w);
                    rect.setAttribute('height', h);
                    rect.setAttribute('fill', 'rgba(56, 189, 248, 0.15)');
                    rect.setAttribute('stroke', '#38bdf8');
                    rect.setAttribute('stroke-width', '2');
                    rect.id = 'svg-rect-' + line.id;
                    svg.appendChild(rect);

                    // Add text label index on box
                    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    text.setAttribute('x', xmin + 3);
                    text.setAttribute('y', ymin + 14);
                    text.setAttribute('fill', '#38bdf8');
                    text.setAttribute('font-size', '12px');
                    text.setAttribute('font-weight', 'bold');
                    text.textContent = line.id;
                    svg.appendChild(text);

                    // Create line inspector card on the right
                    const card = document.createElement('div');
                    card.className = 'line-card';
                    card.id = 'card-' + line.id;
                    card.innerHTML = `
                        <div class="line-index">${line.id}</div>
                        <div class="line-text">${line.text}</div>
                        <div class="line-conf">${line.conf}</div>
                    `;

                    // Highlight box on hover
                    card.onmouseenter = () => {
                        rect.setAttribute('stroke', '#f43f5e');
                        rect.setAttribute('fill', 'rgba(244, 63, 94, 0.3)');
                        card.classList.add('active');
                    };
                    card.onmouseleave = () => {
                        rect.setAttribute('stroke', '#38bdf8');
                        rect.setAttribute('fill', 'rgba(56, 189, 248, 0.15)');
                        card.classList.remove('active');
                    };

                    linesContainer.appendChild(card);
                });

                // Render Structured Schema JSON Tab
                const schemaContainer = document.getElementById('schemaContainer');
                schemaContainer.innerHTML = '';

                for (const [key, val] of Object.entries(data.structured_json)) {
                    if (key === 'line_items') continue;
                    const sc = document.createElement('div');
                    sc.className = 'schema-card';
                    let displayVal = val;
                    if (Array.isArray(val)) {
                        displayVal = val.map(v => `<span class="tag">${v}</span>`).join(' ');
                    }
                    sc.innerHTML = `
                        <div class="schema-title">${key.replace(/_/g, ' ')}</div>
                        <div class="schema-value">${displayVal}</div>
                    `;
                    schemaContainer.appendChild(sc);
                }
            }
        </script>
    </body>
    </html>
    """
