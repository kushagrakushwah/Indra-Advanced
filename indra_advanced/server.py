"""
FastAPI REST API & Interactive Visual Inspection Frontend.
Features:
  1. Model Switcher: Toggle between Indra-Advanced (~90M), Indra Original (1M), or Side-by-Side Comparison
  2. Multi-page document support: Upload entire PDF books or single images with page navigation
  3. SVG bounding-box inspection with line confidence scores and structured schema cards
"""

import os
import uuid
import io
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import pypdfium2 as pdfium

from indra_advanced.pipeline import IndraAdvancedPipeline
from indra_advanced.config import PipelineConfig, OCRModelTier, SerializationMode
from indra_advanced.schemas.base_schema import ExtractionSchema
from indra_advanced.utils.image_utils import load_image

# Optional import of Indra OCR
try:
    import indra_ocr
    INDRA_OCR_AVAILABLE = True
    indra_converter = indra_ocr.load()
except Exception:
    INDRA_OCR_AVAILABLE = False
    indra_converter = None

app = FastAPI(
    title="Indra-Advanced & Indra-Original Document Intelligence",
    description="Multi-page book and document intelligence playground with model switching",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = IndraAdvancedPipeline()

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# In-memory document sessions store
SESSIONS: Dict[str, Dict[str, Any]] = {}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "indra_advanced_ready": True,
        "indra_original_available": INDRA_OCR_AVAILABLE,
    }


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF book or image) and return session info including total page count.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    filename = file.filename or "uploaded_document"
    ext = Path(filename).suffix.lower()
    session_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{session_id}{ext}"

    with open(file_path, "wb") as f:
        f.write(contents)

    is_pdf = ext == ".pdf"
    total_pages = 1

    if is_pdf:
        try:
            doc = pdfium.PdfDocument(str(file_path))
            total_pages = len(doc)
            doc.close()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid PDF file: {e}")

    SESSIONS[session_id] = {
        "file_path": str(file_path),
        "filename": filename,
        "is_pdf": is_pdf,
        "total_pages": total_pages,
    }

    return {
        "session_id": session_id,
        "filename": filename,
        "is_pdf": is_pdf,
        "total_pages": total_pages,
    }


def _get_page_image(session_id: str, page_num: int) -> Image.Image:
    """Helper to render or load PIL Image for a given session and page index."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    sess = SESSIONS[session_id]
    file_path = sess["file_path"]

    if sess["is_pdf"]:
        doc = pdfium.PdfDocument(file_path)
        if page_num < 0 or page_num >= len(doc):
            doc.close()
            raise HTTPException(status_code=400, detail="Page number out of range")
        page = doc[page_num]
        pil_img = page.render(scale=2.0).to_pil().convert("RGB")
        doc.close()
        return pil_img
    else:
        return Image.open(file_path).convert("RGB")


@app.get("/api/page-image/{session_id}/{page_num}")
def get_page_image_endpoint(session_id: str, page_num: int):
    """Returns rendered JPEG image of a document page."""
    pil_img = _get_page_image(session_id, page_num)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


@app.post("/api/process-page")
async def process_page(
    session_id: str = Form(...),
    page_num: int = Form(0),
    model_type: str = Form("indra_advanced"),  # indra_advanced, indra_original, or compare
    schema_type: str = Form("academic_book"),
    custom_labels: Optional[str] = Form(None),
):
    """
    Run extraction on the selected page using selected model:
    - indra_advanced (~90M params, SLANet + PP-OCR + GLiNER)
    - indra_original (1M params, indra-ocr)
    - compare (runs both and returns side-by-side output)
    """
    pil_img = _get_page_image(session_id, page_num)
    width, height = pil_img.size
    img_array = np.array(pil_img)
    sess = SESSIONS[session_id]

    # Resolve schema
    if custom_labels and custom_labels.strip():
        labels = [l.strip() for l in custom_labels.split(",") if l.strip()]
        schema = ExtractionSchema.from_labels("custom", labels)
    elif schema_type == "academic_book":
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

    response_data: Dict[str, Any] = {
        "session_id": session_id,
        "page_num": page_num,
        "total_pages": sess["total_pages"],
        "image_width": width,
        "image_height": height,
        "model_type": model_type,
    }

    # 1. Process with Indra Original
    if model_type in ("indra_original", "compare"):
        if not INDRA_OCR_AVAILABLE or indra_converter is None:
            response_data["indra_original"] = {"error": "indra_ocr package not available"}
        else:
            t0 = time.perf_counter()
            # Save temporary page image for indra_ocr
            temp_page_path = UPLOAD_DIR / f"temp_{session_id}_{page_num}.png"
            pil_img.save(temp_page_path)
            try:
                _, indra_page = next(indra_converter.convert(str(temp_page_path)))
                elapsed_ms = (time.perf_counter() - t0) * 1000

                indra_lines = []
                for idx, line in enumerate(indra_page.lines):
                    indra_lines.append({
                        "id": idx + 1,
                        "text": line.text,
                        "conf": round(float(line.conf), 2),
                        "box": [round(c, 1) for c in line.box],
                    })

                response_data["indra_original"] = {
                    "latency_ms": round(elapsed_ms, 2),
                    "parameter_size": "~1M parameters",
                    "lines": indra_lines,
                    "markdown": indra_page.markdown,
                    "num_lines": len(indra_lines),
                }
            finally:
                if temp_page_path.exists():
                    temp_page_path.unlink()

    # 2. Process with Indra-Advanced
    if model_type in ("indra_advanced", "compare"):
        t0 = time.perf_counter()
        adv_res = pipeline.process(img_array, schema=schema)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        adv_lines = []
        if adv_res.ocr_result and adv_res.ocr_result.boxes:
            for idx, (box, text, score) in enumerate(zip(adv_res.ocr_result.boxes, adv_res.ocr_result.texts, adv_res.ocr_result.scores)):
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                adv_lines.append({
                    "id": idx + 1,
                    "text": text,
                    "conf": round(float(score), 2),
                    "box": [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)],
                })

        response_data["indra_advanced"] = {
            "latency_ms": round(elapsed_ms, 2),
            "parameter_size": "~77M - 90M parameters",
            "lines": adv_lines,
            "structured_json": adv_res.structured_json,
            "markdown_table": adv_res.markdown_table,
            "serialized_text": adv_res.serialized_text,
            "timing_breakdown_ms": adv_res.timing_breakdown_ms,
            "num_lines": len(adv_lines),
        }

    return JSONResponse(response_data)


@app.get("/", response_class=HTMLResponse)
def index_web_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Indra-Advanced & Indra-Original Playground</title>
        <style>
            * { box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; background: #0f172a; color: #f8fafc; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
            header { background: #1e293b; padding: 10px 20px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #334155; flex-wrap: wrap; gap: 10px; }
            .brand { display: flex; align-items: center; gap: 8px; font-weight: bold; font-size: 16px; color: #38bdf8; }
            .badge { background: #0284c7; color: white; padding: 3px 8px; border-radius: 4px; font-size: 11px; }
            .toolbar { display: flex; align-items: center; gap: 12px; }
            select, button, .file-btn { background: #334155; color: white; border: 1px solid #475569; padding: 7px 12px; border-radius: 6px; font-size: 13px; font-weight: 500; cursor: pointer; outline: none; }
            select:hover, button:hover, .file-btn:hover { background: #475569; }
            .btn-primary { background: #38bdf8; color: #0f172a; font-weight: 600; border: none; }
            .btn-primary:hover { background: #7dd3fc; }
            .page-nav { display: flex; align-items: center; gap: 6px; background: #0f172a; padding: 4px 10px; border-radius: 6px; border: 1px solid #334155; font-size: 13px; }
            .page-btn { padding: 3px 8px; font-size: 12px; }
            
            .main-container { display: flex; flex: 1; overflow: hidden; }
            .viewer-pane { flex: 1.1; padding: 15px; overflow-y: auto; display: flex; flex-direction: column; align-items: center; background: #090d16; border-right: 1px solid #334155; position: relative; }
            .canvas-container { position: relative; display: inline-block; box-shadow: 0 4px 25px rgba(0,0,0,0.7); border-radius: 6px; overflow: hidden; }
            #docImg { max-width: 100%; height: auto; display: block; }
            #svgOverlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
            
            .results-pane { flex: 1.3; display: flex; flex-direction: column; background: #1e293b; }
            .tab-bar { display: flex; border-bottom: 1px solid #334155; background: #182234; }
            .tab-btn { flex: 1; padding: 11px; background: none; border: none; color: #94a3b8; font-weight: 600; cursor: pointer; border-bottom: 2px solid transparent; font-size: 13px; }
            .tab-btn.active { color: #38bdf8; border-bottom: 2px solid #38bdf8; background: #1e293b; }
            .tab-body { flex: 1; overflow-y: auto; padding: 16px; }
            
            .line-item { background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 8px 12px; margin-bottom: 6px; display: flex; align-items: center; justify-content: space-between; transition: all 0.15s; }
            .line-item:hover, .line-item.active { border-color: #38bdf8; background: #132238; }
            .line-idx { background: #334155; color: #38bdf8; width: 22px; height: 22px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: bold; margin-right: 10px; }
            .line-txt { flex: 1; font-size: 13px; color: #f1f5f9; }
            .line-score { background: #064e3b; color: #34d399; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 600; margin-left: 8px; }
            
            .schema-card { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 12px 14px; margin-bottom: 10px; }
            .schema-key { font-size: 11px; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.5px; margin-bottom: 4px; }
            .schema-val { font-size: 14px; color: #38bdf8; font-weight: 500; }
            .city-tag { display: inline-block; background: #1e293b; border: 1px solid #475569; padding: 2px 7px; border-radius: 4px; font-size: 12px; margin: 2px; color: #e2e8f0; }
            
            .markdown-view { background: #090d16; padding: 14px; border-radius: 6px; border: 1px solid #334155; font-family: monospace; font-size: 13px; white-space: pre-wrap; color: #a5f3fc; line-height: 1.5; }
            .status-banner { padding: 10px 14px; border-radius: 6px; margin-bottom: 12px; font-size: 13px; display: flex; justify-content: space-between; align-items: center; }
            .banner-info { background: #1e293b; border: 1px solid #38bdf8; color: #38bdf8; }
            input[type="file"] { display: none; }
        </style>
    </head>
    <body>
        <header>
            <div class="brand">
                Indra-Advanced <span class="badge">Model Switcher & Book Reader</span>
            </div>

            <div class="toolbar">
                <!-- Model Switcher Dropdown -->
                <label style="font-size: 12px; color: #94a3b8;">Engine:</label>
                <select id="modelSelect" onchange="triggerReprocess()">
                    <option value="indra_advanced" selected>Indra-Advanced (~90M, Structured JSON)</option>
                    <option value="indra_original">Indra Original (1M, Fast Raw OCR)</option>
                    <option value="compare">Side-by-Side Comparison</option>
                </select>

                <!-- Schema Selector -->
                <label style="font-size: 12px; color: #94a3b8;">Schema:</label>
                <select id="schemaSelect" onchange="triggerReprocess()">
                    <option value="academic_book" selected>Academic Book / Title Page</option>
                    <option value="invoice">Invoice / Bill of Sale</option>
                </select>

                <!-- Page Navigator (For Books / Multi-page PDFs) -->
                <div class="page-nav" id="pageNav" style="display: none;">
                    <button class="page-btn" onclick="prevPage()">◀</button>
                    <span id="pageIndicator">Page 1 / 1</span>
                    <button class="page-btn" onclick="nextPage()">▶</button>
                </div>

                <label class="btn-primary file-btn" for="fileInput">Upload PDF Book or Image</label>
                <input type="file" id="fileInput" accept=".pdf,image/*" onchange="handleFileUpload()">
            </div>
        </header>

        <div class="main-container">
            <!-- Left Pane: Image Canvas with SVG Bounding Boxes -->
            <div class="viewer-pane">
                <div class="canvas-container" id="canvasWrapper" style="display: none;">
                    <img id="docImg" alt="Document Page">
                    <svg id="svgOverlay"></svg>
                </div>
                <div id="emptyNotice" style="margin-top: 150px; text-align: center; color: #64748b;">
                    <h3>No Document Loaded</h3>
                    <p style="font-size: 13px;">Upload an image or a full multi-page PDF book above.</p>
                </div>
            </div>

            <!-- Right Pane: Switchable Tabs (JSON Schema vs OCR Lines vs Raw Markdown) -->
            <div class="results-pane">
                <div class="tab-bar">
                    <button class="tab-btn active" id="tabSchemaBtn" onclick="setTab('schema')">Structured Schema JSON</button>
                    <button class="tab-btn" id="tabLinesBtn" onclick="setTab('lines')">OCR Lines & Bounding Boxes</button>
                    <button class="tab-btn" id="tabMarkdownBtn" onclick="setTab('markdown')">Raw Markdown (Indra Original)</button>
                </div>

                <div class="tab-body" id="tabBodySchema">
                    <div id="schemaView" style="color: #64748b;">Awaiting extraction...</div>
                </div>

                <div class="tab-body" id="tabBodyLines" style="display: none;">
                    <div id="linesView" style="color: #64748b;">Awaiting extraction...</div>
                </div>

                <div class="tab-body" id="tabBodyMarkdown" style="display: none;">
                    <div id="markdownView" class="markdown-view">Awaiting extraction...</div>
                </div>
            </div>
        </div>

        <script>
            let currentSessionId = null;
            let currentPage = 0;
            let totalPages = 1;
            let latestResult = null;

            function setTab(tab) {
                document.getElementById('tabSchemaBtn').classList.toggle('active', tab === 'schema');
                document.getElementById('tabLinesBtn').classList.toggle('active', tab === 'lines');
                document.getElementById('tabMarkdownBtn').classList.toggle('active', tab === 'markdown');

                document.getElementById('tabBodySchema').style.display = tab === 'schema' ? 'block' : 'none';
                document.getElementById('tabBodyLines').style.display = tab === 'lines' ? 'block' : 'none';
                document.getElementById('tabBodyMarkdown').style.display = tab === 'markdown' ? 'block' : 'none';
            }

            async function handleFileUpload() {
                const file = document.getElementById('fileInput').files[0];
                if (!file) return;

                const formData = new FormData();
                formData.append('file', file);

                document.getElementById('emptyNotice').innerHTML = "<div style='color:#38bdf8;'>Uploading & preparing pages...</div>";

                try {
                    const resp = await fetch('/api/upload', { method: 'POST', body: formData });
                    const data = await resp.json();
                    
                    currentSessionId = data.session_id;
                    totalPages = data.total_pages;
                    currentPage = 0;

                    // Configure page navigator
                    const pageNav = document.getElementById('pageNav');
                    if (totalPages > 1) {
                        pageNav.style.display = 'flex';
                        updatePageIndicator();
                    } else {
                        pageNav.style.display = 'none';
                    }

                    document.getElementById('emptyNotice').style.display = 'none';
                    document.getElementById('canvasWrapper').style.display = 'inline-block';

                    loadCurrentPage();
                } catch (err) {
                    alert('Upload failed: ' + err);
                }
            }

            function updatePageIndicator() {
                document.getElementById('pageIndicator').textContent = `Page ${currentPage + 1} / ${totalPages}`;
            }

            function prevPage() {
                if (currentPage > 0) {
                    currentPage--;
                    updatePageIndicator();
                    loadCurrentPage();
                }
            }

            function nextPage() {
                if (currentPage < totalPages - 1) {
                    currentPage++;
                    updatePageIndicator();
                    loadCurrentPage();
                }
            }

            function triggerReprocess() {
                if (currentSessionId) {
                    loadCurrentPage();
                }
            }

            async function loadCurrentPage() {
                if (!currentSessionId) return;

                const docImg = document.getElementById('docImg');
                docImg.src = `/api/page-image/${currentSessionId}/${currentPage}`;

                const modelType = document.getElementById('modelSelect').value;
                const schemaType = document.getElementById('schemaSelect').value;

                document.getElementById('schemaView').innerHTML = `<div class="status-banner banner-info">Processing page with <b>${modelType}</b>...</div>`;
                document.getElementById('linesView').innerHTML = `<div class="status-banner banner-info">Extracting bounding boxes...</div>`;
                document.getElementById('markdownView').textContent = "Processing...";

                const formData = new FormData();
                formData.append('session_id', currentSessionId);
                formData.append('page_num', currentPage);
                formData.append('model_type', modelType);
                formData.append('schema_type', schemaType);

                try {
                    const resp = await fetch('/api/process-page', { method: 'POST', body: formData });
                    latestResult = await resp.json();
                    renderResults(latestResult);
                } catch (err) {
                    alert('Processing error: ' + err);
                }
            }

            function renderResults(data) {
                const svg = document.getElementById('svgOverlay');
                svg.innerHTML = '';
                svg.setAttribute('viewBox', `0 0 ${data.image_width} ${data.image_height}`);

                const modelType = data.model_type;
                let activeLines = [];

                if (modelType === 'indra_advanced' && data.indra_advanced) {
                    activeLines = data.indra_advanced.lines || [];
                    renderSchema(data.indra_advanced.structured_json, data.indra_advanced.latency_ms, data.indra_advanced.parameter_size);
                    renderLines(activeLines);
                    document.getElementById('markdownView').textContent = data.indra_advanced.markdown_table || "(Indra-Advanced outputs structured JSON and table grids)";
                } else if (modelType === 'indra_original' && data.indra_original) {
                    activeLines = data.indra_original.lines || [];
                    document.getElementById('schemaView').innerHTML = `<div class="status-banner banner-info">Indra Original outputs raw text only (no schema extraction). See Raw Markdown tab.</div>`;
                    renderLines(activeLines);
                    document.getElementById('markdownView').textContent = data.indra_original.markdown || "(No markdown returned)";
                    setTab('markdown');
                } else if (modelType === 'compare') {
                    // Compare side by side
                    activeLines = (data.indra_advanced && data.indra_advanced.lines) ? data.indra_advanced.lines : (data.indra_original ? data.indra_original.lines : []);
                    renderComparison(data);
                    renderLines(activeLines);
                }

                // Render SVG Bounding Boxes
                activeLines.forEach(line => {
                    const [xmin, ymin, xmax, ymax] = line.box;
                    const w = xmax - xmin;
                    const h = ymax - ymin;

                    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    rect.setAttribute('x', xmin);
                    rect.setAttribute('y', ymin);
                    rect.setAttribute('width', w);
                    rect.setAttribute('height', h);
                    rect.setAttribute('fill', 'rgba(56, 189, 248, 0.15)');
                    rect.setAttribute('stroke', '#38bdf8');
                    rect.setAttribute('stroke-width', '2');
                    rect.id = 'rect-' + line.id;
                    svg.appendChild(rect);

                    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    text.setAttribute('x', xmin + 3);
                    text.setAttribute('y', ymin + 13);
                    text.setAttribute('fill', '#38bdf8');
                    text.setAttribute('font-size', '11px');
                    text.setAttribute('font-weight', 'bold');
                    text.textContent = line.id;
                    svg.appendChild(text);
                });
            }

            function renderLines(lines) {
                const container = document.getElementById('linesView');
                container.innerHTML = '';
                if (!lines || lines.length === 0) {
                    container.innerHTML = '<div>No text lines detected.</div>';
                    return;
                }

                lines.forEach(l => {
                    const div = document.createElement('div');
                    div.className = 'line-item';
                    div.id = 'card-' + l.id;
                    div.innerHTML = `
                        <div class="line-idx">${l.id}</div>
                        <div class="line-txt">${l.text}</div>
                        <div class="line-score">${l.conf}</div>
                    `;

                    div.onmouseenter = () => {
                        const r = document.getElementById('rect-' + l.id);
                        if (r) { r.setAttribute('stroke', '#f43f5e'); r.setAttribute('fill', 'rgba(244, 63, 94, 0.35)'); }
                        div.classList.add('active');
                    };
                    div.onmouseleave = () => {
                        const r = document.getElementById('rect-' + l.id);
                        if (r) { r.setAttribute('stroke', '#38bdf8'); r.setAttribute('fill', 'rgba(56, 189, 248, 0.15)'); }
                        div.classList.remove('active');
                    };

                    container.appendChild(div);
                });
            }

            function renderSchema(jsonObj, latency, size) {
                const container = document.getElementById('schemaView');
                container.innerHTML = `
                    <div class="status-banner banner-info">
                        <span><b>Model:</b> Indra-Advanced (${size})</span>
                        <span><b>Latency:</b> ${latency} ms</span>
                    </div>
                `;

                if (!jsonObj || Object.keys(jsonObj).length === 0) {
                    container.innerHTML += '<div>No schema entities detected.</div>';
                    return;
                }

                for (const [k, v] of Object.entries(jsonObj)) {
                    if (k === 'line_items') continue;
                    const card = document.createElement('div');
                    card.className = 'schema-card';
                    let displayVal = v;
                    if (Array.isArray(v)) {
                        displayVal = v.map(item => `<span class="city-tag">${item}</span>`).join(' ');
                    }
                    card.innerHTML = `
                        <div class="schema-key">${k.replace(/_/g, ' ')}</div>
                        <div class="schema-val">${displayVal}</div>
                    `;
                    container.appendChild(card);
                }
            }

            function renderComparison(data) {
                const container = document.getElementById('schemaView');
                const adv = data.indra_advanced || {};
                const orig = data.indra_original || {};

                container.innerHTML = `
                    <div style="display:flex; gap:10px; margin-bottom:14px;">
                        <div style="flex:1; background:#0f172a; border:1px solid #a855f7; padding:10px; border-radius:6px;">
                            <div style="color:#c084fc; font-weight:bold; font-size:12px;">INDRA ORIGINAL</div>
                            <div style="font-size:20px; font-weight:bold; margin-top:4px;">${orig.latency_ms || 0} ms</div>
                            <div style="font-size:11px; color:#94a3b8;">~1M params • Raw Markdown Only</div>
                        </div>
                        <div style="flex:1; background:#0f172a; border:1px solid #38bdf8; padding:10px; border-radius:6px;">
                            <div style="color:#38bdf8; font-weight:bold; font-size:12px;">INDRA ADVANCED</div>
                            <div style="font-size:20px; font-weight:bold; margin-top:4px;">${adv.latency_ms || 0} ms</div>
                            <div style="font-size:11px; color:#94a3b8;">~90M params • Typed Schema JSON</div>
                        </div>
                    </div>
                `;

                if (adv.structured_json) {
                    for (const [k, v] of Object.entries(adv.structured_json)) {
                        if (k === 'line_items') continue;
                        const card = document.createElement('div');
                        card.className = 'schema-card';
                        let displayVal = v;
                        if (Array.isArray(v)) {
                            displayVal = v.map(item => `<span class="city-tag">${item}</span>`).join(' ');
                        }
                        card.innerHTML = `
                            <div class="schema-key">${k.replace(/_/g, ' ')}</div>
                            <div class="schema-val">${displayVal}</div>
                        `;
                        container.appendChild(card);
                    }
                }

                document.getElementById('markdownView').textContent = orig.markdown || "(No original markdown)";
            }
        </script>
    </body>
    </html>
    """
