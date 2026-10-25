"""
FastAPI REST API & Interactive Web Interface for Indra-Advanced.
Exposes endpoints for high-throughput document extraction.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import json
import io

from indra_advanced.pipeline import IndraAdvancedPipeline
from indra_advanced.config import (
    PipelineConfig,
    OCRModelTier,
    GLiNERModelVariant,
    SerializationMode,
)
from indra_advanced.schemas.domain_schemas import (
    INVOICE_SCHEMA,
    FINANCIAL_SCHEMA,
    MEDICAL_LAB_SCHEMA,
)
from indra_advanced.schemas.base_schema import ExtractionSchema

app = FastAPI(
    title="Indra-Advanced Document Intelligence API",
    description="Lightweight ~70M-110M parameter table structure and zero-shot schema extraction pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline singleton
pipeline = IndraAdvancedPipeline()


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "indra-advanced",
        "estimated_footprint": pipeline.config.total_estimated_parameters,
    }


@app.post("/extract")
async def extract_document(
    file: UploadFile = File(...),
    schema_type: str = Form("invoice"),
    ocr_tier: str = Form("small"),
    gliner_variant: str = Form("gliner_2_5_small"),
    serialization_mode: str = Form("header_value"),
):
    """
    Upload a document image and extract structured JSON matching specified schema.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Select schema
    selected_schema = None
    if schema_type == "invoice":
        selected_schema = INVOICE_SCHEMA
    elif schema_type in ("financial", "finance"):
        selected_schema = FINANCIAL_SCHEMA
    elif schema_type in ("medical", "lab"):
        selected_schema = MEDICAL_LAB_SCHEMA
    else:
        # Custom labels comma-separated
        labels = [l.strip() for l in schema_type.split(",") if l.strip()]
        selected_schema = ExtractionSchema.from_labels("custom", labels)

    result = pipeline.process(contents, schema=selected_schema)
    return JSONResponse(content=result.to_dict())


@app.get("/", response_class=HTMLResponse)
def index_web_ui():
    """Interactive Web Playground for drag-and-drop document testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Indra-Advanced Document Intelligence</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; background: #0f172a; color: #f8fafc; }
            .container { max-width: 900px; margin: 0 auto; background: #1e293b; padding: 30px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
            h1 { color: #38bdf8; margin-top: 0; }
            .badge { background: #0284c7; color: white; padding: 4px 10px; border-radius: 6px; font-size: 13px; font-weight: 600; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 8px; font-weight: 500; color: #94a3b8; }
            select, input[type="file"], input[type="text"] { width: 100%; padding: 10px; background: #334155; border: 1px solid #475569; border-radius: 6px; color: white; box-sizing: border-box; }
            button { background: #38bdf8; color: #0f172a; font-weight: bold; border: none; padding: 12px 24px; border-radius: 6px; cursor: pointer; font-size: 16px; }
            button:hover { background: #7dd3fc; }
            pre { background: #090d16; padding: 16px; border-radius: 8px; overflow-x: auto; color: #a5f3fc; border: 1px solid #334155; }
            .result-box { margin-top: 30px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Indra-Advanced <span class="badge">~70M Params CPU Pipeline</span></h1>
            <p style="color: #cbd5e1;">Combines <b>SLANet-1M</b> (Stage 1), <b>PP-OCRv4</b> (Stage 2), <b>Custom Associative Serialization</b> (Stage 3), and <b>GLiNER</b> (Stage 4).</p>
            
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="form-group">
                    <label>Upload Document Image (PNG, JPG):</label>
                    <input type="file" name="file" id="fileInput" accept="image/*" required />
                </div>
                <div class="form-group">
                    <label>Extraction Schema:</label>
                    <select name="schema_type">
                        <option value="invoice">Invoice / Purchase Order (Vendor, Total, Line Items)</option>
                        <option value="financial">Financial Report (Revenue, Operating Income, Assets)</option>
                        <option value="medical">Medical / Lab Report (Test Name, Result, Range)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>PP-OCRv4 Tier (Stage 2):</label>
                    <select name="ocr_tier">
                        <option value="small" selected>Small (~7.7M params - Balanced Edge)</option>
                        <option value="tiny">Tiny (~1.5M params - Ultra-lean)</option>
                        <option value="medium">Medium (~34.5M params - High Accuracy 86.2% Hmean)</option>
                    </select>
                </div>
                <button type="button" onclick="submitExtraction()">Extract Structured JSON</button>
            </form>

            <div id="output" class="result-box" style="display:none;">
                <h3 style="color: #38bdf8;">Extraction Results:</h3>
                <pre id="jsonOutput"></pre>
            </div>
        </div>

        <script>
            async function submitExtraction() {
                const form = document.getElementById("uploadForm");
                const formData = new FormData(form);
                const outDiv = document.getElementById("output");
                const jsonPre = document.getElementById("jsonOutput");
                
                jsonPre.textContent = "Processing through 4 stages on CPU...";
                outDiv.style.display = "block";

                try {
                    const resp = await fetch("/extract", { method: "POST", body: formData });
                    const data = await resp.json();
                    jsonPre.textContent = JSON.stringify(data, null, 2);
                } catch (err) {
                    jsonPre.textContent = "Error: " + err;
                }
            }
        </script>
    </body>
    </html>
    """
