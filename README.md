# Indra-Advanced

A lightweight CPU document intelligence pipeline combining table structure recognition, OCR, relational serialization, and zero-shot schema extraction into typed JSON.

## Comparison: Indra Original vs Indra-Advanced

### Architecture Flow

```mermaid
flowchart TD
    subgraph Indra_Original["Indra Original (~1M params)"]
        A1["Input Image"] --> B1["Line Detection + CTC OCR"]
        B1 --> C1["Raw Markdown / Text Dump"]
        C1 -.->|Requires heavy external LLM| D1["Structured JSON"]
    end

    subgraph Indra_Advanced["Indra-Advanced (~77M-90M params)"]
        A2["Input Image"] --> B2["SLANet-1M Table Structure"]
        A2 --> C2["PP-OCRv4 Text Recognition"]
        B2 & C2 --> D2["Associative Serializer (Header: Value Bridge)"]
        D2 --> E2["GLiNER Zero-Shot Extraction"]
        E2 --> F2["Final Typed JSON Output"]
    end
```

### Feature Comparison

| Capability | Indra Original | Indra-Advanced |
| :--- | :--- | :--- |
| Primary Role | OCR and text line detection | End-to-end schema information extraction |
| Recognition Core Size | ~1 million parameters | ~77M to 90M parameters |
| Output Format | Plain Markdown and line boxes | Typed JSON, table rows, and Markdown |
| Schema Extraction | No (requires external LLM) | Yes (direct zero-shot extraction) |
| Relational Serialization | No (line clustering) | Yes (Header-to-Value preservation) |
| Complex Stamp / Fine-Print | Prone to character drops | High-precision DBNet + CRNN |
| Target Runtime | CPU | CPU |

## Quickstart

### Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

### Usage

```python
from indra_advanced import IndraAdvancedPipeline
from indra_advanced.schemas import INVOICE_SCHEMA

pipeline = IndraAdvancedPipeline()
result = pipeline.process("document.png", schema=INVOICE_SCHEMA)

print(result.structured_json)
```

### Web UI & API

```bash
uvicorn indra_advanced.server:app --host 0.0.0.0 --port 8000
```
