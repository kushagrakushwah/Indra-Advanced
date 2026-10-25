---
title: Indra Advanced
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
python_version: '3.12'
app_file: app.py
pinned: false
license: apache-2.0
short_description: Document and table intelligence pipeline
---

# Indra-Advanced

A lightweight document intelligence pipeline that combines table structure recognition, OCR, associative serialization, and zero-shot schema extraction.

## Architecture Comparison

```mermaid
flowchart TD
    subgraph Indra_Original["Indra Original (~1M params)"]
        A1["Input Image"] --> B1["Line Detection + CTC OCR"]
        B1 --> C1["Raw Markdown / Text Dump"]
        C1 -.->|Requires external LLM| D1["Structured JSON"]
    end

    subgraph Indra_Advanced["Indra-Advanced (~77M-90M params)"]
        A2["Input Image"] --> B2["SLANet-1M Table Structure"]
        A2 --> C2["PP-OCRv4 Text Recognition"]
        B2 & C2 --> D2["Associative Serializer (Header: Value Bridge)"]
        D2 --> E2["GLiNER Zero-Shot Extraction"]
        E2 --> F2["Typed JSON Output"]
    end
```

## Feature Comparison

| Capability | Indra Original | Indra-Advanced |
| :--- | :--- | :--- |
| Primary Role | OCR and line detection | End-to-end schema extraction |
| Core Parameter Size | ~1M parameters | ~77M to 90M parameters |
| Output Format | Plain Markdown line dump | Typed JSON, table rows, and Markdown |
| Schema Extraction | No (requires external LLM) | Yes (direct zero-shot extraction) |
| Relational Serialization | No (line clustering) | Yes (Header-to-Value preservation) |
| Multi-Page Books | External loop required | Native multi-page navigation via PDFium |
| Fine-Print / Stamp OCR | Prone to character drops | High-precision DBNet + CRNN |

## Usage

```python
from indra_advanced import IndraAdvancedPipeline
from indra_advanced.schemas import INVOICE_SCHEMA

pipeline = IndraAdvancedPipeline()
result = pipeline.process("document.png", schema=INVOICE_SCHEMA)

print(result.structured_json)
```
