"""
Head-to-Head Benchmark: Indra-OCR (Vik Paruchuri) vs Indra-Advanced 4-Stage Pipeline.
"""

import sys
import time
import json
from pathlib import Path
from PIL import Image

# Ensure UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from indra_advanced import IndraAdvancedPipeline, PipelineConfig
from indra_advanced.schemas import INVOICE_SCHEMA
from indra_advanced.utils.table_renderer import SyntheticTableGenerator

import indra_ocr


def benchmark():
    img_path = ROOT_DIR / "demo_invoice.png"
    if not img_path.exists():
        img_arr, _ = SyntheticTableGenerator.create_invoice_table_image()
        Image.fromarray(img_arr).save(img_path)

    print("=" * 80)
    print("  HEAD-TO-HEAD BENCHMARK: INDRA-OCR vs INDRA-ADVANCED")
    print("=" * 80)

    # ---------------------------------------------------------
    # TEST 1: Indra-OCR (Vik Paruchuri / Datalab)
    # ---------------------------------------------------------
    print("\n[TEST 1] Running Indra-OCR (1M CTC Recognition Model)...")
    converter = indra_ocr.load()

    t0 = time.perf_counter()
    _, page = next(converter.convert(str(img_path)))
    indra_elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"  ✓ Indra-OCR Latency: {indra_elapsed_ms:.2f} ms")
    print(f"  ✓ Text Lines Detected: {len(page.lines)}")
    print(f"  ✓ Layout Blocks: {len(page.blocks)}")
    print(f"  ✓ Output Format: Markdown text & Bounding boxes")
    print("\n--- Indra-OCR Markdown Output ---")
    print(page.markdown)

    # ---------------------------------------------------------
    # TEST 2: Indra-Advanced (4-Stage Pipeline)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("[TEST 2] Running Indra-Advanced (~77M-90M Pipeline)...")
    config = PipelineConfig()
    pipeline = IndraAdvancedPipeline(config)

    t0 = time.perf_counter()
    adv_res = pipeline.process(str(img_path), schema=INVOICE_SCHEMA)
    adv_elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"  ✓ Indra-Advanced Latency: {adv_elapsed_ms:.2f} ms")
    print(f"    - Stage 1 (Structure): {adv_res.timing_breakdown_ms.get('stage1_structure_ms', 0):.2f} ms")
    print(f"    - Stage 2 (OCR): {adv_res.timing_breakdown_ms.get('stage2_ocr_ms', 0):.2f} ms")
    print(f"    - Stage 3 (Serialization): {adv_res.timing_breakdown_ms.get('stage3_serialization_ms', 0):.2f} ms")
    print(f"    - Stage 4 (GLiNER Extraction): {adv_res.timing_breakdown_ms.get('stage4_extraction_ms', 0):.2f} ms")
    print(f"  ✓ Output Format: Structured Typed JSON + Populated HTML + Markdown")
    print("\n--- Indra-Advanced Extracted JSON Snippet ---")
    print(json.dumps({k: v for k, v in list(adv_res.structured_json.items())[:6]}, indent=2))

    print("\n" + "=" * 80)
    print("  SUMMARY COMPARISON")
    print("=" * 80)
    print(f"  Indra-OCR:        ~1M params  | {indra_elapsed_ms:>8.2f} ms | Role: Raw OCR + Markdown layout")
    print(f"  Indra-Advanced:   ~90M params | {adv_elapsed_ms:>8.2f} ms | Role: End-to-End Schema Extraction to JSON")
    print("=" * 80)


if __name__ == "__main__":
    benchmark()
