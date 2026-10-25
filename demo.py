"""
Indra-Advanced: End-to-End Demonstration Script.

Showcases the 4-stage pipeline:
  Stage 1: SLANet-1M / SLANeXt (Structure HTML & Cell BBoxes)
  Stage 2: PP-OCRv4 (Text Detection & Recognition)
  Stage 3: Serialization Engine (Preserves Header: Value Relationships)
  Stage 4: GLiNER Zero-Shot Extraction (Typed Schema JSON)
"""

import sys
import json
from pathlib import Path
from PIL import Image

# Ensure UTF-8 stdout for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from indra_advanced.config import (
    PipelineConfig,
    OCRModelTier,
    GLiNERModelVariant,
    SerializationMode,
)
from indra_advanced.pipeline import IndraAdvancedPipeline
from indra_advanced.schemas.domain_schemas import INVOICE_SCHEMA
from indra_advanced.utils.table_renderer import SyntheticTableGenerator


def run_demo():
    print("=" * 80)
    print("  INDRA-ADVANCED: HIGH-THROUGHPUT DOCUMENT & TABLE INTELLIGENCE")
    print("=" * 80)
    print("Architecture Footprint: ~70M-110M Parameters (Full CPU / Edge Runnable)")
    print("  • Stage 1: SLANet-1M (~9.2M params) -> Predicts HTML table skeleton & cell boxes")
    print("  • Stage 2: PP-OCRv4 (~7.7M params small) -> DBNet text detection + CRNN recognition")
    print("  • Stage 3: Custom Serializer (0 params) -> Preserves 'Header: Value' associative pairs")
    print("  • Stage 4: GLiNER (60M-74M params) -> Zero-shot schema extraction into typed JSON")
    print("=" * 80)

    # 1. Generate realistic test document
    print("\n[Step 1] Synthesizing high-res test invoice document...")
    img_array, expected_md = SyntheticTableGenerator.create_invoice_table_image()
    demo_img_path = ROOT_DIR / "demo_invoice.png"
    Image.fromarray(img_array).save(demo_img_path)
    print(f"  ✓ Created synthetic invoice image at: {demo_img_path}")

    # 2. Configure pipeline
    config = PipelineConfig()
    config.stage2.tier = OCRModelTier.SMALL
    config.stage4.model_variant = GLiNERModelVariant.GLINER_2_5_SMALL
    config.stage3.primary_mode = SerializationMode.HEADER_VALUE

    print(f"\n[Step 2] Initializing Indra-Advanced Pipeline...")
    print(f"  Total footprint: {config.total_estimated_parameters}")
    pipeline = IndraAdvancedPipeline(config)

    # 3. Execute Pipeline
    print("\n[Step 3] Running document extraction through all 4 stages...")
    result = pipeline.process(demo_img_path, schema=INVOICE_SCHEMA)

    # 4. Display Performance & Latencies
    print("\n" + "=" * 50)
    print("  PERFORMANCE BREAKDOWN (CPU INFERENCE)")
    print("=" * 50)
    for stage, ms in result.timing_breakdown_ms.items():
        print(f"  {stage.replace('_', ' ').title():<32}: {ms:>8.2f} ms")
    print("-" * 50)
    print(f"  {'Total Elapsed Latency':<32}: {result.total_elapsed_ms:>8.2f} ms")
    print("=" * 50)

    # 5. Display Stage Outputs
    print("\n[Stage 1 & 2 Output: Populated HTML Skeleton]")
    print(result.populated_html[:300] + ("..." if len(result.populated_html) > 300 else ""))

    print("\n[Stage 3 Output: Serialized Structured Text (Header: Value Bridge)]")
    print("-" * 60)
    print(result.serialized_text)
    print("-" * 60)

    print("\n[Stage 3 Output: Clean Markdown Table]")
    print(result.markdown_table)

    print("\n[Stage 4 Output: Schema-Defined Extracted JSON]")
    print(json.dumps(result.structured_json, indent=2))

    # 6. Save outputs
    output_dir = ROOT_DIR / "demo_output"
    output_dir.mkdir(exist_ok=True)
    json_path = output_dir / "invoice_extracted.json"
    result.save_json(json_path)
    print(f"\n✓ Saved structured output JSON to: {json_path}")
    print("\nDemo complete! All 4 stages executed successfully.")


if __name__ == "__main__":
    run_demo()
