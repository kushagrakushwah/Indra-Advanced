"""
End-to-End integration tests for Indra-Advanced 4-Stage Pipeline.
"""

from indra_advanced.pipeline import IndraAdvancedPipeline
from indra_advanced.config import PipelineConfig, OCRModelTier, SerializationMode
from indra_advanced.schemas.domain_schemas import INVOICE_SCHEMA
from indra_advanced.utils.table_renderer import SyntheticTableGenerator


def test_pipeline_synthetic_invoice_end_to_end():
    # 1. Generate realistic synthetic document table
    img_array, expected_md = SyntheticTableGenerator.create_invoice_table_image()

    # 2. Configure pipeline with small OCR tier
    config = PipelineConfig()
    config.stage2.tier = OCRModelTier.SMALL
    config.stage3.primary_mode = SerializationMode.HEADER_VALUE

    pipeline = IndraAdvancedPipeline(config)

    # 3. Process image
    result = pipeline.process(img_array, schema=INVOICE_SCHEMA)

    # 4. Assert all pipeline stages succeeded
    assert result.populated_html is not None
    assert result.serialized_text is not None
    assert result.total_elapsed_ms > 0
    assert "stage1_structure_ms" in result.timing_breakdown_ms
    assert "stage2_ocr_ms" in result.timing_breakdown_ms
    assert "stage3_serialization_ms" in result.timing_breakdown_ms

    # 5. Check DataFrame conversion
    df = result.to_dataframe()
    assert df is not None

    # 6. Check JSON output structure
    output_dict = result.to_dict()
    assert "parameter_footprint" in output_dict
    assert "timing_breakdown_ms" in output_dict
