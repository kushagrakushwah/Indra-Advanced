"""
Unit tests for Stage 4: Zero-Shot Schema Extraction.
"""

from indra_advanced.schemas.base_schema import ExtractionSchema
from indra_advanced.schemas.domain_schemas import INVOICE_SCHEMA, InvoiceDocument
from indra_advanced.stage4_extraction.gliner_extractor import GLiNERZeroShotExtractor


def test_schema_from_pydantic():
    schema = ExtractionSchema.from_pydantic_model(InvoiceDocument)
    assert "invoice_number" in schema.entity_labels
    assert "total_amount" in schema.entity_labels
    gliner_labels = schema.to_gliner_labels()
    assert "invoice number" in gliner_labels
    assert "total amount" in gliner_labels


def test_heuristic_pattern_extraction():
    extractor = GLiNERZeroShotExtractor()
    sample_text = (
        "Vendor Name: Acme Cloud Inc. | Invoice Number: INV-2024-9988\n"
        "Invoice Date: 2024-10-15 | Total Amount: $1,450.00\n"
        "Item: Cloud Compute | Quantity: 2 | Unit Price: $500.00 | Item Total: $1,000.00"
    )

    result = extractor.extract(sample_text, schema=INVOICE_SCHEMA)

    assert result.structured_json.get("invoice_number") == "INV-2024-9988"
    assert "Acme Cloud Inc." in result.structured_json.get("vendor_name", "")
    assert "$1,450.00" in result.structured_json.get("total_amount", "")
    assert len(result.records) >= 1
    assert any(r.get("item") == "Cloud Compute" for r in result.records)
