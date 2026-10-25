"""
Indra-Advanced: End-to-End Orchestrator Pipeline.
Coordinates:
  Stage 1: Table Structure Recognition (SLANet-1M / SLANeXt)
  Stage 2: Cell Content OCR (PP-OCRv4 DB + CRNN)
  Stage 3: Rule-Based Serialization (Header: Value and Spatial Bridge)
  Stage 4: Zero-Shot Schema Extraction (GLiNER Edge / Boundary / Bi-Encoder)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np

from indra_advanced.config import (
    PipelineConfig,
    SerializationMode,
)
from indra_advanced.stage1_structure.slanet_engine import SLANetEngine, StructureResult
from indra_advanced.stage2_ocr.ppocr_engine import PPOCREngine, OCRResult
from indra_advanced.stage2_ocr.cell_matcher import CellMatcher
from indra_advanced.stage3_serialization.serializer import TableSerializer
from indra_advanced.stage4_extraction.gliner_extractor import (
    GLiNERZeroShotExtractor,
    SchemaExtractionResult,
    ExtractedEntity,
)
from indra_advanced.schemas.base_schema import ExtractionSchema
from indra_advanced.utils.image_utils import load_image


@dataclass
class PipelineResult:
    """Complete results from the 4-stage pipeline execution."""
    populated_html: str
    markdown_table: str
    serialized_text: str
    cell_bboxes: List[List[float]] = field(default_factory=list)
    cell_records: List[Dict[str, Any]] = field(default_factory=list)
    ocr_result: Optional[OCRResult] = None
    structured_json: Dict[str, Any] = field(default_factory=dict)
    records: List[Dict[str, Any]] = field(default_factory=list)
    raw_entities: List[ExtractedEntity] = field(default_factory=list)
    timing_breakdown_ms: Dict[str, float] = field(default_factory=dict)
    total_elapsed_ms: float = 0.0
    parameter_footprint: str = ""

    def to_dataframe(self) -> pd.DataFrame:
        """Converts extracted table records or HTML into a Pandas DataFrame."""
        if self.records:
            return pd.DataFrame(self.records)

        # Fallback: parse populated HTML into DataFrame
        try:
            dfs = pd.read_html(self.populated_html)
            if dfs:
                return dfs[0]
        except Exception:
            pass

        return pd.DataFrame()

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to clean JSON-serializable dictionary."""
        return {
            "parameter_footprint": self.parameter_footprint,
            "timing_breakdown_ms": self.timing_breakdown_ms,
            "total_elapsed_ms": round(self.total_elapsed_ms, 2),
            "structured_json": self.structured_json,
            "table_records": self.records,
            "markdown_table": self.markdown_table,
            "serialized_text": self.serialized_text,
            "num_cells_detected": len(self.cell_bboxes),
            "populated_html": self.populated_html,
        }

    def save_json(self, filepath: Union[str, Path]):
        """Save structured output to JSON file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class IndraAdvancedPipeline:
    """
    Indra-Advanced: High-Throughput CPU/Edge Document & Table Intelligence Pipeline.
    Combines SLANet-1M, PP-OCRv4, Custom Serialization, and GLiNER into ~70M-110M parameter footprint.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # Initialize stages
        self.stage1_structure = SLANetEngine(self.config.stage1)
        self.stage2_ocr = PPOCREngine(self.config.stage2)
        self.stage3_serializer = TableSerializer(self.config.stage3)
        self.stage4_extractor = GLiNERZeroShotExtractor(self.config.stage4)

    def process(
        self,
        image_input: Union[str, Path, bytes, np.ndarray],
        schema: Optional[Union[ExtractionSchema, List[str], Dict[str, Any]]] = None,
        serialization_mode: Optional[SerializationMode] = None,
    ) -> PipelineResult:
        """
        Execute full 4-stage document extraction pipeline.
        
        Args:
            image_input: Document image (file path, PIL Image, bytes, or numpy array).
            schema: Optional schema for Stage 4 zero-shot extraction.
            serialization_mode: Optional Stage 3 serialization mode override.
            
        Returns:
            PipelineResult containing populated HTML, serialized text, and structured JSON.
        """
        t_start_total = time.perf_counter()
        timing: Dict[str, float] = {}

        # Load image
        img = load_image(image_input)

        # -------------------------------------------------------------
        # Stage 2 (Pre-OCR): Detect and recognize text across document
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        ocr_res = self.stage2_ocr.extract(img)
        timing["stage2_ocr_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # -------------------------------------------------------------
        # Stage 1: Table Structure Recognition (SLANet / SLANeXt)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        rapid_ocr_format = ocr_res.to_rapidtable_format() if ocr_res.boxes else None
        struct_res = self.stage1_structure.predict(img, ocr_results=rapid_ocr_format)
        timing["stage1_structure_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # -------------------------------------------------------------
        # Stage 2 (Cell Alignment): Map OCR text to SLANet cell coordinates
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        if ocr_res.boxes and struct_res.cell_bboxes:
            populated_html, cell_records = CellMatcher.match_and_populate_html(
                pred_html=struct_res.pred_html,
                cell_bboxes=struct_res.cell_bboxes,
                ocr_boxes=ocr_res.boxes,
                ocr_texts=ocr_res.texts,
                ocr_scores=ocr_res.scores,
            )
        else:
            populated_html = struct_res.pred_html
            cell_records = []
        timing["stage2_cell_alignment_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # -------------------------------------------------------------
        # Stage 3: Custom Serialization Logic (Critical Bridge)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        mode = serialization_mode or self.config.stage3.primary_mode
        serialized_text = self.stage3_serializer.serialize(
            html_str=populated_html,
            cell_bboxes=struct_res.cell_bboxes,
            mode=mode,
        )
        # Also generate markdown table representation for inspection
        markdown_table = self.stage3_serializer.serialize(
            html_str=populated_html,
            cell_bboxes=struct_res.cell_bboxes,
            mode=SerializationMode.MARKDOWN_TABLE,
        )
        timing["stage3_serialization_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # -------------------------------------------------------------
        # Stage 4: Zero-Shot Schema Extraction (GLiNER)
        # -------------------------------------------------------------
        structured_json: Dict[str, Any] = {}
        records: List[Dict[str, Any]] = []
        raw_entities: List[ExtractedEntity] = []

        if schema is not None:
            t0 = time.perf_counter()
            extraction_res = self.stage4_extractor.extract(
                serialized_text=serialized_text,
                schema=schema,
            )
            structured_json = extraction_res.structured_json
            records = extraction_res.records
            raw_entities = extraction_res.raw_entities
            timing["stage4_extraction_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        total_elapsed = (time.perf_counter() - t_start_total) * 1000

        return PipelineResult(
            populated_html=populated_html,
            markdown_table=markdown_table,
            serialized_text=serialized_text,
            cell_bboxes=struct_res.cell_bboxes,
            cell_records=cell_records,
            ocr_result=ocr_res,
            structured_json=structured_json,
            records=records,
            raw_entities=raw_entities,
            timing_breakdown_ms=timing,
            total_elapsed_ms=round(total_elapsed, 2),
            parameter_footprint=self.config.total_estimated_parameters,
        )
