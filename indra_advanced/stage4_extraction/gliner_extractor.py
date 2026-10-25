"""
Stage 4: Zero-Shot Schema Extraction Engine.
Extracts schema-defined fields into structured, typed JSON using GLiNER.
Supports:
  1. GLiNER2.5 Small (boundary prediction, 74M params, ~296 MB on disk - handles multi-line cells)
  2. GLiNER Bi-Edge v2.0 (60M params, ettin-32m + MiniLM-L6 - ultra-lean for edge)
  3. GLiNER Bi-Small v2.0 (108M params, ettin-68m + MiniLM-L12 - high accuracy)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import re
from indra_advanced.config import Stage4ExtractionConfig, GLiNERModelVariant
from indra_advanced.schemas.base_schema import ExtractionSchema

logger = logging.getLogger("indra_advanced.stage4")


@dataclass
class ExtractedEntity:
    """Represents a single extracted entity with span and confidence."""
    text: str
    label: str
    score: float
    start: int = 0
    end: int = 0
    row_index: Optional[int] = None
    column_header: Optional[str] = None


@dataclass
class SchemaExtractionResult:
    """Final output of Stage 4 zero-shot schema extraction."""
    schema_name: str
    structured_json: Dict[str, Any]
    records: List[Dict[str, Any]] = field(default_factory=list)
    raw_entities: List[ExtractedEntity] = field(default_factory=list)
    model_name: str = "GLiNER"


class GLiNERZeroShotExtractor:
    """
    Stage 4 Engine: Extracts schema fields from serialized text without task-specific retraining.
    """

    MODEL_CHECKPOINTS = {
        GLiNERModelVariant.GLINER_2_5_SMALL: "urchade/gliner_small-v2.1",
        GLiNERModelVariant.GLINER_BI_EDGE: "knowledgator/gliner-bi-edge-v2.0",
        GLiNERModelVariant.GLINER_BI_SMALL: "knowledgator/gliner-bi-small-v1.0",
    }

    def __init__(self, config: Optional[Stage4ExtractionConfig] = None):
        self.config = config or Stage4ExtractionConfig()
        self._model = None
        self._initialized = False

    def _lazy_init(self):
        if self._initialized:
            return

        checkpoint = self.config.model_name_or_path or self.MODEL_CHECKPOINTS.get(
            self.config.model_variant, "urchade/gliner_small-v2.5"
        )

        try:
            from gliner import GLiNER
            logger.info(f"Loading Stage 4 GLiNER model: {checkpoint} on {self.config.device}...")
            self._model = GLiNER.from_pretrained(checkpoint)
            if self.config.device == "cuda":
                self._model.to("cuda")
            elif self.config.device == "cpu":
                self._model.to("cpu")
            self._model.eval()
            logger.info("Successfully loaded GLiNER model.")
        except Exception as e:
            logger.warning(
                f"Could not load GLiNER model from '{checkpoint}' ({e}). "
                "Extractor will use heuristic pattern matching for offline execution."
            )
            self._model = None

        self._initialized = True

    def extract(
        self,
        serialized_text: str,
        schema: Union[ExtractionSchema, List[str], Dict[str, Any]],
        threshold: Optional[float] = None,
    ) -> SchemaExtractionResult:
        """
        Extract fields from serialized structured text according to schema.
        
        Args:
            serialized_text: Linearized table/document text from Stage 3.
            schema: ExtractionSchema instance, list of labels, or field dict.
            threshold: Confidence cutoff threshold.
            
        Returns:
            SchemaExtractionResult containing validated JSON and raw entity spans.
        """
        self._lazy_init()
        thresh = threshold if threshold is not None else self.config.threshold

        # Normalize schema
        schema_obj: ExtractionSchema
        if isinstance(schema, ExtractionSchema):
            schema_obj = schema
        elif isinstance(schema, list):
            schema_obj = ExtractionSchema.from_labels("custom_schema", schema)
        elif isinstance(schema, dict):
            labels = schema.get("entity_labels", list(schema.keys()))
            schema_obj = ExtractionSchema(name=schema.get("name", "custom"), entity_labels=labels)
        else:
            schema_obj = ExtractionSchema(name="generic", entity_labels=["entity"])

        gliner_labels = schema_obj.to_gliner_labels()
        raw_entities: List[ExtractedEntity] = []

        if self._model is not None and len(serialized_text.strip()) > 0:
            try:
                # Process lines or chunks
                lines = serialized_text.splitlines()
                # Run prediction on full text or line-by-line for high resolution
                preds = self._model.predict_entities(
                    serialized_text,
                    gliner_labels,
                    threshold=thresh,
                    flat_ner=self.config.flat_ner,
                    multi_label=self.config.multi_label,
                )

                for p in preds:
                    raw_entities.append(
                        ExtractedEntity(
                            text=p["text"],
                            label=p["label"].replace(" ", "_"),
                            score=float(p.get("score", 1.0)),
                            start=p.get("start", 0),
                            end=p.get("end", 0),
                        )
                    )
            except Exception as e:
                logger.warning(f"Error during GLiNER prediction: {e}. Falling back to pattern extraction.")
                raw_entities = self._heuristic_extract(serialized_text, schema_obj)
        else:
            raw_entities = self._heuristic_extract(serialized_text, schema_obj)

        # Assemble extracted entities into typed JSON and table records
        structured_json, records = self._structure_entities(serialized_text, raw_entities, schema_obj)

        return SchemaExtractionResult(
            schema_name=schema_obj.name,
            structured_json=structured_json,
            records=records,
            raw_entities=raw_entities,
            model_name=self.config.model_variant.value,
        )

    def _structure_entities(
        self,
        serialized_text: str,
        entities: List[ExtractedEntity],
        schema: ExtractionSchema,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Structures extracted entity flat list into coherent document JSON
        and tabular row records.
        """
        output_dict: Dict[str, Any] = {}
        records: List[Dict[str, Any]] = []

        # Map entity labels to clean field names
        for ent in entities:
            clean_label = ent.label.lower()
            if clean_label not in output_dict:
                output_dict[clean_label] = ent.text
            elif isinstance(output_dict[clean_label], list):
                output_dict[clean_label].append(ent.text)
            else:
                # Convert to list if multiple values observed
                output_dict[clean_label] = [output_dict[clean_label], ent.text]

        # Extract table rows if text contains serialized line format (e.g. Header: Value | ...)
        lines = [line.strip() for line in serialized_text.splitlines() if line.strip()]
        for line in lines:
            if ":" in line and ("|" in line or len(line.split(":")) > 2):
                row_record = {}
                parts = line.split(" | ") if " | " in line else [line]
                for p in parts:
                    if ":" in p:
                        k, v = p.split(":", 1)
                        norm_k = k.strip().lower().replace(" ", "_")
                        row_record[norm_k] = v.strip()
                        if norm_k not in output_dict:
                            output_dict[norm_k] = v.strip()
                if row_record:
                    records.append(row_record)

        if records:
            output_dict["line_items"] = records

        return output_dict, records

    def _heuristic_extract(
        self,
        serialized_text: str,
        schema: ExtractionSchema,
    ) -> List[ExtractedEntity]:
        """
        Deterministic extractor when models are offline or for unit testing:
        Extracts key-value patterns and schema matches directly from serialized text.
        """
        entities: List[ExtractedEntity] = []

        for label in schema.entity_labels:
            clean_l = label.replace("_", " ")
            # Check pattern: "Label: Value" or "Label Value"
            pattern = rf"(?i)\b{re.escape(clean_l)}\s*[:\-]\s*([^|\n;]+)"
            matches = re.finditer(pattern, serialized_text)
            for m in matches:
                val = m.group(1).strip()
                if val:
                    entities.append(
                        ExtractedEntity(
                            text=val,
                            label=label,
                            score=0.95,
                            start=m.start(1),
                            end=m.end(1),
                        )
                    )

        # Common document regex patterns
        # Total / Amount
        total_match = re.search(r"(?i)\b(?:total\s*(?:due|amount)?|balance)\s*[:\$]\s*([\$€£]?\s*[\d,]+\.\d{2})", serialized_text)
        if total_match and "total_amount" in schema.entity_labels:
            if not any(e.label == "total_amount" for e in entities):
                entities.append(
                    ExtractedEntity(
                        text=total_match.group(1).strip(),
                        label="total_amount",
                        score=0.98,
                        start=total_match.start(1),
                        end=total_match.end(1),
                    )
                )

        # Dates
        date_match = re.search(r"\b(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4})\b", serialized_text)
        if date_match and "invoice_date" in schema.entity_labels:
            if not any(e.label == "invoice_date" for e in entities):
                entities.append(
                    ExtractedEntity(
                        text=date_match.group(1).strip(),
                        label="invoice_date",
                        score=0.92,
                        start=date_match.start(1),
                        end=date_match.end(1),
                    )
                )

        return entities
