"""
Base Schema System for Indra-Advanced Zero-Shot Extraction.
Supports Pydantic models, JSON Schema, and plain label dictionaries.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class ExtractionSchema(BaseModel):
    """
    Defines a schema specification for Stage 4 GLiNER extraction.
    Holds entity labels, descriptions, and optional hierarchical nesting rules.
    """
    name: str = "generic_extraction"
    description: str = "Generic schema for structured document entity extraction"
    entity_labels: List[str] = Field(default_factory=list)
    label_descriptions: Dict[str, str] = Field(default_factory=dict)
    table_column_mappings: Dict[str, str] = Field(default_factory=dict)
    expected_output_type: str = "dict"  # 'dict' or 'list_of_records'

    @classmethod
    def from_labels(cls, name: str, labels: List[str], descriptions: Optional[Dict[str, str]] = None) -> "ExtractionSchema":
        """Quickly build an extraction schema from a list of zero-shot entity names."""
        return cls(
            name=name,
            entity_labels=labels,
            label_descriptions=descriptions or {},
        )

    @classmethod
    def from_pydantic_model(cls, model_class: type[BaseModel]) -> "ExtractionSchema":
        """Generate an ExtractionSchema automatically from a Pydantic model's fields."""
        fields = model_class.model_fields
        labels = list(fields.keys())
        descriptions = {}
        for fname, finfo in fields.items():
            if finfo.description:
                descriptions[fname] = finfo.description
            else:
                descriptions[fname] = fname.replace("_", " ").title()

        return cls(
            name=model_class.__name__,
            description=model_class.__doc__ or f"Extraction schema for {model_class.__name__}",
            entity_labels=labels,
            label_descriptions=descriptions,
        )

    def to_gliner_labels(self) -> List[str]:
        """Convert schema entities into natural language query labels for GLiNER zero-shot prompting."""
        gliner_labels = []
        for label in self.entity_labels:
            # If description provided, format as human-readable label
            clean_label = label.replace("_", " ").strip()
            gliner_labels.append(clean_label)
        return gliner_labels
