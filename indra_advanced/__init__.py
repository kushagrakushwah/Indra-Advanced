"""
Indra-Advanced: Lightweight High-Throughput Document Structure & Zero-Shot Schema Extraction Pipeline.

4-Stage Pipeline:
  Stage 1: Structure Recognition (SLANet / SLANeXt / SLANet-1M)
  Stage 2: Content Extraction (PP-OCRv4 DBNet + CRNN)
  Stage 3: Serialization (Rule-Based Semantic & Spatial Bridge)
  Stage 4: Schema Extraction (GLiNER Zero-Shot: Bi-Edge / Bi-Small / GLiNER2.5 Small)
"""

__version__ = "1.0.0"

from indra_advanced.config import (
    PipelineConfig,
    Stage1StructureConfig,
    Stage2OCRConfig,
    Stage3SerializationConfig,
    Stage4ExtractionConfig,
    OCRModelTier,
    GLiNERModelVariant,
    SerializationMode,
)
from indra_advanced.pipeline import IndraAdvancedPipeline

__all__ = [
    "IndraAdvancedPipeline",
    "PipelineConfig",
    "Stage1StructureConfig",
    "Stage2OCRConfig",
    "Stage3SerializationConfig",
    "Stage4ExtractionConfig",
    "OCRModelTier",
    "GLiNERModelVariant",
    "SerializationMode",
]
