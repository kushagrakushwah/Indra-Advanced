"""
Configuration for Indra-Advanced 4-Stage Pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union, Dict, Any


class StructureModelVariant(str, Enum):
    """Stage 1: SLANet / SLANeXt table structure models."""
    SLANET_1M = "slanet_1m"                # ~9.2M params; standard lightweight table structure model
    SLANET_PLUS = "slanet_plus"            # ~9.2M params; improved wireless table handling
    SLANEXT_WIRED = "slanext_wired"        # Specialized for bordered / wired tables
    SLANEXT_WIRELESS = "slanext_wireless"  # Specialized for unbordered / wireless tables


class OCRModelTier(str, Enum):
    """Stage 2: PP-OCRv4 (DB + CRNN) Model Tiers."""
    TINY = "tiny"      # ~1.5M params; ultra-lightweight edge deployment
    SMALL = "small"    # ~7.7M params; default balanced edge tier (rec ~5.2M, det ~2.5M)
    MEDIUM = "medium"  # ~34.5M params; highest accuracy (86.2% Det Hmean, 83.2% Rec Hmean)


class SerializationMode(str, Enum):
    """Stage 3: Serialization modes converting HTML table to linear text."""
    HEADER_VALUE = "header_value"                  # "Header: Value" row-wise key-value linearization
    MARKDOWN_TABLE = "markdown_table"              # Clean GitHub-flavored markdown table
    HIERARCHICAL = "hierarchical"                  # Preserves multi-level headers, colspans, and rowspans
    COORDINATE_ANNOTATED = "coordinate_annotated"  # Enriches text with normalized bounding coordinates [x1, y1, x2, y2]
    ROW_JSON = "row_json"                          # Structured JSON lines per row


class GLiNERModelVariant(str, Enum):
    """Stage 4: GLiNER zero-shot schema extraction models."""
    GLINER_2_5_SMALL = "gliner_2_5_small"    # 74M params, ~296 MB on disk; boundary prediction (no span limit)
    GLINER_BI_EDGE = "gliner_bi_edge"        # 60M params (ettin-32m + MiniLM-L6); fastest bi-encoder for edge
    GLINER_BI_SMALL = "gliner_bi_small"      # 108M params (ettin-68m + MiniLM-L12); higher zero-shot accuracy


@dataclass
class Stage1StructureConfig:
    """Configuration for Stage 1: Table Structure Recognition (SLANet/SLANeXt)."""
    model_variant: StructureModelVariant = StructureModelVariant.SLANET_1M
    device: str = "cpu"
    num_threads: int = 4
    input_size: int = 488
    max_length: int = 500
    model_path: Optional[str] = None


@dataclass
class Stage2OCRConfig:
    """Configuration for Stage 2: Content OCR (PP-OCRv4 DB + CRNN)."""
    tier: OCRModelTier = OCRModelTier.SMALL
    device: str = "cpu"
    num_threads: int = 4
    det_db_thresh: float = 0.3
    det_db_box_thresh: float = 0.5
    det_db_unclip_ratio: float = 1.6
    rec_score_thresh: float = 0.5
    lang: str = "en"
    use_angle_cls: bool = False


@dataclass
class Stage3SerializationConfig:
    """Configuration for Stage 3: Custom Serialization Logic."""
    primary_mode: SerializationMode = SerializationMode.HEADER_VALUE
    include_coordinates: bool = False
    preserve_empty_cells: bool = False
    row_delimiter: str = "\n"
    cell_delimiter: str = " | "
    max_cell_length: int = 500


@dataclass
class Stage4ExtractionConfig:
    """Configuration for Stage 4: Zero-Shot Schema Extraction (GLiNER)."""
    model_variant: GLiNERModelVariant = GLiNERModelVariant.GLINER_2_5_SMALL
    model_name_or_path: Optional[str] = None
    device: str = "cpu"
    threshold: float = 0.4
    flat_ner: bool = False
    multi_label: bool = False
    batch_size: int = 8
    max_length: int = 512


@dataclass
class PipelineConfig:
    """Combined configuration for the full Indra-Advanced 4-Stage Pipeline."""
    stage1: Stage1StructureConfig = field(default_factory=Stage1StructureConfig)
    stage2: Stage2OCRConfig = field(default_factory=Stage2OCRConfig)
    stage3: Stage3SerializationConfig = field(default_factory=Stage3SerializationConfig)
    stage4: Stage4ExtractionConfig = field(default_factory=Stage4ExtractionConfig)

    # Global runtime settings
    device: str = "cpu"
    cpu_threads: int = 4
    enable_cache: bool = True
    verbose: bool = False

    def __post_init__(self):
        # Sync device and threads across stages
        self.stage1.device = self.device
        self.stage1.num_threads = self.cpu_threads
        self.stage2.device = self.device
        self.stage2.num_threads = self.cpu_threads
        self.stage4.device = self.device

    @property
    def total_estimated_parameters(self) -> str:
        """Calculate and report total parameter footprint of selected configuration."""
        stage1_params = 9.2  # SLANet ~9.2M
        stage2_params = {
            OCRModelTier.TINY: 1.5,
            OCRModelTier.SMALL: 7.7,
            OCRModelTier.MEDIUM: 34.5,
        }.get(self.stage2.tier, 7.7)
        stage3_params = 0.0  # Rule-based custom logic
        stage4_params = {
            GLiNERModelVariant.GLINER_BI_EDGE: 60.0,
            GLiNERModelVariant.GLINER_2_5_SMALL: 74.0,
            GLiNERModelVariant.GLINER_BI_SMALL: 108.0,
        }.get(self.stage4.model_variant, 74.0)

        total = stage1_params + stage2_params + stage3_params + stage4_params
        return f"~{total:.1f}M parameters (Stage 1: {stage1_params}M + Stage 2: {stage2_params}M + Stage 3: 0M + Stage 4: {stage4_params}M)"
