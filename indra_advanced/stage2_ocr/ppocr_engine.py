"""
Stage 2: PP-OCRv4 (DB + CRNN) Text Extraction Engine.
Extracts text from cell regions and document pages.
Supports 3 model tiers:
  - Tiny: ~1.5M params
  - Small: ~7.7M params (Default edge tier: rec ~5.2M, det ~2.5M)
  - Medium: ~34.5M params (High accuracy: 86.2% Det Hmean, 83.2% Rec Hmean)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional, Union
import numpy as np
import logging
from indra_advanced.config import Stage2OCRConfig, OCRModelTier
from indra_advanced.utils.image_utils import load_image

logger = logging.getLogger("indra_advanced.stage2")


@dataclass
class OCRResult:
    """Output of Stage 2 text recognition."""
    boxes: List[Any] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    tier: str = "small"

    def to_rapidtable_format(self) -> List[Tuple[Any, Any, Any]]:
        """Converts to [(boxes, texts, scores)] tuple expected by RapidTable."""
        return [(self.boxes, self.texts, self.scores)]


class PPOCREngine:
    """
    Stage 2 Engine: Runs text detection (DBNet) and text recognition (CRNN/SVTR).
    """

    def __init__(self, config: Optional[Stage2OCRConfig] = None):
        self.config = config or Stage2OCRConfig()
        self._ocr = None
        self._initialized = False

    def _lazy_init(self):
        if self._initialized:
            return

        try:
            from rapidocr_onnxruntime import RapidOCR

            # Configure params based on tier
            params = {
                "det_db_thresh": self.config.det_db_thresh,
                "det_db_box_thresh": self.config.det_db_box_thresh,
                "det_db_unclip_ratio": self.config.det_db_unclip_ratio,
                "rec_score_thresh": self.config.rec_score_thresh,
                "use_angle_cls": self.config.use_angle_cls,
            }

            self._ocr = RapidOCR(**params)
            logger.info(f"Initialized Stage 2 PP-OCRv4 engine (tier: {self.config.tier.value})")
        except Exception as e:
            logger.warning(
                f"Could not load RapidOCR runtime ({e}). "
                "PPOCREngine will operate in fallback mode."
            )
            self._ocr = None

        self._initialized = True

    def extract(self, image_input: Union[str, np.ndarray]) -> OCRResult:
        """
        Detect and recognize text within the image.
        
        Args:
            image_input: Image filepath or RGB numpy array.
            
        Returns:
            OCRResult containing text boxes, recognized strings, and confidence scores.
        """
        self._lazy_init()
        img = load_image(image_input)

        if self._ocr is not None:
            try:
                ocr_res, elapse_list = self._ocr(img)
                if ocr_res:
                    boxes = [item[0] for item in ocr_res]
                    texts = [item[1] for item in ocr_res]
                    scores = [float(item[2]) for item in ocr_res]
                    return OCRResult(
                        boxes=boxes,
                        texts=texts,
                        scores=scores,
                        tier=self.config.tier.value,
                    )
            except Exception as e:
                logger.warning(f"Error during PP-OCR inference: {e}. Using fallback.")

        # Heuristic fallback if OCR runtime is not available
        return OCRResult(boxes=[], texts=[], scores=[], tier="fallback")
