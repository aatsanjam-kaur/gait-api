"""
Configuration for Knee Osteoarthritis (KOA) X-Ray Inference API
================================================================
Houses all constants, thresholds, paths, class mappings, image constraints,
and environment variable configuration for the X-Ray module.
"""

import os
from typing import Dict, List, Optional

# ==============================================================================
# BASE PATHS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "best_finetuned.keras")

# Model path resolved from environment variable or default local path
MODEL_PATH: str = os.environ.get("XRAY_MODEL_PATH", DEFAULT_MODEL_PATH)

# Optional API key for service protection. Default is None (no auth required).
# Never hardcoded, never raises on import if unset.
API_KEY: Optional[str] = os.environ.get("XRAY_API_KEY", "").strip() or None

# Server networking
HOST: str = os.environ.get("XRAY_API_HOST", "0.0.0.0")
PORT: int = int(os.environ.get("XRAY_API_PORT", "8002"))

# ==============================================================================
# MODEL ARCHITECTURE & TENSOR SPECIFICATIONS
# ==============================================================================
# EfficientNetB0 input dimensions (width, height)
TARGET_IMAGE_SIZE = (224, 224)
TARGET_INPUT_SHAPE = [None, 224, 224, 3]

# 5 Kellgren-Lawrence (KL) grades from MedicalExpert-II dataset
NUM_CLASSES: int = 5
CLASS_LABELS: List[str] = ["KL0", "KL1", "KL2", "KL3", "KL4"]
CLASS_DESCRIPTIONS: Dict[int, str] = {
    0: "Normal",
    1: "Doubtful",
    2: "Mild",
    3: "Moderate",
    4: "Severe",
}

# ==============================================================================
# CLINICAL THRESHOLDS & RISK TIERS
# ==============================================================================
# Binary Osteoarthritis (OA) decision boundary:
# Definite OA corresponds to KL >= 2 (sum of softmax probabilities for KL2, KL3, KL4).
# NOTE: Tune this threshold on the VALIDATION set only, never the test set.
OA_THRESHOLD: float = 0.50

# Risk tier categorization based on oa_probability:
#   - High risk: oa_probability >= 0.65
#   - Monitor:   0.35 <= oa_probability < 0.65
#   - Low risk:  oa_probability < 0.35
RISK_THRESHOLD_HIGH: float = 0.65
RISK_THRESHOLD_LOW: float = 0.35

# Classification outcome strings
PREDICTION_DEFINITE_OA: str = "Definite OA (KL2-4)"
PREDICTION_NO_OR_DOUBTFUL: str = "No or doubtful OA (KL0-1)"
PREDICTION_BORDERLINE: str = "Borderline"

# Standard medical interpretation disclaimer
DEFAULT_INTERPRETATION: str = (
    "Automated KL-grade estimate from a knee X-ray. Screening aid, not a diagnosis."
)

# Prototype status flag
IS_PROTOTYPE: bool = True

# ==============================================================================
# IMAGE VALIDATION CONSTRAINTS
# ==============================================================================
# Maximum uploaded file size: 10 MB (10 * 1024 * 1024 bytes)
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024

# Minimum acceptable spatial dimensions
MIN_IMAGE_WIDTH: int = 64
MIN_IMAGE_HEIGHT: int = 64

# Supported image formats (validated via decoding, not MIME type headers)
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG"}

# Training data (MedicalExpert-II) consisted of horizontal knee-joint strips.
# Typical aspect ratio ranges between ~1.5 and 5.0 (commonly 2:1 to 4:1).
ASPECT_RATIO_MIN: float = 1.5
ASPECT_RATIO_MAX: float = 5.0

# Mean difference between color channels to detect non-grayscale photos
COLOR_DIFF_THRESHOLD: float = 5.0

# Confidence threshold for flagging grade uncertainty
LOW_CONFIDENCE_THRESHOLD: float = 0.50
