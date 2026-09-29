"""
FastAPI Inference Service for KOA X-Ray Analysis
================================================
Provides real-time Kellgren-Lawrence (KL) grading and Osteoarthritis (OA)
screening for knee-joint X-ray radiographs based on EfficientNetB0 transfer learning.

Endpoints:
  - GET  /health         : Service health, model status, and authentication requirements
  - POST /analyze-xray   : Upload knee X-ray image (multipart/form-data), outputs KL grades and OA risk
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any

import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import config
from preprocess import ImageValidationError, preprocess_image, validate_and_inspect_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("xray_api")

# Lazy/conditional import of TensorFlow
try:
    import tensorflow as tf
except ImportError:
    tf = None


# ==============================================================================
# PYDANTIC V2 RESPONSE MODELS
# ==============================================================================

class HealthResponse(BaseModel):
    """Health status and capability report of the inference service."""
    status: str = Field(default="ok", description="Service status indicator")
    model_loaded: bool = Field(..., description="Whether the TensorFlow model is actively loaded")
    model_input_shape: List[Optional[int]] = Field(
        default=config.TARGET_INPUT_SHAPE,
        description="Expected tensor input shape [batch, height, width, channels]"
    )
    num_classes: int = Field(default=config.NUM_CLASSES, description="Total number of output classes")
    api_key_required: bool = Field(..., description="Whether x-api-key authentication header is enforced")


class AnalyzeXRayResponse(BaseModel):
    """Detailed clinical and probabilistic evaluation from knee X-ray analysis."""
    status: str = Field(default="complete", description="Processing status")
    kl_grade: str = Field(..., description="Predicted Kellgren-Lawrence grade (KL0 - KL4)")
    kl_grade_index: int = Field(..., description="Integer index (0-4) of predicted KL grade")
    kl_description: str = Field(..., description="Clinical description of predicted grade")
    kl_grade_confidence: float = Field(..., description="Highest softmax class probability [0.0 - 1.0]")
    kl_probabilities: Dict[str, float] = Field(
        ...,
        description="Full probability distribution over KL grades (KL0 to KL4)"
    )
    oa_present: bool = Field(
        ...,
        description="True if cumulative probability P(KL2..KL4) exceeds OA_THRESHOLD (0.50)"
    )
    oa_probability: float = Field(
        ...,
        description="Cumulative probability of definite osteoarthritis P(KL2)+P(KL3)+P(KL4)"
    )
    risk_score: float = Field(
        ...,
        description="Scaled risk score [0.0 - 100.0] equivalent to oa_probability * 100"
    )
    risk_tier: str = Field(
        ...,
        description="Categorized risk tier: 'high' (>=0.65), 'monitor' (0.35-0.65), or 'low' (<0.35)"
    )
    prediction: str = Field(
        ...,
        description="Clinical classification summary: 'Definite OA (KL2-4)', 'No or doubtful OA (KL0-1)', or 'Borderline'"
    )
    interpretation: str = Field(
        default=config.DEFAULT_INTERPRETATION,
        description="Standard medical interpretation and screening disclaimer"
    )
    # Frontend compatibility fields (for direct frontend integration)
    klGrade: Optional[int] = Field(
        default=None,
        description="Integer KL grade index (0-4) for frontend compatibility"
    )
    findings: Optional[str] = Field(
        default=None,
        description="Clinical summary findings for frontend display"
    )
    riskContribution: Optional[str] = Field(
        default=None,
        description="Risk contribution tier ('low', 'mid', 'high') for frontend display"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal warnings regarding aspect ratio, color artifacts, confidence, or class limits"
    )
    prototype: bool = Field(default=config.IS_PROTOTYPE, description="Prototype indicator flag")


# ==============================================================================
# LIFESPAN & APPLICATION STATE MANAGEMENT
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages startup and shutdown lifecycle:
      1. Loads the fine-tuned Keras model once at startup with compile=False.
      2. Runs a single warm-up prediction on a blank image so the first real request is fast.
      3. Safely records failure if model cannot be loaded, enabling 503 rejection.
    """
    logger.info("Initializing X-RAY API: loading model artifact...")
    app.state.model = None
    app.state.model_loaded = False

    model_path = config.MODEL_PATH
    if os.path.exists(model_path):
        try:
            if tf is not None:
                # Load trained EfficientNetB0 without compilation to optimize load time
                app.state.model = tf.keras.models.load_model(model_path, compile=False)
                app.state.model_loaded = True
                logger.info(f"Model successfully loaded from: {model_path}")

                # Warm-up inference on a blank dummy tensor
                blank_input = tf.zeros((1, config.TARGET_IMAGE_SIZE[0], config.TARGET_IMAGE_SIZE[1], 3), dtype=tf.float32)
                _ = app.state.model.predict(blank_input, verbose=0)
                logger.info("Warm-up prediction completed successfully.")
            else:
                logger.warning("TensorFlow is not installed. Model cannot be executed in this environment.")
        except Exception as err:
            logger.error(f"Failed to load model from {model_path}: {err}", exc_info=True)
            app.state.model = None
            app.state.model_loaded = False
    else:
        logger.warning(
            f"Model file not found at '{model_path}'. "
            "Inference requests will return HTTP 503 until best_finetuned.keras is placed in model/."
        )

    yield

    logger.info("Shutting down X-RAY API service...")


# ==============================================================================
# FASTAPI APP INITIALIZATION
# ==============================================================================

app = FastAPI(
    title="Knee Osteoarthritis (KOA) X-Ray Inference API",
    description="Inference microservice for Kellgren-Lawrence grading from knee radiographs.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware: allow all origins with credentials disabled
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# AUTHENTICATION DEPENDENCY
# ==============================================================================

def verify_optional_api_key(x_api_key: Optional[str] = Header(None, alias="x-api-key")) -> None:
    """
    Enforces API key verification only if the XRAY_API_KEY environment variable is configured.
    If XRAY_API_KEY is not set or empty, open access is allowed.
    Never hardcoded; never raises at import time.
    """
    configured_key = config.API_KEY
    if configured_key:
        if not x_api_key or x_api_key != configured_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key in 'x-api-key' header."
            )


# ==============================================================================
# EXCEPTION HANDLERS (CLEAN JSON, NO LEAKED STACK TRACES)
# ==============================================================================

@app.exception_handler(ImageValidationError)
async def image_validation_error_handler(request: Request, exc: ImageValidationError):
    """Returns clean 400 Bad Request JSON errors for invalid image payloads."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error_type": "ImageValidationError",
            "message": exc.message,
            "detail": exc.message,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Formats HTTP exceptions into standard error payloads."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error_type": "HTTPException",
            "message": exc.detail,
            "detail": exc.detail,
        },
    )


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    """Formats Pydantic request parsing errors into 422 JSON response."""
    error_details = []
    for err in exc.errors():
        loc = " -> ".join(str(l) for l in err.get("loc", []))
        error_details.append(f"{loc}: {err.get('msg', 'validation error')}")
    clean_msg = "; ".join(error_details)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "error_type": "RequestValidationError",
            "message": clean_msg,
            "detail": clean_msg,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Prevents stack traces or internal implementation details from being exposed to clients."""
    logger.error(f"Unexpected internal server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error_type": "InternalServerError",
            "message": "An internal server error occurred while processing the X-ray image.",
            "detail": "Processing failed due to an unexpected error.",
        },
    )


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/health", response_model=HealthResponse)
def health():
    """
    Health check endpoint returning model load state and runtime configuration.
    """
    is_loaded = bool(getattr(app.state, "model", None) is not None and app.state.model_loaded)
    api_key_required = bool(config.API_KEY is not None and len(config.API_KEY) > 0)

    return HealthResponse(
        status="ok",
        model_loaded=is_loaded,
        model_input_shape=config.TARGET_INPUT_SHAPE,
        num_classes=config.NUM_CLASSES,
        api_key_required=api_key_required,
    )


@app.post("/analyze-xray", response_model=AnalyzeXRayResponse)
def analyze_xray(
    file: Optional[UploadFile] = File(None, description="Knee radiograph image file (multipart field 'file')"),
    image: Optional[UploadFile] = File(None, description="Alternative multipart field name ('image') for frontend compatibility"),
    request: Request = None,
):
    """
    Analyzes an uploaded knee X-ray radiograph image:
      1. Validates authentication if XRAY_API_KEY is configured.
      2. Validates image payload by decoding (max 10 MB, min 64x64, JPEG/PNG).
      3. Preprocesses input using the exact notebook pipeline (resize_with_pad to 224x224x3).
      4. Executes model inference synchronously in a threadpool to prevent blocking event loop.
      5. Calculates probabilities, OA presence, risk tiers, and diagnostic warnings.
    """
    # 1. Enforce API key check if configured
    verify_optional_api_key(request.headers.get("x-api-key") if request else None)

    upload_file = file if file is not None else image
    if upload_file is None:
        raise ImageValidationError("No image file provided (upload as 'file' or 'image').")

    # 2. Verify model availability; refuse to serve with 503 if unavailable
    model = getattr(app.state, "model", None)
    if model is None or not app.state.model_loaded:
        logger.error("Inference requested but model is not loaded.")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "error_type": "ServiceUnavailable",
                "message": (
                    "X-ray model is not loaded. Please ensure best_finetuned.keras "
                    "is placed in the model/ directory or configure XRAY_MODEL_PATH."
                ),
                "detail": "Model artifact unavailable.",
            },
        )

    # 3. Read raw file bytes
    try:
        image_bytes = upload_file.file.read()
    except Exception as err:
        raise ImageValidationError("Failed to read uploaded file stream.") from err

    # 4. Validate image by decoding and collect structural warnings
    _, image_warnings = validate_and_inspect_image(image_bytes)

    # 5. Exact notebook preprocessing
    image_tensor = preprocess_image(image_bytes)

    # 6. Execute model prediction
    # model.predict returns softmax probabilities over 5 classes
    try:
        raw_preds = model.predict(image_tensor, verbose=0)
        probs = np.asarray(raw_preds[0], dtype=np.float64)
    except Exception as err:
        logger.error(f"Inference execution failed: {err}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "error_type": "InferenceError",
                "message": "Inference failed during model evaluation.",
                "detail": "Model execution error.",
            },
        )

    # Normalize probabilities defensively to ensure exact sum to 1.0
    prob_sum = float(np.sum(probs))
    if prob_sum > 0:
        probs = probs / prob_sum

    # Top predicted class
    kl_grade_idx = int(np.argmax(probs))
    kl_grade = config.CLASS_LABELS[kl_grade_idx]
    kl_description = config.CLASS_DESCRIPTIONS.get(kl_grade_idx, "Unknown")
    kl_grade_confidence = float(probs[kl_grade_idx])

    # Probability map
    kl_probabilities = {
        label: round(float(p), 4) for label, p in zip(config.CLASS_LABELS, probs)
    }

    # "OA present" = P(KL2) + P(KL3) + P(KL4) > OA_THRESHOLD (0.50)
    p_oa_present = float(np.sum(probs[2:]))
    oa_present = bool(p_oa_present > config.OA_THRESHOLD)
    risk_score = round(p_oa_present * 100.0, 2)

    # Risk tier determination
    if p_oa_present >= config.RISK_THRESHOLD_HIGH:
        risk_tier = "high"
        prediction_text = config.PREDICTION_DEFINITE_OA
    elif p_oa_present >= config.RISK_THRESHOLD_LOW:
        risk_tier = "monitor"
        prediction_text = config.PREDICTION_BORDERLINE
    else:
        risk_tier = "low"
        prediction_text = config.PREDICTION_NO_OR_DOUBTFUL

    # --------------------------------------------------------------------------
    # MODEL PERFORMANCE & CONFIDENCE WARNINGS
    # --------------------------------------------------------------------------
    warnings = list(image_warnings)

    # Low confidence warning
    if kl_grade_confidence < config.LOW_CONFIDENCE_THRESHOLD:
        warnings.append(
            f"KL grade confidence is low ({kl_grade_confidence:.2f} < 0.50); "
            "grade is uncertain, please review the full probability distribution."
        )

    # KL2 weakness warning (KL2 test recall is 11.8%)
    if kl_grade_idx == 2:
        warnings.append(
            "Predicted KL2 is the model's weakest class (test recall 11.8%); "
            "treat as KL1-KL3 borderline and review probabilities."
        )

    # Monitor zone warning
    if config.RISK_THRESHOLD_LOW <= p_oa_present < config.RISK_THRESHOLD_HIGH:
        warnings.append(
            f"OA probability ({p_oa_present:.2f}) falls within the monitor zone "
            f"[{config.RISK_THRESHOLD_LOW}, {config.RISK_THRESHOLD_HIGH}]; "
            "borderline case requires close clinical correlation."
        )

    # Frontend summary fields
    risk_contribution = "high" if p_oa_present >= config.RISK_THRESHOLD_HIGH else ("mid" if p_oa_present >= config.RISK_THRESHOLD_LOW else "low")
    findings_text = f"{kl_grade} ({kl_description}) - {prediction_text}"

    return AnalyzeXRayResponse(
        status="complete",
        kl_grade=kl_grade,
        kl_grade_index=kl_grade_idx,
        kl_description=kl_description,
        kl_grade_confidence=round(kl_grade_confidence, 4),
        kl_probabilities=kl_probabilities,
        oa_present=oa_present,
        oa_probability=round(p_oa_present, 4),
        risk_score=risk_score,
        risk_tier=risk_tier,
        prediction=prediction_text,
        interpretation=config.DEFAULT_INTERPRETATION,
        klGrade=kl_grade_idx,
        findings=findings_text,
        riskContribution=risk_contribution,
        warnings=warnings,
        prototype=config.IS_PROTOTYPE,
    )
