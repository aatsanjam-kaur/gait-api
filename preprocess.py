"""
Image Preprocessing and Validation for KOA X-Ray Inference
===========================================================
Replicates the exact training preprocessing pipeline from OA_xray.ipynb:
  1. tf.image.decode_image(bytes, channels=1, expand_animations=False)
  2. tf.cast(image, tf.float32)
  3. tf.image.resize_with_pad(image, 224, 224)
  4. tf.image.grayscale_to_rgb(image)
  5. tf.expand_dims(image, axis=0) -> shape (1, 224, 224, 3)

NOTE: Do NOT divide by 255 (EfficientNetB0 handles normalization internally).
"""

import io
import logging
from typing import List, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

import config

logger = logging.getLogger("xray_preprocess")

# Lazy/conditional import of TensorFlow to prevent import-time failure
# in lightweight environments where TensorFlow is not installed.
try:
    import tensorflow as tf
except ImportError:
    tf = None
    logger.warning("TensorFlow is not installed in the current environment.")


class ImageValidationError(Exception):
    """Raised when an uploaded image fails validation (empty, corrupt, unsupported, or wrong dimensions)."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def validate_and_inspect_image(image_bytes: bytes) -> Tuple[Image.Image, List[str]]:
    """
    Validates uploaded raw bytes by actively decoding the image.
    Does NOT rely on client Content-Type headers (which are often application/octet-stream).

    Checks performed:
      1. Non-empty byte payload.
      2. Max file size <= 10 MB.
      3. Valid decodable image structure.
      4. Supported formats: JPEG or PNG only.
      5. Minimum dimensions: at least 64x64 pixels.

    Inspections that generate non-blocking warnings:
      1. Color channels differ significantly (detects non-grayscale color photos).
      2. Aspect ratio outside [1.5, 5.0] (joint strips vs full radiographs).

    Returns:
      (PIL.Image, List of warning strings)
    """
    warnings: List[str] = []

    # 1. Empty payload check
    if not image_bytes or len(image_bytes) == 0:
        raise ImageValidationError("Uploaded image file is empty (0 bytes).")

    # 2. Maximum file size check
    if len(image_bytes) > config.MAX_FILE_SIZE_BYTES:
        size_mb = len(image_bytes) / (1024 * 1024)
        raise ImageValidationError(
            f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit of 10 MB."
        )

    # 3. Decoding and header verification
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.verify()  # Verifies file integrity/format headers
        # Re-open after verify() because verify() clears file pointer and image buffers
        pil_img = Image.open(io.BytesIO(image_bytes))
    except (UnidentifiedImageError, OSError, SyntaxError) as err:
        raise ImageValidationError(
            "Corrupt, truncated, or unreadable image file. "
            "Please ensure the file is a valid image."
        ) from err

    # 4. Format validation (strictly JPEG or PNG)
    detected_format = (pil_img.format or "").upper()
    if detected_format not in config.ALLOWED_IMAGE_FORMATS:
        raise ImageValidationError(
            f"Unsupported image format: '{detected_format or 'unknown'}'. "
            f"Only JPEG and PNG formats are supported."
        )

    # 5. Spatial dimensions check
    width, height = pil_img.size
    if width < config.MIN_IMAGE_WIDTH or height < config.MIN_IMAGE_HEIGHT:
        raise ImageValidationError(
            f"Image dimensions ({width}x{height}) are too small. "
            f"Minimum required dimensions are {config.MIN_IMAGE_WIDTH}x{config.MIN_IMAGE_HEIGHT} pixels."
        )

    # --------------------------------------------------------------------------
    # WARNING INSPECTIONS (Appended to warnings, does NOT fail request)
    # --------------------------------------------------------------------------

    # Check A: Aspect ratio outside expected joint-strip range [1.5, 5.0]
    aspect_ratio = float(width) / float(height)
    if aspect_ratio < config.ASPECT_RATIO_MIN or aspect_ratio > config.ASPECT_RATIO_MAX:
        warnings.append(
            f"Image aspect ratio ({aspect_ratio:.2f}) is outside the expected "
            f"joint-strip range [{config.ASPECT_RATIO_MIN}, {config.ASPECT_RATIO_MAX}]. "
            "Training data were knee-joint strips (MedicalExpert-II); full radiographs "
            "or non-strip crops may yield less reliable predictions."
        )

    # Check B: Color photograph detection (knee X-rays must be grayscale radiographs)
    if pil_img.mode in ("RGB", "RGBA"):
        rgb_data = np.asarray(pil_img.convert("RGB"), dtype=np.float32)
        # Difference between R-G and R-B channels across all pixels
        rg_diff = np.abs(rgb_data[:, :, 0] - rgb_data[:, :, 1])
        rb_diff = np.abs(rgb_data[:, :, 0] - rgb_data[:, :, 2])
        mean_channel_diff = float(np.mean(rg_diff + rb_diff))
        if mean_channel_diff > config.COLOR_DIFF_THRESHOLD:
            warnings.append(
                "Image appears to be a color photograph (distinct R/G/B channels detected). "
                "Knee radiographs are expected to be grayscale X-ray images."
            )

    return pil_img, warnings


def preprocess_image(image_bytes: bytes):
    """
    Executes the exact preprocessing pipeline from OA_xray.ipynb:
      tf.image.decode_image(bytes, channels=1, expand_animations=False)
      -> tf.cast(float32)
      -> tf.image.resize_with_pad(image, 224, 224)
      -> tf.image.grayscale_to_rgb
      -> add batch dim

    Returns:
      A batched tensor of shape (1, 224, 224, 3) with float32 values [0.0, 255.0].
      Does NOT divide by 255 (EfficientNetB0 internally rescales inputs).
    """
    if tf is not None:
        # Decode as single-channel grayscale (channels=1)
        image = tf.image.decode_image(image_bytes, channels=1, expand_animations=False)
        # Cast to float32
        image = tf.cast(image, tf.float32)
        # Resize with pad to maintain aspect ratio with zero-padding
        target_h, target_w = config.TARGET_IMAGE_SIZE
        image = tf.image.resize_with_pad(image, target_h, target_w)
        # Convert 1-channel grayscale to 3-channel RGB
        image = tf.image.grayscale_to_rgb(image)
        # Expand batch dimension to (1, 224, 224, 3)
        image = tf.expand_dims(image, axis=0)
        return image

    # Fallback implementation using PIL and NumPy when TensorFlow is not installed
    # (strictly for local unit tests / mock environments)
    logger.info("Using PIL/NumPy preprocessing fallback (TensorFlow not detected).")
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("L")
    w, h = pil_img.size
    target_w, target_h = config.TARGET_IMAGE_SIZE
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    padded_img = Image.new("L", (target_w, target_h), 0)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    padded_img.paste(resized_img, (pad_x, pad_y))

    arr = np.array(padded_img, dtype=np.float32)
    # Stack grayscale into 3 channels
    rgb_arr = np.stack([arr, arr, arr], axis=-1)
    # Add batch dim -> (1, 224, 224, 3)
    batched_arr = np.expand_dims(rgb_arr, axis=0)
    return batched_arr
