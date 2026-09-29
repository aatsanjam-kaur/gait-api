"""
Parity Verification Script for KOA X-Ray Pipeline
===================================================
Run this script in Google Colab (or any environment with real X-ray images and
TensorFlow) to verify that the API's preprocessing matches the notebook's training
pipeline with mathematical parity.

For N images given on the command line, this script:
  1. Executes the notebook's exact `load_image` pipeline.
  2. Executes the API's `preprocess_image` function from `preprocess.py`.
  3. Asserts the two resulting tensors are identical (max absolute diff < 1e-6).
  4. Runs both through `best_finetuned.keras` and verifies predictions match.

Usage:
  python parity_check.py /path/to/knee1.png /path/to/knee2.jpg ...
"""

import os
import sys
import glob
from typing import List

import numpy as np

# Require TensorFlow for parity check
try:
    import tensorflow as tf
except ImportError:
    print("[ERROR] TensorFlow is required to run parity_check.py.")
    print("Please run in an environment with tensorflow==2.20.0 installed (e.g. Google Colab).")
    sys.exit(1)

import config
from preprocess import preprocess_image


def notebook_load_image(filepath: str) -> tf.Tensor:
    """
    Exact implementation of cell 8 from OA_xray.ipynb:
      image = tf.io.read_file(filepath)
      image = tf.image.decode_image(image, channels=1, expand_animations=False)
      image = tf.cast(image, tf.float32)
      image = tf.image.resize_with_pad(image, IMG_SIZE[0], IMG_SIZE[1])
      image = tf.image.grayscale_to_rgb(image)
    """
    image = tf.io.read_file(filepath)
    image = tf.image.decode_image(image, channels=1, expand_animations=False)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, config.TARGET_IMAGE_SIZE[0], config.TARGET_IMAGE_SIZE[1])
    image = tf.image.grayscale_to_rgb(image)
    return tf.expand_dims(image, axis=0)


def resolve_model():
    """Attempts to locate the trained model across candidate paths."""
    candidate_paths = [
        config.MODEL_PATH,
        os.path.join(config.MODEL_DIR, "best_finetuned.keras"),
        "/content/drive/MyDrive/XRAY DATASET/best_finetuned.keras",
        "/content/drive/MyDrive/XRAY_DATASET/best_finetuned.keras",
        "/content/best_finetuned.keras",
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            print(f"[*] Found model at: {p}")
            return tf.keras.models.load_model(p, compile=False)

    print(f"[!] Warning: No model file found among {candidate_paths}.")
    print("    Parity will be verified on input tensors only (skipping model forward pass).")
    return None


def run_parity_check(image_paths: List[str]):
    """Iterates through images and verifies preprocessing parity and model prediction parity."""
    if not image_paths:
        print("[!] No image paths provided.")
        print("Usage: python parity_check.py <image1.png> [image2.png ...]")
        return

    # Expand any shell wildcards if not expanded by the shell
    expanded_paths = []
    for p in image_paths:
        matches = glob.glob(p)
        if matches:
            expanded_paths.extend(matches)
        else:
            expanded_paths.append(p)

    print("=" * 80)
    print(f"RUNNING PARITY VERIFICATION ON {len(expanded_paths)} IMAGE(S)")
    print("=" * 80)

    model = resolve_model()

    passed_count = 0
    for idx, img_path in enumerate(expanded_paths, 1):
        if not os.path.exists(img_path):
            print(f"[{idx}/{len(expanded_paths)}] SKIP: File not found '{img_path}'")
            continue

        print(f"\n[{idx}/{len(expanded_paths)}] Testing: {img_path}")

        # (a) Notebook pipeline
        nb_tensor = notebook_load_image(img_path)

        # (b) API pipeline
        with open(img_path, "rb") as f:
            raw_bytes = f.read()
        api_tensor = preprocess_image(raw_bytes)

        # Convert to numpy arrays
        nb_arr = nb_tensor.numpy()
        api_arr = api_tensor.numpy()

        # Check shape and dtype
        assert nb_arr.shape == api_arr.shape == (1, 224, 224, 3), (
            f"Shape mismatch: notebook={nb_arr.shape} vs api={api_arr.shape}"
        )
        assert nb_arr.dtype == api_arr.dtype == np.float32, (
            f"Dtype mismatch: notebook={nb_arr.dtype} vs api={api_arr.dtype}"
        )

        # Assert max absolute difference < 1e-6
        max_abs_diff = float(np.max(np.abs(nb_arr - api_arr)))
        print(f"    - Input tensor shape: {nb_arr.shape}")
        print(f"    - Max absolute tensor difference: {max_abs_diff:.6e}")
        assert max_abs_diff < 1e-6, (
            f"Parity assertion failed! Max absolute diff {max_abs_diff:.6e} >= 1e-6"
        )
        print("    -> Preprocessing parity ASSERTION PASSED (max_diff < 1e-6)")

        # Run forward pass if model is loaded
        if model is not None:
            nb_preds = model.predict(nb_tensor, verbose=0)[0]
            api_preds = model.predict(api_tensor, verbose=0)[0]

            pred_max_diff = float(np.max(np.abs(nb_preds - api_preds)))
            nb_class = config.CLASS_LABELS[int(np.argmax(nb_preds))]
            api_class = config.CLASS_LABELS[int(np.argmax(api_preds))]

            print(f"    - Notebook prediction: {nb_class} (p={float(np.max(nb_preds)):.4f})")
            print(f"    - API prediction:      {api_class} (p={float(np.max(api_preds)):.4f})")
            print(f"    - Max prediction diff: {pred_max_diff:.6e}")
            assert pred_max_diff < 1e-5, f"Prediction discrepancy detected! Diff: {pred_max_diff}"
            print("    -> Model prediction parity PASSED")

        passed_count += 1

    print("\n" + "=" * 80)
    print(f"PARITY CHECK SUMMARY: {passed_count}/{len(expanded_paths)} images PASSED successfully.")
    print("Preprocessed tensors strictly match the training pipeline.")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python parity_check.py <image1.png> [image2.png ...]")
        print("Example (Colab):")
        print("  python parity_check.py /content/drive/MyDrive/XRAY_DATASET/test/*.png")
        sys.exit(0)

    run_parity_check(sys.argv[1:])
