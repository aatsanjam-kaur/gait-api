"""
Test Suite and Request Verification for KOA X-Ray Inference API
================================================================
Generates synthetic in-memory image payloads and executes automated tests against:
  1. GET  /health          : Status, model loaded flag, input shape, class count
  2. POST /analyze-xray    : Valid synthetic knee radiograph strip (JPEG/PNG)
  3. POST /analyze-xray    : Corrupt / truncated image (HTTP 400 expected)
  4. POST /analyze-xray    : Empty file (0 bytes) (HTTP 400 expected)
  5. POST /analyze-xray    : Wrong file type / plain text (HTTP 400 expected)
  6. POST /analyze-xray    : Authentication validation (if XRAY_API_KEY is configured)

Can run against a live running server (http://localhost:8000) or via FastAPI TestClient.
"""

import io
import json
import os
import sys
from typing import Any, Callable, Dict, Optional

import numpy as np
import requests
from PIL import Image

import config

SERVER_URL = f"http://127.0.0.1:{config.PORT}"


# ==============================================================================
# SYNTHETIC TEST ASSET GENERATION
# ==============================================================================

def generate_synthetic_knee_strip(
    width: int = 600,
    height: int = 200,
    aspect_strip: bool = True
) -> bytes:
    """
    Generates a realistic synthetic grayscale knee joint strip in memory.
    Dimensions 600x200 (aspect ratio 3:1) mimics MedicalExpert-II dataset strips.
    Includes background noise and bright horizontal anatomical bone/joint band.
    """
    # Background baseline noise
    base = np.random.normal(loc=90, scale=15, size=(height, width)).clip(0, 255)

    # Simulate distal femur and proximal tibia density (brighter joint condyle band)
    joint_center = height // 2
    y_coords, _ = np.indices((height, width))
    dist_from_joint = np.abs(y_coords - joint_center)

    # Bright bone regions above and below joint line
    bone_mask = np.exp(-((dist_from_joint - 45) ** 2) / (2 * (25 ** 2)))
    image_data = np.clip(base + 110 * bone_mask, 0, 255).astype(np.uint8)

    # Create grayscale PIL image and export as PNG bytes
    pil_img = Image.fromarray(image_data).convert("L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def generate_corrupt_file() -> bytes:
    """Generates corrupted image bytes with invalid header data."""
    return b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\xff" * 50 + b"\x00\x12\x34CorruptChunkPayload"


def generate_empty_file() -> bytes:
    """Generates empty 0-byte payload."""
    return b""


def generate_text_file() -> bytes:
    """Generates a non-image plain text file payload."""
    return b"Patient Name: John Doe\nModality: Radiograph\nDiagnosis: Pending inspection."


# ==============================================================================
# MOCK MODEL FOR LOCAL / TEST CLIENT VERIFICATION
# ==============================================================================

class MockEfficientNetModel:
    """Mock model simulating 5-class softmax output when running in mock test mode."""
    def predict(self, tensor, verbose=0):
        # Return realistic probability distribution for KL3 moderate OA
        return np.array([[0.02, 0.05, 0.31, 0.52, 0.10]], dtype=np.float32)


# ==============================================================================
# TEST RUNNER SUITE
# ==============================================================================

def run_tests():
    """Executes the complete test suite against live server or TestClient."""

    def execute_suite(
        get_fn: Callable[[str, Optional[Dict]], Any],
        post_fn: Callable[[str, Dict, Optional[Dict]], Any]
    ):
        auth_headers = {}
        if config.API_KEY:
            auth_headers["x-api-key"] = config.API_KEY

        # ----------------------------------------------------------------------
        # TEST 1: GET /health
        # ----------------------------------------------------------------------
        print("\n" + "=" * 75)
        print("TEST 1: GET /health")
        print("=" * 75)
        res = get_fn("/health", None)
        print(f"Status Code: {res.status_code}")
        print("Response:", json.dumps(res.json(), indent=2))
        assert res.status_code == 200, f"Health check failed with status {res.status_code}"
        health_data = res.json()
        assert health_data.get("status") == "ok", "Expected status 'ok'"
        assert "model_loaded" in health_data, "Missing 'model_loaded' field"
        assert health_data.get("model_input_shape") == [None, 224, 224, 3]
        assert health_data.get("num_classes") == 5

        # ----------------------------------------------------------------------
        # TEST 2: POST /analyze-xray (Valid synthetic knee strip)
        # ----------------------------------------------------------------------
        print("\n" + "=" * 75)
        print("TEST 2: POST /analyze-xray (Valid synthetic knee joint strip 600x200)")
        print("=" * 75)
        valid_png = generate_synthetic_knee_strip()
        files = {"file": ("synthetic_knee.png", valid_png, "image/png")}
        res = post_fn("/analyze-xray", files, auth_headers)
        print(f"Status Code: {res.status_code}")
        print("Response:", json.dumps(res.json(), indent=2))
        assert res.status_code == 200, f"Analysis failed: {res.text}"

        body = res.json()
        assert body.get("status") == "complete"
        assert body.get("kl_grade") in config.CLASS_LABELS
        assert 0 <= body.get("kl_grade_index") <= 4
        assert body.get("kl_description") in config.CLASS_DESCRIPTIONS.values()
        assert 0.0 <= body.get("kl_grade_confidence") <= 1.0
        assert "kl_probabilities" in body

        # Assert probabilities sum to approximately 1.0
        probs = body["kl_probabilities"]
        assert len(probs) == 5
        prob_sum = sum(probs.values())
        assert abs(prob_sum - 1.0) < 0.01, f"Probabilities do not sum to 1.0 (sum={prob_sum})"

        # Assert OA presence logic
        expected_oa_prob = round(probs["KL2"] + probs["KL3"] + probs["KL4"], 4)
        assert abs(body.get("oa_probability") - expected_oa_prob) < 0.01
        assert body.get("oa_present") == (body.get("oa_probability") > config.OA_THRESHOLD)
        assert body.get("risk_score") == round(body.get("oa_probability") * 100.0, 2)
        assert body.get("risk_tier") in ("high", "monitor", "low")
        assert "warnings" in body
        assert isinstance(body["warnings"], list)
        assert body.get("prototype") is True

        # ----------------------------------------------------------------------
        # TEST 3: POST /analyze-xray (Corrupted image -> HTTP 400)
        # ----------------------------------------------------------------------
        print("\n" + "=" * 75)
        print("TEST 3: POST /analyze-xray (Corrupt file rejection -> Expected HTTP 400)")
        print("=" * 75)
        corrupt_bytes = generate_corrupt_file()
        files = {"file": ("corrupt.png", corrupt_bytes, "image/png")}
        res = post_fn("/analyze-xray", files, auth_headers)
        print(f"Status Code: {res.status_code}")
        print("Response:", json.dumps(res.json(), indent=2))
        assert res.status_code == 400, f"Expected 400 Bad Request, got {res.status_code}"
        assert res.json().get("error_type") == "ImageValidationError"

        # ----------------------------------------------------------------------
        # TEST 4: POST /analyze-xray (Empty 0-byte file -> HTTP 400)
        # ----------------------------------------------------------------------
        print("\n" + "=" * 75)
        print("TEST 4: POST /analyze-xray (Empty file rejection -> Expected HTTP 400)")
        print("=" * 75)
        empty_bytes = generate_empty_file()
        files = {"file": ("empty.png", empty_bytes, "image/png")}
        res = post_fn("/analyze-xray", files, auth_headers)
        print(f"Status Code: {res.status_code}")
        print("Response:", json.dumps(res.json(), indent=2))
        assert res.status_code == 400, f"Expected 400 Bad Request, got {res.status_code}"
        assert "empty" in res.json().get("message", "").lower()

        # ----------------------------------------------------------------------
        # TEST 5: POST /analyze-xray (Wrong file type / text file -> HTTP 400)
        # ----------------------------------------------------------------------
        print("\n" + "=" * 75)
        print("TEST 5: POST /analyze-xray (Text file rejection -> Expected HTTP 400)")
        print("=" * 75)
        text_bytes = generate_text_file()
        files = {"file": ("report.txt", text_bytes, "text/plain")}
        res = post_fn("/analyze-xray", files, auth_headers)
        print(f"Status Code: {res.status_code}")
        print("Response:", json.dumps(res.json(), indent=2))
        assert res.status_code == 400, f"Expected 400 Bad Request, got {res.status_code}"

        # ----------------------------------------------------------------------
        # TEST 6: POST /analyze-xray (Authentication validation)
        # ----------------------------------------------------------------------
        print("\n" + "=" * 75)
        print("TEST 6: Authentication validation")
        print("=" * 75)
        if config.API_KEY:
            # A: Missing API key header
            print("[*] Testing missing 'x-api-key' header (Expected HTTP 401)...")
            files = {"file": ("synthetic_knee.png", valid_png, "image/png")}
            res = post_fn("/analyze-xray", files, {})
            print(f"Status Code: {res.status_code}")
            assert res.status_code == 401, f"Expected 401 Unauthorized, got {res.status_code}"

            # B: Invalid API key header
            print("[*] Testing invalid 'x-api-key' header (Expected HTTP 401)...")
            files = {"file": ("synthetic_knee.png", valid_png, "image/png")}
            res = post_fn("/analyze-xray", files, {"x-api-key": "invalid-secret-key"})
            print(f"Status Code: {res.status_code}")
            assert res.status_code == 401, f"Expected 401 Unauthorized, got {res.status_code}"

            # C: Correct API key header
            print("[*] Testing correct 'x-api-key' header (Expected HTTP 200)...")
            files = {"file": ("synthetic_knee.png", valid_png, "image/png")}
            res = post_fn("/analyze-xray", files, {"x-api-key": config.API_KEY})
            print(f"Status Code: {res.status_code}")
            assert res.status_code == 200, f"Expected 200 OK, got {res.status_code}"
            print("[*] API key enforcement verified successfully.")
        else:
            print("[*] XRAY_API_KEY is unset; verifying unauthenticated requests succeed...")
            files = {"file": ("synthetic_knee.png", valid_png, "image/png")}
            res = post_fn("/analyze-xray", files, None)
            assert res.status_code == 200
            print("[*] Open access verified successfully.")

        print("\n" + "=" * 75)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 75)

    # Check for live running server first
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=1.0)
        print(f"[*] Detected active live server at {SERVER_URL}")
        def live_get(path, headers):
            return requests.get(f"{SERVER_URL}{path}", headers=headers or {})
        def live_post(path, files, headers):
            return requests.post(f"{SERVER_URL}{path}", files=files, headers=headers or {})
        execute_suite(live_get, live_post)
    except Exception:
        print("[*] No active server detected at http://localhost:8000. Using in-process FastAPI TestClient...")
        from fastapi.testclient import TestClient
        from xray_api import app

        with TestClient(app) as client:
            # If model artifact was not present locally, attach mock model for testing API layer
            if app.state.model is None:
                print("[*] Attaching mock inference model to app.state for unit test verification...")
                app.state.model = MockEfficientNetModel()
                app.state.model_loaded = True

            def test_get(path, headers):
                return client.get(path, headers=headers or {})
            def test_post(path, files, headers):
                return client.post(path, files=files, headers=headers or {})

            execute_suite(test_get, test_post)


if __name__ == "__main__":
    run_tests()
