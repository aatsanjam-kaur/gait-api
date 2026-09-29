from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile
import os
import json
import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = FastAPI(title="OA Gait Analysis API")


# ============================================================
# LANDMARK CONSTANTS
# ============================================================

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_HIP = 23
RIGHT_HIP = 24

LEFT_KNEE = 25
RIGHT_KNEE = 26

LEFT_ANKLE = 27
RIGHT_ANKLE = 28


# ============================================================
# LOAD POSE LANDMARKER
# ============================================================

base_options = python.BaseOptions(
    model_asset_path="pose_landmarker.task"
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)

detector = vision.PoseLandmarker.create_from_options(
    options
)


# ============================================================
# LOAD MODEL + FEATURES
# ============================================================

MODEL_PATH = "logistic_regression.joblib"
FEATURE_COLUMNS_PATH = "feature_columns.json"

model = joblib.load(MODEL_PATH)

with open(FEATURE_COLUMNS_PATH, "r") as f:
    feature_columns = json.load(f)


# ============================================================
# HELPERS
# ============================================================

def calculate_angle(a, b, c):

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(
        ba, bc
    ) / (
        np.linalg.norm(ba)
        * np.linalg.norm(bc)
        + 1e-8
    )

    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    return np.degrees(
        np.arccos(cosine_angle)
    )


def summarize(values, prefix):

    values = values[
        ~np.isnan(values)
    ]

    if len(values) == 0:

        return {
            f"{prefix}_mean": 0,
            f"{prefix}_std": 0,
            f"{prefix}_min": 0,
            f"{prefix}_max": 0,
            f"{prefix}_range": 0
        }

    return {
        f"{prefix}_mean": np.mean(values),
        f"{prefix}_std": np.std(values),
        f"{prefix}_min": np.min(values),
        f"{prefix}_max": np.max(values),
        f"{prefix}_range": np.ptp(values)
    }


# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def extract_landmarks(video_path):

    cap = cv2.VideoCapture(video_path)

    all_landmarks = []

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        result = detector.detect(mp_image)

        if len(result.pose_landmarks) > 0:

            frame_landmarks = []

            for lm in result.pose_landmarks[0]:

                frame_landmarks.append([
                    lm.x,
                    lm.y,
                    lm.z,
                    lm.visibility
                ])

            all_landmarks.append(
                frame_landmarks
            )

        else:

            all_landmarks.append(
                [[np.nan, np.nan, np.nan, np.nan]] * 33
            )

    cap.release()

    if len(all_landmarks) == 0:
        raise ValueError(
            "No landmarks extracted from video"
        )

    return np.array(all_landmarks)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(all_landmarks):

    xy = all_landmarks[:, :, :2]

    left_knee_angles = []
    right_knee_angles = []

    left_hip_angles = []
    right_hip_angles = []

    for frame in range(len(xy)):

        required = [
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
            LEFT_HIP,
            RIGHT_HIP,
            LEFT_KNEE,
            RIGHT_KNEE,
            LEFT_ANKLE,
            RIGHT_ANKLE
        ]

        if np.isnan(
            xy[frame, required]
        ).any():

            left_knee_angles.append(np.nan)
            right_knee_angles.append(np.nan)

            left_hip_angles.append(np.nan)
            right_hip_angles.append(np.nan)

            continue

        left_knee_angles.append(
            calculate_angle(
                xy[frame, LEFT_HIP],
                xy[frame, LEFT_KNEE],
                xy[frame, LEFT_ANKLE]
            )
        )

        right_knee_angles.append(
            calculate_angle(
                xy[frame, RIGHT_HIP],
                xy[frame, RIGHT_KNEE],
                xy[frame, RIGHT_ANKLE]
            )
        )

        left_hip_angles.append(
            calculate_angle(
                xy[frame, LEFT_SHOULDER],
                xy[frame, LEFT_HIP],
                xy[frame, LEFT_KNEE]
            )
        )

        right_hip_angles.append(
            calculate_angle(
                xy[frame, RIGHT_SHOULDER],
                xy[frame, RIGHT_HIP],
                xy[frame, RIGHT_KNEE]
            )
        )

    left_knee_angles = np.array(left_knee_angles)
    right_knee_angles = np.array(right_knee_angles)

    left_hip_angles = np.array(left_hip_angles)
    right_hip_angles = np.array(right_hip_angles)

    features = {}

    features.update(
        summarize(left_knee_angles, "left_knee")
    )

    features.update(
        summarize(right_knee_angles, "right_knee")
    )

    features.update(
        summarize(left_hip_angles, "left_hip")
    )

    features.update(
        summarize(right_hip_angles, "right_hip")
    )

    left_ankle_x = xy[:, LEFT_ANKLE, 0]
    right_ankle_x = xy[:, RIGHT_ANKLE, 0]

    left_ankle_y = xy[:, LEFT_ANKLE, 1]
    right_ankle_y = xy[:, RIGHT_ANKLE, 1]

    features["left_ankle_horizontal_range"] = (
        np.nanmax(left_ankle_x)
        - np.nanmin(left_ankle_x)
    )

    features["right_ankle_horizontal_range"] = (
        np.nanmax(right_ankle_x)
        - np.nanmin(right_ankle_x)
    )

    features["left_ankle_vertical_range"] = (
        np.nanmax(left_ankle_y)
        - np.nanmin(left_ankle_y)
    )

    features["right_ankle_vertical_range"] = (
        np.nanmax(right_ankle_y)
        - np.nanmin(right_ankle_y)
    )

    features["pose_detection_rate"] = np.mean(
        ~np.isnan(
            xy[:, LEFT_HIP, 0]
        )
    )

    feature_df = pd.DataFrame([features])

    feature_df = feature_df[
        feature_columns
    ]

    return feature_df


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def health():

    return {
        "status": "running",
        "model": "OA Gait Logistic Regression"
    }


# ============================================================
# GAIT API
# ============================================================

@app.post("/analyze-gait")
async def analyze_gait(
    file: UploadFile = File(...)
):

    temp_path = None

    try:

        suffix = os.path.splitext(
            file.filename
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(
                await file.read()
            )

            temp_path = temp_file.name

        all_landmarks = extract_landmarks(
            temp_path
        )

        feature_df = build_features(
            all_landmarks
        )

        prediction = int(
            model.predict(
                feature_df
            )[0]
        )

        probability = float(
            model.predict_proba(
                feature_df
            )[0][1]
        )

        return {
            "prediction":
                "KOA"
                if prediction == 1
                else "NM",

            "confidence":
                round(
                    probability,
                    4
                ),

            "risk_score":
                round(
                    probability * 100,
                    2
                )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        if (
            temp_path is not None
            and os.path.exists(temp_path)
        ):
            os.remove(temp_path)
