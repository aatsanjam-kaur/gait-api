# Knee Osteoarthritis (KOA) X-Ray Inference Microservice

A production-ready FastAPI microservice for automated Kellgren-Lawrence (KL) grading and knee osteoarthritis (KOA) screening from plain knee radiographs.

The inference pipeline reproduces the exact preprocessing, normalization, and classification logic established in the training notebook `OA_xray.ipynb`.

---

## 1. Clinical Context, Performance & Model Limitations

### Model Architecture
- **Base Architecture**: EfficientNetB0 transfer learning with custom classification head.
- **Artifact**: `best_finetuned.keras` (trained with TensorFlow 2.20.0).
- **Classification Target**: 5-class Kellgren-Lawrence grading:
  - **KL0**: Normal (no radiographic features of OA)
  - **KL1**: Doubtful (minute osteophytes, doubtful clinical significance)
  - **KL2**: Mild (definite osteophytes, potential joint space narrowing)
  - **KL3**: Moderate (multiple moderate osteophytes, clear joint space narrowing, some sclerosis)
  - **KL4**: Severe (large osteophytes, marked joint space narrowing, severe sclerosis, bone contour deformity)
- **Binary OA Definition**: "OA Present" is defined as definite osteoarthritis ($KL \ge 2$), evaluated by cumulative softmax probability:
  $$\text{OA Present} \iff P(KL2) + P(KL3) + P(KL4) > 0.50$$

### Verified Test-Set Performance
Evaluated on the independent test split ($N = 245$ images, MedicalExpert-II dataset):

| Metric | Score | Clinical Interpretation |
| :--- | :--- | :--- |
| **5-Class Accuracy** | **62.9%** | Overall exact agreement across all 5 discrete KL grades. |
| **Binary OA Accuracy** | **87.3%** | High accuracy in distinguishing non-OA ($KL0-1$) from definite OA ($KL2-4$). |
| **Binary Sensitivity** | **74.2%** | True positive rate for detecting $KL \ge 2$ osteoarthritis. |
| **Binary Specificity** | **95.9%** | High negative predictive reliability; rarely false-alarms on healthy/doubtful joints. |
| **KL0 Recall** | **90.8%** | Excellent identification of normal knees. |
| **KL4 Recall** | **74.2%** | Strong identification of severe end-stage osteoarthritis. |
| **KL2 Recall** | **11.8%** | **Significant clinical limitation.** |

### Key Clinical Limitations & Operational Guidance
1. **KL2 Weakness (Low Recall)**: The model rarely predicts KL2 as the top softmax class (11.8% recall on test data). Mild OA cases are frequently divided into the neighbouring classes ($KL1$ or $KL3$). When `kl_grade_index == 2`, the API injects an explicit warning: `Predicted KL2 is the model's weakest class; treat as KL1-KL3 and review probabilities.`
2. **Aspect Ratio & Crop Sensitivity**: The training dataset (MedicalExpert-II) consists of cropped horizontal **knee-joint strips** with aspect ratios between **1.5:1 and 5.0:1** (typically 2:1 to 4:1). Full uncropped leg radiographs or square non-joint crops will trigger an automated warning and yield less reliable predictions.
3. **Screening Aid Only**: This microservice is an investigational screening aid and triage accelerator. It is **not** an automated diagnostic system and must not replace clinical evaluation or radiologist sign-off.

---

## 2. API Contract

### Endpoints Overview

| Method | Endpoint | Content-Type | Auth | Description |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | `application/json` | None | Service liveness, model load state, expected shapes, auth status. |
| `POST` | `/analyze-xray` | `multipart/form-data` | Header `x-api-key` (Optional) | Accepts knee X-ray radiograph (`file`), returns KL grade & OA risk. |

---

### Request Specification (`POST /analyze-xray`)

- **Field Name**: `file` (required)
- **Supported Formats**: JPEG, PNG
- **Validation**: Active binary decoding (does not rely on incoming `Content-Type` header)
- **Max File Size**: 10 MB
- **Minimum Dimensions**: $64 \times 64$ pixels

### Response Schema (`POST /analyze-xray`)

```json
{
  "status": "complete",
  "kl_grade": "KL3",
  "kl_grade_index": 3,
  "kl_description": "Moderate",
  "kl_grade_confidence": 0.52,
  "kl_probabilities": {
    "KL0": 0.02,
    "KL1": 0.05,
    "KL2": 0.31,
    "KL3": 0.52,
    "KL4": 0.10
  },
  "oa_present": true,
  "oa_probability": 0.93,
  "risk_score": 93.0,
  "risk_tier": "high",
  "prediction": "Definite OA (KL2-4)",
  "interpretation": "Automated KL-grade estimate from a knee X-ray. Screening aid, not a diagnosis.",
  "warnings": [],
  "prototype": true
}
```

### Response Field Descriptions

| Field | Type | Description |
| :--- | :--- | :--- |
| `status` | string | Execution status (`"complete"` or error string). |
| `kl_grade` | string | Top predicted Kellgren-Lawrence grade (`"KL0"` to `"KL4"`). |
| `kl_grade_index` | integer | Discrete index corresponding to predicted grade ($0 \dots 4$). |
| `kl_description` | string | Clinical description (`"Normal"`, `"Doubtful"`, `"Mild"`, `"Moderate"`, `"Severe"`). |
| `kl_grade_confidence` | float | Top softmax class probability ($0.0 \dots 1.0$). |
| `kl_probabilities` | object | Dictionary mapping every class (`"KL0"`–`"KL4"`) to its softmax probability. Sums to 1.0. |
| `oa_present` | boolean | `true` if $P(KL2) + P(KL3) + P(KL4) > 0.50$. |
| `oa_probability` | float | Cumulative probability of definite OA: $\sum_{i=2}^4 P(KL_i)$. |
| `risk_score` | float | Normalized percentage score: `oa_probability * 100.0`. |
| `risk_tier` | string | Categorized risk tier: `"high"` ($\ge 0.65$), `"monitor"` ($0.35 - 0.65$), `"low"` ($< 0.35$). |
| `prediction` | string | Outcome summary: `"Definite OA (KL2-4)"`, `"No or doubtful OA (KL0-1)"`, or `"Borderline"`. |
| `interpretation` | string | Standard clinical disclaimer. |
| `warnings` | array[string] | Active runtime warnings (aspect ratio, non-grayscale color artifacts, low confidence, KL2 weakness). |
| `prototype` | boolean | Set to `true` to flag prototype model deployment. |

---

## 3. Client Integration Examples

> **Important**: Model inference on GPU or CPU can take between 200 ms and 2.5 s depending on hardware. Backend callers should set their HTTP client request timeouts to **at least 30 seconds**.

### cURL

```bash
# Standard request (No API key)
curl -X POST "http://localhost:8000/analyze-xray" \
  -F "file=@/path/to/knee_joint_strip.png"

# Request with API key header (if XRAY_API_KEY is configured)
curl -X POST "http://localhost:8000/analyze-xray" \
  -H "x-api-key: your-secret-api-key" \
  -F "file=@/path/to/knee_joint_strip.png"
```

### Node.js (Axios + Form-Data)

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

async function analyzeKneeXray(imageFilePath) {
  const form = new FormData();
  form.append('file', fs.createReadStream(imageFilePath));

  const headers = {
    ...form.getHeaders(),
    // Required header when tunnelling through free ngrok:
    'ngrok-skip-browser-warning': 'true',
  };

  // Optional: add x-api-key if enabled on the backend
  if (process.env.XRAY_API_KEY) {
    headers['x-api-key'] = process.env.XRAY_API_KEY;
  }

  try {
    const response = await axios.post('http://localhost:8000/analyze-xray', form, {
      headers,
      timeout: 30000, // 30-second timeout
      maxContentLength: 10 * 1024 * 1024,
    });

    console.log('Inference Result:', response.data);
    return response.data;
  } catch (error) {
    if (error.response) {
      console.error(`Inference Error [${error.response.status}]:`, error.response.data);
    } else {
      console.error('Network/Timeout Error:', error.message);
    }
    throw error;
  }
}

// Example invocation
// analyzeKneeXray('./sample_knee_strip.png');
```

---

## 4. Google Colab + ngrok Deployment Guide

To serve the model on a free GPU in Google Colab and expose it to your web/mobile frontend:

### Step 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Clone / Copy Files & Model
```bash
# Copy best_finetuned.keras into the xray/model/ directory
mkdir -p xray/model
cp "/content/drive/MyDrive/XRAY DATASET/best_finetuned.keras" xray/model/best_finetuned.keras
```

### Step 3: Install Dependencies
```bash
pip install -r xray/requirements.txt
```

### Step 4: Configure Environment & Run in Colab
Execute the following cell in your Colab notebook:

```python
import os
import threading
import nest_asyncio
import uvicorn
from pyngrok import ngrok

# Enable nested event loops for Jupyter / Colab
nest_asyncio.apply()

# 1. Configuration (use environment variables, never hardcode tokens)
os.environ["XRAY_MODEL_PATH"] = "xray/model/best_finetuned.keras"
# Optional auth key:
# os.environ["XRAY_API_KEY"] = "your-secure-secret-key"

# 2. Configure ngrok authtoken from Colab secrets or environment
NGROK_TOKEN = os.environ.get("NGROK_AUTHTOKEN")
if not NGROK_TOKEN:
    from google.colab import userdata
    NGROK_TOKEN = userdata.get('NGROK_AUTHTOKEN')

ngrok.set_auth_token(NGROK_TOKEN)

# 3. Start public tunnel
tunnel = ngrok.connect(8000)
print("\n" + "=" * 70)
print(f"PUBLIC INFERENCE URL: {tunnel.public_url}")
print(f"Interactive Docs:     {tunnel.public_url}/docs")
print("=" * 70)

# NOTE: Free ngrok URLs change on every restart.
# Callers must send the header 'ngrok-skip-browser-warning: true' with every request.

# 4. Start Uvicorn in background thread
from xray_api import app

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
```

---

## 5. Preprocessing Parity Check

To prove mathematically that the API preprocessing matches the notebook's training data loader (`load_image`), run `parity_check.py` on real images:

```bash
# In Colab or local environment with TensorFlow:
python xray/parity_check.py /path/to/test/image1.png /path/to/test/image2.png
```

The script:
1. Runs the notebook's exact `load_image` TensorFlow pipeline.
2. Runs the API's `preprocess_image` function from `preprocess.py`.
3. Asserts the maximum absolute difference between both tensors is $< 10^{-6}$ ($0.000000$).
4. Evaluates both tensors through `best_finetuned.keras` and confirms identical softmax distributions.

---

## 6. How to Swap in a Retrained Model

To deploy a newly trained or fine-tuned model checkpoint:

1. **Format Requirements**:
   - Model must accept input tensor of shape `(batch, 224, 224, 3)` with `float32` pixel values $[0.0, 255.0]$ (EfficientNetB0 rescales internally).
   - Output must be a 5-class softmax tensor corresponding to indices $0 \dots 4$ (`KL0`, `KL1`, `KL2`, `KL3`, `KL4`).
2. **Replacing Artifacts**:
   - **Option A**: Overwrite `xray/model/best_finetuned.keras` with your new `.keras` file.
   - **Option B**: Point the service to any location using the environment variable:
     ```bash
     export XRAY_MODEL_PATH="/path/to/new_model.keras"
     ```
3. **Threshold Calibration**:
   - `OA_THRESHOLD` in `config.py` defaults to `0.50`.
   - If adjusting the threshold, **tune strictly on the validation split**, never on the test set.
4. **Restart**:
   - The service will load the new weights on startup, execute the blank image warm-up pass, and report `model_loaded: true` in `/health`.
