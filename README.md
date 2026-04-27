"# KGroup_DiaVision" 
<div align="center">

<img src="https://img.shields.io/badge/K%20Group-DiaVision-22C55E?style=for-the-badge" alt="DiaVision"/>

# DiaVision — Diabetic Retinopathy Screening

**AI-powered fundus image analysis for early DR detection**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![EfficientNet](https://img.shields.io/badge/Model-EfficientNet--B4-4F46E5?style=flat)](https://arxiv.org/abs/1905.11946)
[![License](https://img.shields.io/badge/License-MIT-16A34A?style=flat)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=flat)](https://huggingface.co/spaces)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)

</div>

---

## About This Project

DiaVision is my final year project — a web-based clinical screening tool that uses deep learning to detect signs of **Diabetic Retinopathy (DR)** from retinal fundus photographs. I built this together with my project group (**K Group**) to assist ophthalmologists and healthcare workers with early DR detection, which is the most effective way to prevent vision loss in diabetic patients.

The tool runs an **EfficientNet-B4** model with **Grad-CAM** visualisation so doctors can see exactly which part of the retina triggered the prediction — not just a label, but an explanation.

> ⚠️ **Clinical Disclaimer:** DiaVision is a **decision support tool only** and does not constitute a medical diagnosis. All results must be confirmed by a qualified ophthalmologist.

---

## Features

- **DR detection** from fundus photographs using EfficientNet-B4
- **Grad-CAM heatmap** — highlights the exact retinal regions that influenced the result
- **Three-level output** — DR Detected / Borderline / No DR (threshold: 0.35)
- **Fundus image validator** — rejects non-retinal images (photos, selfies, screenshots) before running the model, with a clear error message
- **Patient record system** — auto-generated patient IDs, name, age, gender, eye selection
- **PDF report export** — printable screening report with Grad-CAM images included
- **Eye Health Info page** — DR stages, anatomy, nutrition, common eye diseases, and clinical resources
- **Dockerised** — deployed on HuggingFace Spaces

---

## How It Works

```
Upload Fundus Image
       │
       ▼
Fundus Validator ──✗──▶ "Not a retinal image" popup
       │ ✓
       ▼
Preprocessing
  ├── Crop black borders
  ├── Resize → 300×300 px
  ├── CLAHE contrast enhancement
  └── Ben Graham normalisation
       │
       ▼
EfficientNet-B4 Inference
       │
       ▼
Grad-CAM Generation
       │
       ▼
DR Probability Score
  ├── ≥ 0.35  →  DR Detected
  ├── 0.20–0.35  →  Borderline
  └── < 0.20  →  No DR
```

---

## Project Structure

```
diavision/
│
├── app.py                      # Flask backend — routes & prediction endpoint
├── model.py                    # EfficientNet-B4, Grad-CAM, fundus validator
├── Dockerfile                  # Container for HuggingFace Spaces
├── requirements.txt            # Python dependencies
│
└── templates/
    ├── dr_detection_gui.html   # Main screening UI
    └── eye_info.html           # Eye health information portal
```

---

## Model Details

| Property | Value |
|---|---|
| Architecture | EfficientNet-B4 |
| Input size | 300 × 300 px |
| DR threshold | 0.35 |
| Borderline range | 0.20 – 0.35 |
| Inference device | CPU |
| Explainability | Grad-CAM |
| Weights hosted on | HuggingFace Hub (`Salonideshmukh/dr-detection-model`) |

---

## Fundus Image Validation

One important feature I added is a **pre-inference validator** that checks if the uploaded image is actually a fundus photograph before running the model. This prevents the model from giving meaningless predictions on random photos.

The validator runs 5 OpenCV-based checks:

| Check | Logic |
|---|---|
| Aspect ratio | Must be roughly square (0.5 – 2.0) |
| Dark border | ≥5% dark pixels — fundus cameras always produce a circular image on black |
| Bright region size | Retinal disc must occupy 30–90% of the image |
| Circularity | The bright region must be approximately circular |
| Colour profile | Retinal tissue is always red-dominant (R > G > B) |

If any check fails, a popup tells the user exactly why the image was rejected.

---

## Running Locally

**Requirements:** Python 3.11+

```bash
# Clone the repo
git clone https://github.com/AnujKanchan/diavision.git
cd diavision

# Install dependencies
pip install -r requirements.txt

# Run
python app.py
```

Open `http://localhost:7860` in your browser.

> First run will download model weights (~100 MB) from HuggingFace Hub automatically.

### Docker

```bash
docker build -t diavision .
docker run -p 7860:7860 diavision
```

---

## Deploying to HuggingFace Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) → select **Docker**
2. Upload files in this structure:

```
/
├── app.py
├── model.py
├── Dockerfile
├── requirements.txt
└── templates/
    ├── dr_detection_gui.html
    └── eye_info.html
```

3. HuggingFace builds automatically. First startup takes ~60 s to download weights.

---

## API

| Method | Route | Description |
|---|---|---|
| GET | `/` | Main screening interface |
| GET | `/info` | Eye health information page |
| POST | `/predict` | Run DR prediction on uploaded image |
| GET | `/health` | Server and model status |

**POST `/predict`** — `multipart/form-data`, field: `image` (JPEG/PNG, max 16 MB)

```json
// Success response
{
  "label": "DR",
  "confidence": 0.94,
  "original": "<base64 PNG>",
  "heatmap":  "<base64 PNG>",
  "overlay":  "<base64 PNG>"
}

// Invalid image (HTTP 422)
{
  "success": false,
  "error": "Invalid image: No dark border detected. Please upload a valid fundus camera retinal photograph."
}
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| Flask | 3.0.0 | Web framework |
| Werkzeug | 3.0.1 | WSGI utilities |
| Pillow | 10.2.0 | Image loading |
| NumPy | 1.26.4 | Array operations |
| OpenCV headless | 4.9.0.80 | Preprocessing & validation |
| PyTorch | 2.2.2 CPU | Model inference |
| timm | 0.9.12 | EfficientNet-B4 |
| huggingface-hub | 0.20.3 | Weight download |
| gunicorn | 21.2.0 | Production server |

---

## Team — K Group

| Name | Role |
|---|---|
| **Anuj Ganesh Kanchan** | Team Lead |
| Viraj Sukhdev Kalbhor 
| Yash Bandu Kalbhor
| Sandesh Sachin Kalbhor 

---

## References

- [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946) — Tan & Le, 2019
- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391) — Selvaraju et al., 2017
- [Ben Graham — Diabetic Retinopathy Preprocessing](https://www.kaggle.com/competitions/diabetic-retinopathy-detection)
- [NIH National Eye Institute — Diabetic Retinopathy](https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/diabetic-retinopathy)
- [WHO — Blindness and Visual Impairment](https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built by **Anuj Ganesh Kanchan** · K Group · Final Year Project

</div>
