FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libxcb1 \
    libxkbcommon0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONUNBUFFERED=1
# Only block timm's fused attention check — do NOT set HF_DATASETS_OFFLINE or
# TRANSFORMERS_OFFLINE here, as that would block hf_hub_download at startup.
ENV TIMM_FUSED_ATTN=0

RUN mkdir -p /app/.cache/huggingface

# Small packages first
RUN pip install --no-cache-dir \
    Flask==3.0.0 \
    Werkzeug==3.0.1 \
    Pillow==10.2.0 \
    numpy==1.26.4 \
    opencv-python-headless==4.9.0.80 \
    gunicorn==21.2.0 \
    huggingface-hub==0.20.3 \
    timm==0.9.12

# PyTorch CPU — large download, separate layer with retries
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --retries 3 \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu

COPY . .

RUN mkdir -p uploads templates

EXPOSE 7860

# timeout=300 covers model download from HuggingFace on first startup
CMD ["gunicorn", "app:app", \
     "--bind", "0.0.0.0:7860", \
     "--workers=1", \
     "--timeout=300", \
     "--log-level=info"]
