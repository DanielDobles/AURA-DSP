# SOTA 2026 ROCm PyTorch Base Image (Ubuntu 24.04 + ROCm 7.2)
FROM rocm/pytorch:latest

# Install critical Audio/Video System Dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ffmpeg \
    libsndfile1 \
    sox \
    libsox-fmt-all \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install CrewAI + PyTorch dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --no-deps git+https://github.com/facebookresearch/demucs#egg=demucs

# Pre-fetch Demucs models to avoid timeouts during execution
RUN python -c "from demucs.api import Separator; Separator(model='htdemucs')" || true

# Copy the AURA-DSP codebase
COPY . .

# Set environment variables for ROCm / MI300X
ENV HSA_OVERRIDE_GFX_VERSION=9.4.2
ENV HIP_VISIBLE_DEVICES=0
ENV PYTHONPATH=/app

CMD ["python", "main.py"]
