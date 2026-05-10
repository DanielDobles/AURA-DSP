# AARS Pipeline — ROCm MI300X Container
FROM rocm/pytorch:rocm6.2.2_ubuntu22.04_py3.10_pytorch_release_2.3.0

# Use specific conda python path for all operations
ENV PATH /opt/conda/envs/py_3.10/bin:$PATH

# System dependencies for audio and DSP
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python stack
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies - we use --no-deps for audio models to protect ROCm torch
RUN pip install --no-cache-dir crewai[litellm] langchain-openai librosa soundfile numpy pydantic scipy ffmpeg-python requests scikit-learn loguru packaging appdirs
RUN pip install --no-cache-dir --no-deps deepfilternet deepfilterlib vocos

# INSTALL MISSING TORCHAUDIO (ROCm version) - Match torch 2.3.0 exactly to avoid ABI mismatch
RUN pip install --no-cache-dir --no-deps torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/rocm6.0

# Copy pipeline code
COPY . .

# Set environment variables for ROCm / MI300X
ENV HSA_OVERRIDE_GFX_VERSION=9.4.2
ENV HIP_VISIBLE_DEVICES=0
ENV MIOPEN_DEBUG=0
ENV PYTHONPATH=/app

CMD ["python", "main.py"]
