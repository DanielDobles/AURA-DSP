# AARS Pipeline — ROCm MI300X Container (Purified DSP version)
FROM rocm/pytorch:rocm6.2.2_ubuntu22.04_py3.10_pytorch_release_2.3.0

# Use specific conda python path
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
RUN pip install --no-cache-dir -r requirements.txt

# Copy pipeline code
COPY . .

# Set environment variables for ROCm / MI300X
ENV HSA_OVERRIDE_GFX_VERSION=9.4.2
ENV HIP_VISIBLE_DEVICES=0
ENV PYTHONPATH=/app

CMD ["python", "main.py"]
