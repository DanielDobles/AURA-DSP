# AARS Pipeline — ROCm MI300X Container
FROM rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.9.1

# System deps
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pipeline code
COPY . .

# Environment
ENV HSA_OVERRIDE_GFX_VERSION=9.4.2
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
