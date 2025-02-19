# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.13-full \
    git \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN python3.13 -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Entrypoint
ENTRYPOINT ["python", "runner.py"]