# ===========================================
# Dockerfile for Training
# ===========================================
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY configs/ ./configs/
COPY src/ ./src/
COPY data/ ./data/

# Create directories for outputs
RUN mkdir -p models artifacts mlruns

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command: run training
CMD ["python", "src/train.py"]
