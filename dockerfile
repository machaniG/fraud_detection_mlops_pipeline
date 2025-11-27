# Production Dockerfile for AWS Deployment
# Optimized for smaller size and faster builds

FROM python:3.10-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY scripts/ ./scripts/
COPY src/ ./src/
COPY tests/ ./tests/

# Copy data file (for initial training)
# In production, this would be fetched from S3
COPY transactions.csv ./transactions.csv

# Create directories for artifacts
RUN mkdir -p /app/models /app/mlruns /app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose FastAPI port
EXPOSE 8000

# Run the complete API
CMD ["python", "scripts/api.py"]