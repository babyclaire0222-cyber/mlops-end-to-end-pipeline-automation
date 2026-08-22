# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and buffering stdout/stderr,
# which keeps container logs flowing in real time.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so this layer is cached unless requirements change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project.
COPY . .

# Create writable directories the pipeline expects at runtime.
RUN mkdir -p data/raw data/processed artifacts/models logs mlruns

# Run as a non-root user for security.
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["python", "main.py"]
CMD ["--run-all"]
