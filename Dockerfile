# ABC Bank Segmentation - data pipeline container (Module 3)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PREFECT_HOME=/app/.prefect

WORKDIR /app

# 1. Dependencies first, so this layer is cached between code changes
COPY requirements-pipeline.txt .
RUN pip install --no-cache-dir -r requirements-pipeline.txt

# 2. Pipeline code and tests (no data: raw data is mounted at run time)
COPY src/ src/
COPY flows/ flows/
COPY tests/ tests/

# 3. Run as a non-root user
RUN useradd --create-home pipeline && chown -R pipeline /app
USER pipeline

# Data, outputs, logs and validation docs live outside the image
VOLUME ["/app/data", "/app/reports", "/app/logs", "/app/gx"]

# ANON_SALT must be passed with -e at run time; it is never stored in the image
CMD ["python", "-m", "flows.pipeline_flow"]
