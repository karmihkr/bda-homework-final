FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY telemetry_failure_modeling.py telemetry_hw.py config.json README.md architecture.md refactoring_report.md ./
COPY generator ./generator

ENTRYPOINT ["python", "telemetry_failure_modeling.py"]
CMD ["--config", "/app/config.json", "--dataset", "/app/dataset", "--output", "/app/processed/telemetry_failure"]
