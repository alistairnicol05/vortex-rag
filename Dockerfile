FROM python:3.12-slim

# HYBRID=1: hybrid search (BM25 + RRF) is the configuration chosen in the evaluation; the code default is off.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src HYBRID=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code, the ADK agent, and the prebuilt index. No PDFs and no API key in the image.
COPY src ./src
COPY agents ./agents
COPY main.py .
COPY data/index ./data/index

# Cloud Run injects PORT (8080). The API keys arrive as secret env vars at runtime.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
