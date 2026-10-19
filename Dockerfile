FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code + the prebuilt index (chunks.jsonl, embeddings.npy, meta.json). No PDFs, no API key in the image.
COPY src ./src
COPY data/index ./data/index

# Cloud Run injects PORT (8080). GEMINI_API_KEY arrives as a secret env var at runtime.
CMD ["sh", "-c", "uvicorn rag.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
