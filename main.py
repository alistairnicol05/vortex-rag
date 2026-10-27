"""One Cloud Run service: your existing RAG API (/ask, /healthz) plus the ADK agent (API and chat UI)."""
import os

from google.adk.cli.fast_api import get_fast_api_app

from rag.app import ask, healthz, lifespan  # your existing endpoints and index-loading startup

# The chat UI is reached through `gcloud run services proxy`, so the browser's Origin is localhost while the
# request arrives at the run.app host. ADK blocks that mismatch as cross-site unless the origin is listed here.
# Safe because the service is private: only callers with Cloud Run IAM access get through the proxy.
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080").split(",")

app = get_fast_api_app(agents_dir="agents", web=True, lifespan=lifespan, allow_origins=ALLOW_ORIGINS)
app.add_api_route("/ask", ask, methods=["POST"])
app.add_api_route("/healthz", healthz, methods=["GET"])
