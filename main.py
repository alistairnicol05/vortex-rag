"""One Cloud Run service: your existing RAG API (/ask, /healthz) plus the ADK agent (API and chat UI)."""
from google.adk.cli.fast_api import get_fast_api_app

from rag.app import ask, healthz, lifespan  # your existing endpoints and index-loading startup

app = get_fast_api_app(agents_dir="agents", web=True, lifespan=lifespan)
app.add_api_route("/ask", ask, methods=["POST"])
app.add_api_route("/healthz", healthz, methods=["GET"])
