"""RAG Assistant Chat UI — a lightweight FastAPI app that proxies chat
messages to a DigitalOcean managed GenAI agent and serves a simple web
interface.

The app self-discovers the agent's deployment URL and API key at startup
using the DO API.

Environment variables (injected by terraform via App Platform):
    AGENT_UUID         — UUID of the managed agent
    DO_API_TOKEN       — DigitalOcean API token
    AGENT_NAME         — Display name of the agent (optional)
    CHAT_AUTH_USERNAME — Username for HTTP Basic Auth (optional; auth disabled if unset)
    CHAT_AUTH_PASSWORD — Password for HTTP Basic Auth (optional; auth disabled if unset)
"""

import json
import logging
import os
import secrets
import sys
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("chat-ui")

app = FastAPI(title="RAG Assistant")

AGENT_UUID = os.environ["AGENT_UUID"]
DO_API_TOKEN = os.environ["DO_API_TOKEN"]
AGENT_NAME = os.environ.get("AGENT_NAME", "RAG Assistant")
DO_API_BASE = os.environ.get("DO_API_BASE", "https://api.digitalocean.com")
CHAT_AUTH_USERNAME = os.environ.get("CHAT_AUTH_USERNAME", "")
CHAT_AUTH_PASSWORD = os.environ.get("CHAT_AUTH_PASSWORD", "")
AUTH_ENABLED = bool(CHAT_AUTH_USERNAME and CHAT_AUTH_PASSWORD)
if not AUTH_ENABLED:
    logger.warning("CHAT_AUTH_USERNAME/CHAT_AUTH_PASSWORD not set — chat UI is unauthenticated")

security = HTTPBasic(auto_error=False)


def require_auth(credentials: HTTPBasicCredentials | None = Depends(security)) -> None:
    if not AUTH_ENABLED:
        return
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": 'Basic realm="RAG Assistant"'},
    )
    if credentials is None:
        raise unauthorized
    # secrets.compare_digest avoids leaking timing information.
    user_ok = secrets.compare_digest(credentials.username.encode("utf-8"), CHAT_AUTH_USERNAME.encode("utf-8"))
    pass_ok = secrets.compare_digest(credentials.password.encode("utf-8"), CHAT_AUTH_PASSWORD.encode("utf-8"))
    if not (user_ok and pass_ok):
        raise unauthorized

# Populated at startup.
AGENT_ENDPOINT = None
AGENT_API_KEY = None

# Serve the static HTML chat page.
INDEX_HTML = (Path(__file__).parent / "static" / "index.html").read_text()


def _do_headers():
    return {"Authorization": f"Bearer {DO_API_TOKEN}", "Content-Type": "application/json"}


def _discover_agent():
    """Fetch agent details from the DO API to get the deployment URL and API key."""
    global AGENT_ENDPOINT, AGENT_API_KEY

    logger.info("Discovering agent %s ...", AGENT_UUID)
    with httpx.Client(timeout=30.0) as client:
        # Get agent details.
        resp = client.get(f"{DO_API_BASE}/v2/gen-ai/agents/{AGENT_UUID}", headers=_do_headers())
        resp.raise_for_status()
        agent = resp.json()["agent"]

        # Extract deployment URL.
        deployment = agent.get("deployment", {})
        deploy_url = deployment.get("url")
        if deploy_url:
            AGENT_ENDPOINT = f"{deploy_url}/api/v1/chat/completions"
            logger.info("Agent endpoint: %s", AGENT_ENDPOINT)
        else:
            logger.error("Agent has no deployment URL. Status: %s", deployment.get("status"))
            raise RuntimeError("Agent deployment URL not available")

        # Create an API key for agent authentication.
        # The auto-generated api_keys[].api_key is a chatbot identifier, not a secret key.
        # We need to create a real API key via the API.
        logger.info("Creating agent API key...")
        key_resp = client.post(
            f"{DO_API_BASE}/v2/gen-ai/agents/{AGENT_UUID}/api_keys",
            headers=_do_headers(),
            json={"name": "chat-ui"},
        )
        key_resp.raise_for_status()
        AGENT_API_KEY = key_resp.json()["api_key_info"]["secret_key"]
        logger.info("Agent API key created")


@app.on_event("startup")
async def startup_event():
    _discover_agent()


@app.get("/", response_class=HTMLResponse)
async def index(_: None = Depends(require_auth)):
    """Serve the chat UI."""
    return INDEX_HTML.replace("{{AGENT_NAME}}", AGENT_NAME)


@app.get("/health")
async def health():
    return {"status": "ok", "agent_ready": AGENT_ENDPOINT is not None}


@app.post("/api/chat")
async def chat(request: Request, _: None = Depends(require_auth)):
    """Proxy a chat message to the managed agent and return the response."""
    if not AGENT_ENDPOINT or not AGENT_API_KEY:
        return JSONResponse(status_code=503, content={"error": "Agent not ready"})

    body = await request.json()
    message = body.get("message", "")
    history = body.get("history", [])

    # Build OpenAI-compatible messages array.
    messages = []
    for h in history:
        messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
    messages.append({"role": "user", "content": message})

    headers = {
        "Authorization": f"Bearer {AGENT_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {"messages": messages, "include_retrieval_info": True}
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(AGENT_ENDPOINT, json=payload, headers=headers)

    try:
        data = resp.json()
    except Exception:
        return JSONResponse(status_code=resp.status_code, content={"error": resp.text})

    _log_response_shape(data)

    # Extract the response text from OpenAI-compatible format.
    content = ""
    if "choices" in data and len(data["choices"]) > 0:
        content = data["choices"][0].get("message", {}).get("content", "")
    elif "detail" in data:
        content = f"Error: {data['detail']}"

    sources = _extract_sources(data)

    return JSONResponse(content={"content": content, "usage": data.get("usage"), "sources": sources})


def _log_response_shape(data: dict) -> None:
    """Log a small summary of the agent response so we can refine source extraction."""
    try:
        logger.info("Agent response top-level keys: %s", list(data.keys()))

        retrieval = data.get("retrieval") or {}
        if isinstance(retrieval, dict):
            logger.info("  retrieval keys: %s", list(retrieval.keys()))
            items = retrieval.get("retrieved_data") or []
            if isinstance(items, list):
                logger.info("  retrieved_data length: %d", len(items))
                for ix, item in enumerate(items[:3]):
                    if isinstance(item, dict):
                        logger.info("    [%d] keys: %s", ix, list(item.keys()))
                        for k, v in item.items():
                            sample = json.dumps(v, default=str)
                            logger.info("    [%d].%s: %s", ix, k, sample[:300])

        citations = data.get("citations")
        if citations is not None:
            logger.info("  citations: %s", json.dumps(citations, default=str)[:500])

        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message", {}) or {}
            logger.info("  choices[0].message keys: %s", list(msg.keys()))
    except Exception as e:
        logger.warning("Failed to log response shape: %s", e)


def _extract_sources(data: dict) -> list[dict]:
    """Pull retrieval/citation entries out of the agent response. The exact
    schema isn't pinned down, so try several common locations and field names."""
    candidates: list = []

    def find_in(obj):
        if not isinstance(obj, dict):
            return None
        for key in ("retrieval", "sources", "citations", "context"):
            val = obj.get(key)
            if isinstance(val, dict):
                for inner in ("retrieved_data", "documents", "sources", "citations", "items"):
                    if isinstance(val.get(inner), list) and val[inner]:
                        return val[inner]
            elif isinstance(val, list) and val:
                return val
        return None

    candidates = find_in(data) or []
    if not candidates:
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            candidates = find_in(choices[0].get("message") or {}) or []

    sources: list[dict] = []
    for idx, item in enumerate(candidates, start=1):
        if not isinstance(item, dict):
            continue
        name = (
            item.get("name")
            or item.get("document_name")
            or item.get("filename")
            or item.get("file_name")
            or item.get("title")
            or item.get("source")
            or f"Source {idx}"
        )
        url = (
            item.get("url")
            or item.get("download_url")
            or item.get("source_url")
            or item.get("file_url")
            or item.get("link")
        )
        sources.append({"index": idx, "name": name, "url": url})
    return sources
