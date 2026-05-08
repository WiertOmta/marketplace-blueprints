"""RAG Assistant Chat UI — a lightweight FastAPI app that proxies chat
messages to a DigitalOcean managed GenAI agent and serves a simple web
interface.

The app self-discovers the agent's deployment URL and API key at startup
using the DO API.

Environment variables (injected by terraform via App Platform):
    AGENT_UUID               — UUID of the managed agent
    DO_API_TOKEN             — DigitalOcean API token
    AGENT_NAME               — Display name of the agent (optional)
    CHAT_AUTH_USERNAME       — Username for HTTP Basic Auth (optional; auth disabled if unset)
    CHAT_AUTH_PASSWORD       — Password for HTTP Basic Auth (optional; auth disabled if unset)
    SPACES_BUCKET            — DO Spaces bucket holding the KB documents (optional)
    SPACES_REGION            — Spaces region, e.g. ams3 / fra1 / nyc3 (optional)
    SPACES_ACCESS_KEY_ID     — Spaces access key (optional)
    SPACES_SECRET_ACCESS_KEY — Spaces secret key (optional)
    SPACES_PRESIGN_TTL       — Presigned URL TTL in seconds (default 900)
"""

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

SPACES_BUCKET = os.environ.get("SPACES_BUCKET", "")
SPACES_REGION = os.environ.get("SPACES_REGION", "")
SPACES_ACCESS_KEY_ID = os.environ.get("SPACES_ACCESS_KEY_ID", "")
SPACES_SECRET_ACCESS_KEY = os.environ.get("SPACES_SECRET_ACCESS_KEY", "")
SPACES_PRESIGN_TTL = int(os.environ.get("SPACES_PRESIGN_TTL", "900"))
SPACES_ENABLED = bool(
    SPACES_BUCKET and SPACES_REGION and SPACES_ACCESS_KEY_ID and SPACES_SECRET_ACCESS_KEY
)

s3_client = None
if SPACES_ENABLED:
    import boto3
    from botocore.client import Config as BotoConfig

    s3_client = boto3.client(
        "s3",
        endpoint_url=f"https://{SPACES_REGION}.digitaloceanspaces.com",
        aws_access_key_id=SPACES_ACCESS_KEY_ID,
        aws_secret_access_key=SPACES_SECRET_ACCESS_KEY,
        region_name=SPACES_REGION,
        config=BotoConfig(signature_version="s3v4"),
    )
    logger.info(
        "Spaces presigning enabled (bucket=%s region=%s ttl=%ds)",
        SPACES_BUCKET, SPACES_REGION, SPACES_PRESIGN_TTL,
    )
else:
    logger.warning("SPACES_* env vars not all set — source download links will be omitted")


def _presign_spaces_url(key: str) -> str | None:
    """Generate a short-lived presigned GET URL for a Spaces object.

    The chunk's `filename` field typically prepends the bucket name as a path
    segment (e.g. 'uvsv-chatbot/foo/bar.docx' for bucket 'uvsv-chatbot'); strip
    that segment if present so the S3 key is the actual object key.
    """
    if not s3_client or not key:
        return None
    if key.startswith(SPACES_BUCKET + "/"):
        key = key[len(SPACES_BUCKET) + 1:]
    try:
        return s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": SPACES_BUCKET, "Key": key},
            ExpiresIn=SPACES_PRESIGN_TTL,
        )
    except Exception as e:
        logger.warning("Failed to presign URL for key %r: %s", key, e)
        return None

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

    # Extract the response text from OpenAI-compatible format.
    content = ""
    if "choices" in data and len(data["choices"]) > 0:
        content = data["choices"][0].get("message", {}).get("content", "")
    elif "detail" in data:
        content = f"Error: {data['detail']}"

    sources = _extract_sources(data)

    return JSONResponse(content={"content": content, "usage": data.get("usage"), "sources": sources})


def _extract_sources(data: dict) -> list[dict]:
    """Pull retrieval entries out of the agent response and presign download URLs.

    The agent returns chunks under data.retrieval.retrieved_data. Each item carries
    a `filename` (Spaces object key, possibly bucket-prefixed) and a `metadata.item_name`
    suitable for display.
    """
    retrieval = data.get("retrieval") or {}
    items = retrieval.get("retrieved_data") if isinstance(retrieval, dict) else None
    if not isinstance(items, list):
        return []

    sources: list[dict] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        filename = item.get("filename") or ""
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        display_name = (
            metadata.get("item_name")
            or (filename.rsplit("/", 1)[-1] if filename else None)
            or f"Source {idx}"
        )
        sources.append({
            "index": idx,
            "name": display_name,
            "url": _presign_spaces_url(filename) if filename else None,
        })
    return sources
