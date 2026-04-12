"""Central configuration — reads .env and models/registry.yaml."""
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

ROOT_DIR = Path(__file__).parent.parent
REGISTRY_PATH = ROOT_DIR / "models" / "registry.yaml"

# --- Server ports ---
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "7910"))
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7920"))
MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", "7930"))
MCP_ENDPOINT = os.getenv("MCP_ENDPOINT", "/mcp")

# --- vLLM ---
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8101")
DEFAULT_SPECIALIST = os.getenv("DEFAULT_SPECIALIST", "pediatrician")

# --- Logging ---
LOG_DIR = Path(os.getenv("LOG_DIR", str(ROOT_DIR / "logs")))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "server.log"


def load_registry() -> dict:
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def get_specialist(specialist_id: str) -> dict | None:
    registry = load_registry()
    for s in registry.get("specialists", []):
        if s["id"] == specialist_id:
            return s
    return None


def get_ready_specialists() -> list[dict]:
    registry = load_registry()
    return [s for s in registry.get("specialists", []) if s["status"] in ("ready", "serving", "training")]
