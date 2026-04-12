"""Runtime model registry — manages live VLLMClient instances per specialist."""
from __future__ import annotations

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))

from backend.config import load_registry, LOG_FILE
from backend.llm.vllm_client import VLLMClient

_clients: dict[str, VLLMClient] = {}


def _log(msg: str):
    print(msg)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def get_client(specialist_id: str) -> VLLMClient | None:
    """Return a VLLMClient for the given specialist, or None if not serving."""
    if specialist_id in _clients:
        return _clients[specialist_id]

    registry = load_registry()
    for s in registry.get("specialists", []):
        if s["id"] == specialist_id:
            port = s.get("vllm_port", 8101)
            base_url = f"http://localhost:{port}/v1"
            client = VLLMClient(base_url=base_url)
            _clients[specialist_id] = client
            _log(f"[registry] Registered client for '{specialist_id}' at {base_url}")
            return client
    return None


def get_system_prompt(specialist_id: str) -> str:
    registry = load_registry()
    for s in registry.get("specialists", []):
        if s["id"] == specialist_id:
            return s.get("system_prompt", "You are a helpful medical AI assistant.")
    return "You are a helpful medical AI assistant."


async def health_check_all() -> dict[str, bool]:
    registry = load_registry()
    results = {}
    for s in registry.get("specialists", []):
        sid = s["id"]
        client = get_client(sid)
        if client:
            results[sid] = await client.is_healthy()
        else:
            results[sid] = False
    return results
