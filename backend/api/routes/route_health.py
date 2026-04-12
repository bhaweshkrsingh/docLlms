"""GET /api/health — check vLLM endpoints and system status."""
from fastapi.responses import JSONResponse
from backend.api.routes._shared import app, log
from backend.config import load_registry


@app.get("/api/health")
async def health():
    import httpx
    registry = load_registry()
    specialists_status = []

    for s in registry.get("specialists", []):
        sid = s["id"]
        port = s.get("vllm_port", 8101)
        status = s.get("status", "unknown")
        vllm_alive = False

        if status in ("ready", "serving"):
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(f"http://localhost:{port}/v1/models")
                    vllm_alive = resp.status_code == 200
            except Exception:
                vllm_alive = False

        specialists_status.append({
            "id": sid,
            "name": s["name"],
            "specialty": s["specialty"],
            "status": status,
            "vllm_port": port,
            "vllm_alive": vllm_alive,
        })
        log(f"[health] {sid}: status={status} vllm_alive={vllm_alive}")

    return JSONResponse({
        "service": "DocLlms",
        "specialists": specialists_status,
    })
