"""POST /api/chat — send a message to a specialist and stream the response."""
from __future__ import annotations

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.api.routes._shared import app, log
from backend.agents.specialist_agent import build_agent
from backend.agents.router_agent import route_query
from backend.config import load_registry


class ChatRequest(BaseModel):
    specialist_id: str | None = None   # if None, auto-route
    history: list[dict] = []           # [{"role": "user"/"assistant", "content": "..."}]
    message: str
    max_tokens: int = 2048
    temperature: float = 0.3


@app.post("/api/chat")
async def chat(req: ChatRequest):
    # Determine which specialist to use
    registry = load_registry()
    available = [
        s["id"] for s in registry.get("specialists", [])
        if s["status"] in ("ready", "serving")
    ]

    specialist_id = req.specialist_id
    if not specialist_id:
        if not available:
            raise HTTPException(503, "No specialist models are currently serving.")
        specialist_id = route_query(req.message, available)
        log(f"[chat] Auto-routed to '{specialist_id}' for query: {req.message[:80]}")
    elif specialist_id not in [s["id"] for s in registry.get("specialists", [])]:
        raise HTTPException(404, f"Specialist '{specialist_id}' not found in registry.")

    agent = build_agent(specialist_id)
    if agent is None:
        raise HTTPException(503, f"Specialist '{specialist_id}' model is not currently serving.")

    if not await agent.is_healthy():
        raise HTTPException(503, f"vLLM endpoint for '{specialist_id}' is not responding.")

    log(f"[chat] specialist={specialist_id} tokens={req.max_tokens} msg={req.message[:80]}")

    async def token_stream():
        yield f"data: {{\"specialist\": \"{specialist_id}\", \"type\": \"start\"}}\n\n"
        async for chunk in agent.chat(
            history=req.history,
            user_message=req.message,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            stream=True,
        ):
            # Escape for SSE
            safe = chunk.replace("\n", "\\n").replace('"', '\\"')
            yield f'data: {{"type": "token", "content": "{safe}"}}\n\n'
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(token_stream(), media_type="text/event-stream")
