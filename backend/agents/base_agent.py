"""Base class for all specialist agents."""
from __future__ import annotations

from typing import AsyncIterator

from backend.llm.vllm_client import VLLMClient


class BaseSpecialistAgent:
    """
    Wraps a VLLMClient with a specialist system prompt and conversation history.
    Each specialist (Pediatrician, Oncologist, etc.) subclasses this.
    """

    specialist_id: str = "base"
    specialist_name: str = "Medical Assistant"
    system_prompt: str = "You are a helpful medical AI assistant."

    def __init__(self, client: VLLMClient):
        self.client = client

    async def chat(
        self,
        history: list[dict],
        user_message: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """Send a message and stream back the response."""
        messages = list(history) + [{"role": "user", "content": user_message}]
        async for chunk in self.client.chat(
            messages=messages,
            system_prompt=self.system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        ):
            yield chunk

    async def is_healthy(self) -> bool:
        return await self.client.is_healthy()
