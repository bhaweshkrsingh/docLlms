"""OpenAI-compatible client for vLLM endpoints.

vLLM serves the fine-tuned models with an OpenAI-compatible API, so we use
the standard openai library. Each specialist runs on its own port.

Serving command (non-docker):
  vllm serve <model_path> --host 0.0.0.0 --port <port> \
    --max-model-len 8192 --gpu-memory-utilization 0.85

Docker version (for reference, not used here):
  docker run -d --name gemma4-31b --gpus all -p 8000:8000 --ipc=host \
    -v /home/bkprity/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN=<token> \
    vllm/vllm-openai:gemma4-cu130 \
    --model nvidia/Gemma-4-31B-IT-NVFP4 --max-model-len 262144 \
    --gpu-memory-utilization 0.9067 \
    --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from openai import AsyncOpenAI


class VLLMClient:
    """Thin async wrapper around the vLLM OpenAI-compatible endpoint."""

    def __init__(self, base_url: str, model_name: str | None = None):
        # vLLM accepts any string as the api_key when auth is disabled
        self.client = AsyncOpenAI(base_url=base_url, api_key="NONE")
        self._base_url = base_url
        self._model_name = model_name  # if None, auto-detected from /v1/models

    async def get_model_name(self) -> str:
        if self._model_name:
            return self._model_name
        models = await self.client.models.list()
        self._model_name = models.data[0].id
        return self._model_name

    async def is_healthy(self) -> bool:
        try:
            await asyncio.wait_for(self.get_model_name(), timeout=5.0)
            return True
        except Exception:
            return False

    async def chat(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """Yield response text chunks (streaming) or the full response."""
        model = await self.get_model_name()

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        if stream:
            async with self.client.chat.completions.stream(
                model=model,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            ) as stream_ctx:
                async for chunk in stream_ctx:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        yield delta
        else:
            response = await self.client.chat.completions.create(
                model=model,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
            yield response.choices[0].message.content or ""
