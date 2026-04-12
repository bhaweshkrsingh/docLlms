"""Concrete specialist agents, one per trained model."""
from __future__ import annotations

from backend.agents.base_agent import BaseSpecialistAgent
from backend.llm.vllm_client import VLLMClient
from backend.llm.model_registry import get_client, get_system_prompt


def build_agent(specialist_id: str) -> BaseSpecialistAgent | None:
    """
    Factory: instantiate the right agent for a given specialist_id.
    Returns None if the vLLM endpoint isn't reachable.
    """
    client = get_client(specialist_id)
    if client is None:
        return None

    agent = BaseSpecialistAgent(client)
    agent.specialist_id = specialist_id
    agent.system_prompt = get_system_prompt(specialist_id)

    # Override names for known specialties
    names = {
        "pediatrician": "PediatricianGemma",
        "oncologist":   "OncologyGemma",
        "cardiologist": "CardiologyGemma",
        "neurologist":  "NeurologyGemma",
        "obgyn":        "ObGynGemma",
    }
    agent.specialist_name = names.get(specialist_id, specialist_id.capitalize() + "Gemma")
    return agent
