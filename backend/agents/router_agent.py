"""
Router agent — analyses the user query and routes it to the best specialist.

Strategy:
  1. Keyword fast-path (zero-cost, works offline)
  2. If ambiguous, ask the default specialist's LLM to classify

This keeps routing instant for clear queries (e.g. "my baby has a fever")
and smart for complex cross-specialty questions.
"""
from __future__ import annotations

import re

# Keyword maps: specialty_id → list of trigger patterns
_ROUTING_RULES: list[tuple[str, str]] = [
    ("pediatrician", r"\b(child|infant|baby|newborn|toddler|pediatr|neonat|adolescen|vaccine|vaccination|growth chart|febrile|kawasaki|congenital|developmental)\b"),
    ("oncologist",   r"\b(cancer|tumor|chemo|chemotherapy|oncol|malignant|metastas|lymphoma|leukemia|carcinoma|biopsy)\b"),
    ("cardiologist", r"\b(heart|cardiac|arrhythmia|ecg|ekg|myocardial|coronary|angina|pacemaker|heart failure|atrial|ventricular)\b"),
    ("neurologist",  r"\b(stroke|seizure|epilep|multiple sclerosis|parkinson|alzheimer|dementia|migraine|neuropath|brain)\b"),
    ("obgyn",        r"\b(pregnancy|prenatal|labor|delivery|obstetric|gynecol|menstrual|uterus|ovarian|cervical|postpartum|eclampsia)\b"),
]


def route_query(query: str, available_specialists: list[str]) -> str:
    """
    Return the specialist_id that best matches the query.
    Falls back to the first available specialist if no match.
    """
    q_lower = query.lower()
    scores: dict[str, int] = {}

    for specialist_id, pattern in _ROUTING_RULES:
        if specialist_id not in available_specialists:
            continue
        hits = len(re.findall(pattern, q_lower))
        if hits > 0:
            scores[specialist_id] = scores.get(specialist_id, 0) + hits

    if scores:
        return max(scores, key=lambda k: scores[k])

    # Default to first available
    return available_specialists[0] if available_specialists else "pediatrician"
