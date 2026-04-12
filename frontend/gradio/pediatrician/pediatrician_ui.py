"""
PediatricianGemma — Gradio chat interface.

Talks directly to the vLLM endpoint (no backend API required to run the UI standalone).
When the backend is up, it can optionally use /api/chat for routing.

Launch:
  python frontend/gradio/pediatrician/launch.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7920"))
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8101/v1")

SYSTEM_PROMPT = """You are PediatricianGemma, an AI assistant fine-tuned on 50,000 pediatric
medical cases covering neonatal care, child development, vaccinations, congenital conditions,
pediatric emergencies, adolescent medicine, and pediatric pharmacology.

Provide detailed, clinically structured, and accurate answers grounded in standard pediatric
practice (Nelson's Textbook, AAP guidelines, ACIP recommendations). When discussing medications,
include weight-based dosing formulas. When discussing developmental concerns, reference appropriate
milestone tables.

Always remind users to consult a licensed pediatrician for actual clinical decisions."""

# ── Example questions — deep/specific, drawn from training data topics ──────

EXAMPLE_QUESTIONS = [
    # Well-child / developmental
    [
        "A 14-month-old boy has no words and doesn't point to objects. His mother says he doesn't "
        "consistently respond to his name. What developmental red flags are present, how do you "
        "evaluate him, and what are your differential diagnoses for this presentation?"
    ],
    [
        "Compare the expected developmental milestones for a healthy child at 6 months, 12 months, "
        "18 months, and 24 months across the four domains: gross motor, fine motor, language, "
        "and social/adaptive. What single red flag at each age should prompt immediate referral?"
    ],
    # Neonatal
    [
        "A 36-hour-old term neonate has a total serum bilirubin of 18 mg/dL. His birth weight was "
        "3.4 kg and he is exclusively breastfed. How do you assess the need for phototherapy versus "
        "exchange transfusion, and what are the risk factors that lower the intervention threshold?"
    ],
    [
        "An 8-week-old premature infant (32 weeks GA) in the NICU develops increasing abdominal "
        "distension, bloody stools, feeding intolerance, and temperature instability. X-ray shows "
        "pneumatosis intestinalis with portal venous gas. Outline the diagnosis, staging (Bell's "
        "criteria), and full management including surgical indications."
    ],
    # Emergencies
    [
        "A 2-year-old presents with a 3-hour history of sudden-onset inconsolable crying, drawing "
        "up his legs in paroxysms every 15–20 minutes, vomiting, and has now passed a red 'currant "
        "jelly' stool. What is the diagnosis, how do you confirm it, and what is your management "
        "algorithm including when to proceed to surgical intervention?"
    ],
    [
        "A 5-year-old boy has had 5 days of fever (≥38.5°C), bilateral non-purulent conjunctivitis, "
        "cracked lips, strawberry tongue, a polymorphous truncal rash, and swollen, erythematous "
        "hands and feet. ESR 95, CRP 14, echo shows mild dilation of the left anterior descending "
        "coronary artery. Confirm the diagnosis, explain the Kobayashi score, and detail the "
        "IVIG + aspirin protocol. What coronary artery z-score thresholds guide long-term "
        "anticoagulation decisions?"
    ],
    [
        "An 18-month-old is brought in after a 2-minute generalised tonic-clonic seizure with fever "
        "(39.4°C). She is now post-ictal but rousable. How do you distinguish a simple febrile "
        "seizure from a complex one? What is the recommended workup, and what do you tell the parents "
        "about recurrence risk, driving implications, and whether anti-epileptic therapy is indicated?"
    ],
    # Gastroenterology / Surgery
    [
        "A 6-week-old boy presents with 4 days of progressive projectile non-bilious vomiting after "
        "every feed. He is hungry immediately after vomiting. On exam you palpate an olive-shaped "
        "mass in the right upper quadrant. What is the diagnosis, what ultrasound findings confirm "
        "it, and describe the pre-operative electrolyte correction and the Ramstedt pyloromyotomy procedure."
    ],
    # Oncology / Haematology
    [
        "A 4-year-old girl presents with 6 weeks of progressive fatigue, pallor, and easy bruising. "
        "She has hepatosplenomegaly. CBC: WBC 85,000/µL (80% blasts), Hgb 6.2 g/dL, platelets "
        "18,000/µL. Describe your diagnostic approach including bone marrow biopsy, immunophenotyping, "
        "cytogenetics, and outline the Berlin-Frankfurt-Münster induction chemotherapy protocol for "
        "paediatric ALL."
    ],
    # Infectious disease
    [
        "An unvaccinated 3-year-old presents with 5 days of fever, coryza, conjunctivitis, and "
        "Koplik spots on the buccal mucosa followed by a maculopapular rash starting at the hairline "
        "and spreading cephalocaudally. Confirm the diagnosis and describe in detail: "
        "(1) the major complications including subacute sclerosing panencephalitis, "
        "(2) post-exposure prophylaxis for unvaccinated contacts, "
        "(3) vitamin A supplementation indications."
    ],
    [
        "A 3-month-old infant presents with 2 weeks of paroxysmal coughing episodes followed by "
        "an inspiratory 'whoop', post-tussive vomiting, and cyanosis during coughing fits. "
        "Lymphocyte count is 22,000/µL. What is the diagnosis, which organism and toxin are "
        "responsible, how do you confirm it microbiologically, and what are the treatment and "
        "household contact prophylaxis protocols?"
    ],
    # Vaccination
    [
        "Walk me through the complete ACIP vaccination schedule for a healthy term infant from "
        "birth through 18 months. For each visit, list the vaccines due, the route and site of "
        "administration, the key contraindications, and explain why the schedule is timed the way "
        "it is (maternal antibody waning, immunological maturity). Include catch-up guidance for "
        "a 12-month-old who received only the birth hepatitis B dose."
    ],
    # Nutrition / Growth
    [
        "A 6-month-old breastfed infant has dropped from the 25th to the 8th weight-for-age "
        "percentile since her 4-month visit. She is not yet on solids. Her mother reports "
        "frequent nursing (10×/day) and no vomiting. Physical exam shows mild hypotonia. "
        "Define failure to thrive, classify this as organic vs non-organic, outline your "
        "diagnostic workup, and detail the nutritional rehabilitation plan including timing "
        "of solid food introduction and supplementation."
    ],
    # Pharmacology
    [
        "A 20 kg, 6-year-old child with community-acquired pneumonia (lobar consolidation, "
        "fever 39°C, RR 35/min, SpO₂ 93% on room air) requires antibiotic therapy. "
        "Compare IV ampicillin vs oral amoxicillin vs azithromycin — indications, weight-based "
        "dosing calculations for this child, duration, and criteria for stepping down from IV "
        "to oral. When would you add coverage for atypical organisms?"
    ],
    # Endocrinology
    [
        "A 10-year-old girl has had 3 months of polyuria, polydipsia, and unintentional 4 kg "
        "weight loss. Random glucose is 340 mg/dL, pH 7.28, bicarbonate 14 mEq/L, glucose in "
        "urine, and moderate ketonuria. Distinguish DKA from HHS in paediatrics. Provide the "
        "step-by-step DKA management protocol including fluid resuscitation rates, insulin "
        "infusion, potassium replacement, and monitoring for cerebral oedema."
    ],
]

# ── Helper: call vLLM directly (streaming) ───────────────────────────────────

def _get_vllm_client() -> OpenAI | None:
    try:
        client = OpenAI(base_url=VLLM_BASE_URL, api_key="NONE")
        models = client.models.list()
        if models.data:
            return client
    except Exception:
        pass
    return None


def _model_status() -> tuple[str, str]:
    """Return (status_text, color_css_class)."""
    client = _get_vllm_client()
    if client:
        try:
            models = client.models.list()
            name = models.data[0].id if models.data else "unknown"
            return f"✅  Model online: {name}", "status-online"
        except Exception:
            pass
    return "⏳  Model not yet serving (training in progress — finishes ~Apr 17 2026)", "status-training"


def respond(message: str, history: list[list[str]]) -> str:
    """Gradio streaming chat function."""
    client = _get_vllm_client()
    if client is None:
        return (
            "⏳ **PediatricianGemma is still training** (~finishes Apr 17 2026).\n\n"
            "The model is currently being fine-tuned on 50,000 pediatric Q&A cases. "
            "Once training completes and the model is served via vLLM, you can ask "
            "any of the example questions and get deep, specialist-level answers.\n\n"
            "In the meantime, you can explore the example questions to see what this "
            "model will be able to answer."
        )

    try:
        models = client.models.list()
        model_id = models.data[0].id
    except Exception:
        return "Error: could not retrieve model name from vLLM."

    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    try:
        stream = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=2048,
            temperature=0.3,
            stream=True,
        )
        full_response = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                full_response += delta
                yield full_response
    except Exception as e:
        yield f"Error calling model: {e}"


# ── Gradio UI ────────────────────────────────────────────────────────────────

CSS = """
.status-online   { color: #16a34a; font-weight: 600; }
.status-training { color: #d97706; font-weight: 600; }
.model-badge     { background: #1e40af; color: white; padding: 4px 12px;
                   border-radius: 999px; font-size: 0.85em; font-weight: 600; }
.specialty-tag   { background: #f0fdf4; border: 1px solid #86efac; color: #166534;
                   padding: 2px 8px; border-radius: 4px; font-size: 0.8em; }
#chatbox         { height: 520px; }
.example-btn     { text-align: left !important; font-size: 0.82em !important; }
"""

DESCRIPTION = """
## PediatricianGemma
<span class="model-badge">Gemma-4-31B</span>&nbsp;
<span class="specialty-tag">Pediatrics</span>

**Base model:** `google/gemma-4-31b-it` &nbsp;|&nbsp;
**Fine-tuned on:** 50,000 pediatric Q&A cases (neonatal · vaccinations · emergencies · development · pharmacology) &nbsp;|&nbsp;
**Served via:** vLLM (OpenAI-compatible)

> ⚕️ For clinical decision support only. Always consult a licensed pediatrician for actual medical decisions.
"""

STATUS_TRAINING = """
> ⏳ **Model status:** Training in progress — fine-tuning on DGX Spark GB10 (~finishes Apr 17 2026).
> The UI is ready; connect to vLLM once the model finishes training.
> Set `VLLM_BASE_URL` in `.env` and restart.
"""


def build_interface():
    status_text, _ = _model_status()

    with gr.Blocks(title="PediatricianGemma") as demo:

        gr.Markdown(DESCRIPTION)
        gr.Markdown(f"**Status:** {status_text}")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="PediatricianGemma",
                    elem_id="chatbox",
                    render_markdown=True,
                    avatar_images=(None, "🩺"),
                )
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Ask a pediatric clinical question…",
                        show_label=False,
                        scale=9,
                        container=False,
                    )
                    send_btn = gr.Button("Send", scale=1, variant="primary")

                with gr.Row():
                    clear_btn = gr.Button("Clear conversation", size="sm")

            with gr.Column(scale=1):
                gr.Markdown("### 📋 Example Questions")
                gr.Markdown(
                    "*Click any question to load it. These are representative of the "
                    "50k pediatric cases the model was trained on.*"
                )
                for q in EXAMPLE_QUESTIONS:
                    # Truncate display to 2 lines
                    display = q[0][:120] + "…" if len(q[0]) > 120 else q[0]
                    gr.Button(display, size="sm", elem_classes=["example-btn"]).click(
                        fn=lambda text=q[0]: text,
                        outputs=msg_box,
                    )

        # Wire up interactions
        def user_submit(message, history):
            return "", history + [[message, None]]

        def bot_respond(history):
            if not history:
                return history
            user_msg = history[-1][0]
            history[-1][1] = ""
            for partial in respond(user_msg, history[:-1]):
                history[-1][1] = partial
                yield history

        msg_box.submit(user_submit, [msg_box, chatbot], [msg_box, chatbot], queue=False).then(
            bot_respond, chatbot, chatbot
        )
        send_btn.click(user_submit, [msg_box, chatbot], [msg_box, chatbot], queue=False).then(
            bot_respond, chatbot, chatbot
        )
        clear_btn.click(lambda: [], outputs=chatbot)

    return demo
