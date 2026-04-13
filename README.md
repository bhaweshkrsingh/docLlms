# docLlms — Multi-Specialist Medical LLM Platform

Full-stack platform for serving fine-tuned specialist doctor LLMs on NVIDIA DGX Spark.
Each specialist model is a `google/gemma-4-31b-it` base fine-tuned on 50,000 domain-specific
medical Q&A records, served via vLLM, and exposed through a Gradio chat UI, FastAPI backend,
and MCP tool server.

Training pipeline: [`/home/ubuntu/dgxsparkfinetune`](../dgxsparkfinetune)

---

## Architecture

```
docLlms/
├── backend/
│   ├── api/            FastAPI — POST /api/chat (SSE streaming), GET /api/health
│   ├── agents/         BaseSpecialistAgent, specialist factory, keyword router
│   ├── llm/            VLLMClient (OpenAI-compatible), model_registry
│   └── mcp/            MCP server — 4 pediatric tools (dosing, vaccines, growth, labs)
├── frontend/
│   └── gradio/
│       └── pediatrician/   Chat UI — 15 deep clinical example questions + mock answers
├── models/
│   └── registry.yaml   Specialist registry: ports, model paths, status, system prompts
├── scripts/
│   ├── post_training.sh    ONE-SHOT: quantise → update registry → serve → restart UI
│   ├── serve_model.sh      Launch vLLM Docker container for one specialist
│   ├── serve_all.sh        Launch all ready specialists
│   └── start_all.sh        Start everything (vLLM + MCP + API + Gradio)
├── .env                Runtime config (gitignored — copy from .env.example)
└── requirements.txt    Python deps (vLLM served via Docker, not pip)
```

## Ports

| Service | Port |
|---------|------|
| Gradio UI | 7920 |
| FastAPI backend | 7910 |
| MCP server | 7930 |
| PediatricianGemma vLLM | 8101 |
| OncologyGemma vLLM | 8102 |
| CardiologyGemma vLLM | 8103 |
| NeurologyGemma vLLM | 8104 |
| ObGynGemma vLLM | 8105 |

## Specialist Models

| Specialist | Dataset | Rows | Status |
|-----------|---------|------|--------|
| PediatricianGemma | `pediatrics_50k.parquet` | 50,000 | **TRAINING** (~finishes Apr 17 2026) |
| OncologyGemma | `oncology_50k.parquet` | 50,000 | Planned |
| CardiologyGemma | `cardiology_50k.parquet` | 50,000 | Planned |
| NeurologyGemma | `neurology_50k.parquet` | 50,000 | Planned |
| ObGynGemma | `obgyn_50k.parquet` | 50,000 | Planned |

Each ~4.4 days training on DGX Spark GB10 (600 steps × 610 s/step).

---

## After Training Completes

One command does everything:

```bash
bash /home/ubuntu/docLlms/scripts/post_training.sh
```

Steps it runs automatically:
1. Verify merged BF16 model at `output/pediatrician_gemma/final_model/merged_model`
2. NVFP4 quantisation via nvidia-modelopt (512 calibration samples, ~30–60 min)
3. Update `models/registry.yaml`: `status: training` → `status: ready`
4. Launch vLLM via Docker (`vllm/vllm-openai:gemma4-cu130`)
5. Wait for vLLM health check at `http://localhost:8101/v1/models`
6. Restart Gradio UI — now connects to live model instead of serving mock answers

---

## Running Services Individually

```bash
cd /home/ubuntu/docLlms
source /home/ubuntu/venv/bin/activate

# Serve one specialist (Docker vLLM — auto-detects NVFP4 or BF16)
bash scripts/serve_model.sh pediatrician

# FastAPI backend
python backend/api/app.py

# MCP server
python -m backend.mcp.server

# Gradio UI only
python frontend/gradio/pediatrician/launch.py
```

## vLLM — Docker Serving

vLLM is served via the pre-pulled Docker image, not pip-installed into the venv.
This ensures the correct CUDA 13 / SM 12.1 (GB10 Blackwell) build is used.

```bash
# Check running vLLM containers
docker ps | grep vllm

# Logs
docker logs -f vllm_pediatrician

# Test endpoint
curl http://localhost:8101/v1/models
```

Docker image: `vllm/vllm-openai:gemma4-cu130` (21.9 GB, already pulled on DGX Spark)

## MCP Tools (Pediatric)

The MCP server at `http://localhost:7930/mcp` exposes four tools:

| Tool | Description |
|------|-------------|
| `get_vaccination_schedule(age_months)` | ACIP schedule by age |
| `get_pediatric_dosing(drug, weight_kg, age_years)` | Weight-based dosing |
| `get_growth_info(age_months, weight_kg, height_cm, sex)` | WHO/CDC growth assessment |
| `get_lab_reference_range(test, age_years, sex)` | Pediatric lab normals |

## Environment Setup

```bash
cp .env.example .env
# Edit .env: set HF_TOKEN, adjust ports if needed
source /home/ubuntu/venv/bin/activate
pip install -r requirements.txt
```
