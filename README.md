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
│       ├── pediatrician/   Chat UI — 15 deep pediatric clinical questions + mock answers
│       └── obgyn/          Chat UI — 15 deep OB/GYN clinical questions + mock answers
├── models/
│   └── registry.yaml   Specialist registry: ports, model paths, status, system prompts
├── scripts/
│   ├── post_training.sh        ONE-SHOT (pediatrician): quantise → registry → serve → UI
│   ├── post_training_obgyn.sh  ONE-SHOT (gynecologist): quantise → registry → serve → UI
│   ├── serve_model.sh          Launch vLLM Docker container for one specialist
│   ├── serve_all.sh            Launch all ready specialists
│   └── start_all.sh            Start everything (vLLM + MCP + API + all Gradio UIs)
├── .env                Runtime config (gitignored — copy from .env.example)
└── requirements.txt    Python deps (vLLM served via Docker, not pip)
```

## Ports

| Service | Port |
|---------|------|
| PediatricianGemma Gradio UI | 7920 |
| GynecologistGemma Gradio UI | 7921 |
| FastAPI backend | 7910 |
| MCP server | 7930 |
| PediatricianGemma vLLM | 8101 |
| OncologyGemma vLLM | 8102 |
| CardiologyGemma vLLM | 8103 |
| NeurologyGemma vLLM | 8104 |
| GynecologistGemma vLLM | 8105 |

## Specialist Models

| Specialist | Dataset | Rows | Status |
|-----------|---------|------|--------|
| PediatricianGemma | `pediatrics_50k.parquet` | 50,000 | **TRAINING** (~finishes Apr 17 2026) |
| GynecologistGemma | `obgyn_50k.parquet` | 50,000 | **NEXT** — data ready, train after Apr 17 |
| OncologyGemma | `oncology_50k.parquet` | 50,000 | Planned |
| CardiologyGemma | `cardiology_50k.parquet` | 50,000 | Planned |
| NeurologyGemma | `neurology_50k.parquet` | 50,000 | Planned |

Each ~4.4 days training on DGX Spark GB10 (600 steps × 610 s/step).

---

## After PediatricianGemma Training Completes (~Apr 17 2026)

One command does everything:

```bash
bash /home/ubuntu/docLlms/scripts/post_training.sh
```

Steps it runs automatically:
1. Verify merged BF16 model at `output/pediatrician_gemma/final_model/merged_model`
2. NVFP4 quantisation via nvidia-modelopt (512 calibration samples, ~30–60 min)
3. Update `models/registry.yaml`: `status: training` → `status: ready`
4. Launch vLLM via Docker (`vllm/vllm-openai:gemma4-cu130`, port 8101)
5. Wait for vLLM health check at `http://localhost:8101/v1/models`
6. Restart Gradio UI — now connects to live model instead of serving mock answers

---

## GynecologistGemma — Training and Serving

### Start Training (immediately after PediatricianGemma finishes)

```bash
cd /home/ubuntu/dgxsparkfinetune
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}
export PYTORCH_ALLOC_CONF=expandable_segments:True
source /home/ubuntu/venv/bin/activate

mkdir -p output/obgyn_gemma
nohup python finetune_dgx_spark.py \
    --model gemma-4-31b --method qlora \
    --dataset /home/ubuntu/medAI/obgyn_50k.parquet \
    --question-col question --answer-col answer \
    --epochs 1 --max-length 2048 \
    --output-dir output/obgyn_gemma \
> output/obgyn_gemma/train.log 2>&1 &
echo $! > output/obgyn_gemma/train.pid

# Start monitor
nohup bash monitor_training.sh \
    output/obgyn_gemma/train.log \
    output/obgyn_gemma/status.log 120 > /dev/null 2>&1 &
```

**Expected:** ~4.4 days (600 steps × ~615 s/step), finishes ~Apr 22 2026.

### After GynecologistGemma Training Completes (~Apr 22 2026)

```bash
bash /home/ubuntu/docLlms/scripts/post_training_obgyn.sh
```

Runs automatically: NVFP4 quantisation → update registry → launch vLLM (port 8105) → launch Gradio UI (port 7921)

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
