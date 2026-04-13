#!/bin/bash
# post_training_obgyn.sh
# ==============================================================
# ONE-SHOT: everything needed after GynecologistGemma training
# completes on DGX Spark.
#
# Steps:
#   1. Verify merged BF16 model exists
#   2. NVFP4 quantisation (skip if already done)
#   3. Update registry.yaml: status planned → ready
#   4. Launch vLLM via Docker (port 8105)
#   5. Wait for vLLM health check
#   6. Launch Gradio UI (port 7921)
#
# Usage:
#   bash /home/ubuntu/docLlms/scripts/post_training_obgyn.sh
# ==============================================================

set -e

SPECIALIST="obgyn"
TRAIN_DIR="/home/ubuntu/dgxsparkfinetune/output/obgyn_gemma"
MERGED_MODEL="${TRAIN_DIR}/final_model/merged_model"
NVFP4_MODEL="/home/ubuntu/dgxsparkfinetune/output/obgyn_gemma_nvfp4"
CALIB_DATA="/home/ubuntu/medAI/obgyn_50k.parquet"
REGISTRY="/home/ubuntu/docLlms/models/registry.yaml"
QUANT_SCRIPT="/home/ubuntu/dgxsparkfinetune/quantize_to_nvfp4.py"
VLLM_PORT=8105
GRADIO_PORT=7921
VLLM_HEALTH="http://localhost:${VLLM_PORT}/v1/models"
DOCLLMS_ROOT="/home/ubuntu/docLlms"
LOG_DIR="${DOCLLMS_ROOT}/logs"

mkdir -p "$LOG_DIR"

source /home/ubuntu/venv/bin/activate
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "========================================================"
echo " GynecologistGemma Post-Training Pipeline"
echo "========================================================"
echo ""

# ── [1/6] Verify merged BF16 model ───────────────────────────
echo "[1/6] Checking merged BF16 model..."
if [ ! -d "$MERGED_MODEL" ]; then
    echo "ERROR: Merged model not found at: $MERGED_MODEL"
    echo "       Training may still be in progress, or merge step failed."
    echo "       Check: ls ${TRAIN_DIR}/final_model/"
    exit 1
fi
if [ ! -f "${MERGED_MODEL}/config.json" ]; then
    echo "ERROR: config.json missing in merged model — incomplete merge."
    exit 1
fi
echo "  ✓ Merged model found: $MERGED_MODEL"
echo ""

# ── [2/6] NVFP4 quantisation ──────────────────────────────────
echo "[2/6] NVFP4 quantisation..."
if [ -d "$NVFP4_MODEL" ] && [ -f "${NVFP4_MODEL}/config.json" ]; then
    echo "  ✓ NVFP4 model already exists at $NVFP4_MODEL — skipping quantisation."
else
    echo "  Running nvidia-modelopt quantisation (~30–60 min)..."
    python3 "$QUANT_SCRIPT" \
        --model-path  "$MERGED_MODEL" \
        --output-path "$NVFP4_MODEL" \
        --calibration-data "$CALIB_DATA" \
        --num-calibration-samples 512 \
        --calibration-batch-size 4
    echo "  ✓ NVFP4 quantisation complete: $NVFP4_MODEL"
fi
echo ""

# ── [3/6] Update registry.yaml ───────────────────────────────
echo "[3/6] Updating registry.yaml (status: planned → ready)..."
/home/ubuntu/venv/bin/python3 - << PYEOF
import yaml, re
with open("$REGISTRY") as f:
    content = f.read()

# Parse YAML to find the obgyn block and update status + nvfp4 path
data = yaml.safe_load(content)
for s in data["specialists"]:
    if s["id"] == "obgyn":
        s["status"] = "ready"
        s["nvfp4_model_path"] = "$NVFP4_MODEL"
        print(f"  Updated: id=obgyn  status→ready  nvfp4={s['nvfp4_model_path']}")

with open("$REGISTRY", "w") as f:
    yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
print("  registry.yaml updated.")
PYEOF
echo ""

# ── [4/6] Launch vLLM Docker container ───────────────────────
echo "[4/6] Launching GynecologistGemma vLLM (port ${VLLM_PORT})..."
bash "${DOCLLMS_ROOT}/scripts/serve_model.sh" "$SPECIALIST"
echo "  ✓ Docker container started."
echo ""

# ── [5/6] Wait for vLLM health ───────────────────────────────
echo "[5/6] Waiting for vLLM to become healthy..."
MAX_WAIT=180
INTERVAL=5
elapsed=0
while [ $elapsed -lt $MAX_WAIT ]; do
    if curl -sf "$VLLM_HEALTH" > /dev/null 2>&1; then
        echo "  ✓ vLLM healthy at ${VLLM_HEALTH}"
        break
    fi
    echo "  ... waiting ${elapsed}s / ${MAX_WAIT}s"
    sleep $INTERVAL
    elapsed=$((elapsed + INTERVAL))
done
if [ $elapsed -ge $MAX_WAIT ]; then
    echo "  WARNING: vLLM did not respond within ${MAX_WAIT}s."
    echo "  Check: docker logs vllm_obgyn"
fi
echo ""

# ── [6/6] Launch Gradio UI ────────────────────────────────────
echo "[6/6] Starting GynecologistGemma Gradio UI (port ${GRADIO_PORT})..."
OBGYN_VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
OBGYN_GRADIO_PORT="${GRADIO_PORT}"
export OBGYN_VLLM_BASE_URL OBGYN_GRADIO_PORT

# Kill any existing process on this port
EXISTING=$(lsof -ti tcp:${GRADIO_PORT} 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
    kill "$EXISTING" 2>/dev/null || true
    sleep 1
fi

cd "$DOCLLMS_ROOT"
nohup /home/ubuntu/venv/bin/python3 frontend/gradio/obgyn/launch.py \
    > "${LOG_DIR}/gradio_obgyn.log" 2>&1 &
echo "$!" > "${LOG_DIR}/gradio_obgyn.pid"
echo "  ✓ Gradio UI PID: $(cat ${LOG_DIR}/gradio_obgyn.pid)"
echo ""

echo "========================================================"
echo " GynecologistGemma pipeline complete!"
echo "========================================================"
echo "  Gradio UI  : http://0.0.0.0:${GRADIO_PORT}"
echo "  vLLM API   : http://localhost:${VLLM_PORT}/v1"
echo "  vLLM logs  : docker logs -f vllm_obgyn"
echo "  UI logs    : tail -f ${LOG_DIR}/gradio_obgyn.log"
echo ""
echo "Next: Plan OncologyGemma training run"
echo "  Dataset: /home/ubuntu/medAI/oncology_50k.parquet"
echo "  Command: cd /home/ubuntu/dgxsparkfinetune && bash run_medical_finetune.sh oncology"
echo "========================================================"
