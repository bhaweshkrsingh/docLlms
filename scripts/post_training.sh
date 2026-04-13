#!/bin/bash
# post_training.sh — Run everything after PediatricianGemma training completes
#
# Steps:
#   1. Verify merged BF16 model exists
#   2. Quantise to NVFP4 (nvidia-modelopt)
#   3. Update registry.yaml status: training → ready
#   4. Launch vLLM via Docker
#   5. Wait for vLLM to be healthy
#   6. Restart Gradio UI (connects to live model instead of mock answers)
#
# Usage:
#   bash /home/ubuntu/docLlms/scripts/post_training.sh

set -e

SPECIALIST="pediatrician"
TRAIN_DIR="/home/ubuntu/dgxsparkfinetune/output/pediatrician_gemma"
MERGED_MODEL="${TRAIN_DIR}/final_model/merged_model"
NVFP4_MODEL="${TRAIN_DIR}/../pediatrician_gemma_nvfp4"
CALIB_DATA="/home/ubuntu/medAI/pediatrics_50k.parquet"
REGISTRY="/home/ubuntu/docLlms/models/registry.yaml"
VLLM_PORT=8101

echo "================================================"
echo "  PediatricianGemma — Post-Training Pipeline"
echo "================================================"

# ── Step 1: Verify merged model ──────────────────────────────────────────────
echo ""
echo "[1/6] Checking merged BF16 model …"
if [ ! -d "$MERGED_MODEL" ]; then
    echo "ERROR: Merged model not found at $MERGED_MODEL"
    echo "  Check training log: tail -100 ${TRAIN_DIR}/train.log"
    exit 1
fi
MODEL_SIZE=$(du -sh "$MERGED_MODEL" | cut -f1)
echo "  OK — merged model found (${MODEL_SIZE})"

# ── Step 2: NVFP4 quantisation ────────────────────────────────────────────────
echo ""
echo "[2/6] Quantising to NVFP4 …"
if [ -d "$NVFP4_MODEL" ]; then
    echo "  NVFP4 model already exists — skipping quantisation."
    echo "  (Delete ${NVFP4_MODEL} to force re-quantise)"
else
    source /home/ubuntu/venv/bin/activate
    export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
    export CUDA_HOME=/usr/local/cuda-13.0
    export PATH=/usr/local/cuda-13.0/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}
    python /home/ubuntu/dgxsparkfinetune/quantize_to_nvfp4.py \
        --model-path        "$MERGED_MODEL" \
        --output-path       "$NVFP4_MODEL" \
        --calibration-data  "$CALIB_DATA" \
        --num-calibration-samples 512
    echo "  NVFP4 quantisation complete → ${NVFP4_MODEL}"
fi

# ── Step 3: Update registry status ───────────────────────────────────────────
echo ""
echo "[3/6] Updating registry.yaml: training → ready …"
python3 -c "
import yaml
with open('$REGISTRY') as f:
    r = yaml.safe_load(f)
for s in r['specialists']:
    if s['id'] == '$SPECIALIST':
        s['status'] = 'ready'
        s['nvfp4_model_path'] = '$NVFP4_MODEL'
with open('$REGISTRY', 'w') as f:
    yaml.dump(r, f, default_flow_style=False, allow_unicode=True)
print('  registry.yaml updated.')
"

# ── Step 4: Launch vLLM via Docker ────────────────────────────────────────────
echo ""
echo "[4/6] Launching vLLM (Docker) …"
cd /home/ubuntu/docLlms
bash scripts/serve_model.sh "$SPECIALIST"

# ── Step 5: Wait for vLLM health ─────────────────────────────────────────────
echo ""
echo "[5/6] Waiting for vLLM to be healthy (up to 120s) …"
for i in $(seq 1 24); do
    sleep 5
    HTTP=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${VLLM_PORT}/v1/models 2>/dev/null)
    if [ "$HTTP" = "200" ]; then
        echo "  vLLM is healthy after $((i*5))s"
        curl -s http://localhost:${VLLM_PORT}/v1/models | python3 -c "
import json,sys; d=json.load(sys.stdin)
print('  Model:', d['data'][0]['id'] if d.get('data') else 'unknown')
"
        break
    fi
    echo "  Waiting … (${i}/24, HTTP ${HTTP})"
done

if [ "$HTTP" != "200" ]; then
    echo "WARNING: vLLM did not become healthy within 120s."
    echo "  Check: docker logs vllm_${SPECIALIST}"
fi

# ── Step 6: Restart Gradio UI ─────────────────────────────────────────────────
echo ""
echo "[6/6] Restarting Gradio UI …"
fuser -k 7920/tcp 2>/dev/null || true
sleep 2
source /home/ubuntu/venv/bin/activate
nohup python frontend/gradio/pediatrician/launch.py > logs/gradio.log 2>&1 &
echo "  Gradio restarted (PID $!)"
sleep 5
HTTP_UI=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:7920/ 2>/dev/null)
echo "  Gradio HTTP: ${HTTP_UI}"

echo ""
echo "================================================"
echo "  ALL DONE"
echo "  Gradio UI  : http://localhost:7920"
echo "  vLLM API   : http://localhost:${VLLM_PORT}/v1"
echo "  vLLM logs  : docker logs -f vllm_${SPECIALIST}"
echo "================================================"
