#!/bin/bash
# serve_model.sh — Launch vLLM for one specialist model (non-docker)
#
# Usage:
#   ./scripts/serve_model.sh pediatrician
#   ./scripts/serve_model.sh oncologist
#
# Docker equivalent (for reference):
#   docker run -d --name gemma4-31b --gpus all -p 8000:8000 --ipc=host \
#     -v /home/bkprity/.cache/huggingface:/root/.cache/huggingface \
#     -e HF_TOKEN=<token> \
#     vllm/vllm-openai:gemma4-cu130 \
#     --model nvidia/Gemma-4-31B-IT-NVFP4 --max-model-len 262144 \
#     --gpu-memory-utilization 0.9067 \
#     --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4

SPECIALIST="${1:-pediatrician}"
REGISTRY="/home/ubuntu/docLlms/models/registry.yaml"

# Parse registry for this specialist
MODEL_PATH=$(python3 -c "
import yaml, sys
with open('$REGISTRY') as f:
    r = yaml.safe_load(f)
for s in r['specialists']:
    if s['id'] == '$SPECIALIST':
        print(s.get('trained_model_path', ''))
        sys.exit(0)
print('')
")

VLLM_PORT=$(python3 -c "
import yaml, sys
with open('$REGISTRY') as f:
    r = yaml.safe_load(f)
for s in r['specialists']:
    if s['id'] == '$SPECIALIST':
        print(s.get('vllm_port', 8101))
        sys.exit(0)
print(8101)
")

GPU_MEM=$(python3 -c "
import yaml, sys
with open('$REGISTRY') as f:
    r = yaml.safe_load(f)
for s in r['specialists']:
    if s['id'] == '$SPECIALIST':
        print(s.get('gpu_memory_utilization', 0.85))
        sys.exit(0)
print(0.85)
")

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: Specialist '$SPECIALIST' not found in registry."
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path not found: $MODEL_PATH"
    echo "Training may still be in progress. Check:"
    echo "  tail -f /home/ubuntu/dgxsparkfinetune/output/${SPECIALIST}_gemma/train.log"
    exit 1
fi

# Environment
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)
source /home/ubuntu/venv/bin/activate

echo "========================================"
echo "  Serving: $SPECIALIST"
echo "  Model:   $MODEL_PATH"
echo "  Port:    $VLLM_PORT"
echo "  GPU mem: ${GPU_MEM}"
echo "========================================"

LOG_FILE="/home/ubuntu/docLlms/logs/vllm_${SPECIALIST}.log"
mkdir -p /home/ubuntu/docLlms/logs

# Use the pre-pulled Docker image (vllm/vllm-openai:gemma4-cu130).
# This avoids installing vLLM into the training venv and ensures the correct
# CUDA 13 / SM 12.1 build is used on DGX Spark GB10.
CONTAINER_NAME="vllm_${SPECIALIST}"

# Stop and remove any existing container for this specialist
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Determine if we have an NVFP4 model (prefer it; fall back to BF16 merged)
NVFP4_PATH=$(python3 -c "
import yaml, sys
with open('$REGISTRY') as f:
    r = yaml.safe_load(f)
for s in r['specialists']:
    if s['id'] == '$SPECIALIST':
        print(s.get('nvfp4_model_path', ''))
        sys.exit(0)
print('')
")

if [ -n "$NVFP4_PATH" ] && [ -d "$NVFP4_PATH" ]; then
    SERVE_PATH="$NVFP4_PATH"
    DTYPE_FLAG="--quantization fp4"
    echo "  Format:  NVFP4 (full 1 PFLOPS inference)"
else
    SERVE_PATH="$MODEL_PATH"
    DTYPE_FLAG="--dtype bfloat16"
    echo "  Format:  BF16 merged (NVFP4 not yet quantised)"
fi

docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    -p "${VLLM_PORT}:8000" \
    --ipc=host \
    -v "${SERVE_PATH}:/model" \
    -v "/home/bkprity/.cache/huggingface:/root/.cache/huggingface" \
    -v "/home/bkprity/.cache/vllm:/root/.cache/vllm" \
    -e HF_TOKEN="$HF_TOKEN" \
    vllm/vllm-openai:gemma4-cu130 \
        --model /model \
        --host 0.0.0.0 \
        --port 8000 \
        --max-model-len 8192 \
        --gpu-memory-utilization "$GPU_MEM" \
        $DTYPE_FLAG \
        --enable-auto-tool-choice \
        --tool-call-parser gemma4 \
    2>&1 | tee "$LOG_FILE"

echo "========================================"
echo "  Container: $CONTAINER_NAME"
echo "  Endpoint:  http://localhost:${VLLM_PORT}/v1"
echo "  Logs:      docker logs -f $CONTAINER_NAME"
echo "========================================"
echo ""
echo "Wait ~60s for model to load, then test:"
echo "  curl http://localhost:${VLLM_PORT}/v1/models"
