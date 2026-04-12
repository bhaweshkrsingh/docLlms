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

nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --max-model-len 8192 \
    --gpu-memory-utilization "$GPU_MEM" \
    --dtype bfloat16 \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    > "$LOG_FILE" 2>&1 &

VLLM_PID=$!
echo "$VLLM_PID" > "/home/ubuntu/docLlms/logs/vllm_${SPECIALIST}.pid"
echo "vLLM started — PID $VLLM_PID"
echo "Log: $LOG_FILE"
echo "Endpoint: http://localhost:${VLLM_PORT}/v1"
echo ""
echo "Wait ~60s for model to load, then test:"
echo "  curl http://localhost:${VLLM_PORT}/v1/models"
