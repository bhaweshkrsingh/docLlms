# ── DocLlms Docker Image (future deployment) ──────────────────────────────────
# NOT used yet — running non-docker on DGX Spark for simplicity.
# This Dockerfile is prepared for when the full multi-specialist platform
# is ready to be containerised and deployed.
#
# Build:
#   docker build -t docllms:latest .
#
# Run (single specialist):
#   docker run -d --name docllms --gpus all \
#     -p 7910:7910 -p 7920:7920 -p 7930:7930 -p 8101:8101 \
#     --ipc=host \
#     -v /home/ubuntu/dgxsparkfinetune/output:/app/models \
#     -e DEFAULT_SPECIALIST=pediatrician \
#     -e HF_TOKEN=$HF_TOKEN \
#     docllms:latest
#
# vLLM serving inside container (Blackwell / CUDA 13):
#   Uses vllm/vllm-openai:gemma4-cu130 as the base image (includes CUDA 13 + SM 12.1 support)
# ---------------------------------------------------------------------------

FROM vllm/vllm-openai:gemma4-cu130

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose ports
#   7910 — FastAPI backend
#   7920 — Gradio frontend
#   7930 — MCP server
#   8101-8109 — vLLM specialist endpoints
EXPOSE 7910 7920 7930 8101 8102 8103 8104 8105

# Default env
ENV BACKEND_HOST=0.0.0.0
ENV BACKEND_PORT=7910
ENV GRADIO_PORT=7920
ENV MCP_PORT=7930
ENV DEFAULT_SPECIALIST=pediatrician
ENV VLLM_BASE_URL=http://localhost:8101/v1

# Entrypoint: start all services
CMD ["bash", "scripts/start_all.sh"]
