#!/bin/bash
# start_all.sh — Start all services (used inside Docker and locally)
# Services: vLLM (specialist models) + MCP server + FastAPI backend + Gradio frontend

set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=== DocLlms — Starting all services ==="

# 1. Serve ready specialist models
echo "[1/4] Starting vLLM specialist endpoints..."
bash "$ROOT_DIR/scripts/serve_all.sh"
echo "       Waiting 60s for models to load..."
sleep 60

# 2. MCP server
echo "[2/4] Starting MCP server on port ${MCP_PORT:-7930}..."
source /home/ubuntu/venv/bin/activate 2>/dev/null || true
nohup python -m backend.mcp.server > "$LOG_DIR/mcp_server.log" 2>&1 &
echo "$!" > "$LOG_DIR/mcp_server.pid"
sleep 2

# 3. FastAPI backend
echo "[3/4] Starting FastAPI backend on port ${BACKEND_PORT:-7910}..."
nohup python backend/api/app.py > "$LOG_DIR/backend.log" 2>&1 &
echo "$!" > "$LOG_DIR/backend.pid"
sleep 2

# 4. Gradio frontend
echo "[4/4] Starting Gradio UI on port ${GRADIO_PORT:-7920}..."
nohup python frontend/gradio/pediatrician/launch.py > "$LOG_DIR/gradio.log" 2>&1 &
echo "$!" > "$LOG_DIR/gradio.pid"

echo ""
echo "=== All services started ==="
echo "  Gradio UI : http://0.0.0.0:${GRADIO_PORT:-7920}"
echo "  API       : http://0.0.0.0:${BACKEND_PORT:-7910}"
echo "  MCP server: http://0.0.0.0:${MCP_PORT:-7930}/mcp"
echo "  vLLM      : http://localhost:8101/v1  (pediatrician)"
echo ""
echo "Logs in: $LOG_DIR"
