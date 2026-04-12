"""Shared FastAPI app, CORS, logging for all routes."""
import os
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.config import BACKEND_PORT, LOG_FILE

app = FastAPI(
    title="DocLlms API",
    description="Multi-specialist medical LLM platform — PediatricianGemma and friends",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def log(message: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} {message}"
    print(line)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass
