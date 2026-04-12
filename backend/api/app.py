"""DocLlms backend API launcher."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

import uvicorn
from backend.api.routes import app  # noqa: F401 — triggers route registration
from backend.config import BACKEND_HOST, BACKEND_PORT, log_to_file  # type: ignore[attr-defined]
from backend.api.routes._shared import log


if __name__ == "__main__":
    log(f"Starting DocLlms API on {BACKEND_HOST}:{BACKEND_PORT}")
    uvicorn.run(app, host=BACKEND_HOST, port=BACKEND_PORT)
