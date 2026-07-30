"""Start the DataScout REST API from the project root."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn


PROJECT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_DIR / "back-end"

# `back-end` contains top-level modules such as api.py, agent.py and pipeline/.
os.chdir(BACKEND_DIR)
sys.path.insert(0, str(BACKEND_DIR))


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
    )
