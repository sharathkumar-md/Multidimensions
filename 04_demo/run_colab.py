"""
Run this in Colab to start the demo.
Each cell is marked — paste them one by one or run the whole file with !python 04_demo/run_colab.py
"""

# ── Cell 1: install deps ──────────────────────────────────────────────────────
import subprocess, sys

pkgs = [
    "streamlit",
    "pyngrok",
    # RAG deps (skip if already installed from eval run)
    "sentence-transformers",
    "chromadb",
    "rank-bm25",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "loguru",
    "pydantic-settings",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
print("deps installed")

# ── Cell 2: pull latest code ──────────────────────────────────────────────────
import subprocess
result = subprocess.run(["git", "-C", "/content/MultiDimensions", "pull"], capture_output=True, text=True)
print(result.stdout or result.stderr)

# ── Cell 3: start streamlit + ngrok ──────────────────────────────────────────
import subprocess, time, os, sys
from pathlib import Path
from pyngrok import ngrok

repo = Path("/content/MultiDimensions")
app = repo / "04_demo" / "app.py"

env = os.environ.copy()
env["HF_HUB_DISABLE_XET"] = "1"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", str(app),
     "--server.port", "8501",
     "--server.headless", "true",
     "--server.enableCORS", "false"],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

time.sleep(4)

tunnel = ngrok.connect(8501)
print(f"\n{'='*50}")
print(f"  Demo URL: {tunnel.public_url}")
print(f"{'='*50}\n")
print("Share this link with the founder.")
print("Model loads on first question (~2 min). Index loads in seconds.")

# ── Cell 4: stream logs (optional, run in separate cell) ─────────────────────
# for line in proc.stdout:
#     print(line.decode(), end="")
