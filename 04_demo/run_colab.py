"""
Run this in Colab to start the demo.
Each cell is marked — paste them one by one or run the whole file with !python 04_demo/run_colab.py
"""

# ── Cell 1: install deps ──────────────────────────────────────────────────────
import subprocess, sys

pkgs = [
    "streamlit",
    # RAG deps
    "sentence-transformers",
    "chromadb",
    "rank-bm25",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "loguru",
    "pydantic-settings",
    # Ingest pipeline deps
    "pdfplumber",
    "PyMuPDF",
    "docling",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
print("deps installed")

# ── Cell 2: pull latest code ──────────────────────────────────────────────────
import subprocess
result = subprocess.run(["git", "-C", "/content/MultiDimensions", "pull"], capture_output=True, text=True)
print(result.stdout or result.stderr)

# ── Cell 3: start streamlit + localtunnel ───────────────────────────────────────
import subprocess, time, os, sys, urllib.request
from pathlib import Path

repo = Path("/content/MultiDimensions")
app = repo / "04_demo" / "app.py"

env = os.environ.copy()
env["HF_HUB_DISABLE_XET"] = "1"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("Starting Streamlit...")
proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", str(app),
     "--server.port", "8501",
     "--server.headless", "true",
     "--server.enableCORS", "false"],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

time.sleep(5)

print("Downloading Cloudflare tunnel...")
subprocess.run(["wget", "-q", "-c", "-nc", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"])
subprocess.run(["chmod", "+x", "cloudflared-linux-amd64"])

print("Starting Cloudflare tunnel...")
import re
lt_proc = subprocess.Popen(
    ["./cloudflared-linux-amd64", "tunnel", "--url", "http://127.0.0.1:8501"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

url = ""
for line in lt_proc.stdout:
    m = re.search(r'https://[\w-]+\.trycloudflare\.com', line)
    if m:
        url = m.group(0)
        break
    
print(f"\n{'='*70}")
print(f"  Demo URL: {url}")
print(f"{'='*70}\n")
print("Share this link with the founder (No password/IP required for Cloudflare!).")
print("Model loads on first question (~2 min). Index loads in seconds.")

# ── Cell 4: stream logs (optional, run in separate cell) ─────────────────────
# for line in proc.stdout:
#     print(line.decode(), end="")
