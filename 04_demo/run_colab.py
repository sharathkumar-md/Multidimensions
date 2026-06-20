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
    "rapidocr-onnxruntime",
    "qwen-vl-utils",
    "torchvision"
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
print("deps installed")

# ── Cell 2: pull latest code ──────────────────────────────────────────────────
import subprocess
result = subprocess.run(["git", "-C", "/content/MultiDimensions", "pull"], capture_output=True, text=True)
print(result.stdout or result.stderr)

# ── Cell 3: run ingestion pipeline (before Streamlit, in a clean process) ─────
import subprocess, sys, os
from pathlib import Path

repo = Path("/content/MultiDimensions")
ingest_script = repo / "03_rag" / "ingest.py"

print("Running ingestion pipeline (skips if already done)...")
env = os.environ.copy()
env["TQDM_DISABLE"] = "1"
env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
env["HF_HUB_DISABLE_XET"] = "1"

result = subprocess.run(
    [sys.executable, str(ingest_script)],
    env=env,
    cwd=str(repo / "03_rag"),
)
if result.returncode != 0:
    print("\nINGESTION FAILED - check errors above before launching Streamlit")
else:
    print("Ingestion complete (or already up to date).")

# ── Cell 4: start streamlit + tunnel ───────────────────────────────────────
import subprocess, time, os, sys, urllib.request, socket
from pathlib import Path

def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

# Automatically find a free port so we never get "Port is not available" errors
port = get_free_port()

repo = Path("/content/MultiDimensions")
app = repo / "04_demo" / "app.py"

env = os.environ.copy()
env["HF_HUB_DISABLE_XET"] = "1"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
env["SKIP_INGEST"] = "1"  # Ingestion already ran in Cell 3

print(f"Starting Streamlit on port {port}...")
proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", str(app),
     "--server.port", str(port),
     "--server.headless", "true",
     "--server.enableCORS", "false",
     "--server.enableXsrfProtection", "false"],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

time.sleep(6)

if proc.poll() is not None:
    print("\nCRITICAL ERROR: Streamlit crashed on startup!")
    print(proc.stdout.read().decode())
    sys.exit(1)

print("Starting localtunnel...")
lt_proc = subprocess.Popen(
    ["npx", "--yes", "localtunnel", "--port", str(port)],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

url = ""
for line in lt_proc.stdout:
    if "your url is:" in line:
        url = line.split("your url is:")[1].strip()
        break
    else:
        print(f"localtunnel logs: {line.strip()}")

ip = urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip()

print(f"\n{'='*70}")
print(f"  Demo URL: {url}")
print(f"  Endpoint IP (password for localtunnel): {ip}")
print(f"{'='*70}\n")
print("Share this link with the founder.")
print("Model loads on first question (~2 min). Index loads in seconds.")

# ── Cell 4: stream logs (optional, run in separate cell) ─────────────────────
# for line in proc.stdout:
#     print(line.decode(), end="")
