"""
MultiDimensions – Colab Demo Runner
=====================================
Run this single script in a Colab cell:

    from google.colab import userdata
    import subprocess, os
    os.environ["GITHUB_TOKEN"] = userdata.get("GITHUB_TOKEN")
    os.environ["HF_TOKEN"]     = userdata.get("HF_TOKEN")
    subprocess.run(["python", "/content/MultiDimensions/04_demo/run_colab.py"], check=True)

Or clone first, then:

    !python /content/MultiDimensions/04_demo/run_colab.py
"""

from __future__ import annotations
import os, sys, subprocess, time, socket, urllib.request
from pathlib import Path

REPO_DIR = Path("/content/MultiDimensions")

# ── Step 1: clone repo if not already present ──────────────────────────────────
if not REPO_DIR.exists():
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        # Try reading from Colab secrets at runtime
        try:
            from google.colab import userdata  # type: ignore
            token = userdata.get("GITHUB_TOKEN")
            os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN", "")
        except Exception:
            pass
    if not token:
        print("ERROR: GITHUB_TOKEN not found. Set it in Colab secrets or as env var.")
        sys.exit(1)
    print("Cloning repository...")
    subprocess.run(
        ["git", "clone",
         f"https://{token}@github.com/sharathkumar-md/Multidimensions.git",
         str(REPO_DIR)],
        check=True,
    )
    print("Cloned!")
else:
    print("Repo already present. Pulling latest...")
    result = subprocess.run(
        ["git", "-C", str(REPO_DIR), "pull"],
        capture_output=True, text=True,
    )
    print(result.stdout or result.stderr)

# ── Step 2: install dependencies ───────────────────────────────────────────────
print("\nInstalling dependencies...")
pkgs = [
    "streamlit",
    "sentence-transformers", "chromadb", "rank-bm25",
    "transformers", "accelerate", "bitsandbytes",
    "loguru", "pydantic-settings",
    "pdfplumber", "PyMuPDF", "docling",
    "rapidocr-onnxruntime", "qwen-vl-utils", "torchvision",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
print("Dependencies installed!")

# ── Step 3: run ingestion in a clean subprocess (no Streamlit interference) ────
print("\nRunning ingestion pipeline (skips if already up to date)...")
env = os.environ.copy()
env["TQDM_DISABLE"] = "1"
env["HF_HUB_DISABLE_XET"] = "1"
env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

ingest_result = subprocess.run(
    [sys.executable, str(REPO_DIR / "03_rag" / "ingest.py")],
    env=env,
    cwd=str(REPO_DIR / "03_rag"),
)
if ingest_result.returncode != 0:
    print("\nINGESTION FAILED — check errors above. Stopping.")
    sys.exit(1)
print("Ingestion complete (or already up to date).")

# ── Step 4: find a free port ────────────────────────────────────────────────────
def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

port = _free_port()

# ── Step 5: launch Streamlit ────────────────────────────────────────────────────
streamlit_env = os.environ.copy()
streamlit_env["HF_HUB_DISABLE_XET"] = "1"
streamlit_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
streamlit_env["SKIP_INGEST"] = "1"   # ingestion already done above

print(f"\nStarting Streamlit on port {port}...")
st_proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run",
     str(REPO_DIR / "04_demo" / "app.py"),
     "--server.port", str(port),
     "--server.headless", "true",
     "--server.enableCORS", "false",
     "--server.enableXsrfProtection", "false"],
    env=streamlit_env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

time.sleep(6)

if st_proc.poll() is not None:
    print("\nCRITICAL ERROR: Streamlit crashed on startup!")
    print(st_proc.stdout.read().decode())
    sys.exit(1)

# ── Step 6: start localtunnel ───────────────────────────────────────────────────
print("Starting localtunnel...")
lt_proc = subprocess.Popen(
    ["npx", "--yes", "localtunnel", "--port", str(port)],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

url = ""
for line in lt_proc.stdout:
    if "your url is:" in line:
        url = line.split("your url is:")[1].strip()
        break
    else:
        print(f"  localtunnel: {line.strip()}")

try:
    ip = urllib.request.urlopen("https://ipv4.icanhazip.com").read().decode().strip()
except Exception:
    ip = "(could not fetch IP)"

print(f"""
{'='*65}
  Demo URL : {url}
  Password : {ip}   ← enter this when localtunnel asks
{'='*65}

Model loads on first question (~2 min). All other responses are fast.
""")
