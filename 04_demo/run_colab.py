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
import os, stat, sys, subprocess, time, socket, tempfile, urllib.request
from pathlib import Path

REPO_DIR = Path("/content/MultiDimensions")

# ── helpers ─────────────────────────────────────────────────────────────────────

def _make_askpass(token: str) -> str:
    """SEC-001: write a throw-away GIT_ASKPASS script so the token never
    appears in the process table, argv, or shell history."""
    script = f"#!/bin/sh\nexec echo '{token}'\n"
    fd, path = tempfile.mkstemp(suffix=".sh")
    os.write(fd, script.encode())
    os.close(fd)
    os.chmod(path, stat.S_IRWXU)
    return path


# ── Step 1: clone repo if not already present ──────────────────────────────────
if not REPO_DIR.exists():
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
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
    askpass = _make_askpass(token)
    clone_env = os.environ.copy()
    clone_env["GIT_ASKPASS"] = askpass
    clone_env["GIT_USERNAME"] = "x-token"
    try:
        subprocess.run(
            ["git", "clone",
             "https://github.com/sharathkumar-md/Multidimensions.git",
             str(REPO_DIR)],
            env=clone_env,
            check=True,
        )
    finally:
        try:
            os.unlink(askpass)
        except OSError:
            pass
    print("Cloned!")
else:
    print("Repo already present. Pulling latest...")
    result = subprocess.run(
        ["git", "-C", str(REPO_DIR), "pull"],
        capture_output=True, text=True,
    )
    print(result.stdout or result.stderr)

# ── Step 2: install dependencies ───────────────────────────────────────────────
# COLAB-001: removed chromadb / rank-bm25 (dead deps), added qdrant-client + fastembed
print("\nInstalling dependencies...")
pkgs = [
    "streamlit",
    "sentence-transformers>=3.0.0",
    "qdrant-client>=1.9.0", "fastembed>=0.3.0",
    "transformers>=4.45.0", "accelerate>=0.30.0", "bitsandbytes>=0.43.0",
    "loguru", "pydantic-settings>=2.3.0",
    "pdfplumber", "PyMuPDF",
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

# COLAB-003: poll until Streamlit responds instead of a blind fixed sleep
print("Waiting for Streamlit to become ready...")
deadline = time.time() + 45
ready = False
while time.time() < deadline:
    if st_proc.poll() is not None:
        # Process died — capture output and abort
        print("\nCRITICAL ERROR: Streamlit crashed on startup!")
        print(st_proc.stdout.read().decode(errors="replace"))
        sys.exit(1)
    try:
        urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=1)
        ready = True
        break
    except Exception:
        time.sleep(1)

if not ready:
    print("\nWARNING: Streamlit did not respond within 45 s — it may still be loading.")

# ── Step 6: start localtunnel ───────────────────────────────────────────────────
print("Starting localtunnel...")
lt_proc = subprocess.Popen(
    ["npx", "--yes", "localtunnel", "--port", str(port)],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

# COLAB-002: timeout the URL-scan loop so it never hangs forever
url = ""
url_deadline = time.time() + 30
for line in lt_proc.stdout:
    print(f"  localtunnel: {line.strip()}")
    if "your url is:" in line.lower():
        parts = line.lower().split("your url is:")
        url = line[line.lower().index("your url is:") + len("your url is:"):].strip()
        break
    if time.time() > url_deadline:
        print("  [WARN] localtunnel URL not detected within 30 s — check output above.")
        break

try:
    ip = urllib.request.urlopen("https://ipv4.icanhazip.com", timeout=5).read().decode().strip()
except Exception:
    ip = "(could not fetch IP)"

print(f"""
{'='*65}
  Demo URL : {url or '(check localtunnel output above)'}
  Password : {ip}   ← enter this when localtunnel asks
{'='*65}

Model loads on first question (~2 min). All other responses are fast.
""")
