"""
MultiDimensions – Colab Demo  (one-shot bootstrap)
====================================================
CELL 1  →  GPU check (1 line)
CELL 2  →  Run this script (2 lines)

That's it.  Everything else (clone, deps, ingest, Streamlit, tunnel) is
handled here automatically.
"""
from __future__ import annotations
import os, stat, sys, subprocess, time, socket, tempfile, urllib.request
from pathlib import Path

REPO_DIR  = Path("/content/MultiDimensions")
REPO_URL  = "https://github.com/sharathkumar-md/Multidimensions.git"

# ── Step 0: read tokens from Colab Secrets (userdata) ─────────────────────────
try:
    from google.colab import userdata          # type: ignore
    _gh  = userdata.get("GITHUB_TOKEN") or ""
    _hf  = userdata.get("HF_TOKEN")     or ""
except Exception:
    _gh, _hf = "", ""

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or _gh
HF_TOKEN     = os.environ.get("HF_TOKEN")     or _hf

if not GITHUB_TOKEN:
    sys.exit("❌  GITHUB_TOKEN not found — add it in Colab Secrets (🔑 icon on the left).")
if not HF_TOKEN:
    sys.exit("❌  HF_TOKEN not found — add it in Colab Secrets (🔑 icon on the left).")

# Propagate into environment for all child processes
os.environ.update({
    "GITHUB_TOKEN":               GITHUB_TOKEN,
    "HF_TOKEN":                   HF_TOKEN,
    "HUGGING_FACE_HUB_TOKEN":     HF_TOKEN,
    "HF_HUB_DISABLE_XET":         "1",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "PYTORCH_CUDA_ALLOC_CONF":    "expandable_segments:True",
    "TOKENIZERS_PARALLELISM":     "false",
    "TQDM_DISABLE":               "1",
})
print("✅  Tokens loaded")


# ── helpers ────────────────────────────────────────────────────────────────────
def _askpass(token: str) -> str:
    """Write a throw-away GIT_ASKPASS helper — token never appears in argv."""
    script = f"#!/bin/sh\nexec echo '{token}'\n"
    fd, path = tempfile.mkstemp(suffix=".sh")
    os.write(fd, script.encode()); os.close(fd)
    os.chmod(path, stat.S_IRWXU)
    return path

def _git_env() -> dict:
    ap = _askpass(GITHUB_TOKEN)
    e  = os.environ.copy()
    e["GIT_ASKPASS"]  = ap
    e["GIT_USERNAME"] = "x-token"
    return e, ap          # caller must os.unlink(ap) in finally


# ── Step 1: clone / pull ───────────────────────────────────────────────────────
if not REPO_DIR.exists():
    print("📦  Cloning repository …")
    git_env, ap = _git_env()
    try:
        subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)],
                       env=git_env, check=True)
    finally:
        try: os.unlink(ap)
        except OSError: pass
    print("✅  Cloned")
else:
    print("🔄  Pulling latest …")
    r = subprocess.run(["git", "-C", str(REPO_DIR), "pull"],
                       capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip() or "Already up to date.")


# ── Step 2: install Python dependencies ───────────────────────────────────────
print("\n📦  Installing dependencies …")
pkgs = [
    "streamlit",
    "sentence-transformers>=3.0.0",
    "qdrant-client>=1.9.0",
    "fastembed>=0.3.0",
    "transformers>=4.45.0",
    "accelerate>=0.30.0",
    "bitsandbytes>=0.43.0",
    "loguru",
    "pydantic-settings>=2.3.0",
    "pdfplumber",
    "PyMuPDF",
    "rapidocr-onnxruntime",
    "qwen-vl-utils",
    "torchvision",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
print("✅  Dependencies ready")


# ── Step 3: ingestion (OCR → chunks → Qdrant index) ──────────────────────────
print("\n⚙️   Running ingestion (skips unchanged PDFs) …")
r = subprocess.run(
    [sys.executable, str(REPO_DIR / "03_rag/ingest.py")],
    cwd=str(REPO_DIR / "03_rag"),
)
if r.returncode != 0:
    sys.exit("❌  Ingestion failed — scroll up for the error.")
print("✅  Ingestion complete")


# ── Step 4: find a free port ──────────────────────────────────────────────────
s = socket.socket(); s.bind(("", 0)); port = s.getsockname()[1]; s.close()


# ── Step 5: launch Streamlit (background) ─────────────────────────────────────
print(f"\n🚀  Starting Streamlit on port {port} …")
st_env = os.environ.copy()
st_env["SKIP_INGEST"] = "1"   # ingestion already done above

st_proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run",
     str(REPO_DIR / "04_demo/app.py"),
     "--server.port",                str(port),
     "--server.headless",            "true",
     "--server.enableCORS",          "false",
     "--server.enableXsrfProtection","false"],
    env=st_env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

# Poll health endpoint (up to 60 s)
deadline = time.time() + 60
while time.time() < deadline:
    if st_proc.poll() is not None:
        print("❌  Streamlit crashed!\n",
              st_proc.stdout.read().decode(errors="replace"))
        sys.exit(1)
    try:
        urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=1)
        print("✅  Streamlit is ready"); break
    except Exception:
        time.sleep(1)
else:
    print("⚠️   Streamlit didn't respond in 60 s — may still be loading.")


# ── Step 6: start localtunnel & print URL ─────────────────────────────────────
lt = subprocess.Popen(
    ["npx", "--yes", "localtunnel", "--port", str(port)],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)

url = ""
for line in lt.stdout:
    print(" ", line.strip())
    if "your url is:" in line.lower():
        url = line[line.lower().index("your url is:") + len("your url is:"):].strip()
        break
    if time.time() > deadline + 90:          # absolute outer timeout
        print("⚠️   localtunnel URL not found — check output above."); break

try:
    ip = urllib.request.urlopen("https://ipv4.icanhazip.com", timeout=5).read().decode().strip()
except Exception:
    ip = "(visit icanhazip.com to find your IP)"

print(f"""
{'='*62}
  🌐  Open    : {url or '(see localtunnel output above)'}
  🔑  Password: {ip}
         ↑ paste this when the localtunnel page asks for it
{'='*62}
  First question loads the model (~2 min). After that it's fast.
""")
