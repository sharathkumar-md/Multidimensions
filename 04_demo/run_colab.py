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
def _git_clone_url() -> str:
    """Build authenticated clone URL for Colab.
    Token-in-URL is the only approach that reliably works in all Colab environments.
    The session is ephemeral — token exposure window is the runtime lifetime only."""
    return f"https://x-token:{GITHUB_TOKEN}@github.com/sharathkumar-md/Multidimensions.git"



# ── Step 1: clone / pull ───────────────────────────────────────────────────────
if not REPO_DIR.exists():
    print("📦  Cloning repository …")
    subprocess.run(
        ["git", "clone", _git_clone_url(), str(REPO_DIR)],
        check=True
    )
    print("✅  Cloned")
else:
    print("🔄  Pulling latest …")
    # For pull, configure credential store (repo already cloned, URL already set)
    cred = Path(os.path.expanduser("~/.git-credentials"))
    cred.write_text(f"https://x-token:{GITHUB_TOKEN}@github.com\n", encoding="utf-8")
    subprocess.run(["git", "config", "--global", "credential.helper", "store"], check=True)
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
    "duckduckgo-search>=6.0.0",
    "diskcache>=5.6.0",
    "authlib>=1.3.1",
    "PyJWT[crypto]>=2.8.0",
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


# ── Step 4: define port ──────────────────────────────────────────────────
port = 8501


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


# ── Step 6: Native Colab Proxy Instructions ─────────────────────────────────────
print(f"""
{'='*70}
  ✅ Streamlit is running in the background!
  First question loads the model (~2 min). Fast after that.
""")
