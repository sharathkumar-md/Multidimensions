from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path("/content/MultiDimensions")
_REPO_HOST = "github.com/sharathkumar-md/Multidimensions.git"


def _get_github_token() -> str | None:
    try:
        from google.colab import userdata
        return userdata.get("GITHUB_TOKEN")
    except Exception:
        return None


def _get_hf_token() -> str | None:
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:
        return None


def _repo_url(token: str | None = None) -> str:
    if token:
        return f"https://{token}@{_REPO_HOST}"
    return f"https://{_REPO_HOST}"


def _pip_install() -> None:
    packages = [
        "loguru",
        "rank_bm25",
        "chromadb",
        "sentence-transformers>=3.0.0",
        "bitsandbytes",
        "accelerate",
        "transformers",
        "pydantic-settings",
    ]
    subprocess.run(
        [sys.executable, "-u", "-m", "pip", "install"] + packages,
        check=True,
    )


def _clone_or_pull(token: str | None = None) -> None:
    if REPO_DIR.exists():
        subprocess.run(["git", "-C", str(REPO_DIR), "pull"], check=True)
    else:
        subprocess.run(["git", "clone", _repo_url(token), str(REPO_DIR)], check=True)


def _run_model(model_id: str, hf_token: str | None = None) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_DISABLE_XET"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["RAG_OCR_OUTPUT_DIR"] = str(REPO_DIR / "data/ocr_output")
    env["RAG_QA_PATH"] = str(REPO_DIR / "data/ocr_output/qa_set.json")
    env["RAG_INDEX_DIR"] = str(REPO_DIR / "03_rag/index")
    env["RAG_RESULTS_DIR"] = "/content/results"
    env["RAG_HF_HOME"] = "/content/hf_cache"
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token

    subprocess.run(
        [sys.executable, "-u", str(REPO_DIR / "03_rag/rag_kaggle.py"), "--model", model_id],
        env=env,
        check=False,
    )


def _push_results(token: str | None = None) -> None:
    if not token:
        print("no GITHUB_TOKEN, skipping push", flush=True)
        return

    remote = _repo_url(token)
    subprocess.run(["git", "-C", str(REPO_DIR), "config", "user.email", "colab@run.local"], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "config", "user.name", "colab"], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "add", "03_rag/results/"], check=True)
    result = subprocess.run(["git", "-C", str(REPO_DIR), "diff", "--cached", "--quiet"])
    if result.returncode != 0:
        subprocess.run(["git", "-C", str(REPO_DIR), "commit", "-m", "eval results"], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "push", remote, "main"], check=True)
        print("results pushed", flush=True)
    else:
        print("nothing new to push", flush=True)


MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B",
    "google/gemma-3-12b-it",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    github_token = _get_github_token()
    hf_token = _get_hf_token()

    print("=== installing deps ===", flush=True)
    _pip_install()

    print("=== cloning/pulling repo ===", flush=True)
    _clone_or_pull(github_token)

    models = [args.model] if args.model else MODELS
    for model in models:
        print(f"\n=== {model} ===", flush=True)
        _run_model(model, hf_token)

    print("\n=== pushing results ===", flush=True)
    _push_results(github_token)
