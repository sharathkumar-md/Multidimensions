from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_DIR = Path("/content/MultiDimensions")
_REPO_HOST = "github.com/sharathkumar-md/Multidimensions.git"
# SEC-001: never embed the token in the URL — use GIT_ASKPASS instead
_REPO_URL_PUBLIC = f"https://{_REPO_HOST}"


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


def _make_askpass(token: str) -> str:
    """Write a temporary GIT_ASKPASS helper script that provides the token.

    Using GIT_ASKPASS keeps the credential out of the process table and
    command history — unlike embedding it in the remote URL.
    """
    script = f"#!/bin/sh\nexec echo '{token}'\n"
    fd, path = tempfile.mkstemp(suffix=".sh")
    os.write(fd, script.encode())
    os.close(fd)
    os.chmod(path, stat.S_IRWXU)
    return path


def _pip_install() -> None:
    packages = [
        "loguru",
        "qdrant-client>=1.9.0",
        "fastembed>=0.3.0",
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
    env = os.environ.copy()
    askpass = None
    if token:
        askpass = _make_askpass(token)
        env["GIT_ASKPASS"] = askpass
        env["GIT_USERNAME"] = "x-token"

    try:
        if REPO_DIR.exists():
            subprocess.run(["git", "-C", str(REPO_DIR), "pull"], env=env, check=True)
        else:
            subprocess.run(
                ["git", "clone", _REPO_URL_PUBLIC, str(REPO_DIR)],
                env=env,
                check=True,
            )
    finally:
        if askpass:
            try:
                os.unlink(askpass)
            except OSError:
                pass


def _run_model(model_id: str, hf_token: str | None = None) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_DISABLE_XET"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["RAG_OCR_OUTPUT_DIR"] = str(REPO_DIR / "data/ocr_output")
    env["RAG_QA_PATH"] = str(REPO_DIR / "data/ocr_output/qa_set.json")
    env["RAG_INDEX_DIR"] = str(REPO_DIR / "03_rag/index")
    env["RAG_RESULTS_DIR"] = str(REPO_DIR / "03_rag/results")
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

    # SEC-001: use GIT_ASKPASS so the token stays out of argv / shell history
    askpass = _make_askpass(token)
    env = os.environ.copy()
    env["GIT_ASKPASS"] = askpass
    env["GIT_USERNAME"] = "x-token"

    try:
        subprocess.run(["git", "-C", str(REPO_DIR), "config", "user.email", "colab@run.local"], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "config", "user.name", "colab"], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "add", "03_rag/results/"], check=True)
        result = subprocess.run(["git", "-C", str(REPO_DIR), "diff", "--cached", "--quiet"])
        if result.returncode != 0:
            subprocess.run(["git", "-C", str(REPO_DIR), "commit", "-m", "eval results"], check=True)
            subprocess.run(
                ["git", "-C", str(REPO_DIR), "push", _REPO_URL_PUBLIC, "main"],
                env=env,
                check=True,
            )
            print("results pushed", flush=True)
        else:
            print("nothing new to push", flush=True)
    finally:
        try:
            os.unlink(askpass)
        except OSError:
            pass


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
