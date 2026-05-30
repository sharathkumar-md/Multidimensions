from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path("/kaggle/working/MultiDimensions")
REPO_URL = "https://github.com/sharathkumar-md/Multidimensions.git"


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
        [sys.executable, "-u", "-m", "pip", "install", "--quiet"] + packages,
        check=True,
    )


def _clone_or_pull() -> None:
    if REPO_DIR.exists():
        subprocess.run(["git", "-C", str(REPO_DIR), "pull"], check=True)
    else:
        subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)


def _set_hf_token() -> None:
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        os.environ["HF_TOKEN"] = token
    except Exception:
        pass


def _run_model(model_id: str) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["RAG_OCR_OUTPUT_DIR"] = str(REPO_DIR / "data/ocr_output")
    env["RAG_QA_PATH"] = str(REPO_DIR / "data/ocr_output/qa_set.json")

    subprocess.run(
        [sys.executable, "-u", str(REPO_DIR / "03_rag/rag_kaggle.py"), "--model", model_id],
        env=env,
        check=False,
    )


def _push_results() -> None:
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        token = secrets.get_secret("GITHUB_TOKEN")
    except Exception as e:
        print(f"no GITHUB_TOKEN secret ({e}), skipping push")
        return

    remote = f"https://{token}@github.com/sharathkumar-md/Multidimensions.git"
    subprocess.run(["git", "-C", str(REPO_DIR), "config", "user.email", "kaggle@run.local"], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "config", "user.name", "kaggle"], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "add", "03_rag/results/"], check=True)
    result = subprocess.run(["git", "-C", str(REPO_DIR), "diff", "--cached", "--quiet"])
    if result.returncode != 0:
        subprocess.run(["git", "-C", str(REPO_DIR), "commit", "-m", "eval results"], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "push", remote, "main"], check=True)
        print("results pushed")
    else:
        print("nothing new to push")


MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B",
    "google/gemma-3-12b-it",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]

if __name__ == "__main__":
    print("=== installing deps ===")
    _pip_install()

    print("=== cloning/pulling repo ===")
    _clone_or_pull()

    _set_hf_token()

    for model in MODELS:
        print(f"\n=== {model} ===")
        _run_model(model)

    print("\n=== pushing results ===")
    _push_results()
