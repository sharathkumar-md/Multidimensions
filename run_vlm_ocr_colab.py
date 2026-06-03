import os
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path("/content/MultiDimensions")
VLM_DIR = REPO_DIR / "01.1_ocr_vlm"


def install():
    subprocess.run([
        sys.executable, "-u", "-m", "pip", "install",
        "transformers>=4.49.0", "qwen-vl-utils", "pymupdf",
        "Pillow", "bitsandbytes", "accelerate", "pydantic-settings", "loguru",
    ], check=True)


def run_ocr():
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_DISABLE_XET": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "VLM_OCR_INPUT_DIR": str(REPO_DIR / "data/input"),
        "VLM_OCR_OUTPUT_DIR": str(REPO_DIR / "data/ocr_output_vlm"),
    })
    subprocess.run(
        [sys.executable, "-u", str(VLM_DIR / "run.py")],
        env=env, cwd=str(VLM_DIR), check=True,
    )


if __name__ == "__main__":
    print("installing deps...", flush=True)
    install()

    print("running VLM OCR...", flush=True)
    run_ocr()

    print(f"\ndone — output at {REPO_DIR / 'data/ocr_output_vlm'}", flush=True)
    print("to use with RAG: set RAG_OCR_OUTPUT_DIR to the path above", flush=True)
