from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).parent.parent / ".env"

_REPO = Path(__file__).resolve().parents[2]


class EvalSettings(BaseSettings):
    """Settings for 02_llm_eval.

    CODE-007 note: This module uses a local Ollama backend and is architecturally
    separate from the production RAG pipeline in 03_rag/ (which uses HuggingFace
    models).  The ocr_output_dir now defaults to the VLM OCR output so both
    modules evaluate the same underlying data source.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_prefix="EVAL_",
        extra="ignore",
    )

    # CODE-007: was "../01_ocr/output" (classic OCR); updated to production VLM output
    ocr_output_dir: Path = Field(default=_REPO / "data" / "ocr_output_vlm")
    results_dir: Path = Field(default=Path("results"))
    ollama_host: str = Field(default="http://localhost:11434")
    models_to_evaluate: list[str] = Field(
        default=["llama3.2:3b", "qwen2.5:7b", "phi3.5:mini"]
    )
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/eval.log")


settings = EvalSettings()
