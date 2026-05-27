from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).parent.parent / ".env"


class EvalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_prefix="EVAL_",
        extra="ignore",
    )

    ocr_output_dir: Path = Field(default=Path("../01_ocr/output"))
    results_dir: Path = Field(default=Path("results"))
    ollama_host: str = Field(default="http://localhost:11434")
    models_to_evaluate: list[str] = Field(
        default=["llama3.2:3b", "qwen2.5:7b", "phi3.5:mini"]
    )
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/eval.log")


settings = EvalSettings()
