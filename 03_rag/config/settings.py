from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).parent.parent / ".env"


class RAGSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_prefix="RAG_",
        extra="ignore",
    )

    ocr_output_dir: Path = Field(default=Path("../data/ocr_output"))
    index_dir: Path = Field(default=Path("index"))
    results_dir: Path = Field(default=Path("results"))

    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=100)

    embed_model: str = Field(default="BAAI/bge-large-en-v1.5")
    top_k_dense: int = Field(default=20)
    top_k_sparse: int = Field(default=20)
    top_k_rerank: int = Field(default=5)
    hyde_enabled: bool = Field(default=True)

    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    nli_model: str = Field(default="cross-encoder/nli-deberta-v3-small")

    models_to_evaluate: list[str] = Field(default=[
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen3-8B",
        "google/gemma-3-12b-it",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ])

    log_level: str = Field(default="INFO")
    log_file: Path = Field(default=Path("logs/rag.log"))


settings = RAGSettings()
