from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).parent.parent / ".env"


class VLMOCRSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_prefix="VLM_OCR_",
        case_sensitive=False,
        extra="ignore",
    )

    input_dir: Path = Field(default=Path("../../data/input"))
    output_dir: Path = Field(default=Path("../../data/ocr_output_vlm"))

    @field_validator("input_dir", "output_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: str | Path) -> Path:
        return Path(v)

    @property
    def markdown_dir(self) -> Path:
        return self.output_dir / "markdown"

    @property
    def manifests_dir(self) -> Path:
        return self.output_dir / "manifests"

    def ensure_dirs(self) -> None:
        for d in (self.markdown_dir, self.manifests_dir):
            d.mkdir(parents=True, exist_ok=True)

    model_id: str = Field(default="Qwen/Qwen2-VL-7B-Instruct")
    page_dpi: int = Field(default=150, ge=72, le=300)
    max_new_tokens: int = Field(default=1024)
    skip_duplicates: bool = Field(default=True)

    log_level: str = Field(default="INFO")


settings = VLMOCRSettings()
