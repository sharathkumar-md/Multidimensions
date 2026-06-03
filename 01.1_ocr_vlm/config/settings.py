from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV),
        env_prefix="VLM_OCR_",
        case_sensitive=False,
        extra="ignore",
    )

    input_dir: Path = Field(default=Path("../../data/input"))
    output_dir: Path = Field(default=Path("../../data/ocr_output_vlm"))
    model_id: str = Field(default="Qwen/Qwen2.5-VL-7B-Instruct")
    page_dpi: int = Field(default=150)
    max_new_tokens: int = Field(default=1024)
    skip_duplicates: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    @field_validator("input_dir", "output_dir", mode="before")
    @classmethod
    def _to_path(cls, v):
        return Path(v)

    @property
    def markdown_dir(self):
        return self.output_dir / "markdown"

    @property
    def manifests_dir(self):
        return self.output_dir / "manifests"

    def ensure_dirs(self):
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
