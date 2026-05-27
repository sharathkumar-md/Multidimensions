from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to this file so the pipeline works from any CWD.
_ENV_FILE = Path(__file__).parent.parent / ".env"


class OCRSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_prefix="OCR_",
        case_sensitive=False,
        extra="ignore",
    )

    input_dir: Path = Field(default=Path("../../data/input"))
    output_dir: Path = Field(default=Path("output"))

    @field_validator("input_dir", "output_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: str | Path) -> Path:
        return Path(v)

    @property
    def markdown_dir(self) -> Path:
        return self.output_dir / "markdown"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def manifests_dir(self) -> Path:
        return self.output_dir / "manifests"

    def ensure_dirs(self) -> None:
        for d in (self.markdown_dir, self.figures_dir, self.manifests_dir):
            d.mkdir(parents=True, exist_ok=True)

    ocr_enabled: bool = Field(default=True)
    ocr_lang: list[str] = Field(default=["en"])
    force_full_ocr: bool = Field(default=False)

    extract_tables: bool = Field(default=True)
    min_table_cells: int = Field(default=4, ge=1)

    extract_figures: bool = Field(default=True)
    figure_dpi: int = Field(default=150, ge=72, le=600)

    strip_repeated_elements: bool = Field(default=True)
    repeated_element_threshold: int = Field(default=3, ge=2)
    pii_redaction_enabled: bool = Field(default=True)
    pii_redaction_entities: list[str] = Field(
        default=[
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "IBAN_CODE",
            "IP_ADDRESS",
        ]
    )

    max_workers: int = Field(default=2, ge=1, le=16)
    skip_duplicates: bool = Field(default=True)

    log_level: str = Field(default="INFO")
    log_file: Path | None = Field(default=Path("logs/ocr_pipeline.log"))
    log_rotation: str = Field(default="10 MB")
    log_retention: str = Field(default="30 days")


settings = OCRSettings()
