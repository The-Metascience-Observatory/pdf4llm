"""
Configuration management for the PDF-to-Markdown pipeline.

Supports three extraction modes:
- full-grobid: Best quality, uses GROBID for everything (~5s/PDF with GPU)
- hybrid: Fast extraction + GROBID fallback for failed tables
- fast: No GROBID, PyMuPDF + pdfplumber only
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import os


def _default_workers() -> int:
    """Default workers: min(10, cpu_count)."""
    try:
        cpus = os.cpu_count() or 4
    except Exception:
        cpus = 4
    return min(10, cpus)


class ExtractionMode(Enum):
    """Extraction mode determines which tools are used for PDF processing."""

    FULL_GROBID = "full-grobid"  # Best quality, GROBID for everything
    HYBRID = "hybrid"            # Fast + GROBID fallback for tables
    FAST = "fast"                # No GROBID, PyMuPDF + pdfplumber only

    @classmethod
    def from_string(cls, value: str) -> "ExtractionMode":
        """Convert string to ExtractionMode."""
        value = value.lower().replace("_", "-")
        for mode in cls:
            if mode.value == value:
                return mode
        raise ValueError(f"Unknown extraction mode: {value}. "
                        f"Valid modes: {[m.value for m in cls]}")


@dataclass
class Config:
    """Configuration for the PDF-to-Markdown pipeline."""

    # Extraction mode
    mode: ExtractionMode = ExtractionMode.FULL_GROBID

    # GROBID settings
    grobid_url: str = "http://localhost:8070"
    grobid_timeout: int = 120  # seconds
    grobid_retry_count: int = 3
    grobid_retry_delay: float = 1.0  # seconds between retries

    # Docker support
    use_docker_grobid: bool = False  # Whether to expect Docker-based GROBID
    docker_compose_path: Optional[Path] = None  # Path to docker-compose.yml

    # Output settings
    output_json: bool = True  # Also save JSON alongside Markdown
    output_tei: bool = False  # Save raw TEI XML (for debugging)

    # Batch processing settings
    workers: int = field(default_factory=_default_workers)
    checkpoint_enabled: bool = True
    checkpoint_file: Optional[Path] = None

    # Table extraction settings
    table_quality_threshold: float = 0.8  # Trigger fallback if below (increased for higher accuracy)

    # Chart/Figure extraction settings (uses Ollama by default - FREE)
    extract_charts: bool = False  # Enable chart analysis with vision model
    chart_provider: str = "ollama"  # "ollama" (free), "anthropic", or "openai"
    chart_model: Optional[str] = None  # Model name (default: llava:13b for ollama)
    ollama_url: str = "http://localhost:11434"

    # Logging
    verbose: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        return cls(
            mode=ExtractionMode.from_string(
                os.getenv("PDF2MD_MODE", "full-grobid")
            ),
            grobid_url=os.getenv("PDF2MD_GROBID_URL", "http://localhost:8070"),
            grobid_timeout=int(os.getenv("PDF2MD_GROBID_TIMEOUT", "120")),
            workers=int(os.getenv("PDF2MD_WORKERS", str(_default_workers()))),
            output_json=os.getenv("PDF2MD_OUTPUT_JSON", "true").lower() == "true",
            verbose=os.getenv("PDF2MD_VERBOSE", "false").lower() == "true",
            use_docker_grobid=os.getenv("PDF2MD_USE_DOCKER_GROBID", "false").lower() == "true",
            docker_compose_path=Path(os.getenv("PDF2MD_DOCKER_COMPOSE_PATH", "docker/docker-compose.yml")) if os.getenv("PDF2MD_DOCKER_COMPOSE_PATH") else None,
        )

    def requires_grobid(self) -> bool:
        """Check if the current mode requires GROBID."""
        return self.mode in (ExtractionMode.FULL_GROBID, ExtractionMode.HYBRID)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.grobid_timeout < 1:
            raise ValueError("grobid_timeout must be at least 1 second")
        if not 0 <= self.table_quality_threshold <= 1:
            raise ValueError("table_quality_threshold must be between 0 and 1")


# Default configuration instance
DEFAULT_CONFIG = Config()
