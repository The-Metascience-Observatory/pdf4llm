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
    grobid_url: str = "http://127.0.0.1:8070"
    grobid_timeout: int = 120  # seconds
    # Auto-start GROBID via Docker if not already running.
    # Requires Docker to be installed and running.
    # grobid_docker_mode: "crf" (lightweight, ~500 MB) or "delft" (highest accuracy, ~8 GB).
    auto_start_grobid: bool = True
    grobid_docker_mode: str = "crf"  # "crf" or "delft"

    # Output settings
    output_json: bool = True  # Also save JSON alongside Markdown
    output_tei: bool = False  # Save raw TEI XML (for debugging)
    abstract_only: bool = False  # Only extract abstract → {DOI}_abstract.md
    single_markdown: bool = False  # Combine everything into one flat <stem>.md (no subfolder, no JSON)
    no_ocr: bool = False  # Disable Tesseract OCR fallback
    use_docling: bool = False  # Use docling instead of GROBID for extraction (docling-only mode)
    docling_use_gpu: bool = False  # Use CUDA for docling (requires GPU and CUDA-enabled docling install)

    # Parallel extraction architecture (new default for FULL_GROBID mode)
    # When True: runs GROBID AND docling in parallel per PDF, merges outputs
    # using the best of each tool (GROBID for headers/refs, docling for body/tables)
    parallel_extraction: bool = True
    docling_timeout: int = 300  # docling per-PDF timeout (seconds)
    use_marker_fallback: bool = True  # Try marker-pdf when both GROBID and docling fail
    marker_timeout: int = 600  # marker per-PDF timeout (seconds)
    # Final-last-resort fallback: pymupdf4llm (fast, pure Python, no ML).
    # Runs AFTER marker-pdf has already failed. Produces flat markdown with
    # heading-based section splitting; no structured references or tables.
    use_pymupdf_fallback: bool = True

    # Batch processing settings
    workers: int = field(default_factory=_default_workers)
    checkpoint_enabled: bool = True
    checkpoint_file: Optional[Path] = None

    # Table extraction settings
    table_quality_threshold: float = 0.8  # Trigger fallback if below (increased for higher accuracy)

    # CrossRef DOI enrichment: run when extracted DOI rate is below this fraction.
    # Set to 0.0 to disable, or use --no-crossref CLI flag.
    crossref_enrich_threshold: float = 0.5
    # Optional email for CrossRef polite pool (faster rate limits).
    crossref_mailto: Optional[str] = None

    # Chart/Figure extraction settings (uses Ollama by default - FREE)
    extract_charts: bool = False  # Enable chart analysis with vision model
    chart_provider: str = "ollama"  # "ollama" (free), "anthropic", or "openai"
    chart_model: Optional[str] = None  # Model name (default: llava:13b for ollama)
    ollama_url: str = "http://localhost:11434"

    # Plain image extraction (no vision model, no network, no cost).
    # Saves each figure as its own PNG via docling's layout model, which crops
    # the RENDERED page region -- so vector figures (matplotlib/R plots) are
    # captured, unlike charts.py's PyMuPDF embedded-raster scan.
    extract_images: bool = False
    # Render scale for cropped figures: 1.0 == 72 DPI, so 2.0 == 144 DPI.
    # Memory cost grows quadratically; 2.0 is a good default for figure crops.
    images_scale: float = 2.0

    # Logging
    verbose: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        return cls(
            mode=ExtractionMode.from_string(
                os.getenv("PDF4LLM_MODE", "full-grobid")
            ),
            grobid_url=os.getenv("PDF4LLM_GROBID_URL", "http://127.0.0.1:8070"),
            grobid_timeout=int(os.getenv("PDF4LLM_GROBID_TIMEOUT", "120")),
            workers=int(os.getenv("PDF4LLM_WORKERS", str(_default_workers()))),
            output_json=os.getenv("PDF4LLM_OUTPUT_JSON", "true").lower() == "true",
            verbose=os.getenv("PDF4LLM_VERBOSE", "false").lower() == "true",
            crossref_enrich_threshold=float(os.getenv("PDF4LLM_CROSSREF_THRESHOLD", "0.5")),
            crossref_mailto=os.getenv("PDF4LLM_CROSSREF_MAILTO"),
        )

    def requires_grobid(self) -> bool:
        """Check if the current mode requires GROBID."""
        if self.use_docling:
            return False
        return self.mode in (ExtractionMode.FULL_GROBID, ExtractionMode.HYBRID)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.grobid_timeout < 1:
            raise ValueError("grobid_timeout must be at least 1 second")
        if not 0 <= self.table_quality_threshold <= 1:
            raise ValueError("table_quality_threshold must be between 0 and 1")
        if not 0 <= self.crossref_enrich_threshold <= 1:
            raise ValueError("crossref_enrich_threshold must be between 0 and 1")


# Default configuration instance
DEFAULT_CONFIG = Config()
