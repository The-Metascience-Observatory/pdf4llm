"""
marker-pdf extractor (final fallback tier).

Used only when BOTH GROBID and docling fail to produce usable output.
marker is a high-quality PDF→markdown converter with good OCR and layout
handling; it's especially strong on HTML-to-PDF layouts (Collabra, etc.)
that GROBID's DeLFT models can struggle with. Tradeoff: slow and heavy
(~3GB of models; ~20-30s per PDF on CPU).

Models are downloaded on first use and cached in ~/.cache/datalab/models/.
"""
import logging
import re
from pathlib import Path
from typing import Optional

from ..models import DocumentModel, ExtractionMetadata, Section
from ..extractors.fast import _extract_doi_from_filename, _validate_title, REF_HEADING_PATTERNS

logger = logging.getLogger(__name__)

# Process-local cache for marker models. They're expensive to load (~90s
# first time, faster after disk cache) and take ~3 GB of RAM. Loading once
# per process and reusing is essential.
_MARKER_MODELS_CACHE = None


def _get_marker_models():
    """Load marker models once per process (they're ~3 GB, ~90s first load)."""
    global _MARKER_MODELS_CACHE
    if _MARKER_MODELS_CACHE is None:
        from marker.models import create_model_dict
        logger.info("Loading marker models (first use may take 30-90s)...")
        _MARKER_MODELS_CACHE = create_model_dict()
        logger.info("marker models loaded")
    return _MARKER_MODELS_CACHE


def extract_with_marker(pdf_path: Path, config) -> DocumentModel:
    """
    Convert a PDF to markdown using marker-pdf, then parse into a DocumentModel.

    Only used as the final fallback tier. Slower and more memory-intensive
    than both GROBID and docling, but handles unusual layouts well.
    """
    try:
        from marker.converters.pdf import PdfConverter
        from marker.output import text_from_rendered
    except ImportError as e:
        raise RuntimeError(
            f"marker-pdf not installed. Install with: "
            f"pip install --user --break-system-packages marker-pdf  ({e})"
        )

    model_dict = _get_marker_models()
    converter = PdfConverter(artifact_dict=model_dict)

    # Run conversion (this is the slow step — ~20-60s per PDF on CPU)
    rendered = converter(str(pdf_path))
    text, _metadata, _images = text_from_rendered(rendered)

    if not text or len(text.strip()) < 100:
        raise RuntimeError(f"marker produced empty or trivially small markdown for {pdf_path.name}")

    # Parse marker's markdown into abstract + body + references section
    abstract, body_text, refs_md = _split_marker_markdown(text)

    # Extract title from first non-citation H1 heading
    title = _extract_title(text)

    doi = _extract_doi_from_filename(pdf_path)

    metadata = ExtractionMetadata(
        source_pdf=pdf_path.name,
        extraction_mode="marker-fallback",
    )

    sections = [Section(heading="Body", level=1, content=body_text)] if body_text else []

    doc = DocumentModel(
        metadata=metadata,
        doi=doi,
        title=title or (pdf_path.stem if _validate_title(pdf_path.stem) else ""),
        abstract=abstract,
        sections=sections,
        tables=[],        # marker embeds tables in markdown; no structured extraction
        figures=[],       # marker embeds figure placeholders; no structured extraction
        references=[],    # marker doesn't produce structured references
    )

    # Stash raw markdown and references section for downstream rendering
    # (references.md file will be written from _marker_refs_md)
    doc._marker_raw_md = text
    doc._marker_refs_md = refs_md

    return doc


def _extract_title(md_text: str) -> str:
    """Extract title from marker's markdown output.

    marker often starts with a citation (plain text) then a section header
    like `#### Social Psychology` then the real title as `# Title`.
    We want the first H1 heading that looks like a title, not a citation.
    """
    lines = md_text.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Prefer top-level # headings that aren't too short
        if stripped.startswith("# ") and not stripped.startswith("##"):
            candidate = stripped[2:].strip()
            if len(candidate) >= 15 and not candidate.lower().startswith("abstract"):
                return candidate
    # Fallback: first non-empty line that looks titular
    for line in lines[:20]:
        s = line.strip()
        if s and not s.startswith("!") and not s.startswith("#") and len(s) >= 15:
            # Skip lines that look like citations (e.g., "Imada, H., Chan, W. F.,...")
            if not re.match(r"^[A-Z][a-z]+,\s+[A-Z]\.", s):
                return s
    return ""


def _split_marker_markdown(md_text: str) -> tuple[Optional[str], str, Optional[str]]:
    """
    Split marker's markdown output into (abstract, body, references_md).

    Uses heading detection for the Abstract and References sections.
    If headings aren't present, abstract will be None and references_md will
    be None; the full text goes into body.
    """
    abstract = None
    references_md = None
    body = md_text

    # Locate Abstract (heading at any level 1-6 that matches "abstract")
    abs_match = re.search(r"(?im)^\s{0,3}#{1,6}\s*abstract\s*$", md_text)
    if abs_match:
        abstract_start = abs_match.end()
        # Abstract ends at next heading
        next_heading = re.search(r"^\s{0,3}#{1,6}\s", md_text[abstract_start:], re.MULTILINE)
        if next_heading:
            abstract = md_text[abstract_start:abstract_start + next_heading.start()].strip()
        else:
            abstract = md_text[abstract_start:].strip()

    # Locate References / Bibliography / Works Cited (heading)
    for pat in REF_HEADING_PATTERNS:
        m = re.search(pat, md_text)
        if m:
            references_md = md_text[m.end():].strip()
            body = md_text[:m.start()].strip()
            break

    return abstract, body, references_md
