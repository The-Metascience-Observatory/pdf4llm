"""
pymupdf4llm extractor (final-last-resort fallback tier).

Used only when GROBID, docling, AND marker-pdf have all failed. pymupdf4llm
is a pure-PyMuPDF markdown wrapper: no ML models, no GROBID server, no
layout networks — just PyMuPDF's text-extraction engine with heading
detection and a markdown renderer on top. It's fast (~1-3s per PDF), tiny
in memory, and produces *something* on virtually any text-layer PDF, which
makes it the ideal last-resort tier: when marker-pdf has already failed,
we'd rather have flat markdown than nothing.

Tradeoffs versus the upstream tiers:
  - No structured references (references.json will be empty — the
    Reference section is pulled from the raw markdown via
    `_extract_references_section_from_markdown()` so references.md may
    still be populated).
  - No structured tables.
  - Abstract extraction is heuristic (first paragraph after the title
    heading) and may be missing on papers with unusual front-matter.
  - Title extraction falls back to PDF metadata /Title if no H1 heading
    is detected in the markdown output.

PyMuPDF is already a dependency of pdf4llm; `pymupdf4llm` is a thin
additional package (pure Python wrapper). If `pymupdf4llm` is not
installed, `_try_pymupdf4llm()` in pipeline.py will silently skip this
tier and mark the document as failed.
"""
import contextlib
import logging
import os
import re
import sys
from pathlib import Path
from typing import Iterator, Optional

from ..models import DocumentModel, ExtractionMetadata, Section
from ..extractors.fast import _extract_doi_from_filename, _validate_title, REF_HEADING_PATTERNS

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _suppress_native_output() -> Iterator[None]:
    """Temporarily redirect C-level stdout+stderr to /dev/null.

    `pymupdf4llm.to_markdown()` (and the underlying MuPDF C library) print
    "=== Document parser messages ===" banners and per-page OCR notices
    directly to fd 1 / fd 2, bypassing Python's logging. Those messages
    pollute batch output. This helper duplicates the approach in
    `ocr_fallback._suppress_c_stderr` and extends it to stdout as well.

    If the FDs can't be duplicated (e.g. running in an environment where
    sys.stdout has no fileno()), yields without suppression rather than
    raising.
    """
    try:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError):
        yield
        return

    # Flush Python-level buffers before we grab the FDs
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    old_stdout = os.dup(stdout_fd)
    old_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stdout_fd)
    os.dup2(devnull, stderr_fd)
    try:
        yield
    finally:
        os.dup2(old_stdout, stdout_fd)
        os.dup2(old_stderr, stderr_fd)
        os.close(devnull)
        os.close(old_stdout)
        os.close(old_stderr)


def extract_with_pymupdf4llm(pdf_path: Path, config) -> DocumentModel:
    """
    Convert a PDF to markdown using pymupdf4llm and wrap it into a DocumentModel.

    Only used as the final-last-resort fallback tier. Fast, memory-light,
    produces flat markdown with best-effort heading detection. Does not
    produce structured references or tables.
    """
    try:
        import pymupdf4llm
    except ImportError as e:
        raise RuntimeError(
            f"pymupdf4llm not installed. Install with: "
            f"pip install --user --break-system-packages pymupdf4llm  ({e})"
        )

    # Run conversion — this is the fast step (~1-3s per PDF on CPU).
    # page_chunks=False returns a single markdown string for the whole doc.
    # pymupdf4llm / MuPDF print status banners and per-page OCR notices
    # directly to native stdout/stderr, so we suppress both for the
    # duration of the call to keep batch output clean.
    try:
        with _suppress_native_output():
            text = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=False)
    except Exception as e:
        raise RuntimeError(f"pymupdf4llm conversion failed for {pdf_path.name}: {e}") from e

    if not text or len(text.strip()) < 100:
        raise RuntimeError(
            f"pymupdf4llm produced empty or trivially small markdown for {pdf_path.name}"
        )

    # Parse the markdown into abstract + body + references section
    abstract, body_text, refs_md = _split_pymupdf_markdown(text)

    # Title: first H1 heading in the markdown, else PDF metadata /Title
    title = _extract_title(text) or _extract_title_from_pdf_metadata(pdf_path) or (pdf_path.stem if _validate_title(pdf_path.stem) else "")

    doi = _extract_doi_from_filename(pdf_path)

    metadata = ExtractionMetadata(
        source_pdf=pdf_path.name,
        extraction_mode="pymupdf4llm-fallback",
    )

    # Build sections from the body markdown's headings so that downstream
    # quality scoring doesn't trip the "Very few sections extracted (1)"
    # warning on every pymupdf4llm-fallback doc. If no headings are found,
    # we still produce a single 'Body' section so body.md is never empty.
    sections = _split_body_into_sections(body_text)
    if not sections and body_text:
        sections = [Section(heading="Body", level=1, content=body_text)]

    doc = DocumentModel(
        metadata=metadata,
        doi=doi,
        title=title,
        abstract=abstract,
        sections=sections,
        tables=[],        # pymupdf4llm doesn't extract structured tables
        figures=[],       # pymupdf4llm doesn't extract figures
        references=[],    # pymupdf4llm doesn't produce structured references
    )

    # Stash raw markdown and references section for downstream rendering.
    # The pipeline's references.md writer checks these attributes as part
    # of its fallback chain (_docling_raw_md → _marker_refs_md → _marker_raw_md
    # → _pymupdf_raw_md / _pymupdf_refs_md).
    doc._pymupdf_raw_md = text
    doc._pymupdf_refs_md = refs_md

    return doc


def _extract_title(md_text: str) -> str:
    """Extract title from the pymupdf4llm markdown output.

    Prefer the first H1 heading of reasonable length. Skip headings that
    look like citations or 'Abstract'.
    """
    lines = md_text.split("\n")
    for line in lines[:40]:
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("##"):
            candidate = stripped[2:].strip()
            if len(candidate) >= 15 and not candidate.lower().startswith("abstract"):
                return candidate
    # Fallback: first non-empty, non-heading, non-citation line
    for line in lines[:20]:
        s = line.strip()
        if s and not s.startswith("!") and not s.startswith("#") and len(s) >= 15:
            if not re.match(r"^[A-Z][a-z]+,\s+[A-Z]\.", s):
                return s
    return ""


def _extract_title_from_pdf_metadata(pdf_path: Path) -> Optional[str]:
    """Fallback: read /Title from the PDF's XMP metadata via PyMuPDF."""
    try:
        import fitz
        with fitz.open(str(pdf_path)) as pdf:
            title = (pdf.metadata or {}).get("title") or ""
            title = title.strip()
            return title if len(title) >= 5 else None
    except Exception:
        return None


def _split_pymupdf_markdown(md_text: str) -> tuple[Optional[str], str, Optional[str]]:
    """
    Split pymupdf4llm's markdown output into (abstract, body, references_md).

    Uses heading detection for Abstract and References sections. If the
    headings aren't present, abstract will be None and references_md will
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


def _split_body_into_sections(body_md: str) -> list:
    """
    Split the body markdown into Section objects by ATX headings (#, ##, ###).

    Returns a list of Section(heading, level, content). If no headings are
    found, returns an empty list (the caller will wrap the full body in a
    single fallback section).
    """
    if not body_md:
        return []

    heading_re = re.compile(r"^(?P<hashes>#{1,3})\s+(?P<title>.+)$", re.MULTILINE)
    matches = list(heading_re.finditer(body_md))
    if not matches:
        return []

    sections = []
    for i, m in enumerate(matches):
        level = len(m.group("hashes"))
        heading = m.group("title").strip()
        content_start = m.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(body_md)
        content = body_md[content_start:content_end].strip()
        # Skip empty pseudo-sections (e.g. heading immediately followed by another heading)
        if heading or content:
            sections.append(Section(heading=heading or None, level=level, content=content))
    return sections
