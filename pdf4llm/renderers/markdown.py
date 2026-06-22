"""
LLM-optimized Markdown renderer.

Generates Markdown output designed for efficient LLM processing:
- YAML frontmatter with DOI for easy extraction
- Tables prefixed with [TABLE N: caption] for identification
- References with DOI prominently displayed
- Minimal formatting, maximum information density
"""

from typing import Optional
from ..models import DocumentModel, Section
from ..extractors.tables import table_to_markdown, normalize_table


def _render_section(section: Section, depth: int = 0) -> str:
    """Render a section recursively."""
    lines = []

    # Heading
    if section.heading:
        level = min(section.level, 6)
        prefix = "#" * level
        lines.append(f"{prefix} {section.heading}")
        lines.append("")

    # Content
    if section.content:
        lines.append(section.content)
        lines.append("")

    # Subsections
    for subsection in section.subsections:
        lines.append(_render_section(subsection, depth + 1))

    return "\n".join(lines)


def render_abstract_md(document: DocumentModel) -> str:
    """
    Render abstract.md: title and abstract text only.

    Citations appear as [N] markers linking to references.json.
    """
    lines = []
    lines.append(f"# {document.title}")
    lines.append("")
    if document.abstract:
        lines.append(document.abstract)
    return "\n".join(lines)


def render_body_md(document: DocumentModel) -> str:
    """
    Render body.md: full body text with section headers.

    Citations appear as [N] markers. No tables, figures, or references.
    """
    lines = []
    for section in document.sections:
        lines.append(_render_section(section))
    return "\n".join(lines).strip()


def render_tables_md(document: DocumentModel) -> str:
    """
    Render tables.md: all tables as markdown tables with captions.
    """
    if not document.tables:
        return ""

    lines = []
    for table in document.tables:
        normalized = normalize_table(table)
        lines.append(table_to_markdown(normalized))
        lines.append("")

    return "\n".join(lines).strip()


def render_single_markdown(document: DocumentModel, docling_raw_md: Optional[str] = None) -> str:
    """
    Render the whole document as one self-contained markdown file.

    Layout: title + metadata block → ## Abstract → body sections →
    ## Tables (omitted if none) → ## References (numbered; falls back
    to docling refs prose when document.references is empty).
    """
    parts: list[str] = []

    parts.append(f"# {document.title or 'Untitled'}")
    parts.append("")

    meta_bits = []
    if document.doi:
        meta_bits.append(f"**DOI:** {document.doi}")
    if document.year:
        meta_bits.append(f"**Year:** {document.year}")
    if meta_bits:
        parts.append("  \n".join(meta_bits))
        parts.append("")

    if document.abstract:
        parts.append("## Abstract")
        parts.append("")
        parts.append(document.abstract.strip())
        parts.append("")

    body = render_body_md(document)
    if body:
        parts.append(body)
        parts.append("")

    tables_md = render_tables_md(document)
    if tables_md:
        parts.append("## Tables")
        parts.append("")
        parts.append(tables_md)
        parts.append("")

    refs_md = _render_single_refs(document, docling_raw_md)
    if refs_md:
        parts.append("## References")
        parts.append("")
        parts.append(refs_md)
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def _render_single_refs(document: DocumentModel, docling_raw_md: Optional[str]) -> str:
    """Numbered references from structured list; fall back to docling refs prose."""
    if document.references:
        lines = []
        for i, ref in enumerate(document.references, start=1):
            bits = []
            if ref.authors:
                authors = ", ".join(ref.authors[:3]) + (", et al." if len(ref.authors) > 3 else "")
                bits.append(authors)
            if ref.year:
                bits.append(f"({ref.year})")
            if ref.title:
                bits.append(ref.title.strip().rstrip("."))
            if ref.journal:
                bits.append(ref.journal.strip().rstrip("."))
            if ref.doi:
                bits.append(f"doi:{ref.doi}")
            text = ". ".join(b for b in bits if b)
            if not text:
                text = (ref.raw_text or "").strip() or "[reference data unavailable]"
            lines.append(f"{i}. {text}")
        return "\n".join(lines)

    if docling_raw_md:
        # Find a References / Bibliography heading and return everything after it.
        import re
        m = re.search(r"(?im)^\s*#+\s*(references|bibliography)\s*$", docling_raw_md)
        if m:
            return docling_raw_md[m.end():].strip()
    return ""


