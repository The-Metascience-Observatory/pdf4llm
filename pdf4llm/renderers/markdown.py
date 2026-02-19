"""
LLM-optimized Markdown renderer.

Generates Markdown output designed for efficient LLM processing:
- YAML frontmatter with DOI for easy extraction
- Tables prefixed with [TABLE N: caption] for identification
- References with DOI prominently displayed
- Minimal formatting, maximum information density
"""

from typing import Optional
from ..models import DocumentModel, Section, Reference, Figure
from ..extractors.tables import table_to_markdown, normalize_table


def render_markdown(document: DocumentModel) -> str:
    """
    Render DocumentModel as LLM-optimized Markdown.

    Args:
        document: Parsed document model

    Returns:
        Markdown string
    """
    sections = []

    # YAML frontmatter
    sections.append(_render_frontmatter(document))

    # Abstract
    if document.abstract:
        sections.append("# Abstract\n")
        sections.append(document.abstract)
        sections.append("")

    # Main sections
    for section in document.sections:
        sections.append(_render_section(section))

    # Tables (if not already embedded in sections)
    if document.tables:
        # Check if we need a separate tables section
        sections.append("\n# Tables\n")
        for table in document.tables:
            normalized = normalize_table(table)
            sections.append(table_to_markdown(normalized))
            sections.append("")

    # Figures with AI-generated descriptions
    figures_with_descriptions = [f for f in document.figures if f.description]
    if figures_with_descriptions:
        sections.append(_render_figures(figures_with_descriptions))

    # References
    if document.references:
        sections.append(_render_references(document.references))

    return "\n".join(sections)


def _render_frontmatter(document: DocumentModel) -> str:
    """Render YAML frontmatter with document metadata."""
    lines = ["---"]

    if document.doi:
        lines.append(f'doi: "{document.doi}"')

    # Escape quotes in title
    title = document.title.replace('"', '\\"')
    lines.append(f'title: "{title}"')

    lines.append(f'source_pdf: "{document.metadata.source_pdf}"')
    lines.append(f'extraction_mode: "{document.metadata.extraction_mode}"')

    # Add quality metrics
    if document.metadata.quality_score is not None:
        lines.append(f"quality_score: {document.metadata.quality_score}")
    lines.append(f"tables_extracted: {document.metadata.tables_extracted}")
    lines.append(f"references_with_doi: {document.metadata.references_with_doi}")

    lines.append("---")
    lines.append("")

    return "\n".join(lines)


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


def _render_figures(figures: list[Figure]) -> str:
    """
    Render figures section with AI-generated descriptions.

    Format:
    ## Figure fig_1_1 (Page 3)
    [Caption if available]

    **Description**: [AI-generated description]

    **Extracted Data**:
    | ... |
    """
    lines = ["# Figures", ""]

    for fig in figures:
        # Figure header
        header = f"## Figure {fig.id}"
        if fig.page_number:
            header += f" (Page {fig.page_number})"
        lines.append(header)
        lines.append("")

        # Caption if available
        if fig.caption:
            lines.append(f"*{fig.caption}*")
            lines.append("")

        # AI-generated description
        if fig.description:
            lines.append(fig.description)
            lines.append("")

        # If chart_data is separate and not already in description
        if fig.chart_data and fig.chart_data not in (fig.description or ""):
            lines.append("**Extracted Data**:")
            lines.append(fig.chart_data)
            lines.append("")

    return "\n".join(lines)


def _render_references(references: list[Reference]) -> str:
    """
    Render references section with DOIs prominently displayed.

    Format:
    1. Author A, Author B. "Title." Journal. Year;Volume:Pages. DOI:10.xxx
    2. Author C. "Title." Journal. Year. [NO-DOI]
    """
    lines = ["# References", ""]

    for ref in references:
        parts = []

        # Reference number
        parts.append(f"{ref.num}.")

        # Authors
        if ref.authors:
            authors_str = ", ".join(ref.authors[:3])  # Limit to 3 authors
            if len(ref.authors) > 3:
                authors_str += ", et al"
            parts.append(authors_str + ".")

        # Title
        if ref.title:
            # Clean title
            title = ref.title.strip().rstrip(".")
            parts.append(f'"{title}."')

        # Journal
        if ref.journal:
            parts.append(ref.journal + ".")

        # Year, volume, pages
        pub_info = []
        if ref.year:
            pub_info.append(str(ref.year))
        if ref.volume:
            pub_info.append(f";{ref.volume}")
        if ref.pages:
            pub_info.append(f":{ref.pages}")
        if pub_info:
            parts.append("".join(pub_info) + ".")

        # DOI (CRITICAL - prominently displayed)
        if ref.doi:
            parts.append(f"DOI:{ref.doi}")
        else:
            parts.append("[NO-DOI]")

        lines.append(" ".join(parts))

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


def render_markdown_minimal(document: DocumentModel) -> str:
    """
    Render minimal Markdown for maximum token efficiency.

    This version strips formatting to bare minimum while
    preserving all critical information.

    Args:
        document: Parsed document model

    Returns:
        Minimal Markdown string
    """
    sections = []

    # Minimal header
    if document.doi:
        sections.append(f"DOI: {document.doi}")
    sections.append(f"TITLE: {document.title}")
    sections.append("")

    # Abstract
    if document.abstract:
        sections.append("ABSTRACT:")
        sections.append(document.abstract)
        sections.append("")

    # Sections as flat text
    for section in document.sections:
        sections.append(_render_section_minimal(section))

    # Tables
    for table in document.tables:
        normalized = normalize_table(table)
        sections.append(table_to_markdown(normalized, include_caption=True))
        sections.append("")

    # Figures (compact format)
    figures_with_descriptions = [f for f in document.figures if f.description]
    if figures_with_descriptions:
        sections.append("FIGURES:")
        for fig in figures_with_descriptions:
            sections.append(f"{fig.id}: {fig.description[:500]}...")
            if fig.chart_data:
                sections.append(fig.chart_data)
            sections.append("")

    # References (compact format)
    if document.references:
        sections.append("REFERENCES:")
        for ref in document.references:
            if ref.doi:
                sections.append(f"{ref.num}. {ref.title or 'Unknown'} | DOI:{ref.doi}")
            else:
                sections.append(f"{ref.num}. {ref.title or ref.raw_text or 'Unknown'} | [NO-DOI]")

    return "\n".join(sections)


def _render_section_minimal(section: Section) -> str:
    """Render section in minimal format."""
    lines = []

    if section.heading:
        lines.append(f"## {section.heading}")

    if section.content:
        lines.append(section.content)

    for subsection in section.subsections:
        lines.append(_render_section_minimal(subsection))

    return "\n".join(lines)
