"""
JSON output renderer.

Generates structured JSON output for programmatic access
to extracted document data.
"""

import json
from typing import Any, Dict
from ..models import DocumentModel


def render_json(document: DocumentModel, indent: int = 2) -> str:
    """
    Render DocumentModel as JSON string.

    Args:
        document: Parsed document model
        indent: JSON indentation (default 2, use None for compact)

    Returns:
        JSON string
    """
    data = document_to_dict(document)
    return json.dumps(data, indent=indent, ensure_ascii=False)


def document_to_dict(document: DocumentModel) -> Dict[str, Any]:
    """
    Convert DocumentModel to dictionary suitable for JSON serialization.

    This creates a clean structure optimized for LLM/programmatic access.

    Args:
        document: Parsed document model

    Returns:
        Dictionary representation
    """
    return {
        "metadata": {
            "doi": document.doi,
            "title": document.title,
            "source_pdf": document.metadata.source_pdf,
            "extraction_mode": document.metadata.extraction_mode,
            "extraction_timestamp": document.metadata.extraction_timestamp,
            "grobid_version": document.metadata.grobid_version,
            "pipeline_version": document.metadata.pipeline_version,
            "quality": {
                "overall_score": document.metadata.quality_score,
                "issues": document.metadata.quality_issues,
                "tables_extracted": document.metadata.tables_extracted,
                "tables_from_fallback": document.metadata.tables_from_fallback,
                "references_total": document.metadata.references_total,
                "references_with_doi": document.metadata.references_with_doi,
            },
            "warnings": document.metadata.warnings,
        },
        "abstract": document.abstract,
        "sections": [_section_to_dict(s) for s in document.sections],
        "tables": [_table_to_dict(t) for t in document.tables],
        "figures": [_figure_to_dict(f) for f in document.figures],
        "references": [_reference_to_dict(r) for r in document.references],
    }


def _section_to_dict(section) -> Dict[str, Any]:
    """Convert Section to dictionary."""
    return {
        "heading": section.heading,
        "level": section.level,
        "content": section.content,
        "subsections": [_section_to_dict(s) for s in section.subsections],
    }


def _table_to_dict(table) -> Dict[str, Any]:
    """Convert Table to dictionary."""
    return {
        "id": table.id,
        "caption": table.caption,
        "headers": table.headers,
        "rows": table.rows,
        "footnotes": table.footnotes,
        "page_number": table.page_number,
        "quality_score": table.quality_score,
        "source": table.source,
    }


def _figure_to_dict(figure) -> Dict[str, Any]:
    """Convert Figure to dictionary."""
    return {
        "id": figure.id,
        "caption": figure.caption,
        "page_number": figure.page_number,
    }


def _reference_to_dict(ref) -> Dict[str, Any]:
    """
    Convert Reference to dictionary.

    DOI is placed first for easy access.
    """
    return {
        "num": ref.num,
        "doi": ref.doi,  # CRITICAL - placed prominently
        "title": ref.title,
        "authors": ref.authors,
        "journal": ref.journal,
        "year": ref.year,
        "volume": ref.volume,
        "issue": ref.issue,
        "pages": ref.pages,
    }


def render_references_json(document: DocumentModel, indent: int = 2) -> str:
    """
    Render references as a bare JSON array for the modular output format.

    Each object has an 'id' field matching [N] citation markers in the markdown.

    Returns:
        JSON array string: [{id, authors, title, journal, volume, issue, pages, year, doi}, ...]
    """
    refs = []
    for ref in document.references:
        refs.append({
            "id": ref.num,
            "authors": ref.authors,
            "title": ref.title,
            "journal": ref.journal,
            "volume": ref.volume,
            "issue": ref.issue,
            "pages": ref.pages,
            "year": ref.year,
            "doi": ref.doi,
        })
    return json.dumps(refs, indent=indent, ensure_ascii=False)


def render_json_compact(document: DocumentModel) -> str:
    """
    Render compact JSON (no indentation) for space efficiency.

    Args:
        document: Parsed document model

    Returns:
        Compact JSON string
    """
    return render_json(document, indent=None)


def render_references_only(document: DocumentModel, indent: int = 2) -> str:
    """
    Render only the references as JSON.

    Useful for extracting just the bibliography for DOI lookups.

    Args:
        document: Parsed document model
        indent: JSON indentation

    Returns:
        JSON string with references only
    """
    data = {
        "source_pdf": document.metadata.source_pdf,
        "document_doi": document.doi,
        "references_total": len(document.references),
        "references_with_doi": sum(1 for r in document.references if r.doi),
        "references": [_reference_to_dict(r) for r in document.references],
    }
    return json.dumps(data, indent=indent, ensure_ascii=False)


def render_tables_only(document: DocumentModel, indent: int = 2) -> str:
    """
    Render only the tables as JSON.

    Useful for extracting just numerical data.

    Args:
        document: Parsed document model
        indent: JSON indentation

    Returns:
        JSON string with tables only
    """
    data = {
        "source_pdf": document.metadata.source_pdf,
        "document_doi": document.doi,
        "tables_count": len(document.tables),
        "tables": [_table_to_dict(t) for t in document.tables],
    }
    return json.dumps(data, indent=indent, ensure_ascii=False)
