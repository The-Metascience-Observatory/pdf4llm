"""
PDF-to-Markdown Pipeline for LLM Analysis

A modular PDF converter optimized for LLM consumption in metascience research.
Supports three extraction modes:
- full-grobid: Best quality, uses GROBID for everything
- hybrid: Fast extraction + GROBID fallback for tables
- fast: No GROBID, PyMuPDF + pdfplumber only
"""

__version__ = "1.0.0"

from .config import Config, ExtractionMode
from .models import DocumentModel, Reference, Table, Section

__all__ = [
    "Config",
    "ExtractionMode",
    "DocumentModel",
    "Reference",
    "Table",
    "Section",
]
