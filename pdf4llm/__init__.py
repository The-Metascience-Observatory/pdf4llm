"""
PDF-to-Markdown Pipeline for LLM Analysis

A modular PDF converter optimized for LLM consumption in metascience research.
Supports three extraction modes:
- full-grobid: Best quality, uses GROBID for everything
- hybrid: Fast extraction + GROBID fallback for tables
- fast: No GROBID, PyMuPDF + pdfplumber only
"""

__version__ = "1.0.0"

# Lazy re-exports: avoid pulling pydantic/.models at package import time so
# the pure-stdlib launcher can `import pdf4llm.launcher` from any Python env
# (e.g. base anaconda with broken numpy) without dragging in heavy deps.
__all__ = [
    "Config",
    "ExtractionMode",
    "DocumentModel",
    "Reference",
    "Table",
    "Section",
]

_LAZY = {
    "Config": (".config", "Config"),
    "ExtractionMode": (".config", "ExtractionMode"),
    "DocumentModel": (".models", "DocumentModel"),
    "Reference": (".models", "Reference"),
    "Table": (".models", "Table"),
    "Section": (".models", "Section"),
}


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module
        mod_name, attr = _LAZY[name]
        return getattr(import_module(mod_name, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
