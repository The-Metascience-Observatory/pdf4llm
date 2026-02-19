"""
Extractors for PDF content.

- grobid: GROBID-based extraction (highest quality)
- fast: PyMuPDF + pdfplumber (no GROBID dependency)
- tables: Table extraction utilities
- charts: Chart/figure extraction and analysis with vision models
"""

from .grobid import extract_with_grobid, parse_tei_xml
from .fast import extract_fast
from .tables import assess_table_quality, normalize_table, table_to_markdown
from .charts import ChartExtractor, extract_and_analyze_charts, check_ollama_available

__all__ = [
    "extract_with_grobid",
    "parse_tei_xml",
    "extract_fast",
    "assess_table_quality",
    "normalize_table",
    "table_to_markdown",
    "ChartExtractor",
    "extract_and_analyze_charts",
    "check_ollama_available",
]
