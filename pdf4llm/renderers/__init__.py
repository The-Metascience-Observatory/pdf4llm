"""
Output renderers for converted documents.

- markdown: LLM-optimized Markdown output
- json_output: Structured JSON output
"""

from .markdown import (
    render_abstract_md,
    render_body_md,
    render_tables_md,
    render_single_markdown,
)
from .json_output import render_json, render_references_json

__all__ = [
    "render_abstract_md",
    "render_body_md",
    "render_tables_md",
    "render_single_markdown",
    "render_json",
    "render_references_json",
]
