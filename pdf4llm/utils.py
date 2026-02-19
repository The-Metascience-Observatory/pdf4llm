"""
Utility functions for the PDF-to-Markdown pipeline.
"""

import re
from pathlib import Path
from typing import Optional


def sanitize_filename(name: str, max_length: int = 100) -> str:
    """
    Sanitize a string for use as a filename.

    Args:
        name: String to sanitize
        max_length: Maximum filename length

    Returns:
        Safe filename string
    """
    # Remove or replace unsafe characters
    safe = re.sub(r'[<>:"/\\|?*]', '', name)
    safe = re.sub(r'\s+', '_', safe)
    safe = safe.strip('._')

    # Truncate if too long
    if len(safe) > max_length:
        safe = safe[:max_length].rsplit('_', 1)[0]

    return safe or "unnamed"


def normalize_doi(doi: str) -> Optional[str]:
    """
    Normalize a DOI string.

    Handles common variations:
    - https://doi.org/10.xxx
    - http://doi.org/10.xxx
    - doi:10.xxx
    - DOI: 10.xxx
    - 10.xxx (raw)

    Args:
        doi: DOI string to normalize

    Returns:
        Normalized DOI (just the 10.xxx part) or None if invalid
    """
    if not doi:
        return None

    doi = doi.strip()

    # Remove URL prefixes
    for prefix in ["https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/"]:
        if doi.lower().startswith(prefix.lower()):
            doi = doi[len(prefix):]
            break

    # Remove doi: prefix
    if doi.lower().startswith("doi:"):
        doi = doi[4:].strip()

    # Validate DOI format (should start with 10.)
    if not doi.startswith("10."):
        return None

    # Remove trailing punctuation
    doi = doi.rstrip(".,;:)")

    return doi


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a text string.

    This is a rough approximation (actual depends on tokenizer).
    Rule of thumb: ~4 characters per token for English.

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    # Simple approximation
    return len(text) // 4


def truncate_text(text: str, max_tokens: int = 4000) -> str:
    """
    Truncate text to approximately max_tokens.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens (approximate)

    Returns:
        Truncated text
    """
    max_chars = max_tokens * 4

    if len(text) <= max_chars:
        return text

    # Try to truncate at sentence boundary
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.8:
        return truncated[:last_period + 1] + " [TRUNCATED]"

    return truncated + "... [TRUNCATED]"


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable form.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_pdf_info(pdf_path: Path) -> dict:
    """
    Get basic PDF file information.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Dictionary with file info
    """
    pdf_path = Path(pdf_path)

    info = {
        "path": str(pdf_path),
        "name": pdf_path.name,
        "size": format_file_size(pdf_path.stat().st_size) if pdf_path.exists() else None,
        "exists": pdf_path.exists(),
    }

    return info
