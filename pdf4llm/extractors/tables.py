"""
Table extraction utilities.

Provides:
- Table quality assessment
- Table normalization
- Markdown rendering for tables
"""

import re
from typing import List, Optional
from ..models import Table


def assess_table_quality(table: Table) -> float:
    """
    Assess the quality of extracted table data.

    Returns a score from 0.0 to 1.0 where:
    - 1.0 = Perfect extraction
    - 0.6 = Threshold for triggering fallback
    - 0.0 = Completely failed extraction

    Args:
        table: Table to assess

    Returns:
        Quality score between 0 and 1
    """
    scores = []

    # 1. Header detection (0.3 weight)
    if table.headers:
        # Check that headers look reasonable (not empty, not all numeric)
        non_empty = sum(1 for h in table.headers if h.strip())
        all_numeric = all(re.match(r"^[\d\.\,\-\s]*$", h) for h in table.headers if h.strip())

        if non_empty > 0 and not all_numeric:
            scores.append(1.0)
        elif non_empty > 0:
            scores.append(0.5)
        else:
            scores.append(0.0)
    else:
        scores.append(0.0)

    # 2. Cell population (0.3 weight)
    total_cells = sum(len(row) for row in table.rows)
    if total_cells > 0:
        non_empty = sum(1 for row in table.rows for cell in row if cell.strip())
        cell_ratio = non_empty / total_cells
        scores.append(cell_ratio)
    else:
        scores.append(0.0)

    # 3. Column consistency (0.2 weight)
    if table.rows:
        col_counts = [len(row) for row in table.rows]
        if len(set(col_counts)) == 1:
            # All rows have same number of columns
            scores.append(1.0)
        else:
            # Some variation
            most_common = max(set(col_counts), key=col_counts.count)
            consistent_ratio = col_counts.count(most_common) / len(col_counts)
            scores.append(consistent_ratio * 0.7)
    else:
        scores.append(0.0)

    # 4. Header-row alignment (0.2 weight)
    if table.headers and table.rows:
        header_cols = len(table.headers)
        row_cols = [len(row) for row in table.rows]
        avg_row_cols = sum(row_cols) / len(row_cols) if row_cols else 0

        if header_cols == round(avg_row_cols):
            scores.append(1.0)
        elif abs(header_cols - avg_row_cols) <= 1:
            scores.append(0.7)
        else:
            scores.append(0.3)
    else:
        scores.append(0.5)

    # Weighted average
    weights = [0.3, 0.3, 0.2, 0.2]
    weighted_score = sum(s * w for s, w in zip(scores, weights))

    return round(weighted_score, 3)


def normalize_table(table: Table) -> Table:
    """
    Normalize table data for cleaner output.

    - Strips whitespace from all cells
    - Normalizes unicode characters
    - Ensures consistent column counts
    - Removes empty rows

    Args:
        table: Table to normalize

    Returns:
        Normalized Table
    """
    # Normalize headers
    headers = [_normalize_cell(h) for h in table.headers] if table.headers else []

    # Determine target column count
    target_cols = len(headers) if headers else (
        max(len(row) for row in table.rows) if table.rows else 0
    )

    # Normalize rows
    normalized_rows = []
    for row in table.rows:
        normalized_row = [_normalize_cell(cell) for cell in row]

        # Skip completely empty rows
        if not any(cell for cell in normalized_row):
            continue

        # Pad or trim to target column count
        if len(normalized_row) < target_cols:
            normalized_row.extend([""] * (target_cols - len(normalized_row)))
        elif len(normalized_row) > target_cols and target_cols > 0:
            normalized_row = normalized_row[:target_cols]

        normalized_rows.append(normalized_row)

    # Normalize footnotes
    footnotes = [_normalize_cell(f) for f in table.footnotes if _normalize_cell(f)]

    return Table(
        id=table.id,
        caption=_normalize_cell(table.caption) if table.caption else None,
        headers=headers,
        rows=normalized_rows,
        footnotes=footnotes,
        page_number=table.page_number,
        quality_score=table.quality_score,
        source=table.source,
    )


def _normalize_cell(text: Optional[str]) -> str:
    """Normalize a single cell value."""
    if not text:
        return ""

    # Strip whitespace
    text = text.strip()

    # Normalize common unicode characters
    replacements = {
        "\u2212": "-",  # minus sign
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u00b1": "+-",  # plus-minus
        "\u2264": "<=",  # less than or equal
        "\u2265": ">=",  # greater than or equal
        "\u03b1": "alpha",  # Greek alpha
        "\u03b2": "beta",  # Greek beta
        "\u03c7": "chi",  # Greek chi
        "\u2019": "'",  # right single quote
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\xa0": " ",  # non-breaking space
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def table_to_markdown(table: Table, include_caption: bool = True) -> str:
    """
    Render table as Markdown pipe table.

    Args:
        table: Table to render
        include_caption: Whether to include table caption

    Returns:
        Markdown string
    """
    lines = []

    # Caption
    if include_caption and table.caption:
        lines.append(f"[TABLE {table.id.upper()}: {table.caption}]")
        lines.append("")

    # Get column count
    num_cols = len(table.headers) if table.headers else (
        max(len(row) for row in table.rows) if table.rows else 0
    )

    if num_cols == 0:
        return "\n".join(lines) if lines else ""

    # Headers
    if table.headers:
        header_cells = [_escape_markdown(h) for h in table.headers]
        # Pad if needed
        while len(header_cells) < num_cols:
            header_cells.append("")
        lines.append("| " + " | ".join(header_cells) + " |")

        # Separator
        lines.append("| " + " | ".join(["---"] * num_cols) + " |")
    else:
        # No headers - create empty header row
        lines.append("| " + " | ".join([""] * num_cols) + " |")
        lines.append("| " + " | ".join(["---"] * num_cols) + " |")

    # Data rows
    for row in table.rows:
        cells = [_escape_markdown(cell) for cell in row]
        # Pad if needed
        while len(cells) < num_cols:
            cells.append("")
        # Trim if too many
        cells = cells[:num_cols]
        lines.append("| " + " | ".join(cells) + " |")

    # Footnotes
    if table.footnotes:
        lines.append("")
        for footnote in table.footnotes:
            lines.append(f"*{footnote}*")

    return "\n".join(lines)


def _escape_markdown(text: str) -> str:
    """Escape special Markdown characters in table cells."""
    if not text:
        return ""

    # Replace pipe characters (breaks table format)
    text = text.replace("|", "\\|")

    # Replace newlines (breaks table format)
    text = text.replace("\n", " ").replace("\r", "")

    return text


def merge_tables(tables: List[Table]) -> List[Table]:
    """
    Merge tables from different sources, preferring higher quality.

    Args:
        tables: List of tables (may have duplicates from different sources)

    Returns:
        Deduplicated list with best quality version of each table
    """
    # Group by caption similarity
    merged = {}

    for table in tables:
        # Use caption as key (or id if no caption)
        key = (table.caption or "").lower().strip()
        if not key:
            key = table.id

        if key not in merged or table.quality_score > merged[key].quality_score:
            merged[key] = table

    return list(merged.values())
