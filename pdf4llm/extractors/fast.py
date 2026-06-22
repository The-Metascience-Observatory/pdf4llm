"""
Fast PDF extraction without GROBID.

Uses PyMuPDF for text extraction and pdfplumber for tables.
This mode is faster but cannot reliably extract reference DOIs.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from ..config import Config
from ..models import (
    DocumentModel, Reference, Table, Section, Figure,
    ExtractionMetadata
)
from .tables import assess_table_quality
from .doi_lookup import fetch_title_from_doi

logger = logging.getLogger(__name__)

# Reference-section heading patterns (shared with marker_ext, pymupdf4llm_ext, pipeline)
REF_HEADING_PATTERNS = [
    r"(?im)^\s{0,3}#{1,6}\s*references\s*$",
    r"(?im)^\s{0,3}#{1,6}\s*bibliography\s*$",
    r"(?im)^\s{0,3}#{1,6}\s*works\s+cited\s*$",
]


def extract_fast(pdf_path: Path, config: Optional[Config] = None) -> DocumentModel:
    """
    Extract document content using PyMuPDF and pdfplumber.

    This is the fast extraction mode that doesn't require GROBID.
    Note: Reference DOI extraction is LIMITED in this mode.

    Args:
        pdf_path: Path to PDF file
        config: Optional configuration

    Returns:
        DocumentModel with extracted content
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF (fitz) is required for fast extraction. "
                         "Install with: pip install pymupdf")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Extract text with PyMuPDF
    doc = fitz.open(pdf_path)

    try:
        # Extract full text
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n\n"

        # Extract DOI early (needed for title fallback)
        doi = _extract_doi_from_text(full_text)
        if not doi:
            doi = _extract_doi_from_filename(pdf_path)

        # Extract title (font-based first, then DOI lookup as fallback)
        title = _extract_title_from_text(full_text, doc)
        if not title or title == "Untitled" or len(title) > 200 or _is_bad_title(title):
            doi_title = fetch_title_from_doi(doi)
            if doi_title:
                title = doi_title

        # Extract abstract (regex first, then font-based fallback)
        abstract = _extract_abstract_from_text(full_text)
        if not abstract:
            abstract = _extract_abstract_by_font(doc)

        # Extract sections (pass title to avoid title contamination)
        sections = _extract_sections_from_text(full_text, title=title)

        # Extract tables using pdfplumber
        tables = _extract_tables_pdfplumber(pdf_path)

        # Extract references (limited DOI extraction)
        references = _extract_references_from_text(full_text)

        # Build metadata
        metadata = ExtractionMetadata(
            source_pdf=pdf_path.name,
            extraction_mode="fast",
            tables_extracted=len(tables),
            references_total=len(references),
            references_with_doi=sum(1 for r in references if r.doi),
            warnings=["Fast mode: Reference DOI extraction is limited"] if references else [],
        )

        return DocumentModel(
            metadata=metadata,
            doi=doi,
            title=title,
            abstract=abstract,
            sections=sections,
            tables=tables,
            figures=[],  # Figure extraction not implemented in fast mode
            references=references,
        )

    finally:
        doc.close()


def _extract_title_from_text(text: str, doc) -> str:
    """
    Extract title using font-size heuristics (primary) and multi-line
    text accumulation (fallback).
    """
    # Strategy 1: Font-size based detection from PyMuPDF
    title = _extract_title_by_font_size(doc)
    if title and _validate_title(title):
        return title

    # Strategy 2: Multi-line accumulation from raw text
    title = _extract_title_from_lines(text)
    if title and _validate_title(title):
        return title

    return "Untitled"


# Patterns that indicate a line is a page artifact, not a title
TITLE_ARTIFACT_PATTERNS = [
    r"^Vol\.\s*[:(]",                      # "Vol.:(0123456789)"
    r"^\d{4}\.\d{2}\.\d{2}",              # dates like "2021.03.24"
    r"^Received\s*:",                       # "Received: ..."
    r"^Accepted\s*:",                       # "Accepted: ..."
    r"^Published\s*:",                      # "Published: ..."
    r"^Open\s+Access$",                    # "Open Access"
    r"^https?://",                         # URLs
    r"^doi\s*:",                           # DOI lines
    r"^\d+$",                              # bare numbers
    r"^ISSN",                              # ISSN numbers
    r"^Copyright",                         # copyright notices
    r"^[A-Z][a-z]+\s+\d{4}$",            # "March 2021"
    r"^Research\s+Article$",               # article type labels
    r"^Original\s+Article$",
    r"^Review\s+Article$",
    r"^Case\s+Report$",
    r"^Short\s+Communication$",
    r"^Letter\s+to",
    r"^www\.",                             # website URLs
    r"^\(\d+\)$",                          # bare numbers in parens
    r"^[A-Z]+\s*\d+$",                    # journal codes like "J123" (must fill whole title)
    r"^Page\s+\d+",                        # page numbers
    r"^Cite\s+this",                       # citation prompts
    r"^\d+\s*[-–]\s*\d+$",               # page ranges
    r"^PLOS\s+",                           # PLOS journal headers
    r"^Nature\s+\w+$",                    # Nature journal headers
    r"^Science\s+\w+$",                   # Science journal headers
]
_ARTIFACT_RE = [re.compile(p, re.IGNORECASE) for p in TITLE_ARTIFACT_PATTERNS]

# Lines that indicate we've passed the title area
TITLE_STOP_PATTERNS = [
    r"^abstract\b",
    r"^introduction\b",
    r"^summary\b",
    r"^keywords?\b",
    r"^background\b",
    r"^1\.\s",
    r"^methods?\b",
    r"^materials?\s+and\s+methods",
]
_STOP_RE = [re.compile(p, re.IGNORECASE) for p in TITLE_STOP_PATTERNS]


def _validate_title(title: str) -> bool:
    """Check if a candidate title looks legitimate (not an artifact)."""
    if not title or len(title) < 10:
        return False
    if len(title) > 500:
        return False
    for pattern in _ARTIFACT_RE:
        if pattern.match(title):
            return False
    # Reject titles that are mostly digits/punctuation
    alpha_count = sum(c.isalpha() for c in title)
    if alpha_count / max(len(title), 1) < 0.5:
        return False
    return True


def _extract_title_by_font_size(doc) -> Optional[str]:
    """
    Extract title by finding the largest font text on page 1.

    Uses PyMuPDF's structured text output to identify title spans
    by their font size.
    """
    if len(doc) == 0:
        return None

    # Check first 1-2 pages (some papers have a cover page)
    pages_to_check = min(len(doc), 2)
    best_title = None
    best_font_size = 0

    for page_idx in range(pages_to_check):
        page = doc[page_idx]
        page_height = page.rect.height

        try:
            page_dict = page.get_text("dict")
        except Exception:
            continue

        # Collect spans with font sizes from the top portion of the page
        spans_with_size = []
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:  # text blocks only
                continue
            block_y = block.get("bbox", [0, 0, 0, 0])[1]
            # Only consider top 50% of page for title
            if block_y > page_height * 0.5:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    size = span.get("size", 0)
                    y_pos = span.get("bbox", [0, 0, 0, 0])[1]
                    if text and size > 0:
                        spans_with_size.append((size, text, y_pos, block_y))

        if not spans_with_size:
            continue

        # Get unique font sizes, sorted descending (largest first)
        unique_sizes = sorted(set(s[0] for s in spans_with_size), reverse=True)

        # Try each font size tier, starting from largest
        for target_size in unique_sizes:
            size_tolerance = target_size * 0.1

            # Collect spans at this font size
            title_spans = []
            for size, text, y_pos, block_y in spans_with_size:
                if abs(size - target_size) <= size_tolerance:
                    # Skip artifacts
                    if any(p.match(text) for p in _ARTIFACT_RE):
                        continue
                    title_spans.append((y_pos, text))

            if not title_spans:
                continue

            # Sort by y-position
            title_spans.sort(key=lambda x: x[0])

            # Cluster spatially: only keep spans that are vertically
            # close together (within ~3x the font size = ~3 lines).
            # This avoids picking up running headers or section titles
            # at the same font size but far from the actual title.
            clustered = [title_spans[0]]
            max_gap = target_size * 3
            for span in title_spans[1:]:
                if span[0] - clustered[-1][0] < max_gap:
                    clustered.append(span)
                else:
                    break
            title_spans = clustered

            candidate = " ".join(t[1] for t in title_spans).strip()

            # Clean up: collapse whitespace, remove line-break hyphens
            candidate = re.sub(r"\s+", " ", candidate)
            candidate = re.sub(r"(\w)- (\w)", r"\1\2", candidate)

            if _validate_title(candidate) and len(candidate) > len(best_title or ""):
                best_title = candidate
                best_font_size = target_size
                break  # found a valid title at this font size, stop

    return best_title


def _extract_title_from_lines(text: str) -> Optional[str]:
    """
    Extract title by accumulating consecutive non-blank lines from the
    top of the document. Handles multi-line titles.
    """
    lines = text.strip().split("\n")
    title_lines = []
    started = False

    for line in lines[:30]:  # only check first 30 lines
        stripped = line.strip()

        # Skip empty lines before title starts
        if not stripped:
            if started:
                break  # blank line after title lines = end of title
            continue

        # Stop if we hit a section header or metadata boundary
        if any(p.match(stripped) for p in _STOP_RE):
            break

        # Skip artifact lines
        if any(p.match(stripped) for p in _ARTIFACT_RE):
            continue

        # Skip very short lines that look like metadata (page numbers, etc.)
        if len(stripped) < 5:
            continue

        # Skip lines that look like author names (contain commas and are
        # mostly proper nouns, or have email-like patterns)
        if started and ("@" in stripped or re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5}(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})+", stripped)):
            break

        title_lines.append(stripped)
        started = True

        # If we've accumulated a reasonable title, check if next line
        # looks like it continues or is something else
        if len(" ".join(title_lines)) > 200:
            break

    if not title_lines:
        return None

    title = " ".join(title_lines)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _extract_abstract_from_text(text: str) -> Optional[str]:
    """
    Extract abstract section from text using multiple strategies.

    Tries various formatting patterns found in academic papers:
    - "Abstract" on its own line followed by text
    - "Abstract:" inline with content
    - "ABSTRACT" in all caps
    - "Summary" as alternative label
    """
    # End boundary: common sections that follow the abstract
    end_boundary = r"(?:introduction|background|keywords?|key\s*words|1\.\s|methods?|materials?\s+and\s+methods|objectives?|aims?\b|highlights?)"

    patterns = [
        # "Abstract" on its own line, text follows on subsequent lines
        rf"(?im)^abstract\s*$\n+(.*?)(?=\n\s*(?:{end_boundary}))",
        # "ABSTRACT" in all caps on its own line
        rf"(?m)^ABSTRACT\s*$\n+(.*?)(?=\n\s*(?:{end_boundary}|INTRODUCTION|BACKGROUND|KEYWORDS|1\.))",
        # "Abstract:" or "Abstract —" inline, content on same/next lines
        rf"(?i)abstract\s*[:—–-]\s*(.*?)(?=\n\s*(?:{end_boundary}))",
        # "Abstract" followed immediately by text on the same line
        r"(?im)^abstract\s+(.{50,?})(?=\n\s*$)",
        # "Summary" variant
        rf"(?im)^summary\s*$\n+(.*?)(?=\n\s*(?:{end_boundary}))",
        rf"(?i)summary\s*[:—–-]\s*(.*?)(?=\n\s*(?:{end_boundary}))",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            # Clean up: collapse whitespace but preserve paragraph breaks
            abstract = re.sub(r"[ \t]+", " ", abstract)
            abstract = re.sub(r"\n{3,}", "\n\n", abstract)
            # Validate length
            if 50 < len(abstract) < 3000:
                return abstract

    return None


def _extract_abstract_by_font(doc) -> Optional[str]:
    """
    Extract abstract using font-size/style heuristics on page 1.

    Fallback for journals like PNAS where the abstract has no "Abstract" label
    but uses a distinct bold font that differs from both the title and body text.
    """
    if len(doc) == 0:
        return None

    page = doc[0]
    blocks = page.get_text('dict')['blocks']

    # Find the largest font size on the page (title)
    max_font_size = 0
    for block in blocks:
        if 'lines' not in block:
            continue
        for line in block['lines']:
            for span in line['spans']:
                if span['size'] > max_font_size:
                    max_font_size = span['size']

    if max_font_size == 0:
        return None

    # Look for a multi-line bold text block that:
    # - Has at least 3 lines (abstracts are multi-line)
    # - Uses a bold font (name contains 'Bold', '.B', or 'bold')
    # - Is smaller than the title font
    # - Contains 50-3000 chars (reasonable abstract length)
    for block in blocks:
        if 'lines' not in block or len(block['lines']) < 3:
            continue

        first_span = block['lines'][0]['spans'][0]
        font_name = first_span['font']
        font_size = first_span['size']

        # Must be bold and smaller than title
        is_bold = ('Bold' in font_name or '.B' in font_name
                   or 'bold' in font_name.lower())
        is_smaller_than_title = font_size < max_font_size - 2

        if not (is_bold and is_smaller_than_title):
            continue

        # Verify that the majority of lines are bold
        bold_line_count = 0
        for line in block['lines']:
            if any('Bold' in s['font'] or '.B' in s['font']
                   or 'bold' in s['font'].lower()
                   for s in line['spans']):
                bold_line_count += 1

        if bold_line_count < len(block['lines']) * 0.8:
            continue

        # Build text from all spans
        text = ' '.join(
            span['text']
            for line in block['lines']
            for span in line['spans']
        ).strip()

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Validate length
        if 50 < len(text) < 3000:
            return text

    return None


def _is_bad_title(title: str) -> bool:
    """Detect titles that are clearly not real paper titles."""
    t = title.lower()
    # URLs or download watermarks
    if "http" in t or "downloaded from" in t:
        return True
    # Repeated words (e.g. "Study Study Study Study")
    words = title.split()
    if len(words) >= 4:
        from collections import Counter
        counts = Counter(words)
        most_common_count = counts.most_common(1)[0][1]
        if most_common_count >= len(words) * 0.5:
            return True
    return False



def _extract_sections_from_text(text: str, title: str = "") -> List[Section]:
    """
    Extract sections from document text using strict header detection.

    Only recognizes lines as section headers if they:
    - Have a numbered prefix (1., 2.1, etc.), OR
    - Match a known academic section name, OR
    - Are ALL CAPS (common journal formatting)

    The extracted title is used to skip lines that are title fragments.
    """
    sections = []
    current_section = None
    current_content = []

    # Normalize title for substring matching
    title_lower = title.lower().strip() if title else ""

    for line in text.split("\n"):
        stripped = line.strip()

        # Skip lines that are part of the title
        if title_lower and stripped and _is_title_fragment(stripped, title_lower):
            continue

        # Check if line is a section header
        is_header, heading, level = _detect_section_header(stripped)

        if is_header:
            # Save previous section
            if current_section:
                current_section.content = "\n".join(current_content).strip()
                sections.append(current_section)

            current_section = Section(
                heading=heading,
                level=level,
                content="",
            )
            current_content = []
        elif current_section:
            current_content.append(stripped)

    # Save last section
    if current_section:
        current_section.content = "\n".join(current_content).strip()
        sections.append(current_section)

    return sections


# Known academic section names (lowercase, for matching)
KNOWN_SECTIONS = {
    "abstract", "introduction", "background", "methods", "methodology",
    "materials and methods", "materials & methods", "results", "discussion",
    "conclusions", "conclusion", "acknowledgments", "acknowledgements",
    "references", "bibliography", "supplementary", "supplementary materials",
    "appendix", "funding", "conflict of interest", "conflicts of interest",
    "competing interests", "data availability", "data availability statement",
    "ethics", "ethics statement", "study design", "statistical analysis",
    "limitations", "results and discussion", "experimental",
    "experimental procedures", "procedures", "literature review",
    "related work", "future work", "abbreviations", "declarations",
    "author contributions", "additional information",
}


def _is_title_fragment(line: str, title_lower: str) -> bool:
    """Check if a line is a fragment of the document title."""
    line_lower = line.lower().strip()
    # Line is contained in the title
    if line_lower in title_lower:
        return True
    # Title is contained in the line (less common but possible)
    if len(title_lower) > 20 and title_lower in line_lower:
        return True
    return False


def _detect_section_header(line: str) -> tuple:
    """
    Detect if a line is a section header.

    Returns: (is_header: bool, heading_text: str, level: int)
    """
    if not line or len(line) > 100:
        return False, "", 0

    # Pattern 1: Numbered header (e.g., "1. Introduction", "2.1 Methods", "3.2.1 Analysis")
    # Section numbers use 1-3 digit segments (not 6-digit grant numbers, etc.)
    match = re.match(r"^(\d{1,3}(?:\.\d{1,3})*)\s*\.?\s+(.+)$", line)
    if match:
        numbering = match.group(1)
        heading = match.group(2).strip()
        # Additional check: heading text should look like a heading, not body text
        # Reject if heading starts with lowercase or is very long
        if len(heading) < 80 and (heading[0].isupper() or heading[0].isdigit()):
            level = min(numbering.count(".") + 1, 6)
            return True, heading, level

    # Pattern 2: Known section name (exact or close match)
    clean = line.strip().rstrip(":").lower()
    if clean in KNOWN_SECTIONS:
        return True, line.strip().rstrip(":"), 1

    # Pattern 3: ALL CAPS heading (e.g., "INTRODUCTION", "RESULTS AND DISCUSSION")
    # Must be >3 chars, mostly letters, and all uppercase
    if (len(line) > 3 and line == line.upper()
            and sum(c.isalpha() for c in line) / max(len(line), 1) > 0.7):
        # Also check against known sections (case-insensitive)
        if clean in KNOWN_SECTIONS:
            return True, line.strip().title(), 1
        # Accept ALL CAPS lines that look like headers (short, no punctuation except spaces)
        if len(line) < 60 and re.match(r"^[A-Z\s]+$", line):
            return True, line.strip().title(), 1

    return False, "", 0


def _extract_tables_pdfplumber(pdf_path: Path) -> List[Table]:
    """Extract tables using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed, skipping table extraction")
        return []

    tables = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_tables = page.extract_tables()

                for idx, table_data in enumerate(page_tables):
                    if not table_data or len(table_data) < 2:
                        continue

                    # First row as headers
                    headers = [str(cell or "").strip() for cell in table_data[0]]

                    # Rest as rows
                    rows = []
                    for row in table_data[1:]:
                        row_data = [str(cell or "").strip() for cell in row]
                        if any(cell for cell in row_data):  # Skip empty rows
                            rows.append(row_data)

                    if headers or rows:
                        table = Table(
                            id=f"tab{len(tables) + 1}",
                            caption=None,  # pdfplumber doesn't extract captions
                            headers=headers,
                            rows=rows,
                            page_number=page_num,
                            source="pdfplumber",
                        )
                        table.quality_score = assess_table_quality(table)
                        tables.append(table)

    except Exception as e:
        logger.warning(f"pdfplumber table extraction failed: {e}")

    return tables


def _extract_references_from_text(text: str) -> List[Reference]:
    """
    Extract references from text (limited DOI extraction).

    This is best-effort and won't be as accurate as GROBID.
    """
    references = []

    # Find references section
    ref_pattern = r"(?i)(?:references|bibliography|works cited)\s*\n+(.*?)(?=\n\s*(?:appendix|supplementary|acknowledgment)|$)"
    match = re.search(ref_pattern, text, re.DOTALL)

    if not match:
        return references

    ref_text = match.group(1)

    # Split into individual references
    # Common patterns: numbered [1], 1., (1), or just line breaks
    ref_entries = re.split(r"\n\s*(?:\[\d+\]|\d+\.\s|\(\d+\))", ref_text)

    for idx, entry in enumerate(ref_entries, 1):
        entry = entry.strip()
        if len(entry) < 20:
            continue

        # Try to extract DOI
        doi = _extract_doi_from_reference(entry)

        # Try to extract year
        year = None
        year_match = re.search(r"\b(19|20)\d{2}\b", entry)
        if year_match:
            year = int(year_match.group())

        # Try to extract title (usually in quotes or after authors)
        title = _extract_title_from_reference(entry)

        references.append(Reference(
            num=idx,
            doi=doi,
            title=title,
            year=year,
            raw_text=entry[:500],  # Keep raw text for fallback
        ))

    return references


def _extract_doi_from_text(text: str) -> Optional[str]:
    """
    Extract document DOI from text.

    Searches the full document (not just the beginning) because DOIs
    appear in headers, footers, colophons, and copyright notices.
    Prefers DOIs found earlier in the document.
    """
    patterns = [
        r"(?i)doi[:\s]+\s*(10\.\d{4,}/[^\s,;\"'<>\]]+)",
        r"(?i)https?://(?:dx\.)?doi\.org/(10\.\d{4,}/[^\s,;\"'<>\]]+)",
        r"(?i)digital\s+object\s+identifier[:\s]+(10\.\d{4,}/[^\s,;\"'<>\]]+)",
    ]

    candidates = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            doi = match.group(1).rstrip(".,;:)")
            position = match.start()
            candidates.append((doi, position))

    if not candidates:
        return None

    # Prefer DOIs found earlier in the document (more likely the paper's own DOI)
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def _extract_doi_from_filename(pdf_path: Path) -> Optional[str]:
    """
    Extract DOI from filename convention: 10.XXXX--suffix.pdf

    Many pipelines encode the DOI in the filename by replacing ALL '/' with
    '--'. Multi-segment DOIs (e.g. `10.1093/rheumatology/kew394`) therefore
    appear in filenames as `10.1093--rheumatology--kew394.pdf`, and every
    `--` must be replaced with `/` on the way back — not just the first one.

    Examples:
      - 10.1002--jnr.22753.pdf               -> 10.1002/jnr.22753
      - 10.1093--rheumatology--kew394.pdf    -> 10.1093/rheumatology/kew394
      - 110.1002--smj.2585.pdf               -> 10.1002/smj.2585 (spurious leading digit)
    """
    stem = pdf_path.stem
    # Match leading DOI prefix (optionally after spurious leading digits),
    # then replace ALL remaining `--` in the suffix with `/`.
    match = re.match(r"^\d*(10\.\d{4,})--(.+)$", stem)
    if match:
        suffix = match.group(2).replace("--", "/")
        return f"{match.group(1)}/{suffix}"
    return None


def _extract_doi_from_reference(text: str) -> Optional[str]:
    """Extract DOI from a reference entry.

    NOTE: earlier versions of this function used `[10]\\.` (a character class
    matching a single char from {1, 0}) instead of the literal `10\\.`. That
    was a silent bug — no real DOI starts with just "1." or "0.", so the
    function returned None on every input. Fixed to use the literal "10."
    prefix like `_extract_doi_from_text` does.
    """
    patterns = [
        r"(?i)doi[:\s]+\s*(10\.\d{4,}/[^\s,;\"'<>\]]+)",
        r"(?i)https?://(?:dx\.)?doi\.org/(10\.\d{4,}/[^\s,;\"'<>\]]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            doi = match.group(1).rstrip(".,;:)")
            return doi

    return None


def _extract_title_from_reference(text: str) -> Optional[str]:
    """Extract title from reference entry."""
    # Look for text in quotes
    match = re.search(r'["\u201c]([^"\u201d]+)["\u201d]', text)
    if match:
        return match.group(1).strip()

    # Look for text after year
    match = re.search(r"\b(?:19|20)\d{2}\b[.\s]+([^.]+\.)", text)
    if match:
        title = match.group(1).strip()
        if len(title) > 10:
            return title

    return None
