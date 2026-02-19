"""
GROBID-based PDF extraction.

This module handles:
1. Sending PDFs to GROBID server
2. Parsing TEI XML response
3. Extracting references with DOIs (critical for metascience)
4. Extracting tables from TEI
"""

import logging
import re
import time
from pathlib import Path
from typing import Optional, List, Tuple
from lxml import etree
import requests

from ..config import Config
from ..models import (
    DocumentModel, Reference, Table, Section, Figure,
    ExtractionMetadata
)

logger = logging.getLogger(__name__)

# TEI namespace
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


class GrobidError(Exception):
    """Exception raised for GROBID-related errors."""
    pass


class GrobidScannedPdfError(GrobidError):
    """Raised when GROBID detects a scanned PDF with no extractable text blocks.

    This typically means the PDF contains only images (no text layer) and
    requires OCR to extract content.
    """
    pass


class GrobidClient:
    """Client for interacting with GROBID server."""

    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.grobid_url.rstrip("/")

    def is_alive(self, timeout: int = 300) -> bool:
        """
        Check if GROBID server is running.

        Note: GROBID loads models lazily on first request, which can take
        several minutes. Default timeout is 300s (5 minutes) to accommodate this.
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/isalive",
                timeout=timeout
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def process_citation(self, raw_citation: str, timeout: int = 15) -> Optional["Reference"]:
        """
        Parse a single raw citation string using GROBID /api/processCitation.

        Returns a partially-populated Reference (num=0) or None on failure.
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/processCitation",
                data={"citations": raw_citation, "consolidateCitations": "0"},
                timeout=timeout,
            )
            if response.status_code == 200 and response.text.strip():
                return _parse_citation_tei_fragment(response.text)
        except requests.RequestException as e:
            logger.debug(f"GROBID processCitation failed: {e}")
        return None

    def get_version(self, timeout: int = 60) -> Optional[str]:
        """Get GROBID version."""
        try:
            response = requests.get(
                f"{self.base_url}/api/version",
                timeout=timeout
            )
            if response.status_code == 200:
                return response.text.strip()
        except requests.RequestException:
            pass
        return None

    def process_fulltext(self, pdf_path: Path) -> str:
        """
        Send PDF to GROBID and get TEI XML response.

        Args:
            pdf_path: Path to PDF file

        Returns:
            TEI XML string

        Raises:
            GrobidError: If GROBID processing fails
        """
        if not pdf_path.exists():
            raise GrobidError(f"PDF file not found: {pdf_path}")

        url = f"{self.base_url}/api/processFulltextDocument"

        last_error = None
        for attempt in range(self.config.grobid_retry_count):
            try:
                with open(pdf_path, "rb") as pdf_file:
                    response = requests.post(
                        url,
                        files={"input": (pdf_path.name, pdf_file, "application/pdf")},
                        data={
                            "consolidateHeader": "1",
                            "consolidateCitations": "0",
                            "includeRawCitations": "1",
                            "teiCoordinates": "persName,figure,ref,formula,biblStruct,s,head",
                        },
                        timeout=self.config.grobid_timeout
                    )

                if response.status_code == 200:
                    return response.text
                elif response.status_code == 503:
                    # Server busy, wait and retry
                    logger.warning(f"GROBID busy, retrying in {self.config.grobid_retry_delay}s...")
                    time.sleep(self.config.grobid_retry_delay * (attempt + 1))
                    continue
                else:
                    error_text = response.text[:300]
                    if "[NO_BLOCKS]" in response.text:
                        raise GrobidScannedPdfError(
                            f"GROBID returned status {response.status_code}: {error_text}"
                        )
                    raise GrobidError(
                        f"GROBID returned status {response.status_code}: {error_text}"
                    )

            except requests.Timeout:
                last_error = GrobidError(f"GROBID timeout after {self.config.grobid_timeout}s")
                logger.warning(f"GROBID timeout, attempt {attempt + 1}/{self.config.grobid_retry_count}")
            except requests.ConnectionError:
                # GROBID process has died - likely out of memory
                raise GrobidError(
                    "GROBID server has crashed (connection refused). "
                    "This usually means GROBID ran out of memory. "
                    "Try reducing --workers (e.g. --workers 2) or restarting GROBID with more memory (-Xmx8g)."
                )
            except requests.RequestException as e:
                last_error = GrobidError(f"GROBID request failed: {e}")
                logger.warning(f"GROBID request error: {e}")

            if attempt < self.config.grobid_retry_count - 1:
                time.sleep(self.config.grobid_retry_delay)

        raise last_error or GrobidError("GROBID processing failed after retries")


def extract_with_grobid(
    pdf_path: Path,
    config: Config,
    client: Optional["GrobidClient"] = None,
    grobid_version: Optional[str] = None,
) -> Tuple[str, DocumentModel]:
    """
    Extract document content using GROBID.

    Args:
        pdf_path: Path to PDF file
        config: Configuration
        client: Optional pre-created GrobidClient (avoids per-file health checks)
        grobid_version: Optional cached GROBID version string

    Returns:
        Tuple of (TEI XML string, DocumentModel)
    """
    if client is None:
        client = GrobidClient(config)
        if not client.is_alive():
            raise GrobidError(f"GROBID server not available at {config.grobid_url}")
        grobid_version = client.get_version()

    tei_xml = client.process_fulltext(pdf_path)

    document = parse_tei_xml(tei_xml, pdf_path.name, grobid_version)

    return tei_xml, document


def parse_tei_xml(
    tei_xml: str,
    source_pdf: str = "unknown.pdf",
    grobid_version: Optional[str] = None
) -> DocumentModel:
    """
    Parse GROBID TEI XML into DocumentModel.

    Args:
        tei_xml: TEI XML string from GROBID
        source_pdf: Source PDF filename
        grobid_version: GROBID version string

    Returns:
        DocumentModel with extracted content
    """
    try:
        root = etree.fromstring(tei_xml.encode("utf-8"))
    except etree.XMLSyntaxError as e:
        raise GrobidError(f"Invalid TEI XML: {e}")

    # Extract document DOI
    doi = _extract_document_doi(root)

    # Extract title
    title = _extract_title(root)

    # Extract references FIRST to build citation map (CRITICAL for DOI extraction)
    references, ref_id_map = _extract_references(root)

    # Extract abstract (with citation-aware text extraction)
    abstract = _extract_abstract(root, ref_id_map)

    # Extract sections (with citation-aware text extraction)
    sections = _extract_sections(root, ref_id_map)

    # Extract tables
    tables = _extract_tables(root)

    # Extract figures
    figures = _extract_figures(root)

    # Build metadata
    metadata = ExtractionMetadata(
        source_pdf=source_pdf,
        extraction_mode="full-grobid",
        grobid_version=grobid_version,
        tables_extracted=len(tables),
        references_total=len(references),
        references_with_doi=sum(1 for r in references if r.doi),
    )

    return DocumentModel(
        metadata=metadata,
        doi=doi,
        title=title,
        abstract=abstract,
        sections=sections,
        tables=tables,
        figures=figures,
        references=references,
    )


def _extract_document_doi(root: etree._Element) -> Optional[str]:
    """Extract document DOI from TEI header."""
    # Try idno in biblStruct (most reliable)
    doi_elem = root.find(".//tei:teiHeader//tei:idno[@type='DOI']", TEI_NS)
    if doi_elem is not None and doi_elem.text:
        return _normalize_doi(doi_elem.text)

    # Try sourceDesc
    doi_elem = root.find(".//tei:sourceDesc//tei:idno[@type='DOI']", TEI_NS)
    if doi_elem is not None and doi_elem.text:
        return _normalize_doi(doi_elem.text)

    return None


def _extract_title(root: etree._Element) -> str:
    """Extract document title."""
    title_elem = root.find(".//tei:titleStmt/tei:title", TEI_NS)
    if title_elem is not None:
        return _get_text_content(title_elem).strip()

    # Fallback to analytic title
    title_elem = root.find(".//tei:analytic/tei:title", TEI_NS)
    if title_elem is not None:
        return _get_text_content(title_elem).strip()

    return "Untitled"


def _extract_abstract(root: etree._Element, ref_id_map: Optional[dict] = None) -> Optional[str]:
    """Extract document abstract with citation-aware text extraction."""
    abstract_elem = root.find(".//tei:abstract", TEI_NS)
    if abstract_elem is not None:
        paragraphs = []
        for p in abstract_elem.findall(".//tei:p", TEI_NS):
            if ref_id_map:
                text = _get_text_with_citations(p, ref_id_map).strip()
            else:
                text = _get_text_content(p).strip()
            if text:
                paragraphs.append(text)
        if paragraphs:
            return "\n\n".join(paragraphs)

        # Try direct text content
        if ref_id_map:
            text = _get_text_with_citations(abstract_elem, ref_id_map).strip()
        else:
            text = _get_text_content(abstract_elem).strip()
        if text:
            return text

    return None


def _extract_sections(root: etree._Element, ref_id_map: Optional[dict] = None) -> List[Section]:
    """Extract document sections from body with citation-aware text extraction."""
    body = root.find(".//tei:body", TEI_NS)
    if body is None:
        return []

    return _parse_div_recursive(body, ref_id_map=ref_id_map, level=1)


def _parse_div_recursive(parent: etree._Element, ref_id_map: Optional[dict] = None, level: int = 1) -> List[Section]:
    """Recursively parse div elements into sections."""
    sections = []

    for div in parent.findall("./tei:div", TEI_NS):
        # Get heading (headings don't need citation replacement)
        head = div.find("./tei:head", TEI_NS)
        heading = _get_text_content(head).strip() if head is not None else None

        # Determine level from heading text patterns (e.g., "1.", "1.1", etc.)
        detected_level = _detect_heading_level(heading) if heading else level
        actual_level = min(detected_level, 6)

        # Get paragraph content (citation-aware)
        paragraphs = []
        for p in div.findall("./tei:p", TEI_NS):
            if ref_id_map:
                text = _get_text_with_citations(p, ref_id_map).strip()
            else:
                text = _get_text_content(p).strip()
            if text:
                paragraphs.append(text)

        content = "\n\n".join(paragraphs)

        # Recursively get subsections
        subsections = _parse_div_recursive(div, ref_id_map=ref_id_map, level=actual_level + 1)

        if heading or content or subsections:
            sections.append(Section(
                heading=heading,
                level=actual_level,
                content=content,
                subsections=subsections,
            ))

    return sections


def _detect_heading_level(heading: str) -> int:
    """Detect heading level from numbering pattern."""
    if not heading:
        return 1

    # Count dots in numbering prefix
    match = re.match(r"^(\d+(?:\.\d+)*)\s*\.?\s*", heading)
    if match:
        numbering = match.group(1)
        dots = numbering.count(".")
        return min(dots + 1, 6)

    return 1


def _extract_tables(root: etree._Element) -> List[Table]:
    """Extract tables from TEI."""
    tables = []

    for idx, fig in enumerate(root.findall(".//tei:figure[@type='table']", TEI_NS)):
        table_id = fig.get("{http://www.w3.org/XML/1998/namespace}id", f"tab{idx + 1}")

        # Get caption
        head = fig.find("./tei:head", TEI_NS)
        caption = _get_text_content(head).strip() if head is not None else None

        # Get table element
        table_elem = fig.find("./tei:table", TEI_NS)

        headers = []
        rows = []

        if table_elem is not None:
            # Extract rows
            for row_elem in table_elem.findall("./tei:row", TEI_NS):
                role = row_elem.get("role", "")
                cells = []

                for cell in row_elem.findall("./tei:cell", TEI_NS):
                    cell_text = _get_text_content(cell).strip()
                    cells.append(cell_text)

                if role == "head" or row_elem == table_elem.find("./tei:row", TEI_NS):
                    if not headers:  # First header row
                        headers = cells
                else:
                    if cells:
                        rows.append(cells)

        # Get footnotes
        footnotes = []
        for note in fig.findall(".//tei:note", TEI_NS):
            note_text = _get_text_content(note).strip()
            if note_text:
                footnotes.append(note_text)

        tables.append(Table(
            id=table_id,
            caption=caption,
            headers=headers,
            rows=rows,
            footnotes=footnotes,
            source="grobid",
        ))

    return tables


def _extract_figures(root: etree._Element) -> List[Figure]:
    """Extract figure metadata from TEI."""
    figures = []

    for idx, fig in enumerate(root.findall(".//tei:figure", TEI_NS)):
        # Skip tables (they have type="table")
        if fig.get("type") == "table":
            continue

        fig_id = fig.get("{http://www.w3.org/XML/1998/namespace}id", f"fig{idx + 1}")

        # Get caption
        head = fig.find("./tei:head", TEI_NS)
        figdesc = fig.find("./tei:figDesc", TEI_NS)

        caption_parts = []
        if head is not None:
            caption_parts.append(_get_text_content(head).strip())
        if figdesc is not None:
            caption_parts.append(_get_text_content(figdesc).strip())

        caption = " ".join(filter(None, caption_parts)) or None

        figures.append(Figure(
            id=fig_id,
            caption=caption,
        ))

    return figures


def _extract_references(root: etree._Element) -> Tuple[List[Reference], dict]:
    """
    Extract references with DOIs from TEI.

    This is CRITICAL for metascience - we need DOIs to find original studies.

    Returns:
        Tuple of (references list, ref_id_map) where ref_id_map maps
        GROBID xml:id (e.g. "b0") to 1-indexed reference number.
    """
    references = []
    ref_id_map: dict = {}  # xml:id -> 1-indexed ref number

    listbibl = root.find(".//tei:listBibl", TEI_NS)
    if listbibl is None:
        return references, ref_id_map

    for idx, biblstruct in enumerate(listbibl.findall("./tei:biblStruct", TEI_NS)):
        ref_num = idx + 1

        # Map GROBID's xml:id to our 1-indexed number
        xml_id = biblstruct.get("{http://www.w3.org/XML/1998/namespace}id")
        if xml_id:
            ref_id_map[xml_id] = ref_num

        # Extract DOI (CRITICAL)
        doi = _extract_reference_doi(biblstruct)

        # Extract title
        title = _extract_reference_title(biblstruct)

        # Extract authors
        authors = _extract_reference_authors(biblstruct)

        # Extract journal
        journal = _extract_reference_journal(biblstruct)

        # Extract year
        year = _extract_reference_year(biblstruct)

        # Extract volume and pages
        volume = _extract_bibl_scope(biblstruct, "volume")
        issue = _extract_bibl_scope(biblstruct, "issue")
        pages = _extract_bibl_scope(biblstruct, "page")

        # Get raw text as fallback
        raw_text = None
        note = biblstruct.find(".//tei:note[@type='raw_reference']", TEI_NS)
        if note is not None:
            raw_text = _get_text_content(note).strip()

        references.append(Reference(
            num=ref_num,
            doi=doi,
            title=title,
            authors=authors,
            journal=journal,
            year=year,
            volume=volume,
            issue=issue,
            pages=pages,
            raw_text=raw_text,
        ))

    return references, ref_id_map


def _extract_reference_doi(biblstruct: etree._Element) -> Optional[str]:
    """
    Extract DOI from a biblStruct element.

    Tries multiple locations where GROBID places DOIs.
    """
    # Primary: idno type="DOI"
    doi_elem = biblstruct.find(".//tei:idno[@type='DOI']", TEI_NS)
    if doi_elem is not None and doi_elem.text:
        return _normalize_doi(doi_elem.text)

    # Secondary: ptr element with doi.org URL
    ptr = biblstruct.find(".//tei:ptr", TEI_NS)
    if ptr is not None:
        target = ptr.get("target", "")
        if "doi.org/" in target:
            return _normalize_doi(target.split("doi.org/")[-1])

    # Tertiary: Check ref element
    ref = biblstruct.find(".//tei:ref[@type='url']", TEI_NS)
    if ref is not None:
        target = ref.get("target", "") or _get_text_content(ref)
        if "doi.org/" in target:
            return _normalize_doi(target.split("doi.org/")[-1])

    return None


def _normalize_doi(doi: str) -> str:
    """Normalize DOI string."""
    doi = doi.strip()
    # Remove common prefixes
    for prefix in ["https://doi.org/", "http://doi.org/", "doi:", "DOI:"]:
        if doi.lower().startswith(prefix.lower()):
            doi = doi[len(prefix):]
    return doi.strip()


def _extract_reference_title(biblstruct: etree._Element) -> Optional[str]:
    """Extract reference title."""
    # Try analytic title (article title)
    title = biblstruct.find(".//tei:analytic/tei:title", TEI_NS)
    if title is not None:
        text = _get_text_content(title).strip()
        if text:
            return text

    # Try monogr title (book/journal title as fallback)
    title = biblstruct.find(".//tei:monogr/tei:title", TEI_NS)
    if title is not None:
        text = _get_text_content(title).strip()
        if text:
            return text

    return None


def _extract_reference_authors(biblstruct: etree._Element) -> List[str]:
    """Extract reference authors."""
    authors = []

    for author in biblstruct.findall(".//tei:author", TEI_NS):
        persname = author.find("./tei:persName", TEI_NS)
        if persname is not None:
            forename = persname.find("./tei:forename", TEI_NS)
            surname = persname.find("./tei:surname", TEI_NS)

            parts = []
            if surname is not None and surname.text:
                parts.append(surname.text.strip())
            if forename is not None and forename.text:
                # Add initial
                parts.append(forename.text.strip()[0] if forename.text.strip() else "")

            if parts:
                authors.append(" ".join(parts))

    return authors


def _extract_reference_journal(biblstruct: etree._Element) -> Optional[str]:
    """Extract reference journal name."""
    title = biblstruct.find(".//tei:monogr/tei:title[@level='j']", TEI_NS)
    if title is not None:
        return _get_text_content(title).strip() or None

    # Try any monogr title
    title = biblstruct.find(".//tei:monogr/tei:title", TEI_NS)
    if title is not None:
        return _get_text_content(title).strip() or None

    return None


def _extract_reference_year(biblstruct: etree._Element) -> Optional[int]:
    """Extract reference publication year."""
    date = biblstruct.find(".//tei:date[@type='published']", TEI_NS)
    if date is None:
        date = biblstruct.find(".//tei:date", TEI_NS)

    if date is not None:
        when = date.get("when", "")
        if when:
            # Extract year from date string
            match = re.match(r"(\d{4})", when)
            if match:
                return int(match.group(1))

        # Try text content
        text = _get_text_content(date).strip()
        match = re.search(r"(\d{4})", text)
        if match:
            return int(match.group(1))

    return None


def _extract_bibl_scope(biblstruct: etree._Element, unit: str) -> Optional[str]:
    """Extract biblScope value by unit type."""
    scope = biblstruct.find(f".//tei:biblScope[@unit='{unit}']", TEI_NS)
    if scope is not None:
        # Check 'from' and 'to' attributes for ranges
        from_val = scope.get("from", "")
        to_val = scope.get("to", "")
        if from_val and to_val:
            return f"{from_val}-{to_val}"
        if from_val:
            return from_val
        return _get_text_content(scope).strip() or None
    return None


def _get_text_content(elem: etree._Element) -> str:
    """Get all text content from an element, including nested elements."""
    if elem is None:
        return ""
    return "".join(elem.itertext())


_TEI_REF_TAG = f"{{{TEI_NS['tei']}}}ref"


def _parse_citation_tei_fragment(tei_text: str) -> Optional[Reference]:
    """
    Parse a TEI fragment returned by GROBID /api/processCitation.

    The response is a full TEI document or a bare <biblStruct> element.
    Reuses the same extraction helpers used for full-document references.
    """
    try:
        root = etree.fromstring(tei_text.encode("utf-8"))
    except etree.XMLSyntaxError:
        return None

    # Find biblStruct — try with namespace first, then without
    biblstruct = root.find(".//tei:biblStruct", TEI_NS)
    if biblstruct is None:
        biblstruct = root.find(".//biblStruct")
    if biblstruct is None:
        # Maybe root itself is the biblStruct
        local = root.tag.split("}")[-1] if "}" in root.tag else root.tag
        if local == "biblStruct":
            biblstruct = root
    if biblstruct is None:
        return None

    from ..models import Reference as _Ref
    return _Ref(
        num=0,  # caller sets the real number
        doi=_extract_reference_doi(biblstruct),
        title=_extract_reference_title(biblstruct),
        authors=_extract_reference_authors(biblstruct),
        journal=_extract_reference_journal(biblstruct),
        year=_extract_reference_year(biblstruct),
        volume=_extract_bibl_scope(biblstruct, "volume"),
        issue=_extract_bibl_scope(biblstruct, "issue"),
        pages=_extract_bibl_scope(biblstruct, "page"),
    )


def _get_text_with_citations(elem: etree._Element, ref_id_map: dict) -> str:
    """
    Get text content, replacing <ref type="bibr"> elements with normalized [N] markers.

    Only replaces citations whose target maps to a known reference.
    Falls back to the original element text when unmapped.
    """
    if elem is None:
        return ""

    parts = []
    if elem.text:
        parts.append(elem.text)

    for child in elem:
        if child.tag == _TEI_REF_TAG and child.get("type") == "bibr":
            target = child.get("target", "")
            if target.startswith("#"):
                xml_id = target[1:]
                ref_num = ref_id_map.get(xml_id)
                if ref_num is not None:
                    parts.append(f"[{ref_num}]")
                else:
                    parts.append(_get_text_content(child))
            else:
                parts.append(_get_text_content(child))
        else:
            # Recurse for non-citation elements
            parts.append(_get_text_with_citations(child, ref_id_map))
        if child.tail:
            parts.append(child.tail)

    return "".join(parts)
