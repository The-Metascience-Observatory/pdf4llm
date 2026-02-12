"""
OCR fallback extraction for scanned PDFs or poor-quality extractions.

Uses PyMuPDF's built-in OCR capabilities (via Tesseract) when GROBID
produces low-quality results (quality score < 0.6).
"""

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

from ..config import Config
from ..models import (
    DocumentModel, Reference, Table, Section,
    ExtractionMetadata
)

logger = logging.getLogger(__name__)

# Minimum quality score threshold for triggering OCR fallback
OCR_FALLBACK_THRESHOLD = 0.6


def needs_ocr_fallback(quality_score: float) -> bool:
    """Check if quality score warrants OCR fallback."""
    return quality_score < OCR_FALLBACK_THRESHOLD


def extract_with_ocr(pdf_path: Path, config: Optional[Config] = None) -> DocumentModel:
    """
    Extract document content using OCR.
    
    This is a fallback for scanned PDFs or documents where GROBID
    produces poor results. Uses PyMuPDF with Tesseract OCR.
    
    Args:
        pdf_path: Path to PDF file
        config: Optional configuration
        
    Returns:
        DocumentModel with OCR-extracted content
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF (fitz) is required for OCR extraction. "
                         "Install with: pip install pymupdf")
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    
    try:
        full_text = ""
        ocr_pages = 0
        
        for page_num, page in enumerate(doc):
            # First try normal text extraction
            text = page.get_text()
            
            # If page has very little text, try OCR
            if len(text.strip()) < 100:
                try:
                    # Use PyMuPDF's OCR capability
                    # This requires Tesseract to be installed
                    ocr_text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
                    
                    # If still no text, try with OCR flag
                    if len(ocr_text.strip()) < 100:
                        # Get page as image and OCR it
                        pix = page.get_pixmap(dpi=300)
                        ocr_text = _ocr_pixmap(pix)
                        ocr_pages += 1
                    
                    text = ocr_text
                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num}: {e}")
            
            full_text += text + "\n\n"
        
        # Parse the OCR'd text into document structure
        title = _extract_title_ocr(full_text)
        abstract = _extract_abstract_ocr(full_text)
        sections = _extract_sections_ocr(full_text)
        references = _extract_references_ocr(full_text)
        
        # Build metadata
        metadata = ExtractionMetadata(
            source_pdf=pdf_path.name,
            extraction_mode="ocr-fallback",
            tables_extracted=0,  # Tables not extracted in OCR mode
            references_total=len(references),
            references_with_doi=sum(1 for r in references if r.doi),
            warnings=[
                f"OCR fallback used ({ocr_pages} pages OCR'd)",
                "Table extraction not available in OCR mode",
                "Reference DOI extraction is limited",
            ],
        )
        
        # Try to extract DOI from text
        doi = _extract_doi_ocr(full_text)
        
        return DocumentModel(
            metadata=metadata,
            doi=doi,
            title=title,
            abstract=abstract,
            sections=sections,
            tables=[],
            figures=[],
            references=references,
        )
        
    finally:
        doc.close()


def _ocr_pixmap(pix) -> str:
    """
    OCR a PyMuPDF pixmap using Tesseract.
    
    Requires: pip install pytesseract, and Tesseract installed on system.
    """
    try:
        import pytesseract
        from PIL import Image
        import io
        
        # Convert pixmap to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        # Run OCR
        text = pytesseract.image_to_string(img, lang='eng')
        return text
        
    except ImportError:
        logger.warning("pytesseract not installed. Install with: pip install pytesseract")
        return ""
    except Exception as e:
        logger.warning(f"Tesseract OCR failed: {e}")
        return ""


def _extract_title_ocr(text: str) -> str:
    """Extract title from OCR'd text (usually first non-empty line)."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    for line in lines[:10]:  # Check first 10 lines
        # Skip very short lines or lines that look like headers
        if len(line) < 10:
            continue
        if line.upper() == line and len(line) < 50:
            continue  # Likely a header like "RESEARCH ARTICLE"
        if re.match(r'^(volume|vol\.|issue|page|doi|http)', line.lower()):
            continue
        
        # This is likely the title
        return line
    
    return "Untitled"


def _extract_abstract_ocr(text: str) -> str:
    """Extract abstract from OCR'd text."""
    # Look for "Abstract" section
    patterns = [
        r'(?:^|\n)\s*(?:ABSTRACT|Abstract|Summary|SUMMARY)\s*[:\.]?\s*\n(.*?)(?=\n\s*(?:Introduction|INTRODUCTION|Keywords|KEYWORDS|1\.|I\.)|\Z)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            abstract = match.group(1).strip()
            # Clean up OCR artifacts
            abstract = re.sub(r'\s+', ' ', abstract)
            if len(abstract) > 50:
                return abstract
    
    return ""


def _extract_sections_ocr(text: str) -> list:
    """Extract sections from OCR'd text."""
    sections = []
    
    # Split by common section headers
    section_pattern = r'\n\s*(?:(\d+\.?\s+)?([A-Z][A-Za-z\s]+))\s*\n'
    
    parts = re.split(section_pattern, text)
    
    current_title = "Introduction"
    current_content = []
    
    for i, part in enumerate(parts):
        if part and part.strip():
            # Check if this looks like a section header
            if re.match(r'^[A-Z][a-z]+', part.strip()) and len(part.strip()) < 50:
                if current_content:
                    sections.append(Section(
                        title=current_title,
                        content='\n'.join(current_content),
                        level=1,
                    ))
                current_title = part.strip()
                current_content = []
            else:
                current_content.append(part.strip())
    
    # Add last section
    if current_content:
        sections.append(Section(
            title=current_title,
            content='\n'.join(current_content),
            level=1,
        ))
    
    return sections


def _extract_references_ocr(text: str) -> list:
    """Extract references from OCR'd text (limited capability)."""
    references = []
    
    # Find references section
    ref_match = re.search(
        r'(?:^|\n)\s*(?:REFERENCES|References|Bibliography|BIBLIOGRAPHY|Literature Cited)\s*\n(.*)',
        text, re.DOTALL | re.IGNORECASE
    )
    
    if not ref_match:
        return references
    
    ref_text = ref_match.group(1)
    
    # Split by reference numbers or newlines
    ref_entries = re.split(r'\n\s*(?:\[\d+\]|\d+\.)\s*', ref_text)
    
    for i, entry in enumerate(ref_entries):
        entry = entry.strip()
        if len(entry) < 20:
            continue
            
        # Try to extract DOI
        doi_match = re.search(r'10\.\d{4,}/[^\s]+', entry)
        doi = doi_match.group(0).rstrip('.,') if doi_match else None
        
        references.append(Reference(
            num=i + 1,
            title=entry[:200] if len(entry) > 200 else entry,
            doi=doi,
        ))
    
    return references


def _extract_doi_ocr(text: str) -> Optional[str]:
    """Extract DOI from OCR'd text."""
    # Look for DOI patterns
    patterns = [
        r'(?:doi[:\s]+|DOI[:\s]+|https?://doi\.org/)?(10\.\d{4,}/[^\s]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            doi = match.group(1).rstrip('.,;')
            return doi
    
    return None
