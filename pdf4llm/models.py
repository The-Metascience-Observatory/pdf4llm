"""
Pydantic data models for the PDF-to-Markdown pipeline.

These models represent the structured output from PDF extraction,
optimized for LLM consumption in metascience research.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class Reference(BaseModel):
    """
    A bibliographic reference extracted from the paper.

    The DOI field is critical for finding original studies being replicated.
    """

    num: int = Field(..., description="Reference number in the paper (1-indexed)")
    doi: Optional[str] = Field(None, description="DOI if available (CRITICAL for metascience)")
    title: Optional[str] = Field(None, description="Paper/article title")
    authors: List[str] = Field(default_factory=list, description="List of author names")
    journal: Optional[str] = Field(None, description="Journal or publication venue")
    year: Optional[int] = Field(None, description="Publication year")
    volume: Optional[str] = Field(None, description="Volume number")
    issue: Optional[str] = Field(None, description="Issue number")
    pages: Optional[str] = Field(None, description="Page range (e.g., '100-110')")
    raw_text: Optional[str] = Field(None, description="Original reference text if parsing failed")


class Table(BaseModel):
    """
    A table extracted from the paper.

    Tables are critical for extracting numerical data like patient counts,
    drug responses, statistical results, etc.
    """

    id: str = Field(..., description="Table identifier (e.g., 'tab1')")
    caption: Optional[str] = Field(None, description="Table caption/title")
    headers: List[str] = Field(default_factory=list, description="Column headers")
    rows: List[List[str]] = Field(default_factory=list, description="Table data rows")
    footnotes: List[str] = Field(default_factory=list, description="Table footnotes")
    page_number: Optional[int] = Field(None, description="Page number where table appears")
    quality_score: float = Field(1.0, ge=0, le=1, description="Extraction quality score (0-1)")
    source: str = Field("grobid", description="Extraction source (grobid/pdfplumber/camelot)")


class Section(BaseModel):
    """
    A document section with hierarchical structure.
    """

    heading: Optional[str] = Field(None, description="Section heading text")
    level: int = Field(1, ge=1, le=6, description="Heading level (1-6, like HTML h1-h6)")
    content: str = Field("", description="Section text content")
    subsections: List["Section"] = Field(default_factory=list, description="Nested subsections")


# Enable forward reference resolution for nested Section
try:
    Section.model_rebuild()  # Pydantic v2
except AttributeError:
    Section.update_forward_refs()  # Pydantic v1


class Figure(BaseModel):
    """
    A figure reference with optional AI-generated description.
    """

    id: str = Field(..., description="Figure identifier (e.g., 'fig1')")
    caption: Optional[str] = Field(None, description="Figure caption")
    page_number: Optional[int] = Field(None, description="Page number where figure appears")
    description: Optional[str] = Field(None, description="AI-generated description of figure content")
    chart_data: Optional[str] = Field(None, description="Extracted chart data as markdown table")


class ExtractionMetadata(BaseModel):
    """
    Metadata about the extraction process.
    """

    source_pdf: str = Field(..., description="Source PDF filename")
    extraction_mode: str = Field(..., description="Extraction mode used")
    extraction_timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="ISO timestamp of extraction"
    )
    grobid_version: Optional[str] = Field(None, description="GROBID version if used")
    pipeline_version: str = Field("1.0.0", description="Pipeline version")

    # Quality metrics
    tables_extracted: int = Field(0, description="Number of tables extracted")
    tables_from_fallback: int = Field(0, description="Tables extracted via fallback method")
    references_total: int = Field(0, description="Total references found")
    references_with_doi: int = Field(0, description="References with DOI extracted")

    # Quality validation
    quality_score: Optional[float] = Field(None, ge=0, le=1, description="Overall extraction quality (0-1)")
    quality_issues: List[str] = Field(default_factory=list, description="Quality issues found during validation")

    # Warnings and errors
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")


class DocumentModel(BaseModel):
    """
    Complete document model representing a parsed scientific paper.

    This is the intermediate representation between PDF extraction
    and Markdown/JSON output.
    """

    # Metadata
    metadata: ExtractionMetadata

    # Document identifiers
    doi: Optional[str] = Field(None, description="Paper DOI if found")
    title: str = Field("", description="Paper title")
    year: Optional[int] = Field(None, description="Publication year")

    # Content
    abstract: Optional[str] = Field(None, description="Paper abstract")
    sections: List[Section] = Field(default_factory=list, description="Document sections")
    tables: List[Table] = Field(default_factory=list, description="Extracted tables")
    figures: List[Figure] = Field(default_factory=list, description="Figure references")
    references: List[Reference] = Field(default_factory=list, description="Bibliography")

    def count_references_with_doi(self) -> int:
        """Count references that have DOI extracted."""
        return sum(1 for ref in self.references if ref.doi)

    def get_reference_by_num(self, num: int) -> Optional[Reference]:
        """Get a reference by its number."""
        for ref in self.references:
            if ref.num == num:
                return ref
        return None

    def get_table_by_id(self, table_id: str) -> Optional[Table]:
        """Get a table by its ID."""
        for table in self.tables:
            if table.id == table_id:
                return table
        return None


class ProcessingError(BaseModel):
    """
    An error that occurred during processing.
    """

    stage: str = Field(..., description="Processing stage where error occurred")
    error_type: str = Field(..., description="Error type/class name")
    message: str = Field(..., description="Error message")
    recoverable: bool = Field(True, description="Whether processing could continue")


class ProcessingResult(BaseModel):
    """
    Result of processing a single PDF.
    """

    pdf_path: str = Field(..., description="Path to source PDF")
    status: str = Field(..., description="success/partial/failed")
    output_dir: Optional[str] = Field(None, description="Path to output directory (contains abstract.md, body.md, references.json)")
    errors: List[ProcessingError] = Field(default_factory=list, description="Errors encountered")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    processing_time_seconds: float = Field(0, description="Processing time in seconds")


class BatchResult(BaseModel):
    """
    Result of batch processing multiple PDFs.
    """

    total_files: int = Field(0, description="Total PDFs processed")
    successful: int = Field(0, description="Fully successful extractions")
    partial: int = Field(0, description="Partial extractions (some errors)")
    failed: int = Field(0, description="Complete failures")
    results: List[ProcessingResult] = Field(default_factory=list, description="Individual results")
    error_summary: dict = Field(default_factory=dict, description="Error type counts")
    total_time_seconds: float = Field(0, description="Total processing time")

    def generate_report(self) -> str:
        """Generate a human-readable summary report."""
        lines = [
            "=" * 60,
            "PDF-to-Markdown Batch Processing Report",
            "=" * 60,
            f"Total files:  {self.total_files}",
            f"Successful:   {self.successful} ({self.successful/max(self.total_files,1)*100:.1f}%)",
            f"Partial:      {self.partial} ({self.partial/max(self.total_files,1)*100:.1f}%)",
            f"Failed:       {self.failed} ({self.failed/max(self.total_files,1)*100:.1f}%)",
            f"Total time:   {self.total_time_seconds:.1f} seconds",
            f"Avg per PDF:  {self.total_time_seconds/max(self.total_files,1):.2f} seconds",
            "",
        ]

        if self.error_summary:
            lines.append("Error Summary:")
            for error_type, count in sorted(self.error_summary.items(),
                                           key=lambda x: -x[1]):
                lines.append(f"  {error_type}: {count}")

        lines.append("=" * 60)
        return "\n".join(lines)
