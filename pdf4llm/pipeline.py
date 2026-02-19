"""
Pipeline orchestration for PDF-to-Markdown conversion.

Handles:
- Single file conversion
- Batch processing with parallelism
- Checkpoint/resume for interrupted batches
- Progress tracking and reporting
"""

import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional, Set, List

from tqdm import tqdm

from .config import Config, ExtractionMode
from .models import (
    DocumentModel, ProcessingResult, ProcessingError, BatchResult, Figure
)
from .extractors.grobid import extract_with_grobid, GrobidError, GrobidScannedPdfError, GrobidClient
from .extractors.fast import extract_fast, _extract_doi_from_filename
from .extractors.tables import assess_table_quality
from .extractors.ocr_fallback import needs_ocr_fallback, extract_with_ocr
from .renderers.markdown import render_markdown, render_abstract_md, render_body_md, render_tables_md
from .renderers.json_output import render_json, render_references_json
from .validation import validate_document

logger = logging.getLogger(__name__)


_GROBID_CRASH_MARKERS = ("crashed", "connection refused", "out of memory")

# Minimum body.md character count — multi-page PDFs below this threshold are
# candidates for OCR re-extraction (likely scanned or very sparse text layer).
_BODY_MD_MIN_SIZE = 2000


def _is_grobid_crash(error: Exception) -> bool:
    """Return True when an error indicates GROBID process crash/OOM."""
    message = str(error).lower()
    return any(marker in message for marker in _GROBID_CRASH_MARKERS)


def _build_grobid_oom_error(pdf_name: str, processed_count: int) -> GrobidError:
    """Create consistent, actionable error message for GROBID crashes."""
    return GrobidError(
        f"\n\nGROBID OUT OF MEMORY - server crashed while processing {pdf_name}.\n"
        f"Successfully processed {processed_count} PDFs before crash.\n"
        "To fix: restart GROBID and rerun with fewer workers (e.g. --workers 2 --resume)\n"
    )


def _get_pdf_page_count(pdf_path: Path) -> int:
    """Return the number of pages in a PDF, or 0 on error."""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        count = doc.page_count
        doc.close()
        return count
    except Exception:
        return 0  # Unknown — skip the size check rather than risk false positives


def _build_failed_processing_result(pdf_path: Path, error: Exception) -> ProcessingResult:
    """Create a standardized failed processing result for batch mode."""
    return ProcessingResult(
        pdf_path=str(pdf_path),
        status="failed",
        errors=[ProcessingError(
            stage="execution",
            error_type=type(error).__name__,
            message=str(error),
            recoverable=False,
        )],
        processing_time_seconds=0,
    )


def convert_single(
    pdf_path: Path,
    output_dir: Path,
    config: Config,
) -> ProcessingResult:
    """
    Convert a single PDF to Markdown and JSON.

    Args:
        pdf_path: Path to input PDF
        output_dir: Directory for output files
        config: Pipeline configuration

    Returns:
        ProcessingResult with status and output paths
    """
    start_time = time.time()
    errors = []
    warnings = []

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = pdf_path.stem
    document = None
    tei_xml = None

    try:
        # Extract based on mode
        if config.mode == ExtractionMode.FULL_GROBID:
            try:
                tei_xml, document = extract_with_grobid(pdf_path, config)
            except GrobidScannedPdfError as e:
                # Scanned PDF: skip fast extraction (useless for image-only PDFs),
                # go directly to Tesseract OCR
                logger.warning(f"Scanned PDF detected ({pdf_path.name}), using Tesseract OCR directly")
                warnings.append("Scanned PDF detected (no text layer); using Tesseract OCR")
                document = extract_with_ocr(pdf_path, config)
                document.metadata.extraction_mode = "ocr-scanned"
            except GrobidError as e:
                if _is_grobid_crash(e):
                    raise  # OOM / crash - don't silently fall back, abort
                logger.warning(f"GROBID failed for {pdf_path.name}, falling back to OCR: {e}")
                warnings.append(f"GROBID failed, falling back to Tesseract OCR: {e}")
                document = extract_with_ocr(pdf_path, config)
                document.metadata.extraction_mode = "ocr-fallback"

        elif config.mode == ExtractionMode.FAST:
            document = extract_fast(pdf_path, config)

        elif config.mode == ExtractionMode.HYBRID:
            # Try fast first, then GROBID for any deficient fields
            document = extract_fast(pdf_path, config)

            # Determine what needs GROBID fallback
            fallback_reasons = _check_hybrid_fallback_needed(document, config)

            if fallback_reasons:
                try:
                    tei_xml, grobid_doc = extract_with_grobid(pdf_path, config)

                    # Selectively merge GROBID results based on what failed
                    if "title" in fallback_reasons:
                        document.title = grobid_doc.title

                    if "abstract" in fallback_reasons:
                        document.abstract = grobid_doc.abstract

                    if "doi" in fallback_reasons and grobid_doc.doi:
                        document.doi = grobid_doc.doi

                    if "tables" in fallback_reasons and grobid_doc.tables:
                        document.tables = grobid_doc.tables
                        document.metadata.tables_from_fallback = 0

                    # Always prefer GROBID references (better DOI extraction)
                    if grobid_doc.references:
                        document.references = grobid_doc.references
                        document.metadata.references_total = len(grobid_doc.references)
                        document.metadata.references_with_doi = sum(
                            1 for r in grobid_doc.references if r.doi
                        )

                    warnings.append(
                        f"Used GROBID fallback for: {', '.join(fallback_reasons)}"
                    )

                except GrobidError as e:
                    warnings.append(f"GROBID fallback failed: {e}")

        # Extract and analyze charts if enabled
        if config.extract_charts and document:
            try:
                from .extractors.charts import ChartExtractor

                logger.info(f"Extracting charts from {pdf_path}")
                extractor = ChartExtractor(
                    provider=config.chart_provider,
                    model=config.chart_model,
                    ollama_url=config.ollama_url,
                )

                # Extract figures from PDF
                extracted_figures = extractor.extract_figures_from_pdf(pdf_path)

                if extracted_figures:
                    # Analyze figures with vision model
                    analyzed_figures = extractor.analyze_all_figures(extracted_figures)

                    # Save figure images to separate files
                    effective_output_dir.mkdir(parents=True, exist_ok=True)
                    for fig in analyzed_figures:
                        # Extract page number and figure index from figure_id (e.g., "fig_8_1" -> page 8, fig 1)
                        parts = fig.figure_id.split("_")
                        if len(parts) >= 3:
                            page_num = parts[1]
                            fig_num = parts[2]
                            img_filename = f"page_{page_num}_fig_{fig_num}.png"
                        else:
                            img_filename = f"{fig.figure_id}.png"

                        img_path = effective_output_dir / img_filename
                        img_path.write_bytes(fig.image_data)
                        logger.info(f"Saved figure to {img_path}")

                    # Convert to Figure model and add to document
                    document.figures = [
                        Figure(
                            id=fig.figure_id,
                            caption=fig.caption,
                            page_number=fig.page_number,
                            description=fig.description,
                            chart_data=fig.chart_data,
                        )
                        for fig in analyzed_figures
                    ]

                    logger.info(f"Analyzed {len(analyzed_figures)} figures")

            except Exception as e:
                warnings.append(f"Chart extraction failed: {e}")
                logger.warning(f"Chart extraction failed for {pdf_path}: {e}")

    except GrobidError as e:
        if _is_grobid_crash(e):
            raise  # OOM / crash - propagate to stop the batch
        errors.append(ProcessingError(
            stage="grobid",
            error_type="GrobidError",
            message=str(e),
            recoverable=False,
        ))
        logger.error(f"GROBID error for {pdf_path}: {e}")

    except Exception as e:
        errors.append(ProcessingError(
            stage="extraction",
            error_type=type(e).__name__,
            message=str(e),
            recoverable=False,
        ))
        logger.error(f"Extraction error for {pdf_path}: {e}")

    # Fallback: extract DOI from filename if not found by extractor
    if document and not document.doi:
        filename_doi = _extract_doi_from_filename(pdf_path)
        if filename_doi:
            document.doi = filename_doi
            logger.debug(f"DOI extracted from filename: {filename_doi}")

    # Output dir: subfolder named by PDF stem
    if document:
        effective_output_dir = output_dir / stem
        effective_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        effective_output_dir = output_dir

    abstract_path = effective_output_dir / "abstract.md"
    body_path = effective_output_dir / "body.md"
    refs_path = effective_output_dir / "references.json"
    tables_path = effective_output_dir / "tables.md"
    tei_path = effective_output_dir / f"{stem}.tei.xml"

    # Validate extraction quality
    if document:
        quality_score, quality_issues = validate_document(document)
        document.metadata.quality_score = quality_score
        document.metadata.quality_issues = quality_issues
        if quality_issues:
            logger.info(f"Quality score {quality_score:.2f} for {pdf_path.name}: {', '.join(quality_issues)}")
        
        # OCR fallback: if quality is below threshold, retry with OCR
        if needs_ocr_fallback(quality_score):
            logger.info(f"Quality score {quality_score:.2f} below threshold, attempting OCR fallback...")
            try:
                ocr_document = extract_with_ocr(pdf_path, config)
                ocr_score, ocr_issues = validate_document(ocr_document)
                
                # Use OCR result if it's better than GROBID result
                if ocr_score > quality_score:
                    logger.info(f"OCR fallback improved quality: {quality_score:.2f} -> {ocr_score:.2f}")
                    document = ocr_document
                    document.metadata.quality_score = ocr_score
                    document.metadata.quality_issues = ocr_issues
                    warnings.append(f"OCR fallback used (quality improved from {quality_score:.2f} to {ocr_score:.2f})")
                else:
                    logger.info(f"OCR fallback did not improve quality ({ocr_score:.2f} vs {quality_score:.2f}), keeping original")
                    warnings.append(f"OCR fallback attempted but did not improve quality")
            except Exception as e:
                logger.warning(f"OCR fallback failed: {e}")
                warnings.append(f"OCR fallback failed: {e}")

        # Body size check: if GROBID produced very little body content and the PDF
        # has more than one page, the text layer is likely sparse/scanned — retry
        # with Tesseract OCR. Single-page PDFs are excluded because a small body
        # is perfectly normal (e.g. a letter, abstract-only doc, or short note).
        if (document
                and "ocr" not in (document.metadata.extraction_mode or "")
                and config.mode == ExtractionMode.FULL_GROBID):
            body_md_preview = render_body_md(document)
            if len(body_md_preview) < _BODY_MD_MIN_SIZE:
                pdf_pages = _get_pdf_page_count(pdf_path)
                if pdf_pages > 1:
                    logger.info(
                        f"Body content very small ({len(body_md_preview)} chars) for "
                        f"{pdf_path.name} ({pdf_pages} pages), retrying with Tesseract OCR"
                    )
                    try:
                        ocr_doc = extract_with_ocr(pdf_path, config)
                        ocr_body_preview = render_body_md(ocr_doc)
                        if len(ocr_body_preview) > len(body_md_preview):
                            logger.info(
                                f"OCR retry improved body: {len(body_md_preview)} → "
                                f"{len(ocr_body_preview)} chars"
                            )
                            document = ocr_doc
                            warnings.append(
                                f"GROBID body too small ({len(body_md_preview)} chars), "
                                f"switched to Tesseract OCR ({len(ocr_body_preview)} chars)"
                            )
                        else:
                            logger.info(
                                f"OCR retry did not improve body size "
                                f"({len(ocr_body_preview)} vs {len(body_md_preview)}), keeping GROBID result"
                            )
                    except Exception as e:
                        logger.warning(f"OCR retry for small body failed: {e}")

    # Re-parse OCR references via GROBID citation parser for structured output
    if (document
            and "ocr" in (document.metadata.extraction_mode or "")
            and config.requires_grobid()):
        try:
            _enrich_ocr_references(document, config)
        except Exception as e:
            logger.warning(f"OCR reference enrichment failed: {e}")

    # Render outputs if we have a document
    result_output_dir = None

    if document:
        try:
            # Render abstract.md
            abstract_md = render_abstract_md(document)
            abstract_path.write_text(abstract_md, encoding="utf-8")

            # Render body.md
            body_md = render_body_md(document)
            body_path.write_text(body_md, encoding="utf-8")

            # Render references.json
            refs_json = render_references_json(document)
            refs_path.write_text(refs_json, encoding="utf-8")

            # Render tables.md (only if tables exist)
            if document.tables:
                tables_md = render_tables_md(document)
                tables_path.write_text(tables_md, encoding="utf-8")

            result_output_dir = str(effective_output_dir)

            # Save TEI XML if enabled and available
            if config.output_tei and tei_xml:
                tei_path.write_text(tei_xml, encoding="utf-8")

        except Exception as e:
            errors.append(ProcessingError(
                stage="render",
                error_type=type(e).__name__,
                message=str(e),
                recoverable=True,
            ))
            logger.error(f"Render error for {pdf_path}: {e}")

    # Determine status
    if not errors:
        status = "success"
    elif document and result_output_dir:
        status = "partial"
    else:
        status = "failed"

    processing_time = time.time() - start_time

    return ProcessingResult(
        pdf_path=str(pdf_path),
        status=status,
        output_dir=result_output_dir,
        errors=errors,
        warnings=warnings,
        processing_time_seconds=round(processing_time, 2),
    )


def _enrich_ocr_references(document: DocumentModel, config: Config) -> None:
    """
    Re-parse OCR-extracted references using GROBID /api/processCitation.

    OCR reference extraction is very basic (raw text → title field).
    This sends each raw reference string to GROBID's citation parser to
    get proper structured output (authors, journal, year, volume, pages, DOI).

    Modifies document.references in-place.
    """
    if not document.references:
        return

    client = GrobidClient(config)
    try:
        if not client.is_alive(timeout=5):
            logger.debug("GROBID not available for OCR reference enrichment")
            return
    except Exception:
        return

    logger.info(f"Enriching {len(document.references)} OCR references via GROBID citation parser")
    enriched = []
    improved = 0

    for ref in document.references:
        raw = ref.raw_text or ref.title or ""
        if not raw.strip():
            enriched.append(ref)
            continue

        parsed = client.process_citation(raw)
        if parsed and (parsed.title or parsed.authors):
            # Preserve original num, raw_text, and any DOI already found
            enriched.append(parsed.model_copy(update={
                "num": ref.num,
                "raw_text": raw,
                "doi": parsed.doi or ref.doi,
            }))
            improved += 1
        else:
            enriched.append(ref)

    document.references = enriched
    document.metadata.references_with_doi = sum(1 for r in enriched if r.doi)
    if improved:
        logger.info(f"GROBID structured {improved}/{len(enriched)} OCR references")


def _check_hybrid_fallback_needed(
    document: DocumentModel, config: Config
) -> List[str]:
    """
    Check if fast extraction produced deficient results that warrant
    GROBID fallback. Returns a list of reason strings (empty = no fallback needed).
    """
    reasons = []

    # Title: missing, too short, or looks like an artifact
    title = document.title or ""
    if title in ("Untitled", "") or len(title) < 15:
        reasons.append("title")
    else:
        # Import the artifact patterns from fast extractor
        from .extractors.fast import _ARTIFACT_RE
        if any(p.match(title) for p in _ARTIFACT_RE):
            reasons.append("title")

    # Abstract: missing or too short
    if not document.abstract or len(document.abstract) < 50:
        reasons.append("abstract")

    # DOI: missing
    if not document.doi:
        reasons.append("doi")

    # Table quality: low quality or no tables detected
    low_quality_tables = [
        t for t in document.tables
        if t.quality_score < config.table_quality_threshold
    ]
    if low_quality_tables or not document.tables:
        reasons.append("tables")

    return reasons


class BatchProcessor:
    """
    Batch processor for converting multiple PDFs.

    Features:
    - Parallel processing with ProcessPoolExecutor
    - Checkpoint/resume for interrupted batches
    - Progress tracking with tqdm
    """

    def __init__(self, config: Config, checkpoint_file: Optional[Path] = None):
        self.config = config
        self.checkpoint_file = checkpoint_file
        self.processed: Set[str] = self._load_checkpoint()

    def _load_checkpoint(self) -> Set[str]:
        """Load checkpoint file if it exists."""
        if self.checkpoint_file and self.checkpoint_file.exists():
            try:
                data = json.loads(self.checkpoint_file.read_text())
                return set(data.get("processed", []))
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return set()

    @staticmethod
    def _move_pdf(pdf_path: Path, output_dir: Path):
        """Move the source PDF into its output folder."""
        src = Path(pdf_path).resolve()
        dst = output_dir / src.name
        if src != dst and src.exists():
            shutil.move(str(src), str(dst))
            logger.info(f"Moved PDF to {dst}")

    def _save_checkpoint(self, pdf_path: str):
        """Save checkpoint after successful processing."""
        if not self.checkpoint_file:
            return

        self.processed.add(pdf_path)

        try:
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            self.checkpoint_file.write_text(json.dumps({
                "processed": list(self.processed),
                "last_updated": datetime.utcnow().isoformat(),
            }))
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def process_folder(
        self,
        input_dir: Path,
        output_dir: Path,
        resume: bool = True,
        skip_existing: bool = False,
        move_pdf: bool = False,
    ) -> BatchResult:
        """
        Process all PDFs in a folder.

        Args:
            input_dir: Directory containing PDFs
            output_dir: Directory for output files
            resume: Whether to skip already-processed files (using checkpoint)
            skip_existing: Whether to skip PDFs that already have output files

        Returns:
            BatchResult with processing summary
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all PDFs — glob("**/*.pdf") misses root-level files in Python <3.12,
        # so explicitly combine root-level and recursive results.
        pdfs = list(dict.fromkeys(
            list(input_dir.glob("*.pdf")) + list(input_dir.glob("**/*.pdf"))
        ))
        logger.info(f"Found {len(pdfs)} PDFs in {input_dir}")

        # Filter already processed if resuming from checkpoint
        if resume and self.processed:
            original_count = len(pdfs)
            pdfs = [p for p in pdfs if str(p) not in self.processed]
            logger.info(f"Resuming: {original_count - len(pdfs)} already processed, "
                       f"{len(pdfs)} remaining")

        # Filter PDFs that already have output folders with content
        if skip_existing:
            original_count = len(pdfs)
            pdfs_to_process = []
            for pdf in pdfs:
                stem = pdf.stem
                out = output_dir / stem
                if out.exists() and (out / "abstract.md").exists() and (out / "body.md").exists():
                    continue
                pdfs_to_process.append(pdf)
            pdfs = pdfs_to_process
            skipped = original_count - len(pdfs)
            if skipped > 0:
                logger.info(f"Skipping {skipped} PDFs with existing output folders, "
                           f"{len(pdfs)} remaining")

        if not pdfs:
            return BatchResult(
                total_files=len(self.processed),
                successful=len(self.processed),
            )

        # Check GROBID availability if needed
        if self.config.requires_grobid():
            client = GrobidClient(self.config)
            if not client.is_alive():
                raise GrobidError(
                    f"GROBID server not available at {self.config.grobid_url}. "
                    "Start GROBID with: docker run --gpus all -p 8070:8070 lfoppiano/grobid:0.8.0"
                )
            logger.info(f"GROBID server ready (version: {client.get_version()})")

        # Process files
        start_time = time.time()
        results: List[ProcessingResult] = []
        error_summary = {}

        # Use ProcessPoolExecutor for parallelism
        # Note: We process sequentially if workers=1 for easier debugging
        if self.config.workers == 1:
            for pdf_path in tqdm(pdfs, desc="Converting PDFs"):
                try:
                    result = convert_single(pdf_path, output_dir, self.config)
                except GrobidError as e:
                    if _is_grobid_crash(e):
                        raise _build_grobid_oom_error(pdf_path.name, len(results))
                    raise
                results.append(result)

                if result.status != "failed":
                    self._save_checkpoint(str(pdf_path))
                    if move_pdf and result.output_dir:
                        self._move_pdf(pdf_path, Path(result.output_dir))

                for error in result.errors:
                    error_summary[error.error_type] = error_summary.get(error.error_type, 0) + 1
        else:
            # Parallel processing (ThreadPool since work is I/O-bound: GROBID HTTP calls)
            with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
                futures = {
                    executor.submit(_process_single_wrapper, pdf, output_dir, self.config): pdf
                    for pdf in pdfs
                }

                for future in tqdm(as_completed(futures), total=len(pdfs), desc="Converting PDFs"):
                    pdf_path = futures[future]
                    try:
                        result = future.result()
                        results.append(result)

                        if result.status != "failed":
                            self._save_checkpoint(str(pdf_path))
                            if move_pdf and result.output_dir:
                                self._move_pdf(pdf_path, Path(result.output_dir))

                        for error in result.errors:
                            error_summary[error.error_type] = error_summary.get(error.error_type, 0) + 1

                    except GrobidError as e:
                        if _is_grobid_crash(e):
                            # Cancel remaining futures
                            for f in futures:
                                f.cancel()
                            raise _build_grobid_oom_error(pdf_path.name, len(results))
                        logger.error(f"Failed to process {pdf_path}: {e}")
                        results.append(_build_failed_processing_result(pdf_path, e))
                    except Exception as e:
                        logger.error(f"Failed to process {pdf_path}: {e}")
                        results.append(_build_failed_processing_result(pdf_path, e))

        total_time = time.time() - start_time

        # Count results
        successful = sum(1 for r in results if r.status == "success")
        partial = sum(1 for r in results if r.status == "partial")
        failed = sum(1 for r in results if r.status == "failed")

        batch_result = BatchResult(
            total_files=len(results),
            successful=successful,
            partial=partial,
            failed=failed,
            results=results,
            error_summary=error_summary,
            total_time_seconds=round(total_time, 2),
        )

        # Log summary
        logger.info(batch_result.generate_report())

        return batch_result


def _process_single_wrapper(pdf_path: Path, output_dir: Path, config: Config) -> ProcessingResult:
    """Wrapper for convert_single to work with ProcessPoolExecutor."""
    return convert_single(pdf_path, output_dir, config)


def check_grobid_health(config: Config) -> dict:
    """
    Check GROBID server health.

    Returns:
        Dictionary with health check results
    """
    client = GrobidClient(config)

    return {
        "url": config.grobid_url,
        "alive": client.is_alive(),
        "version": client.get_version(),
    }
