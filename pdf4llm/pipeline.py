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
import os
import re
import shutil
import subprocess
import sys
import time
import gc
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
from pathlib import Path
from typing import Optional, Set, List, Dict

from tqdm import tqdm

from .config import Config, ExtractionMode
from .models import (
    DocumentModel, ProcessingResult, ProcessingError, BatchResult, Figure
)
from .extractors.grobid import extract_with_grobid, GrobidError, GrobidScannedPdfError, GrobidClient, _normalize_doi
from .extractors.fast import extract_fast, _extract_doi_from_filename, _validate_title, REF_HEADING_PATTERNS
from .extractors.doi_lookup import fetch_title_from_doi
from .extractors.ocr_fallback import needs_ocr_fallback, extract_with_ocr
from .renderers.markdown import render_abstract_md, render_body_md, render_single_markdown
from .renderers.json_output import render_references_json
from .validation import validate_document

logger = logging.getLogger(__name__)

# Path to the docker-compose.yml bundled with this package
_DOCKER_DIR = Path(__file__).resolve().parent.parent / "docker"


def _start_grobid_docker(mode: str = "crf", wait_seconds: int = 120) -> None:
    """Start GROBID via Docker Compose if not already running.

    Args:
        mode: "crf" (lightweight) or "delft" (highest accuracy, ~8 GB image)
        wait_seconds: seconds to wait for GROBID to become ready
    """
    import urllib.request

    def _is_alive(url: str = "http://127.0.0.1:8070/api/isalive") -> bool:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                return r.read().strip() == b"true"
        except Exception:
            return False

    if _is_alive():
        logger.info("GROBID already running at http://127.0.0.1:8070")
        return

    compose_file = _DOCKER_DIR / "docker-compose.yml"
    if not compose_file.exists():
        raise FileNotFoundError(
            f"docker-compose.yml not found at {compose_file}. "
            "Cannot auto-start GROBID."
        )

    print(f"Starting GROBID ({mode} mode) via Docker...", flush=True)
    subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "down"],
        capture_output=True,
    )
    subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "--profile", mode, "up", "-d"],
        check=True,
    )

    print(f"Waiting for GROBID to be ready (up to {wait_seconds}s)...", end="", flush=True)
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if _is_alive():
            print(" ready.", flush=True)
            return
        print(".", end="", flush=True)
        time.sleep(3)

    print("", flush=True)
    raise RuntimeError(
        f"GROBID did not become ready within {wait_seconds}s. "
        "Check: docker compose logs grobid-crf (or grobid-delft)"
    )


class _TqdmLoggingHandler(logging.StreamHandler):
    """Route log output through tqdm.write() so messages don't corrupt the progress bar."""

    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


GROBID_CRASH_MARKERS = ("crashed", "connection refused", "out of memory")

# Minimum body.md character count — multi-page PDFs below this threshold are
# candidates for OCR re-extraction (likely scanned or very sparse text layer).
_BODY_MD_MIN_SIZE = 2000


def _is_grobid_crash(error: Exception) -> bool:
    """Return True when an error indicates GROBID process crash/OOM."""
    message = str(error).lower()
    return any(marker in message for marker in GROBID_CRASH_MARKERS)


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


def _patch_docling_hyperlink_uri():
    """Monkey-patch PdfHyperlink to accept malformed URIs and AnyUrl objects.

    Same fix as /home/dan/Dropbox/AAA_METASCIENCE_OBSERVATORY/convert_pdfs/convert_pdfs.py —
    some academic PDFs have malformed link URIs that docling's strict AnyUrl field rejects.
    """
    try:
        from docling_core.types.doc import page as page_mod
        from typing import Any
        orig_cls = page_mod.PdfHyperlink
        if orig_cls.model_fields["uri"].annotation is not Any:
            orig_cls.model_fields["uri"].annotation = Any
            orig_cls.model_rebuild(force=True)
    except Exception:
        pass  # best effort; older docling versions may not have PdfHyperlink


# Module-level flag so we only silence the noisy docling loggers once per process
_DOCLING_LOGS_QUIETED = False


# Semaphore that caps concurrent CUDA docling conversions.
# Default: 1 slot (one GPU worker at a time). Override with --gpu-slots N or
# PDF4LLM_GPU_SLOTS=N env var. With --workers 4 --gpu-slots 2, two threads run
# docling on CUDA simultaneously while the other two run on CPU.
import threading as _threading
_GPU_SLOT: Optional[_threading.Semaphore] = None
_GPU_SLOT_LOCK = _threading.Lock()
_GPU_AVAILABLE_CACHE: Optional[bool] = None


def _get_gpu_slot() -> _threading.Semaphore:
    """Return the GPU semaphore, initializing it on first call.

    The permit count comes from the PDF4LLM_GPU_SLOTS env var (set by the
    --gpu-slots CLI flag before any worker threads start). Lazy init means
    the CLI flag is always read before the semaphore is created.
    """
    global _GPU_SLOT
    if _GPU_SLOT is not None:
        return _GPU_SLOT
    with _GPU_SLOT_LOCK:
        if _GPU_SLOT is None:
            try:
                slots = int(os.environ.get("PDF4LLM_GPU_SLOTS", "1"))
                slots = max(1, slots)
            except (ValueError, TypeError):
                slots = 1
            _GPU_SLOT = _threading.Semaphore(slots)
    return _GPU_SLOT


def _cuda_available() -> bool:
    """Check CUDA availability. Caches True; re-polls on False.

    Why not cache False: a transient failure on first call (e.g. torch imported
    before CUDA context ready, or sibling thread holding the import lock) would
    permanently lock the process into CPU mode. Re-polling is cheap once torch
    is in sys.modules, and `torch.cuda.is_available()` is a sub-millisecond check.
    """
    global _GPU_AVAILABLE_CACHE
    if _GPU_AVAILABLE_CACHE is True:
        return True
    try:
        import torch
        result = bool(torch.cuda.is_available())
        if result:
            _GPU_AVAILABLE_CACHE = True
        return result
    except Exception:
        return False


def _quiet_docling_noise():
    """Suppress docling's most noisy INFO/ERROR log messages.

    Specifically silences:
      - `docling.models.stages.ocr.tesseract_ocr_cli_model` — logs every
        `command: tesseract --psm 0 -l osd /tmp/tmpXXX.png stdout` at INFO
        (one per page, so ~10-30 spam lines per PDF)
      - `docling.models.stages.ocr.tesseract_ocr_model` — same pattern
      - `docling.document_converter` — "detected formats" / "Going to convert" /
        "Initializing pipeline" / "Loading plugin" / "Registered ... engines" /
        "Accelerator device" / "Processing document" / "Finished converting"
        are all at INFO; most are redundant with pdf4llm's tqdm progress bar.

    We raise these to WARNING so genuine errors and warnings still surface.
    The `OSD failed ... Too few characters` messages are logged at ERROR but
    are benign (tesseract just can't detect orientation on sparse pages and
    docling falls back gracefully). We install a filter that drops those
    specific messages while still letting real errors through.

    Only runs once per process — subsequent calls are no-ops.
    """
    global _DOCLING_LOGS_QUIETED
    if _DOCLING_LOGS_QUIETED:
        return
    _DOCLING_LOGS_QUIETED = True

    import logging as _logging

    # Raise the log level on the chatty loggers to WARNING — this kills the
    # "command: tesseract" INFO spam and most docling pipeline-init chatter.
    noisy_loggers = [
        "docling.models.stages.ocr.tesseract_ocr_cli_model",
        "docling.models.stages.ocr.tesseract_ocr_model",
        "docling.document_converter",
        "docling.pipeline.standard_pdf_pipeline",
        "docling.pipeline.base_pipeline",
        "docling.models.factories.base_factory",
        "docling.datamodel.pipeline_options",
        "docling.datamodel.document",  # "An unexpected error occurred while opening the document"
        "docling.utils.accelerator_utils",  # "Accelerator device: 'cuda:0'"
    ]
    for name in noisy_loggers:
        _logging.getLogger(name).setLevel(_logging.WARNING)

    # Tesseract OCR loggers only ever emit noise (OSD failures, command spam).
    # Fully disable them by setting level above CRITICAL so Python won't even
    # create a LogRecord — this is more reliable than a filter-based approach.
    for name in [
        "docling.models.stages.ocr.tesseract_ocr_cli_model",
        "docling.models.stages.ocr.tesseract_ocr_model",
    ]:
        _logging.getLogger(name).setLevel(_logging.CRITICAL + 1)

    # Install a filter that drops known-benign ERROR messages from the document
    # logger that our pipeline already handles gracefully (caught by _try_docling).
    class _DoclingNoisyFilter(_logging.Filter):
        _DROP_PATTERNS = (
            # PdfiumError / corrupt PDF — caught by _try_docling, pipeline falls through
            ("An unexpected error occurred while opening the document",),
        )

        def filter(self, record):
            msg = record.getMessage()
            for patterns in self._DROP_PATTERNS:
                if all(p in msg for p in patterns):
                    return False  # drop
            return True

    noisy_filter = _DoclingNoisyFilter()
    _logging.getLogger("docling.datamodel.document").addFilter(noisy_filter)


def _extract_with_docling(pdf_path: Path, config: Config) -> tuple:
    """Extract document content using docling's Python API.

    Based on the approach in /home/dan/Dropbox/AAA_METASCIENCE_OBSERVATORY/convert_pdfs/convert_pdfs.py.
    Uses the Python API directly (no subprocess) for speed and reliability.

    Does NOT extract figure images — figures are placeholder-only in the output.
    If GPU (CUDA) is available and `config.docling_use_gpu` is True, uses
    ThreadedPdfPipelineOptions with CUDA acceleration.

    Returns (DocumentModel, raw_markdown_text).
    """
    import os
    import warnings
    from .models import ExtractionMetadata, Section

    # GPU selection policy:
    #   1. Explicit opt-out: PDF4LLM_DOCLING_CPU=1 forces CPU for every worker.
    #   2. Otherwise, if CUDA is available at runtime (torch.cuda.is_available),
    #      try to claim the single-permit _GPU_SLOT. Whoever claims it uses CUDA;
    #      the other concurrent worker(s) fall back to CPU until the slot frees.
    #   3. Explicit --docling-gpu (config.docling_use_gpu=True) still forces the
    #      caller to wait for the slot rather than silently dropping to CPU, so
    #      users who explicitly asked for GPU get it.
    #
    # Historically this function clobbered CUDA_VISIBLE_DEVICES="" before
    # importing docling to protect against broken CUDA installs. That's still
    # safe in pure-CPU mode, but with the mixed GPU/CPU thread pool we must NOT
    # mutate CUDA_VISIBLE_DEVICES once a sibling thread has bound to CUDA —
    # hence the env-var reset only fires in the strict-CPU branch below.
    cpu_forced = os.environ.get("PDF4LLM_DOCLING_CPU", "").strip() not in ("", "0", "false", "False")
    explicit_gpu = bool(getattr(config, "docling_use_gpu", False))
    cuda_ok = (not cpu_forced) and _cuda_available()

    gpu_claimed = False
    if cuda_ok:
        if explicit_gpu:
            # User explicitly asked for GPU — block until a slot is free.
            _get_gpu_slot().acquire()
            gpu_claimed = True
        else:
            # Auto-mode — grab GPU opportunistically, else run on CPU.
            gpu_claimed = _get_gpu_slot().acquire(blocking=False)

    use_gpu = gpu_claimed

    # NOTE: previously we set os.environ["CUDA_VISIBLE_DEVICES"] = "" when
    # cuda_ok was False, to "hide" CUDA from docling probing. That was a trap:
    # once set, CUDA becomes permanently invisible to torch in this process
    # even if _cuda_available later recovers. Since we pass an explicit
    # AcceleratorDevice.CPU in the CPU branch below, docling already honors
    # that directly — no need to clobber the process env.
    warnings.filterwarnings("ignore", message="CUDA is not available")

    try:
        logger.info(
            f"docling device: {'CUDA' if use_gpu else 'CPU'} for {pdf_path.name}"
        )
    except Exception:
        pass

    _patch_docling_hyperlink_uri()
    _quiet_docling_noise()

    try:
        from docling.document_converter import DocumentConverter, FormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            ThreadedPdfPipelineOptions,
            TesseractCliOcrOptions,
        )
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    except ImportError as e:
        if gpu_claimed:
            _get_gpu_slot().release()
        raise RuntimeError(
            f"docling not installed. Install with: pip install docling  (import error: {e})"
        )

    # Build pipeline options.
    #
    # OCR engine selection:
    #   We explicitly use TesseractCliOcrOptions (tesseract CLI binary, which
    #   pdf4llm already depends on for its own OCR fallback) instead of the
    #   default rapidocr. rapidocr has a bug on current versions where it
    #   assigns a PosixPath to an OmegaConf config field that rejects non-
    #   primitive types (omegaconf.errors.UnsupportedValueType:
    #   "Value 'PosixPath' is not a supported primitive type, full_key:
    #   Global.model_root_dir"). Tesseract CLI bypasses this.
    #
    # do_ocr stays TRUE so we get full OCR coverage on image-embedded text
    # (tables rendered as images, equation labels, scanned pages).
    #
    # We also explicitly force the accelerator device (CPU unless user opted
    # into GPU) because docling's default probing can crash with
    # `libnvrtc-builtins.so.13.0 not found` on CUDA-installed-but-incomplete
    # systems.
    ocr_options = TesseractCliOcrOptions()

    if use_gpu:
        accel = AcceleratorOptions(device=AcceleratorDevice.CUDA)
        pipeline_options = ThreadedPdfPipelineOptions(
            accelerator_options=accel,
            layout_batch_size=64,
            ocr_batch_size=64,
            do_ocr=True,
            ocr_options=ocr_options,
            generate_picture_images=False,
        )
    else:
        accel = AcceleratorOptions(device=AcceleratorDevice.CPU)
        pipeline_options = PdfPipelineOptions(
            accelerator_options=accel,
            do_ocr=True,
            ocr_options=ocr_options,
            generate_picture_images=False,
        )

    format_options = {
        InputFormat.PDF: FormatOption(
            pipeline_options=pipeline_options,
            backend=DoclingParseDocumentBackend,
            pipeline_cls=StandardPdfPipeline,
        ),
    }
    converter = DocumentConverter(format_options=format_options)

    # Convert — raises_on_error=False so we get partial results on recoverable errors.
    # The GPU slot (if claimed) is released as soon as converter.convert returns,
    # BEFORE the cheap markdown post-processing below. That way the next waiting
    # worker can start its own CUDA conversion while this thread keeps processing
    # the extracted document in Python.
    try:
        try:
            result = converter.convert(pdf_path, raises_on_error=False)
        except Exception as e:
            raise RuntimeError(f"docling convert failed: {e}")
    finally:
        if gpu_claimed:
            _get_gpu_slot().release()
            gpu_claimed = False

    if result is None or result.document is None:
        raise RuntimeError("docling produced no document")

    doc_obj = result.document
    md_text = doc_obj.export_to_markdown()
    # Release docling's intermediate page/image data immediately — the docling
    # ConversionResult and DocumentConverter both hold heavy in-memory state
    # (page images, model tensors) that won't release until GC otherwise.
    del doc_obj, result, converter

    if not md_text or len(md_text.strip()) < 20:
        raise RuntimeError("docling produced empty or trivially small markdown")

    # Split into abstract and body (same logic as before)
    abstract = None
    body_lines = []
    lines = md_text.split("\n")
    in_abstract = False

    for line in lines:
        stripped = line.strip().lower()
        if stripped in ("# abstract", "## abstract", "### abstract", "abstract"):
            in_abstract = True
            continue
        if in_abstract:
            # End abstract at next heading or after sufficient content
            if line.startswith("#") and stripped not in ("", "abstract"):
                in_abstract = False
                abstract = "\n".join(body_lines).strip()
                body_lines = [line]
                continue
            body_lines.append(line)
        else:
            body_lines.append(line)

    if in_abstract and body_lines:
        # Abstract ran to end of document (unusual but possible)
        abstract = "\n".join(body_lines).strip()
        body_lines = []

    if abstract is None:
        # No explicit abstract heading found — use first non-title paragraph
        paragraphs = md_text.split("\n\n")
        non_empty = [p.strip() for p in paragraphs if p.strip() and not p.strip().startswith("#")]
        if len(non_empty) >= 2:
            abstract = non_empty[1]
        elif non_empty:
            abstract = non_empty[0]

    body_text = "\n".join(body_lines).strip() if not in_abstract else ""
    if not body_text:
        body_text = md_text  # fallback: use full text as body

    # Extract title — prefer docling's metadata, fall back to first H1
    title = ""
    try:
        if hasattr(doc_obj, "name") and doc_obj.name:
            title = str(doc_obj.name)
    except Exception:
        pass
    if not title:
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break

    # DOI from filename (docling doesn't extract DOIs)
    doi = _extract_doi_from_filename(pdf_path)

    metadata = ExtractionMetadata(
        source_pdf=pdf_path.name,
        extraction_mode="docling-gpu" if use_gpu else "docling",
    )

    sections = [Section(heading="Body", level=1, content=body_text)] if body_text else []

    doc = DocumentModel(
        metadata=metadata,
        doi=doi,
        title=title or (pdf_path.stem if _validate_title(pdf_path.stem) else ""),
        abstract=abstract,
        sections=sections,
    )
    return doc, md_text


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


# ══════════════════════════════════════════════════════════════════════════
# Parallel extraction helpers (new architecture)
# ══════════════════════════════════════════════════════════════════════════
#
# The "parallel" mode runs GROBID and docling concurrently for each PDF and
# merges their outputs, using the best-of-each strengths:
#   - abstract.md   ← GROBID (DeLFT header model is strongest on titles/abstracts)
#   - body.md       ← docling (best layout/table handling on non-standard PDFs)
#   - references.json ← GROBID (structured bibliographic parsing)
#   - references.md ← docling (new file, refs as prose markdown)
# If both fail, we fall back to marker-pdf as a final tier.


def _try_grobid(pdf_path: Path, config: Config) -> tuple:
    """Run GROBID, returning (document, tei_xml, exception).

    Does NOT re-raise non-crash GrobidError — those are expected failures
    that let the caller fall through to the next tier. Crash errors DO
    propagate (they indicate GROBID is unrecoverable and the batch should abort).
    """
    try:
        tei_xml, doc = extract_with_grobid(pdf_path, config)
        return doc, tei_xml, None
    except GrobidScannedPdfError as e:
        return None, None, e
    except GrobidError as e:
        if _is_grobid_crash(e):
            raise  # escalate crash to abort the batch
        return None, None, e
    except Exception as e:
        return None, None, e


def _try_docling(pdf_path: Path, config: Config) -> tuple:
    """Run docling, returning (document, exception)."""
    try:
        doc, raw_md = _extract_with_docling(pdf_path, config)
        # Stash raw markdown on the doc for later references.md extraction
        doc._docling_raw_md = raw_md
        return doc, None
    except Exception as e:
        return None, e


def _try_marker(pdf_path: Path, config: Config) -> tuple:
    """Run marker-pdf as a last-resort fallback. Returns (document, exception)."""
    try:
        from .extractors.marker_ext import extract_with_marker
    except ImportError as e:
        return None, RuntimeError(f"marker-pdf not available: {e}")
    try:
        doc = extract_with_marker(pdf_path, config)
        return doc, None
    except Exception as e:
        return None, e


def _try_pymupdf4llm(pdf_path: Path, config: Config) -> tuple:
    """Run pymupdf4llm as the final-last-resort fallback.

    This tier runs after marker-pdf has already failed. It uses PyMuPDF's
    text-extraction engine via the `pymupdf4llm` package — no ML models,
    fast, small memory footprint — trading accuracy for the ability to
    produce *something* on virtually any text-layer PDF.

    Returns (document, exception).
    """
    try:
        from .extractors.pymupdf4llm_ext import extract_with_pymupdf4llm
    except ImportError as e:
        return None, RuntimeError(f"pymupdf4llm not available: {e}")
    try:
        doc = extract_with_pymupdf4llm(pdf_path, config)
        return doc, None
    except Exception as e:
        return None, e


def _merge_grobid_docling(grobid_doc: DocumentModel, docling_doc: DocumentModel) -> DocumentModel:
    """Merge GROBID + docling outputs into a single DocumentModel.

    Strategy: keep each tool's strongest output fields. For the title,
    pick the first candidate (GROBID preferred) that passes _validate_title;
    if neither does, leave it empty for the DOI API fallback in convert_single.
    """
    # Start from GROBID's metadata (has grobid_version etc.)
    merged_meta = grobid_doc.metadata
    merged_meta.extraction_mode = "merged-grobid-docling"

    merged = DocumentModel(
        metadata=merged_meta,
        # Prefer GROBID for the structured header fields
        doi=grobid_doc.doi or docling_doc.doi,
        title=next(
            (t for t in [grobid_doc.title, docling_doc.title]
             if t and _validate_title(t)),
            "",
        ),
        abstract=grobid_doc.abstract or docling_doc.abstract,
        # Prefer docling for body/tables
        sections=docling_doc.sections if docling_doc.sections else grobid_doc.sections,
        tables=docling_doc.tables if docling_doc.tables else grobid_doc.tables,
        figures=docling_doc.figures if docling_doc.figures else grobid_doc.figures,
        # Prefer GROBID for structured references
        references=grobid_doc.references if grobid_doc.references else docling_doc.references,
    )

    # Preserve docling's raw markdown on the merged doc for references.md extraction
    raw_md = getattr(docling_doc, "_docling_raw_md", None)
    if raw_md is not None:
        merged._docling_raw_md = raw_md

    return merged


def _extract_references_section_from_markdown(md_text: str) -> Optional[str]:
    """Find and return the text content under a 'References' heading in markdown.

    Returns None if no references heading found.
    """
    if not md_text:
        return None
    for pat in REF_HEADING_PATTERNS:
        m = re.search(pat, md_text)
        if m:
            return md_text[m.end():].strip()
    return None


def convert_single(
    pdf_path: Path,
    output_dir: Path,
    config: Config,
    output_subdir_name: Optional[str] = None,
) -> ProcessingResult:
    """
    Convert a single PDF to Markdown and JSON.

    Args:
        pdf_path: Path to input PDF
        output_dir: Directory for output files
        config: Pipeline configuration
        output_subdir_name: Override for the per-PDF output subfolder name.
            Defaults to the PDF's stem. Used by batch processing to
            disambiguate stem collisions (e.g. multiple ``paper.pdf`` files).

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
    subdir_name = output_subdir_name or stem
    document = None
    tei_xml = None
    docling_raw_md = None

    # Provenance tracking: which extractor produced which field
    provenance = {
        "extraction_mode": None,
        "components": {},
        "grobid": {"attempted": False},
        "docling": {"attempted": False},
        "marker": {"attempted": False},
        "pymupdf4llm": {"attempted": False},
    }

    try:
        # ─── Docling-only mode ─────────────────────────────────────────────
        # Explicitly disables GROBID (set via --docling-only flag).
        if config.use_docling:
            try:
                document, docling_raw_md = _extract_with_docling(pdf_path, config)
                provenance["extraction_mode"] = "docling-only"
                provenance["docling"]["attempted"] = True
                provenance["docling"]["succeeded"] = True
                provenance["components"] = {
                    "abstract": "docling", "body": "docling",
                    "references_json": "docling", "references_md": "docling",
                }
            except Exception as e:
                import click
                click.secho(f"  SKIP {pdf_path.name}: docling failed: {e}", fg="yellow")
                errors.append(ProcessingError(
                    stage="docling",
                    error_type=type(e).__name__,
                    message=str(e),
                    recoverable=False,
                ))
                provenance["docling"]["attempted"] = True
                provenance["docling"]["succeeded"] = False
                provenance["docling"]["error"] = str(e)[:300]

        # ─── Parallel GROBID + docling (new default for FULL_GROBID) ────────
        elif config.mode == ExtractionMode.FULL_GROBID and config.parallel_extraction:
            grobid_doc = None
            grobid_err = None
            docling_doc = None
            docling_err = None
            grobid_t0 = docling_t0 = time.time()
            grobid_elapsed = docling_elapsed = 0.0

            provenance["grobid"]["attempted"] = True
            provenance["docling"]["attempted"] = True

            # Run both extractors in parallel
            with ThreadPoolExecutor(max_workers=2) as ex:
                grobid_future = ex.submit(_try_grobid, pdf_path, config)
                docling_future = ex.submit(_try_docling, pdf_path, config)

                try:
                    grobid_doc, tei_xml, grobid_err = grobid_future.result(
                        timeout=config.grobid_timeout + 30
                    )
                    grobid_elapsed = time.time() - grobid_t0
                except GrobidError as e:
                    if _is_grobid_crash(e):
                        raise  # abort batch on crash
                    grobid_err = e
                    grobid_elapsed = time.time() - grobid_t0
                except Exception as e:
                    grobid_err = e
                    grobid_elapsed = time.time() - grobid_t0

                try:
                    docling_doc, docling_err = docling_future.result(
                        timeout=config.docling_timeout + 30
                    )
                    docling_elapsed = time.time() - docling_t0
                except Exception as e:
                    docling_err = e
                    docling_elapsed = time.time() - docling_t0

            provenance["grobid"]["succeeded"] = grobid_doc is not None
            provenance["grobid"]["processing_time_s"] = round(grobid_elapsed, 2)
            if grobid_err:
                provenance["grobid"]["error"] = str(grobid_err)[:300]
            provenance["docling"]["succeeded"] = docling_doc is not None
            provenance["docling"]["processing_time_s"] = round(docling_elapsed, 2)
            if docling_err:
                provenance["docling"]["error"] = str(docling_err)[:300]

            # Decide merge strategy based on what succeeded
            if grobid_doc is not None and docling_doc is not None:
                # Happy path: both worked, merge
                document = _merge_grobid_docling(grobid_doc, docling_doc)
                provenance["extraction_mode"] = "merged-grobid-docling"
                provenance["components"] = {
                    "abstract": "grobid",
                    "body": "docling",
                    "references_json": "grobid",
                    "references_md": "docling",
                    "tables": "docling",
                }
            elif grobid_doc is not None:
                # docling failed, use grobid for everything
                document = grobid_doc
                document.metadata.extraction_mode = "grobid-only"
                provenance["extraction_mode"] = "grobid-only"
                provenance["components"] = {
                    "abstract": "grobid", "body": "grobid",
                    "references_json": "grobid", "tables": "grobid",
                }
                logger.warning(
                    f"docling failed for {pdf_path.name}, using GROBID-only: {docling_err}"
                )
                warnings.append(f"docling failed: {docling_err}")
            elif docling_doc is not None:
                # grobid failed, use docling for everything
                document = docling_doc
                document.metadata.extraction_mode = "docling-only-fallback"
                provenance["extraction_mode"] = "docling-only-fallback"
                provenance["components"] = {
                    "abstract": "docling", "body": "docling",
                    "references_json": "docling", "references_md": "docling",
                    "tables": "docling",
                }
                logger.warning(
                    f"GROBID failed for {pdf_path.name}, using docling-only: {grobid_err}"
                )
                warnings.append(f"GROBID failed: {grobid_err}")
            else:
                # Both failed — try marker if enabled
                logger.warning(
                    f"Both GROBID and docling failed for {pdf_path.name}; "
                    f"GROBID err: {grobid_err}; docling err: {docling_err}"
                )
                warnings.append(f"GROBID failed: {grobid_err}; docling failed: {docling_err}")

                if config.use_marker_fallback:
                    provenance["marker"]["attempted"] = True
                    marker_t0 = time.time()
                    logger.warning(f"Falling back to marker-pdf for {pdf_path.name}")
                    marker_doc, marker_err = _try_marker(pdf_path, config)
                    provenance["marker"]["processing_time_s"] = round(time.time() - marker_t0, 2)
                    if marker_doc is not None:
                        document = marker_doc
                        document.metadata.extraction_mode = "marker-fallback"
                        provenance["extraction_mode"] = "marker-fallback"
                        provenance["marker"]["succeeded"] = True
                        provenance["components"] = {
                            "abstract": "marker", "body": "marker",
                            "references_json": "marker", "references_md": "marker",
                        }
                        warnings.append("Recovered via marker-pdf fallback")
                    else:
                        provenance["marker"]["succeeded"] = False
                        provenance["marker"]["error"] = str(marker_err)[:300]

                # Final-last-resort tier: pymupdf4llm runs if we still don't
                # have a document and the fallback is enabled. This fires when
                # marker-pdf was disabled, wasn't installed, or failed — we'd
                # rather have flat markdown than record a total failure.
                if document is None and config.use_pymupdf_fallback:
                    provenance["pymupdf4llm"]["attempted"] = True
                    pymupdf_t0 = time.time()
                    logger.warning(
                        f"Falling back to pymupdf4llm (last-resort) for {pdf_path.name}"
                    )
                    pymupdf_doc, pymupdf_err = _try_pymupdf4llm(pdf_path, config)
                    provenance["pymupdf4llm"]["processing_time_s"] = round(
                        time.time() - pymupdf_t0, 2
                    )
                    if pymupdf_doc is not None:
                        document = pymupdf_doc
                        document.metadata.extraction_mode = "pymupdf4llm-fallback"
                        provenance["extraction_mode"] = "pymupdf4llm-fallback"
                        provenance["pymupdf4llm"]["succeeded"] = True
                        provenance["components"] = {
                            "abstract": "pymupdf4llm", "body": "pymupdf4llm",
                            "references_json": "pymupdf4llm",
                            "references_md": "pymupdf4llm",
                        }
                        warnings.append(
                            "Recovered via pymupdf4llm last-resort fallback "
                            "(no structured references — downstream consumers "
                            "should treat this record as best-effort)"
                        )
                    else:
                        provenance["pymupdf4llm"]["succeeded"] = False
                        provenance["pymupdf4llm"]["error"] = str(pymupdf_err)[:300]

                # If nothing recovered the document, record the failure now.
                if document is None:
                    if config.use_marker_fallback and provenance["marker"].get("error"):
                        errors.append(ProcessingError(
                            stage="marker-fallback",
                            error_type="MarkerFailed",
                            message=provenance["marker"]["error"],
                            recoverable=False,
                        ))
                    if config.use_pymupdf_fallback and provenance["pymupdf4llm"].get("error"):
                        errors.append(ProcessingError(
                            stage="pymupdf4llm-fallback",
                            error_type="PymupdfFailed",
                            message=provenance["pymupdf4llm"]["error"],
                            recoverable=False,
                        ))
                    if not (config.use_marker_fallback or config.use_pymupdf_fallback):
                        errors.append(ProcessingError(
                            stage="parallel-extraction",
                            error_type="BothExtractorsFailed",
                            message=f"GROBID: {grobid_err}; docling: {docling_err}",
                            recoverable=False,
                        ))

        # ─── Legacy GROBID sequential path (--no-parallel) ─────────────────
        elif config.mode == ExtractionMode.FULL_GROBID:
            provenance["grobid"]["attempted"] = True
            try:
                tei_xml, document = extract_with_grobid(pdf_path, config)
                provenance["grobid"]["succeeded"] = True
                provenance["extraction_mode"] = "grobid-only"
            except GrobidScannedPdfError as e:
                if config.no_ocr:
                    import click
                    click.secho(f"  SKIP {pdf_path.name}: scanned PDF (no text layer)", fg="yellow")
                    errors.append(ProcessingError(
                        stage="grobid",
                        error_type="GrobidScannedPdfError",
                        message=f"Scanned PDF (no text layer); OCR disabled via --noocr: {e}",
                        recoverable=False,
                    ))
                    warnings.append("Scanned PDF detected; OCR disabled via --noocr")
                    provenance["grobid"]["succeeded"] = False
                else:
                    logger.warning(f"Scanned PDF detected ({pdf_path.name}), using Tesseract OCR directly")
                    warnings.append("Scanned PDF detected (no text layer); using Tesseract OCR")
                    document = extract_with_ocr(pdf_path, config)
                    document.metadata.extraction_mode = "ocr-scanned"
                    provenance["extraction_mode"] = "ocr-scanned"
            except GrobidError as e:
                if _is_grobid_crash(e):
                    raise
                if config.no_ocr:
                    import click
                    click.secho(f"  SKIP {pdf_path.name}: {e}", fg="yellow")
                    errors.append(ProcessingError(
                        stage="grobid",
                        error_type="GrobidError",
                        message=f"GROBID failed; OCR disabled via --noocr: {e}",
                        recoverable=False,
                    ))
                    warnings.append(f"GROBID failed; OCR disabled via --noocr: {e}")
                    provenance["grobid"]["succeeded"] = False
                else:
                    logger.warning(f"GROBID failed for {pdf_path.name}, falling back to OCR: {e}")
                    warnings.append(f"GROBID failed, falling back to Tesseract OCR: {e}")
                    document = extract_with_ocr(pdf_path, config)
                    document.metadata.extraction_mode = "ocr-fallback"
                    provenance["extraction_mode"] = "ocr-fallback"

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

    # Fallback: fetch title from DOI API if local extraction failed
    if document and document.doi:
        title = document.title or ""
        if not title or not _validate_title(title):
            api_title = fetch_title_from_doi(document.doi)
            if api_title:
                document.title = api_title
                logger.info(f"Title from DOI API for {pdf_path.name}: {api_title[:80]}")

    # Single-markdown mode: write one flat <slug>.md in output_dir; skip subfolder/json/etc.
    if config.single_markdown and document:
        slug = re.sub(r'[^\w.\-]', '--', (document.doi or stem))
        md = render_single_markdown(document, docling_raw_md)
        out_path = output_dir / f"{slug}.md"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        processing_time = time.time() - start_time
        return ProcessingResult(
            pdf_path=str(pdf_path),
            status="success",
            output_dir=str(output_dir),
            errors=errors,
            warnings=warnings,
            processing_time_seconds=round(processing_time, 2),
        )

    # Abstract-only mode: save {DOI}_abstract.md directly in output_dir, skip everything else
    if config.abstract_only and document:
        doi_slug = re.sub(r'[^\w.\-]', '--', (document.doi or stem))
        abstract_md = render_abstract_md(document)
        abstract_path = output_dir / f"{doi_slug}_abstract.md"
        abstract_path.write_text(abstract_md, encoding="utf-8")

        # In docling mode, also save the full markdown
        if docling_raw_md:
            full_md_path = output_dir / f"{doi_slug}.md"
            full_md_path.write_text(docling_raw_md, encoding="utf-8")

        processing_time = time.time() - start_time
        return ProcessingResult(
            pdf_path=str(pdf_path),
            status="success",
            output_dir=str(output_dir),
            errors=errors,
            warnings=warnings,
            processing_time_seconds=round(processing_time, 2),
        )

    # Output dir: subfolder named by PDF stem
    # NOTE: we do NOT create the folder here — we create it lazily right
    # before writing the first output file, so that failed/crashed extractions
    # don't leave empty "phantom" folders behind.
    if document:
        effective_output_dir = output_dir / subdir_name
    else:
        effective_output_dir = output_dir

    abstract_path = effective_output_dir / "abstract.md"
    body_path = effective_output_dir / "body.md"
    refs_path = effective_output_dir / "references.json"
    refs_md_path = effective_output_dir / "references.md"
    tei_path = effective_output_dir / f"{stem}.tei.xml"
    provenance_path = effective_output_dir / "provenance.json"

    # Validate extraction quality
    if document:
        quality_score, quality_issues = validate_document(document)
        document.metadata.quality_score = quality_score
        document.metadata.quality_issues = quality_issues
        if quality_issues:
            logger.info(f"Quality score {quality_score:.2f} for {pdf_path.name}: {', '.join(quality_issues)}")
        
        # OCR fallback: if quality is below threshold, retry with OCR
        if needs_ocr_fallback(quality_score) and not config.no_ocr:
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
                and not config.no_ocr
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

    # CrossRef DOI enrichment: trigger when DOI rate is below threshold
    if document and document.references and config.crossref_enrich_threshold > 0:
        ref_count = len(document.references)
        doi_count = sum(1 for r in document.references if r.doi)
        doi_rate = doi_count / ref_count
        if doi_rate < config.crossref_enrich_threshold:
            logger.info(
                f"DOI rate {doi_rate:.0%} ({doi_count}/{ref_count}) below threshold "
                f"{config.crossref_enrich_threshold:.0%} — running CrossRef enrichment"
            )
            try:
                added = _enrich_references_crossref(document, config)
                if added:
                    # Re-score now that more DOIs are present
                    quality_score, quality_issues = validate_document(document)
                    document.metadata.quality_score = quality_score
                    document.metadata.quality_issues = quality_issues
            except Exception as e:
                logger.warning(f"CrossRef DOI enrichment failed: {e}")

    # Render outputs if we have a document
    result_output_dir = None

    if document:
        try:
            # Render content first, THEN create the output folder, so that
            # any render failure doesn't leave an empty phantom folder.
            abstract_md = render_abstract_md(document)
            body_md = render_body_md(document)
            refs_json = render_references_json(document)

            # references.md: extract the References section from docling's raw
            # markdown (if available). Only present when docling produced output.
            refs_md_text = None
            docling_raw = getattr(document, "_docling_raw_md", None) \
                or getattr(document, "_marker_refs_md", None)
            if docling_raw:
                refs_md_text = _extract_references_section_from_markdown(docling_raw)
            # Fallback: check if marker stashed its raw markdown
            if refs_md_text is None:
                marker_raw = getattr(document, "_marker_raw_md", None)
                if marker_raw:
                    refs_md_text = _extract_references_section_from_markdown(marker_raw)
            # Final fallback: pymupdf4llm last-resort tier
            if refs_md_text is None:
                pymupdf_refs = getattr(document, "_pymupdf_refs_md", None)
                if pymupdf_refs:
                    refs_md_text = pymupdf_refs
                else:
                    pymupdf_raw = getattr(document, "_pymupdf_raw_md", None)
                    if pymupdf_raw:
                        refs_md_text = _extract_references_section_from_markdown(pymupdf_raw)

            # All renders succeeded — create folder and write files atomically
            effective_output_dir.mkdir(parents=True, exist_ok=True)
            abstract_path.write_text(abstract_md, encoding="utf-8")
            body_path.write_text(body_md, encoding="utf-8")
            refs_path.write_text(refs_json, encoding="utf-8")
            if refs_md_text is not None:
                refs_md_path.write_text(refs_md_text, encoding="utf-8")

            # Write provenance.json recording which extractor produced what
            try:
                provenance_path.write_text(
                    json.dumps(provenance, indent=2), encoding="utf-8"
                )
            except Exception as pe:
                logger.warning(f"Failed to write provenance.json: {pe}")

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
            # Clean up partial output: remove any files written and the folder
            # if we created it but ultimately failed.
            try:
                for p in (abstract_path, body_path, refs_path, refs_md_path,
                          tei_path, provenance_path):
                    if p.exists():
                        p.unlink()
                if effective_output_dir.exists() and effective_output_dir != output_dir \
                        and not any(effective_output_dir.iterdir()):
                    effective_output_dir.rmdir()
            except OSError:
                pass

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


def _enrich_references_crossref(document: DocumentModel, config: Config) -> int:
    """
    For references lacking a DOI, query the CrossRef API to recover missing DOIs.

    Uses the bibliographic query endpoint with title + first author + year.
    A match is accepted only when the CrossRef relevance score exceeds
    _CROSSREF_MIN_SCORE and the result year agrees with the reference year
    (within 1 year) when both are available.

    Modifies document.references in-place.
    Returns the count of DOIs newly added.
    """
    import requests as _requests

    _CROSSREF_MIN_SCORE = 80.0
    _CROSSREF_BASE = "https://api.crossref.org/works"
    _RATE_LIMIT_SLEEP = 0.11  # ~9 req/s, well within polite-pool limits

    # Skip references with suspicious titles (filename-derived DOIs like
    # "10.1109--metric.2002.1011340", or titles that look like raw DOIs).
    # _validate_title is already imported from fast.py.
    _doi_like = re.compile(r"^10\.\d{4,}/|^10\.\d{4,}--")

    candidates = [
        r for r in document.references
        if not r.doi
        and r.title
        and len(r.title) >= 15
        and not _doi_like.match(r.title)
        and _validate_title(r.title)
    ]
    if not candidates:
        return 0

    logger.info(
        f"CrossRef DOI enrichment: querying {len(candidates)} references missing DOIs"
    )

    mailto = config.crossref_mailto or ""
    added = 0

    for ref in candidates:
        query_parts = [ref.title]
        if ref.authors:
            query_parts.append(ref.authors[0].split()[-1])  # first author surname
        query = " ".join(query_parts)

        params: dict = {
            "query.bibliographic": query,
            "rows": "3",
            "select": "DOI,title,published,score",
        }
        if mailto:
            params["mailto"] = mailto

        try:
            resp = _requests.get(
                _CROSSREF_BASE,
                params=params,
                timeout=10,
                headers={"User-Agent": "pdf4llm/1.0 (mailto:crossref@pdf4llm)"},
            )
            time.sleep(_RATE_LIMIT_SLEEP)

            if resp.status_code != 200:
                continue

            data = resp.json()
            items = data.get("message", {}).get("items", [])
            if not items:
                continue

            top = items[0]
            score = top.get("score", 0.0)
            if score < _CROSSREF_MIN_SCORE:
                continue

            # Year guard: reject if years disagree by more than 1
            if ref.year:
                cr_year = None
                published = top.get("published", {})
                parts = published.get("date-parts", [[]])
                if parts and parts[0]:
                    try:
                        cr_year = int(parts[0][0])
                    except (ValueError, TypeError):
                        pass
                if cr_year and abs(cr_year - ref.year) > 1:
                    continue

            raw_doi = top.get("DOI", "")
            doi = _normalize_doi(raw_doi) if raw_doi else None
            if not doi:
                continue

            # Update the reference in-place (Pydantic models are mutable via assignment)
            idx = document.references.index(ref)
            document.references[idx] = ref.model_copy(update={"doi": doi})
            added += 1

        except Exception as e:
            logger.debug(f"CrossRef lookup failed for '{ref.title[:60]}': {e}")
            time.sleep(_RATE_LIMIT_SLEEP)
            continue

    if added:
        document.metadata.references_with_doi = sum(
            1 for r in document.references if r.doi
        )
        logger.info(
            f"CrossRef DOI enrichment: found {added}/{len(candidates)} missing DOIs"
        )
    else:
        logger.info("CrossRef DOI enrichment: no new DOIs found")

    return added


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
    - Parallel processing with ThreadPoolExecutor
    - Checkpoint/resume for interrupted batches
    - Progress tracking with tqdm
    - Clean shutdown on Ctrl+C (cancels pending work, finishes in-flight only)
    """

    def __init__(self, config: Config, checkpoint_file: Optional[Path] = None):
        self.config = config
        self.checkpoint_file = checkpoint_file
        self.processed: Set[str] = self._load_checkpoint()
        # Set by the SIGINT handler to request cooperative shutdown.
        # Worker loop polls this and stops scheduling new work when True.
        self._shutdown_requested = False

    def _load_checkpoint(self) -> Set[str]:
        """Load checkpoint file and validate entries against disk.

        Drops any entry whose output files don't actually exist — avoids
        "phantom" entries where a previous run crashed mid-render, marked a
        PDF as processed, but never wrote the output files. Those PDFs will
        be retried on the next --resume.
        """
        if self.checkpoint_file and self.checkpoint_file.exists():
            try:
                data = json.loads(self.checkpoint_file.read_text())
                raw = set(data.get("processed", []))
                # Determine the output_dir where we'd validate — it's the parent
                # of the checkpoint file (pdf4llm convention).
                output_dir = self.checkpoint_file.parent
                validated = set()
                stale = 0
                for path_str in raw:
                    p = Path(path_str)
                    stem = p.stem
                    # When stems collide (e.g. all PDFs named paper.pdf in
                    # per-DOI subdirs), the output lives under the parent dir
                    # name, not the stem. Try both so stale-detection works.
                    folder = output_dir / stem
                    if not folder.exists() and p.parent.name:
                        folder = output_dir / p.parent.name
                    # A valid entry MUST have all three core files on disk.
                    # abstract_only mode uses a different layout, so skip this
                    # check when running in that mode.
                    if self.config.abstract_only or self.config.single_markdown:
                        validated.add(path_str)
                        continue
                    if (folder / "abstract.md").exists() and \
                       (folder / "body.md").exists() and \
                       (folder / "references.json").exists():
                        validated.add(path_str)
                    else:
                        stale += 1
                        # Clean up any empty phantom folder
                        if folder.exists() and not any(folder.iterdir()):
                            try:
                                folder.rmdir()
                            except OSError:
                                pass
                if stale > 0:
                    logger.warning(
                        f"Checkpoint had {stale} stale entries (missing output "
                        f"on disk) — removing them and will retry those PDFs"
                    )
                    # Rewrite the checkpoint file with only valid entries
                    try:
                        self.checkpoint_file.write_text(json.dumps({
                            "processed": sorted(validated),
                            "last_updated": datetime.utcnow().isoformat(),
                        }))
                    except Exception as e:
                        logger.warning(f"Failed to rewrite validated checkpoint: {e}")
                return validated
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return set()

    @staticmethod
    def _compute_output_names(pdfs: List[Path]) -> Dict[Path, str]:
        """Map each PDF to a unique output-subfolder name.

        Defaults to the PDF stem. When stems collide (e.g. scraper layouts
        where every file is named ``paper.pdf`` under a per-paper directory),
        falls back to the parent directory name, then to
        ``<parent>__<stem>`` if needed. Without this, all colliding inputs
        would write to the same output folder and overwrite each other.
        """
        stem_counts: Dict[str, int] = {}
        for p in pdfs:
            stem_counts[p.stem] = stem_counts.get(p.stem, 0) + 1
        names: Dict[Path, str] = {}
        used: Set[str] = set()
        for p in pdfs:
            if stem_counts[p.stem] == 1:
                name = p.stem
            elif p.parent.name and p.parent.name not in used:
                name = p.parent.name
            elif p.parent.name:
                name = f"{p.parent.name}__{p.stem}"
            else:
                name = p.stem
            names[p] = name
            used.add(name)
        return names

    @staticmethod
    def _move_pdf(pdf_path: Path, output_dir: Path):
        """Move the source PDF into its output folder."""
        src = Path(pdf_path).resolve()
        dst = output_dir / src.name
        if src != dst and src.exists():
            shutil.move(str(src), str(dst))

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

        # Filter already processed if resuming from checkpoint.
        # When --movepdf is used with input_dir == output_dir, the PDF moves
        # from ./foo.pdf → ./foo/foo.pdf after processing, so the path stored
        # in the checkpoint (original location) won't match what the glob finds
        # on the next run (moved location). Guard against this with a stem-based
        # fallback: if any processed entry shares the same filename stem, skip.
        if resume and self.processed:
            original_count = len(pdfs)
            processed_stems = {Path(p).stem for p in self.processed}
            pdfs = [
                p for p in pdfs
                if str(p) not in self.processed and p.stem not in processed_stems
            ]
            logger.info(f"Resuming: {original_count - len(pdfs)} already processed, "
                       f"{len(pdfs)} remaining")

        # Disambiguate output subfolder names so PDFs that share a stem
        # (e.g. every paper.pdf in a <doi>/paper.pdf layout) don't overwrite
        # each other's output.
        out_names = self._compute_output_names(pdfs)

        # Filter PDFs that already have output files
        if skip_existing:
            original_count = len(pdfs)
            pdfs_to_process = []
            for pdf in pdfs:
                stem = pdf.stem
                if self.config.abstract_only:
                    # Check stem-based name and DOI-from-filename-based name
                    if (output_dir / f"{stem}_abstract.md").exists():
                        continue
                    doi_from_fn = _extract_doi_from_filename(pdf)
                    if doi_from_fn:
                        doi_slug = re.sub(r'[^\w.\-]', '--', doi_from_fn)
                        if (output_dir / f"{doi_slug}_abstract.md").exists():
                            continue
                elif self.config.single_markdown:
                    if (output_dir / f"{stem}.md").exists():
                        continue
                    doi_from_fn = _extract_doi_from_filename(pdf)
                    if doi_from_fn:
                        doi_slug = re.sub(r'[^\w.\-]', '--', doi_from_fn)
                        if (output_dir / f"{doi_slug}.md").exists():
                            continue
                else:
                    out = output_dir / out_names[pdf]
                    if out.exists() and (out / "abstract.md").exists() and (out / "body.md").exists():
                        continue
                pdfs_to_process.append(pdf)
            pdfs = pdfs_to_process
            skipped = original_count - len(pdfs)
            if skipped > 0:
                logger.info(f"Skipping {skipped} PDFs with existing output, "
                           f"{len(pdfs)} remaining")

        if not pdfs:
            return BatchResult(
                total_files=len(self.processed),
                successful=len(self.processed),
            )

        # Check GROBID availability if needed
        if self.config.requires_grobid() and not self.config.use_docling:
            client = GrobidClient(self.config)
            if not client.is_alive():
                if self.config.auto_start_grobid:
                    _start_grobid_docker(mode=self.config.grobid_docker_mode)
                    if not client.is_alive():
                        raise GrobidError(
                            f"GROBID still not available at {self.config.grobid_url} "
                            "after auto-start attempt."
                        )
                else:
                    raise GrobidError(
                        f"GROBID server not available at {self.config.grobid_url}. "
                        "Start GROBID with: docker run --gpus all -p 8070:8070 lfoppiano/grobid:0.8.0"
                    )
            logger.info(f"GROBID server ready (version: {client.get_version()})")

        # Process files
        start_time = time.time()
        # Track counts directly instead of accumulating all results in memory.
        # For 33K PDFs the full results list would hold tens of thousands of
        # objects. Only keep failures for diagnostics.
        n_successful = 0
        n_partial = 0
        n_failed = 0
        failed_results: List[ProcessingResult] = []
        error_summary: Dict[str, int] = {}

        # Install SIGINT handler for clean shutdown: first Ctrl+C sets the
        # shutdown flag (finish in-flight, skip remaining). Second Ctrl+C
        # forces immediate exit.
        import signal as _signal
        prior_sigint = _signal.getsignal(_signal.SIGINT)
        shutdown_count = [0]
        force_exit = [False]

        def _sigint_handler(signum, frame):
            shutdown_count[0] += 1
            if shutdown_count[0] == 1:
                self._shutdown_requested = True
                tqdm.write("\n⚠️  Ctrl+C received. Finishing in-flight PDFs then stopping. "
                           "Press Ctrl+C again to force immediate exit.\n", file=sys.stderr)
            else:
                force_exit[0] = True
                tqdm.write("\n⚠️  Force exit.\n", file=sys.stderr)
                # Silence further logging from threads still running
                logging.getLogger().setLevel(logging.CRITICAL)
                # Restore original handler and re-raise
                _signal.signal(_signal.SIGINT, prior_sigint)
                raise KeyboardInterrupt()

        try:
            _signal.signal(_signal.SIGINT, _sigint_handler)
        except (ValueError, OSError):
            # Can't install signal handler (e.g., not in main thread)
            pass

        # Swap in a tqdm-aware logging handler for the duration of the batch so
        # that log messages don't corrupt the progress bar.
        root_logger = logging.getLogger()
        _original_handlers = root_logger.handlers[:]
        _original_level = root_logger.level
        _tqdm_handler = _TqdmLoggingHandler(stream=sys.stderr)
        _tqdm_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root_logger.handlers = [_tqdm_handler]

        def _tally(result: ProcessingResult, pdf_path: Path) -> None:
            """Update counters and checkpoint for one completed result."""
            nonlocal n_successful, n_partial, n_failed
            if result.status == "success":
                n_successful += 1
            elif result.status == "partial":
                n_partial += 1
            else:
                n_failed += 1
                failed_results.append(result)
            if result.status != "failed":
                self._save_checkpoint(str(pdf_path))
                if move_pdf and result.output_dir:
                    self._move_pdf(pdf_path, Path(result.output_dir))
            for error in result.errors:
                error_summary[error.error_type] = error_summary.get(error.error_type, 0) + 1

        try:
            # Sequential path (easier to debug)
            if self.config.workers == 1:
                for pdf_path in tqdm(pdfs, desc="Converting PDFs"):
                    if self._shutdown_requested:
                        logger.info("Shutdown requested — stopping sequential loop")
                        break
                    try:
                        result = convert_single(
                            pdf_path, output_dir, self.config,
                            output_subdir_name=out_names.get(pdf_path),
                        )
                    except GrobidError as e:
                        if _is_grobid_crash(e):
                            raise _build_grobid_oom_error(pdf_path.name, n_successful + n_partial + n_failed)
                        raise
                    _tally(result, pdf_path)
            else:
                # Parallel processing — use a throttled sliding window so we
                # never hold more than (workers * 4) completed futures in memory
                # at once. Submitting all N futures upfront would keep every
                # ProcessingResult alive until the last one finishes.
                max_in_flight = self.config.workers * 4
                pending: Dict = {}  # future → pdf_path
                pdf_iter = iter(pdfs)
                progress = tqdm(total=len(pdfs), desc="Converting PDFs")

                def _submit_next() -> bool:
                    try:
                        pdf = next(pdf_iter)
                        f = executor.submit(
                            _process_single_wrapper, pdf, output_dir, self.config,
                            out_names.get(pdf),
                        )
                        pending[f] = pdf
                        return True
                    except StopIteration:
                        return False

                executor = ThreadPoolExecutor(max_workers=self.config.workers)
                try:
                    # Fill the initial window
                    for _ in range(max_in_flight):
                        if not _submit_next():
                            break

                    while pending:
                        if self._shutdown_requested:
                            cancelled = sum(1 for f in list(pending) if not f.done() and f.cancel())
                            logger.info(f"Shutdown requested — cancelled {cancelled} pending futures")
                            break

                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            pdf_path = pending.pop(future)
                            try:
                                result = future.result()
                                _tally(result, pdf_path)
                            except GrobidError as e:
                                if _is_grobid_crash(e):
                                    for f in list(pending):
                                        f.cancel()
                                    raise _build_grobid_oom_error(pdf_path.name, n_successful + n_partial + n_failed)
                                logger.error(f"Failed to process {pdf_path}: {e}")
                                _tally(_build_failed_processing_result(pdf_path, e), pdf_path)
                            except Exception as e:
                                logger.error(f"Failed to process {pdf_path}: {e}")
                                _tally(_build_failed_processing_result(pdf_path, e), pdf_path)
                            finally:
                                # Drop the future reference so the result can be GC'd,
                                # then run GC periodically to release docling memory.
                                del future
                                progress.update(1)
                            # Submit one more to maintain the window
                            if not self._shutdown_requested:
                                _submit_next()
                finally:
                    progress.close()
                    if force_exit[0]:
                        # Force exit: mark threads daemon so Python's atexit handler
                        # won't block joining them, then skip waiting.
                        for t in getattr(executor, '_threads', set()):
                            t.daemon = True
                        try:
                            executor.shutdown(wait=False, cancel_futures=True)
                        except TypeError:
                            executor.shutdown(wait=False)
                    else:
                        # Graceful shutdown: wait for in-flight tasks to finish.
                        # shutdown(cancel_futures=True) was added in Python 3.9.
                        try:
                            executor.shutdown(wait=True, cancel_futures=self._shutdown_requested)
                        except TypeError:
                            executor.shutdown(wait=not self._shutdown_requested)
                    # Final GC pass to release docling model state from all threads.
                    gc.collect()
        finally:
            # Restore the original logging handlers and level
            root_logger.handlers = _original_handlers
            root_logger.setLevel(_original_level)
            # Restore the prior SIGINT handler
            try:
                _signal.signal(_signal.SIGINT, prior_sigint)
            except (ValueError, OSError):
                pass

        total_time = time.time() - start_time

        batch_result = BatchResult(
            total_files=n_successful + n_partial + n_failed,
            successful=n_successful,
            partial=n_partial,
            failed=n_failed,
            results=failed_results,
            error_summary=error_summary,
            total_time_seconds=round(total_time, 2),
        )

        return batch_result


def _process_single_wrapper(
    pdf_path: Path,
    output_dir: Path,
    config: Config,
    output_subdir_name: Optional[str] = None,
) -> ProcessingResult:
    """Wrapper for convert_single to work with ProcessPoolExecutor."""
    return convert_single(pdf_path, output_dir, config, output_subdir_name=output_subdir_name)


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
