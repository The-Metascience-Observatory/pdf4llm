# pdf4llm — Agent Orientation

High-quality PDF → structured Markdown/JSON converter for scientific papers. Optimized for downstream LLM pipelines. AGPL-3.0-or-later licensed (PyMuPDF dependency forces AGPL).

---

## Package layout

```
pdf4llm/
├── cli.py               — Click CLI: `convert`, `batch`, `health-check` commands
├── config.py            — Config dataclass + ExtractionMode enum
├── models.py            — Pydantic models: DocumentModel, ProcessingResult, BatchResult, Reference, Section, Table
├── pipeline.py          — convert_single(), BatchProcessor, merge logic, docling wrapper
├── validation.py        — validate_document() → (score: float, issues: list[str])
├── utils.py             — misc helpers
├── extractors/
│   ├── grobid.py        — GROBID HTTP client, TEI parser, _normalize_doi
│   ├── fast.py          — PyMuPDF + pdfplumber path, _extract_doi_from_filename, _validate_title
│   ├── ocr_fallback.py  — Tesseract OCR path (extract_with_ocr)
│   ├── tables.py        — table quality assessment
│   ├── charts.py        — vision-model chart extraction (optional, Ollama/Anthropic/OpenAI)
│   ├── marker_ext.py    — marker-pdf fallback tier integration
│   ├── pymupdf4llm_ext.py — last-resort tier (flat markdown, no ML)
│   └── doi_lookup.py    — DOI→title lookup (Crossref/DataCite/Unpaywall/S2/OpenAlex)
└── renderers/
    ├── markdown.py      — render_abstract_md, render_body_md, render_tables_md
    └── json_output.py   — render_references_json, render_json
```

---

## Extraction pipeline (`full-grobid` default mode)

For each PDF, `convert_single()` in `pipeline.py` runs:

```
GROBID ──┐
          ├──► _merge_grobid_docling() → merged DocumentModel
docling ──┘
     │
     └─ both fail → marker-pdf → pymupdf4llm → hard fail
```

- GROBID (`_try_grobid`): structured header + references. Best for title/abstract/refs JSON.
- docling (`_try_docling` → `_extract_with_docling`): body text, tables, multi-column layouts. No structured references.
- Merge (`_merge_grobid_docling`): GROBID wins on metadata/refs; docling wins on body/tables.
- Quality score < threshold → OCR fallback (`extract_with_ocr`) then CrossRef DOI enrichment.
- marker-pdf fallback fires if both GROBID and docling fail.
- pymupdf4llm fires if marker also fails.

**Key function signatures:**
```python
# pipeline.py
convert_single(pdf_path, output_dir, config, output_subdir_name=None) -> ProcessingResult
BatchProcessor(config, checkpoint_file=None).process_folder(input_dir, output_dir, resume, skip_existing, move_pdf) -> BatchResult

# validation.py
validate_document(document: DocumentModel) -> (score: float, issues: list[str])
# score < 0.6 triggers OCR fallback; quality issues are logged at INFO

# config.py
Config(mode=ExtractionMode.FULL_GROBID, workers=2, parallel_extraction=True, ...)
```

---

## Output layout (one subfolder per PDF)

```
{subdir_name}/          ← named by PDF stem, or parent dir when stems collide
├── abstract.md         — title + abstract (GROBID)
├── body.md             — main text with sections/tables (docling)
├── references.json     — [{authors, title, journal, year, doi, …}, …] (GROBID)
├── references.md       — references as prose (docling raw markdown)
├── tables.md           — extracted tables
└── provenance.json     — per-field attribution: which extractor produced what
```

**Stem collision:** when all PDFs in a tree are named `paper.pdf` (e.g. `<doi>/paper.pdf` scraper layouts), `BatchProcessor._compute_output_names()` falls back to using the parent directory name as the subfolder name. This prevents all outputs overwriting a single `./paper/` folder.

---

## Memory-conscious batch design

For large runs (tens of thousands of PDFs), the parallel path uses a **throttled sliding window** (`workers * 4` in-flight futures at once) so completed results are GC'd immediately rather than accumulating. `convert_single` explicitly `del`s the docling `converter` and `ConversionResult` after markdown extraction to release GPU/CPU memory promptly.

---

## Config reference (key fields)

| Field | Default | Notes |
|---|---|---|
| `mode` | `FULL_GROBID` | `FULL_GROBID`, `HYBRID`, `FAST` |
| `workers` | `min(10, cpu_count)` | 2 = sweet spot on 32 GB; 4 on 64 GB+ |
| `parallel_extraction` | `True` | GROBID + docling in parallel per PDF |
| `use_docling` | `False` | `True` = docling-only (no GROBID, no structured refs) |
| `docling_use_gpu` | `False` | Enable CUDA for docling |
| `use_marker_fallback` | `True` | marker-pdf tier after GROBID+docling fail |
| `use_pymupdf_fallback` | `True` | pymupdf4llm last-resort tier |
| `abstract_only` | `False` | Write `{doi}_abstract.md` only, no subfolder |
| `crossref_enrich_threshold` | `0.5` | Run CrossRef DOI enrichment when DOI rate < 50% |
| `auto_start_grobid` | `True` | Auto-start GROBID at `../grobid/` |

**Env vars:**
- `PDF4LLM_GPU_SLOTS=N` — how many workers may use CUDA simultaneously (default 1)
- `PDF4LLM_DOCLING_CPU=1` — force CPU for all workers (overrides `docling_use_gpu`)
- `PDF4LLM_MODE`, `PDF4LLM_WORKERS`, `PDF4LLM_GROBID_URL`, `PDF4LLM_CROSSREF_THRESHOLD`, `PDF4LLM_CROSSREF_MAILTO`

---

## Common CLI usage

```bash
# Default: GROBID + docling parallel merge
pdf4llm convert paper.pdf -o output/

# Batch a directory (throttled window, checkpoint/resume, move PDFs on success)
pdf4llm batch ./pdfs/ -o ./output/ --workers 2 --movepdf --resume

# Batch when PDFs are already in <doi>/paper.pdf layout
pdf4llm batch ./snpedia_pdfs/ -o ./snpedia_pdfs/ --workers 2

# docling-only (no GROBID; no structured refs)
pdf4llm convert paper.pdf -o output/ --docling-only

# Fast mode (PyMuPDF only; no ML)
pdf4llm convert paper.pdf -o output/ --mode fast

# Debug with verbose logging
pdf4llm convert paper.pdf -o output/ -v
```

---

## GROBID setup (required for full-grobid mode)

GROBID lives at `../grobid/` relative to this repo (sibling directory). pdf4llm auto-starts it via the local build; DeLFT (deep learning models) requires a Python 3.10 venv at `~/venv_grobid` with:
```
jep==4.2.2  delft==0.3.4  transformers==4.44.2  Pillow
```
Missing `Pillow` causes every GROBID request to return HTTP 500 while `/api/isalive` still returns 200.

Docker alternative: `./docker/start-grobid.sh delft` (then pass `--no-auto-grobid`).

---

## Key invariants

- **Output is always a subfolder**, never flat files in `output_dir` (except `abstract_only` mode which writes `{doi}_abstract.md` flat).
- **Skip-existing** checks for `abstract.md + body.md` in the per-PDF subfolder. No partial-output retries without deleting the folder first.
- **Checkpoint** (`BatchProcessor`) tracks absolute PDF paths; stem-collision-aware via `_compute_output_names`.
- **`BatchResult.results`** in a large batch contains **only failures** (not all 33K entries) — don't rely on it for success records.
- **GROBID crash detection**: `_is_grobid_crash()` distinguishes OOM from normal 500 errors and aborts the whole batch (re-raise), because a crashed GROBID JVM won't recover on its own.
- Quality threshold for OCR fallback is set inside `needs_ocr_fallback()` in `ocr_fallback.py` (currently 0.6).

---

## Testing

No test suite yet. Validate changes by:
1. `python -c "from pdf4llm.pipeline import convert_single, BatchProcessor; print('ok')"` — import smoke test
2. `pdf4llm convert <a_real_paper.pdf> -o /tmp/test_out/ -v` — single-PDF end-to-end
3. `pdf4llm batch <dir_with_few_pdfs> -o /tmp/batch_out/ --workers 1` — batch smoke test
4. Check `/tmp/test_out/<stem>/abstract.md`, `body.md`, `references.json`, `provenance.json` are non-empty

---

## Common pitfalls

- Don't accumulate large lists of `ProcessingResult` objects during batch — use the counter pattern in `_tally()`.
- Don't call `DocumentConverter(...)` outside `_extract_with_docling` — models are heavy; they must be `del`'d promptly.
- The `futures` dict in the parallel path is a **throttled window** (not all-at-once) — don't revert to submitting all futures upfront.
- `_compute_output_names()` must be called before both the skip-existing loop and the conversion loop to ensure consistent naming.
- When adding new output files, write them inside `effective_output_dir` (the per-PDF subfolder), not `output_dir`.
