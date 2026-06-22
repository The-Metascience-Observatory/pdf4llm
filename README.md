# pdf4llm

High-quality PDF → structured-markdown converter optimized for LLM ingestion of scientific papers.

**pdf4llm's default mode runs two best-in-class extractors in parallel per PDF and merges their output**:

- **GROBID (with DeLFT)** — state-of-the-art for academic headers and structured references
- **docling** — state-of-the-art for body text and table layout on non-standard PDFs
- **marker-pdf** — fallback tier when both GROBID and docling fail (ML-based, ~3 GB models)
- **pymupdf4llm** — last-resort safety net after marker-pdf fails (fast, pure Python, no ML)

For each paper, the output matches the expectations of downstream LLM pipelines (abstract, body, structured references, references as prose, conversion provenance trail).

---

## Why parallel merge?

No single PDF extractor is best at everything:

| Tool       | Best at                                       | Weak at                                    |
|------------|-----------------------------------------------|--------------------------------------------|
| **GROBID** + DeLFT  | Title/author/abstract extraction, reference parsing into structured JSON (authors, journal, year, DOI) | HTML-derived PDFs (Collabra, some preprint servers), multi-column layouts with embedded images |
| **docling**         | Body text, complex tables, multi-column layouts, non-standard publisher templates | No structured references — it produces markdown prose only |
| **marker-pdf**      | Heavily scanned / image-heavy PDFs, broken text layers, layout-corrupted files | Slower; no structured references |
| **pymupdf4llm**     | Speed (~1–3 s per PDF), low memory, works on virtually any text-layer PDF — ideal "can we at least get *something* out" safety net | Flat markdown only; weak heading detection; no structured references, tables, or figures |

By running GROBID and docling in parallel and taking the best output from each, pdf4llm produces significantly better extractions than either tool alone — especially for the ~10% of papers that GROBID-only or docling-only pipelines fail on.

---

## Output layout

Each converted PDF produces a subfolder named by its file stem:

```
{stem}/
├── {stem}.pdf            # the source PDF (moved here via --movepdf)
├── abstract.md           # title + abstract (GROBID)
├── body.md               # main text with sections and inline tables (docling)
├── references.json       # structured [{authors, title, journal, year, doi, …}, …] (GROBID)
├── references.md         # references section as prose markdown (docling)
├── tables.md             # extracted tables (when tables are present)
└── provenance.json       # per-field attribution: which extractor produced which file
```

`provenance.json` example:

```json
{
  "extraction_mode": "merged-grobid-docling",
  "components": {
    "abstract": "grobid",
    "body": "docling",
    "references_json": "grobid",
    "references_md": "docling",
    "tables": "docling"
  },
  "grobid": {
    "attempted": true,
    "succeeded": true,
    "processing_time_s": 15.87
  },
  "docling": {
    "attempted": true,
    "succeeded": true,
    "processing_time_s": 17.48
  },
  "marker": {"attempted": false},
  "pymupdf4llm": {"attempted": false}
}
```

This lets downstream consumers (LLM extraction pipelines, analysis scripts) know which fields to trust.

---

## Fallback chain

When a PDF is processed in `--mode full-grobid` (the default), pdf4llm follows this cascade:

```
             ┌─── GROBID succeeds + docling succeeds → merge-best-of-each
             │
PDF ─────────┼─── only GROBID succeeds              → GROBID-only output
             │
             ├─── only docling succeeds              → docling-only output
             │                                         (references.json may be empty)
             │
             └─── both fail → marker-pdf             → marker-fallback output
                                   │                   (references.json empty)
                                   │
                                   └─ marker fails → pymupdf4llm    ← last-resort
                                                        │            (flat markdown
                                                        │             only; refs/tables
                                                        │             empty)
                                                        │
                                                        └─ pymupdf4llm fails → failed,
                                                                               logged for
                                                                               retry
```

- `--no-parallel` forces sequential single-extractor mode (GROBID + OCR fallback only). Use for debugging.
- `--no-marker-fallback` disables the marker tier.
- `--no-pymupdf-fallback` disables the pymupdf4llm last-resort tier.
- `--docling-only` disables GROBID entirely. Use if you don't need structured references or GROBID is unavailable.

---

## Installation

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.9 | 3.10 required if using local GROBID+DeLFT |
| Java (JDK) | 21 | Only needed for local GROBID build; not needed with Docker |
| Tesseract OCR | any | System package; see step 2 |

**Minimum viable install (docling only, no GROBID):** complete steps 1–3, then run with `--docling-only`. You get body text and prose references but no structured `references.json`.

**Full install (recommended):** complete all steps 1–6 for parallel GROBID+docling merge with structured references.

---

### 1. Install pdf4llm

```bash
cd /path/to/pdf4llm
pip install -e .
```

On Ubuntu 24+, you may need `--break-system-packages` or a venv due to PEP 668. The package installs `pdf4llm` as a CLI command into `~/.local/bin/` (or your venv's `bin/`).

### 2. Install Tesseract OCR (required for fallback + docling OCR engine)

pdf4llm uses Tesseract for two things:
1. As a fallback when GROBID reports a scanned/empty PDF
2. As docling's OCR engine (via `TesseractCliOcrOptions`) to bypass a known rapidocr PosixPath bug

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

### 3. Install docling (for the parallel merge default)

```bash
pip install --user docling
# or, if you hit PEP 668:
pip install --user --break-system-packages docling
```

**Pinned version note:** docling versions above 2.76.0 (at time of writing) have broken `rapidocr` config handling. pdf4llm works around this by explicitly using `TesseractCliOcrOptions`, so any recent docling should work, but if you see `UnsupportedValueType: Value 'PosixPath' is not a supported primitive type`, it means rapidocr is being picked up — check the troubleshooting section below.

### 4. Install marker-pdf (optional — primary fallback after GROBID+docling)

```bash
pip install --user marker-pdf
```

First use downloads several GB of layout/OCR models to `~/.cache/datalab/models/`. After download, marker runs entirely offline and CPU-only. Models load once per process (~90 s first time, ~2 s subsequent).

If you don't install marker, pdf4llm still works — it just won't have the marker fallback tier (the pymupdf4llm last-resort tier below still runs). Use `--no-marker-fallback` to disable the tier explicitly.

### 5. pymupdf4llm (last-resort fallback tier)

`pymupdf4llm` is a thin wrapper around PyMuPDF that outputs markdown with basic heading detection. It's the final safety-net tier — runs only after marker-pdf fails — and produces *something* on virtually any text-layer PDF.

```bash
pip install --user pymupdf4llm
```

It's listed as a core dependency (it's tiny — PyMuPDF is already pulled in by pdf4llm), so `pip install -e .` should install it automatically. If you want to disable this tier, pass `--no-pymupdf-fallback`.

### 6. Install GROBID with DeLFT

GROBID provides the structured header and reference extraction. DeLFT is GROBID's deep-learning integration for higher-accuracy models.

#### 6a. Docker (easiest)

```bash
cd /path/to/pdf4llm
./docker/start-grobid.sh delft   # high accuracy, ~8 GB image
# or
./docker/start-grobid.sh crf     # lightweight CRF-only, ~500 MB image
```

Then run pdf4llm with `--no-auto-grobid`:

```bash
pdf4llm batch ./pdfs -o ./output --no-auto-grobid
```

#### 6b. Local build (requires Java 21)

```bash
sudo apt install openjdk-21-jdk
git clone https://github.com/kermitt2/grobid.git
cd grobid
./gradlew clean install
```

pdf4llm auto-detects GROBID as a sibling directory. When you run `pdf4llm batch`, it starts GROBID with DeLFT from the sibling `grobid/` checkout, using a Python 3.10 venv at `~/venv_grobid` to host DeLFT's Python deps. **You must create this venv manually:**

```bash
# Requires Python 3.10 specifically — DeLFT is not compatible with 3.11+
python3.10 -m venv ~/venv_grobid

~/venv_grobid/bin/pip install --upgrade pip

# Install DeLFT and its dependencies
# Pin jep to 4.2.2 — version 4.3.1 causes a SIGSEGV with Java 21
~/venv_grobid/bin/pip install \
    "jep==4.2.2" \
    "delft==0.3.4" \
    "transformers==4.44.2" \
    "Pillow"
```

If you don't have Python 3.10, install it first:
```bash
sudo apt install python3.10 python3.10-venv
```

**Critical gotcha: `~/venv_grobid` must have `Pillow` installed.** DeLFT imports `transformers`, which transitively imports `PIL._imaging`. If Pillow is missing from the venv, JEP (the Python-in-Java bridge) falls back to the system PIL, which is usually compiled for the wrong Python version and crashes with `cannot import name '_imaging' from 'PIL'`. The symptom is **every `/api/processFulltextDocument` returns HTTP 500** while `/api/isalive` still returns 200. Fix:

```bash
~/venv_grobid/bin/pip install Pillow
```

### 7. GROBID auto-start

By default, `pdf4llm batch` and `pdf4llm convert` will auto-start GROBID from the sibling build (`../grobid`). You can disable this with `--no-auto-grobid` and manage GROBID yourself.

Verify GROBID is up:

```bash
curl -s http://localhost:8070/api/isalive
# expected: true
```

---

## Usage

### Convert a single PDF (parallel merge — default)

```bash
pdf4llm convert paper.pdf -o output/
```

Runs GROBID and docling in parallel, merges outputs, writes the 6 files listed above.

### Batch convert a folder

```bash
pdf4llm batch ./pdfs/ -o ./output/ --workers 2 --movepdf --resume
```

- `--workers 2` — 2 papers processed concurrently. Each worker runs GROBID + docling in parallel for its PDF, so effective parallelism is 4. On a 64 GB machine, `--workers 2` is the sweet spot; `--workers 4` can OOM on long PDFs.
- `--movepdf` — moves each successfully-processed PDF into its output subfolder (so the source dir ends up empty of processed files).
- `--resume` — reads `.pdf4llm_checkpoint.json` and skips PDFs already processed. Safe to Ctrl+C and restart.

### Sequential / non-parallel mode

```bash
pdf4llm convert paper.pdf -o output/ --no-parallel
```

Falls back to the classic single-extractor path: GROBID → Tesseract OCR fallback → fail. Use for debugging or when docling is unavailable.

### docling-only mode (skip GROBID entirely)

```bash
pdf4llm convert paper.pdf -o output/ --docling-only
```

Use when you don't need structured references or GROBID isn't installed. `references.json` will be empty; `references.md` will contain the references as prose.

### Disable the marker-pdf fallback tier

```bash
pdf4llm batch ./pdfs/ -o ./output/ --no-marker-fallback
```

### Disable the pymupdf4llm last-resort tier

```bash
pdf4llm batch ./pdfs/ -o ./output/ --no-pymupdf-fallback
```

By default, pymupdf4llm runs after marker-pdf fails and produces flat markdown as a final safety net. Disable it only if you'd rather have a hard failure (e.g. to surface problem PDFs for manual review) than a best-effort record.

### Fast mode (no GROBID, no docling, no OCR)

```bash
pdf4llm convert paper.pdf -o output/ --mode fast
```

PyMuPDF + pdfplumber only. Very fast (~1–2 s per PDF) but lower quality, especially for references and tables. Useful for quick previews or pre-filtering.

### Check GROBID status

```bash
pdf4llm health-check
```

### Extract abstracts only

```bash
pdf4llm batch ./pdfs/ -o ./out/ --extract-abstract-only
```

Saves just `{stem}_abstract.md` per paper at the top level (no subfolders).

---

## CLI flags reference

### Common flags

| Flag | Description |
|---|---|
| `-o, --output PATH` | Output directory (required) |
| `--mode [full-grobid\|hybrid\|fast]` | Extraction mode (default: `full-grobid`) |
| `--workers INT` | Parallel workers (default: `min(10, CPU count)`) |
| `--movepdf` | Move source PDFs into output subfolders on success |
| `--resume` | Skip already-processed PDFs via checkpoint |
| `--skip-existing` | Skip PDFs whose output files already exist (on by default) |
| `-v, --verbose` | Verbose logging |

### Extractor / merge control

| Flag | Description |
|---|---|
| *(default)* | Parallel GROBID + docling merge |
| `--docling-only` | Use docling alone, skip GROBID (no structured references) |
| `--no-parallel` | Disable parallel merge; fall back to classic single-extractor cascade |
| `--no-marker-fallback` | Disable the marker-pdf fallback tier (runs after GROBID+docling fail) |
| `--no-pymupdf-fallback` | Disable the pymupdf4llm last-resort fallback tier (runs after marker) |
| `--docling-gpu` | Enable CUDA for docling (requires GPU + CUDA-enabled PyTorch) |

### GROBID management

| Flag | Description |
|---|---|
| `--grobid-url URL` | GROBID server URL (default `http://localhost:8070`) |
| `--no-auto-grobid` | Don't auto-start GROBID (assume already running) |
| `--grobid-home PATH` | Path to GROBID source directory for auto-start |
| `--run-without-delft` | Allow CRF-only mode when DeLFT is unavailable (lower accuracy) |
| `--docker` / `--docker-mode [delft\|crf]` | Use Docker-based GROBID |
| `--timeout INT` | GROBID per-request timeout (default 120 s) |

### OCR / quality

| Flag | Description |
|---|---|
| `--noocr` | Disable Tesseract OCR fallback (skip PDFs that fail GROBID extraction) |
| `--extract-abstract-only` | Extract only the abstract, no body/refs/tables |

### Output

| Flag | Description |
|---|---|
| `--no-json` | Don't write structured JSON files |
| `--save-tei` | Save raw GROBID TEI XML for debugging |
| `--checkpoint PATH` | Custom checkpoint file path |

---

## Python API

### Single-PDF conversion

```python
from pdf4llm import Config, ExtractionMode
from pdf4llm.pipeline import convert_single
from pathlib import Path

config = Config(
    mode=ExtractionMode.FULL_GROBID,  # default
    parallel_extraction=True,         # GROBID + docling in parallel
    use_marker_fallback=True,         # marker tier (after GROBID+docling fail)
    use_pymupdf_fallback=True,        # pymupdf4llm last-resort tier (after marker fails)
)
result = convert_single(Path("paper.pdf"), Path("output/"), config)

print(result.status)        # "success" / "partial" / "failed"
print(result.output_dir)    # path to the {stem}/ subfolder
print(result.processing_time_seconds)
```

### Batch conversion

```python
from pdf4llm import Config
from pdf4llm.pipeline import BatchProcessor
from pathlib import Path

config = Config(workers=2)
processor = BatchProcessor(config, checkpoint_file=Path("output/.pdf4llm_checkpoint.json"))
result = processor.process_folder(
    input_dir=Path("./pdfs"),
    output_dir=Path("./output"),
    resume=True,
    skip_existing=True,
    move_pdf=True,
)
print(result.generate_report())
```

---

## Extraction modes

| Mode | What it runs | When to use |
|---|---|---|
| `full-grobid` (default) | GROBID + docling in parallel, merged; marker fallback; pymupdf4llm last-resort fallback | **Default for production** — best overall quality |
| `full-grobid --no-parallel` | GROBID → Tesseract OCR fallback → fail | Classic cascade; useful when docling is unavailable |
| `full-grobid --docling-only` | docling only | No structured references needed, or GROBID is broken |
| `hybrid` | Fast extraction (PyMuPDF+pdfplumber) with selective GROBID fallback for deficient fields | Speed-sensitive batches with a quality floor |
| `fast` | PyMuPDF + pdfplumber only | Quick previews, pre-filtering, or when no GROBID available |

---

## Performance

On CPU (no GPU):

| Config                                            | Per-paper time      | Quality         |
|---------------------------------------------------|---------------------|-----------------|
| `full-grobid` (GROBID + DeLFT + docling merge)    | ~15–25 s            | **Best**        |
| `full-grobid --no-parallel` (GROBID + DeLFT)      | ~8–15 s             | Good            |
| `full-grobid --docling-only`                      | ~10–20 s            | Good (no refs)  |
| `marker-pdf` (via fallback)                       | ~20–60 s            | Good (no refs)  |
| `pymupdf4llm` (via last-resort fallback)          | ~1–3 s              | Basic (no refs, no tables) |
| `fast`                                            | ~1–2 s              | Basic           |

Memory footprint (per worker):
- GROBID + DeLFT JVM: ~6 GB
- docling models: ~1–2 GB
- marker models: ~3 GB (only loaded when the fallback fires)
- pymupdf4llm: <100 MB (only loaded when the last-resort fallback fires)
- Total for parallel merge: ~8–9 GB per worker

**Recommended**: `--workers 2` on a 32 GB machine, `--workers 4` on 64 GB+. No swap = be conservative.

---

## Troubleshooting

### `GROBID returned status 500: [GENERAL] An exception occurred while running Grobid` on every PDF

**Cause**: DeLFT failed to initialize, usually because the GROBID Python venv (`~/venv_grobid`) is missing `Pillow`. JEP falls back to the system PIL which is compiled for the wrong Python version and crashes on import.

**Fix**:
```bash
~/venv_grobid/bin/pip install Pillow
```

Then restart GROBID (or let `pdf4llm batch` auto-restart it). Verify the fix by checking the GROBID log for `JEP initialization failed` — if present, Pillow (or a transitive dep) still isn't importable.

### `omegaconf.errors.UnsupportedValueType: Value 'PosixPath' is not a supported primitive type, full_key: Global.model_root_dir`

**Cause**: docling is trying to initialize `rapidocr` as its OCR engine, and rapidocr's current version has a bug assigning a PosixPath to an OmegaConf config field that only accepts primitives.

**Fix**: pdf4llm already works around this by explicitly using `TesseractCliOcrOptions` instead of the default rapidocr. If you still hit this error, make sure you're running a recent pdf4llm and that Tesseract is installed. Alternatively downgrade: `pip install 'docling==2.76.0'`.

### `cannot import name '_imaging' from 'PIL'`

**Cause**: Mismatch between the Pillow installed in the GROBID venv vs the system PIL. The `.so` binary was compiled for a different Python version.

**Fix**: Install Pillow explicitly in the GROBID venv using the venv's `pip`:
```bash
~/venv_grobid/bin/pip install --upgrade --force-reinstall Pillow
```

### `pdf4llm batch` silently freezes

**Cause**: GROBID zombies or hung child processes after an earlier crash. `/api/isalive` may still return 200 while `/api/processFulltextDocument` hangs forever.

**Fix**:
```bash
pkill -9 -f grobid-service
sleep 2
# Re-run pdf4llm batch; auto-start will spin up a fresh GROBID
```

pdf4llm's signal handler cleanly stops GROBID on Ctrl+C (SIGTERM → SIGKILL after 3 s), but if GROBID is in a JEP-deadlocked state, SIGKILL is the only option.

### Ctrl+C doesn't stop the run cleanly

**Behavior**: pdf4llm installs a SIGINT handler that on **first** Ctrl+C sets a shutdown flag, letting in-flight PDFs finish but cancelling pending work. A **second** Ctrl+C forces immediate exit. Both paths stop the GROBID child process via SIGTERM (3 s timeout) then SIGKILL.

If Ctrl+C seems unresponsive, press it again for force exit.

### `GROBID OUT OF MEMORY - server crashed while processing X.pdf`

**Cause**: GROBID's JVM ran out of heap. Usually triggered by parallel processing of large PDFs on a machine without swap.

**Fix**:
1. Ctrl+C to abort
2. Restart with fewer workers: `--workers 1`
3. Optionally add swap: `sudo fallocate -l 16G /swapfile; sudo mkswap /swapfile; sudo swapon /swapfile`
4. Re-run with `--resume` to skip the papers that already succeeded

### Phantom checkpoint entries

**Symptom**: `.pdf4llm_checkpoint.json` has more entries than there are output folders on disk.

**Fix**: pdf4llm's `BatchProcessor._load_checkpoint()` automatically validates each entry on startup, dropping any whose `abstract.md + body.md + references.json` don't all exist on disk. The stale entries will be retried on the next run. If you want to force a full re-validation:
```bash
rm /path/to/output/.pdf4llm_checkpoint.json
# Next --resume run will re-check all source PDFs against existing output folders
```

### `docling not found` error

**Cause**: docling isn't installed in the Python that `pdf4llm` uses. Check with `which pdf4llm`; the shebang in that file tells you which Python is being invoked.

**Fix**: Install docling in the same environment:
```bash
# System python with PEP 668 workaround:
pip install --user --break-system-packages docling

# Or in a venv:
/path/to/venv/bin/pip install docling
```

### `marker not installed` warning in logs

If you haven't installed marker-pdf, pdf4llm will skip the marker fallback tier with a warning and proceed straight to the pymupdf4llm last-resort tier. This is fine — most PDFs don't need marker. Install if you want the extra coverage:
```bash
pip install --user --break-system-packages marker-pdf
```

### `pymupdf4llm not available` warning in logs

If the last-resort tier can't import `pymupdf4llm`, pdf4llm will skip it and record a hard failure when marker-pdf has already failed. Install it to keep the safety net active:
```bash
pip install --user --break-system-packages pymupdf4llm
```

pymupdf4llm is listed as a core dependency in `pyproject.toml` / `requirements.txt`, so this warning usually only appears in partial installs or bespoke environments.

### Record has `extraction_mode: pymupdf4llm-fallback` — what does that mean for downstream?

This means GROBID, docling, AND marker-pdf all failed on that PDF, and pdf4llm recovered via the pure-PyMuPDF last-resort tier. The output folder is complete (`abstract.md`, `body.md`, `references.md`, `provenance.json`) but:

- `references.json` is an empty array `[]` — no structured references.
- `body.md` is flat markdown with heuristic heading-based section splitting.
- `abstract.md` is extracted via heading detection and may be empty if the PDF doesn't use an "Abstract" heading.
- `tables` are not extracted.

Treat these records as best-effort. Downstream consumers (e.g. `extract.py` in `claude_code_replications/`) already handle empty `references.json` gracefully (same path as docling-only fallback) and will produce degraded-but-valid results.

---

## Architecture notes

### File layout of the package

```
pdf4llm/
├── cli.py                    # Click CLI entry points (convert, batch, health-check)
├── config.py                 # Config dataclass
├── models.py                 # DocumentModel and related Pydantic models
├── pipeline.py               # convert_single(), BatchProcessor, merge logic
├── validation.py             # quality scoring
├── extractors/
│   ├── grobid.py             # GROBID HTTP client + TEI parser
│   ├── fast.py               # PyMuPDF + pdfplumber path
│   ├── ocr_fallback.py       # Tesseract OCR (the extract_with_ocr function)
│   ├── tables.py             # table quality assessment
│   ├── charts.py             # chart extraction via vision models (optional)
│   ├── marker_ext.py         # marker-pdf integration (fallback tier after docling)
│   ├── pymupdf4llm_ext.py    # pymupdf4llm integration (last-resort tier after marker)
│   └── doi_lookup.py         # DOI→title lookup via Crossref/DataCite/Unpaywall/S2/OpenAlex
└── renderers/
    ├── markdown.py           # abstract.md, body.md, tables.md renderers
    └── json_output.py        # references.json and full-document JSON
```

### Merge logic

```python
# In pipeline.py: _merge_grobid_docling()
merged = DocumentModel(
    metadata=grobid_doc.metadata,
    doi=grobid_doc.doi or docling_doc.doi,
    title=<first valid title from [grobid, docling]; then DOI API fallback>,
    abstract=grobid_doc.abstract or docling_doc.abstract,
    # Prefer docling for body/tables
    sections=docling_doc.sections or grobid_doc.sections,
    tables=docling_doc.tables or grobid_doc.tables,
    figures=docling_doc.figures or grobid_doc.figures,
    # Prefer GROBID for structured references
    references=grobid_doc.references or docling_doc.references,
)
```

`references.md` is extracted from docling's raw markdown output via a heading-detection regex that looks for `# References`, `# Bibliography`, or `# Works Cited`.

---

## What pdf4llm is not

- **Not a general-purpose PDF converter.** It's optimized for academic papers with a specific output shape that matches downstream LLM-extraction pipelines. For general PDFs (invoices, books, legal documents) you may prefer raw docling or marker-pdf directly.
- **Not a reference resolver.** It extracts references but doesn't look them up in Crossref/Semantic Scholar. Use a separate resolver for that.
- **Not a figure extractor by default.** We disable `generate_picture_images` in docling to save disk space. Enable `--extract-charts` if you need it (requires Ollama for vision-model analysis).

---

## License

pdf4llm is distributed under the **GNU Affero General Public License v3 or later (AGPL-3.0-or-later)**. The full license text is in the [LICENSE](LICENSE) file at the root of this repository, and per-dependency attributions are in [NOTICE](NOTICE).

**Why AGPL?** pdf4llm depends on [PyMuPDF](https://pymupdf.readthedocs.io/) (dual-licensed AGPL-3.0 / Artifex Commercial) for PDF text extraction, page rendering, and page-count helpers, and on [pymupdf4llm](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/) (same dual license) for the last-resort markdown extraction tier. Python's `import` is dynamic linking and creates a combined work — any distribution of pdf4llm must therefore be licensed under a license compatible with AGPL-3.0. The simplest compatible choice is AGPL-3.0-or-later itself, which is what we use.

**What this means for you as a user:**

- You are free to install, run, modify, and redistribute pdf4llm, including for commercial purposes, subject to AGPL-3.0's terms.
- If you modify pdf4llm and distribute the modified version (as a GitHub fork, a pip package, a Docker image, etc.), the modified version must also be licensed under AGPL-3.0-or-later and you must make the modified source code available to recipients.
- **AGPL §13 — the "network service" clause** — if you host a modified pdf4llm on a server and let users interact with it over a network (e.g. as a web API or SaaS endpoint), you must offer those remote users access to the modified source code. This clause does *not* apply to local CLI use: running `pdf4llm batch ./pdfs/` on your own machine is not "network interaction."
- GPL-3.0-or-later code (like `marker-pdf`) is compatible with AGPL-3.0 under AGPL §13 — using the marker fallback tier does not add any new constraints beyond what PyMuPDF already imposes.

**If you need a non-AGPL distribution:** obtain a commercial license for PyMuPDF from Artifex Software and relicense your downstream variant of pdf4llm accordingly, or fork pdf4llm and replace the PyMuPDF-dependent extractors (`fast.py`, `ocr_fallback.py`, `pymupdf4llm_ext.py`, `charts.py`, the page-count helper in `pipeline.py`) with permissive alternatives such as [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) (BSD-3-Clause / Apache-2.0) + [pdfminer.six](https://github.com/pdfminer/pdfminer.six) (MIT). We do not currently maintain such a variant.

**No copy-pasted code:** every line of pdf4llm source is original to the Metascience Observatory; we use all upstream tools (GROBID, docling, marker-pdf, pymupdf, etc.) exclusively via their public APIs. See [NOTICE](NOTICE) for the full dependency chain and per-package license.

---

## Dependencies

### Python
- `click` — CLI
- `pydantic` — data models
- `tqdm` — progress bars
- `requests` — GROBID HTTP
- `pymupdf`, `pdfplumber` — fast mode
- `pymupdf4llm` — last-resort fallback tier (wraps `pymupdf`; pulled in as a core dep)
- `pytesseract`, `Pillow` — OCR fallback
- `docling` — parallel merge (optional but strongly recommended)
- `marker-pdf` — fallback tier after GROBID+docling (optional)

### System
- `tesseract-ocr` — for OCR fallback
- `openjdk-21-jdk` — for GROBID
- Working GROBID build with DeLFT (at `../grobid/`) OR Docker

### GROBID Python venv (for DeLFT)
- Python 3.10 venv at `~/venv_grobid`
- `Pillow`, `transformers==4.44.2`, `delft==0.3.4`, `jep==4.2.2`
- See [GROBID installation notes](#6-install-grobid-with-delft)
