# pdf4llm

PDF-to-Markdown and JSON converter optimized for LLM analysis of scientific papers. Extracts structured content (sections, tables, figures, references with DOIs) from academic PDFs using [GROBID](https://github.com/kermitt2/grobid).

## Installation

### 1. Install pdf4llm

```bash
pip install -e .
```

This gives you the `pdf4llm` CLI command.

### 1b. Install Tesseract OCR (recommended)

Tesseract is used as a fallback when PDF extraction quality is below 60%. Without it, low-quality extractions cannot be retried with OCR.

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

The Python wrapper (`pytesseract`) is installed automatically with pdf4llm. A yellow warning will appear at runtime if the Tesseract binary is missing.

### 2a. Install GROBID via Docker (Recommended for DeLFT)

**For highest accuracy extraction using Deep Learning models (DeLFT), use Docker:**

**Prerequisites:** Docker and Docker Compose installed.

```bash
# Start GROBID with DeLFT models (high accuracy, CPU)
cd /path/to/pdf4llm
./docker/start-grobid.sh delft

# Or for fastest CPU performance (CRF-only)
./docker/start-grobid.sh crf
```

**Docker Images:**
- `grobid/grobid:0.8.2-full` - Includes DeLFT deep learning models (~8GB)
- `grobid/grobid:0.8.2-crf` - CRF-only, lightweight (~500MB)

**Performance on CPU:**
- **DeLFT mode:** ~15-30s per PDF (highest accuracy, +3-5% improvement over CRF)
- **CRF mode:** ~3-5s per PDF (fast, good accuracy)

**Using Docker GROBID with pdf4llm:**

```bash
# Automatic Docker startup
pdf4llm convert paper.pdf -o output/ --docker --docker-mode delft

# Or start Docker manually and connect
./docker/start-grobid.sh delft
pdf4llm convert paper.pdf -o output/ --no-auto-grobid
```

**Configuration:**

Edit `docker/grobid-delft-cpu.yaml` to customize which models use DeLFT:
- Set `engine: "delft"` for deep learning (slower, more accurate)
- Set `engine: "wapiti"` for CRF (faster)

**Memory Requirements:**
- DeLFT mode: 6-8GB RAM
- CRF mode: 4GB RAM

**Stop GROBID:**
```bash
cd /path/to/pdf4llm
docker compose -f docker/docker-compose.yml down
```

### 2b. Install GROBID (local build, no Docker)

GROBID is required for the `full-grobid` (default) and `hybrid` extraction modes. The `fast` mode does not need GROBID.

**Prerequisites:** OpenJDK 21 or higher. Install path must not contain spaces.

```bash
# Install JDK 21 (Ubuntu/Debian)
sudo apt install openjdk-21-jdk

# Install JDK 21 (macOS)
brew install openjdk@21
```

Clone and build GROBID as a sibling directory (the default location pdf4llm looks for):

```bash
# From the parent directory of pdf4llm
cd ..
git clone https://github.com/kermitt2/grobid.git
cd grobid
./gradlew clean install
```

The build downloads models and dependencies on first run. This takes a while.

### GROBID Auto-Start (Recommended)

**pdf4llm automatically starts GROBID with DeLFT (highest accuracy) when needed!**

When you run `pdf4llm convert` or `pdf4llm batch`, it will:
1. Check if GROBID is already running on port 8070
2. If not, automatically start GROBID with DeLFT deep learning models (if available)
3. Use the pre-built JAR for proper JEP/DeLFT integration
4. Stop GROBID when processing completes

No manual GROBID management required! Just run:
```bash
pdf4llm convert paper.pdf -o output/
```

### Manual GROBID Start (Optional)

If you prefer to manage GROBID manually or want to keep it running between commands:

**With DeLFT (highest accuracy):**
```bash
cd /path/to/grobid
java -Xmx6g \
  -Djava.library.path=$HOME/.local/lib/python3.12/site-packages/jep \
  -jar grobid-service/build/libs/grobid-service-*-onejar.jar \
  server grobid-home/config/grobid.yaml
```

**Without DeLFT (faster, CRF-only):**
```bash
cd /path/to/grobid
./gradlew run
```

Then use `--no-auto-grobid` flag:
```bash
pdf4llm convert paper.pdf -o output/ --no-auto-grobid
```

**Note:** GROBID starts on **port 8070** by default. Verify it's up with:
```bash
curl http://localhost:8070/api/isalive
```

## Usage

### Convert a single PDF

```bash
pdf4llm convert paper.pdf -o output/
```

### Batch convert a folder of PDFs

```bash
pdf4llm batch ./pdfs/ -o ./output/ --workers 4
```

Resume an interrupted batch:

```bash
pdf4llm batch ./pdfs/ -o ./output/ --resume
```

Skip already-converted PDFs:

```bash
pdf4llm batch ./pdfs/ -o ./output/ --skip-existing
```

### Fast mode (no GROBID)

Uses PyMuPDF + pdfplumber only. Faster but lower quality, especially for references and tables.

```bash
pdf4llm convert paper.pdf -o output/ --mode fast
```

### Hybrid mode

Fast extraction with GROBID fallback for low-quality fields:

```bash
pdf4llm convert paper.pdf -o output/ --mode hybrid
```

### Check GROBID status

```bash
pdf4llm health-check
```

### Chart extraction (experimental)

Requires [Ollama](https://ollama.ai/) running locally with a vision model:

```bash
pdf4llm convert paper.pdf -o output/ --extract-charts
```

## Extraction modes

| Mode | Speed | Quality | GROBID Backend | Requires GROBID |
|------|-------|---------|----------------|-----------------|
| `full-grobid` (default) | ~15-30s/PDF (CPU+DeLFT) | **Best** | DeLFT | Yes |
| `full-grobid` | ~3-5s/PDF (CPU+CRF) | **Good** | CRF | Yes |
| `full-grobid` | ~5s/PDF (GPU+DeLFT) | **Best** | DeLFT | Yes |
| `hybrid` | Varies | Good | CRF/DeLFT | Yes (fallback) |
| `fast` | Fastest (~1-2s/PDF) | Basic | None | No |

**Note:** DeLFT (Deep Learning) provides +3-5% accuracy improvement over CRF, especially for citations, references, and author affiliations.

## Output

Each PDF produces a DOI-named subfolder containing:

- `<stem>.md` — structured Markdown with sections, tables, figures, and references
- `<stem>.json` — machine-readable JSON with the same content (disable with `--no-json`)
- `<stem>.tei.xml` — raw GROBID TEI XML (opt-in with `--save-tei`)

## Python API

```python
from pdf4llm import Config, ExtractionMode
from pdf4llm.pipeline import convert_single
from pathlib import Path

config = Config(mode=ExtractionMode.FULL_GROBID)
result = convert_single(Path("paper.pdf"), Path("output/"), config)

print(result.status)        # "success", "partial", or "failed"
print(result.markdown_path)
print(result.json_path)
```

## Requirements

- Python >= 3.9
- OpenJDK >= 21 (for GROBID)
