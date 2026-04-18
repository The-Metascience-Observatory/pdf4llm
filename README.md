# Data Ingestor

An ETL (Extract, Transform, Load) pipeline for ingesting, enriching, and standardizing replication experiment data into the master replications database.

## Overview

The data ingestor takes CSV files containing replication study data, automatically fetches missing metadata from academic APIs, converts effect sizes to a standardized format (Pearson's r), generates HTML citations, and appends new entries to the master database.

## Pipeline Steps

```
Input CSV
  → [1] Metadata Enrichment (fetch authors, titles, journals via DOI/title lookup)
  → [2] Effect Size Conversion (convert 20+ effect size types to Pearson's r)
  → [3] Citation Generation (APA-style HTML citations with DOI links)
  → [4] Column Filtering (validate against data_dictionary.csv)
  → [5] Duplicate Detection (skip entries already in the database)
  → [6] Save & Version (timestamped CSV export + version_history.txt update)
```

## Usage

```bash
python data_ingestor.py <input_csv> [--skip-api-calls] [--discipline "discipline_name"]
```

- `input_csv` — CSV file with new replication data to ingest
- `--skip-api-calls` — Skip metadata enrichment (faster, useful for re-processing)
- `--discipline` — Apply a discipline label to all rows (e.g., `"cancer biology"`)

## Key Files

| File | Description |
|------|-------------|
| `data_ingestor.py` | Main orchestration script; runs the 6-step pipeline, handles effect size conversions and validation |
| `fetch_metadata_from_doi.py` | Fetches metadata (authors, title, journal, year, etc.) from a DOI using a cascade of 6 APIs |
| `fetch_metadata_from_title.py` | Fetches metadata starting from a paper title when DOI is unavailable |
| `generate_citation_html_for_website.py` | Formats bibliographic data into APA-style HTML citations with clickable DOI links |
| `fetch_pdf_from_doi.py` | Downloads full-text PDFs using 8+ fallback sources |
| `pull_pdfs.ipynb` | Notebook for batch PDF downloading |
| `make_ground_truth_dataset.ipynb` | Notebook for creating a validated/curated subset of replications |
| `data_dictionary.csv` | Schema definition for all 42 database columns |
| `version_history.txt` | Tracks all database versions |

## External APIs

Metadata is fetched progressively—the system tries each API in order and stops when all fields are filled:

1. **OpenAlex** (`api.openalex.org`)
2. **Crossref** (`api.crossref.org`)
3. **DataCite** (`api.datacite.org`)
4. **Unpaywall** (`api.unpaywall.org`)
5. **Europe PMC** (`ebi.ac.uk/europepmc`)
6. **Semantic Scholar** (`api.semanticscholar.org`)

PDF retrieval additionally uses OSF, DuckDuckGo search, direct DOI resolution, and Sci-Hub as fallbacks.

## Effect Size Conversions

The pipeline normalizes effect sizes to Pearson's r from 20+ input types including Cohen's d, odds ratios, eta-squared, Cohen's f/f², R², and parsed test statistics (t, F, z, χ²). Conversion functions are defined in `data_ingestor.py`.

## Duplicate Detection

New rows are checked against the master database using a composite key of `original_url` + `replication_url` + `description`. Duplicates are skipped during ingestion.

## Output

Each run produces a timestamped CSV file (`replications_database_YYYY_MM_DD_HHMMSS.csv`) in the `../data/` directory and updates `version_history.txt`.
