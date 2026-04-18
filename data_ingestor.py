"""
Ingestion Engine for Replications Database

This script processes CSV or JSON files containing replication experiment data
and adds them to the master replications database.

Usage:
    python data_ingestor.py <input_file.csv|input_file.json>
    python data_ingestor.py --skip-api-calls <input_file.csv>

JSON format: {"replications": [{"original_url": "...", "replication_url": "...", ...}, ...]}
"""

import pandas as pd
import numpy as np
import argparse
import time
import os
import re
import math
import logging
import shutil
import json
import concurrent.futures
import threading
import random
from datetime import datetime
from difflib import SequenceMatcher
from fetch_metadata_from_doi import fetch_metadata_from_doi, _new_authors_are_better, _authors_have_abbreviations
from fetch_metadata_from_title import fetch_metadata_from_title


# Configure logging for the ingestion pipeline
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# Get the directory where this script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
BACKUP_DIR = os.path.join(DATA_DIR, 'backup')
VERSION_HISTORY_PATH = os.path.join(DATA_DIR, 'version_history.txt')
API_CACHE_PATH = os.path.join(SCRIPT_DIR, 'api_cache.json')
ONTOLOGY_PATH = os.path.join(DATA_DIR, 'metascience_observatory_topic_ontology.json')
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, 'ingestion_checkpoint.csv')
CHECKPOINT_META_PATH = os.path.join(SCRIPT_DIR, 'ingestion_checkpoint_meta.json')


def save_checkpoint_metadata(input_file, row_count):
    """Save metadata about the checkpoint for validation on next run"""
    metadata = {
        'input_file': os.path.basename(input_file),
        'input_file_abspath': os.path.abspath(input_file),
        'row_count': row_count,
        'timestamp': datetime.now().isoformat()
    }
    with open(CHECKPOINT_META_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)


def is_checkpoint_valid(input_file, row_count):
    """Check if checkpoint is valid for the current input file"""
    if not os.path.exists(CHECKPOINT_PATH):
        return False
    if not os.path.exists(CHECKPOINT_META_PATH):
        logger.warning("Checkpoint found but no metadata - treating as stale")
        return False

    try:
        with open(CHECKPOINT_META_PATH, 'r') as f:
            metadata = json.load(f)

        # Check if input file matches (by basename or absolute path)
        current_basename = os.path.basename(input_file)
        current_abspath = os.path.abspath(input_file)

        if (metadata.get('input_file') != current_basename and
            metadata.get('input_file_abspath') != current_abspath):
            logger.warning(f"Checkpoint is for different file: {metadata.get('input_file')} vs {current_basename}")
            return False

        # Row count should match (allow some tolerance for header differences)
        if metadata.get('row_count') != row_count:
            logger.warning(f"Checkpoint row count mismatch: {metadata.get('row_count')} vs {row_count}")
            return False

        return True
    except Exception as e:
        logger.error(f"Error validating checkpoint: {e}")
        return False


def clear_checkpoint():
    """Remove checkpoint and metadata files"""
    for path in [CHECKPOINT_PATH, CHECKPOINT_META_PATH]:
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Removed stale checkpoint: {path}")


def load_api_cache():
    """Load persistent API cache from disk. Returns (doi_cache, title_cache) dicts."""
    doi_cache = {}
    title_cache = {}
    if os.path.exists(API_CACHE_PATH):
        try:
            with open(API_CACHE_PATH, 'r') as f:
                data = json.load(f)
            doi_cache = data.get('doi', {})
            title_cache = data.get('title', {})
            logger.info(f"Loaded API cache: {len(doi_cache)} DOIs, {len(title_cache)} titles")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not load API cache: {e}")
    return doi_cache, title_cache


def save_api_cache(doi_cache, title_cache):
    """Save API caches to disk as JSON."""
    data = {'doi': doi_cache, 'title': title_cache}
    with open(API_CACHE_PATH, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved API cache: {len(doi_cache)} DOIs, {len(title_cache)} titles")


def get_latest_master_database():
    """Get the latest master database filename from version_history.txt.
    Walks backwards through the file to find the most recent entry that
    actually exists on disk."""
    if not os.path.exists(VERSION_HISTORY_PATH):
        return None

    with open(VERSION_HISTORY_PATH, 'r') as f:
        lines = f.readlines()

    # Walk backwards to find the last entry that exists on disk
    for line in reversed(lines):
        line = line.strip()
        # Skip empty lines and comments
        if line and not line.startswith('#'):
            # Extract just the filename (remove any path prefix and comments)
            filename = line.split('#')[0].strip()
            # Handle both relative paths and just filenames
            if filename.startswith('../data/'):
                filename = filename.replace('../data/', '')
            # Verify the file actually exists
            full_path = os.path.join(DATA_DIR, filename)
            if os.path.exists(full_path):
                return filename
            else:
                logger.warning(f"Version history references missing file: {filename}, skipping")

    return None

def extract_doi_from_url(url):
    """Extract DOI from URL like 'https://doi.org/10.1234/xyz'"""
    if not isinstance(url, str) or not url.strip():
        return None
    url = url.strip()
    doi = None
    if url.startswith("http://doi.org/"):
        doi = url.replace("http://doi.org/", "")
    elif url.startswith("https://doi.org/"):
        doi = url.replace("https://doi.org/", "")
    # Ensure DOI starts with 10. (fix malformed DOIs like 0.1037/... → 10.1037/...)
    if doi and not doi.startswith("10."):
        doi = "10." + doi.lstrip("0.")
    return doi

def normalize_doi(doi):
    """
    Normalize a DOI by removing any URL prefix.
    Handles cases where DOI might already be a full URL.
    Returns just the DOI part (e.g., '10.1234/xyz')
    """
    if not isinstance(doi, str) or not doi.strip():
        return None
    doi = doi.strip()
    # Strip common URL prefixes
    if doi.startswith("http://doi.org/"):
        doi = doi.replace("http://doi.org/", "")
    elif doi.startswith("https://doi.org/"):
        doi = doi.replace("https://doi.org/", "")
    elif doi.startswith("http://dx.doi.org/"):
        doi = doi.replace("http://dx.doi.org/", "")
    elif doi.startswith("https://dx.doi.org/"):
        doi = doi.replace("https://dx.doi.org/", "")
    # Ensure DOI starts with 10. (fix malformed DOIs like 0.1037/... → 10.1037/...)
    if doi and not doi.startswith("10."):
        doi = "10." + doi.lstrip("0.")
    return doi if doi else None

def is_valid_doi(doi):
    """Validate that a DOI has a plausible format (starts with 10. followed by registrant code)."""
    if not isinstance(doi, str) or not doi.strip():
        return False
    return bool(re.match(r'^10\.\d{4,}/.+$', doi.strip()))

def is_empty(value):
    """Check if value is empty/missing"""
    return pd.isna(value) or value == "" or value == "NaN" or (isinstance(value, str) and not value.strip())


# =============================================================================
# EFFECT SIZE CONVERSION FUNCTIONS
# =============================================================================
# Based on formulas from effect_size_conversions.md and effect_size_transformations.R
# All conversions target Pearson's r as the common metric.

# Effect size types that cannot be reliably converted to r
CANNOT_CONVERT = {
    "beta (std)", "χ2", "b (unstd)", "b",
    "cramer's v", "dz", "beta", "percentage",
    "squared seminpartial correlation (sr2)", "regression coefficient",
    "unstandardized coefficient", "cohen's h", "h",
    "semi-partial correlation", "cliff's delta", "w", "cohen's w"
}

# Mapping of effect size type aliases to canonical names
ESTYPE_MAP = {
    # Odds ratios and hazard ratios (HR uses same conversion as OR)
    "or": "or",
    "odds ratio": "or",
    "hr": "or",
    "hazard ratio": "or",
    "hazards ratio": "or",

    # Standardized mean differences (Cohen's d, Hedges' g, Glass' delta)
    "d": "d",
    "cohen's d": "d",
    "hedges' g": "d",
    "hedges'g": "d",
    "hedge's g": "d",
    "hedges g": "d",
    "hedgesg": "d",
    "smd": "d",
    "glass' delta": "d",
    "glass's delta": "d",
    "glass delta": "d",

    # Eta-squared (η²) and partial eta-squared (η²_p)
    "etasq": "eta2",
    "etaq": "eta2",
    "eta^2": "eta2",
    "η²": "eta2",
    "eta-squared": "eta2",
    "partial etasq": "eta2",
    "etasq (partial)": "eta2",
    "partial eta-squared": "eta2",
    "partial eta squared": "eta2",
    "partial η²": "eta2",
    "partial eta^2": "eta2",

    # Cohen's f
    "f": "f",
    "cohen's f": "f",

    # Cohen's f²
    "f2": "f2",
    "f^2": "f2",
    "f²": "f2",
    "cohen's f^2": "f2",

    # Correlations (r / phi / Spearman's r)
    "r": "r",
    "phi": "r",
    "φ": "r",
    "pearson's r": "r",
    "pearson r": "r",
    "correlation": "r",
    "spearman's r": "r",
    "spearman r": "r",
    "spearman": "r",

    # R-squared (R²)
    "r2": "r2",
    "r^2": "r2",
    "r²": "r2",
    "r-square": "r2",
    "r-squared": "r2",

    # Test statistics
    "test statistic": "test-stat",
    "test statistics": "test-stat",
    "test": "test-stat",
}


def d_to_r(d, n1=None, n2=None):
    """
    Convert Cohen's d to Pearson's r.

    If sample sizes are provided, uses the exact formula:
        r = d / sqrt(d^2 + (n1 + n2)^2 / (n1 * n2))

    Otherwise uses the approximation (assumes equal sample sizes):
        r = d / sqrt(d^2 + 4)

    Sign is preserved.
    """
    if d is None or (isinstance(d, float) and math.isnan(d)):
        return None

    if n1 is not None and n2 is not None and n1 > 0 and n2 > 0:
        # Exact formula with sample sizes
        a = (n1 + n2) ** 2 / (n1 * n2)
        return d / math.sqrt(d ** 2 + a)
    else:
        # Approximation assuming equal sample sizes
        return d / math.sqrt(d ** 2 + 4)


def or_to_r(odds_ratio):
    """
    Convert Odds Ratio to Pearson's r.

    Two-step conversion:
    1. Convert OR to d: d = ln(OR) * sqrt(3) / π
    2. Convert d to r: r = d / sqrt(d^2 + 4)

    Sign is preserved (OR < 1 implies negative r).
    """
    if odds_ratio is None or (isinstance(odds_ratio, float) and math.isnan(odds_ratio)):
        return None
    if odds_ratio <= 0:
        return None

    d = math.log(odds_ratio) * math.sqrt(3) / math.pi
    return d_to_r(d)


def eta2_to_r(eta2):
    """
    Convert Eta-squared to Pearson's r.

    Two-step conversion:
    1. Convert η² to d: d = 2 * sqrt(η² / (1 - η²))
    2. Convert d to r: r = d / sqrt(d^2 + 4)

    Always positive.
    """
    if eta2 is None or (isinstance(eta2, float) and math.isnan(eta2)):
        return None
    if eta2 < 0 or eta2 >= 1:
        return None

    d = 2 * math.sqrt(eta2 / (1 - eta2))
    return d_to_r(d)


def f_to_r(f):
    """
    Convert Cohen's f to Pearson's r.

    Two-step conversion:
    1. Convert f to d: d = 2f
    2. Convert d to r: r = d / sqrt(d^2 + 4)

    Always positive.
    """
    if f is None or (isinstance(f, float) and math.isnan(f)):
        return None

    d = 2 * f
    return d_to_r(d)


def f2_to_r(f2):
    """
    Convert Cohen's f² to Pearson's r.

    Two-step conversion:
    1. Convert f² to R²: R² = f² / (1 + f²)
    2. Convert R² to r: r = sqrt(R²)

    Always positive.
    """
    if f2 is None or (isinstance(f2, float) and math.isnan(f2)):
        return None
    if f2 < 0:
        return None

    r2 = f2 / (1 + f2)
    return math.sqrt(r2)


def r2_to_r(r2):
    """
    Convert R-squared to Pearson's r.

    r = sqrt(R²)

    Always positive.
    """
    if r2 is None or (isinstance(r2, float) and math.isnan(r2)):
        return None
    if r2 < 0 or r2 > 1:
        return None

    return math.sqrt(r2)


def parse_test_statistic(stat_string):
    """
    Parse APA-formatted test statistics and convert to r.

    Supported formats:
    - t(df) = value        e.g., "t(10) = 2.5"
    - F(df1, df2) = value  e.g., "F(1, 20) = 4.5" (df1 must be 1)
    - z = value, N = value e.g., "z = 2.81, N = 34"
    - χ2(1, N = value) = value  e.g., "χ2(1, N = 12) = 5" (df must be 1)

    Returns r value or None if cannot be parsed/converted.
    """
    if not isinstance(stat_string, str):
        return None

    stat_string = stat_string.strip()

    # t-test: t(df) = value
    t_match = re.match(r'^t\((\d+)\)\s*=\s*(-?\d+\.?\d*)$', stat_string, re.IGNORECASE)
    if t_match:
        df = float(t_match.group(1))
        t_val = float(t_match.group(2))
        return t_val / math.sqrt(t_val ** 2 + df)

    # F-test: F(df1, df2) = value
    f_match = re.match(r'^f\((\d+)\s*,\s*(\d+)\)\s*=\s*(\d+\.?\d*)$', stat_string, re.IGNORECASE)
    if f_match:
        df1 = float(f_match.group(1))
        df2 = float(f_match.group(2))
        f_val = float(f_match.group(3))
        if df1 == 1:
            t_val = math.sqrt(f_val)
            return t_val / math.sqrt(t_val ** 2 + df2)
        else:
            return None  # Cannot convert F with df1 > 1

    # z-test: z = value, N = value
    z_match = re.match(r'^z\s*=\s*(-?\d+\.?\d*)\s*,\s*n\s*=\s*(\d+)$', stat_string, re.IGNORECASE)
    if z_match:
        z_val = float(z_match.group(1))
        n_val = float(z_match.group(2))
        return z_val / math.sqrt(z_val ** 2 + n_val)

    # Chi-squared: χ2(1, N = value) = value or x2(1, N = value) = value
    # Replace χ with x for matching
    normalized_stat = re.sub(r'^[χΧ]', 'x', stat_string)
    chi_match = re.match(r'^x2\(\s*1\s*,\s*n\s*=\s*(\d+)\s*\)\s*=\s*(\d+\.?\d*)$', normalized_stat, re.IGNORECASE)
    if chi_match:
        n_val = float(chi_match.group(1))
        chi_val = float(chi_match.group(2))
        return math.sqrt(chi_val / n_val)

    return None


def convert_effect_size(es_value, es_type, n1=None, n2=None):
    """
    Convert a single effect size to Pearson's r.

    Args:
        es_value: The effect size value (numeric or string for test statistics)
        es_type: The type of effect size (e.g., "d", "r", "or", "etasq")
        n1: Sample size for group 1 (optional, used for Cohen's d conversion)
        n2: Sample size for group 2 (optional, used for Cohen's d conversion)

    Returns:
        Pearson's r value, or None if conversion is not possible.
    """
    if es_value is None or es_type is None:
        return None

    # Handle pandas NA values
    if pd.isna(es_value) or pd.isna(es_type):
        return None

    # Normalize effect size type
    es_type_lower = str(es_type).lower().strip()

    # Replace curly apostrophes with straight ones
    es_type_lower = es_type_lower.replace("'", "'")

    # Check if this is a non-convertible type
    if es_type_lower in [t.lower() for t in CANNOT_CONVERT]:
        return None

    # Get canonical type
    canonical_type = ESTYPE_MAP.get(es_type_lower)

    if canonical_type is None:
        # Unknown type
        return None

    # Try to convert es_value to float (for non-test-statistic types)
    if canonical_type != "test-stat":
        try:
            es_value = float(es_value)
        except (ValueError, TypeError):
            return None

    # Perform conversion based on canonical type
    if canonical_type == "r":
        # Already r or phi, return as-is
        return es_value

    elif canonical_type == "r2":
        return r2_to_r(es_value)

    elif canonical_type == "d":
        return d_to_r(es_value, n1, n2)

    elif canonical_type == "or":
        return or_to_r(es_value)

    elif canonical_type == "eta2":
        return eta2_to_r(es_value)

    elif canonical_type == "f":
        return f_to_r(es_value)

    elif canonical_type == "f2":
        return f2_to_r(es_value)

    elif canonical_type == "test-stat":
        return parse_test_statistic(str(es_value))

    return None


def calculate_effect_sizes(df):
    """
    Calculate original_es_r and replication_es_r from effect size data.

    Only fills in values where:
    1. The target column (original_es_r or replication_es_r) is currently empty/missing
    2. All necessary source data is available (es value, es type, and N if needed for d)

    Args:
        df: DataFrame with effect size columns

    Returns:
        DataFrame with original_es_r and replication_es_r filled in where possible
    """
    print("\nCalculating effect sizes (converting to Pearson's r)...")

    # Ensure target columns exist
    if 'original_es_r' not in df.columns:
        df['original_es_r'] = None
    if 'replication_es_r' not in df.columns:
        df['replication_es_r'] = None

    original_filled = 0
    replication_filled = 0

    for idx, row in df.iterrows():
        # Process original effect size
        if is_empty(row.get('original_es_r')):
            original_es = row.get('original_es')
            original_es_type = row.get('original_es_type')
            original_n = row.get('original_n')

            if not is_empty(original_es) and not is_empty(original_es_type):
                # Get sample sizes for d conversion if available
                n1 = None
                n2 = None
                if not is_empty(original_n):
                    try:
                        # Assume equal groups if only total N is given
                        total_n = float(original_n)
                        n1 = total_n / 2
                        n2 = total_n / 2
                    except (ValueError, TypeError):
                        pass

                r_value = convert_effect_size(original_es, original_es_type, n1, n2)
                # Only store non-zero values (0 indicates failed conversion or meaningless data)
                if r_value is not None and r_value != 0:
                    df.at[idx, 'original_es_r'] = r_value
                    original_filled += 1

        # Process replication effect size
        if is_empty(row.get('replication_es_r')):
            replication_es = row.get('replication_es')
            replication_es_type = row.get('replication_es_type')
            replication_n = row.get('replication_n')

            if not is_empty(replication_es) and not is_empty(replication_es_type):
                # Get sample sizes for d conversion if available
                n1 = None
                n2 = None
                if not is_empty(replication_n):
                    try:
                        total_n = float(replication_n)
                        n1 = total_n / 2
                        n2 = total_n / 2
                    except (ValueError, TypeError):
                        pass

                r_value = convert_effect_size(replication_es, replication_es_type, n1, n2)
                # Only store non-zero values (0 indicates failed conversion or meaningless data)
                if r_value is not None and r_value != 0:
                    df.at[idx, 'replication_es_r'] = r_value
                    replication_filled += 1

    print(f"  ✓ Filled {original_filled} original_es_r values")
    print(f"  ✓ Filled {replication_filled} replication_es_r values")

    return df

def format_author_initial(name):
    """
    Add periods after single-letter initials in an author name.
    e.g., "Jonathan W Schooler" → "Jonathan W. Schooler"
         "J Lukas Thürmer" → "J. Lukas Thürmer"
    """
    if not isinstance(name, str) or not name.strip():
        return name
    parts = name.strip().split()
    formatted_parts = []
    for part in parts:
        if len(part) == 1 and part.isalpha():
            formatted_parts.append(part + '.')
        else:
            formatted_parts.append(part)
    return ' '.join(formatted_parts)


def format_authors_string(authors_str):
    """
    Format a semicolon-separated authors string, adding periods after single-letter initials.
    """
    if not isinstance(authors_str, str) or not authors_str.strip():
        return authors_str
    authors = [format_author_initial(a.strip()) for a in authors_str.split(';')]
    return '; '.join(authors)


def is_abbreviated_journal(journal_name):
    """
    Detect if a journal name is likely abbreviated.

    Criteria:
    - Contains periods (excluding trailing period only)
    - This catches patterns like "Dev. Sci", "J. Exp. Psychol.", "Psychol. Sci"

    Returns True if likely abbreviated, False otherwise.
    """
    if not isinstance(journal_name, str) or not journal_name.strip():
        return False

    name = journal_name.strip()

    # Remove trailing period for analysis
    if name.endswith('.'):
        name = name[:-1]

    # Check for periods inside the name (strong indicator of abbreviation)
    # e.g., "Dev. Sci" or "J. Exp. Psychol"
    if '.' in name:
        return True

    return False


def needs_enrichment(row, prefix):
    """Check if any key metadata fields are missing or abbreviated"""
    fields_to_check = ['authors', 'title', 'journal', 'volume', 'issue', 'pages', 'year']

    for field in fields_to_check:
        col_name = f"{prefix}_{field}"
        if col_name not in row.index or is_empty(row.get(col_name)):
            return True
        # Special handling for journal field - check if abbreviated
        if field == 'journal' and is_abbreviated_journal(row.get(col_name)):
            return True
        # Special handling for authors - check if first names are abbreviated
        if field == 'authors' and _authors_have_abbreviations(row.get(col_name)):
            return True

    return False

def enrich_from_metadata(row, prefix, metadata):
    """Fill row with metadata from API calls"""
    if not metadata:
        return row

    field_mapping = {
        'authors': f'{prefix}_authors',
        'title': f'{prefix}_title',
        'journal': f'{prefix}_journal',
        'volume': f'{prefix}_volume',
        'issue': f'{prefix}_issue',
        'pages': f'{prefix}_pages',
        'year': f'{prefix}_year',
    }

    for meta_key, col_name in field_mapping.items():
        # Fill if column doesn't exist or current value is empty
        # OR if it's journal field and current value is abbreviated
        current_val = row.get(col_name) if col_name in row.index else None
        value = metadata.get(meta_key)
        if not value:
            continue

        should_fill = (
            col_name not in row.index
            or is_empty(current_val)
            or (meta_key == 'journal' and is_abbreviated_journal(current_val))
            or (meta_key == 'authors' and _new_authors_are_better(current_val, format_authors_string(value)))
        )

        if should_fill:
            # Format author names (add periods after single-letter initials)
            if meta_key == 'authors':
                value = format_authors_string(value)
            # Normalize year to int and validate range
            if meta_key == 'year':
                try:
                    year_int = int(float(value))
                    if 1800 <= year_int <= 2030:
                        value = year_int
                    else:
                        logger.warning(f"Year {year_int} out of range for {col_name}, skipping")
                        continue
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse year '{value}' for {col_name}, skipping")
                    continue
            row[col_name] = value

    return row

def _authors_overlap(existing_authors, fetched_authors):
    """Check if at least one author surname appears in both author strings.
    Handles formats like 'Given Family' and 'Family, Given'."""
    if not existing_authors or not fetched_authors:
        return False
    existing_str = str(existing_authors).lower()
    fetched_str = str(fetched_authors).lower()
    # Extract surnames: split by semicolon, take last word of each name
    def get_surnames(s):
        surnames = set()
        for name in s.split(';'):
            name = name.strip()
            if not name:
                continue
            # Handle "Family, Given" format
            if ',' in name:
                surnames.add(name.split(',')[0].strip())
            else:
                # "Given Family" format - take last word
                parts = name.split()
                if parts:
                    surnames.add(parts[-1].strip().rstrip('.'))
        return surnames
    existing_surnames = get_surnames(existing_str)
    fetched_surnames = get_surnames(fetched_str)
    return bool(existing_surnames & fetched_surnames)


def sanity_check_metadata(row, prefix, metadata):
    """
    Check if fetched metadata matches existing data.
    Returns True if metadata is likely correct, False otherwise.
    Checks year field and title similarity when available.
    Allows a year window of +/- 5 years if title and authors align
    (common for working papers published later as journal articles).
    """
    if not metadata:
        return False

    # Check title similarity when both exist
    title_match = False
    title_col = f"{prefix}_title"
    if title_col in row.index and not is_empty(row[title_col]) and metadata.get('title'):
        existing_title = str(row[title_col]).lower().strip()
        fetched_title = str(metadata['title']).lower().strip()
        similarity = SequenceMatcher(None, existing_title, fetched_title).ratio()
        if similarity < 0.6:
            logger.warning(f"Title mismatch for {prefix} (similarity={similarity:.2f}): "
                           f"existing='{existing_title[:60]}' vs fetched='{fetched_title[:60]}'")
            return False
        title_match = similarity >= 0.6

    # Check author overlap
    authors_col = f"{prefix}_authors"
    authors_match = False
    if authors_col in row.index and not is_empty(row[authors_col]) and metadata.get('authors'):
        authors_match = _authors_overlap(row[authors_col], metadata['authors'])

    # Check year field
    year_col = f"{prefix}_year"
    if year_col in row.index and not is_empty(row[year_col]):
        existing_value = str(row[year_col]).strip()
        fetched_value = str(metadata.get('year', "")).strip()

        if fetched_value:
            existing_year = existing_value.replace(".0", "")
            fetched_year = fetched_value.replace(".0", "")

            if existing_year != fetched_year:
                # Allow +/- 5 year window if title AND authors match
                # (working papers often published years later as journal articles)
                try:
                    year_diff = abs(int(existing_year) - int(fetched_year))
                    if year_diff <= 5 and (title_match and authors_match):
                        logger.info(f"Year differs by {year_diff}y for {prefix} "
                                    f"(existing={existing_year}, fetched={fetched_year}) "
                                    f"but title+authors match — accepting")
                    elif year_diff <= 5 and title_match:
                        logger.info(f"Year differs by {year_diff}y for {prefix} "
                                    f"(existing={existing_year}, fetched={fetched_year}) "
                                    f"title matches but no author data to confirm — accepting")
                    else:
                        logger.warning(f"Year mismatch for {prefix}: existing={existing_value}, fetched={fetched_value}")
                        return False
                except ValueError:
                    logger.warning(f"Year mismatch for {prefix}: existing={existing_value}, fetched={fetched_value}")
                    return False

    return True

def _cache_result_is_good(result, require_doi=False):
    """Check if a cached result is worth keeping or should be re-fetched.
    Returns True if the result has enough data to skip re-fetching."""
    if result is None:
        return False
    if not isinstance(result, dict):
        return False
    if require_doi and not result.get('doi'):
        return False
    # Count how many fields have real values
    filled = sum(1 for k, v in result.items() if v not in [None, "", "NaN"])
    # Re-fetch if less than 3 fields filled (too sparse to be useful)
    return filled >= 3


def _cache_get_or_fetch(cache, key, fetch_fn, cache_lock=None, require_doi=False):
    """Thread-safe cache lookup. On miss, calls fetch_fn() outside the lock,
    then stores the result. Re-fetches if cached result is incomplete
    (missing DOI when require_doi=True, or too few fields filled)."""
    if cache_lock:
        with cache_lock:
            if key in cache and _cache_result_is_good(cache[key], require_doi):
                return cache[key], True  # (result, was_cached)
    elif key in cache and _cache_result_is_good(cache[key], require_doi):
        return cache[key], True

    # Fetch outside the lock so other threads aren't blocked on I/O
    result = fetch_fn()

    if cache_lock:
        with cache_lock:
            cache[key] = result
    else:
        cache[key] = result

    return result, False


def process_row(row, row_idx, total_rows, doi_cache=None, title_cache=None, cache_lock=None):
    """Process a single row to enrich metadata.
    cache_lock: optional threading.Lock for thread-safe cache access."""
    if doi_cache is None:
        doi_cache = {}
    if title_cache is None:
        title_cache = {}

    def _evict_title_cache(key):
        """Remove a bad result from the title cache so it's re-fetched next run."""
        if cache_lock:
            with cache_lock:
                title_cache.pop(key, None)
        else:
            title_cache.pop(key, None)

    # Stagger worker startup to avoid simultaneous API bursts — skip if no enrichment needed
    if needs_enrichment(row, 'original') or needs_enrichment(row, 'replication'):
        time.sleep(random.uniform(0, 2))
    print(f"\nProcessing row {row_idx + 1}/{total_rows}...")

    # ===== PROCESS ORIGINAL STUDY =====
    original_url = row.get('original_url')
    original_doi = extract_doi_from_url(original_url)

    # Validate DOI format before making API calls
    if original_doi and not is_valid_doi(original_doi):
        logger.warning(f"Invalid DOI format for original: {original_doi}, skipping API lookup")
        original_doi = None

    if original_doi and needs_enrichment(row, 'original'):
        metadata, was_cached = _cache_get_or_fetch(
            doi_cache, original_doi,
            lambda: fetch_metadata_from_doi(original_doi),
            cache_lock
        )
        if was_cached:
            print(f"  Using cached metadata for original DOI: {original_doi}")
        else:
            print(f"  Fetched metadata for original DOI: {original_doi}")
            time.sleep(0.3)  # Rate limiting
        row = enrich_from_metadata(row, 'original', metadata)

    # If no DOI URL but title exists, try to fetch DOI from title
    elif is_empty(original_url) and not is_empty(row.get('original_title')):
        original_title = row.get('original_title')
        original_authors = row.get('original_authors')
        original_journal = row.get('original_journal')
        original_year = row.get('original_year')
        original_volume = row.get('original_volume')
        # Include year in cache key so same title with different expected years
        # gets separate cache entries (avoids returning a 2025 paper for a 2020 query)
        year_suffix = f"|{str(original_year).replace('.0', '')}" if original_year and str(original_year).strip() not in ('', 'nan', 'NaN') else ""
        title_key = original_title.lower().strip() + year_suffix
        metadata, was_cached = _cache_get_or_fetch(
            title_cache, title_key,
            lambda: fetch_metadata_from_title(
                original_title,
                authors=original_authors,
                journal=original_journal,
                year=original_year,
                volume=original_volume
            ),
            cache_lock,
            require_doi=True
        )
        if was_cached:
            print(f"  Using cached metadata for original title: {original_title}")
        else:
            print(f"  Searched by title: {original_title}")

        if metadata and metadata.get('doi'):
            if sanity_check_metadata(row, 'original', metadata):
                normalized_doi = normalize_doi(metadata['doi'])
                if normalized_doi:
                    print(f"  ✓ Found and verified DOI: {normalized_doi}")
                    row['original_url'] = f"https://doi.org/{normalized_doi}"
                    row = enrich_from_metadata(row, 'original', metadata)
                else:
                    print(f"  ✗ Could not normalize DOI: {metadata['doi']}")
            else:
                print(f"  ✗ DOI failed sanity check, not using: {metadata['doi']}")
                _evict_title_cache(title_key)
        elif metadata and metadata.get('pmid'):
            if sanity_check_metadata(row, 'original', metadata):
                pmid = metadata['pmid']
                pmid_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                print(f"  ✓ No DOI but found PMID: {pmid_url}")
                row['original_url'] = pmid_url
                row = enrich_from_metadata(row, 'original', metadata)
            else:
                print(f"  ✗ PMID failed sanity check, not using: {metadata['pmid']}")
                _evict_title_cache(title_key)
        elif metadata and metadata.get('url'):
            if sanity_check_metadata(row, 'original', metadata):
                print(f"  ✓ No DOI/PMID but found URL: {metadata['url']}")
                row['original_url'] = metadata['url']
                row = enrich_from_metadata(row, 'original', metadata)
            else:
                print(f"  ✗ URL failed sanity check")
                _evict_title_cache(title_key)
        else:
            print(f"  ✗ Could not find DOI/PMID from title")

            # Retry with first part of title before colon (e.g., subtitles often cause search failures)
            if ':' in original_title:
                short_title = original_title.split(':')[0].strip()
                if len(short_title) >= 15:  # Only retry if the first part is meaningful
                    print(f"  ↻ Retrying with shortened title: {short_title}")
                    short_key = short_title.lower().strip() + year_suffix
                    metadata2, was_cached2 = _cache_get_or_fetch(
                        title_cache, short_key,
                        lambda: fetch_metadata_from_title(
                            short_title,
                            authors=original_authors,
                            journal=original_journal,
                            year=original_year,
                            volume=original_volume
                        ),
                        cache_lock,
                        require_doi=True
                    )
                    if metadata2 and metadata2.get('doi'):
                        if sanity_check_metadata(row, 'original', metadata2):
                            normalized_doi = normalize_doi(metadata2['doi'])
                            if normalized_doi:
                                print(f"  ✓ Found DOI via shortened title: {normalized_doi}")
                                row['original_url'] = f"https://doi.org/{normalized_doi}"
                                row = enrich_from_metadata(row, 'original', metadata2)
                            else:
                                print(f"  ✗ Could not normalize DOI: {metadata2['doi']}")
                        else:
                            print(f"  ✗ DOI from shortened title failed sanity check")
                            _evict_title_cache(short_key)
                    elif metadata2 and metadata2.get('pmid'):
                        if sanity_check_metadata(row, 'original', metadata2):
                            pmid = metadata2['pmid']
                            pmid_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                            print(f"  ✓ Found PMID via shortened title: {pmid_url}")
                            row['original_url'] = pmid_url
                            row = enrich_from_metadata(row, 'original', metadata2)
                    elif metadata2 and metadata2.get('url'):
                        if sanity_check_metadata(row, 'original', metadata2):
                            print(f"  ✓ Found URL via shortened title: {metadata2['url']}")
                            row['original_url'] = metadata2['url']
                            row = enrich_from_metadata(row, 'original', metadata2)
                    else:
                        print(f"  ✗ Shortened title also failed")

        if not was_cached:
            time.sleep(0.3)  # Rate limiting

    # ===== PROCESS REPLICATION STUDY =====
    replication_url = row.get('replication_url')
    replication_doi = extract_doi_from_url(replication_url)

    # Validate DOI format before making API calls
    if replication_doi and not is_valid_doi(replication_doi):
        logger.warning(f"Invalid DOI format for replication: {replication_doi}, skipping API lookup")
        replication_doi = None

    if replication_doi and needs_enrichment(row, 'replication'):
        metadata, was_cached = _cache_get_or_fetch(
            doi_cache, replication_doi,
            lambda: fetch_metadata_from_doi(replication_doi),
            cache_lock
        )
        if was_cached:
            print(f"  Using cached metadata for replication DOI: {replication_doi}")
        else:
            print(f"  Fetched metadata for replication DOI: {replication_doi}")
            time.sleep(0.3)  # Rate limiting
        row = enrich_from_metadata(row, 'replication', metadata)

    # If no DOI URL but title exists, try to fetch DOI from title
    elif is_empty(replication_url) and not is_empty(row.get('replication_title')):
        replication_title = row.get('replication_title')
        replication_authors = row.get('replication_authors')
        replication_journal = row.get('replication_journal')
        replication_year = row.get('replication_year')
        replication_volume = row.get('replication_volume')
        year_suffix = f"|{str(replication_year).replace('.0', '')}" if replication_year and str(replication_year).strip() not in ('', 'nan', 'NaN') else ""
        title_key = replication_title.lower().strip() + year_suffix
        metadata, was_cached = _cache_get_or_fetch(
            title_cache, title_key,
            lambda: fetch_metadata_from_title(
                replication_title,
                authors=replication_authors,
                journal=replication_journal,
                year=replication_year,
                volume=replication_volume
            ),
            cache_lock,
            require_doi=True
        )
        if was_cached:
            print(f"  Using cached metadata for replication title: {replication_title}")
        else:
            print(f"  Searched by title: {replication_title}")

        if metadata and metadata.get('doi'):
            if sanity_check_metadata(row, 'replication', metadata):
                normalized_doi = normalize_doi(metadata['doi'])
                if normalized_doi:
                    print(f"  ✓ Found and verified DOI: {normalized_doi}")
                    row['replication_url'] = f"https://doi.org/{normalized_doi}"
                    row = enrich_from_metadata(row, 'replication', metadata)
                else:
                    print(f"  ✗ Could not normalize DOI: {metadata['doi']}")
            else:
                print(f"  ✗ DOI failed sanity check, not using: {metadata['doi']}")
                _evict_title_cache(title_key)
        elif metadata and metadata.get('pmid'):
            if sanity_check_metadata(row, 'replication', metadata):
                pmid = metadata['pmid']
                pmid_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                print(f"  ✓ No DOI but found PMID: {pmid_url}")
                row['replication_url'] = pmid_url
                row = enrich_from_metadata(row, 'replication', metadata)
            else:
                print(f"  ✗ PMID failed sanity check, not using: {metadata['pmid']}")
                _evict_title_cache(title_key)
        elif metadata and metadata.get('url'):
            if sanity_check_metadata(row, 'replication', metadata):
                print(f"  ✓ No DOI/PMID but found URL: {metadata['url']}")
                row['replication_url'] = metadata['url']
                row = enrich_from_metadata(row, 'replication', metadata)
            else:
                print(f"  ✗ URL failed sanity check")
                _evict_title_cache(title_key)
        else:
            print(f"  ✗ Could not find DOI/PMID from title")

            # Retry with first part of title before colon
            if ':' in replication_title:
                short_title = replication_title.split(':')[0].strip()
                if len(short_title) >= 15:
                    print(f"  ↻ Retrying with shortened title: {short_title}")
                    short_key = short_title.lower().strip() + year_suffix
                    metadata2, was_cached2 = _cache_get_or_fetch(
                        title_cache, short_key,
                        lambda: fetch_metadata_from_title(
                            short_title,
                            authors=replication_authors,
                            journal=replication_journal,
                            year=replication_year,
                            volume=replication_volume
                        ),
                        cache_lock,
                        require_doi=True
                    )
                    if metadata2 and metadata2.get('doi'):
                        if sanity_check_metadata(row, 'replication', metadata2):
                            normalized_doi = normalize_doi(metadata2['doi'])
                            if normalized_doi:
                                print(f"  ✓ Found DOI via shortened title: {normalized_doi}")
                                row['replication_url'] = f"https://doi.org/{normalized_doi}"
                                row = enrich_from_metadata(row, 'replication', metadata2)
                            else:
                                print(f"  ✗ Could not normalize DOI: {metadata2['doi']}")
                        else:
                            print(f"  ✗ DOI from shortened title failed sanity check")
                            _evict_title_cache(short_key)
                    elif metadata2 and metadata2.get('pmid'):
                        if sanity_check_metadata(row, 'replication', metadata2):
                            pmid = metadata2['pmid']
                            pmid_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                            print(f"  ✓ Found PMID via shortened title: {pmid_url}")
                            row['replication_url'] = pmid_url
                            row = enrich_from_metadata(row, 'replication', metadata2)
                    elif metadata2 and metadata2.get('url'):
                        if sanity_check_metadata(row, 'replication', metadata2):
                            print(f"  ✓ Found URL via shortened title: {metadata2['url']}")
                            row['replication_url'] = metadata2['url']
                            row = enrich_from_metadata(row, 'replication', metadata2)
                    else:
                        print(f"  ✗ Shortened title also failed")

        if not was_cached:
            time.sleep(0.3)  # Rate limiting

    return row


def filter_columns(df, data_dict_path=None):
    """Keep only columns that appear in data_dictionary.csv, preserving order from data dictionary"""
    if data_dict_path is None:
        data_dict_path = os.path.join(DATA_DIR, 'data_dictionary.csv')
    print("\nFiltering columns based on data_dictionary.csv...")

    data_dict = pd.read_csv(data_dict_path)
    valid_columns = data_dict['column_name'].tolist()

    # Keep only columns that exist in both the dataframe and the valid columns list
    # Order them according to the order in data_dictionary.csv
    columns_to_keep = [col for col in valid_columns if col in df.columns]

    print(f"  Keeping {len(columns_to_keep)} valid columns out of {len(df.columns)} total")
    print(f"  Columns ordered according to data_dictionary.csv")

    return df[columns_to_keep]

def normalize_year_columns(df):
    """Convert original_year and replication_year from float strings (e.g. 2018.0) to integers."""
    for col in ['original_year', 'replication_year']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    return df


def normalize_discipline_column(df):
    """Convert discipline column values to lowercase"""
    if 'discipline' in df.columns:
        print("\nNormalizing discipline column (converting to lowercase)...")
        df['discipline'] = df['discipline'].apply(
            lambda x: x.lower() if pd.notna(x) and isinstance(x, str) else x
        )
        print(f"  ✓ Converted discipline values to lowercase")
    return df

def populate_field_from_discipline(df):
    """Fill empty 'field' values using the topic ontology and the 'discipline' column.

    Expects discipline to already be lowercase (run after normalize_discipline_column).
    Builds a lowercased discipline→field lookup from the ontology, then fills any
    rows where 'field' is empty but 'discipline' matches a known ontology entry.
    """
    if 'discipline' not in df.columns:
        return df

    if not os.path.exists(ONTOLOGY_PATH):
        print(f"  ⚠ Ontology file not found: {ONTOLOGY_PATH}, skipping field population")
        return df

    with open(ONTOLOGY_PATH, 'r') as f:
        ontology = json.load(f)

    disc_to_field = {
        disc_name.lower(): field_name
        for field_name, disciplines in ontology.items()
        for disc_name in disciplines
    }

    if 'field' not in df.columns:
        df['field'] = ''

    filled = 0
    unmatched = set()
    for idx, row in df.iterrows():
        if not is_empty(row.get('field')):
            continue
        disc = str(row.get('discipline') or '').strip().lower()
        if not disc:
            continue
        if disc in disc_to_field:
            df.at[idx, 'field'] = disc_to_field[disc]
            filled += 1
        else:
            unmatched.add(disc)

    print(f"\nPopulating 'field' from discipline (ontology lookup)...")
    print(f"  ✓ Filled {filled} 'field' values")
    if unmatched:
        print(f"  ⚠ {len(unmatched)} discipline value(s) not found in ontology: {sorted(unmatched)}")

    return df


def _normalize_journal_key(s):
    """Normalize a journal name for lookup: lowercase, strip periods, collapse spaces."""
    s = s.lower().replace('.', '').strip()
    return re.sub(r'\s+', ' ', s)


def normalize_journal_names(df):
    """Normalize journal names using journal_name_mappings.json

    Keys in the JSON are normalized (lowercase, no periods, single spaces).
    Input journal names are normalized the same way before lookup, making
    matching case-insensitive and period-insensitive.

    Applies in order:
    1. Abbreviation expansion (e.g., "Dev. Sci" → "Developmental Science")
    2. Variant form standardization (e.g., "PLOS ONE" → "PLOS One")
    3. HTML entity fixes (e.g., "&amp;" → "&")
    """
    mappings_path = os.path.join(DATA_DIR, 'journal_name_mappings.json')

    if not os.path.exists(mappings_path):
        print(f"  ⚠ Journal name mappings file not found: {mappings_path}")
        return df

    try:
        with open(mappings_path, 'r') as f:
            mappings = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  ⚠ Could not load journal name mappings: {e}")
        return df

    abbreviations = mappings.get('abbreviations', {})
    variant_forms = mappings.get('variant_forms', {})

    # Build set of known full names (values) to avoid false-positive expansions.
    # A journal that is already a canonical full name should not be re-expanded
    # even if its normalized form collides with an abbreviation key for a
    # different journal (e.g., "Social Psychology" is a real journal, not an
    # abbreviation for "Social Psychology Quarterly").
    known_full_names = set()
    for v in abbreviations.values():
        known_full_names.add(_normalize_journal_key(v))
    for v in variant_forms.values():
        known_full_names.add(_normalize_journal_key(v))

    print("\nNormalizing journal names...")

    total_replacements = 0

    for col in ['replication_journal', 'original_journal']:
        if col not in df.columns:
            continue

        col_replacements = 0

        # Apply mappings to each value
        def apply_mappings(journal_name):
            nonlocal col_replacements

            if pd.isna(journal_name) or not isinstance(journal_name, str):
                return journal_name

            original = journal_name.strip()

            # Normalize for lookup (lowercase, strip periods, collapse spaces)
            normalized = _normalize_journal_key(original)

            # Skip abbreviation expansion if the name is already a known full name.
            # This prevents false positives where a full journal name collides
            # with an abbreviation key for a different journal.
            if normalized not in known_full_names:
                # Step 1: Apply abbreviations (normalized key match)
                if normalized in abbreviations:
                    expansion = abbreviations[normalized]
                    # Guard against NLM entries that merely add a location qualifier
                    # or subtitle to what is already the full name, e.g.:
                    #   "AIDS" → "AIDS (London, England)"
                    #   "BMJ"  → "BMJ (Clinical research ed.)"
                    # If the expansion starts with the original text followed by a
                    # separator (space, paren, semicolon, colon), the input is
                    # already the canonical name and should not be changed.
                    # But allow cases where a truncated word gets completed, e.g.:
                    #   "Death Stud" → "Death Studies" (word continues, not a qualifier)
                    exp_norm = _normalize_journal_key(expansion)
                    is_qualifier_expansion = (
                        exp_norm.startswith(normalized) and
                        len(exp_norm) > len(normalized) and
                        exp_norm[len(normalized)] in ' ;:,('
                    )
                    if not is_qualifier_expansion:
                        col_replacements += 1
                        return expansion

            # Step 2: Apply variant forms (normalized key match)
            if normalized in variant_forms:
                col_replacements += 1
                return variant_forms[normalized]

            # Step 3: Apply HTML entity fixes
            if '&amp;' in original:
                col_replacements += 1
                return original.replace('&amp;', '&')

            return original

        df[col] = df[col].apply(apply_mappings)

        if col_replacements > 0:
            print(f"  ✓ Normalized {col_replacements} journal names in {col}")
            total_replacements += col_replacements

    if total_replacements == 0:
        print(f"  No journal names required normalization")

    return df

def reorder_columns(df, data_dict_path=None):
    """Reorder columns according to the order in data_dictionary.csv"""
    if data_dict_path is None:
        data_dict_path = os.path.join(DATA_DIR, 'data_dictionary.csv')
    data_dict = pd.read_csv(data_dict_path)
    valid_columns = data_dict['column_name'].tolist()
    
    # Get columns that exist in both the dataframe and the data dictionary
    # Order them according to the order in data_dictionary.csv
    columns_in_order = [col for col in valid_columns if col in df.columns]
    
    # Add any columns that exist in df but not in data dictionary (shouldn't happen after filtering, but just in case)
    remaining_columns = [col for col in df.columns if col not in columns_in_order]
    
    # Combine: ordered columns first, then any remaining columns
    final_column_order = columns_in_order + remaining_columns
    
    return df[final_column_order]

def _normalize_val(v):
    """Normalize a cell value to a comparable string (nan/None/empty → '').
    Strips trailing .0 from floats so 123.0 == 123 after CSV round-tripping."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ''
    if isinstance(v, float) and v == int(v):
        s = str(int(v))
    else:
        s = str(v).strip()
    if s.lower() == 'nan':
        return ''
    # Also handle string "123.0" → "123"
    if s.endswith('.0') and s[:-2].lstrip('-').isdigit():
        s = s[:-2]
    return s


def _normalize_url_to_doi(url_val):
    """Extract DOI from a URL for comparison, falling back to stripped string."""
    s = _normalize_val(url_val)
    if not s:
        return ''
    doi = extract_doi_from_url(s)
    return doi.lower() if doi else s.lower()


AUTO_DUP_FIELDS = ['original_url', 'replication_url', 'description',
                    'replication_n', 'replication_es', 'replication_es_type']

# Fields where we compare by DOI rather than raw string
_DOI_FIELDS = {'original_url', 'replication_url'}


def is_auto_duplicate(incoming_row, master_row):
    """Check if incoming row is an automatic duplicate of a master row.
    Returns True if all 6 key fields match (same effect, same replication data)."""
    for col in AUTO_DUP_FIELDS:
        if col in _DOI_FIELDS:
            inc_val = _normalize_url_to_doi(incoming_row.get(col))
            mst_val = _normalize_url_to_doi(master_row.get(col)) if col in master_row.index else ''
        else:
            inc_val = _normalize_val(incoming_row.get(col))
            mst_val = _normalize_val(master_row.get(col)) if col in master_row.index else ''
        if inc_val != mst_val:
            return False
    return True


def find_duplicate_matches(row, master_df):
    """
    Find potential duplicate rows in master based on DOI match.
    Extracts DOIs from original_url and replication_url for comparison,
    so http://doi.org/... and https://doi.org/... are treated as identical.
    Returns list of matching master row indices.
    """
    if master_df.empty:
        return []
    orig_doi = _normalize_url_to_doi(row.get('original_url', ''))
    rep_doi = _normalize_url_to_doi(row.get('replication_url', ''))
    if not orig_doi and not rep_doi:
        return []
    matches = master_df[
        (master_df['original_url'].fillna('').apply(_normalize_url_to_doi) == orig_doi) &
        (master_df['replication_url'].fillna('').apply(_normalize_url_to_doi) == rep_doi)
    ]
    return matches.index.tolist()


def merge_into_master(master_df, match_idx, incoming_row, force_replace_fields=None):
    """Merge incoming row into an existing master row, filling empty fields.
    For 'description', keeps whichever is longer.
    For 'result', keeps existing unless 'result' is in force_replace_fields.
    Returns the number of fields that were filled/updated."""
    if force_replace_fields is None:
        force_replace_fields = set()
    filled = 0
    for col in incoming_row.index:
        if col not in master_df.columns:
            continue
        incoming_val = incoming_row[col]
        existing_val = master_df.at[match_idx, col]
        if col in force_replace_fields and not is_empty(incoming_val):
            master_df.at[match_idx, col] = incoming_val
            filled += 1
        elif col == 'result' and not is_empty(existing_val):
            continue  # keep existing result unless force-replaced
        elif col == 'description':
            inc_len = len(str(incoming_val).strip()) if not is_empty(incoming_val) else 0
            ext_len = len(str(existing_val).strip()) if not is_empty(existing_val) else 0
            if inc_len > ext_len:
                master_df.at[match_idx, col] = incoming_val
                filled += 1
        elif not is_empty(incoming_val) and is_empty(existing_val):
            master_df.at[match_idx, col] = incoming_val
            filled += 1
    return filled


def prompt_duplicate_action(new_row, master_df, match_indices, ingest_idx, total_ingest,
                            dup_number=None, total_dups=None):
    """
    Display comparison between new row and existing master row(s) and prompt user for action.
    Returns: (action, target_master_row_index_or_None)
      action: 'add', 'replace', 'skip', 'merge', 'add_all', 'skip_all', or 'merge_all'
      target: specific master row index for merge/replace, None for skip/add/batch actions
    """
    table_fields = ['original_es', 'replication_es', 'original_es_type',
                     'replication_es_type', 'result', 'original_n', 'replication_n',
                     'original_es_r', 'replication_es_r', 'discipline', 'subdiscipline',
                     'original_title', 'original_authors', 'original_year',
                     'replication_title', 'replication_authors', 'replication_year']

    n_matches = len(match_indices)
    dup_label = ""
    if dup_number is not None and total_dups is not None:
        dup_label = f"  [duplicate {dup_number}/{total_dups}]"

    print(f"\n{'='*120}")
    print(f"  POSSIBLE DUPLICATE: Ingest row {ingest_idx + 1}/{total_ingest}{dup_label}")
    print(f"  Matches {n_matches} existing row{'s' if n_matches > 1 else ''} in master")
    print(f"{'='*120}")

    # Show incoming row info
    new_desc = _normalize_val(new_row.get('description'))
    print(f"  INCOMING ROW:")
    print(f"    original_url:    {_normalize_val(new_row.get('original_url'))}")
    print(f"    replication_url: {_normalize_val(new_row.get('replication_url'))}")
    print(f"    description:     {new_desc if new_desc else '(empty)'}")
    print(f"    result: {_normalize_val(new_row.get('result'))}  |  "
          f"orig_es: {_normalize_val(new_row.get('original_es'))} ({_normalize_val(new_row.get('original_es_type'))})  |  "
          f"rep_es: {_normalize_val(new_row.get('replication_es'))} ({_normalize_val(new_row.get('replication_es_type'))})")

    # Show each match with a number
    for i, mi in enumerate(match_indices, 1):
        master_row = master_df.loc[mi]
        existing_desc = _normalize_val(master_row.get('description'))

        print(f"\n  ── [{i}] master row {mi} ──")
        print(f"  description: {existing_desc if existing_desc else '(empty)'}")
        if existing_desc and new_desc and existing_desc == new_desc:
            print(f"  (description identical to incoming)")

        # Diff table
        print(f"  {'FIELD':<24} {'EXISTING':<45} {'NEW':<45} NOTE")
        print(f"  {'─'*24} {'─'*45} {'─'*45} {'─'*20}")
        for field in table_fields:
            existing_val = _normalize_val(master_row.get(field))
            new_val = _normalize_val(new_row.get(field))
            # Truncate for table columns
            ev_display = (existing_val[:42] + "...") if len(existing_val) > 42 else existing_val
            nv_display = (new_val[:42] + "...") if len(new_val) > 42 else new_val
            if existing_val == new_val:
                marker, note = " ", ""
            elif not existing_val and new_val:
                marker, note = "+", "← merge would fill"
            elif existing_val and not new_val:
                marker, note = " ", "(incoming empty)"
            else:
                marker, note = "*", "DIFFERS"
            print(f" {marker}{field:<23} {ev_display:<45} {nv_display:<45} {note}")

    # Options
    print(f"\n  What would you like to do?")
    print(f"    [s]  Skip (do not add this row)")
    print(f"    [a]  Add as new row (different effect, not a duplicate)")
    if n_matches == 1:
        print(f"    [m]  Merge into existing row {match_indices[0]} (fill empty fields)")
        print(f"    [r]  Replace existing row {match_indices[0]} with this row")
    else:
        for i, mi in enumerate(match_indices, 1):
            print(f"    [m{i}] Merge into master row {mi}")
        for i, mi in enumerate(match_indices, 1):
            print(f"    [r{i}] Replace master row {mi}")
    print(f"    [S]  Skip ALL remaining duplicates")
    print(f"    [M]  Merge ALL remaining (into first match)")
    print(f"    [A]  Add ALL remaining as new rows")

    # Build set of valid choices
    valid_single = {'s', 'a', 'S', 'M', 'A'}
    if n_matches == 1:
        valid_single.update({'m', 'r'})

    while True:
        choice = input("  > ").strip()

        if choice in valid_single:
            if choice == 's':
                return ('skip', None)
            elif choice == 'a':
                return ('add', None)
            elif choice == 'm':
                return ('merge', match_indices[0])
            elif choice == 'r':
                return ('replace', match_indices[0])
            elif choice == 'S':
                return ('skip_all', None)
            elif choice == 'M':
                return ('merge_all', None)
            elif choice == 'A':
                return ('add_all', None)

        # Handle numbered merge/replace (m1, m2, r1, r2, etc.)
        num_match = re.match(r'^([mr])(\d+)$', choice)
        if num_match and n_matches > 1:
            action_char = num_match.group(1)
            num = int(num_match.group(2))
            if 1 <= num <= n_matches:
                target_idx = match_indices[num - 1]
                action = 'merge' if action_char == 'm' else 'replace'
                return (action, target_idx)

        # Invalid input
        if n_matches == 1:
            print("  Invalid choice. Enter s, a, m, r, S, M, or A.")
        else:
            print(f"  Invalid choice. Enter s, a, m1-m{n_matches}, r1-r{n_matches}, S, M, or A.")

def ingest_data(input_csv, skip_api_calls=False, discipline=None, initiative_tag=None, workers=2, no_gui=False, skip_duplication_check=False):
    """Main ingestion function"""
    print(f"\n{'='*60}")
    print(f"REPLICATIONS DATABASE INGESTION ENGINE")
    print(f"{'='*60}")
    if skip_api_calls:
        print("  [Skipping API calls - metadata enrichment disabled]")
    else:
        print(f"  [Parallel enrichment with {workers} worker(s)]")
    print(f"{'='*60}")

    # Load input data (support both CSV and JSON)
    print(f"\nLoading input file: {input_csv}")

    if input_csv.lower().endswith('.json'):
        # Load JSON file
        with open(input_csv, 'r') as f:
            data = json.load(f)

        # Extract replications list
        if 'replications' in data and isinstance(data['replications'], list):
            input_df = pd.DataFrame(data['replications'])
            print(f"  Loaded {len(input_df)} replication rows from JSON")
        else:
            raise ValueError("JSON file must contain a 'replications' array")
    else:
        # Load CSV file
        input_df = pd.read_csv(input_csv)
        print(f"  Loaded {len(input_df)} rows from CSV")

    # Skip rows where contains_replications is NO or False
    if 'contains_replications' in input_df.columns:
        before_count = len(input_df)
        input_df = input_df[~input_df['contains_replications'].apply(
            lambda x: (isinstance(x, str) and x.strip().lower() in ('no', 'false'))
                      or (isinstance(x, bool) and not x)
        )]
        skipped = before_count - len(input_df)
        if skipped:
            print(f"  Skipped {skipped} rows where contains_replications is NO/False")
        input_df = input_df.reset_index(drop=True)

    # Skip rows where original_url is literal "na" (not a real URL)
    if 'original_url' in input_df.columns:
        before_count = len(input_df)
        input_df = input_df[~input_df['original_url'].apply(
            lambda x: isinstance(x, str) and x.strip().lower() == 'na'
        )]
        skipped = before_count - len(input_df)
        if skipped:
            print(f"  Skipped {skipped} rows where original_url is 'na'")
        input_df = input_df.reset_index(drop=True)

    # Hard requirement: all rows must have a non-empty description
    if 'description' not in input_df.columns:
        print("\n\033[91m❌ ERROR: Input file is missing the 'description' column.")
        print("  'description' is required for all rows. Add descriptions and retry.\033[0m")
        return
    missing_desc = [i for i, v in input_df['description'].items() if is_empty(v)]
    if missing_desc:
        print(f"\n\033[91m❌ ERROR: {len(missing_desc)} row(s) are missing a 'description':")
        for i in missing_desc[:10]:
            orig_url = _normalize_val(input_df.loc[i].get('original_url', ''))
            print(f"  Row {i + 1}: original_url = {orig_url or '(empty)'}")
        if len(missing_desc) > 10:
            print(f"  ... and {len(missing_desc) - 10} more")
        print("  All rows must have a 'description'. Add descriptions and retry.\033[0m")
        return

    # Clean Unicode artifacts (zero-width spaces, trailing periods) from URL columns
    def clean_doi_url(url):
        if not isinstance(url, str):
            return url
        for char in ['\u200b', '\u200c', '\u200d', '\uFEFF']:
            url = url.replace(char, '')
        url = url.rstrip('.')
        return url

    # Convert bare DOIs and various URL forms to canonical https://doi.org/ URLs.
    def normalize_url_to_doi_url(url):
        if not isinstance(url, str) or not url.strip():
            return url
        url = url.strip()
        # Add missing scheme (e.g. "doi.org/10.xxx")
        if url.startswith("doi.org/"):
            url = "https://" + url
        # Upgrade http to https
        if url.startswith("http://"):
            url = "https://" + url[7:]
        # Normalize dx.doi.org → doi.org
        if url.startswith("https://dx.doi.org/"):
            url = "https://doi.org/" + url[len("https://dx.doi.org/"):]
        if url.startswith("https://"):
            return url
        # Convert bare DOIs (starting with "10.") to full URL
        if url.startswith("10."):
            return f"https://doi.org/{url}"
        return url

    for col in ['original_url', 'replication_url']:
        if col in input_df.columns:
            input_df[col] = input_df[col].apply(clean_doi_url)
            input_df[col] = input_df[col].apply(normalize_url_to_doi_url)
            print(f"  Cleaned and normalized {col} (bare DOIs → https://doi.org/ URLs, http → https)")

    # Apply discipline to all rows if specified
    if discipline:
        input_df['discipline'] = discipline.lower()
        print(f"  Applied discipline '{discipline.lower()}' to all rows")

    # Apply initiative tag to all rows if specified
    if initiative_tag:
        input_df['replication_initiative_tag'] = initiative_tag
        print(f"  Applied initiative tag '{initiative_tag}' to all rows")

    # Rename "version" column to "ai_version" and validate integer values
    if 'version' in input_df.columns:
        print(f"\n  Found 'version' column, renaming to 'ai_version'...")
        input_df = input_df.rename(columns={'version': 'ai_version'})

        # Validate that all values are integers
        non_integer_rows = []
        for idx, val in input_df['ai_version'].items():
            if pd.notna(val):
                try:
                    # Try to convert to int
                    int_val = int(float(val))
                    # Check if conversion changed the value (e.g., 1.5 → 1)
                    if float(val) != float(int_val):
                        non_integer_rows.append((idx, val))
                except (ValueError, TypeError):
                    non_integer_rows.append((idx, val))

        if non_integer_rows:
            print(f"\n  \033[93m{'='*80}")
            print(f"  ⚠️  WARNING: Found {len(non_integer_rows)} non-integer values in 'ai_version' column!")
            print(f"  {'='*80}\033[0m")
            print(f"\n  \033[93mRows with non-integer ai_version values:\033[0m")
            for idx, val in non_integer_rows[:10]:  # Show first 10
                print(f"    Row {idx + 1}: ai_version = {val}")
            if len(non_integer_rows) > 10:
                print(f"    ... and {len(non_integer_rows) - 10} more")
            print(f"\n  \033[93m{'='*80}\033[0m\n")
        else:
            print(f"  ✓ All ai_version values are valid integers")

    # Find latest master database from version_history.txt
    latest_master = get_latest_master_database()
    if not latest_master:
        # Fallback: find newest replications_database_*.csv in data dir
        import glob as _glob
        candidates = sorted(_glob.glob(os.path.join(DATA_DIR, 'replications_database_*.csv')))
        if candidates:
            latest_master = os.path.basename(candidates[-1])
            print(f"\n⚠ version_history.txt missing/empty — falling back to newest file: {latest_master}")
        else:
            print(f"\nNo master database found, will create new one")

    if latest_master:
        master_csv = os.path.join(DATA_DIR, latest_master)
        print(f"\nLoading master database: {master_csv}")
        try:
            master_df = pd.read_csv(master_csv)
            print(f"  Loaded {len(master_df)} existing rows")
        except FileNotFoundError:
            print(f"  Master database not found at {master_csv}, will create new one")
            master_df = pd.DataFrame()
    else:
        master_df = pd.DataFrame()

    if master_df.empty:
        print("\n⚠ WARNING: Master database is empty — all rows will be treated as new!")
        resp = input("  Continue? (y/n): ").strip().lower()
        if resp != 'y':
            print("  Aborted.")
            return

    # Back up the current master database before modifying
    if latest_master:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        backup_path = os.path.join(BACKUP_DIR, latest_master)
        if not os.path.exists(backup_path):
            shutil.copy2(master_csv, backup_path)
            print(f"  ✓ Backed up {latest_master} → data/backup/")
        else:
            print(f"  Backup already exists: data/backup/{latest_master}")

    # Check for checkpoint from a previous interrupted run
    if is_checkpoint_valid(input_csv, len(input_df)):
        print(f"\n  Found valid checkpoint from previous run: {CHECKPOINT_PATH}")
        print(f"  Loading checkpoint (skipping steps 1-4)...")
        processed_df = pd.read_csv(CHECKPOINT_PATH)
        print(f"  ✓ Loaded {len(processed_df)} processed rows from checkpoint")
    elif os.path.exists(CHECKPOINT_PATH):
        print(f"\n  ⚠ Found stale checkpoint (different input file or row count)")
        print(f"  Clearing stale checkpoint and processing from scratch...")
        clear_checkpoint()
        processed_df = None
    else:
        processed_df = None

    if processed_df is None:
        # Process each row (skip API calls if flag is set)
        if skip_api_calls:
            print(f"\n{'='*60}")
            print(f"STEP 1: SKIPPING METADATA ENRICHMENT (--skip-api-calls flag set)")
            print(f"{'='*60}")
            processed_df = input_df.copy()
        else:
            print(f"\n{'='*60}")
            print(f"STEP 1: ENRICHING METADATA")
            print(f"{'='*60}")

            doi_cache, title_cache = load_api_cache()
            cache_lock = threading.Lock()
            total = len(input_df)
            completed_count = 0

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(process_row, row, idx, total, doi_cache, title_cache, cache_lock): idx
                        for idx, row in input_df.iterrows()
                    }
                    processed_rows = [None] * total
                    for future in concurrent.futures.as_completed(futures):
                        idx = futures[future]
                        processed_rows[idx] = future.result()
                        completed_count += 1
                        if completed_count % 50 == 0:
                            save_api_cache(doi_cache, title_cache)
                            logger.info(f"API cache saved ({completed_count}/{total} rows done)")
            except (KeyboardInterrupt, SystemExit):
                logger.warning("Interrupted — saving API cache before exit...")
                save_api_cache(doi_cache, title_cache)
                print(f"\nAPI cache saved to {API_CACHE_PATH} ({len(doi_cache)} DOIs, {len(title_cache)} titles cached)")
                raise
            finally:
                save_api_cache(doi_cache, title_cache)
            processed_df = pd.DataFrame(processed_rows)

        # Calculate effect sizes (convert to r)
        print(f"\n{'='*60}")
        print(f"STEP 2: CALCULATING EFFECT SIZES (converting to r)")
        print(f"{'='*60}")
        processed_df = calculate_effect_sizes(processed_df)

        # Filter columns
        print(f"\n{'='*60}")
        print(f"STEP 3: FILTERING COLUMNS")
        print(f"{'='*60}")
        processed_df = filter_columns(processed_df)

        # Normalize discipline column
        processed_df = normalize_discipline_column(processed_df)

        # Populate field from discipline using ontology
        processed_df = populate_field_from_discipline(processed_df)

        # Normalize journal names
        processed_df = normalize_journal_names(processed_df)

        # Normalize year columns (convert float strings like 2018.0 → 2018)
        processed_df = normalize_year_columns(processed_df)

        # Save checkpoint so restarts skip straight to step 5
        processed_df.to_csv(CHECKPOINT_PATH, index=False)
        save_checkpoint_metadata(input_csv, len(input_df))
        print(f"  ✓ Checkpoint saved — restart will skip to duplicate review")

    # Step 4b: Detect identical title pairs (likely self-paired / within-paper errors)
    print(f"\n{'='*60}")
    print(f"STEP 4b: CHECKING FOR IDENTICAL TITLE PAIRS")
    print(f"{'='*60}")
    identical_title_indices = set()
    for idx, row in processed_df.iterrows():
        orig_t = str(row.get('original_title', '') or '').strip().lower()
        repl_t = str(row.get('replication_title', '') or '').strip().lower()
        if orig_t and repl_t and orig_t == repl_t:
            identical_title_indices.add(idx)
    if identical_title_indices:
        print(f"  ⚠ {len(identical_title_indices)} row(s) with identical original/replication titles:")
        for idx in sorted(identical_title_indices):
            row = processed_df.loc[idx]
            print(f"    Row {idx + 1}: '{str(row.get('replication_title', ''))[:80]}'")
    else:
        print(f"  ✓ No identical title pairs found")

    # Check for duplicates and append
    print(f"\n{'='*60}")
    print(f"STEP 5: CHECKING DUPLICATES AND APPENDING")
    print(f"{'='*60}")
    print(f"  Master: {len(master_df)} rows | Incoming: {len(processed_df)} rows")

    if skip_duplication_check:
        # Skip all duplication checking - add all rows directly
        print(f"\n  ⚠ Skipping duplication check (--skip-duplication-check flag set)")
        print(f"  Adding all {len(processed_df)} rows directly to master")
        rows_to_append = [processed_df.loc[i] for i in processed_df.index]
        duplicates_found = 0
        replaced_count = 0
        merged_count = 0
        identical_count = 0
        total_dups = 0
    else:
        # Pre-scan: classify each row as new, auto-duplicate, or potential duplicate
        auto_skipped = []   # list of (processed_df idx, match_indices)
        potential_dups = []  # list of (processed_df idx, match_indices)
        new_row_indices = []

        for idx, row in processed_df.iterrows():
            if idx in identical_title_indices:
                continue  # handled separately in identical-titles GUI tab
            match_indices = find_duplicate_matches(row, master_df)
            if not match_indices:
                new_row_indices.append(idx)
                continue
            # Check if any existing match is an auto-duplicate (6 key fields match)
            if any(is_auto_duplicate(row, master_df.loc[mi]) for mi in match_indices):
                auto_skipped.append((idx, match_indices))
            else:
                potential_dups.append((idx, match_indices))

        identical_count = len(auto_skipped)
        total_dups = len(potential_dups)
        print(f"\n  {len(new_row_indices)} new rows (no match in master)")
        print(f"  {len(identical_title_indices)} identical-title rows (flagged for review)")
        print(f"  {identical_count} auto-duplicate rows (will be skipped)")
        print(f"  {total_dups} potential duplicate(s) to review")

        # Add all genuinely new rows
        rows_to_append = [processed_df.loc[i] for i in new_row_indices]
        duplicates_found = 0
        replaced_count = 0
        merged_count = 0

    if not skip_duplication_check and (potential_dups or auto_skipped or identical_title_indices) and not no_gui:
        # ── GUI duplicate review ──
        from duplicate_review_gui import launch_duplicate_review
        print(f"\n  Launching GUI for duplicate review...")
        gui_results = launch_duplicate_review(
            potential_dups, auto_skipped, processed_df, master_df,
            identical_title_list=sorted(identical_title_indices)
        )

        if gui_results:
            # Handle auto-skip overrides
            override_set = set(gui_results.get('auto_skip_overrides', []))
            for idx, match_indices in auto_skipped:
                if idx in override_set:
                    rows_to_append.append(processed_df.loc[idx])
                    print(f"  Override: importing auto-skipped row {idx + 1}")

            # Handle potential duplicate decisions
            for decision in gui_results.get('potential_dups', []):
                idx = decision['incoming_idx']
                action = decision['action']
                target = decision.get('target_master_idx')
                force_replace = decision.get('force_replace_fields', set())
                row = processed_df.loc[idx]

                if action == 'add':
                    rows_to_append.append(row)
                elif action == 'replace' and target is not None:
                    master_df = master_df.drop([target])
                    rows_to_append.append(row)
                    replaced_count += 1
                elif action == 'merge' and target is not None:
                    filled = merge_into_master(master_df, target, row, force_replace)
                    merged_count += 1
                    print(f"  ✓ Merged row {idx + 1} into master row {target} ({filled} fields filled)")
                else:
                    duplicates_found += 1

            # Handle identical-title decisions
            reviewed_identical = set()
            for decision in gui_results.get('identical_titles', []):
                idx = decision['incoming_idx']
                reviewed_identical.add(idx)
                action = decision['action']
                if action == 'skip':
                    duplicates_found += 1
                    print(f"  Skipped identical-title row {idx + 1}")
                elif action == 'correct':
                    if decision.get('corrected_original_title'):
                        processed_df.at[idx, 'original_title'] = decision['corrected_original_title']
                    if decision.get('corrected_original_url'):
                        processed_df.at[idx, 'original_url'] = decision['corrected_original_url']
                    rows_to_append.append(processed_df.loc[idx])
                    print(f"  Corrected and added identical-title row {idx + 1}")
                else:  # 'add'
                    rows_to_append.append(processed_df.loc[idx])
                    print(f"  Added identical-title row {idx + 1} as-is")
            # Unreviewed identical-title rows default to add as-is
            for idx in sorted(identical_title_indices):
                if idx not in reviewed_identical:
                    rows_to_append.append(processed_df.loc[idx])
        else:
            # User closed window without applying — skip all potential dups,
            # but add identical-title rows as-is so data is not silently lost
            print("  GUI closed without applying — skipping all potential duplicates")
            duplicates_found = total_dups
            for idx in sorted(identical_title_indices):
                rows_to_append.append(processed_df.loc[idx])

    elif not skip_duplication_check and potential_dups:
        # ── CLI fallback (--no-gui) ──
        batch_decision = None

        for dup_counter, (idx, match_indices) in enumerate(potential_dups, 1):
            row = processed_df.loc[idx]

            if batch_decision == 'add_all':
                rows_to_append.append(row)
                continue
            elif batch_decision == 'skip_all':
                duplicates_found += 1
                continue
            elif batch_decision == 'merge_all':
                filled = merge_into_master(master_df, match_indices[0], row)
                merged_count += 1
                if filled:
                    print(f"  Merged row {idx + 1} → master row {match_indices[0]} ({filled} fields filled)")
                continue

            action, target_idx = prompt_duplicate_action(
                row, master_df, match_indices, idx, len(processed_df),
                dup_number=dup_counter, total_dups=total_dups)

            if action == 'add':
                rows_to_append.append(row)
            elif action == 'add_all':
                rows_to_append.append(row)
                batch_decision = 'add_all'
            elif action == 'replace':
                master_df = master_df.drop([target_idx])
                rows_to_append.append(row)
                replaced_count += 1
            elif action == 'merge':
                filled = merge_into_master(master_df, target_idx, row)
                merged_count += 1
                print(f"  ✓ Merged into master row {target_idx} ({filled} fields filled)")
            elif action == 'merge_all':
                filled = merge_into_master(master_df, match_indices[0], row)
                merged_count += 1
                print(f"  ✓ Merged into master row {match_indices[0]} ({filled} fields filled)")
                batch_decision = 'merge_all'
            elif action == 'skip':
                duplicates_found += 1
            elif action == 'skip_all':
                duplicates_found += 1
                batch_decision = 'skip_all'

    # CLI path for identical-title rows (--no-gui or no GUI available)
    if not skip_duplication_check and identical_title_indices and (no_gui or not (potential_dups or auto_skipped)):
        print(f"\n{'='*60}")
        print(f"IDENTICAL TITLE PAIRS — {len(identical_title_indices)} row(s) to review")
        print(f"{'='*60}")
        for idx in sorted(identical_title_indices):
            row = processed_df.loc[idx]
            title = str(row.get('replication_title', ''))[:80]
            print(f"\n  Row {idx + 1}: '{title}'")
            print(f"    original_url:    {row.get('original_url', '')}")
            print(f"    replication_url: {row.get('replication_url', '')}")
            while True:
                choice = input("  [s]kip / [a]dd as-is / [c]orrect: ").strip().lower()
                if choice == 's':
                    duplicates_found += 1
                    print(f"  Skipped row {idx + 1}")
                    break
                elif choice == 'a':
                    rows_to_append.append(row)
                    break
                elif choice == 'c':
                    new_title = input("  New original_title (Enter to keep current): ").strip()
                    new_url = input("  New original_url (Enter to keep current): ").strip()
                    if new_title:
                        processed_df.at[idx, 'original_title'] = new_title
                    if new_url:
                        processed_df.at[idx, 'original_url'] = new_url
                    rows_to_append.append(processed_df.loc[idx])
                    print(f"  Corrected and added row {idx + 1}")
                    break
                else:
                    print("  Invalid choice. Enter s, a, or c.")
    elif not skip_duplication_check and identical_title_indices and not no_gui:
        pass  # handled in GUI block above

    print(f"\n  Auto-duplicates skipped: {identical_count}")
    print(f"  Duplicates skipped: {duplicates_found}")
    print(f"  Existing rows merged: {merged_count}")
    print(f"  Existing rows replaced: {replaced_count}")
    print(f"  New rows to add: {len(rows_to_append)}")

    # Append new rows to master
    if rows_to_append:
        new_rows_df = pd.DataFrame(rows_to_append)
        # Set validated to "no" for any rows where it's empty
        if 'validated' in new_rows_df.columns:
            new_rows_df['validated'] = new_rows_df['validated'].apply(
                lambda x: 'no' if pd.isna(x) or x == '' or (isinstance(x, str) and not x.strip()) else x
            )
        else:
            new_rows_df['validated'] = 'no'
        updated_master_df = pd.concat([master_df, new_rows_df], ignore_index=True)
    else:
        updated_master_df = master_df

    # Calculate effect sizes for ALL rows (including existing ones that may be missing conversions)
    print(f"\n{'='*60}")
    print(f"STEP 5b: CALCULATING EFFECT SIZES FOR ALL ROWS")
    print(f"{'='*60}")
    updated_master_df = calculate_effect_sizes(updated_master_df)

    # Normalize journal names for ALL rows (including existing ones)
    print(f"\n{'='*60}")
    print(f"STEP 5c: NORMALIZING JOURNAL NAMES FOR ALL ROWS")
    print(f"{'='*60}")
    updated_master_df = normalize_journal_names(updated_master_df)

    # Reorder columns according to data_dictionary.csv
    updated_master_df = reorder_columns(updated_master_df)

    # Normalize year columns (convert float strings like 2018.0 → 2018)
    updated_master_df = normalize_year_columns(updated_master_df)

    # Save with timestamp
    print(f"\n{'='*60}")
    print(f"STEP 6: SAVING UPDATED DATABASE")
    print(f"{'='*60}")

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    output_filename = f"replications_database_{timestamp}.csv"
    output_path = os.path.join(DATA_DIR, output_filename)
    updated_master_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved updated database to: {output_path}")
    print(f"  Total rows in database: {len(updated_master_df)}")

    # Update version history
    print(f"\nUpdating {VERSION_HISTORY_PATH}...")
    input_basename = os.path.basename(input_csv)
    if os.path.exists(VERSION_HISTORY_PATH):
        with open(VERSION_HISTORY_PATH, 'r') as f:
            content = f.read()
        with open(VERSION_HISTORY_PATH, 'w') as f:
            f.write(content.rstrip() + '\n' + output_filename + f' # added {input_basename}' + '\n')
    else:
        with open(VERSION_HISTORY_PATH, 'w') as f:
            f.write(output_filename + f' # added {input_basename}' + '\n')
    print(f"✓ Added {output_filename} to version_history.txt")

    # Clear checkpoint files after successful completion
    clear_checkpoint()

    print(f"\n{'='*60}")
    print(f"INGESTION COMPLETE!")
    print(f"{'='*60}")
    print(f"Summary:")
    print(f"  - Input rows: {len(input_df)}")
    print(f"  - Auto-duplicates skipped: {identical_count}")
    print(f"  - Duplicates skipped: {duplicates_found}")
    print(f"  - Existing rows merged: {merged_count}")
    print(f"  - Existing rows replaced: {replaced_count}")
    print(f"  - New rows added: {len(rows_to_append)}")
    print(f"  - Total rows in database: {len(updated_master_df)}")
    print(f"  - Output file: {output_path}")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingestion Engine for Replications Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_ingestor.py cancer_biology_replications_data.csv --discipline "cancer biology"
  python data_ingestor.py --skip-api-calls psych_file_drawer_data_to_ingest.csv
  python data_ingestor.py 10.1073--pnas.2402315121_result_full.json
  python data_ingestor.py --skip-duplication-check large_dataset.csv
        """
    )
    parser.add_argument('input_csv', help='Input CSV or JSON file to ingest (JSON must have "replications" array)')
    parser.add_argument('--skip-api-calls', action='store_true',
                       help='Skip metadata enrichment API calls (faster but no metadata updates)')
    parser.add_argument('--discipline', type=str, default=None,
                       help='Set discipline value for all rows (e.g., "cancer biology")')
    parser.add_argument('--initiative_tag', type=str, default=None,
                       help='Set replication_initiative_tag for all rows (e.g., "SMR", "RRR")')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers for metadata enrichment (default: 4)')
    parser.add_argument('--no-gui', action='store_true',
                       help='Use CLI prompts instead of GUI for duplicate review')
    parser.add_argument('--skip-duplication-check', action='store_true',
                       help='Skip all duplication checking and add all rows directly (for manual review later)')
    parser.add_argument('--fresh', action='store_true',
                       help='Clear checkpoint and re-run steps 1-3 from scratch')

    args = parser.parse_args()

    if args.fresh:
        for f in [CHECKPOINT_PATH, CHECKPOINT_META_PATH, API_CACHE_PATH]:
            if os.path.exists(f):
                os.remove(f)
                print(f"  Cleared {f}")

    ingest_data(args.input_csv, skip_api_calls=args.skip_api_calls,
                discipline=args.discipline, initiative_tag=args.initiative_tag,
                workers=args.workers, no_gui=args.no_gui,
                skip_duplication_check=args.skip_duplication_check)
