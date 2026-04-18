#!/usr/bin/env python3
"""
Fill missing URLs in the database by searching for DOIs using the metadata enrichment pipeline.
Automatically finds the latest database CSV and saves a new versioned copy.
"""

import pandas as pd
import sys
import time
import glob
from pathlib import Path
from datetime import datetime

from fetch_metadata_from_title import fetch_metadata_from_title

DATA_DIR = Path(__file__).parent.parent / "data"
VERSION_HISTORY = DATA_DIR / "version_history.txt"


def _find_latest_csv():
    """Find the most recent replications_database_*.csv in the data directory."""
    pattern = str(DATA_DIR / "replications_database_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No replications_database_*.csv found in {DATA_DIR}")
    return files[-1]


def _is_empty(val):
    """Check if a value is effectively empty (NaN, None, empty string, 'NaN')."""
    if pd.isna(val):
        return True
    return str(val).strip() in ('', 'NaN', 'nan', 'None')


def _clean(val):
    """Return None if value is empty, otherwise return stripped string."""
    if _is_empty(val):
        return None
    return str(val).strip()


def _extract_url(metadata):
    """Extract the best URL from metadata, preferring DOI > PMID > raw URL.
    Returns (url, label) tuple or (None, None)."""
    if not metadata:
        return None, None
    if metadata.get('doi'):
        url = f"https://doi.org/{metadata['doi']}"
        return url, f"DOI: {url}"
    if metadata.get('pmid'):
        url = f"https://pubmed.ncbi.nlm.nih.gov/{metadata['pmid']}/"
        return url, f"PMID: {url}"
    if metadata.get('url'):
        return metadata['url'], f"URL: {metadata['url']}"
    return None, None


def _process_missing_urls(df, indices, prefix, url_col):
    """Fill missing URLs for a set of rows.

    Args:
        df: DataFrame (modified in place)
        indices: Index of rows to process
        prefix: Column prefix ('original' or 'replication')
        url_col: Name of the URL column to fill

    Returns:
        Number of URLs updated
    """
    updated = 0
    total = len(indices)

    for i, idx in enumerate(indices):
        row = df.loc[idx]
        title = _clean(row[f'{prefix}_title'])
        if not title:
            continue

        print(f"\n  [{i+1}/{total}] Row {idx + 1}: {title}")

        metadata = fetch_metadata_from_title(
            title,
            authors=_clean(row.get(f'{prefix}_authors')),
            journal=_clean(row.get(f'{prefix}_journal')),
            year=_clean(row.get(f'{prefix}_year')),
            volume=_clean(row.get(f'{prefix}_volume')),
        )

        url, label = _extract_url(metadata)
        if url:
            df.at[idx, url_col] = url
            print(f"    Found {label}")
            updated += 1
        else:
            print(f"    Could not find URL")

        time.sleep(0.5)

    return updated


def fill_missing_urls(input_csv, output_csv=None):
    """Fill missing original_url and replication_url by searching with titles."""
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")

    missing_original = df['original_url'].apply(_is_empty) & ~df['original_title'].apply(_is_empty)
    missing_replication = df['replication_url'].apply(_is_empty) & ~df['replication_title'].apply(_is_empty)

    print(f"\nRows missing original_url with title available: {missing_original.sum()}")
    print(f"Rows missing replication_url with title available: {missing_replication.sum()}")

    print("\n" + "="*80)
    print("Processing missing original URLs...")
    print("="*80)
    updated_original = _process_missing_urls(
        df, df[missing_original].index, 'original', 'original_url'
    )

    print("\n" + "="*80)
    print("Processing missing replication URLs...")
    print("="*80)
    updated_replication = _process_missing_urls(
        df, df[missing_replication].index, 'replication', 'replication_url'
    )

    total_updated = updated_original + updated_replication

    print("\n" + "="*80)
    print(f"Summary: Updated {total_updated} URLs ({updated_original} original, {updated_replication} replication)")
    print("="*80)

    if total_updated == 0:
        print("\nNo URLs updated, skipping save.")
        return df

    # Save output
    if output_csv is None:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        output_csv = str(DATA_DIR / f"replications_database_{timestamp}.csv")

    df.to_csv(output_csv, index=False)
    print(f"\nSaved to: {output_csv}")

    # Update version history
    output_name = Path(output_csv).name
    comment = f"# filled {total_updated} missing URLs ({updated_original} original_url, {updated_replication} replication_url)"
    with open(VERSION_HISTORY, "a") as f:
        f.write(f"{output_name} {comment}\n")
    print(f"Updated {VERSION_HISTORY}")

    return df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = _find_latest_csv()
        print(f"Auto-detected latest database: {input_file}")

    fill_missing_urls(input_file)
