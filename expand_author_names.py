#!/usr/bin/env python3
"""
Expand abbreviated author names to full names (First Middle Last) using
metadata APIs. Processes both original_authors and replication_authors columns.

Strategy:
1. Extract DOI from original_url / replication_url
2. If DOI available: use fetch_metadata_from_doi to get full author names
3. If no DOI: use fetch_metadata_from_title as fallback
4. Only update if the fetched names are "better" (more complete) than existing

Automatically finds the latest database CSV and saves a new versioned copy.
"""

import pandas as pd
import sys
import time
import glob
import re
import json
from pathlib import Path
from datetime import datetime

from fetch_metadata_from_doi import fetch_metadata_from_doi
from fetch_metadata_from_title import fetch_metadata_from_title, normalize_doi

DATA_DIR = Path(__file__).parent.parent / "data"
VERSION_HISTORY = DATA_DIR / "version_history.txt"
CACHE_FILE = Path(__file__).parent / ".author_cache.json"


def _find_latest_csv():
    """Find the most recent replications_database_*.csv in the data directory."""
    pattern = str(DATA_DIR / "replications_database_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No replications_database_*.csv found in {DATA_DIR}")
    return files[-1]


def _extract_doi(url):
    """Extract DOI from a URL. Returns None if not a DOI URL."""
    if pd.isna(url):
        return None
    url = str(url).strip()
    if url.startswith('10.'):
        return url
    if 'doi.org/' in url:
        doi = url.split('doi.org/')[-1]
        doi = doi.split('#')[0].split('?')[0]
        if not doi.startswith('10.'):
            doi = '10.' + doi.lstrip('0.')
        return doi
    return None


def _is_empty(val):
    """Check if a value is effectively empty."""
    if pd.isna(val):
        return True
    return str(val).strip() in ('', 'NaN', 'nan', 'None')


def _has_abbreviated_names(authors_str):
    """Check if any author name has an abbreviated first or last name (single initial)."""
    if _is_empty(authors_str):
        return False
    names = str(authors_str).split(';')
    for name in names:
        parts = [p.strip('.,').strip() for p in name.strip().split() if p.strip()]
        if len(parts) < 2:
            continue
        first = parts[0]
        last = parts[-1]
        # First name is a single initial
        if len(first) <= 2 and first[0].isupper() and (len(first) == 1 or first[1] == '.'):
            return True
        # Last name is a single initial (rare but happens in "Last, F." format)
        if len(last) <= 2 and last[0].isupper() and (len(last) == 1 or last[1] == '.'):
            return True
    return False


def _is_last_first_format(authors_str):
    """Check if authors string uses 'Last, First' format."""
    if _is_empty(authors_str):
        return False
    first_author = str(authors_str).split(';')[0].strip()
    # "Last, First" has a comma within the name
    return ',' in first_author


def _normalize_to_first_last(authors_str):
    """Convert 'Last, First' format to 'First Last' format."""
    if _is_empty(authors_str):
        return authors_str
    names = str(authors_str).split(';')
    normalized = []
    for name in names:
        name = name.strip()
        if not name:
            continue
        if ',' in name:
            parts = name.split(',', 1)
            last = parts[0].strip()
            first = parts[1].strip() if len(parts) > 1 else ''
            if first and last:
                normalized.append(f"{first} {last}")
            else:
                normalized.append(name)
        else:
            normalized.append(name)
    return '; '.join(normalized)


def _count_full_names(authors_str):
    """Count how many author names have full (non-abbreviated) first names."""
    if _is_empty(authors_str):
        return 0
    names = str(authors_str).split(';')
    full_count = 0
    for name in names:
        parts = [p.strip('.,').strip() for p in name.strip().split() if p.strip()]
        if len(parts) < 2:
            continue
        first = parts[0]
        # Full first name: more than 2 chars (not just an initial)
        if len(first) > 2 or (len(first) == 2 and first[1] != '.'):
            full_count += 1
    return full_count


def _count_authors(authors_str):
    """Count number of authors in a semicolon-separated string."""
    if _is_empty(authors_str):
        return 0
    return len([n for n in str(authors_str).split(';') if n.strip()])


def _is_better_authors(new_authors, old_authors):
    """Check if new author string is better than old one.

    Better means: same or more authors, and more full (non-abbreviated) names.
    """
    if _is_empty(new_authors):
        return False
    if _is_empty(old_authors):
        return True

    new_count = _count_authors(new_authors)
    old_count = _count_authors(old_authors)

    # Don't accept if we lost authors (wrong paper match)
    if new_count < old_count * 0.7:
        return False

    new_full = _count_full_names(new_authors)
    old_full = _count_full_names(old_authors)

    # Better if more full names
    return new_full > old_full


def _load_cache():
    """Load DOI -> authors cache from disk."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_cache(cache):
    """Save DOI -> authors cache to disk."""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except IOError as e:
        print(f"  Warning: could not save cache: {e}")


def expand_author_names(input_csv, output_csv=None):
    """Expand abbreviated author names using DOI/title metadata lookups."""
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")

    cache = _load_cache()
    print(f"Loaded {len(cache)} cached DOI -> authors mappings")

    # Identify rows needing expansion
    needs_original = df.apply(
        lambda r: _has_abbreviated_names(r.get('original_authors'))
                  or _is_last_first_format(r.get('original_authors'))
                  or _is_empty(r.get('original_authors')),
        axis=1
    )
    needs_replication = df.apply(
        lambda r: _has_abbreviated_names(r.get('replication_authors'))
                  or _is_last_first_format(r.get('replication_authors'))
                  or _is_empty(r.get('replication_authors')),
        axis=1
    )

    print(f"\nRows needing original_authors expansion: {needs_original.sum()}")
    print(f"Rows needing replication_authors expansion: {needs_replication.sum()}")

    # Collect unique DOIs to fetch (avoid redundant API calls)
    doi_set = set()
    for idx in df[needs_original].index:
        doi = _extract_doi(df.at[idx, 'original_url'])
        if doi:
            doi_set.add(doi)
    for idx in df[needs_replication].index:
        doi = _extract_doi(df.at[idx, 'replication_url'])
        if doi:
            doi_set.add(doi)

    # Remove DOIs already in cache
    dois_to_fetch = [d for d in doi_set if d not in cache]
    print(f"\nUnique DOIs to fetch: {len(doi_set)} ({len(dois_to_fetch)} not cached)")

    # Fetch metadata for uncached DOIs
    if dois_to_fetch:
        print(f"\nFetching metadata for {len(dois_to_fetch)} DOIs...")
        for i, doi in enumerate(dois_to_fetch):
            print(f"  [{i+1}/{len(dois_to_fetch)}] {doi}", end="", flush=True)
            try:
                meta = fetch_metadata_from_doi(doi)
                if meta and meta.get('authors'):
                    authors = meta['authors']
                    # Normalize to First Last format if needed
                    if _is_last_first_format(authors):
                        authors = _normalize_to_first_last(authors)
                    cache[doi] = authors
                    print(f"  -> {_count_authors(authors)} authors")
                else:
                    cache[doi] = None
                    print(f"  -> no authors found")
            except Exception as e:
                cache[doi] = None
                print(f"  -> error: {e}")
            time.sleep(0.3)

            # Save cache periodically
            if (i + 1) % 50 == 0:
                _save_cache(cache)

        _save_cache(cache)
        print("DOI fetch complete.")

    # Now process rows
    updated_original = 0
    updated_replication = 0
    title_lookups = 0

    # Process original_authors
    print("\n" + "=" * 80)
    print("Expanding original_authors...")
    print("=" * 80)

    for idx in df[needs_original].index:
        old_authors = df.at[idx, 'original_authors']
        new_authors = None

        # Try DOI first
        doi = _extract_doi(df.at[idx, 'original_url'])
        if doi and doi in cache and cache[doi]:
            new_authors = cache[doi]
        elif not doi and not _is_empty(df.at[idx, 'original_title']):
            # Fallback: title search
            title = str(df.at[idx, 'original_title']).strip()
            print(f"  Row {idx+1}: title lookup for: {title}")
            try:
                meta = fetch_metadata_from_title(title)
                if meta and meta.get('authors'):
                    new_authors = meta['authors']
                    if _is_last_first_format(new_authors):
                        new_authors = _normalize_to_first_last(new_authors)
                title_lookups += 1
                time.sleep(0.5)
            except Exception as e:
                print(f"    Error: {e}")

        if new_authors and _is_better_authors(new_authors, old_authors):
            df.at[idx, 'original_authors'] = new_authors
            updated_original += 1
            if updated_original <= 10:
                old_display = str(old_authors)[:80] if not _is_empty(old_authors) else "(empty)"
                new_display = str(new_authors)[:80]
                print(f"  Row {idx+1}: {old_display}")
                print(f"        -> {new_display}")

    # Process replication_authors
    print("\n" + "=" * 80)
    print("Expanding replication_authors...")
    print("=" * 80)

    for idx in df[needs_replication].index:
        old_authors = df.at[idx, 'replication_authors']
        new_authors = None

        # Try DOI first
        doi = _extract_doi(df.at[idx, 'replication_url'])
        if doi and doi in cache and cache[doi]:
            new_authors = cache[doi]
        elif not doi and not _is_empty(df.at[idx, 'replication_title']):
            # Fallback: title search
            title = str(df.at[idx, 'replication_title']).strip()
            print(f"  Row {idx+1}: title lookup for: {title}")
            try:
                meta = fetch_metadata_from_title(title)
                if meta and meta.get('authors'):
                    new_authors = meta['authors']
                    if _is_last_first_format(new_authors):
                        new_authors = _normalize_to_first_last(new_authors)
                title_lookups += 1
                time.sleep(0.5)
            except Exception as e:
                print(f"    Error: {e}")

        if new_authors and _is_better_authors(new_authors, old_authors):
            df.at[idx, 'replication_authors'] = new_authors
            updated_replication += 1
            if updated_replication <= 10:
                old_display = str(old_authors)[:80] if not _is_empty(old_authors) else "(empty)"
                new_display = str(new_authors)[:80]
                print(f"  Row {idx+1}: {old_display}")
                print(f"        -> {new_display}")

    total_updated = updated_original + updated_replication

    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  original_authors updated: {updated_original}")
    print(f"  replication_authors updated: {updated_replication}")
    print(f"  Total updated: {total_updated}")
    print(f"  Title lookups (no DOI): {title_lookups}")
    print("=" * 80)

    if total_updated == 0:
        print("\nNo authors updated, skipping save.")
        return df

    # Save output
    if output_csv is None:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        output_csv = str(DATA_DIR / f"replications_database_{timestamp}.csv")

    df.to_csv(output_csv, index=False)
    print(f"\nSaved to: {output_csv}")

    # Update version history
    output_name = Path(output_csv).name
    comment = f"# expanded author names ({updated_original} original, {updated_replication} replication)"
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

    expand_author_names(input_file)
