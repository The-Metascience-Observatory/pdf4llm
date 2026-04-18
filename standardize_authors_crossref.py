#!/usr/bin/env python3
"""
Standardize author names using CrossRef API.

This script:
1. Reads the replications database
2. For each unique DOI, queries CrossRef API for standardized author names
3. Formats names consistently with periods after initials
4. Updates the original_authors column
5. Saves to a new database file

Usage:
    python standardize_authors_crossref.py
"""

import pandas as pd
import requests
import time
from datetime import datetime
from collections import defaultdict
import re
from tqdm import tqdm

# Configuration
CROSSREF_EMAIL = "dan@metascienceobservatory.org"
CROSSREF_API_BASE = "https://api.crossref.org/works/"
RATE_LIMIT_DELAY = 0.5  # seconds between requests (polite pool)
INPUT_FILE = "data/replications_database_2026_02_15_084408.csv"


def extract_doi(url):
    """Extract DOI from URL."""
    if pd.isna(url):
        return None

    url = str(url)

    # Handle direct DOIs
    if url.startswith('10.'):
        return url

    # Extract from doi.org URL
    if 'doi.org/' in url:
        doi = url.split('doi.org/')[-1]
        # Remove any trailing fragments or query params
        doi = doi.split('#')[0].split('?')[0]

        # Add 10. prefix if missing (some DOIs stored as 0.1111/... instead of 10.1111/...)
        if not doi.startswith('10.'):
            doi = '10.' + doi.lstrip('0.')

        return doi

    return None


def format_author_name(author_dict):
    """
    Format author name consistently.

    Args:
        author_dict: Dict with 'given' and 'family' keys from CrossRef

    Returns:
        Formatted name string with periods after single-letter initials
    """
    given = author_dict.get('given', '').strip()
    family = author_dict.get('family', '').strip()

    if not family:
        return None

    if not given:
        return family

    # Add periods after single letters if not present
    # e.g., "J Lukas" -> "J. Lukas", "J. Lukas" stays "J. Lukas"
    parts = given.split()
    formatted_parts = []

    for part in parts:
        # Check if it's a single letter without a period
        if len(part) == 1 and part.isalpha():
            formatted_parts.append(part + '.')
        # Check if it's already a single letter with period
        elif len(part) == 2 and part[0].isalpha() and part[1] == '.':
            formatted_parts.append(part)
        # Otherwise keep as is
        else:
            formatted_parts.append(part)

    formatted_given = ' '.join(formatted_parts)
    return f"{formatted_given} {family}"


def get_authors_from_crossref(doi):
    """
    Query CrossRef API for author information.

    Args:
        doi: DOI string

    Returns:
        List of formatted author names, or None if error
    """
    if not doi:
        return None

    url = f"{CROSSREF_API_BASE}{doi}"
    headers = {
        'User-Agent': f'MetascienceObservatory/1.0 (mailto:{CROSSREF_EMAIL})'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            authors_data = data['message'].get('author', [])

            if not authors_data:
                return None

            # Format each author
            formatted_authors = []
            for author_dict in authors_data:
                formatted_name = format_author_name(author_dict)
                if formatted_name:
                    formatted_authors.append(formatted_name)

            return formatted_authors if formatted_authors else None

        elif response.status_code == 404:
            return None
        else:
            return None

    except requests.exceptions.Timeout:
        return None
    except Exception as e:
        return None


def get_authors_from_openalex(doi):
    """
    Query OpenAlex API for author information (fallback).

    Args:
        doi: DOI string

    Returns:
        List of author names, or None if error
    """
    if not doi:
        return None

    url = f"https://api.openalex.org/works/doi:{doi}"
    headers = {
        'User-Agent': f'MetascienceObservatory/1.0 (mailto:{CROSSREF_EMAIL})'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            authorships = data.get('authorships', [])

            if not authorships:
                return None

            # Extract display names
            authors = []
            for authorship in authorships:
                author = authorship.get('author', {})
                display_name = author.get('display_name', '').strip()
                if display_name:
                    authors.append(display_name)

            return authors if authors else None

        else:
            return None

    except Exception as e:
        return None


def get_authors_with_fallback(doi):
    """
    Try to get authors from CrossRef, fall back to OpenAlex if not found.

    Args:
        doi: DOI string

    Returns:
        Tuple of (authors_list, source) where source is 'crossref' or 'openalex'
    """
    # Try CrossRef first
    authors = get_authors_from_crossref(doi)
    if authors:
        return authors, 'crossref'

    # Fall back to OpenAlex
    time.sleep(0.2)  # Small delay before OpenAlex query
    authors = get_authors_from_openalex(doi)
    if authors:
        return authors, 'openalex'

    return None, None


def main():
    print("=" * 80)
    print("Author Name Standardization via CrossRef API")
    print("=" * 80)
    print(f"\nStarting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using email: {CROSSREF_EMAIL}")
    print(f"Rate limit delay: {RATE_LIMIT_DELAY}s between requests\n")

    # Read database
    print(f"Reading database: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"Total rows: {len(df)}")

    # Count rows with authors and DOIs
    has_authors = df['original_authors'].notna()
    has_doi = df['original_url'].notna()
    print(f"Rows with original_authors: {has_authors.sum()}")
    print(f"Rows with DOIs: {has_doi.sum()}")

    # Extract DOIs
    print("\nExtracting DOIs...")
    df['doi'] = df['original_url'].apply(extract_doi)
    valid_dois = df['doi'].notna()
    print(f"Valid DOIs extracted: {valid_dois.sum()}")

    # Get unique DOIs to query
    unique_dois = df[valid_dois]['doi'].unique()
    print(f"Unique DOIs to query: {len(unique_dois)}")

    # Create mapping of DOI -> authors from CrossRef
    print(f"\nQuerying CrossRef API...")
    print(f"Estimated time: {len(unique_dois) * RATE_LIMIT_DELAY / 60:.1f} minutes")
    print("-" * 80)

    doi_to_authors = {}
    success_count = 0
    error_count = 0
    crossref_count = 0
    openalex_count = 0

    for doi in tqdm(unique_dois, desc="Querying APIs", unit="DOI"):
        authors, source = get_authors_with_fallback(doi)

        if authors:
            doi_to_authors[doi] = '; '.join(authors)
            success_count += 1

            if source == 'crossref':
                crossref_count += 1
            elif source == 'openalex':
                openalex_count += 1
        else:
            error_count += 1

        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)

    print(f"\nAPI query complete:")
    print(f"  Successful: {success_count}")
    print(f"    - From CrossRef: {crossref_count}")
    print(f"    - From OpenAlex: {openalex_count}")
    print(f"  Failed/Not found: {error_count}")

    # Update database with standardized authors
    print("\nUpdating database with standardized author names...")

    updates_made = 0
    no_crossref_data = 0
    already_correct = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Updating database", unit="row"):
        doi = row['doi']
        current_authors = row['original_authors']

        if pd.isna(doi):
            continue

        if doi in doi_to_authors:
            crossref_authors = doi_to_authors[doi]

            # Only update if different
            if current_authors != crossref_authors:
                df.at[idx, 'original_authors'] = crossref_authors
                updates_made += 1
            else:
                already_correct += 1
        else:
            no_crossref_data += 1

    print(f"  Updates made: {updates_made}")
    print(f"  Already correct: {already_correct}")
    print(f"  No CrossRef data: {no_crossref_data}")

    # Save updated database
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    output_file = f"data/replications_database_{timestamp}.csv"

    # Drop temporary DOI column
    df_output = df.drop(columns=['doi'])

    df_output.to_csv(output_file, index=False)
    print(f"\nSaved updated database to: {output_file}")

    # Update version history
    version_entry = f"replications_database_{timestamp}.csv # standardized author names via CrossRef API ({updates_made} rows updated)"

    with open('data/version_history.txt', 'a') as f:
        f.write(version_entry + '\n')

    print(f"Updated version_history.txt")

    # Generate comparison report
    print("\n" + "=" * 80)
    print("COMPARISON REPORT")
    print("=" * 80)

    # Show some examples of changes
    print("\nExample changes (first 20):")
    print("-" * 80)

    changes_shown = 0
    for idx, row in df.iterrows():
        doi = row['doi']
        if pd.notna(doi) and doi in doi_to_authors:
            old_authors = df_output.at[idx, 'original_authors']

            # Re-read original to compare
            df_orig = pd.read_csv(INPUT_FILE)
            orig_authors = df_orig.at[idx, 'original_authors']

            if orig_authors != old_authors and changes_shown < 20:
                print(f"\nRow {idx}:")
                print(f"  Before: {orig_authors}")
                print(f"  After:  {old_authors}")
                changes_shown += 1

    print(f"\n{'=' * 80}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
