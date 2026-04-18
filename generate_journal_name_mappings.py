#!/usr/bin/env python3
"""
Generate journal_name_mappings.json from NLM's authoritative J_Medline.txt.

Downloads the NLM bulk journal list (~35k records) from:
  https://ftp.ncbi.nlm.nih.gov/pubmed/J_Medline.txt

Each record maps MedlineTA abbreviation → full JournalTitle.

Output keys are normalized: lowercase, periods stripped, whitespace collapsed.
This matches the lookup logic in data_ingestor.py's normalize_journal_names().

Hand-curated entries in the existing journal_name_mappings.json are preserved
and take priority over NLM data (they contain domain-specific overrides).

Usage:
    python generate_journal_name_mappings.py                # download fresh + merge
    python generate_journal_name_mappings.py --cached       # use previously downloaded file
    python generate_journal_name_mappings.py --dry-run      # show what would change, don't write
"""

import argparse
import json
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
MAPPINGS_PATH = os.path.join(DATA_DIR, 'journal_name_mappings.json')
CACHE_PATH = os.path.join(SCRIPT_DIR, 'J_Medline.txt')

NLM_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/J_Medline.txt"


def normalize_key(s):
    """Normalize a journal name for lookup: lowercase, strip periods, collapse spaces."""
    s = s.lower().replace('.', '').strip()
    return re.sub(r'\s+', ' ', s)


def download_j_medline(url=NLM_URL, output_path=CACHE_PATH):
    """Download J_Medline.txt from NLM FTP."""
    import requests
    print(f"Downloading {url} ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(resp.text)
    print(f"  Saved to {output_path} ({len(resp.text) / 1024:.0f} KB)")
    return output_path


def parse_j_medline(path):
    """Parse J_Medline.txt into a list of record dicts.

    File format: records separated by lines of dashes, fields as 'Key: Value'.
    """
    records = []
    current = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('---'):
                if current:
                    records.append(current)
                current = {}
            elif ': ' in line:
                key, _, val = line.partition(': ')
                current[key.strip()] = val.strip()
    if current:
        records.append(current)
    return records


def build_abbreviation_dict(records):
    """Build normalized abbreviation → full title dict from NLM records.

    Returns:
        dict: {normalized_abbrev: full_title}
        stats: dict with counts
    """
    abbrevs = {}
    stats = {
        'total_records': len(records),
        'self_mappings_skipped': 0,
        'missing_fields_skipped': 0,
        'duplicates_overwritten': 0,
        'valid_mappings': 0,
    }

    for rec in records:
        title = rec.get('JournalTitle', '').strip()
        med_abbr = rec.get('MedAbbr', '').strip()

        if not title or not med_abbr:
            stats['missing_fields_skipped'] += 1
            continue

        key = normalize_key(med_abbr)
        val_normalized = normalize_key(title)

        # Skip self-mappings (abbreviation equals full name after normalization)
        if key == val_normalized:
            stats['self_mappings_skipped'] += 1
            continue

        if key in abbrevs:
            stats['duplicates_overwritten'] += 1

        abbrevs[key] = title
        stats['valid_mappings'] += 1

    return abbrevs, stats


def load_existing_mappings(path=MAPPINGS_PATH):
    """Load current journal_name_mappings.json."""
    if not os.path.exists(path):
        return {"abbreviations": {}, "variant_forms": {}, "html_entity_fixes": {}}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_mappings(nlm_abbrevs, existing):
    """Merge NLM abbreviations into existing mappings.

    Hand-curated entries take priority over NLM data.

    Returns:
        merged: dict ready to write as JSON
        stats: dict with merge counts
    """
    hand_curated = existing.get('abbreviations', {})
    variant_forms = existing.get('variant_forms', {})
    html_entity_fixes = existing.get('html_entity_fixes', {})

    # Start with NLM data
    merged_abbrevs = dict(nlm_abbrevs)

    # Overlay hand-curated (they win on conflict)
    overridden = 0
    added_by_hand = 0
    for key, val in hand_curated.items():
        if key in merged_abbrevs and merged_abbrevs[key] != val:
            overridden += 1
        elif key not in merged_abbrevs:
            added_by_hand += 1
        merged_abbrevs[key] = val

    stats = {
        'nlm_entries': len(nlm_abbrevs),
        'hand_curated_entries': len(hand_curated),
        'hand_curated_overrides': overridden,
        'hand_curated_additions': added_by_hand,
        'total_abbreviations': len(merged_abbrevs),
        'variant_forms': len(variant_forms),
    }

    merged = {
        "_comment": (
            "Maps variant/abbreviated journal names to their canonical full names. "
            "Used by data_ingestor.py to normalize journal names on ingest. "
            "Keys are normalized: lowercase, periods stripped, single spaces. "
            "Bulk abbreviations sourced from NLM J_Medline.txt (~35k journals). "
            "Hand-curated entries override NLM data where they differ."
        ),
        "abbreviations": dict(sorted(merged_abbrevs.items())),
        "variant_forms": variant_forms,
        "html_entity_fixes": html_entity_fixes,
    }

    return merged, stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate journal_name_mappings.json from NLM J_Medline.txt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--cached', action='store_true',
                        help='Use previously downloaded J_Medline.txt instead of re-downloading')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would change without writing the file')
    args = parser.parse_args()

    # Step 1: Get J_Medline.txt
    if args.cached and os.path.exists(CACHE_PATH):
        print(f"Using cached {CACHE_PATH}")
    else:
        download_j_medline()

    # Step 2: Parse
    print("Parsing J_Medline.txt ...")
    records = parse_j_medline(CACHE_PATH)
    print(f"  Parsed {len(records)} records")

    # Step 3: Build abbreviation dict
    nlm_abbrevs, build_stats = build_abbreviation_dict(records)
    print(f"\nNLM data:")
    print(f"  Total records:         {build_stats['total_records']}")
    print(f"  Self-mappings skipped: {build_stats['self_mappings_skipped']}")
    print(f"  Missing fields:        {build_stats['missing_fields_skipped']}")
    print(f"  Valid mappings:        {build_stats['valid_mappings']}")

    # Step 4: Load existing and merge
    existing = load_existing_mappings()
    merged, merge_stats = merge_mappings(nlm_abbrevs, existing)

    print(f"\nMerge results:")
    print(f"  NLM abbreviations:        {merge_stats['nlm_entries']}")
    print(f"  Hand-curated entries:      {merge_stats['hand_curated_entries']}")
    print(f"  Hand-curated overrides:    {merge_stats['hand_curated_overrides']}")
    print(f"  Hand-curated additions:    {merge_stats['hand_curated_additions']}")
    print(f"  Total abbreviations:       {merge_stats['total_abbreviations']}")
    print(f"  Variant forms (preserved): {merge_stats['variant_forms']}")

    # Step 5: Show hand-curated overrides
    if merge_stats['hand_curated_overrides'] > 0:
        print(f"\nHand-curated overrides (our value wins over NLM):")
        hand = existing.get('abbreviations', {})
        for key in sorted(hand):
            if key in nlm_abbrevs and nlm_abbrevs[key] != hand[key]:
                print(f"  '{key}': NLM='{nlm_abbrevs[key]}' -> ours='{hand[key]}'")

    # Step 6: Write
    if args.dry_run:
        print(f"\n[DRY RUN] Would write {len(merged['abbreviations'])} abbreviations to {MAPPINGS_PATH}")
    else:
        with open(MAPPINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        size_kb = os.path.getsize(MAPPINGS_PATH) / 1024
        print(f"\n  Wrote {MAPPINGS_PATH} ({size_kb:.0f} KB)")

    print("\nDone.")


if __name__ == '__main__':
    main()
