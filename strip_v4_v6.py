"""Strip rows with ai_version 4.0 or 6.0 from the master replications database.

Produces a new timestamped CSV in data/ and appends a line to version_history.txt.
Run this before re-ingesting the rerun_v4_v6 collated CSV so that the old rows
are cleanly removed instead of relying on duplicate detection.
"""
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = (SCRIPT_DIR.parent / "data").resolve()
VERSION_HISTORY_PATH = DATA_DIR / "version_history.txt"
VERSIONS_TO_STRIP = {"4.0", "6.0"}


def get_latest_master() -> Path:
    """Walk version_history.txt backwards to find the most recent existing master file."""
    if not VERSION_HISTORY_PATH.exists():
        raise FileNotFoundError(f"version_history.txt not found at {VERSION_HISTORY_PATH}")
    for line in reversed(VERSION_HISTORY_PATH.read_text().splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        filename = line.split("#")[0].strip()
        if filename.startswith("../data/"):
            filename = filename.replace("../data/", "")
        full_path = DATA_DIR / filename
        if full_path.exists():
            return full_path
    raise FileNotFoundError("No existing master CSV referenced in version_history.txt")


def main() -> int:
    master_path = get_latest_master()
    print(f"Loading master: {master_path.name}")
    df = pd.read_csv(master_path)
    before = len(df)
    print(f"  {before} rows")

    # Ensure ai_version is string for comparison (it's stored as object dtype)
    mask = df["ai_version"].astype(str).isin(VERSIONS_TO_STRIP)
    removed = int(mask.sum())
    df_stripped = df[~mask].copy()
    after = len(df_stripped)

    print(f"Removing {removed} rows with ai_version in {sorted(VERSIONS_TO_STRIP)}")
    print(f"  {before} -> {after} rows")

    if removed == 0:
        print("Nothing to strip; exiting without writing.")
        return 1

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    output_name = f"replications_database_{timestamp}.csv"
    output_path = DATA_DIR / output_name
    df_stripped.to_csv(output_path, index=False)
    print(f"\nWrote {output_path}")

    # Append to version_history.txt
    history_line = f"{output_name} # stripped {removed} rows with ai_version in {{4.0, 6.0}}\n"
    with open(VERSION_HISTORY_PATH, "a") as f:
        f.write(history_line)
    print(f"Appended to version_history.txt: {history_line.strip()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
