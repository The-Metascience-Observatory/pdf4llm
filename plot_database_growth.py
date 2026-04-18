"""
Plot the growth of the replications database over time.
Uses all backup CSVs in data/backup/ plus the current CSV in data/.
"""

import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKUP_DIR = REPO_ROOT / "data" / "backup"
DATA_DIR = REPO_ROOT / "data"
PATTERN = re.compile(r"replications_database_(\d{4}_\d{2}_\d{2}_\d{6})\.csv$")


def parse_timestamp(filename: str) -> datetime | None:
    m = PATTERN.search(filename)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y_%m_%d_%H%M%S")


def collect_snapshots() -> list[tuple[datetime, int]]:
    snapshots: list[tuple[datetime, int]] = []

    for folder in (BACKUP_DIR, DATA_DIR):
        for f in folder.glob("replications_database_*.csv"):
            ts = parse_timestamp(f.name)
            if ts is None:
                continue
            try:
                df = pd.read_csv(f, low_memory=False)
                snapshots.append((ts, len(df)))
            except Exception as e:
                print(f"Skipping {f.name}: {e}")

    # Group by date, take max row count per day
    by_day: dict[datetime, int] = {}
    for ts, count in snapshots:
        day = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        by_day[day] = max(by_day.get(day, 0), count)

    return sorted(by_day.items())


def _draw_chart(ax: plt.Axes, dates: list, counts: list, title_fontsize: int = 14) -> None:
    ax.plot(dates, counts, marker="o", markersize=4, linewidth=1.5, color="#2563eb")
    ax.fill_between(dates, counts, alpha=0.08, color="#2563eb")
    ax.annotate(
        f"{counts[0]:,}",
        xy=(dates[0], counts[0]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
        color="#374151",
    )
    ax.annotate(
        f"{counts[-1]:,}",
        xy=(dates[-1], counts[-1]),
        xytext=(-8, 8),
        textcoords="offset points",
        fontsize=9,
        color="#374151",
        ha="right",
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.set_title("Replications Database: Row Count Over Time", fontsize=title_fontsize, pad=12)
    ax.set_ylabel("Number of rows")
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot(snapshots: list[tuple[datetime, int]]) -> None:
    dates = [s[0] for s in snapshots]
    counts = [s[1] for s in snapshots]

    # ── Wide figure (standard) ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    _draw_chart(ax, dates, counts)
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    out = DATA_DIR / "database_growth.png"
    fig.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.close(fig)

    # ── Square figure (social media) ─────────────────────────────────
    fig_sq, ax_sq = plt.subplots(figsize=(8, 8))
    _draw_chart(ax_sq, dates, counts, title_fontsize=13)
    fig_sq.autofmt_xdate(rotation=45, ha="right")
    fig_sq.tight_layout()
    out_sq = DATA_DIR / "database_growth_social.png"
    fig_sq.savefig(out_sq, dpi=150)
    print(f"Saved to {out_sq}")
    plt.close(fig_sq)

    plt.show()


if __name__ == "__main__":
    snapshots = collect_snapshots()
    print(f"Found {len(snapshots)} snapshots")
    for ts, count in snapshots:
        print(f"  {ts.strftime('%Y-%m-%d %H:%M')}  →  {count:,} rows")
    plot(snapshots)
