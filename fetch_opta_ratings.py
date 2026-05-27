#!/usr/bin/env python3
"""Scrape current Opta Power Rankings for Portuguese clubs.

The rankings are exposed at https://theanalyst.com/articles/...-opta-power-rankings,
which embeds an iframe widget loaded from dataviz.theanalyst.com. The widget
ships its full dataset (~16k clubs globally) inlined inside its JS bundle as
a JSON-encoded array. There's no separate REST endpoint to call — we have to
download the bundle and parse it out.

We filter to Portugal-domiciled clubs (~122: Primeira Liga + Segunda Liga +
Liga 3 + Campeonato + women's first division) and write the slim slice to
`opta_ratings.parquet` alongside the other Dashboard data files.

Run manually any time, or let the scheduled-refresh workflow trigger it.

Columns produced (per club):
    contestantId, contestantName, contestantClubName, contestantCode,
    country, confederation,
    domesticLeagueId, domesticLeagueName,
    currentRating, seasonAverageRating,
    highestSeasonRating, lowestSeasonRating, lastWeekRating, lastWeekSeasonAverageRating,
    rank,             # global rank (out of ~14k clubs)
    currentGlobalRank, lastWeekGlobalRank,
    currentConfederationRank, lastWeekConfederationRank,
    fetched_at        # UTC timestamp of this scrape
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests


# Bundle URL. The article page embeds an iframe at this origin; the iframe
# loads index.js which has the data inlined.
BUNDLE_URL = "https://dataviz.theanalyst.com/opta-power-rankings/index.js"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Each team record starts with this signature.
_OPENER = re.compile(
    r'\{"rank":\d+,"contestantId":"[a-z0-9]+","contestantName"'
)


def _extract_records(js: str) -> list[dict]:
    """Walk the JS bundle and pull every embedded team object.
    Uses a brace-aware scanner that respects JSON string boundaries so
    embedded quotes / braces don't break parsing."""
    records: list[dict] = []
    i = 0
    n = len(js)
    while True:
        m = _OPENER.search(js, i)
        if not m:
            break
        start = m.start()
        depth = 0
        in_str = False
        esc = False
        end = start
        for k in range(start, n):
            ch = js[k]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = k
                    break
        try:
            records.append(json.loads(js[start:end + 1]))
        except json.JSONDecodeError:
            # Skip malformed (shouldn't happen for Opta's well-formed bundle).
            pass
        i = end + 1
    return records


# Columns we keep. Everything else (image URLs, week-ago timestamps, etc.)
# is dropped to keep the parquet small.
_KEEP_COLS = [
    "contestantId", "contestantName", "contestantClubName",
    "contestantShortName", "contestantCode",
    "country", "countryId", "confederation",
    "domesticLeagueId", "domesticLeagueName",
    "currentRating", "seasonAverageRating",
    "highestSeasonRating", "lowestSeasonRating",
    "lastWeekRating", "lastWeekSeasonAverageRating",
    "rank", "currentGlobalRank", "lastWeekGlobalRank",
    "currentConfederationRank", "lastWeekConfederationRank",
]


def fetch_opta_ratings(countries: list[str] | None = None) -> pd.DataFrame:
    """Download the bundle, extract every team, optionally filter by country."""
    print(f"Downloading bundle: {BUNDLE_URL}")
    r = requests.get(BUNDLE_URL, timeout=60, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    # The bundle is UTF-8 but requests.text often guesses Latin-1 when the
    # server doesn't send an explicit charset; that re-encoding mangles
    # accented club names (Atlético, Académica, etc.). Decode raw bytes.
    js = r.content.decode("utf-8")
    print(f"  bundle size: {len(js)/1024/1024:.1f} MB")

    records = _extract_records(js)
    print(f"  extracted records: {len(records):,}")

    df = pd.DataFrame(records)
    if countries:
        df = df[df["country"].isin(countries)].copy()
        print(f"  filtered to countries {countries}: {len(df)} rows")

    # Keep only the columns we care about; tolerate any that are missing.
    keep = [c for c in _KEEP_COLS if c in df.columns]
    df = df[keep].copy()
    # numericize the ranks/ratings (Opta sometimes ships them as strings)
    for col in ("rank", "currentGlobalRank", "lastWeekGlobalRank",
                "currentConfederationRank", "lastWeekConfederationRank"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ("currentRating", "seasonAverageRating",
                "highestSeasonRating", "lowestSeasonRating",
                "lastWeekRating", "lastWeekSeasonAverageRating"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["fetched_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    return df.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--output", type=Path, default=here / "opta_ratings.parquet",
        help="Where to write the parquet (default: Dashboard/opta_ratings.parquet)",
    )
    parser.add_argument(
        "--countries", default="Portugal",
        help='Comma-separated country names to keep (default "Portugal"). '
             'Use "" to keep all 14k+ clubs.',
    )
    args = parser.parse_args(argv)

    countries = [c.strip() for c in args.countries.split(",") if c.strip()] or None
    df = fetch_opta_ratings(countries)
    df.to_parquet(args.output, index=False, compression="zstd")

    # Tiny summary print so the cron output is informative.
    print(f"\nWrote {len(df)} rows to {args.output}")
    if "domesticLeagueName" in df.columns and not df.empty:
        per_league = (
            df.groupby("domesticLeagueName")
              .agg(n=("contestantId", "count"),
                   avg_rating=("currentRating", "mean"))
              .round(2)
              .sort_values("avg_rating", ascending=False)
        )
        print("\nBy domestic league:")
        print(per_league.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
