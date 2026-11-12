"""Every match in raw_events must have OBV rows in the three per-match engine
files the match report reads. Exits 1 and lists the matches when one is
missing, so the rebuild job fails loudly instead of the report silently
dropping its OBV sections (30 Oct - 8 Nov 2026: 20 matches had none while
the rebuild was failing).

    python scripts/check_obv_coverage.py [dashboard_dir]
"""
import os
import sys

import pandas as pd

OBV_FILES = ('obv_match_player.parquet', 'obv_match_minute.parquet', 'obv_pass_pairs.parquet')


def missing_matches(root):
    """{file: sorted matchIds in raw_events with no row in that file}."""
    raw = pd.read_parquet(os.path.join(root, 'raw_events.parquet'), columns=['matchId'])
    want = set(raw['matchId'].dropna().astype('int64').unique())
    out = {}
    for f in OBV_FILES:
        path = os.path.join(root, f)
        have = set()
        if os.path.exists(path):
            have = set(pd.read_parquet(path, columns=['matchId'])['matchId'].dropna().astype('int64').unique())
        gone = sorted(int(m) for m in want - have)
        if gone:
            out[f] = gone
    return out, len(want)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    missing, n = missing_matches(root)
    if not missing:
        print(f'OBV coverage: all {n} matches in raw_events have OBV rows in {len(OBV_FILES)} files.')
        return 0
    for f, ids in missing.items():
        print(f'OBV coverage: {f} has no rows for {len(ids)} of {n} matches: '
              f'{ids[:30]}{" ..." if len(ids) > 30 else ""}')
    return 1


if __name__ == '__main__':
    sys.exit(main())
