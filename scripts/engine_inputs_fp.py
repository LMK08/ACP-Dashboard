"""Fingerprint of everything the engine rebuild reads that the daily data
refresh can change: the HF Space's raw_events.parquet (LFS sha256 — no
download) plus the committed per-match minutes, lineups and schedule.

    python scripts/engine_inputs_fp.py            # print the fingerprint
    python scripts/engine_inputs_fp.py --changed  # 'true'/'false' vs the
                                                  # stored models/ratings/engine_inputs.sha

engine_rebuild.yml skips a scheduled rebuild when nothing changed (the
2026-10 refreshes logged "No new events fetched" five runs in a row and
each still ran a 30-60 min rebuild). Run from the Dashboard dir.
"""
import hashlib
import os
import sys

STORED = os.path.join('models', 'ratings', 'engine_inputs.sha')
FILES = ['match_player_minutes.parquet', 'match_lineups.pkl', 'matches_summary.parquet']


def _content_hash(f):
    """Hash of the CONTENT that matters, not the file bytes: the refresh
    rewrites these files every run (parquet metadata, pickle ordering), so a
    byte hash changed daily and triggered a full rebuild with no new data
    (first live test, 2026-10-21)."""
    import pandas as pd
    if not os.path.exists(f):
        return 'missing'
    if f == 'matches_summary.parquet':
        d = pd.read_parquet(f, columns=['matchId', 'status', 'score'])
        d = d[d['status'] == 'Played'].astype(str).sort_values(['matchId']).to_csv(index=False)
    elif f == 'match_player_minutes.parquet':
        d = pd.read_parquet(f).astype(str).sort_values(['matchId', 'playerId']).to_csv(index=False)
    elif f == 'match_lineups.pkl':
        import pickle
        with open(f, 'rb') as fh:
            lu = pickle.load(fh)
        d = repr(sorted((int(k), bool(v)) for k, v in lu.items()))
    else:
        return hashlib.sha256(open(f, 'rb').read()).hexdigest()
    return hashlib.sha256(d.encode()).hexdigest()


def fingerprint():
    from huggingface_hub import HfApi
    info = HfApi(token=os.environ.get('HF_TOKEN') or None).get_paths_info(
        'ACP-Analytics/ACP-Dashboard', ['raw_events.parquet'], repo_type='space')
    parts = [f"raw_events={info[0].lfs.sha256 if info and info[0].lfs else 'missing'}"]
    for f in FILES:
        parts.append(f"{f}={_content_hash(f)}")
    return '\n'.join(parts) + '\n'


if __name__ == '__main__':
    fp = fingerprint()
    if '--changed' in sys.argv:
        old = open(STORED).read() if os.path.exists(STORED) else ''
        print('false' if old == fp else 'true')
    elif '--write' in sys.argv:
        open(STORED, 'w').write(fp)
        print(f'wrote {STORED}')
    else:
        print(fp, end='')
