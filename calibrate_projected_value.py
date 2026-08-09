#!/usr/bin/env python3
"""Fee calibration pass for the dashboard's Projected value (engine value).

Compares the engine value (the bio-headline EUR from load_player_engine's
_eng_eur in app.py) against every fee in valuations/reported_fees.csv and
reports fee/value ratios by league. Run it after adding a new fee:

    python calibrate_projected_value.py

How to read the output (loop agreed with Lucas, 2026-07-26):
  * The MEDIAN ratio per league is the calibration signal; individual
    ratios are dominated by selling leverage and contract situation
    (observed range 0.20-5.10) and are NOT evidence by themselves.
  * Jul 2026 baseline: L3 median 1.09 (n=10 real sales — well centred,
    constants left alone), Camp 0.79 after the 0.85 Camp penalty was
    applied to engine values (n=4).
  * Only propose constant changes (PROJECTED_EUR_COEF/EXP, position
    multipliers, CAMP_PROJECTED_EUR_PENALTY, _ENGINE_VALUE_TEMPER) when
    a median drifts materially AND the sample behind it has grown.
    Changes go through Lucas's sign-off before touching app.py.

Implementation note: the CVI/age/curve functions are extracted verbatim
from app.py source at runtime (no hand transcription, so this script can
never drift from the app's math). The _eng_eur wrapper below MUST mirror
the nested _eng_eur inside load_player_engine() in app.py — it cannot be
extracted because it is a closure. If you change one, change both.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CAMP_SEASON_IDS = {190230, 191779, 192925}
ENGINE_VALUE_TEMPER = 0.8  # mirror of _ENGINE_VALUE_TEMPER in app.py
ROLE2CVI = {'Striker': 'ST', 'Wide Attacker': 'AM_WG',
            'Advanced Midfielder': 'AM_WG', 'Deep Midfielder': 'CM',
            'Wide Defender': 'FB', 'Central Defender': 'CB'}

_SRC = (HERE / 'app.py').read_text()


def _grab(name: str, kind: str = 'def') -> str:
    """Extract a top-level def or assignment block from app.py source."""
    if kind == 'def':
        m = re.search(rf"^def {name}\(.*?(?=^\S)", _SRC, re.M | re.S)
    else:
        m = re.search(rf"^{name}\s*=\s*\{{.*?^\}}", _SRC, re.M | re.S)
        if not m:
            m = re.search(rf"^{name}\s*=.*", _SRC, re.M)
    if not m:
        raise RuntimeError(f"couldn't extract {name} from app.py — "
                           f"update calibrate_projected_value.py")
    return m.group(0)


def _load_app_math() -> dict:
    ns = {'pd': pd, 'np': np, 'math': math}
    for blk in [
        _grab('CVI_AGE_VALUE_PARAMS', 'const'),
        _grab('POSITION_EUR_MULTIPLIER', 'const'),
        _grab('CAMP_PROJECTED_EUR_PENALTY', 'const'),
        _grab('PROJECTED_EUR_COEF', 'const'),
        _grab('PROJECTED_EUR_EXP', 'const'),
        _grab('PROJECTED_EUR_CAP', 'const'),
        _grab('_cvi_expected_perf_at'),
        _grab('_cvi_cum_remaining_career'),
        _grab('_CVI_MAX_CAREER_VALUE', 'const'),
        _grab('_cvi_age_value_multiplier'),
        _grab('cvi_to_projected_eur'),
    ]:
        exec(blk, ns)  # noqa: S102 — our own source, extracted verbatim
    return ns


def main() -> None:
    ns = _load_app_math()
    eng = pd.read_parquet(HERE / 'models' / 'ratings' / 'player_engine.parquet')
    eng['playerId'] = pd.to_numeric(eng['playerId'], errors='coerce').astype('Int64')
    eng['seasonId'] = pd.to_numeric(eng['seasonId'], errors='coerce').astype('Int64')
    pool = eng['projection_abs'].dropna()

    def eng_eur(row):
        """Mirror of _eng_eur in app.py load_player_engine()."""
        pa = row.get('projection_abs')
        if pa is None or pd.isna(pa) or len(pool) == 0:
            return None
        perf = float((pool < float(pa)).mean()
                     + 0.5 * (pool == float(pa)).mean()) * 100.0
        grp = ROLE2CVI.get(row.get('role'))
        am = ns['_cvi_age_value_multiplier'](row.get('age'), grp)
        try:
            comp = 702 if int(row.get('seasonId')) in CAMP_SEASON_IDS else 43324
        except (TypeError, ValueError):
            comp = None
        v = ns['cvi_to_projected_eur'](perf * am, position_group=grp,
                                       competition_id=comp)
        return None if v is None else v * ENGINE_VALUE_TEMPER

    fees = pd.read_csv(HERE / 'valuations' / 'reported_fees.csv', comment='#')
    fees['playerId'] = pd.to_numeric(fees['playerId'], errors='coerce')
    rows = []
    for _, f in fees.iterrows():
        er = eng[(eng['playerId'] == int(f['playerId']))
                 & (eng['seasonId'] == int(f['season_id']))]
        ev = eng_eur(er.iloc[0]) if len(er) else None
        rows.append({
            'player': f['player_name'],
            'type': f['transfer_type'],
            'synthetic': int(f.get('synthetic_flag', 0) or 0),
            'season': int(f['season_id']),
            'league': 'Camp' if int(f['season_id']) in CAMP_SEASON_IDS else 'L3',
            'fee': float(f['fee_eur']),
            'engine_value': ev,
            'role': er.iloc[0].get('role') if len(er) else None,
        })
    t = pd.DataFrame(rows)
    t['ratio'] = t['fee'] / t['engine_value']

    print("=== fee vs engine value (dashboard Projected value) ===")
    print(t.sort_values('ratio').to_string(index=False, formatters={
        'fee': '{:,.0f}'.format,
        'engine_value': lambda v: f"{v:,.0f}" if pd.notna(v) else '—',
        'ratio': lambda v: f"{v:.2f}" if pd.notna(v) else '—'}))

    ok = t.dropna(subset=['engine_value'])
    real = ok[(ok['type'] == 'permanent') & (ok['synthetic'] == 0)]
    print(f"\ncoverage: {len(ok)}/{len(t)} fee rows have an engine row "
          f"for the fee season")
    print("\nCalibration medians (real permanent sales only):")
    for label, d in [('L3', real[real['league'] == 'L3']),
                     ('Camp', real[real['league'] == 'Camp'])]:
        if len(d):
            print(f"  {label:5s}: n={len(d):2d}  median fee/value = "
                  f"{d['ratio'].median():.2f}   (Jul 2026 baseline: "
                  f"{'1.09' if label == 'L3' else '0.79'})")
    print("\nInterpretation: medians near 1.0 = curve centred. Only propose "
          "constant changes on material, sample-backed drift (see docstring).")


if __name__ == '__main__':
    main()
