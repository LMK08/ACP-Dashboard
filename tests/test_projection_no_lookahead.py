"""No lookahead in the AS-OF projections: a season's as-of projection (the
basis of its season value and of every fee pairing) never depends on later
seasons (2026-10-29).

Builds build_rating -> build_projection -> league_conversion into temp dirs
on all seasons and with every input cut after CUTOFF (ACP_MAX_YEAR), and
requires every as-of number of every season up to the cutoff to be
identical: the projection fit per year (asof.fit_by_year), the career
blocks on one scale, the career role and role shares, the evidence and the
Campeonato recruit gap of each year.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DASH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DASH))
from league_config import SEASON_YEAR  # noqa: E402

CHAIN = ['models/ratings/build_rating.py', 'models/ratings/league_conversion.py',
         'models/ratings/build_projection.py']
CUTOFFS = (2023, 2024)
COLS = ['projection_asof', 'projection_asof_abs', 'career_asof', 'evidence', 'w_evidence',
        'age', 'age_delta', 'band_sd', 'value_role']
GPA_EVENTS = DASH.parent.parent / 'GPA Model Project v2' / 'parquet_data' / 'liga3_portugal_events.parquet'

pytestmark = [
    pytest.mark.skipif(not (DASH / 'models' / 'duels' / 'contests.parquet').exists(),
                       reason='engine inputs not present'),
    pytest.mark.skipif(not GPA_EVENTS.exists(), reason='GPA event files not present'),
]


def _chain(out_dir, max_year=None):
    env = {**os.environ, 'ACP_ENGINE_OUT': str(out_dir)}
    env.pop('ACP_MAX_YEAR', None)
    if max_year is not None:
        env['ACP_MAX_YEAR'] = str(max_year)
    for script in CHAIN:
        subprocess.run([sys.executable, str(DASH / script)], cwd=DASH, env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {f: pd.read_parquet(Path(out_dir) / f'acp_projection_{f}.parquet') for f in ('asof', 'pools', 'sales')}


@pytest.fixture(scope='module')
def builds(tmp_path_factory):
    root = tmp_path_factory.mktemp('asof_chain')
    out = {}
    for c in (None,) + CUTOFFS:
        d = root / ('full' if c is None else f'cut{c}')
        d.mkdir()
        out['full' if c is None else c] = _chain(d, c)
    return out


@pytest.mark.parametrize('cutoff', CUTOFFS)
def test_asof_projections_up_to_cutoff_ignore_later_seasons(builds, cutoff):
    full, cut = builds['full']['asof'], builds[cutoff]['asof']
    early = full[full['seasonId'].map(lambda s: SEASON_YEAR.get(int(s))) <= cutoff]
    m = early.merge(cut, on=['playerId', 'seasonId'], how='outer', suffixes=('_full', '_cut'), indicator=True)
    assert (m['_merge'] == 'both').all(), 'later seasons changed which rows exist'
    leaks = {}
    for c in COLS:
        if m[f'{c}_full'].dtype == object:
            bad = (m[f'{c}_full'].fillna('') != m[f'{c}_cut'].fillna('')).to_numpy()
        else:
            a, b = m[f'{c}_full'].astype(float).to_numpy(), m[f'{c}_cut'].astype(float).to_numpy()
            bad = ~(np.isclose(a, b, rtol=0, atol=1e-9) | (np.isnan(a) & np.isnan(b)))
        if bad.any():
            leaks[c] = int(bad.sum())
    assert not leaks, f'as-of projections <= {cutoff} depend on later seasons: {leaks}'


@pytest.mark.parametrize('cutoff', CUTOFFS)
def test_pools_and_sale_values_up_to_cutoff_ignore_later_seasons(builds, cutoff):
    """The pools a past value ranks in, and the sale-date projections the fee
    pairing reads, are identical with every later season cut off."""
    key = ['yr', 'playerId', 'seasonId']
    fp, cp = builds['full']['pools'], builds[cutoff]['pools']
    a = fp[fp['yr'] <= cutoff].sort_values(key).reset_index(drop=True)
    b = cp.sort_values(key).reset_index(drop=True)
    assert len(a) == len(b) and (a[key].to_numpy() == b[key].to_numpy()).all()
    assert np.allclose(a['projection_abs'], b['projection_abs'], rtol=0, atol=1e-9, equal_nan=True)
    fs, cs = builds['full']['sales'], builds[cutoff]['sales']
    early = fs[fs['season_id'].map(lambda s: SEASON_YEAR.get(int(s))) <= cutoff]
    m = early.merge(cs, on=['playerId', 'as_of_date'], how='outer', suffixes=('_f', '_c'), indicator=True)
    assert (m['_merge'] == 'both').all()
    for c in ('projection_sale_abs', 'age', 'w_evidence', 'evidence', 'base_seasonId'):
        assert np.allclose(m[f'{c}_f'].astype(float), m[f'{c}_c'].astype(float), rtol=0, atol=1e-9, equal_nan=True), c
    assert (m['value_role_f'] == m['value_role_c']).all()
