"""No lookahead upstream of the rating: DefR and RAPM for a season never depend on later seasons.

Same check as tests/test_rating_no_lookahead.py, one step earlier in the chain:
models/defr/compute_defr.py and models/ratings/rapm_v4.py are run on all
seasons and with every season after 2023 cut off (ACP_MAX_YEAR), and every
2021-23 value must be identical.

Slow (~8 min) and needs the GPA repo's event files beside the Dashboard, so
it is opt-in: ACP_SLOW_TESTS=1 python -m pytest tests/test_engine_no_lookahead_upstream.py
Run it after changing either script.
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

GPA_EVENTS = DASH.parent.parent / 'GPA Model Project v2' / 'parquet_data' / 'liga3_portugal_events.parquet'
CUTOFF = 2023

pytestmark = [
    pytest.mark.skipif(os.environ.get('ACP_SLOW_TESTS') != '1', reason='slow: set ACP_SLOW_TESTS=1'),
    pytest.mark.skipif(not GPA_EVENTS.exists(), reason='GPA event files not present'),
]


def _run(script, out_env, out_dir, max_year=None):
    env = {**os.environ, out_env: str(out_dir)}
    env.pop('ACP_MAX_YEAR', None)
    if max_year is not None:
        env['ACP_MAX_YEAR'] = str(max_year)
    subprocess.run([sys.executable, str(DASH / script)], cwd=DASH, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _assert_same_early(full, cut, cols):
    early = full[full['seasonId'].map(lambda s: SEASON_YEAR.get(int(s))) <= CUTOFF]
    m = early.merge(cut, on=['playerId', 'seasonId'], how='outer',
                    suffixes=('_full', '_cut'), indicator=True)
    assert (m['_merge'] == 'both').all(), 'later seasons changed which rows exist'
    leaks = {}
    for c in cols:
        a, b = m[f'{c}_full'].astype(float).to_numpy(), m[f'{c}_cut'].astype(float).to_numpy()
        bad = ~(np.isclose(a, b, rtol=0, atol=1e-9) | (np.isnan(a) & np.isnan(b)))
        if bad.any():
            leaks[c] = int(bad.sum())
    assert not leaks, f'seasons <= {CUTOFF} depend on later seasons: {leaks}'


@pytest.mark.parametrize('script,out_env,fname,cols', [
    ('models/defr/compute_defr.py', 'ACP_DEFR_OUT', 'defr_per_player_season.parquet',
     ['expected_def_actions', 'actual_def_actions', 'defr_per90_vs_position', 'defr_adj',
      'defr_dwae', 'defr_career', 'n_seasons']),
    ('models/ratings/rapm_v4.py', 'ACP_RAPM_OUT', 'rapm_v4_per_season.parquet', ['rapm_v4']),
])
def test_upstream_seasons_ignore_later_seasons(tmp_path, script, out_env, fname, cols):
    (tmp_path / 'full').mkdir(); (tmp_path / 'cut').mkdir()
    _run(script, out_env, tmp_path / 'full')
    _run(script, out_env, tmp_path / 'cut', CUTOFF)
    _assert_same_early(pd.read_parquet(tmp_path / 'full' / fname),
                       pd.read_parquet(tmp_path / 'cut' / fname), cols)
