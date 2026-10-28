"""No lookahead in the ACP Rating: a season's numbers never depend on later seasons.

Builds models/ratings/build_rating.py three times into temp dirs — on all
seasons, and with every input truncated after 2023 and after 2024
(ACP_MAX_YEAR) — and requires every exported value of every season up to the
cutoff to be identical. Covers what build_rating.py itself does with its
inputs (percentiles, offence reliability weights, duel expectations,
blending, shrinkage); the upstream input files (GPA values, DefR, roles,
RAPM, contests) are taken as committed.

Excluded by design: acp_rating_career and n_seasons, the whole-career
summary AS OF TODAY that only the profile's all-seasons view shows.
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

BUILD = DASH / 'models' / 'ratings' / 'build_rating.py'
CUTOFFS = (2023, 2024)
WHOLE_CAREER = {'acp_rating_career', 'n_seasons'}

pytestmark = pytest.mark.skipif(
    not (DASH / 'models' / 'duels' / 'contests.parquet').exists(),
    reason='engine inputs not present')


def _build(out_dir, max_year=None):
    env = {**os.environ, 'ACP_ENGINE_OUT': str(out_dir)}
    env.pop('ACP_MAX_YEAR', None)
    if max_year is not None:
        env['ACP_MAX_YEAR'] = str(max_year)
    subprocess.run([sys.executable, str(BUILD)], cwd=DASH, env=env, check=True,
                   stdout=subprocess.DEVNULL)
    return pd.read_parquet(Path(out_dir) / 'acp_rating_per_player_season.parquet')


@pytest.fixture(scope='module')
def builds(tmp_path_factory):
    out = {None: _build(tmp_path_factory.mktemp('full'))}
    for y in CUTOFFS:
        out[y] = _build(tmp_path_factory.mktemp(f'upto{y}'), y)
    return out


@pytest.mark.parametrize('cutoff', CUTOFFS)
def test_seasons_up_to_cutoff_ignore_later_seasons(builds, cutoff):
    full, cut = builds[None], builds[cutoff]
    keys = ['playerId', 'seasonId', 'league']
    early = full[full['seasonId'].map(SEASON_YEAR) <= cutoff]
    assert len(cut) == len(early), 'a later season changed which rows exist'
    m = early.merge(cut, on=keys, suffixes=('_full', '_cut'), validate='1:1')
    assert len(m) == len(early)
    num = [c for c in full.columns if c not in keys and c not in WHOLE_CAREER
           and pd.api.types.is_numeric_dtype(full[c])]
    leaks = {}
    for c in num:
        a, b = m[f'{c}_full'].to_numpy(float), m[f'{c}_cut'].to_numpy(float)
        bad = ~(np.isclose(a, b, rtol=0, atol=1e-9) | (np.isnan(a) & np.isnan(b)))
        if bad.any():
            leaks[c] = (int(bad.sum()), float(np.nanmax(np.abs(a - b))))
    assert not leaks, f'values for seasons <= {cutoff} depend on later seasons: {leaks}'
