"""models/value/availability — club windows, the one-season cap, the
regular-chance factor and its wiring into the engine value; plus the
transfer-record helpers in fetch_transfers.

Pure-Python tests on a synthetic season; the data-backed check skips without
the committed availability.parquet.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

DASH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DASH)

from league_config import SEASON_YEAR, is_reserve_team  # noqa: E402
from models.value import availability as av  # noqa: E402
from models.value import eur_intervals as ei  # noqa: E402
from models.value import fetch_transfers as ft  # noqa: E402

SID = 191782                    # Liga 3 2025/26 -> year 2025
L = 95.0                        # every synthetic match lasts 95'
T1, T2 = 101, 202               # two clubs
DAYS = pd.date_range('2025-08-10', periods=30, freq='7D')   # the synthetic season's match days


def _season(n=30):
    """n weekly matches per club, Aug -> Apr; both clubs play every week."""
    days = pd.date_range('2025-08-10', periods=n, freq='7D')
    ms, ev, iv, lineups = [], [], [], {}
    for i, d in enumerate(days):
        for k, t in enumerate((T1, T2)):
            mid = 1000 + 2 * i + k
            ms.append({'matchId': mid, 'seasonId': SID, 'dateutc': d.strftime('%Y-%m-%d 15:00:00')})
            ev.append({'matchId': mid, 'seasonId': SID, 'team.id': t, 'player.id': 9000 + t})   # a fixture player
            iv.append({'matchId': mid, 'u_out': L})
            lineups[mid] = {f'team{t}': {'lineup': [{'playerId': 9000 + t}], 'bench': [], 'substitutions': []}}
    return days, ms, ev, iv, lineups


def _inputs(players, transfers=(), n=30):
    """players: {pid: [(team, match_index, flag)]}, flag 'S' start / 'B' came on / 'U' unused."""
    days, ms, ev, iv, lineups = _season(n)
    for pid, apps in players.items():
        for t, i, flag in apps:
            mid = 1000 + 2 * i + (0 if t == T1 else 1)
            side = lineups[mid][f'team{t}']
            if flag == 'S':
                side['lineup'].append({'playerId': pid})
            else:
                side['bench'].append({'playerId': pid})
                if flag == 'B':
                    side['substitutions'].append({'playerIn': pid, 'playerOut': 9000 + t})
            if flag in ('S', 'B'):
                ev.append({'matchId': mid, 'seasonId': SID, 'team.id': t, 'player.id': pid})
    ms = pd.DataFrame(ms)
    ms['d'] = pd.to_datetime(ms['dateutc'])
    tr = pd.DataFrame(list(transfers), columns=['playerId', 'startDate', 'fromTeamId', 'toTeamId'])
    return {'ms': ms, 'ev': pd.DataFrame(ev), 'lineups': lineups, 'iv': pd.DataFrame(iv), 'tr': tr,
            'eng': pd.DataFrame(columns=['playerId', 'seasonId', 'team', 'mins_lineup'])}, days


def _avail(inp, pid):
    t = av.availability_table(inp)
    return t[t['playerId'] == pid].iloc[0]


# --- the factor ----------------------------------------------------------------------
def test_factor_is_one_at_and_above_the_anchor_and_monotone_below():
    assert av.regular_factor(av.ANCHOR_SHARE, 0.0) == pytest.approx(1.0)
    assert av.regular_factor(0.9, 0.0) == 1.0
    shares = [0.033, 0.07, 0.11, 0.22, 0.30]
    f = [av.regular_factor(s, 0.0) for s in shares]
    assert all(0 < a < b <= 1 for a, b in zip(f, f[1:])), f
    # same minutes off the bench -> a lower chance of becoming a regular
    assert av.regular_factor(0.11, 1.0) < av.regular_factor(0.11, 0.0)
    assert av.regular_factor(float('nan'), 0.0) is None and av.regular_factor(0.1, None) is None


def test_reserve_team_label():
    assert is_reserve_team('Sporting CP II') and is_reserve_team('Vitória Guimarães II')
    assert not is_reserve_team('Atlético CP') and not is_reserve_team('Vila Real') and not is_reserve_team(None)


# --- club windows --------------------------------------------------------------------
def test_full_season_without_a_move():
    inp, _ = _inputs({1: [(T1, i, 'S') for i in range(3)]})
    r = _avail(inp, 1)
    assert not r['window_cut'] and r['avail_mins'] == pytest.approx(30 * L)


def test_mid_season_arrival_cuts_the_window():
    inp, days = _inputs({2: [(T1, i, 'S') for i in range(20, 30)]},
                        transfers=[(2, DAYS[20].strftime('%Y-%m-%d'), 555, T1)])
    r = _avail(inp, 2)
    assert r['window_cut'] and r['avail_mins'] == pytest.approx(10 * L)


def test_window_widens_to_cover_every_squad_appearance():
    # recorded arrival at match 20, but he was on the bench from match 15
    apps = [(T1, 15, 'U')] + [(T1, i, 'S') for i in range(20, 30)]
    inp, days = _inputs({3: apps}, transfers=[(3, DAYS[20].strftime('%Y-%m-%d'), 555, T1)])
    assert _avail(inp, 3)['avail_mins'] == pytest.approx(15 * L)


def test_tiny_window_falls_back_to_the_full_season():
    # arrives with 5 matches left (inside the season, so the window IS cut,
    # then reverted: fewer than MIN_WINDOW matches)
    inp, days = _inputs({4: [(T1, i, 'S') for i in range(25, 30)]},
                        transfers=[(4, DAYS[25].strftime('%Y-%m-%d'), 555, T1)])
    r = _avail(inp, 4)
    assert not r['window_cut'] and r['avail_mins'] == pytest.approx(30 * L)


def test_two_clubs_never_exceed_one_season():
    # moved T1 -> T2 at match 12; only the arrival at T2 is on record
    apps = [(T1, i, 'S') for i in range(0, 10)] + [(T2, i, 'S') for i in range(12, 30)]
    inp, days = _inputs({5: apps}, transfers=[(5, DAYS[12].strftime('%Y-%m-%d'), 777, T2)])
    r = _avail(inp, 5)
    assert r['n_clubs'] == 2 and r['avail_mins'] <= 30 * L + 1e-6


def test_two_clubs_without_dates_get_one_season():
    apps = [(T1, i, 'S') for i in range(0, 5)] + [(T2, i, 'S') for i in range(20, 25)]
    inp, _ = _inputs({6: apps})
    assert _avail(inp, 6)['avail_mins'] == pytest.approx(30 * L)


def test_nothing_is_inferred_from_the_lineups():
    # first squad after 1 January with 5+ team matches before, but no dated
    # move: no window (the January-arrival rule was removed 2026-10-29)
    jan = [i for i, d in enumerate(DAYS) if d >= pd.Timestamp('2026-01-01')]
    inp, _ = _inputs({7: [(T1, i, 'S') for i in jan]})
    r = _avail(inp, 7)
    assert not r['window_cut'] and r['avail_mins'] == pytest.approx(30 * L)


def test_a_renewal_is_not_a_move():
    # a record from the club to itself mid-season must not cut the window
    inp, _ = _inputs({9: [(T1, i, 'S') for i in range(20, 30)]},
                     transfers=[(9, DAYS[20].strftime('%Y-%m-%d'), T1, T1)])
    r = _avail(inp, 9)
    assert not r['window_cut'] and r['avail_mins'] == pytest.approx(30 * L)


def test_sub_fraction_counts_appearances_only():
    apps = [(T1, i, 'S') for i in range(4)] + [(T1, i, 'B') for i in range(4, 10)] + [(T1, 10, 'U')]
    inp, _ = _inputs({8: apps})
    r = _avail(inp, 8)
    assert (r['starts'], r['sub_apps']) == (4, 6) and r['sub_frac'] == pytest.approx(0.6)


# --- build / attach / engine value -----------------------------------------------------
def test_build_exempts_reserve_sides_and_scales_by_share():
    inp, _ = _inputs({10: [(T1, 0, 'S')], 11: [(T1, 0, 'S')], 12: [(T1, i, 'S') for i in range(20)]})
    inp['eng'] = pd.DataFrame({'playerId': [10, 11, 12], 'seasonId': [SID] * 3,
                               'team': ['Leça', 'Sporting CP II', 'Leça'], 'mins_lineup': [95, 95, 1900]})
    out = av.build(inp).set_index('playerId')
    assert out.loc[10, 'regular_factor'] < 0.5           # a one-match cameo
    assert out.loc[11, 'regular_factor'] == 1.0          # same cameo at a reserve side
    assert out.loc[12, 'regular_factor'] == 1.0          # 1,900' of 2,850' available
    assert out.loc[10, 'avail_share'] == pytest.approx(95 / (30 * L))
    assert SEASON_YEAR[SID] == out.loc[10, 'yr']


def test_attach_defaults_to_one(tmp_path):
    eng = pd.DataFrame({'playerId': [1, 2, 3], 'seasonId': [SID, SID, SID]})
    assert av.attach(eng, tmp_path / 'missing.parquet')['regular_factor'].tolist() == [1.0, 1.0, 1.0]
    p = tmp_path / 'a.parquet'
    pd.DataFrame({'playerId': pd.array([1, 2], dtype='Int64'), 'seasonId': pd.array([SID, SID], dtype='Int64'),
                  'regular_factor': [0.4, np.nan]}).to_parquet(p)
    assert av.attach(eng, p)['regular_factor'].tolist() == [0.4, 1.0, 1.0]


def test_attach_carries_the_share_columns(tmp_path):
    eng = pd.DataFrame({'playerId': [1, 2], 'seasonId': [SID, SID], 'avail_mins': [9.0, 9.0]})
    p = tmp_path / 'a.parquet'
    pd.DataFrame({'playerId': pd.array([1], dtype='Int64'), 'seasonId': pd.array([SID], dtype='Int64'),
                  'regular_factor': [0.4], 'yr': [2025], 'mins_year': [300], 'avail_mins': [3000.0]}).to_parquet(p)
    out = av.attach(eng, p)
    assert out.loc[0, ['yr', 'mins_year', 'avail_mins']].tolist() == [2025, 300, 3000.0]
    assert out.loc[1, ['yr', 'mins_year', 'avail_mins']].isna().all()   # stale input column replaced


def test_scoped_share_counts_each_year_once():
    # one year as L3 + Camp rows (same player-year availability), plus a second year
    rows = pd.DataFrame({'yr': [2024, 2024, 2025], 'mins_year': [300, 300, 1500],
                         'avail_mins': [3000.0, 3000.0, 2000.0]})
    s = av.scoped_share(rows)
    assert s['share'] == pytest.approx(1800 / 5000)
    assert (s['mins'], s['avail'], s['years']) == (1800.0, 5000.0, [2024, 2025])
    assert av.scoped_share(rows.iloc[[2]])['share'] == pytest.approx(0.75)
    # an ever-present starter a few minutes past the typical-length schedule
    over = av.scoped_share(pd.DataFrame({'yr': [2026], 'mins_year': [685], 'avail_mins': [679.0]}))
    assert (over['share'], over['avail']) == (1.0, 685.0)


def test_scoped_share_none_without_data():
    assert av.scoped_share(pd.DataFrame()) is None
    assert av.scoped_share(pd.DataFrame({'playerId': [1]})) is None      # no availability attached
    assert av.scoped_share(pd.DataFrame({'yr': [2025], 'mins_year': [94], 'avail_mins': [np.nan]})) is None


def test_engine_value_scales_by_the_factor():
    pool = np.sort(np.array([30.0, 40.0, 50.0, 60.0]))
    base = ei.engine_value_eur(50.0, pool, 'Striker', 22.0, SID)
    assert ei.engine_value_eur(50.0, pool, 'Striker', 22.0, SID, 0.5) == pytest.approx(base * 0.5)
    assert ei.engine_value_eur(50.0, pool, 'Striker', 22.0, SID, None) == base
    assert ei.engine_value_eur(50.0, pool, 'Striker', 22.0, SID, float('nan')) == base
    df = pd.DataFrame({'projection_abs': [50.0, 50.0], 'role': ['Striker'] * 2, 'age': [22.0] * 2,
                       'seasonId': [SID] * 2, 'regular_factor': [1.0, 0.25]})
    v = ei.engine_value_eur_frame(df).tolist()
    assert v[1] == pytest.approx(v[0] * 0.25)


@pytest.mark.skipif(not av.OUT.exists(), reason='no committed availability.parquet')
def test_committed_availability_is_sane():
    a = pd.read_parquet(av.OUT)
    assert a['regular_factor'].between(0, 1).all()
    assert (a.loc[a['reserve_team'], 'regular_factor'] == 1.0).all()
    assert a['avail_mins'].dropna().max() < 3600, 'a season never offers more than one season of minutes'


# --- transfer records ----------------------------------------------------------------
def test_transfer_rows_and_upsert():
    assert ft._rows({'transfer': [{'a': 1}]}) == [{'a': 1}] and ft._rows([]) == [] and ft._rows(None) == []
    old = ft._normalise([{'playerId': 1, 'startDate': '2026-01-20', 'fromTeamId': 5, 'toTeamId': 6, 'type': 'Loan',
                          'toTeamName': 'old'},
                         {'playerId': 2, 'startDate': '0000-00-00', 'fromTeamId': 5, 'toTeamId': 6, 'type': 'Loan'}], 'seed')
    assert len(old) == 1, 'an undated move is dropped'
    new = ft._normalise([{'playerId': 1, 'startDate': '2026-01-20', 'fromTeamId': 5, 'toTeamId': 6, 'type': 'Loan',
                          'toTeamName': 'new'}], 'api')
    out = ft.upsert(old, new)
    assert len(out) == 1 and out.iloc[0]['toTeamName'] == 'new' and out.iloc[0]['source'] == 'api'


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


class _Session:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc = resp, exc

    def get(self, *a, **k):
        if self.exc:
            raise self.exc
        return self.resp


def test_get_reads_the_refusal_from_the_body_and_survives_network_errors():
    import requests
    body, err = ft._get(_Session(_Resp(400, {'error': {'code': 403, 'message': 'Forbidden'}})), None, '/x')
    assert body is None and err.startswith('403')
    body, err = ft._get(_Session(exc=requests.ConnectionError('down')), None, '/x')
    assert body is None and err.startswith('network')
    body, err = ft._get(_Session(_Resp(200, {'transfer': []})), None, '/x')
    assert err is None and ft._rows(body) == []


def test_ranges_never_exceed_one_year_and_cover_the_span():
    from datetime import date
    r = list(ft._ranges(date(2025, 6, 1), date(2026, 10, 28)))
    assert all((b - a).days <= 364 for a, b in r) and len(r) == 2
    assert r[0][0] == date(2025, 6, 1) and r[-1][1] == date(2026, 10, 28)
    assert all((r[i + 1][0] - r[i][1]).days == 1 for i in range(len(r) - 1))


def test_unchanged_rows_keep_their_stamp():
    row = {'playerId': 1, 'startDate': '2026-01-20', 'fromTeamId': 5, 'toTeamId': 6, 'type': 'Loan',
           'toTeamName': 'X', 'active': 1}
    old = ft._normalise([row], 'seed').assign(fetched_at='2026-05-21T00:00:00Z')
    again = ft.upsert(old, ft._normalise([dict(row, active=True)], 'api'))
    assert again.iloc[0]['source'] == 'seed' and again.iloc[0]['fetched_at'] == '2026-05-21T00:00:00Z'
    changed = ft.upsert(old, ft._normalise([dict(row, toTeamName='Y')], 'api'))
    assert changed.iloc[0]['source'] == 'api' and len(changed) == 1


def test_stamp_follows_the_factors(tmp_path):
    p = tmp_path / 'a.parquet'
    assert av.stamp(p) == ''
    pd.DataFrame({'playerId': [1], 'seasonId': [SID], 'regular_factor': [0.5]}).to_parquet(p)
    s1 = av.stamp(p)
    import time
    time.sleep(0.01)
    pd.DataFrame({'playerId': [1], 'seasonId': [SID], 'regular_factor': [0.6]}).to_parquet(p)
    assert s1 and av.stamp(p) != s1
