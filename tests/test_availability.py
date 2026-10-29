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
    """n weekly matches per club, Aug -> Apr; both clubs play every week (each
    against an opponent outside the data); every match has API minutes of L."""
    days = pd.date_range('2025-08-10', periods=n, freq='7D')
    ms, ev, mins, lineups = [], [], [], {}
    for i, d in enumerate(days):
        for k, t in enumerate((T1, T2)):
            mid = 1000 + 2 * i + k
            ms.append({'matchId': mid, 'seasonId': SID, 'dateutc': d.strftime('%Y-%m-%d 15:00:00'),
                       'homeTeamName': f'Clube {t}', 'awayTeamName': f'Visitante {mid}'})
            ev.append({'matchId': mid, 'seasonId': SID, 'team.id': t, 'team.name': f'Clube {t}',
                       'player.id': 9000 + t})   # a fixture player
            mins.append({'matchId': mid, 'playerId': 9000 + t, 'minutes': L})
            lineups[mid] = {f'Clube {t}': {'lineup': [{'playerId': 9000 + t}], 'bench': [], 'substitutions': []}}
    return days, ms, ev, mins, lineups


def _inputs(players, transfers=(), n=30):
    """players: {pid: [(team, match_index, flag)]}, flag 'S' start / 'B' came on / 'U' unused."""
    days, ms, ev, mins, lineups = _season(n)
    for pid, apps in players.items():
        for t, i, flag in apps:
            mid = 1000 + 2 * i + (0 if t == T1 else 1)
            side = lineups[mid][f'Clube {t}']
            if flag == 'S':
                side['lineup'].append({'playerId': pid})
            else:
                side['bench'].append({'playerId': pid})
                if flag == 'B':
                    side['substitutions'].append({'playerIn': pid, 'playerOut': 9000 + t})
            if flag in ('S', 'B'):
                ev.append({'matchId': mid, 'seasonId': SID, 'team.id': t, 'team.name': f'Clube {t}', 'player.id': pid})
    ms = pd.DataFrame(ms)
    ms['d'] = pd.to_datetime(ms['dateutc'])
    tr = pd.DataFrame(list(transfers), columns=['playerId', 'startDate', 'fromTeamId', 'toTeamId'])
    return {'ms': ms, 'ev': pd.DataFrame(ev), 'lineups': lineups, 'mins': pd.DataFrame(mins), 'tr': tr,
            'clock': pd.DataFrame(), 'exact': pd.DataFrame(columns=['playerId', 'seasonId', 'minutes']),
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


# --- exact available minutes -------------------------------------------------------------
def test_schedule_counts_matches_without_event_data():
    # the club's matches 10-19 have no events (only the fixture list knows them)
    inp, _ = _inputs({13: [(T1, i, 'S') for i in range(10)]})
    gone = {1000 + 2 * i for i in range(10, 20)}
    inp['ev'] = inp['ev'][~inp['ev']['matchId'].isin(gone)]
    assert _avail(inp, 13)['avail_mins'] == pytest.approx(30 * L)


def test_unplayed_fixtures_and_stat_less_matches_are_not_available():
    inp, _ = _inputs({17: [(T1, i, 'S') for i in range(10)]})
    ms = inp['ms'].assign(status='Played')
    future = ms[ms['matchId'] == 1000].assign(matchId=99999, status='Fixture', d=pd.Timestamp('2026-05-30'))
    inp['ms'] = pd.concat([ms, future], ignore_index=True)
    assert _avail(inp, 17)['avail_mins'] == pytest.approx(30 * L)        # the fixture adds nothing
    inp['excluded'] = {1000 + 2 * i for i in range(25, 30)}               # no per-match minutes for 5 of T1's
    assert _avail(inp, 17)['avail_mins'] == pytest.approx(25 * L)


def test_squad_view_counts_only_matchday_squads():
    # 10 starts, 5 unused on the bench, 15 club matches without him in the squad
    apps = [(T1, i, 'S') for i in range(10)] + [(T1, i, 'U') for i in range(10, 15)]
    inp, _ = _inputs({18: apps})
    r = _avail(inp, 18)
    assert (r['n_squad'], r['n_club'], r['squad_clubs']) == (15, 30, f'Clube {T1}:15:30')
    assert r['squad_mins'] == pytest.approx(15 * L) and r['avail_mins'] == pytest.approx(30 * L)
    s = av.scoped_share(pd.DataFrame({'yr': [2025], 'mins_year': [10 * L], **{k: [r[k]] for k in
                        ('avail_mins', 'len_api', 'squad_mins', 'n_squad', 'n_club', 'squad_clubs')}}))
    assert s['squad_share'] == pytest.approx(10 / 15) and s['share'] == pytest.approx(10 / 30)
    assert s['clubs'] == [(f'Clube {T1}', 15, 30)] and s['squad_played'] == pytest.approx(10 * L)


def test_squad_match_between_his_two_clubs_counts_once():
    # T1 -> T2 at match 12 (dated); T1's match 20 is against T2, where he sat on T2's bench
    apps = [(T1, i, 'S') for i in range(10)] + [(T2, i, 'S') for i in range(12, 20)]
    inp, _ = _inputs({19: apps}, transfers=[(19, DAYS[12].strftime('%Y-%m-%d'), T1, T2)])
    ab = 1000 + 2 * 20
    inp['ms'].loc[inp['ms']['matchId'] == ab, 'awayTeamName'] = f'Clube {T2}'
    inp['lineups'][ab][f'Clube {T2}'] = {'lineup': [{'playerId': 9000 + T2}], 'bench': [{'playerId': 19}],
                                         'substitutions': []}
    r = _avail(inp, 19)
    clubs = dict((c.rsplit(':', 2)[0], int(c.rsplit(':', 2)[1])) for c in r['squad_clubs'].split('|'))
    assert r['n_squad'] == 19 and clubs == {f'Clube {T1}': 10, f'Clube {T2}': 9}


def test_an_empty_lineup_entry_is_not_known():
    inp, _ = _inputs({20: [(T1, i, 'S') for i in range(10)]})
    inp['lineups'][1000 + 2 * 25] = {}                    # scanned before kick-off
    assert _avail(inp, 20)['n_club'] == 29


def test_match_length_api_then_clock_then_median():
    inp, _ = _inputs({14: [(T1, i, 'S') for i in range(3)]}, n=4)
    m_api, m_clock, m_none = 1000, 1002, 1004           # club T1's matches 0, 1, 2
    inp['mins'] = pd.DataFrame({'matchId': [m_api, m_api], 'playerId': [1, 2], 'minutes': [96, 71]})
    inp['clock'] = pd.DataFrame({'1H': [47.5], '2H': [93.0]}, index=pd.Index([m_clock], name='matchId'))
    ln = av.match_lengths(inp)
    assert ln[m_api] == (96.0, 'api')                    # the most any player got
    assert ln[m_clock] == (pytest.approx(95.5), 'clock')  # 47.5 + 93 - 45
    assert ln[m_none][1] == 'median'
    r = _avail(inp, 14)
    assert r['len_api'] == pytest.approx(96.0 / r['avail_mins'])


def test_team_names_match_across_accents_and_spacing():
    inp, _ = _inputs({15: [(T1, i, 'S') for i in range(3)]})
    inp['ev'] = inp['ev'].assign(**{'team.name': inp['ev']['team.name'].str.replace('Clube', 'Clubé ') + ' '})
    assert _avail(inp, 15)['avail_mins'] == pytest.approx(30 * L)


def test_build_adds_display_rows_for_players_the_engine_does_not_rate(tmp_path):
    # 16 is a keeper (no engine row): a display-only row with his exact minutes
    inp, _ = _inputs({12: [(T1, i, 'S') for i in range(20)], 16: [(T2, i, 'S') for i in range(30)]})
    inp['eng'] = pd.DataFrame({'playerId': [12], 'seasonId': [SID], 'team': ['Leça'], 'mins_lineup': [1900]})
    inp['exact'] = pd.DataFrame({'playerId': [12, 16], 'seasonId': [SID, SID], 'minutes': [1900, 2850]})
    out = av.build(inp).set_index('playerId')
    assert bool(out.loc[12, 'in_engine']) and not bool(out.loc[16, 'in_engine'])
    assert np.isnan(out.loc[16, 'regular_factor']) and out.loc[16, 'avail_share'] == pytest.approx(1.0)
    eng = pd.DataFrame({'playerId': [12], 'seasonId': [SID]})
    p = tmp_path / 'a.parquet'
    out.reset_index().to_parquet(p)
    assert av.attach(eng, p)['regular_factor'].tolist() == [1.0]   # the keeper row never joins


def test_scoped_share_counts_each_year_once():
    # one year as L3 + Camp rows (same player-year availability), plus a second year
    rows = pd.DataFrame({'yr': [2024, 2024, 2025], 'mins_year': [300, 300, 1500],
                         'avail_mins': [3000.0, 3000.0, 2000.0], 'len_api': [1.0, 1.0, 1.0]})
    s = av.scoped_share(rows)
    assert s['share'] == pytest.approx(1800 / 5000)
    assert (s['mins'], s['avail'], s['years'], s['exact']) == (1800.0, 5000.0, [2024, 2025], True)
    assert av.scoped_share(rows.iloc[[2]])['share'] == pytest.approx(0.75)
    # a clock-derived schedule a few seconds short of an ever-present player's minutes
    over = av.scoped_share(pd.DataFrame({'yr': [2026], 'mins_year': [2524], 'avail_mins': [2518.0], 'len_api': [0.0]}))
    assert (over['share'], over['avail'], over['exact']) == (1.0, 2524.0, False)


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
    e = a[a['in_engine']]
    assert e['regular_factor'].between(0, 1).all()
    assert (e.loc[e['reserve_team'], 'regular_factor'] == 1.0).all()
    assert a.loc[~a['in_engine'], 'regular_factor'].isna().all(), 'display-only rows carry no factor'
    assert a['avail_mins'].dropna().max() < 3600, 'a season never offers more than one season of minutes'
    assert a['len_api'].dropna().between(0, 1).all()


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
