"""match_minutes.py: how Wyscout answers are classified, and which matches
count as per-match / excluded. No network — a fake session replays answers."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import match_minutes as mm  # noqa: E402


class _Resp:
    def __init__(self, status, payload=None, text=''):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f'HTTP {self.status_code}')


class _Session:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def get(self, *a, **k):
        self.calls += 1
        return self.answers.pop(0)


_PLAYERS = {'players': [
    {'playerId': 7, 'seasonId': 192831, 'competitionId': 43324,
     'total': {'minutesOnField': 67, 'matchesInStart': 0},
     'positions': [{'position': {'code': 'rcb'}, 'percent': 100}]},
    {'playerId': 8, 'seasonId': 192831, 'competitionId': 43324,
     'total': {'minutesOnField': 0}, 'positions': []},     # unused sub: dropped
]}


def test_fetch_match_rows(monkeypatch):
    monkeypatch.setattr(mm.time, 'sleep', lambda s: None)
    rows = mm.fetch_match(_Session([_Resp(200, _PLAYERS)]), None, 5871526)
    assert rows == [{'matchId': 5871526, 'seasonId': 192831, 'competitionId': 43324,
                     'playerId': 7, 'minutes': 67, 'started': False, 'positions': 'rcb:100'}]


def test_transient_400_is_retried(monkeypatch):
    monkeypatch.setattr(mm.time, 'sleep', lambda s: None)
    s = _Session([_Resp(400, text='{"error":{"code":400}}'), _Resp(200, _PLAYERS)])
    assert mm.fetch_match(s, None, 1)[0]['minutes'] == 67
    assert s.calls == 2


def test_no_stats_and_forbidden_become_sentinels_without_retry(monkeypatch):
    monkeypatch.setattr(mm.time, 'sleep', lambda s: None)
    for text in ('{"error":{"code":404,"message":"Not Found: No statistical data available"}}',
                 '{"error":{"code":403,"message":"Forbidden"}}'):
        s = _Session([_Resp(400, text=text)])
        rows = mm.fetch_match(s, None, 99)
        assert s.calls == 1                       # definitive answer: no retries
        assert rows[0]['playerId'] == 0 and rows[0]['minutes'] == 0


def _cache(tmp_path, rows):
    p = tmp_path / 'mm.parquet'
    pd.DataFrame(rows, columns=mm.COLUMNS).to_parquet(p, index=False)
    return str(p)


def test_per_match_seasons_need_coverage_and_exclusions(tmp_path):
    # season 1: 3 played matches, 2 with minutes, 1 sentinel -> per-match,
    #           the sentinel match is excluded from per-90 work
    # season 2: 4 played, 1 stray row -> below 50% -> season totals, nothing excluded
    path = _cache(tmp_path, [
        (10, 1, 1, 7, 90, True, 'cf:100'), (11, 1, 1, 7, 90, True, 'cf:100'),
        (12, 0, 0, 0, 0, False, ''),
        (20, 2, 2, 9, 90, True, 'cf:100'),
    ])
    ms = pd.DataFrame({'matchId': [10, 11, 12, 20, 21, 22, 23],
                       'seasonId': [1, 1, 1, 2, 2, 2, 2]})
    assert mm.per_match_seasons(path, ms) == {1}
    assert mm.excluded_match_ids(ms, path) == {12}
    assert mm.matches_without_stats(path) == {12}
    assert 0 not in set(mm.load_cache(path)['playerId'])  # sentinels hidden


def test_unfetched_match_in_per_match_season_is_excluded(tmp_path):
    # a played match the fetch has not succeeded on yet must not count either
    path = _cache(tmp_path, [(10, 1, 1, 7, 90, True, ''), (11, 1, 1, 7, 90, True, '')])
    ms = pd.DataFrame({'matchId': [10, 11, 13], 'seasonId': [1, 1, 1]})
    assert mm.excluded_match_ids(ms, path) == {13}


def test_refetch_sentinel_never_replaces_real_minutes(tmp_path, monkeypatch):
    path = _cache(tmp_path, [(10, 1, 43324, 7, 90, True, 'cf:100')])
    ms = pd.DataFrame({'matchId': [10], 'seasonId': [1], 'status': ['Played'],
                       'dateutc': [pd.Timestamp.now(tz='UTC')]})
    monkeypatch.setattr(mm, 'current_season_ids', lambda: {43324: 1})
    monkeypatch.setattr(mm, 'competition_for_season', lambda sid: 43324)
    monkeypatch.setattr(mm, 'get_credentials', lambda c: ('u', 'p'))
    monkeypatch.setattr(mm, '_session', lambda: _Session(
        [_Resp(400, text='{"error":{"code":403,"message":"Forbidden"}}')]))
    out = mm.update_cache(ms, path=path)
    assert out.loc[out['matchId'] == 10, 'minutes'].tolist() == [90]


def test_rows_before_a_kept_sentinel_are_still_saved(tmp_path, monkeypatch):
    # match 10 (new) fetches real minutes; match 11 is the LAST in the batch and
    # answers 403 on re-fetch while holding real rows -> 10's rows must be saved
    path = _cache(tmp_path, [(11, 1, 43324, 8, 90, True, 'cb:100')])
    now = pd.Timestamp.now(tz='UTC')
    ms = pd.DataFrame({'matchId': [10, 11], 'seasonId': [1, 1], 'status': ['Played'] * 2,
                       'dateutc': [now, now]})
    monkeypatch.setattr(mm, 'current_season_ids', lambda: {43324: 1})
    monkeypatch.setattr(mm, 'competition_for_season', lambda sid: 43324)
    monkeypatch.setattr(mm, 'get_credentials', lambda c: ('u', 'p'))
    monkeypatch.setattr(mm, '_session', lambda: _Session(
        [_Resp(200, _PLAYERS), _Resp(400, text='{"error":{"code":403,"message":"Forbidden"}}')]))
    out = mm.update_cache(ms, path=path)
    assert set(out['matchId']) == {10, 11}
    assert mm.load_cache(path).groupby('matchId')['minutes'].sum().to_dict() == {10: 67, 11: 90}
