"""Transfermarkt links: club-name matching, page parsing, coach matching by
dates, and the coach-naming hook. No network: pages are small fixtures."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transfermarkt_links as tl  # noqa: E402
from models.coaches import build_coaches as bc  # noqa: E402


def test_similarity_rules():
    assert tl.similarity('Fafe', 'AD Fafe') == pytest.approx(1.0)
    assert tl.similarity('Sporting Braga II', 'SC Braga B') == pytest.approx(0.95)    # reserve side, 'Sporting' dropped
    assert tl.similarity('União de Leiria', 'UD Leiria') == pytest.approx(0.95)      # likely; needs the data check
    assert tl.similarity('União de Coimbra', 'Académica Coimbra') < tl.TEAM_OK_SCORE
    assert tl.similarity('Sporting de Tomar', 'União de Tomar') < 1.0
    # a first team never matches a reserve side, either way round
    assert tl.similarity('Fafe', 'AD Fafe B') == 0.0 and tl.similarity('Sporting CP II', 'Sporting CP') == 0.0
    # contained names link; youth sides never get that lift
    assert tl.similarity('Paredes', 'USC Paredes') >= 0.9
    assert tl.similarity('Paredes', 'USC Paredes Youth') < tl.TEAM_OK_SCORE
    assert tl.similarity('Fafe', 'AD Fafe U19') < tl.TEAM_OK_SCORE


def test_pick_club_needs_a_clear_portuguese_senior_club():
    c = [{'tm_team_id': 1, 'tm_slug': 'a', 'tm_name': 'AD Sanjoanense', 'country': 'Portugal',
          'league': 'Campeonato de Portugal - Série B'},
         {'tm_team_id': 2, 'tm_slug': 'b', 'tm_name': 'SC Sanjoanense', 'country': 'Portugal', 'league': None},
         {'tm_team_id': 3, 'tm_slug': 'c', 'tm_name': 'Sanjoanense', 'country': 'Brazil', 'league': 'Liga 3'}]
    best, status, _ = tl.pick_club('Sanjoanense', c)
    assert best['tm_team_id'] == 1 and status == 'ok'
    # two senior clubs with the same name: a person decides
    c[1]['league'] = 'Liga 3'
    assert tl.pick_club('Sanjoanense', c)[1] == 'review'
    # a near name that is not in a senior league is not linked automatically
    near = [{'tm_team_id': 9, 'tm_slug': 'x', 'tm_name': 'Clube Oriental Lissabon', 'country': 'Portugal', 'league': None}]
    assert tl.pick_club('Oriental Lisboa', near)[1] == 'review'
    assert tl.pick_club('Oriental Lisboa', [dict(near[0], country='Spain')]) == (None, 'not_found', '')


SEARCH = '''<div class="box"><h2>Search results: Clubs - 2 Hits</h2><table class="items"><tbody>
<tr><td><a href="/ad-fafe/startseite/verein/3326"><img></a></td>
    <td><a href="/ad-fafe/startseite/verein/3326">AD Fafe</a></td>
    <td><img class="flaggenrahmen" title="Portugal"></td>
    <td><a href="/liga-3/startseite/wettbewerb/PT3A">Liga 3</a></td></tr>
<tr><td><a href="/ad-fafe-u19/startseite/verein/20727">AD Fafe U19</a></td>
    <td><img class="flaggenrahmen" title="Portugal"></td></tr>
</tbody></table></div>
<div class="box"><h2>Search results: Players</h2><table class="items"><tbody>
<tr><td><a href="/x/startseite/verein/1">Not a club row</a></td></tr></tbody></table></div>'''

STAFF = '''<table class="items"><tbody>
<tr><td><a href="/luis-pinto/profil/trainer/51375"><img></a><a href="/luis-pinto/profil/trainer/51375">Luís Pinto</a> 01/04/1989</td>
    <td></td><td>23/01/2023</td><td>30/06/2024</td><td>524 days</td><td>42</td><td>1.64</td></tr>
<tr><td><a href="/mario-ferreira/profil/trainer/120583">Mário Ferreira</a> 13/01/1995</td>
    <td></td><td>17/09/2025</td><td></td><td>421 days</td><td>43</td><td>1.67</td></tr>
<tr><td>No link here</td><td></td><td>01/01/2020</td><td></td><td></td><td>1</td><td></td></tr>
</tbody></table>'''


def test_parsers():
    clubs = tl.parse_club_search(SEARCH)
    assert [(c['tm_team_id'], c['tm_name'], c['country'], c['league']) for c in clubs] == [
        (3326, 'AD Fafe', 'Portugal', 'Liga 3'), (20727, 'AD Fafe U19', 'Portugal', None)]
    m = tl.parse_manager_history(STAFF)
    assert [(r['tm_coach_id'], r['name'], r['matches']) for r in m] == [(51375, 'Luís Pinto', 42), (120583, 'Mário Ferreira', 43)]
    assert m[0]['dob'] == pd.Timestamp('1989-04-01')                 # day/month/year
    assert m[0]['start'] == pd.Timestamp('2023-01-23') and m[0]['end'] == pd.Timestamp('2024-06-30')
    assert pd.isna(m[1]['end'])                                       # still in charge


def _matches(coach, team, days):
    return pd.DataFrame({'matchId': [hash((coach, team, d)) % 10**9 for d in days], 'teamId': team, 'team': f'T{team}',
                         'coachId': coach, 'date': pd.to_datetime(days) + pd.Timedelta(hours=15)})


def test_link_coaches_by_dates():
    mg = pd.DataFrame([
        {'teamId': 1, 'tm_coach_id': 500, 'name': 'A', 'dob': pd.Timestamp('1980-01-01'),
         'start': pd.Timestamp('2023-01-23'), 'end': pd.Timestamp('2024-06-30')},
        {'teamId': 1, 'tm_coach_id': 501, 'name': 'B', 'dob': pd.Timestamp('1975-01-01'),
         'start': pd.Timestamp('2024-07-01'), 'end': pd.NaT}])
    ten = pd.date_range('2023-01-21', periods=10, freq='14D').strftime('%Y-%m-%d')       # first match 2 days BEFORE the appointment date
    tm = pd.concat([
        _matches(7, 1, ten),                                           # all inside A's stint (with the slack)
        _matches(8, 1, pd.date_range('2024-08-01', periods=9, freq='7D').strftime('%Y-%m-%d')),   # B, open-ended stint
        _matches(9, 1, list(pd.date_range('2024-05-01', periods=5, freq='7D').strftime('%Y-%m-%d'))
                 + list(pd.date_range('2024-08-01', periods=5, freq='7D').strftime('%Y-%m-%d'))),  # split 5 / 5
        _matches(10, 2, ten),                                          # a club with no manager history
        _matches(11, 1, ['2010-01-01'] * 1)], ignore_index=True)       # before any stint
    out = tl.link_coaches(tm, mg).set_index('coachId')
    assert out.loc[7, ['tm_coach_id', 'status', 'matches_covered']].tolist() == [500, 'ok', 10]
    assert out.loc[8, 'tm_coach_id'] == 501 and out.loc[8, 'status'] == 'ok'
    assert out.loc[9, 'status'] == 'review' and out.loc[9, 'share'] == pytest.approx(0.5)
    assert out.loc[10, 'status'] == 'no_linked_club' and out.loc[11, 'status'] == 'no_stint'
    # too few matches is never an automatic link, however clean
    few = tl.link_coaches(_matches(7, 1, ten[:tl.COACH_OK_MATCHES - 1]), mg)
    assert few.status.iloc[0] == 'review'


def test_add_transfermarkt_names_only_the_unnamed(tmp_path):
    coaches = pd.DataFrame({'coachId': [1, 2, 3], 'name': ['Coach #1', 'Known Name', 'Coach #3'],
                            'fullName': ['', 'Known Name', ''], 'birthDate': [None, '1970-05-05', None],
                            'nationality': [None, 'Portugal', None], 'resolved': [False, True, False]})
    links = tmp_path / 'links.csv'
    pd.DataFrame({'coachId': [1, 2, 3], 'tm_coach_id': [500, 600, 700], 'tm_name': ['From TM', 'Other', 'Unsure'],
                  'tm_dob': ['1989-04-01', '1960-01-01', '1990-01-01'], 'status': ['ok', 'ok', 'review']}).to_csv(links, index=False)
    out = bc.add_transfermarkt(coaches, links).set_index('coachId')
    assert out.loc[1, 'name'] == 'From TM' and out.loc[1, 'birthDate'] == '1989-04-01' and out.loc[1, 'tm_coach_id'] == 500
    assert out.loc[2, 'name'] == 'Known Name' and out.loc[2, 'birthDate'] == '1970-05-05' and out.loc[2, 'tm_coach_id'] == 600
    assert out.loc[3, 'name'] == 'Coach #3' and pd.isna(out.loc[3, 'tm_coach_id'])      # 'review' links are never applied
    # no link file: unchanged, with an empty id column
    none = bc.add_transfermarkt(coaches, tmp_path / 'missing.csv')
    assert none.name.tolist() == coaches.name.tolist() and none.tm_coach_id.isna().all()


def test_contained_name_must_be_clear_of_every_namesake():
    """'Vitória' is inside two clubs' names: the one in a national league must
    not win by default - the club we mean may have dropped out of them."""
    c = [{'tm_team_id': 1, 'tm_slug': 'a', 'tm_name': 'Vitória Guimarães SC', 'country': 'Portugal', 'league': 'Liga Portugal'},
         {'tm_team_id': 2, 'tm_slug': 'b', 'tm_name': 'Vitória Setúbal FC', 'country': 'Portugal', 'league': None}]
    assert tl.pick_club('Vitória FC', c)[1] == 'review'
    c = [{'tm_team_id': 1, 'tm_slug': 'a', 'tm_name': 'Académica Coimbra', 'country': 'Portugal', 'league': 'Liga Portugal 2'},
         {'tm_team_id': 2, 'tm_slug': 'b', 'tm_name': 'CF União de Coimbra Youth', 'country': 'Portugal', 'league': None}]
    assert tl.pick_club('União de Coimbra', c)[1] != 'ok'
    # a clear contained name with only youth namesakes still links
    c = [{'tm_team_id': 5, 'tm_slug': 'p', 'tm_name': 'USC Paredes', 'country': 'Portugal', 'league': 'Liga 3'},
         {'tm_team_id': 6, 'tm_slug': 'q', 'tm_name': 'USC Paredes Youth', 'country': 'Portugal', 'league': None},
         {'tm_team_id': 7, 'tm_slug': 'r', 'tm_name': 'SU 1º Dezembro Y17', 'country': 'Portugal', 'league': None}]
    assert tl.pick_club('Paredes', c)[1] == 'ok'
    for youth in ('SU 1º Dezembro Y17', 'SU Sintrense Youth19', 'Académica de Coimbra - SF', 'AD Fafe Sub-19'):
        assert tl._YOUTH.search(' ' + tl.fold(youth) + ' '), youth


def test_changeover_and_stale_open_stints():
    mg = pd.DataFrame([
        {'teamId': 1, 'tm_coach_id': 1, 'name': 'Old', 'dob': pd.NaT, 'start': pd.Timestamp('2020-07-01'), 'end': pd.NaT},      # end left blank
        {'teamId': 1, 'tm_coach_id': 2, 'name': 'Out', 'dob': pd.NaT, 'start': pd.Timestamp('2023-07-01'), 'end': pd.Timestamp('2024-06-30')},
        {'teamId': 1, 'tm_coach_id': 3, 'name': 'In', 'dob': pd.NaT, 'start': pd.Timestamp('2024-07-01'), 'end': pd.NaT}])
    closed = tl.close_stints(mg).set_index('tm_coach_id')
    assert closed.loc[1, 'end'] == pd.Timestamp('2023-06-30') and pd.isna(closed.loc[3, 'end'])
    tm = _matches(7, 1, ['2024-06-29', '2024-07-02', '2021-05-05', '2023-06-29'])
    who = dict(zip(pd.to_datetime(tl.stint_hits(tm, mg).date).dt.strftime('%Y-%m-%d'), tl.stint_hits(tm, mg).name))
    assert who['2024-06-29'] == 'Out'        # the outgoing man's last days are his, not the successor's
    assert who['2024-07-02'] == 'In'
    assert who['2021-05-05'] == 'Old'
    assert who['2023-06-29'] == 'Old'        # inside Old's closed stint; the slack only applies to a match in NO stint
    # a match two days before the first appointment of all falls back on the slack
    early = tl.stint_hits(_matches(7, 1, ['2020-06-29']), mg)
    assert early.name.tolist() == ['Old']
    assert tl.stint_hits(_matches(7, 1, ['2020-06-20']), mg).empty


def test_verify_team_links_uses_named_coaches():
    mg = pd.DataFrame([
        {'teamId': 1, 'tm_coach_id': 1, 'name': 'Luís Pinto', 'dob': pd.NaT, 'start': pd.Timestamp('2023-01-01'), 'end': pd.NaT},
        {'teamId': 2, 'tm_coach_id': 2, 'name': 'Someone Else', 'dob': pd.NaT, 'start': pd.Timestamp('2023-01-01'), 'end': pd.NaT}])
    days = pd.date_range('2023-02-01', periods=12, freq='7D').strftime('%Y-%m-%d')
    tm = pd.concat([_matches(10, 1, days), _matches(20, 2, days)], ignore_index=True)
    coaches = pd.DataFrame({'coachId': [10, 20], 'name': ['L. Pinto', 'Rui Borges'], 'fullName': ['Luís Miguel Pinto', 'Rui Borges'],
                            'resolved': [True, True]})
    links = pd.DataFrame({'teamId': [1, 2], 'team': ['A', 'B'], 'status': ['ok', 'ok'], 'score': [0.9, 0.9]})
    out = tl.verify_team_links(tm, mg, coaches, links).set_index('teamId')
    assert out.loc[1, 'status'] == 'ok' and out.loc[1, 'agree_share'] == 1.0
    assert out.loc[2, 'status'] == 'review' and out.loc[2, 'agree_share'] == 0.0
    # an exact-name link survives a lack of evidence, but not clear disagreement
    exact = tl.verify_team_links(tm.iloc[:3], mg, coaches, links.assign(score=1.0)).set_index('teamId')
    assert exact.loc[1, 'status'] == 'ok'
    assert tl.verify_team_links(tm, mg, coaches, links.assign(score=1.0)).set_index('teamId').loc[2, 'status'] == 'review'


def test_stale_transfermarkt_name_is_withdrawn(tmp_path):
    """A name taken from a link that is no longer 'ok' goes back to unnamed."""
    coaches = pd.DataFrame({'coachId': [1], 'name': ['From TM'], 'fullName': ['From TM'], 'birthDate': ['1989-04-01'],
                            'nationality': [None], 'resolved': [False], 'tm_coach_id': pd.array([500], dtype='Int64')})
    links = tmp_path / 'links.csv'
    pd.DataFrame({'coachId': [1], 'tm_coach_id': [500], 'tm_name': ['From TM'], 'tm_dob': ['1989-04-01'],
                  'status': ['review']}).to_csv(links, index=False)
    out = bc.add_transfermarkt(coaches, links).iloc[0]
    assert out['name'] == 'Coach #1' and out['birthDate'] is None and pd.isna(out['tm_coach_id'])
