"""The market-value scraper reads a player's VERIFIED Transfermarkt profile
(transfermarkt_links.py), never a namesake's. No request is made here."""
import pandas as pd
import pytest

pytest.importorskip('bs4')
pytest.importorskip('requests')
import scrape_transfermarkt as st  # noqa: E402


def _links(tmp_path):
    p = tmp_path / 'links.csv'
    pd.DataFrame([
        dict(playerId=1, status='ok', tm_player_id=111.0, tm_player_slug='right-man',
             tm_name='Right Man', tm_dob='1990-01-01'),
        dict(playerId=2, status='review', tm_player_id=222.0, tm_player_slug='x',
             tm_name='X', tm_dob='1991-01-01'),
        dict(playerId=3, status='not_in_squad', tm_player_id=None, tm_player_slug=None,
             tm_name=None, tm_dob=None),
    ]).to_csv(p, index=False)
    return p


def test_only_ok_links_are_verified(tmp_path):
    v = st.load_verified_links(_links(tmp_path))
    assert set(v) == {1}
    assert v[1].tm_id == 111
    assert v[1].tm_url.endswith('/right-man/profil/spieler/111')


def test_purge_drops_only_the_wrong_profile(tmp_path):
    v = st.load_verified_links(_links(tmp_path))
    val = tmp_path / 'valuations.parquet'
    pd.DataFrame([
        dict(playerId=1, source='transfermarkt', notes='tm_id=999 club_at_time=A'),  # namesake
        dict(playerId=1, source='transfermarkt', notes='tm_id=111'),                 # right man
        dict(playerId=1, source='manual', notes='club told us'),                     # not scraped
        dict(playerId=5, source='transfermarkt', notes='tm_id=555'),                 # unverified: kept
        dict(playerId=7, source='transfermarkt', notes='tm_id=777'),                 # named in `also`
        dict(playerId=1, source='transfermarkt', notes='no profile id'),             # unknown: kept
    ]).to_parquet(val, index=False)
    tr = tmp_path / 'transfers.parquet'
    pd.DataFrame([dict(playerId=1, tm_id=999), dict(playerId=1, tm_id=111.0),
                  dict(playerId=5, tm_id=555)]).to_parquet(tr, index=False)
    out = st.purge_wrong_profiles(v, also=[7], paths=(val, tr, tmp_path / 'absent.parquet'))
    assert out == {'valuations.parquet': (2, 2), 'transfers.parquet': (1, 1)}
    left = pd.read_parquet(val)
    assert sorted(left['notes']) == ['club told us', 'no profile id', 'tm_id=111', 'tm_id=555']
    assert pd.read_parquet(tr)['tm_id'].tolist() == [111, 555]
    # A second pass finds nothing more.
    assert st.purge_wrong_profiles(v, paths=(val, tr)) == {
        'valuations.parquet': (0, 0), 'transfers.parquet': (0, 0)}


def _run(monkeypatch, tmp_path, universe, dobs, verified, profile_dob):
    """run() with every network call replaced; returns (searched, log, metadata)."""
    searched, saved = [], {}
    monkeypatch.setattr(st, 'load_player_universe', lambda: pd.DataFrame(universe))
    monkeypatch.setattr(st, 'load_player_dob_map', lambda: dobs)
    monkeypatch.setattr(st, 'load_verified_links', lambda path=None: verified)
    monkeypatch.setattr(st, 'load_existing_valuations', lambda: pd.DataFrame())
    monkeypatch.setattr(st, 'make_session', lambda: None)

    def search(session, name, **kw):
        searched.append(name)
        return [st.TMMatch(tm_id=999, tm_url='u/999', name=name, dob=None, club=None, score=150.0)]
    monkeypatch.setattr(st, 'search_transfermarkt', search)
    monkeypatch.setattr(st, 'fetch_tm_player_profile', lambda s, url: {'dob': profile_dob})
    monkeypatch.setattr(st, 'fetch_market_value_history',
                        lambda s, tm_id: [{'value_eur': 1.0, 'as_of_date': '2026-01-01'}])
    monkeypatch.setattr(st, 'fetch_tm_transfer_history', lambda s, tm_id: [{'date': '2025-07-01'}])
    for name in ('append_rows', 'append_metadata', 'append_transfers'):
        monkeypatch.setattr(st, name, lambda rows, name=name: saved.__setitem__(name, list(rows)))
    monkeypatch.setattr(st, 'MATCH_LOG_PATH', tmp_path / 'log.csv')
    st.run()
    return searched, pd.read_csv(tmp_path / 'log.csv'), saved


def test_verified_link_skips_the_name_search(monkeypatch, tmp_path):
    v = {1: st.TMMatch(tm_id=111, tm_url='u/111', name='Right Man', dob='1990-01-01', club=None)}
    searched, log, saved = _run(monkeypatch, tmp_path,
                                [dict(playerId=1, playerName='Bruno Alves')],
                                {1: '1990-01-01'}, v, '1990-01-01')
    assert searched == []
    assert log['status'].tolist() == ['verified'] and log['tm_id'].tolist() == [111]
    assert {r['notes'] for r in saved['append_rows']} == {'tm_id=111'}
    assert [r['tm_id'] for r in saved['append_transfers']] == [111]


def test_name_search_link_with_another_birth_date_writes_nothing(monkeypatch, tmp_path):
    searched, log, saved = _run(monkeypatch, tmp_path,
                                [dict(playerId=1, playerName='Bruno Alves')],
                                {1: '1990-01-01'}, {}, '1981-11-27')
    assert searched == ['Bruno Alves']
    assert log['status'].tolist() == ['dob_mismatch']
    assert saved == {'append_rows': [], 'append_metadata': [], 'append_transfers': []}


def test_name_search_still_used_when_birth_dates_agree_or_ours_is_unknown(monkeypatch, tmp_path):
    for dobs in ({1: '1990-01-01'}, {}):
        _, log, saved = _run(monkeypatch, tmp_path, [dict(playerId=1, playerName='A B')],
                             dobs, {}, '1990-01-01')
        assert log['status'].tolist() == ['matched']
        assert len(saved['append_rows']) == 1 and len(saved['append_metadata']) == 1


def test_partial_run_keeps_other_players_log_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(st, 'MATCH_LOG_PATH', tmp_path / 'log.csv')
    pd.DataFrame([dict(playerId=1, status='matched'), dict(playerId=2, status='no_match')]
                 ).to_csv(tmp_path / 'log.csv', index=False)
    st.write_match_log([dict(playerId=1, status='verified')])
    log = pd.read_csv(tmp_path / 'log.csv').set_index('playerId')['status'].to_dict()
    assert log == {2: 'no_match', 1: 'verified'}


def test_relink_targets_conflict_players_not_yet_read(monkeypatch, tmp_path):
    v = {p: st.TMMatch(tm_id=t, tm_url='u', name='', dob=None, club=None)
         for p, t in ((1, 111), (2, 222), (3, 333))}
    conf, meta = tmp_path / 'conf.csv', tmp_path / 'meta.parquet'
    pd.DataFrame({'playerId': [1, 2, 3, 9]}).to_csv(conf, index=False)   # 9 has no verified link
    pd.DataFrame({'playerId': [1, 3], 'tm_id': [111.0, 999.0]}).to_parquet(meta, index=False)
    monkeypatch.setattr(st, 'LINK_CONFLICTS_PATH', conf)
    monkeypatch.setattr(st, 'METADATA_PATH', meta)
    assert st.players_to_relink(v) == [2, 3]      # 1 is already read from his verified profile


def test_verified_run_targets_unread_and_stale_profiles(monkeypatch, tmp_path):
    v = {p: st.TMMatch(tm_id=t, tm_url='u', name='', dob=None, club=None)
         for p, t in ((1, 111), (2, 222), (3, 333), (4, 444))}
    meta = tmp_path / 'meta.parquet'
    pd.DataFrame({'playerId': [1, 2, 3], 'tm_id': [111, 222, 999],
                  'fetched_at': ['2026-11-01T00:00:00', '2026-05-29T00:00:00', '2026-11-01T00:00:00']}
                 ).to_parquet(meta, index=False)
    monkeypatch.setattr(st, 'METADATA_PATH', meta)
    # 1 read recently: skipped. 2 stale, 3 read from another profile, 4 never read.
    assert st.players_to_read_verified(v, today='2026-11-12') == [2, 3, 4]


def test_long_run_saves_as_it_goes(monkeypatch, tmp_path):
    saves = []
    monkeypatch.setattr(st, 'SAVE_EVERY', 2)
    monkeypatch.setattr(st, 'load_player_universe',
                        lambda: pd.DataFrame([dict(playerId=i, playerName=f'P {i}') for i in range(1, 6)]))
    monkeypatch.setattr(st, 'load_player_dob_map', lambda: {})
    monkeypatch.setattr(st, 'load_verified_links', lambda path=None: {
        i: st.TMMatch(tm_id=100 + i, tm_url='u', name='', dob=None, club=None) for i in range(1, 6)})
    monkeypatch.setattr(st, 'load_existing_valuations', lambda: pd.DataFrame())
    monkeypatch.setattr(st, 'make_session', lambda: None)
    monkeypatch.setattr(st, 'fetch_tm_player_profile', lambda s, url: {'dob': None})
    monkeypatch.setattr(st, 'fetch_market_value_history', lambda s, tm_id: [])
    monkeypatch.setattr(st, 'fetch_tm_transfer_history', lambda s, tm_id: [{'date': '2025-07-01'}])
    monkeypatch.setattr(st, 'append_rows', lambda rows: None)
    monkeypatch.setattr(st, 'append_transfers', lambda rows: None)
    monkeypatch.setattr(st, 'append_metadata', lambda rows: saves.append([r['playerId'] for r in rows]))
    monkeypatch.setattr(st, 'MATCH_LOG_PATH', tmp_path / 'log.csv')
    st.run()
    assert saves == [[1, 2], [3, 4], [5]]                       # nothing saved twice, nothing lost
    assert sorted(pd.read_csv(tmp_path / 'log.csv').playerId) == [1, 2, 3, 4, 5]


def test_failed_transfer_read_is_not_recorded_as_read(monkeypatch, tmp_path):
    saved = {}
    monkeypatch.setattr(st, 'load_player_universe',
                        lambda: pd.DataFrame([dict(playerId=1, playerName='A'), dict(playerId=2, playerName='B')]))
    monkeypatch.setattr(st, 'load_player_dob_map', lambda: {})
    monkeypatch.setattr(st, 'load_verified_links', lambda path=None: {
        i: st.TMMatch(tm_id=100 + i, tm_url='u', name='', dob=None, club=None) for i in (1, 2)})
    monkeypatch.setattr(st, 'load_existing_valuations', lambda: pd.DataFrame())
    monkeypatch.setattr(st, 'make_session', lambda: None)
    monkeypatch.setattr(st, 'fetch_tm_player_profile', lambda s, url: {'dob': None})
    monkeypatch.setattr(st, 'fetch_market_value_history',
                        lambda s, tm_id: [{'value_eur': 1.0, 'as_of_date': '2026-01-01'}])

    def transfers(s, tm_id):
        if tm_id == 101:
            raise st.FetchFailed('HTTP 429')
        return [{'date': '2025-07-01'}]
    monkeypatch.setattr(st, 'fetch_tm_transfer_history', transfers)
    for name in ('append_rows', 'append_metadata', 'append_transfers'):
        monkeypatch.setattr(st, name, lambda rows, name=name: saved.__setitem__(name, [r['playerId'] for r in rows]))
    monkeypatch.setattr(st, 'MATCH_LOG_PATH', tmp_path / 'log.csv')
    st.run()
    assert saved == {'append_rows': [2], 'append_transfers': [2], 'append_metadata': [2]}
    assert pd.read_csv(tmp_path / 'log.csv').set_index('playerId')['status'].to_dict() == {1: 'fetch_failed', 2: 'verified'}
