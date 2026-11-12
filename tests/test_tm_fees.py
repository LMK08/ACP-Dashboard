"""Transfermarkt sales as fee rows: which transfers qualify, how they join the
hand-curated file, and that reserve-side sales stay out of the range."""
import pandas as pd

import models.value.eur_intervals as ei
from valuations import build_tm_fees as bf


def _sim(a, b):
    return 1.0 if a.replace(' FC', '') == b else 0.0


def _matches():
    rows = []
    for pid, team in ((1, 'Varzim'), (2, 'Varzim'), (3, 'Varzim'), (4, 'Varzim'), (5, 'Varzim'), (6, 'Varzim')):
        for d in ('2024-03-01', '2024-05-10'):
            rows.append(dict(playerId=pid, matchId=hash((pid, d)) % 10**6, seasonId=10, team=team, dateutc=pd.Timestamp(d)))
    rows.append(dict(playerId=9, matchId=1, seasonId=10, team='Varzim', dateutc=pd.Timestamp('2024-05-18')))   # season's last match
    rows.append(dict(playerId=5, matchId=2, seasonId=11, team='Leixões', dateutc=pd.Timestamp('2024-09-01')))
    return pd.DataFrame(rows)


def _transfers():
    base = dict(transfer_type='-', is_loan=False, to_club='Buyer')
    return pd.DataFrame([
        dict(base, playerId=1, tm_id=101, date='2024-07-01', from_club='Varzim FC', fee_eur=150000.0),   # summer sale: IN
        dict(base, playerId=1, tm_id=999, date='2024-07-02', from_club='Varzim FC', fee_eur=9e6),        # a namesake's profile
        dict(base, playerId=2, tm_id=102, date='2024-07-01', from_club='Other Club', fee_eur=80000.0),   # another club sold him
        dict(base, playerId=3, tm_id=103, date='2024-12-01', from_club='Varzim FC', fee_eur=80000.0),    # 205 days after his last match
        dict(base, playerId=4, tm_id=104, date='2024-04-01', from_club='Varzim FC', fee_eur=40000.0),    # mid-season: IN
        dict(base, playerId=5, tm_id=105, date='2023-07-01', from_club='Varzim FC', fee_eur=50000.0),    # before any match here
        dict(base, playerId=6, tm_id=106, date='2024-07-01', from_club='Varzim FC', fee_eur=0.0),        # free
        dict(base, playerId=6, tm_id=106, date='2024-07-05', from_club='Varzim FC', fee_eur=20000.0, transfer_type='loan transfer'),
        dict(base, playerId=7, tm_id=107, date='2024-07-01', from_club='Varzim FC', fee_eur=70000.0),    # link not sure
    ])


def _links():
    return pd.DataFrame([dict(playerId=p, status='ok', tm_player_id=100.0 + p, our_name=f'P{p}', tm_player_slug=f'p{p}')
                         for p in (1, 2, 3, 4, 5, 6)] + [dict(playerId=7, status='review', tm_player_id=107.0,
                                                               our_name='P7', tm_player_slug='p7')])


def test_only_sales_by_his_club_with_data_before_them_become_fee_rows():
    out = bf.fee_rows(_transfers(), _links(), _matches(), similarity=_sim)
    assert out.playerId.tolist() == [4, 1]
    r = out.set_index('playerId')
    assert r.loc[1, ['from_team', 'fee_eur', 'as_of_date', 'season_id', 'transfer_type', 'synthetic_flag']].tolist() == \
        ['Varzim', 150000.0, '2024-07-01', 10, 'permanent', 0]
    assert r.loc[1, 'gap_days'] == 52 and r.loc[4, 'gap_days'] == 31
    assert set(bf.COLS) <= set(out.columns)


def test_hand_row_wins_over_a_transfermarkt_row_for_the_same_sale(tmp_path):
    hand, tm = tmp_path / 'hand.csv', tmp_path / 'tm.csv'
    cols = bf.COLS
    pd.DataFrame([dict.fromkeys(cols) | dict(playerId=1, fee_eur=100000, as_of_date='2024-07-15', season_id=10)]
                 )[cols].to_csv(hand, index=False)
    pd.DataFrame([dict.fromkeys(cols) | dict(playerId=1, fee_eur=150000, as_of_date='2024-07-01', season_id=10),
                  dict.fromkeys(cols) | dict(playerId=1, fee_eur=60000, as_of_date='2022-07-01', season_id=8),
                  dict.fromkeys(cols) | dict(playerId=2, fee_eur=70000, as_of_date='2024-07-01', season_id=10)]
                 )[cols].to_csv(tm, index=False)
    got = ei.read_fee_rows(hand, tm)
    assert sorted(zip(got.playerId, got.fee_eur)) == [(1, 60000), (1, 100000), (2, 70000)]
    assert len(ei.read_fee_rows(hand, tmp_path / 'absent.csv')) == 1


def test_reserve_side_players_get_no_range_and_their_sales_stay_out():
    calib = {'support': {'min_value_eur': 25000}, 'headline_level': '0.50',
             'levels': {'0.50': {'q': 0.5, 'shipped': True}}}
    assert ei.projected_eur_interval(150_000, 0.5, calib, team='Varzim')[0] is not None
    assert ei.projected_eur_interval(150_000, 0.5, calib, team='Sporting CP II') == (None, None)
    assert ei.range_support_reason(150_000, calib, team='Sporting CP II') == ei.RESERVE_REASON
    assert ei.range_support_reason(150_000, calib, team=None) is None
    # the shipped calibration holds no reserve-side sale, and says how many it left out
    shipped = ei.load_calibration()
    assert not any(str(r.get(k) or '').endswith(' II') for r in shipped.get('calibration', [])
                   for k in ('engine_team', 'from_team'))
    assert shipped['reserve_side']['n'] == shipped['excluded'].get('reserve_side', 0)
