"""Career section: the full club list from the transfer record, and when the
Player Information card shows Transfermarkt's club instead of ours."""
import pandas as pd

from views import player_career as pc


def _transfers():
    return pd.DataFrame([
        dict(date='2023-07-01', from_club='B', to_club='C', transfer_type='free transfer', fee_eur=None, is_loan=False),
        dict(date='2020-07-01', from_club='Youth', to_club='A', transfer_type='-', fee_eur=None, is_loan=False),
        dict(date='2022-01-15', from_club='A', to_club='B', transfer_type='loan transfer', fee_eur=None, is_loan=False),
        dict(date='2022-06-30', from_club='B', to_club='A', transfer_type='End of loan', fee_eur=None, is_loan=False),
        dict(date='2022-07-01', from_club='A', to_club='B', transfer_type='x', fee_eur=50000.0, is_loan=False),
        dict(date=None, from_club='?', to_club='?', transfer_type='-', fee_eur=None, is_loan=False),
    ])


def test_clubs_lists_every_spell_newest_first():
    got = pc.clubs(_transfers())
    assert got.Club.tolist() == ['C', 'B', 'A', 'B', 'A', 'Youth']
    assert got.iloc[0].tolist() == ['2023-07-01', 'now', 'C', 'Free transfer']
    assert got.iloc[1].tolist() == ['2022-07-01', '2023-07-01', 'B', 'Transfer, €50,000']
    assert got.iloc[2]['Joined as'] == 'Back from loan'
    assert got.iloc[3].tolist() == ['2022-01-15', '2022-06-30', 'B', 'Loan']
    assert got.iloc[5].tolist() == ['', '2020-07-01', 'Youth', '']      # the club he left in his first recorded move
    assert pc.clubs(pd.DataFrame()).empty


def _links():
    return pd.DataFrame([dict(playerId=1, status='ok', tm_player_id=111.0),
                         dict(playerId=2, status='review', tm_player_id=222.0)])


def _meta():
    return pd.DataFrame([
        dict(playerId=1, tm_id=999, current_club='Namesake FC', joined_date='2026-07-01', on_loan_from=None,
             contract_until=None, fetched_at='2026-11-12T10:00:00'),
        dict(playerId=1, tm_id=111, current_club='New Club', joined_date='2026-07-01', on_loan_from='Parent FC',
             contract_until='2027-06-30', fetched_at='2026-11-12T10:00:00'),
        dict(playerId=2, tm_id=222, current_club='Other', joined_date='2026-07-01', on_loan_from=None,
             contract_until=None, fetched_at='2026-11-12T10:00:00'),
    ])


def test_current_club_only_from_the_sure_links_profile():
    cur = pc.current_club(1, _links(), _meta())
    assert cur['club'] == 'New Club' and cur['on_loan_from'] == 'Parent FC'
    assert cur['joined'] == pd.Timestamp('2026-07-01') and cur['read'] == pd.Timestamp('2026-11-12')
    assert pc.current_club(2, _links(), _meta()) is None      # link not sure
    assert pc.current_club(3, _links(), _meta()) is None
    assert pc.current_club(1, _links(), pd.DataFrame()) is None


def test_team_card_changes_only_when_he_joined_after_his_last_match_here():
    cur = pc.current_club(1, _links(), _meta())
    team, note = pc.team_now('Old Club', pd.Timestamp('2026-05-10 18:00'), cur)
    assert team == 'New Club'
    assert 'on loan from Parent FC' in note and '01 Jul 2026' in note and 'Old Club on 10 May 2026' in note
    # he was already at that club when he last played in our data: ours stands
    assert pc.team_now('Old Club', pd.Timestamp('2026-09-01'), cur) == ('Old Club', None)
    # same day counts as already there
    assert pc.team_now('Old Club', pd.Timestamp('2026-07-01 20:00'), cur) == ('Old Club', None)
    assert pc.team_now('Old Club', pd.Timestamp('2026-05-10'), None) == ('Old Club', None)
    assert pc.team_now('Old Club', pd.Timestamp('2026-05-10'), dict(cur, joined=None)) == ('Old Club', None)
    team, note = pc.team_now('Old Club', pd.Timestamp('2026-05-10'), dict(cur, club='Without Club', on_loan_from=None))
    assert team == 'Without club' and note.startswith('Transfermarkt lists him as without club since')


def test_same_day_moves_follow_the_chain_of_clubs():
    rows = [   # stored newest first, both on the same day
        dict(date='2024-01-31', from_club='B', to_club='C', transfer_type='free transfer', fee_eur=None, is_loan=False),
        dict(date='2024-01-31', from_club='A', to_club='B', transfer_type='End of loan', fee_eur=None, is_loan=False),
        dict(date='2023-07-01', from_club='B', to_club='A', transfer_type='loan transfer', fee_eur=None, is_loan=False),
    ]
    got = pc.clubs(pd.DataFrame(rows))
    assert got.Club.tolist() == ['C', 'B', 'A', 'B']
    assert got.Until.tolist() == ['now', '2024-01-31', '2024-01-31', '2023-07-01']


def test_aware_match_date_does_not_break_the_team_rule():
    cur = pc.current_club(1, _links(), _meta())
    team, note = pc.team_now('Old Club', pd.Timestamp('2026-05-10 18:00', tz='UTC'), cur)
    assert team == 'New Club' and note
