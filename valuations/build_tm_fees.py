"""Paid permanent sales from Transfermarkt transfer records, in the columns of
reported_fees.csv, for the fee calibration (models/value/eur_intervals.py)
and the sale-date projections (models/ratings/build_projection.py).

    python valuations/build_tm_fees.py          # writes valuations/tm_fees.csv

A transfer becomes a fee row only when ALL of these hold:

  * the player's Transfermarkt link is sure (tm_player_links.csv 'ok') and
    the transfer was read from that profile;
  * a fee above zero was paid and it is not a loan or a loan's end;
  * he has a match in our data BEFORE the transfer date (no look-ahead: a
    fee can only meet a value built from what came before it);
  * the SELLING club on Transfermarkt is the club he last played for in our
    data (name similarity >= 0.9). A fee paid to a later club, or for a
    first-team player who made a cameo for the reserve side, is not a
    price for what our data shows;
  * the sale is during that season, or within 120 days of his last match
    for that club (the summer window after it).

Market values are NOT used anywhere: only what a club actually paid.
Rows already in the hand-curated file are dropped when read
(eur_intervals.read_fee_rows), so a hand row always wins. Sales from
reserve sides are written here but kept out of the calibration by the
pairing ('reserve_side'): they sell at a multiple of the engine value.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DASH = HERE.parent
sys.path.insert(0, str(DASH))

OUT = HERE / 'tm_fees.csv'
MIN_CLUB_SIMILARITY = 0.9
SUMMER_WINDOW_DAYS = 120
COLS = ['playerId', 'player_name', 'from_team', 'to_team', 'transfer_type', 'fee_eur', 'add_ons_eur',
        'sell_on_pct', 'as_of_date', 'season_id', 'contract_until', 'loan_until', 'source_url', 'notes',
        'synthetic_flag']


def our_matches(events_path=None, matches_path=None):
    """One row per (playerId, matchId): kick-off, season and his team's name."""
    ev = pd.read_parquet(events_path or DASH / 'raw_events.parquet',
                         columns=['player.id', 'matchId', 'seasonId', 'team.name']).drop_duplicates()
    ms = pd.read_parquet(matches_path or DASH / 'matches_summary.parquet', columns=['matchId', 'dateutc'])
    ev = ev.merge(ms, on='matchId').rename(columns={'player.id': 'playerId', 'team.name': 'team'})
    ev['dateutc'] = pd.to_datetime(ev['dateutc'], errors='coerce')
    return ev.dropna(subset=['playerId', 'dateutc'])


def fee_rows(transfers, links, matches, similarity=None):
    """The fee rows (COLS plus `gap_days`, `tm_from`) the rules above allow."""
    if similarity is None:
        from transfermarkt_links import similarity
    ok = links[(links['status'] == 'ok') & links['tm_player_id'].notna()].drop_duplicates('playerId').set_index('playerId')
    t = transfers[transfers['playerId'].isin(ok.index)].copy()
    t = t[pd.to_numeric(t['tm_id'], errors='coerce') == t['playerId'].map(ok['tm_player_id'])]
    t['date'] = pd.to_datetime(t['date'], errors='coerce')
    kind = t['transfer_type'].astype(str)
    loan = kind.str.contains('oan') | t.get('is_loan', False).astype(bool)
    t = t[(pd.to_numeric(t['fee_eur'], errors='coerce').fillna(0) > 0) & ~loan & t['date'].notna()]
    season_end = matches.groupby('seasonId')['dateutc'].max()
    by_player = {int(p): g.sort_values('dateutc') for p, g in matches[matches['playerId'].isin(t['playerId'])].groupby('playerId')}
    rows = []
    for r in t.itertuples():
        m = by_player.get(int(r.playerId))
        if m is None:
            continue
        before = m[m['dateutc'] < r.date]
        if not len(before):
            continue                                     # no data before the fee
        last = before.iloc[-1]
        if similarity(str(r.from_club), str(last['team'])) < MIN_CLUB_SIMILARITY:
            continue                                     # another club sold him
        gap = (r.date - last['dateutc']).days
        mid_season = r.date < season_end[last['seasonId']]
        if not mid_season and gap > SUMMER_WINDOW_DAYS:
            continue
        slug = ok.loc[r.playerId].get('tm_player_slug')
        slug = slug if isinstance(slug, str) and slug else '-'
        rows.append({
            'playerId': int(r.playerId), 'player_name': ok.loc[r.playerId].get('our_name'),
            'from_team': last['team'], 'to_team': r.to_club, 'transfer_type': 'permanent',
            'fee_eur': float(r.fee_eur), 'add_ons_eur': None, 'sell_on_pct': None,
            'as_of_date': r.date.strftime('%Y-%m-%d'), 'season_id': int(last['seasonId']),
            'contract_until': None, 'loan_until': None,
            'source_url': f"https://www.transfermarkt.com/{slug}/transfers/spieler/{int(r.tm_id)}",
            'notes': f"Transfermarkt transfer record: {r.from_club} to {r.to_club}; "
                     f"last match here {gap} days before",
            'synthetic_flag': 0, 'gap_days': int(gap), 'tm_from': r.from_club})
    out = pd.DataFrame(rows, columns=COLS + ['gap_days', 'tm_from'])
    # one row per sale even when the player sits under two of our ids
    try:
        from league_config import PLAYER_ID_ALIASES as aliases
    except Exception:
        aliases = {}
    out['_canon'] = out['playerId'].map(lambda p: aliases.get(int(p), int(p)))
    out = out.drop_duplicates(['_canon', 'as_of_date']).drop(columns='_canon')
    return out.sort_values(['as_of_date', 'playerId']).reset_index(drop=True)


def main():
    transfers = pd.read_parquet(HERE / 'tm_transfer_history.parquet')
    links = pd.read_csv(HERE / 'tm_player_links.csv')
    out = fee_rows(transfers, links, our_matches())
    out[COLS].to_csv(OUT, index=False)
    print(f'{len(out)} sales written to {OUT}')
    print(out[['player_name', 'as_of_date', 'from_team', 'to_team', 'fee_eur', 'gap_days']].to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
