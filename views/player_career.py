"""Player Profile → Career: what Transfermarkt records for the player.

Two tables, both Transfermarkt's own figures (transfermarkt_links.py and
scrape_transfermarkt.py), kept apart from our exact minutes:

  seasons    appearances and minutes per club-season, for the clubs and
             seasons we have linked (all competitions of that season)
  transfers  his moves between clubs, where the transfer history was read

A player is shown only when his link is sure (same club, same season, same
date of birth as one man in that squad).
"""
import os

import pandas as pd
import streamlit as st

_VAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'valuations')
LINKS = os.path.join(_VAL, 'tm_player_links.csv')
SEASONS = os.path.join(_VAL, 'tm_player_club_seasons.parquet')
TRANSFERS = os.path.join(_VAL, 'tm_transfer_history.parquet')


def _mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


@st.cache_data(show_spinner=False)
def _load(sig):          # `sig` (the three files' modification times) IS the cache key: no leading underscore
    links = pd.read_csv(LINKS) if os.path.exists(LINKS) else pd.DataFrame(columns=['playerId', 'status'])
    seasons = pd.read_parquet(SEASONS) if os.path.exists(SEASONS) else pd.DataFrame()
    transfers = pd.read_parquet(TRANSFERS) if os.path.exists(TRANSFERS) else pd.DataFrame()
    return links, seasons, transfers


def tables():
    return _load((_mtime(LINKS), _mtime(SEASONS), _mtime(TRANSFERS)))


def career(player_id, links, seasons, transfers):
    """(link row or None, seasons frame, transfers frame) for one player."""
    row = links[(links.playerId == int(player_id)) & (links.status == 'ok')]
    link = row.iloc[0] if len(row) else None
    sea = pd.DataFrame()
    if link is not None and len(seasons):
        s = seasons[seasons.tm_player_id == int(link.tm_player_id)].sort_values(['year', 'minutes'], ascending=False)
        s = s.drop_duplicates(['tm_team_id', 'year']) if 'tm_team_id' in s.columns else s

        def col(name):            # a column Transfermarkt's page did not have is shown blank, not a crash
            return s[name].to_numpy() if name in s.columns else [None] * len(s)
        sea = pd.DataFrame({
            'Season': [f'{int(y)}/{str(int(y) + 1)[-2:]}' for y in s.year], 'Club': col('team'),
            'In squad': col('in_squad'), 'Appearances': col('appearances'),
            'Goals': col('goals'), 'Assists': col('assists'), 'Minutes': col('minutes')})
    tr = pd.DataFrame()
    # only transfers read from the SAME profile as the sure link: the older
    # name-search links (tm_match_log.csv) put some players on a namesake's
    # profile (140 found 2026-11, every one with a different date of birth)
    if link is not None and len(transfers) and {'playerId', 'tm_id'} <= set(transfers.columns):
        t = transfers[(transfers.playerId == int(player_id))
                      & (pd.to_numeric(transfers.tm_id, errors='coerce') == int(link.tm_player_id))]
        t = t.sort_values('date', ascending=False)
        if len(t):
            fee = t.fee_eur.map(lambda v: '' if pd.isna(v) else f'€{v:,.0f}')
            tr = pd.DataFrame({'Date': t.date.astype(str).str[:10].to_numpy(), 'Season': t.season.to_numpy(),
                               'From': t.from_club.to_numpy(), 'To': t.to_club.to_numpy(),
                               'Loan': t.is_loan.map(lambda v: 'yes' if v is True or v == 1 else '').to_numpy(),
                               'Fee': fee.to_numpy()})
    return link, sea, tr


def render(player_id):
    links, seasons, transfers = tables()
    link, sea, tr = career(player_id, links, seasons, transfers)
    st.subheader('Career (Transfermarkt)')
    if link is None:
        st.info('No sure Transfermarkt link for this player yet. A player is linked when his date of birth '
                'matches one man in the Transfermarkt squad of the same club and season.')
        return
    if link is not None:
        st.caption(f'[Transfermarkt profile](https://www.transfermarkt.com/{link.tm_player_slug}/profil/spieler/'
                   f'{int(link.tm_player_id)}) · listed there as {link.tm_name}.')
        st.markdown('**Seasons at clubs in our data**')
        if sea.empty:
            st.caption('No season statistics read for him yet.')
        else:
            st.dataframe(sea, hide_index=True, use_container_width=True)
            st.caption("Transfermarkt's own count for that club and season, all competitions (league, cups). It "
                       'credits a full match as 90 minutes, so it runs a little under our exact minutes for the '
                       'same matches and over them where cup matches are included. Seasons at clubs we have '
                       'not linked are not listed.')
    if not tr.empty:
        st.markdown('**Transfers**')
        st.dataframe(tr, hide_index=True, use_container_width=True)
        st.caption('Moves recorded on Transfermarkt, newest first. Not every player has had his history read.')
