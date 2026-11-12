"""Player Profile → Career: what Transfermarkt records for the player.

All of it Transfermarkt's own record (transfermarkt_links.py and
scrape_transfermarkt.py), kept apart from our exact minutes:

  clubs      every club he has been at and from when to when, built from
             his transfer history (youth sides and loans included)
  seasons    appearances and minutes per club-season, for the clubs and
             seasons we have linked (all competitions of that season)
  current    the club his profile lists now; the Player Information card
             shows it when he joined it after his last match in our data

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
METADATA = os.path.join(_VAL, 'tm_player_metadata.parquet')


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


@st.cache_data(show_spinner=False)
def _load_metadata(sig):
    return pd.read_parquet(METADATA) if os.path.exists(METADATA) else pd.DataFrame()


def metadata():
    return _load_metadata(_mtime(METADATA))


def _day(v):
    d = pd.to_datetime(v, errors='coerce')
    if pd.isna(d):
        return None
    return (d.tz_localize(None) if d.tzinfo is not None else d).normalize()


def _how(kind, fee, loan):
    kind = str(kind or '')
    if 'End of loan' in kind:
        return 'Back from loan'
    if loan is True or loan == 1 or 'oan' in kind:
        return 'Loan'
    if pd.notna(fee) and fee > 0:
        return f'Transfer, €{fee:,.0f}'
    return 'Free transfer' if 'free' in kind.lower() else ''


def clubs(transfers):
    """One row per spell at a club, newest first, from ONE player's transfer
    rows: a move opens a spell at `to_club` that lasts until his next move.
    The club he left in his first recorded move gets a spell with no start."""
    cols = ['From', 'Until', 'Club', 'Joined as']
    if transfers is None or not len(transfers):
        return pd.DataFrame(columns=cols)
    t = transfers.assign(_d=pd.to_datetime(transfers['date'], errors='coerce')).dropna(subset=['_d'])
    t = t.sort_values('_d', kind='stable').reset_index(drop=True)
    if not len(t):
        return pd.DataFrame(columns=cols)
    # moves dated the same day (a loan ending and the next move): the one
    # that leaves the club he is at comes first
    order, at = [], None
    for _, g in t.groupby('_d', sort=True):
        left = list(g.index)
        while left:
            nxt = next((i for i in left if at is not None and t.from_club[i] == at), left[0])
            left.remove(nxt)
            order.append(nxt)
            at = t.to_club[nxt]
    t = t.loc[order].reset_index(drop=True)
    fmt = lambda d: d.strftime('%Y-%m-%d')
    rows = [{'From': '', 'Until': fmt(t._d.iloc[0]), 'Club': t.from_club.iloc[0], 'Joined as': ''}]
    for i, r in t.iterrows():
        until = fmt(t._d.iloc[i + 1]) if i + 1 < len(t) else 'now'
        rows.append({'From': fmt(r._d), 'Until': until, 'Club': r.to_club,
                     'Joined as': _how(r.get('transfer_type'), r.get('fee_eur'), r.get('is_loan'))})
    out = pd.DataFrame(rows, columns=cols)
    out = out[out.Club.notna() & (out.Club.astype(str).str.strip() != '')]
    return out.iloc[::-1].reset_index(drop=True)


def current_club(player_id, links, meta):
    """What his Transfermarkt profile lists now, or None. Only from the profile
    of his sure link: {club, joined, on_loan_from, contract_until, read}."""
    row = links[(links.playerId == int(player_id)) & (links.status == 'ok')]
    if not len(row) or meta is None or not len(meta) or not {'playerId', 'tm_id', 'current_club'} <= set(meta.columns):
        return None
    m = meta[(meta.playerId == int(player_id))
             & (pd.to_numeric(meta.tm_id, errors='coerce') == int(row.iloc[0].tm_player_id))]
    if not len(m) or not isinstance(m.iloc[0].current_club, str) or not m.iloc[0].current_club.strip():
        return None
    m = m.iloc[0]
    loan = m.get('on_loan_from')
    return {'club': m.current_club.strip(), 'joined': _day(m.get('joined_date')),
            'on_loan_from': loan if isinstance(loan, str) and loan.strip() else None,
            'contract_until': _day(m.get('contract_until')), 'read': _day(m.get('fetched_at'))}


def team_now(our_team, last_match, cur):
    """(team for the Player Information card, note or None).

    Transfermarkt's club replaces ours only when he JOINED it after his last
    match in our data: then our team is where he was, not where he is. When
    he joined it before that match he was already there, so ours stands (the
    two names are the same club). With no joining date nothing is replaced."""
    if not cur:
        return our_team, None
    read = f", read {cur['read']:%d %b %Y}" if cur.get('read') is not None else ''
    last = _day(last_match)
    joined = cur.get('joined')
    if joined is None or (last is not None and joined <= last):
        return our_team, None
    club = {'without club': 'Without club', 'retired': 'Retired'}.get(cur['club'].lower(), cur['club'])
    loan = f", on loan from {cur['on_loan_from']}" if cur.get('on_loan_from') else ''
    was = f' In our data he last played for {our_team} on {last:%d %b %Y}.' if last is not None else ''
    if club in ('Without club', 'Retired'):
        what = f"Transfermarkt lists him as {club.lower()} since {joined:%d %b %Y}{read}."
    else:
        what = f"Team is his club on Transfermarkt{loan} since {joined:%d %b %Y}{read}."
    return club, what + was


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
    spells = pd.DataFrame()
    if link is not None and len(transfers) and {'playerId', 'tm_id'} <= set(transfers.columns):
        spells = clubs(transfers[(transfers.playerId == int(player_id))
                                 & (pd.to_numeric(transfers.tm_id, errors='coerce') == int(link.tm_player_id))])
    st.subheader('Career (Transfermarkt)')
    if link is None:
        st.info('No sure Transfermarkt link for this player yet. A player is linked when his date of birth '
                'matches one man in the Transfermarkt squad of the same club and season.')
        return
    if link is not None:
        st.caption(f'[Transfermarkt profile](https://www.transfermarkt.com/{link.tm_player_slug}/profil/spieler/'
                   f'{int(link.tm_player_id)}) · listed there as {link.tm_name}.')
        st.markdown('**Clubs**')
        if spells.empty:
            st.caption('His transfer history has not been read yet, so his full list of clubs is not available.')
        else:
            st.dataframe(spells, hide_index=True, use_container_width=True)
            st.caption('Every club in his Transfermarkt transfer history, newest first, youth sides and loans '
                       'included. Dates are the dates of the moves; Transfermarkt often records a summer move '
                       'as 1 July.')
        st.markdown('**Seasons at clubs in our data**')
        if sea.empty:
            st.caption('No season statistics read for him yet.')
        else:
            st.dataframe(sea, hide_index=True, use_container_width=True)
            st.caption("Transfermarkt's own count for that club and season, all competitions (league, cups). It "
                       'credits a full match as 90 minutes, so it runs a little under our exact minutes for the '
                       'same matches and over them where cup matches are included. Seasons at clubs we have '
                       'not linked are not listed.')
