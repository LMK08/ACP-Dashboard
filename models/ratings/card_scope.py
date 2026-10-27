"""Which engine row the ACP Index card shows for a player in a scope.

Pure pandas — no streamlit. The Player Profile's ACP Index card and the
Player Comparison page both read it, so the two pages can never show a
player with different ratings for the same scope (it was inline in
views/player_profile.py until 2026-10).
"""
import numpy as np
import pandas as pd

# Columns the All-Seasons career view aggregates minutes-weighted.
CAREER_WEIGHTED_COLS = ['off_pct', 'qual_pct', 'rapm_pct', 'defr_pct',
                        'datt_pct', 'setpiece_pct', 'aerial_grade_pct',
                        'ground_grade_pct', 'Shooting_pct', 'Creating_pct',
                        'Linking_pct', 'Receiving_pct', 'Dribbling_pct',
                        'career_asof', 'w_evidence']
PROJECTION_COLS = ('projection', 'projection_abs', 'band_sd', 'proj_delta', 'seasons_ago')


def season_id_list(active_season_ids):
    """active_season_ids can be None (All Seasons), an int, or a list."""
    if isinstance(active_season_ids, (list, tuple, set)):
        return [int(s) for s in active_season_ids if s is not None]
    if active_season_ids is not None:
        return [int(active_season_ids)]
    return None


def card_row(eng_df, player_id, active_season_ids):
    """Resolve the card's engine row for ``player_id`` in a scope.

    Returns a dict:
      erows       — every engine row of the player (all seasons, leagues)
      escope      — the rows the scope covers (falls back to the last rated
                    season when the scope has none — ``stale``)
      e           — the card row (Series) or None when the player is unrated
      e_sids      — the scope's season ids (None = All Seasons)
      career_view — All Seasons with more than one rated season: rating and
                    every percentile / grade minutes-weighted across them;
                    mins_played = the career total
      stale       — not rated in the scope; showing the last rated season

    Row pick: the scope's highest-minutes row. The projection is a single
    player-level forward look anchored to the player's most recent season,
    and seasonId is NOT chronological across leagues (Camp 23/24 = 190230 >
    L3 24/25 = 190090) — so it is always taken from the row that actually
    carries one (Lucas 2026-06-24).
    """
    erows = (eng_df[eng_df['playerId'] == int(player_id)]
             if eng_df is not None and not eng_df.empty else pd.DataFrame())
    e_sids = season_id_list(active_season_ids)
    career_view = False
    if erows.empty:
        escope = erows
    elif e_sids:
        escope = erows[erows['seasonId'].isin(e_sids)]
    else:   # All Seasons → CAREER view: aggregate every rated season
        escope = erows
        career_view = len(erows) > 1
    stale = False
    if escope.empty and not erows.empty:
        escope = erows[erows['seasonId'] == erows['seasonId'].max()]
        stale = True
    out = dict(erows=erows, escope=escope, e=None, e_sids=e_sids,
               career_view=career_view, stale=stale)
    if escope.empty:
        return out

    e = escope.sort_values('mins_played').iloc[-1]
    if career_view:
        # representative row = highest-minutes season (role / league /
        # shares); rating + every percentile / grade minutes-weighted
        e = e.copy()
        wts = pd.to_numeric(escope['mins_played'], errors='coerce').fillna(0.0).to_numpy()
        for c in CAREER_WEIGHTED_COLS:
            if c in escope.columns and wts.sum() > 0:
                v = pd.to_numeric(escope[c], errors='coerce').to_numpy()
                m = ~np.isnan(v)
                if m.any():
                    e[c] = float(np.average(v[m], weights=wts[m]))
        if pd.notna(e.get('acp_rating_career')):
            e['acp_rating'] = float(e['acp_rating_career'])
        e['mins_played'] = float(wts.sum())
        # projection = the LATEST season's forward look; lineup minutes =
        # career total (for the radar header)
        latest = escope.sort_values('seasonId').iloc[-1]
        for c in ('projection', 'projection_abs', 'band_sd',
                  'proj_delta', 'seasons_ago', 'age'):
            if c in latest:
                e[c] = latest[c]
        if 'mins_lineup' in escope.columns:
            e['mins_lineup'] = float(pd.to_numeric(
                escope['mins_lineup'], errors='coerce').fillna(0.0).sum())
    proj_src = erows[erows['projection'].notna()]
    if not proj_src.empty:
        pr = proj_src.sort_values('seasonId').iloc[-1]
        for c in PROJECTION_COLS:
            if c in pr.index:
                e[c] = pr[c]
    out['e'] = e
    return out
