"""TODAY's projection per player (2026-11-12) — what the analysis boards and
tables show for a scope holding the season in progress, or All Seasons.

Lucas: the live projection should mean "how good a player will be moving
forward (including for the rest of the current season)" rather than next
season. build_projection writes each live row's 12-month FORWARD projection
(projection_fwd: the shipped predictor with half a year of the age step;
engine columns fwd / fwd_abs). A player's projection today is that forward
projection once his club has played enough league matches (engine live_ok,
the live value's switch), else his summer projection — the projection of the
row the app's row pick takes (passed in, so the pick rule stays in one place).

A past season keeps the projection made at its end. Pure pandas;
app.load_player_engine calls attach(), app.merge_engine_values_into_stats
reads the columns when includes_live_season(scope).
"""
import numpy as np
import pandas as pd

NOW_COLS = ('projection_now', 'projection_now_abs', 'band_now', 'projection_now_live')
_NEEDS = {'playerId', 'mins_played', 'fwd', 'fwd_abs', 'live_ok'}


def attach(df, summer_pick):
    """df plus projection_now / projection_now_abs / band_now /
    projection_now_live on EVERY row of each player.

    summer_pick: one row per player indexed by playerId with projection /
    projection_abs / band_sd (the app's row pick). A player with a live_ok
    row carrying fwd_abs takes that row's forward projection (the higher-
    minutes live row for a winter mover) and its band (fwd_band: today's
    role band, the summer projection's — conservative for 12 months; NaN on
    an engine without it); everyone else
    takes the summer pick's projection and band (NaN when he has none).
    An engine file without fwd
    (built before 2026-11-12) is returned unchanged, so a code deploy that
    lands before the engine rebuild never labels a summer projection live.
    """
    if not _NEEDS <= set(df.columns):
        return df
    out = df.copy()
    pid = out['playerId']
    out['projection_now'] = pid.map(summer_pick['projection']).astype(float)
    out['projection_now_abs'] = pid.map(summer_pick['projection_abs']).astype(float)
    out['band_now'] = pid.map(summer_pick['band_sd']).astype(float)
    out['projection_now_live'] = False
    fw = (out[out['live_ok'].fillna(False).astype(bool) & out['fwd_abs'].notna()]
          .sort_values('mins_played', kind='stable')
          .drop_duplicates('playerId', keep='last').set_index('playerId'))
    has = pid.isin(fw.index).fillna(False).to_numpy(dtype=bool)
    if has.any():
        p = pid[has]
        out.loc[has, 'projection_now'] = p.map(fw['fwd']).to_numpy(dtype=float)
        out.loc[has, 'projection_now_abs'] = p.map(fw['fwd_abs']).to_numpy(dtype=float)
        out.loc[has, 'band_now'] = (p.map(fw['fwd_band']).to_numpy(dtype=float)
                                    if 'fwd_band' in fw.columns else np.nan)
        out.loc[has, 'projection_now_live'] = True
    return out


def _sids(season_ids):
    s = (season_ids if isinstance(season_ids, (list, tuple, set, frozenset, pd.Series, np.ndarray))
         else [season_ids])
    return [int(x) for x in s if x is not None and pd.notna(x)]


def includes_live_season(season_ids, in_progress=None):
    """True for All Seasons (None) or a scope holding a season in progress —
    the scopes that show today's projection."""
    if season_ids is None:
        return True
    if in_progress is None:
        from league_config import IN_PROGRESS_SEASON_IDS as in_progress
    return any(s in in_progress for s in _sids(season_ids))


def latest_completed_seasons(competitions=None, in_progress=None, season_year=None):
    """{seasonId}: each competition's newest season that is not in progress —
    the season whose rows carry every player's summer projection."""
    if competitions is None or in_progress is None or season_year is None:
        import league_config as lc
        competitions = lc.COMPETITIONS if competitions is None else competitions
        in_progress = lc.IN_PROGRESS_SEASON_IDS if in_progress is None else in_progress
        season_year = lc.SEASON_YEAR if season_year is None else season_year
    out = set()
    for comp in competitions.values():
        done = [int(s) for s in comp['seasons'] if int(s) not in in_progress and int(s) in season_year]
        if done:
            out.add(max(done, key=lambda s: season_year[s]))
    return out


def projection_is_headline(season_ids, in_progress=None, latest_completed=None):
    """Whether a scope's boards rank by projection: All Seasons, a scope
    holding the season in progress (today's projection) or only latest
    completed seasons (their summer projection covers every player). An
    older season's rows carry a projection only for players last seen then
    (35-45% of them), so its boards rank by rating."""
    if includes_live_season(season_ids, in_progress):
        return True
    if latest_completed is None:
        latest_completed = latest_completed_seasons(in_progress=in_progress)
    sids = _sids(season_ids)
    return bool(sids) and all(s in latest_completed for s in sids)
