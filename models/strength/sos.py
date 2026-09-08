"""Strength-of-schedule (SOS) adjustment for the team Attacking / Defending
Strength ratings shown in the Match Predictor's "Team Strength Ratings" table.

Pure pandas — no streamlit. ``app.calculate_sos_adjusted_strength`` is the
cached wrapper around :func:`sos_adjust`; tests import this module directly.

Conventions (from ``app.calculate_team_strength``)::

    Attacking Strength = 0.3 * GF/match + 0.7 * xGF/match   (higher = better)
    Defending Strength = 0.3 * GA/match + 0.7 * xGA/match   (LOWER  = better)

Direction — a team is CREDITED for a tough schedule::

    att_factor = league_avg_def / avg_opp_def   # > 1 when opponents conceded
                                                #     less than the league average
    def_factor = league_avg_att / avg_opp_att   # < 1 when opponents scored
                                                #     more than the league average
    sos_att = raw_att * att_factor
    sos_def = raw_def * def_factor

Opponent strength is each opponent's PRE-match cumulative strength from
``app.calculate_rolling_team_strength``; opponents with no prior match are
skipped, and the adjustment only switches on once a team has
``MIN_MATCHES_WITH_OPP_DATA`` such opponents (after matchweek 4 in a league
where everyone plays every week — and the factors are noisy for a while
after that, since each opponent's strength rests on a handful of games).

History: until 2026-09 both factors were inverted (opponents' conceded ÷
league, opponents' scored ÷ league), i.e. the adjustment REWARDED easy
schedules. After MW4 2026/27 that put Caldas (15th on raw Att − Def) above
Atlético CP and Mafra purely for having faced weak opponents.

``sos_factor`` summarises schedule difficulty: > 1 = tougher than average.

Both factors are clipped to ``FACTOR_BOUNDS`` (0.5–2.0): with only three
opponents counted, one that entered on a couple of clean sheets would
otherwise multiply a team's attack four-fold or more.
"""
import numpy as np
import pandas as pd

MIN_MATCHES_WITH_OPP_DATA = 3
_FLOOR = 0.01  # guards the league averages and per-team opponent means against 0
FACTOR_BOUNDS = (0.5, 2.0)  # cap on either multiplier (early-season noise guard)

RESULT_COLUMNS = [
    'raw_att', 'raw_def', 'avg_opp_att', 'avg_opp_def', 'matches_with_opp_data',
    'sos_att_factor', 'sos_def_factor', 'sos_att', 'sos_def', 'sos_factor',
]


def sos_adjust(rolling_strength_df, team_strength_df,
               min_matches=MIN_MATCHES_WITH_OPP_DATA):
    """SOS-adjust end-of-window team strengths.

    Parameters
    ----------
    rolling_strength_df : DataFrame with one row per (matchId, team) carrying
        that team's PRE-match ``att_strength`` / ``def_strength`` (NaN before
        its first match).
    team_strength_df : DataFrame indexed by team with ``Attacking Strength``
        and ``Defending Strength`` (the season totals being adjusted).

    Returns
    -------
    DataFrame indexed by team with :data:`RESULT_COLUMNS`; empty when either
    input is empty.
    """
    if (rolling_strength_df is None or team_strength_df is None
            or rolling_strength_df.empty or team_strength_df.empty):
        return pd.DataFrame()

    league_avg_att = max(team_strength_df['Attacking Strength'].mean(), _FLOOR)
    league_avg_def = max(team_strength_df['Defending Strength'].mean(), _FLOOR)

    # For each (matchId, team) find the other team in the same match and its
    # pre-match strength.
    match_teams = rolling_strength_df[['matchId', 'team', 'att_strength', 'def_strength']].copy()
    opp = match_teams.merge(match_teams, on='matchId', suffixes=('', '_opp'))
    opp = opp[opp['team'] != opp['team_opp']]

    opp_valid = opp.dropna(subset=['att_strength_opp', 'def_strength_opp'])
    avg_opp = opp_valid.groupby('team').agg(
        avg_opp_att=('att_strength_opp', 'mean'),
        avg_opp_def=('def_strength_opp', 'mean'),
        matches_with_opp_data=('att_strength_opp', 'count'),
    )

    result = team_strength_df[['Attacking Strength', 'Defending Strength']].copy()
    result.columns = ['raw_att', 'raw_def']
    result = result.join(avg_opp, how='left')

    # Credit tough schedules: stingy opponents (low avg_opp_def) scale the
    # attack UP; potent opponents (high avg_opp_att) scale conceded DOWN.
    lo, hi = FACTOR_BOUNDS
    result['sos_att_factor'] = (league_avg_def / result['avg_opp_def'].clip(lower=_FLOOR)).clip(lo, hi)
    result['sos_def_factor'] = (league_avg_att / result['avg_opp_att'].clip(lower=_FLOOR)).clip(lo, hi)

    has_enough = result['matches_with_opp_data'].fillna(0) >= min_matches
    result['sos_att'] = np.where(has_enough, result['raw_att'] * result['sos_att_factor'], result['raw_att'])
    result['sos_def'] = np.where(has_enough, result['raw_def'] * result['sos_def_factor'], result['raw_def'])
    # Schedule difficulty: both terms exceed 1 when opponents were better
    # than average (conceded less / scored more).
    result['sos_factor'] = np.where(
        has_enough,
        (result['sos_att_factor'] + 1.0 / result['sos_def_factor']) / 2,
        np.nan,
    )
    return result[RESULT_COLUMNS]
