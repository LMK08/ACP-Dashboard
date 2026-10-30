#!/usr/bin/env python3
"""ACP Rating v5 — the production player rating.

v2 adds the DUEL component (Glicko ladders, models/duels/): a player's
trait ratings (aerial/takeon/stopper/shield/press, >=30 contests each)
percentiled within role x league and volume-weighted into one duel
score. Career-level trait merged on playerId, like RAPM. Weights
rebalanced; duel quality now enters via both DWAE (season-level,
static-expectation) and the ladders (career-level, opponent-adjusted).

Architecture settled by the GPA audit + ratings harness + gated RAPM:
  - GPA possession value is the on-ball ENGINE (use Total OFFENSIVE
    Value — the clean part; GPA's defensive 'Interrupting' is near-noise
    at the player level, YoY r~0.07, so it is NOT the defensive axis).
  - Defence enters OUTSIDE the engine via DefR (workload above role
    expectation) + DWAE (contested-duel quality) — both validated.
  - RAPM v3 is a small "intangibles" nudge (off-ball impact; passed its
    gates; near-orthogonal to GPA/DefR).
  - Everything is normalised within OBSERVED ROLE x LEAGUE (Futi:
    "scaled by tactical role") so a CB and a striker are comparable.
  - Single-season percentiles are noisy (harness), so the headline
    rating is minutes-shrunk and a recency-weighted CAREER rating is the
    trustworthy scouting number.
  - Bespoke template scores are NOT in the rating (demoted to role-fit).

Blend (role-percentiles, 0-1, weights sum to 1):
    0.50 offence (Total Offensive Value)
    0.20 DefR workload (defr_adj)
    0.15 DWAE quality (defr_dwae_p90)
    0.15 RAPM v3 intangibles
Weights are deliberately fixed + documented (v2 candidate: role-specific
weighting). Role-percentile means a striker's defensive axes compare him
to other strikers, so low-defending roles aren't penalised — only
players unusual FOR THEIR ROLE move on those axes.

Outputs models/ratings/acp_rating_per_player_season.parquet:
    acp_rating            single-season, minutes-shrunk, 0-100
    acp_rating_career     recency-weighted across seasons, 0-100
    off_pct/defr_pct/dwae_pct/rapm_pct  the components (0-1, role-fair)
    role, side, n_seasons

Run from the Dashboard dir: python models/ratings/build_rating.py
"""
from pathlib import Path
import sys as _sys_lc
_sys_lc.path.insert(0, str(Path(__file__).resolve().parents[2]))
from league_config import SEASON_YEAR, CAMP_SEASON_IDS, IN_PROGRESS_SEASON_IDS  # noqa: E402
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

_HERE = Path(__file__).resolve().parent
_DASH = _HERE.parent.parent
SEASON_YR = SEASON_YEAR   # from league_config (every season, incl. the current one)

# v7.0 lookahead test hook (tests/test_rating_no_lookahead.py): ACP_MAX_YEAR
# drops every season after that year from every input, ACP_ENGINE_OUT
# redirects the output. A rating with no lookahead is identical for every
# season <= ACP_MAX_YEAR whether or not the later seasons exist.
import os as _os
_MAX_YR = int(_os.environ['ACP_MAX_YEAR']) if _os.environ.get('ACP_MAX_YEAR') else None
_OUT = Path(_os.environ.get('ACP_ENGINE_OUT', str(_HERE)))


def _read(path, **kw):
    """read_parquet, truncated to seasons <= ACP_MAX_YEAR when it is set."""
    frame = pd.read_parquet(path, **kw)
    if _MAX_YR is None:
        return frame
    yr = pd.to_numeric(frame['seasonId'], errors='coerce').map(SEASON_YEAR)
    return frame[yr <= _MAX_YR].copy()
CAMP = set(CAMP_SEASON_IDS)
# v3 ROLE-SPECIFIC weights over 5 axes (Lucas-confirmed design):
#   off  = GPA Total Offensive (role-pct)
#   resp = DefR — defensive RESPONSIBILITY ABOVE EXPECTATION (not raw
#           volume: presence-conditioned expected vs actual)
#   qual = defensive quality, MERGED dwae+defensive-ladders (r=0.43
#           overlap — two measurements of one skill, one weight)
#   datt = take-on/shield ladders. NOT redundant with off: within-role
#           r = -0.05..+0.11 — press resistance & 1v1 are skills the
#           possession-value engine doesn't capture. Small weight.
#   rapm = intangibles.
# Quality > responsibility in EVERY role (Lucas: doing it well beats
# doing lots of it).
# v4 (Lucas): qual ~3x resp everywhere; datt trimmed to 0.05 flat; rapm
# HELD at 0.10 — raising a career-level one-number trait mechanically
# inflates measured YoY (same value both seasons) without adding truth,
# and RAPM's own gates passed only marginally (split-half 0.31).
# off weights HELD at v3 levels (off is the noisiest season axis — the
# first v4 draft raised them and rating YoY fell 0.41->0.30; reverted);
# the weight freed from resp/datt goes to QUAL per Lucas.
# v5.2 post outcome-audit (commit 7f8ce5d): resp KEPT small (micro-
# anchored, YoY 0.41, orthogonal to qual — the team-level null is the
# test's blindness, not the metric's); qual UP (cleanest defensive
# outcome link, -0.38 xGA; ceiling acknowledged — it only measures
# contested defence); datt UP for ball-playing roles (links both ends
# +0.38 xGF/-0.21 xGA, measured non-overlap with off); rapm HELD 0.10
# (gate-earned; split-half 0.31 caps it).
# v6.5 (Lucas): Off +0.075 for every role — on-ball value is the
# engine's best-validated axis and should carry more of the headline.
# Taken from qual (and a sliver of datt for attackers); rapm flat 0.10.
ROLE_WEIGHTS = {
    'Striker':              {'off': 0.60,  'resp': 0.05,  'qual': 0.20,  'datt': 0.05,  'rapm': 0.10},
    'Wide Attacker':        {'off': 0.60,  'resp': 0.05,  'qual': 0.20,  'datt': 0.05,  'rapm': 0.10},
    'Advanced Midfielder':  {'off': 0.575, 'resp': 0.05,  'qual': 0.225, 'datt': 0.05,  'rapm': 0.10},
    'Deep Midfielder':      {'off': 0.525, 'resp': 0.075, 'qual': 0.25,  'datt': 0.05,  'rapm': 0.10},
    'Wide Defender':        {'off': 0.475, 'resp': 0.10,  'qual': 0.275, 'datt': 0.05,  'rapm': 0.10},
    'Central Defender':     {'off': 0.425, 'resp': 0.10,  'qual': 0.325, 'datt': 0.05,  'rapm': 0.10},
}
DEFAULT_W = {'off': 0.525, 'resp': 0.075, 'qual': 0.25, 'datt': 0.05, 'rapm': 0.10}

# v6.5 (Lucas): ROLE-RELEVANCE multipliers inside the Off axis. The
# pure reliability x variance weighting drifted from what each role is
# FOR (striker Shooting carried 11% of striker offence; DM Shooting
# 39%). Relevance re-anchors influence to role needs while the lambda
# reliability shrink keeps damping the noisy categories — influence is
# now relevance x reliability x variance. Multipliers kept modest
# (0.5-1.5) so reliability still dominates.
# v6.8 (Lucas): relevance grid informed by the transfer matrix (does a
# category deviation predict NEXT-season offence EXCLUDING itself?).
# Style traits (high self-persistence, ~zero transfer: WA Dribbling
# 0.53/-0.01, AM/DM Creating, DM Shooting) get no amplification;
# quality markers (AM Receiving 0.45/+0.36 — strongest in the matrix;
# DM Linking/Dribbling/Receiving; WD Receiving) get the boosts.
# lambda keeps doing reliability shrinkage; relevance stays hand-set,
# bounded 0.4-1.7, moved only where evidence and football logic agree.
ROLE_CAT_RELEVANCE = {
    'Striker':             {'Shooting': 1.5,  'Receiving': 1.25, 'Creating': 1.0,  'Dribbling': 0.85, 'Linking': 0.7},
    'Wide Attacker':       {'Shooting': 1.15, 'Receiving': 1.25, 'Creating': 1.25, 'Dribbling': 1.0,  'Linking': 0.7},
    'Advanced Midfielder': {'Shooting': 1.05, 'Receiving': 1.3,  'Creating': 1.3,  'Dribbling': 0.9,  'Linking': 0.9},
    'Deep Midfielder':     {'Shooting': 0.4,  'Receiving': 1.0,  'Creating': 1.05, 'Dribbling': 1.0,  'Linking': 1.7},
    'Wide Defender':       {'Shooting': 0.6,  'Receiving': 1.1,  'Creating': 1.25, 'Dribbling': 1.0,  'Linking': 1.2},
    'Central Defender':    {'Shooting': 0.6,  'Receiving': 1.0,  'Creating': 0.9,  'Dribbling': 0.8,  'Linking': 1.5},
}
# v4 offence axis: reliability x relevance weighted GPA CATEGORY blend.
# Measured (2026-06): 53% of striker offence is Shooting Value at YoY
# 0.09 — finishing variance, not skill; receiving/dribbling/set-piece
# craft repeat far better. Category weight = value-share x max(YoY,.05),
# estimated per role on all seasons (meta-parameters, noted in-sample).
# v4.1: big-4 + ONE merged Dead-Ball category. Separately the four
# dead-ball cats are zero-inflated (percentile noise -> excluded in the
# first v4), but exclusion DELETED real skill: dead-ball delivery is
# 14-26% of wide/AM role offence (Joao Pais: 51%!) and the MOST
# repeatable skill measured (corners YoY 0.44-0.56). Merging the four
# into one category concentrates the non-zero mass.
# v6.0 UNITS FIX: the dashboard's 'X Value' columns are ALREADY per-90
# (caught by the pass-split sanity check, 2026-06-12) — this builder had
# been dividing them by minutes again since v1, ranking players on
# per-90/minutes and systematically understating high-minute players'
# offence. Now: RAW season sums loaded, per-90 computed exactly once.
# Passing additionally split into CREATING (final-third/box/cross/
# through/shot-assist passes) vs LINKING (futi adoption #2), from
# pass_split.parquet (raw action-value sums, set-piece passes excluded).
GPA_RAW_CATS = ['Shooting', 'Receiving', 'Dribbling',
                 'SetPiece', 'Corner', 'FreeKick', 'ThrowIn']
# v5.1 (Lucas): dead-ball SEPARATED from the overall rating — it is a
# specialist skill the club wants visible as its own score, not folded
# into the headline. Offence axis = big-4 open-play value only;
# setpiece_pct exported alongside.
GPA_CATS = ['Shooting', 'Creating', 'Linking', 'Receiving', 'Dribbling']
SHRINK_K = 300.0          # residual blend shrink, renormalized to full season (v5.4)
CAREER_DECAY = 0.5        # recency weight = mins x 0.5^(seasons_back)
MIN_MINS = 500            # percentile COHORT floor — defines the reference distribution
MIN_MINS_ELIG = 90        # rating ELIGIBILITY floor (Lucas 2026-06): rate sub-cohort
                          # players too. They are scored against the >=MIN_MINS cohort
                          # and minutes-shrunk as usual (a 90' player keeps ~26% of his
                          # z, pulled toward replacement) — low minutes are penalised by
                          # the shrink, not by exclusion. Cohort stays >=MIN_MINS so the
                          # reference scale is unmoved.

print("[1/4] assemble components…", flush=True)
g = _read(_DASH / 'gpa_player_season_values.parquet',
                      columns=['playerId', 'seasonId', 'name', 'position_group',
                                'mins_played', 'Total Offensive Value'] + GPA_RAW_CATS)
g['DeadBall'] = (g['SetPiece'] + g['Corner'] + g['FreeKick'] + g['ThrowIn'])
g['playerId'] = pd.to_numeric(g['playerId'], errors='coerce').astype('Int64')
g['seasonId'] = pd.to_numeric(g['seasonId'], errors='coerce').astype('Int64')
ps = _read(_HERE / 'pass_split.parquet')
g = g.merge(ps.rename(columns={'Creating Value': 'Creating',
                                  'Linking Value': 'Linking'}),
              on=['playerId', 'seasonId'], how='left')
g[['Creating', 'Linking']] = g[['Creating', 'Linking']].fillna(0.0)
d = _read(_DASH / 'models/defr/defr_per_player_season.parquet',
                      columns=['playerId', 'seasonId', 'defr_adj', 'defr_dwae',
                                'dwae_n'])
r = _read(_DASH / 'models/roles/role_assignments_season.parquet',
                      columns=['playerId', 'seasonId', 'primary_role_name', 'side',
                                'primary_role'] + [f'role_share_{k}' for k in range(6)])
_idn = r.dropna(subset=['primary_role', 'primary_role_name']).drop_duplicates('primary_role')
ROLE_ID2NAME = dict(zip(_idn['primary_role'].astype(int), _idn['primary_role_name']))
# v6.6 (Lucas): RAPM per (player, season) — leak-free walk-forward with
# 0.7/yr decay (rapm_v4.py); replaces the single career coefficient
rapm = _read(_HERE / 'rapm_v4_per_season.parquet')
# v7.0 NO LOOKAHEAD: the Glicko duel ladders (models/duels/duel_ratings.parquet)
# are no longer read here. They are ONE career rating per player per trait
# (as of the latest contest), and v6.9 mapped them onto every season row —
# a 2021/22 rating carried duels from every later season. Every duel input
# below is the season's own contests with as-of expectations.

df = (g.merge(d, on=['playerId', 'seasonId'], how='left')
        .merge(r, on=['playerId', 'seasonId'], how='left')
        .merge(rapm, on=['playerId', 'seasonId'], how='left'))
df['role'] = df['primary_role_name'].fillna(df['position_group'])
df['league'] = np.where(df['seasonId'].isin(CAMP), 'CAMP', 'L3')
df['yr'] = df['seasonId'].map(SEASON_YR)
# v6.4 (Lucas): eligibility pools minutes ACROSS competitions within a
# season — a winter mover (e.g. 558' Camp + 391' L3) has a full season
# of evidence and deserves a row in each league he played. Percentile
# cohorts are still built from >=MIN_MINS rows only; sub-floor rows are
# scored against those distributions, then minutes-shrunk as usual.
MIN_MINS_ROW = 90         # per-row floor, lowered 180->90 (Lucas 2026-06) to admit
                          # single-competition sub-180' rows; matches MIN_MINS_ELIG
_season_mins = df.groupby(['playerId', 'yr'])['mins_played'].transform('sum')
df = df[(_season_mins >= MIN_MINS_ELIG) & (df['mins_played'] >= MIN_MINS_ROW)
          & (df['position_group'] != 'GK') & df['role'].notna()].copy()
df['_cohort'] = df['mins_played'] >= MIN_MINS
# v6.3 ROLE BLENDING (Lucas): median primary-role share is 0.70 — 63% of
# player-seasons spend >20% of matches in another role. Percentiles are
# share-weighted blends across the role cohorts actually occupied; axis
# weights and the age curve blend the same way.
ROLE_NAMES = [ROLE_ID2NAME[k] for k in range(6)]
for k in range(6):
    df['sh_' + ROLE_ID2NAME[k]] = df[f'role_share_{k}'].fillna(0.0)
_shsum = sum(df['sh_' + nm] for nm in ROLE_NAMES)
_fallback = _shsum < 0.5
ROLE_UNIVERSE = list(ROLE_NAMES)
for lbl in df.loc[_fallback, 'role'].dropna().unique():
    col = 'sh_' + str(lbl)
    if col not in df.columns:
        df[col] = 0.0
        ROLE_UNIVERSE.append(str(lbl))
    df.loc[_fallback & (df['role'] == lbl), col] = 1.0
_shsum = sum(df['sh_' + nm] for nm in ROLE_UNIVERSE).replace(0, 1)
for nm in ROLE_UNIVERSE:
    df['sh_' + nm] = df['sh_' + nm] / _shsum
print(f"  role blending: {int((~_fallback).sum()):,} share-based, "
       f"{int(_fallback.sum()):,} one-hot fallback")
print(f"  {len(df):,} eligible player-seasons (outfield; floor {MIN_MINS_ELIG}'; "
       f"{int((~df['_cohort']).sum())} sub-{MIN_MINS}' rows scored vs the cohort + shrunk)")


def role_pct(col):
    """v6.4 share-weighted blended percentile: per (league, season, role)
    a weighted ECDF (weights = role shares) built from COHORT rows
    (>= MIN_MINS) only; every valid row — including sub-floor rows
    admitted by season pooling — is scored against it by midpoint
    interpolation. Ties share a midpoint pct. NaN inputs -> 0.5."""
    vals = df[col].to_numpy(float)
    out = np.zeros(len(df))
    wtot = np.zeros(len(df))
    coh = df['_cohort'].to_numpy(bool)
    for _, gidx in df.groupby(['league', 'seasonId']).groups.items():
        pos = df.index.get_indexer(np.asarray(list(gidx)))
        v = vals[pos]
        valid = ~np.isnan(v)
        for rn in ROLE_UNIVERSE:
            w = df['sh_' + rn].to_numpy(float)[pos]
            ms = valid & (w > 1e-9)            # scored: every valid row
            mc = ms & coh[pos]                  # cohort: full rows only
            if mc.sum() < 3:
                continue
            ci = np.flatnonzero(mc)
            order = np.argsort(v[ci], kind='stable')
            cv = v[ci][order]
            cum = np.cumsum(w[ci][order])
            si = np.flatnonzero(ms)
            lo = np.searchsorted(cv, v[si], side='left')
            hi = np.searchsorted(cv, v[si], side='right')
            wlo = np.where(lo > 0, cum[lo - 1], 0.0)
            whi = np.where(hi > 0, cum[hi - 1], 0.0)
            pct = (wlo + (whi - wlo) / 2.0) / cum[-1]
            out[pos[si]] += w[si] * pct
            wtot[pos[si]] += w[si]
    res = np.where(wtot > 0, out / np.where(wtot > 0, wtot, 1), 0.5)
    res = np.where(np.isnan(vals), 0.5, res)
    return pd.Series(res, index=df.index)


# --- v4 offence: category blend ------------------------------------------
from scipy.stats import pearsonr as _pr
for c in GPA_CATS:
    df[c + '90'] = df[c] / df['mins_played'] * 90
    df[c + '_pct'] = role_pct(c + '90')
# v5 offence: combine in VALUE space, not percentile space. GPA
# categories are all in goals, so they SUM — percentile-averaging is
# ordinal arithmetic that caps specialists (the root cause of the
# v4 share-weighting machinery, now deleted). Each category deviation
# from the cohort mean is shrunk by its per-role rank reliability
# (regression-to-mean: best estimate of true value = r x observed),
# then summed in goals/90 and percentiled ONCE for display.
#
# TESTED AND REJECTED 2026-06-30 (Lucas + audit): z-scored offence variants.
#   (a) rawz — z the combined off_blend within role cohort (skew-preserving,
#       cap ±4σ, z-space minutes shrink): rating YoY 0.436 -> 0.459.
#   (b) catz — z EACH CATEGORY value first, then the λ·relevance sum:
#       YoY 0.469 (held in both leagues, on 24->25 only, and team-switch
#       0.501 -> 0.524; share-blended variant equivalent, 0.467).
# REJECTED because the audit showed the YoY gain is a WEIGHT-SHIFT artifact,
# not truth: z-ing each category strips the sd(dev) term from influence
# (rel·λ·sd -> rel·λ), which crushed striker Shooting 17% -> 6% of the off
# axis — mechanically re-walking the v6.5 "role fidelity over reliability"
# trade backwards (rel=1.5 was hand-set to RAISE shooting to ~17%). External
# validity (rating_T -> next-season raw off value/90, attackers, n=340) was
# NEUTRAL: pct 0.090 vs catz 0.081, rawz 0.060 — no outcome-level gain, so
# per the v6.5 decision framework it does not ship. Fourth member of the
# YoY-flattering artifact family (pooled stability / mechanical career-trait
# / smoothed-target / weight-shift). IF REVISITED at the 26/27 refit: catz
# with the relevance grid RE-CALIBRATED in z-space to restore the intended
# influence shares, gated on external validity, not YoY.
_off_adj = pd.Series(0.0, index=df.index)
_lam_log = []
for role, sub in df.groupby('role'):
    for c in GPA_CATS:
        P = []
        # reliability from COMPLETED seasons only: half-season pairs into a
        # season still in progress would move every season's weights each
        # matchday (found when 2026/27 was first rated, 2026-10-21)
        _done = sub['_cohort'] & ~sub['seasonId'].isin(IN_PROGRESS_SEASON_IDS)
        # a winter mover has two rows in one year (one per league); pairs use
        # his MAIN row (most minutes). v6.9 paired whichever an unstable sort
        # put first, so LATER seasons' rows could decide it (caught by the
        # lookahead test)
        _main = (sub[_done].sort_values(['yr', 'mins_played', 'seasonId'], kind='stable')
                 .groupby(['playerId', 'yr']).tail(1))
        for pid, gg in _main.sort_values('yr', kind='stable').groupby('playerId'):
            rr = gg[[c + '_pct', 'yr']].to_dict('records')
            for a, b in zip(rr, rr[1:]):
                if b['yr'] - a['yr'] == 1:
                    P.append((a[c + '_pct'], b[c + '_pct'], b['yr']))
        P = pd.DataFrame(P, columns=['a', 'b', 'yr'])
        # v7.0 NO LOOKAHEAD: reliability AS-OF each season-year — a row is
        # weighted with pairs ending in its own season or earlier only (v6.9
        # pooled every completed season, so 2021/22 offence was weighted
        # with reliabilities measured on 2022-26). < 30 pairs -> the 0.15
        # prior, as before. The current season is unchanged (all its prior
        # pairs are complete seasons either way).
        lam_y = {}
        for y in sorted(sub['yr'].unique()):
            Py = P[P['yr'] <= y]
            lam = _pr(Py['a'], Py['b'])[0] if len(Py) >= 30 else 0.15
            lam_y[y] = float(np.clip(lam, 0.05, 1.0)) if np.isfinite(lam) else 0.15
        lam_row = sub['yr'].map(lam_y)
        _cm = (sub[c + '90'].where(sub['_cohort'])
                 .groupby([sub['league'], sub['seasonId']]).transform('mean'))
        dev = (sub[c + '90'] - _cm).fillna(0.0)
        rel = ROLE_CAT_RELEVANCE.get(role, {}).get(c, 1.0)
        _off_adj.loc[sub.index] += rel * lam_row * dev
        lam = lam_y[max(lam_y)]          # log the latest season's weights
        _lam_log.append({'role': role, 'cat': c.replace(' Value', ''),
                           'lam': round(lam, 2), 'rel': rel,
                           'infl': rel * lam * float(dev.std())})
df['off_blend'] = _off_adj
# standalone set-piece score (not in the rating)
df['DeadBall90'] = df['DeadBall'] / df['mins_played'] * 90
df['setpiece_pct'] = role_pct('DeadBall90')
df['off_pct'] = role_pct('off_blend')   # re-uniform within role x league x season
df['off_total_pct'] = role_pct('Total Offensive Value')   # kept for reference
df['defr_pct'] = role_pct('defr_adj')
# v6.2: DWAE count-shrunk before per-90 (Lucas audit): wins-above-
# expectation is a raw count deviation — a 20-engagement fluke spiked
# the per-90 with no damping. EB factor n/(n+80): half-shrink at the
# 10th-pct player (59 engagements), light at the median (123).
# Measured: dwae pctile YoY 0.229 -> 0.240.
df['dwae_shrunk_p90'] = (df['defr_dwae'] * df['dwae_n']
                           / (df['dwae_n'] + 80.0)
                           / df['mins_played'] * 90.0)
df['dwae_pct'] = role_pct('dwae_shrunk_p90')
df['rapm_pct'] = role_pct('rapm_v4')
# duel axes split by side: defensive contests feed QUALITY (Aerial + Ground
# Grade, below); take-on/shield (attacking side) are their own small axis,
# Off Duel Grade (datt_pct), also below. v7.0: both from the season's own
# contests — the v6.9 career-ladder composites (duel_composite) are gone.

# v6.9 TYPE-MATCHED Def Quality (Lucas): pair the fresh wins-above-
# expectation signal with the opponent-adjusted ladder PER CONTEST TYPE
# — Aerial Grade (aerial WOE + aerial Glicko) and Ground Grade (ground
# WOE + stopper/press Glicko) — then blend by engagement counts.
# Expectation = OPPONENT QUALITY, never physical attributes (Lucas —
# same locked principle that removed height from the aerial ladder):
# p_hat = what an AVERAGE player does vs THIS opponent, so beating an
# excellent aerial player earns more than beating a poor one. Ground
# keeps the situational bucket (kind x zone x phase = context, which
# IS adjusted out) plus the opponent-quality term. Audited 2026-06-12:
# within-role WOE YoY unchanged vs the height/situational version
# (aerial 0.33-0.71, ground 0.23-0.44) and still beats pooled DWAE.
# Opponent strengths are AS-OF shrunk win shares (K0=40): cumulative
# through the contest's season-year only — no future information
# (v6.9.2, Lucas; same leak-free convention as career_asof / RAPM v4).
# For the current season as-of == career, so live boards are unchanged;
# only historical rows get honest. Verified: within-role WOE YoY
# identical, current-season agreement r = 1.000.
#
# TESTED AND REJECTED 2026-07 (Lucas + audit): DANGER-WEIGHTED duel blending.
# Hypothesis (Lucas, confirmed empirically): being dribbled past is far
# costlier than losing a hold-up duel — measured possession-xG swing
# (beaten − won, zone-controlled) take-on 0.0123 vs off-duel 0.0055 =
# 2.25x, sharpest in the own third (+0.027 xG); middle-third hold-up
# losses are ~free. Variant: weight take-on contests 2.25x in the ground
# WOE sum and the stopper/press + takeon/shield composites (volume
# weighting underweights the costlier skill: take-ons are only ~15% of
# contests). REJECTED on the gates: qual axis YoY 0.542 -> 0.521 (noise
# from upweighting the thin 15% overwhelms the importance gain — take-on
# defence has k~54 contests to stabilize vs a median ~8 faced/season);
# rating YoY 0.436 -> 0.432, team-switch 0.501 -> 0.498; and the DECISIVE
# outcome link (within-role qual vs team xGA) did NOT improve (-0.128 ->
# -0.124) while players moved up to 12.6 rating pts. The system already
# prices the danger where measurable: WOE expectations are kind x zone x
# phase bucketed, and the beaten defender's consequences are charged via
# the possession-value ledger. The 2.25x finding LIVES in the scouting
# layer instead: 'Dribbled past % (proj)' (k=54 EB shrink) in the
# dashboard's Defensive metrics. IF REVISITED: try danger-weighting only
# own-third take-ons (where the swing concentrates), gate on the xGA
# outcome link, not YoY.
# v7.1 NO LOOKAHEAD WITHIN THE SEASON (2026-10-29): every contest is judged
# only against contests from matches that kicked off BEFORE it — strengths,
# bucket rates and the strength centring (duel_expectations.py). v7.0's were
# cumulative through the contest's season-year, its own result included.
from models.ratings.duel_expectations import expectations as _duel_expectations  # noqa: E402
_ct = _read(_DASH / 'models/duels/contests.parquet',
                        columns=['ladder', 'seasonId', 'date', 'playerA', 'playerB',
                                  'scoreA', 'att_kind', 'zx', 'phase'])
_a2, _g = _duel_expectations(_ct, CAMP, SEASON_YR)
_g['woe'] = _g['scoreA'] - _g['phat']
_grd = (_g.groupby(['playerA', 'seasonId'])['woe']
          .agg(woe_ground='sum', n_ground='size').reset_index()
          .rename(columns={'playerA': 'playerId'}))
# v7.0 Off Duel Grade, season-only: the ATTACKER side of the same ground
# contests ('dribble' = take-on, 'offensive_duel' = shield / press
# resistance). Expected attacker win = 1 - what an average attacker gets vs
# THIS defender (bucket rate + the defender's strength before the match), so
# beating a strong defender earns more — the mirror of the defensive WOE above.
_g['woe_att'] = _g['pdef'] - _g['scoreA']    # (1 - scoreA) - (1 - pdef)
_att = (_g.groupby(['playerB', 'seasonId'])['woe_att']
          .agg(woe_att='sum', n_att='size').reset_index()
          .rename(columns={'playerB': 'playerId'}))
_sA = (_a2[['playerA', 'seasonId']]
         .assign(woe=_a2['scoreA'] - _a2['pA']))
_sB = (_a2[['playerB', 'seasonId']].rename(columns={'playerB': 'playerA'})
         .assign(woe=(1 - _a2['scoreA'].values) - _a2['pB'].values))
_aer = (pd.concat([_sA, _sB]).groupby(['playerA', 'seasonId'])['woe']
          .agg(woe_aerial='sum', n_aerial='size').reset_index()
          .rename(columns={'playerA': 'playerId'}))
df = df.merge(_grd, on=['playerId', 'seasonId'], how='left')
df = df.merge(_aer, on=['playerId', 'seasonId'], how='left')
df = df.merge(_att, on=['playerId', 'seasonId'], how='left')
for _t in ('aerial', 'ground', 'att'):
    _nn = df[f'n_{_t}'].fillna(0.0)
    df[f'woe_{_t}_p90'] = (df[f'woe_{_t}'].fillna(0.0) * _nn / (_nn + 60.0)
                            / df['mins_played'] * 90.0)
df['aer_woe_pct'] = role_pct('woe_aerial_p90')
df['grd_woe_pct'] = role_pct('woe_ground_p90')
df['datt_pct'] = role_pct('woe_att_p90')
# v7.0 (prototype, 2026-10-27 evaluation): the grades are the SEASON's wins
# over expectation only. v6.9 averaged each with a duel-ladder percentile,
# but the ladders (models/duels/duel_ratings.parquet) are ONE career rating
# per player per trait, mapped to every season: half of Def Quality was a
# career constant (historical rows also saw later seasons' duels), which
# made it look stable (YoY 0.57; the season half alone is 0.32, 0.09 at
# 300-900') and is why it escaped the minutes shrink below. Career duel
# information belongs in the projection, not the descriptive season index —
# NOT moved there yet: qual_pct was the projection's only route to the
# ladders (build_projection.py FEATS), and a leak-free as-of ladder feature
# has to be built first (the stored ladders include later seasons' duels).
df['aerial_grade_pct'] = df['aer_woe_pct']
df['ground_grade_pct'] = df['grd_woe_pct']
_n_a = df['n_aerial'].fillna(0.0)
_n_g = df['n_ground'].fillna(0.0)
_den_t = (_n_a + _n_g)
df['qual_raw'] = np.where(
    _den_t > 0,
    (df['aerial_grade_pct'] * _n_a + df['ground_grade_pct'] * _n_g)
    / _den_t.replace(0, 1),
    0.5)
df['qual_pct'] = role_pct('qual_raw')
# kept for backward compat in exports (nothing reads it): v7.0 = the
# season's defensive duel grade before the minutes shrink, not a ladder
df['duel_pct'] = df['qual_pct']

_lt = pd.DataFrame(_lam_log)
_lt['infl_share'] = _lt['infl'] / _lt.groupby('role')['infl'].transform('sum')
print("  off-axis lambda (reliability shrink) and influence share:")
print(_lt.pivot(index='role', columns='cat', values='lam').to_string())
print((_lt.pivot(index='role', columns='cat', values='infl_share') * 100)
        .round(0).to_string())

# v5.3 per-component shrinkage: reliability scales with minutes
# DIFFERENTLY per axis (off YoY 0.09 at 500-900min vs 0.21 at 1500+;
# qual nearly flat 0.48->0.58 because its inputs are already shrunk at
# source: DWAE/defr EB-shrunk, ladders RD-shrunk, rapm ridge-shrunk).
# So: shrink the OFF axis individually (K=2000) where low-minute noise
# actually leaks in, and lighten the blanket blend shrink (900->300)
# to avoid double-shrinking the source-shrunk axes.
# v5.4 form KEPT after Lucas's reexamination of universal (v5.3)
# shrinkage. Fixed-target A/B (predicting face-value next-season
# rating with each input version): v5.4 inputs career 0.497 /
# replacement 0.507; universal inputs 0.499 / 0.482 — a tie on carry,
# v5.4 better for the shipped replacement form. Universal shrinkage's
# apparent YoY/predictive edge (0.56 vs 0.47) is a SMOOTHED-TARGET
# artifact: shrunk ratings are easier to predict because they are
# shrunk. Conclusion: full seasons at face value; prediction-side
# shrinkage lives in the projection only.
_anchor = 2500.0 / (2500.0 + 2000.0)
_s_off = np.minimum((df['mins_played'] / (df['mins_played'] + 2000.0)) / _anchor, 1.0)
df['off_pct'] = 0.5 + (df['off_pct'] - 0.5) * _s_off
# v7.0 (prototype): the two DEFENSIVE axes get the same shrink. Exempting
# them rested on Def Quality's flat reliability, which was the career
# ladder's; season-only it is the noisiest axis at low minutes. Exempt,
# they carried more of the rating than their weights (CB off:qual variance
# 1.8:1 without the off shrink -> 1.1:1 with it, vs 1.3:1 nominal).
# Off Duel Grade joins them: season-only now (no career ladder), it carries
# the same low-minutes noise. RAPM keeps its own multi-season ridge shrink.
for _c in ('qual_pct', 'defr_pct', 'datt_pct'):
    df[_c] = 0.5 + (df[_c] - 0.5) * _s_off

print("[2/4] role-weighted blend + minutes shrink…", flush=True)
AXES = {'off': 'off_pct', 'resp': 'defr_pct', 'qual': 'qual_pct',
         'datt': 'datt_pct', 'rapm': 'rapm_pct'}
_AX = ['off', 'resp', 'qual', 'datt', 'rapm']
W = pd.DataFrame(0.0, index=df.index, columns=_AX)
for rn in ROLE_UNIVERSE:
    wts = ROLE_WEIGHTS.get(rn, DEFAULT_W)
    for a in _AX:
        W[a] += df['sh_' + rn] * wts[a]
raw = sum(W[a] * df[col] for a, col in AXES.items())        # 0-1
# shrink the player-vs-role deviation toward 0.5 by minutes reliability
# v6.0 (futi adoptions #1 + #3): blend -> z-score within role x league
# x season, low-minute players shrunk toward REPLACEMENT level (z=-0.4,
# our measured convergence target: 44.7 vs mean 48.8 on the old scale —
# playing time is information; pulling unknowns to AVERAGE flattered
# them), full-season anchor kept (2,500 min = no shrink). Display on
# futi's variance-preserving scale: 50 + 17*z, clipped [1, 99].
Z_REPL = -0.4
# blended percentiles are role-fair by construction -> pool z within
# league x season
# moments from cohort (>=MIN_MINS) rows only — sub-floor rows are scored
# on the same scale without shifting it
_rc = raw.where(df['_cohort'])
zraw = (raw - _rc.groupby([df['league'], df['seasonId']]).transform('mean')) \
         / _rc.groupby([df['league'], df['seasonId']]).transform('std')
_ab = 2500.0 / (2500.0 + SHRINK_K)
shrink = np.minimum((df['mins_played'] / (df['mins_played'] + SHRINK_K)) / _ab, 1.0)
zsh = Z_REPL + (zraw - Z_REPL) * shrink
df['acp_rating'] = np.clip(50.0 + 17.0 * zsh, 1.0, 99.0)

print("[3/4] recency-weighted career rating…", flush=True)
car_rows = []
for pid, gg in df.sort_values(['yr', 'league', 'seasonId'], kind='stable').groupby('playerId'):
    gg = gg.reset_index(drop=True)
    latest = gg['yr'].max()
    w = gg['mins_played'].to_numpy() * (CAREER_DECAY ** (latest - gg['yr'].to_numpy()))
    car = float((gg['acp_rating'] * w).sum() / w.sum()) if w.sum() > 0 else np.nan
    for i in range(len(gg)):
        h = gg.iloc[:i + 1]
        wa = h['mins_played'].to_numpy() * (CAREER_DECAY ** (gg.loc[i, 'yr'] - h['yr'].to_numpy()))
        car_rows.append({'playerId': pid, 'seasonId': gg.loc[i, 'seasonId'],
                           'acp_rating_career': car,   # full career (display; has lookahead on old rows)
                           'career_asof': float((h['acp_rating'] * wa).sum() / wa.sum()),
                           'n_seasons': int(gg['seasonId'].nunique())})
df = df.merge(pd.DataFrame(car_rows), on=['playerId', 'seasonId'], how='left')

out_cols = ['playerId', 'seasonId', 'name', 'role', 'side', 'league',
              'position_group', 'mins_played', 'off_pct', 'defr_pct',
              'dwae_pct', 'qual_pct', 'datt_pct', 'duel_pct', 'rapm_pct', 'setpiece_pct', 'acp_rating', 'acp_rating_career',
              'n_seasons']
# per-category offence percentiles (engine radar surfaces Off split
# into its five categories)
out_cols = out_cols + [c + '_pct' for c in GPA_CATS]
# v6.9 type-matched defensive sub-grades + their raw WOE halves
out_cols = out_cols + ['aerial_grade_pct', 'ground_grade_pct',
                         'woe_aerial_p90', 'woe_ground_p90', 'woe_att_p90']
out_cols = out_cols + ['sh_' + nm for nm in ROLE_NAMES]
out = df[out_cols].copy()
out['rating_version'] = 'v7.1'   # v7.1: duel expectations as of kick-off (duel_expectations.py)
out.to_parquet(_OUT / 'acp_rating_per_player_season.parquet')
print(f"  saved acp_rating_per_player_season.parquet ({len(out):,} rows)")

print("[4/4] validation…", flush=True)


def yoy(col):
    P = []
    for pid, gg in df.sort_values('yr', kind='stable').groupby('playerId'):
        rr = gg.dropna(subset=[col]).to_dict('records')
        for a, b in zip(rr, rr[1:]):
            if b['yr'] - a['yr'] == 1 and a['position_group'] == b['position_group']:
                P.append((a[col], b[col]))
    P = pd.DataFrame(P, columns=['a', 'b'])
    return (pearsonr(P['a'], P['b'])[0] if len(P) >= 3 else float('nan')), len(P)

r_s, n = yoy('acp_rating')
print(f"  acp_rating YoY (same pos): r = {r_s:.3f} (n={n})")
# futi adoption #4: ratings should hold when players SWITCH TEAMS
# within a league — the cleanest test that we measure players, not
# team contexts.
_pt = _read(_HERE / 'player_teams.parquet')
df = df.merge(_pt, on=['playerId', 'seasonId'], how='left')
_sw = {'stay': [], 'switch': []}
for pid, gg in df.sort_values('yr', kind='stable').groupby('playerId'):
    rr = gg.to_dict('records')
    for a, b in zip(rr, rr[1:]):
        if (b['yr'] - a['yr'] == 1 and a['position_group'] == b['position_group']
                and a['league'] == b['league'] and pd.notna(a['team'])
                and pd.notna(b['team'])):
            _sw['stay' if a['team'] == b['team'] else 'switch'].append(
                (a['acp_rating'], b['acp_rating']))
for k, v in _sw.items():
    V = pd.DataFrame(v)
    if len(V) >= 3:   # a truncated (ACP_MAX_YEAR) build can have too few pairs
        print(f"  YoY team-{k:<7}: r = {pearsonr(V[0], V[1])[0]:.3f} (n={len(V)})")
# component contribution check: corr of each component with the rating
cur = df[df['seasonId'].isin({191782, 191779})]
print("  component corr with rating (25/26):")
for c in ['off_pct', 'defr_pct', 'qual_pct', 'datt_pct', 'rapm_pct']:
    if len(cur) >= 3:
        print(f"    {c:<10} {pearsonr(cur[c], cur['acp_rating'])[0]:+.2f}")
print(f"\n  Top 12 by career rating (>=2 seasons):")
top = (df.drop_duplicates('playerId')
         .query('n_seasons >= 2')
         .nlargest(12, 'acp_rating_career'))
for _, x in top.iterrows():
    print(f"    {str(x['name'])[:22]:<24} {x['role'][:18]:<19} {x['side']} "
           f"career={x['acp_rating_career']:.0f}  "
           f"(off {x['off_pct']*100:.0f} def {x['defr_pct']*100:.0f} "
           f"dwae {x['dwae_pct']*100:.0f} rapm {x['rapm_pct']*100:.0f})")
print(f"\n  Top 8 current-season (25/26) by acp_rating:")
for _, x in cur.nlargest(8, 'acp_rating').iterrows():
    print(f"    {str(x['name'])[:22]:<24} {x['role'][:18]:<19} {x['league']:<5} "
           f"rating={x['acp_rating']:.0f}")
