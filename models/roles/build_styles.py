#!/usr/bin/env python3
"""Player styles v3 — archetypes DERIVED FROM THE TENDENCY VECTOR.

Supersedes v2 (per-role k-means on an action-mix feature block, commit
0973bec). Why the rebuild, per the approved spec:

  * v2 had no neutral centre. k-means partitions everybody, so every player got
    a "type" even when he was simply a normal player for his role, and the WA
    partition had no Arriver at all — the taxonomy was whatever k-means found,
    not what a scout would name.
  * v2's hard labels flipped ~35% year on year: a cluster boundary drawn through
    the middle of a dense blob reassigns anyone who drifts slightly.

v3 follows futi's architecture: TENDENCIES are the primitive, styles are read
off them. A player's style is the POLE OF HIS MOST EXTREME TENDENCY, but only
if that lean is real (|z| >= a per-role threshold); otherwise he is
"Conventional <Role>" — an explicit, populated centre, which is both the honest
answer for most players and what stops labels flipping on noise.

Order of resolution (composites FIRST, per spec):
  1. COMPOSITE styles — the multi-tendency archetypes futi names (Wingback =
     crosses + gets forward; Target Striker = aerial + arrives). Checked first
     because a composite is a stronger claim than any single pole.
  2. SINGLE POLE — the most extreme style-defining tendency, if |z| >= thr AND
     it beats the runner-up named pole by the role's margin (v3.1).
  3. CONVENTIONAL <Role>.

v3.1 RELIABILITY RETUNE (2026-07-16, Lucas-approved "full sticky retune"):
  * STICKY LABELS — styles are assigned on a recency-weighted blend of this
    season's z with the player's same-role prior season (minutes-weighted,
    LAMBDA_PRIOR discount; ~futi's rolling-12-month assignment). Display
    tendencies stay strictly per-season; only the label + fit read the blend.
  * MARGIN RULE — near-tied top poles no longer coin-flip the argmax between
    seasons; ties fall to Conventional. Tuned jointly with the threshold.
  * Both the sticky label churn (what users see) AND the unsmoothed
    same-config YoY (the underlying signal) are reported, so the improvement
    from smoothing is never mistaken for signal — this project's artifact
    family (pooled/mechanical/smoothed-target/weight-shift) demands it.
  * NOT done, deliberately: reliability-weighting the axes (tested — raw
    agreement up, kappa DOWN via taxonomy collapse; rejected).

Not every displayed tendency is style-defining. Panels show context axes
(defensive-line height, near/far post) that describe a player without making an
archetype; STYLE_AXES names only the poles that earn a style. Deliberate:
naming all 2xN poles would splinter occupancy into unusable slivers.

STYLES ARE DESCRIPTIVE, NEVER QUALITY, and never enter the rating or projection.

Outputs:
  style_assignments_season.parquet — v2 columns (playerId, seasonId,
      primary_role_name, style_id, style_name) + style, style_fit, style_2,
      style_2_fit, role, mins_played, name, thin_sample
  style_profiles.csv — per-style n, mean z-profile, exemplars

Run from the Dashboard dir:  python models/roles/build_styles.py
  --split-half   also run the odd/even-match reliability check (re-reads the
                 event store; slow)
"""
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from build_tendencies import (ALL_TENDENCIES, ROLE_TENDENCY_MENU,
                              SEASON_ONLY_TENDENCIES, MIN_MINS_COHORT,
                              MIN_MINS_SCORED, load_events, prepare,
                              tendency_table, load_roles, load_defr)

_ROLES_DIR = Path(os.environ.get('ACP_ROLES_DIR', _HERE))
_TEND = _ROLES_DIR / 'tendencies_season.parquet'
_OUT = _HERE / 'style_assignments_season.parquet'
_PROF = _HERE / 'style_profiles.csv'
_V2 = _ROLES_DIR / 'style_assignments_season_v2_backup.parquet'

THR_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
            0.80, 0.85, 0.90]
MARGIN_GRID = [0.0, 0.10, 0.20, 0.30]   # runner-up margin, tuned with thr
# v3.2 (Lucas, 2026-07-16): Conventional MINIMIZED — band lowered from the
# spec's original 20-45% to 10-25%. Evidence behind the floor: centre size vs
# stability is U-shaped (uniform thr 0.2-0.4 scored WORSE than either a real
# centre or pure argmax — a thin shell maximises boundary-hopping), so the
# tuner keeps a floor rather than chasing zero; the sticky blend carries the
# stability the big centre used to provide. Pure argmax reference: YoY 64.6%,
# kappa 0.513, with a 3-9% residual centre anyway (players leaning only
# toward unnamed poles).
# v4: band widened to (0.10, 0.45) — with 2-3 named styles per role the
# Conventional centre naturally carries more mass (the v3.2 minimize
# directive is superseded by the 2026-07-17 "fewer, consistent styles"
# directive). The tuner still prefers the most stable feasible config, so
# smaller centres win where the data supports them.
CONVENTIONAL_TARGET = (0.10, 0.45)
MAX_STYLE_SHARE = 0.60
MIN_STYLE_SHARE = 0.08

# v3.1 sticky blend: the ASSIGNMENT vector is this season's z blended with the
# player's same-role prior-season z, weighted by minutes with the prior
# discounted (w_cur = m_t / (m_t + LAMBDA_PRIOR * m_{t-1}); equal minutes ->
# 2/3 current). futi assigns styles on a rolling 12 months; strict per-season
# labelling was noisier by construction (Lucas approved the divergence-from-
# per-season 2026-07-16). DISPLAY tendencies remain purely per-season — only
# the label (and its fit) read the blended vector.
LAMBDA_PRIOR = 0.5

YEAR = {188221: 2021, 188222: 2022, 189147: 2023, 190090: 2024, 191782: 2025,
        190230: 2023, 191779: 2025}

# ---------------------------------------------------------------------------
# STYLE AXES — which tendency poles earn a name, per role.
# Names adopt futi's where they map cleanly onto our roles, and keep the June v2
# names where those read better for this league (Target / Link-Up / Running
# Striker, No-Nonsense).
#
# SIZE OF THE TAXONOMY (settled 2026-07-16 on evidence): the first cut named
# 2xN poles per role and produced 5-8 styles per role. That is more than futi
# shows (their panels name ~3-4 per role, which is also the spec's own name
# list) and it cost real stability: the more named poles compete for the argmax,
# the more often two near-tied axes swap places between seasons and the label
# flips for no footballing reason (YoY 44% same-role at 5-8 styles vs 63% here).
# The set below is pruned to the spec's named list — every style here is one
# futi names or one of the June v2 names the spec asked us to keep.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# v4 BOOK (2026-07-17, Lucas: "consistent and descriptive; don't need to match
# futi; detect how many styles per role"). Book sized per role by three
# independent measures, all in the blended tendency space, cohort only:
#   1. GMM (diag) BIC + k-means split-half ARI (does the same cluster
#      structure reappear in random player halves): ST k=3 (ARI 0.67 — the
#      strongest structure), WA 2-3 (0.53/0.51), AM k=2 (0.75, emphatic),
#      DM 2-3 (0.60/0.56), WD k=2 (0.72; k=3 collapses to 0.34), CB 2-3
#      (0.62/0.55).
#   2. Per-style split-half retention (odd/even matches, n=2,615 pairs —
#      the big-sample gate the thin YoY panel can't give): keepers Defensive
#      Fullback 62%, Aggressive 60%, Dribbler 52%, Combining 50%, both
#      Playmakers 47%; casualties Poacher 23% (churns into Target — a box
#      shot-diet is a CORRELATE of the target-man blob, never its own
#      cluster), Arriving Midfielder 27% (the AM confusion hub), Stay-Home
#      27% (top confusion = No-Nonsense BOTH ways -> one conservative
#      persona), DLP-composite 13% (two coupled conditions shatter on half
#      samples), Pivot 38%, Destroyer 38% (scheme axis; Lucas's earlier
#      keep-ruling explicitly reversed with the new evidence, 2026-07-17).
#   3. Cluster-centroid inspection to name what survives.
# RESULT: 16 named + 6 Conventional (was 20+6). Composites pruned to Target
# Striker alone; DLP is the plain build-up-hub pole (long-ball condition
# dropped), Wingback is the crossing pole opposite Combining (the axis is
# stable, YoY 0.469 — only the two-condition form was fragile).
# History: v3.3 cut Attacking Fullback (mushy middle), renamed futi's
# 'Defensive Stopper' -> 'Stay-Home Defender' (their name, not their
# mechanic) before its merge here; Destroyer's 2026-07-16 keep-ruling and the
# full v3.x evidence trail live in the branch commit messages.
# ---------------------------------------------------------------------------
STYLE_AXES = {
    'Striker': {
        'ground_aerial':          {'high': 'Target Striker', 'low': None},
        'come_short_run_behind':  {'high': 'Running Striker', 'low': None},
        'carry_pass':             {'high': 'Link-Up Striker', 'low': None},
    },
    'Wide Attacker': {
        'create_arrive':          {'high': 'Wide Arriver',
                                   'low': 'Wide Playmaker'},
        'carry_pass':             {'high': None, 'low': 'Wide Dribbler'},
    },
    'Advanced Midfielder': {
        'create_arrive':          {'high': None, 'low': 'Advanced Playmaker'},
        'carry_pass':             {'high': None, 'low': 'Ball Carrier'},
    },
    'Deep Midfielder': {
        'passive_active_buildup': {'high': 'Deep-Lying Playmaker', 'low': None},
        'carry_pass':             {'high': None, 'low': 'Carrying Midfielder'},
    },
    'Wide Defender': {
        'passive_active':         {'high': 'Defensive Fullback', 'low': None},
        'combine_cross':          {'high': 'Wingback',
                                   'low': 'Combining Fullback'},
    },
    'Central Defender': {
        # 'Conservative Defender' (Lucas, 2026-07-17 name workshop) — was
        # June's 'No-Nonsense': accurate idiom but temperament-flavoured and
        # the only CB label without the noun. Conservative = the measured
        # mechanic (safe pass selection + the absorbed stay-home positioning),
        # quality-neutral, natural mirror of Ball-Playing.
        'secure_progressive':     {'high': 'Ball-Playing Defender',
                                   'low': 'Conservative Defender'},
        'passive_active':         {'high': 'Aggressive Defender', 'low': None},
    },
}

# ---------------------------------------------------------------------------
# COMPOSITES — futi's multi-tendency archetypes. Checked before single poles.
#   (name, [(tendency, 'high'|'low', min_z), ...])  — ALL conditions must hold.
# ---------------------------------------------------------------------------
COMPOSITES = {
    # A target man is aerial AND on the end of things — not merely tall. The
    # Arrive bar is the strict one so a link-up striker who happens to win
    # headers (Elias Franco: aerial 75th, arrive 65th) does NOT read Target,
    # while Balotelli (aerial 66th, arrive 99th) does. The ONLY surviving
    # composite (v4): its half-sample retention (43%) holds where the DLP and
    # Wingback composites shattered (13% / 33%) — those are single-pole now.
    'Striker': [
        ('Target Striker', [('ground_aerial', 'high', 0.25),
                            ('create_arrive', 'high', 0.80)]),
    ],
}


def conventional(role):
    return f'Conventional {role}'


def zscores(T, cohort_mask):
    """z-score each role's MENU tendencies within the role, fitting mean/sd on
    the >=900' cohort only (thin players are scored on the cohort's scale, they
    do not move it)."""
    Z = pd.DataFrame(index=T.index, columns=[f'z_{k}' for k in ALL_TENDENCIES],
                     dtype=float)
    stats = {}
    for role, g in T.groupby('role', observed=True):
        ref = g[cohort_mask.reindex(g.index, fill_value=False)]
        for k in ROLE_TENDENCY_MENU.get(role, []):
            v = ref[f't_{k}'].dropna()
            if len(v) < 20:
                continue
            mu, sd = v.mean(), v.std()
            if not sd or not np.isfinite(sd):
                continue
            Z.loc[g.index, f'z_{k}'] = (g[f't_{k}'] - mu) / sd
            stats[(role, k)] = (mu, sd)
    return Z, stats


def apply_zstats(T, stats):
    """Apply an existing (role, tendency) -> (mu, sd) map. Used by the
    split-half check so both halves land on the season-fitted scale."""
    Z = pd.DataFrame(index=T.index, columns=[f'z_{k}' for k in ALL_TENDENCIES],
                     dtype=float)
    for role, g in T.groupby('role', observed=True):
        for k in ROLE_TENDENCY_MENU.get(role, []):
            if (role, k) not in stats:
                continue
            mu, sd = stats[(role, k)]
            Z.loc[g.index, f'z_{k}'] = (g[f't_{k}'] - mu) / sd
    return Z


def _prior_z_table(T, Z):
    """(playerId, role, year+1) -> minutes-weighted mean of the PRIOR season's
    raw z + that season's minutes. Keyed at year+1 so it merges directly onto
    the current row. A player with rows in both leagues the same year (loan /
    mid-season move) contributes both, minutes-weighted."""
    zcols = list(Z.columns)
    D = pd.concat([T[['playerId', 'role', 'year', 'mins_played']]
                   .reset_index(drop=True), Z.reset_index(drop=True)], axis=1)
    D = D[D['year'].notna()]
    rows = []
    for (pid, role, yr), g in D.groupby(['playerId', 'role', 'year']):
        w = g['mins_played'].to_numpy(float)
        Zg = g[zcols].to_numpy(float)
        vals = np.where(np.isnan(Zg), 0.0, Zg)
        wm = np.where(np.isnan(Zg), 0.0, w[:, None])
        den = wm.sum(axis=0)
        avg = np.divide((vals * wm).sum(axis=0), den,
                        out=np.full(len(zcols), np.nan), where=den > 0)
        rows.append((pid, role, yr + 1, w.sum(), *avg))
    return pd.DataFrame(rows, columns=['playerId', 'role', 'year',
                                       'prior_mins'] + zcols)


def blend_z(T, Z, prior, lam=LAMBDA_PRIOR):
    """Sticky assignment space: blend this season's z with the same-role prior
    season's (from _prior_z_table). Rows with no prior pass through unchanged;
    a NaN on either side falls back to the other."""
    zcols = list(Z.columns)
    D = T[['playerId', 'role', 'year', 'mins_played']].reset_index(drop=True)
    M = D.merge(prior, on=['playerId', 'role', 'year'], how='left')
    cur = Z.to_numpy(float)
    prv = M[zcols].to_numpy(float)
    pm = M['prior_mins'].to_numpy(float)
    w = np.where(np.isnan(pm), 1.0,
                 D['mins_played'].to_numpy(float)
                 / (D['mins_played'].to_numpy(float) + lam * np.nan_to_num(pm)))
    w = w[:, None]
    out = np.where(np.isnan(prv), cur,
                   np.where(np.isnan(cur), prv, w * cur + (1.0 - w) * prv))
    return pd.DataFrame(out, index=T.index, columns=zcols)


def assign_styles(T, Z, thr_by_role, margin_by_role=None):
    """Composites -> most-extreme named single pole -> Conventional.

    v3.1 margin rule: the winning pole must clear the role's |z| threshold AND
    beat the runner-up NAMED pole by the role's margin. Two near-tied leans are
    an ambiguous read — before this rule they swapped the argmax between
    seasons for no footballing reason (the largest churn source); now they fall
    to Conventional, which is the honest label for a player without one
    dominant lean."""
    margin_by_role = margin_by_role or {}
    out = []
    zcols = set(Z.columns)
    for i, role in zip(T.index, T['role']):
        axes = STYLE_AXES.get(role, {})
        thr = thr_by_role.get(role, 0.75)
        mar = margin_by_role.get(role, 0.0)
        lab = None
        for name, conds in COMPOSITES.get(role, []):
            ok = True
            for k, side, mz in conds:
                z = Z.at[i, f'z_{k}'] if f'z_{k}' in zcols else np.nan
                if pd.isna(z) or (z < mz if side == 'high' else z > -mz):
                    ok = False
                    break
            if ok:
                lab = name
                break
        if lab is None:
            cands = []
            for k, pole in axes.items():
                z = Z.at[i, f'z_{k}'] if f'z_{k}' in zcols else np.nan
                if pd.isna(z):
                    continue
                nm = pole['high'] if z > 0 else pole['low']
                if nm is None:
                    continue
                cands.append((abs(z), nm))
            cands.sort(key=lambda t: -t[0])
            if (cands and cands[0][0] >= thr
                    and (len(cands) == 1 or cands[0][0] - cands[1][0] >= mar)):
                lab = cands[0][1]
        out.append(lab if lab is not None else conventional(role))
    return pd.Series(out, index=T.index, name='style')


def fit_and_mix(T, Z, styles):
    """style_fit (0-100) + top-2 style mix, from position in z-space.

    Fit is how far a player travels in his style's DIRECTION, not how close he
    sits to its centroid. That distinction matters and the first version got it
    wrong: a style's centroid is the mean of its members, so the most extreme
    exemplar is the FURTHEST point from it — centroid distance rated Balotelli
    (Arrive 99th) a 24% Target Striker and put a rival style above him. Since
    Conventional sits at the origin by construction, each named style's centroid
    doubles as its direction vector, and projecting a player onto that unit
    direction measures exactly what the label claims: how strongly he expresses
    the style. The archetype scores ~100.

      named style  -> score = z . unit(centroid direction)
      Conventional -> score = -(max |z| over the role's style axes), i.e. how
                      centrally he sits; the most style-less player scores ~100

    Scores become 0-100 by ECDF against EVERY cohort player in the role, not
    just that style's own members. Referencing to members looked natural but is
    a trap: an ECDF within members is uniform by construction, so every style's
    fit averaged 50 and a fit could not be compared with the runner-up's (a rank
    among Target Strikers and a rank among Running Strikers are different
    populations — that is how Balotelli came out 47% Target with a 74% second).
    Against one common population, style_fit and style_2_fit are on one scale
    and read as a genuine mix: "88% Target Striker, 74% Running Striker"."""
    fits = pd.Series(np.nan, index=T.index)
    s2 = pd.Series(None, index=T.index, dtype=object)
    f2 = pd.Series(np.nan, index=T.index)

    def _pct(scores, member_scores):
        ref = np.sort(np.asarray(member_scores, dtype=float))
        ref = ref[~np.isnan(ref)]
        if len(ref) == 0:
            return np.full(len(scores), np.nan)
        lo = np.searchsorted(ref, scores, side='left')
        hi = np.searchsorted(ref, scores, side='right')
        p = (lo + hi) / 2.0 / len(ref) * 100.0
        return np.where(np.isnan(scores), np.nan, p)

    for role, g in T.groupby('role', observed=True):
        axes = STYLE_AXES.get(role, {})
        dims = [f'z_{k}' for k in ROLE_TENDENCY_MENU.get(role, [])
                if f'z_{k}' in Z.columns]
        if not dims:
            continue
        M = np.nan_to_num(Z.loc[g.index, dims].to_numpy(dtype=float))
        lab = styles.loc[g.index]
        named = sorted(n for n in lab.unique() if n != conventional(role))

        # direction per named style = its centroid (Conventional is the origin)
        proj = {}
        for nm in named:
            c = M[(lab == nm).to_numpy()].mean(axis=0)
            nrm = np.linalg.norm(c)
            if nrm < 1e-9:
                continue
            proj[nm] = M @ (c / nrm)

        # centrality score for the Conventional bucket
        acols = [f'z_{k}' for k in axes if f'z_{k}' in Z.columns]
        central = -np.nanmax(np.abs(Z.loc[g.index, acols].to_numpy(dtype=float)),
                             axis=1) if acols else np.zeros(len(g))

        # one common reference population: the role's cohort players
        ref_m = (T.loc[g.index, 'mins_played'] >= MIN_MINS_COHORT).to_numpy()
        if ref_m.sum() < 20:
            ref_m = np.ones(len(g), dtype=bool)
        pct = {nm: _pct(proj[nm], proj[nm][ref_m]) for nm in proj}
        pct[conventional(role)] = _pct(central, central[ref_m])

        fits.loc[g.index] = np.array(
            [pct[l][i] if l in pct else np.nan for i, l in enumerate(lab)])

        # secondary lean = the next named style he most expresses
        sec, secw = [], []
        for i, l in enumerate(lab):
            alts = [(pct[nm][i], nm) for nm in proj if nm != l]
            if not alts:
                sec.append(None); secw.append(np.nan); continue
            v, nm = max(alts)
            sec.append(nm)
            secw.append(v)
        s2.loc[g.index] = sec
        f2.loc[g.index] = secw
    return fits, s2, f2


def _yoy(frame, styles, roles=None, same_role=True):
    """Hard-label agreement for the same player in consecutive seasons.

    same_role=True by default: styles are defined WITHIN a role, so a player who
    changes role necessarily changes style — counting those as disagreements
    measures role churn, not style stability.

    Also returns Cohen's kappa. Raw agreement is NOT comparable across
    taxonomies of different size — v2 clustered each role into k=2 or 3, where
    chance agreement alone is 33-50%, while v3 names 3-5 styles per role. Kappa
    corrects for that and is the honest v2-vs-v3 number."""
    s = pd.DataFrame({'playerId': frame['playerId'].values,
                      'year': frame['year'].values,
                      'style': styles.values})
    if roles is not None:
        s['role'] = roles.values
    a = s.copy(); a['year'] = a['year'] + 1
    m = a.merge(s, on=['playerId', 'year'], suffixes=('_1', '_2'))
    if roles is not None and same_role:
        m = m[m['role_1'] == m['role_2']]
    if len(m) < 10:
        return float('nan'), 0, float('nan')
    obs = (m['style_1'] == m['style_2']).mean()
    # chance agreement from the marginal label distribution
    p = pd.concat([m['style_1'], m['style_2']]).value_counts(normalize=True)
    exp = float((p ** 2).sum())
    kappa = (obs - exp) / (1 - exp) if exp < 1 else float('nan')
    return obs, len(m), kappa


def tune_thresholds(T, Z, cohort):
    """Pick each role's (|z| threshold, runner-up margin) JOINTLY: among grid
    combos that put Conventional in the spec's 20-45% band and keep every named
    style within 8-60%, take the one with the best YoY hard-label agreement.
    If none satisfies occupancy, take the combo closest to the Conventional
    band (and say so).

    Occupancy is a hard gate BEFORE YoY on purpose — it is what stops this
    tuner from rediscovering the reliability-flattering collapse (label
    everything Conventional and YoY soars). Kappa per role is reported
    downstream as the second guard."""
    chosen, report = {}, []
    for role in ROLE_TENDENCY_MENU:
        g = T[(T['role'] == role) & cohort]
        if g.empty:
            chosen[role] = (0.75, 0.0)
            continue
        best = None
        for thr in THR_GRID:
            for mar in MARGIN_GRID:
                st = assign_styles(g, Z.loc[g.index], {role: thr}, {role: mar})
                share = st.value_counts(normalize=True)
                conv = share.get(conventional(role), 0.0)
                named = share.drop(labels=[conventional(role)], errors='ignore')
                ok_occ = bool(len(named)
                              and CONVENTIONAL_TARGET[0] <= conv <= CONVENTIONAL_TARGET[1]
                              and (named < MAX_STYLE_SHARE).all()
                              and (named >= MIN_STYLE_SHARE).all())
                y, n, _ = _yoy(g, st, g['role'])
                report.append((role, thr, mar, conv, y, ok_occ, len(named)))
                dist = max(0.0, CONVENTIONAL_TARGET[0] - conv,
                           conv - CONVENTIONAL_TARGET[1])
                cand = ((1 if ok_occ else 0), -dist, y if y == y else -1,
                        thr, mar)
                if best is None or cand[:3] > best[:3]:
                    best = cand
        chosen[role] = (best[3], best[4])
    thr = {r: v[0] for r, v in chosen.items()}
    mar = {r: v[1] for r, v in chosen.items()}
    return thr, mar, report


def split_half(T, stats, thr, mar, prior, cohort):
    """Odd/even-match reliability: rebuild tendencies on each half of a player's
    matches through the SAME code path, label each half, and ask whether the two
    halves agree. This is v2's headline check (76.7%) run on v3.

    Reported in BOTH labelling modes:
      margin-only — each half labelled from its own z alone. The honest
          within-season reliability of the underlying signal.
      sticky      — each half blended with the player's prior FULL season,
          exactly as production labels are. Mechanically flattered (both
          halves share the same prior anchor) but it IS the reliability of
          the label users see; read it with that caveat.

    CAVEAT, disclosed: DefR is published per player-season, so passive_active
    (SEASON_ONLY_TENDENCIES) cannot be halved — it is held at its season value
    in both halves, which flatters the roles whose style axes use it (DM, WD,
    CB). Read those three roles' figures as an upper bound."""
    print("  rebuilding tendencies on odd/even match halves…", flush=True)
    ev = prepare(load_events())
    keys = ['playerId', 'seasonId', 'parity']
    roles = load_roles().copy()
    # each half is ~half a season: keep the per-90 rates comparable
    roles['mins_played'] = roles['mins_played'] / 2.0
    H, _ = tendency_table(ev, keys, roles, load_defr())
    H = H.merge(roles[['playerId', 'seasonId', 'role', 'mins_played']]
                .rename(columns={'mins_played': '_half_mins'}),
                on=['playerId', 'seasonId'], how='left')
    # blend weight uses the half's minutes vs the prior FULL season's — the
    # as-deployed behaviour for a part-season sample
    H['mins_played'] = H['_half_mins'] if 'mins_played' not in H.columns \
        else H['mins_played'].fillna(H['_half_mins'])
    keep = set(map(tuple, T.loc[cohort, ['playerId', 'seasonId']].to_numpy()))
    H = H[[tuple(r) in keep for r in H[['playerId', 'seasonId']].to_numpy()]]
    H['year'] = H['seasonId'].map(YEAR)
    ZH = apply_zstats(H, stats)
    ZHb = blend_z(H, ZH, prior)
    out = {}
    for tag, Zx in (('margin-only', ZH), ('sticky', ZHb)):
        lab = assign_styles(H, Zx, thr, mar)
        Hx = H.assign(style=lab)
        h0 = Hx[Hx['parity'] == 0][['playerId', 'seasonId', 'style']]
        h1 = Hx[Hx['parity'] == 1][['playerId', 'seasonId', 'style']]
        m = h0.merge(h1, on=['playerId', 'seasonId'], suffixes=('_0', '_1'))
        agree = (m['style_0'] == m['style_1']).mean()
        out[tag] = agree
        print(f"  v3 split-half agreement [{tag}]: {agree * 100:.1f}%  "
              f"(n={len(m)})")
        per = []
        for role, g in m.merge(T[['playerId', 'seasonId', 'role']],
                               on=['playerId', 'seasonId']).groupby('role'):
            per.append((role, (g['style_0'] == g['style_1']).mean(), len(g)))
        for role, a, n in sorted(per, key=lambda t: -t[1]):
            print(f"      {role:<22}{a * 100:5.1f}%  (n={n})")
    print(f"    NOTE: passive_active held at its season value in both halves "
          f"(DefR is season-level) — flatters DM/WD/CB in both modes; the "
          f"sticky mode additionally shares the prior-season anchor across "
          f"halves by construction.")
    return out


def per_role_stability(frame, styles, roles):
    """Per-role YoY agreement + Cohen's kappa.

    Kappa must be computed WITHIN a role: the choice set a player faces is his
    own role's styles, so pooling the marginals across roles pretends the label
    space is larger than it is and inflates kappa (it flattered v2's 0.439 to
    0.674 when pooled)."""
    d = pd.DataFrame({'playerId': frame['playerId'].values,
                      'year': frame['year'].values,
                      'role': roles.values, 'lab': styles.values})
    a = d.copy(); a['year'] = a['year'] + 1
    m = a.merge(d, on=['playerId', 'year'], suffixes=('_1', '_2'))
    m = m[m['role_1'] == m['role_2']]
    rows = []
    for r, g in m.groupby('role_1'):
        obs = (g['lab_1'] == g['lab_2']).mean()
        p = pd.concat([g['lab_1'], g['lab_2']]).value_counts(normalize=True)
        exp = float((p ** 2).sum())
        rows.append((r, obs, (obs - exp) / (1 - exp) if exp < 1 else np.nan,
                     len(g), g['lab_1'].nunique()))
    return pd.DataFrame(rows, columns=['role', 'obs', 'kappa', 'n', 'n_styles'])


def _v2_frame():
    v2 = pd.read_parquet(_V2)
    v2['year'] = v2['seasonId'].map(YEAR)
    lab = (v2['primary_role_name'].astype(str) + '#'
           + v2['style_id'].astype(str))
    return v2, lab, v2['primary_role_name']


def v2_baselines():
    """v2's YoY on the SAME pair definition, for an apples-to-apples baseline
    (the spec quotes ~65% YoY and 76.7% split-half)."""
    if not _V2.exists():
        print("  (v2 backup missing — using the spec's quoted baselines)")
        return
    v2 = pd.read_parquet(_V2)
    v2['year'] = v2['seasonId'].map(YEAR)
    lab = (v2['primary_role_name'].astype(str) + '#'
           + v2['style_id'].astype(str))
    y, n, k = _yoy(v2, lab, v2['primary_role_name'])
    nst = v2.groupby('primary_role_name')['style_id'].nunique().to_dict()
    print(f"  v2 YoY hard-label agreement (same role): {y * 100:.1f}%  "
          f"(n={n})   kappa {k:.3f}   [spec quotes ~65%]")
    print(f"    v2 taxonomy: {nst}")
    print("    v2 clusters each role into k=2-3, where chance agreement alone "
          "is 33-50%; kappa is the fair v2-vs-v3 comparison.")


def main():
    print("[1/6] tendencies…", flush=True)
    T = pd.read_parquet(_TEND)
    T['year'] = T['seasonId'].map(YEAR)
    cohort = T['mins_played'] >= MIN_MINS_COHORT
    print(f"  {len(T):,} rows  ({cohort.sum():,} cohort)")

    print("[2/6] z-scores within role…", flush=True)
    Z, stats = zscores(T, cohort)
    prior = _prior_z_table(T, Z)
    Zb = blend_z(T, Z, prior)
    has_prior = (T[['playerId', 'role', 'year']]
                 .merge(prior[['playerId', 'role', 'year']].drop_duplicates(),
                        on=['playerId', 'role', 'year'], how='left',
                        indicator=True)['_merge'] == 'both')
    print(f"  sticky blend (lam={LAMBDA_PRIOR}): "
          f"{int(has_prior.sum()):,}/{len(T):,} rows have a same-role prior "
          f"season; the rest label from this season alone")

    print("[3/6] tuning (|z| threshold, runner-up margin) per role…", flush=True)
    thr, mar, report = tune_thresholds(T, Zb, cohort)
    feas = {}
    for role, t, m, conv, y, ok, n in report:
        feas[role] = feas.get(role, 0) + (1 if ok else 0)
    print(f"  {'role':<22}{'thr':>6}{'mar':>6}{'conv%':>8}{'stickyYoY':>11}"
          f"{'feasible':>10}")
    for role in thr:
        row = next(r for r in report
                   if r[0] == role and abs(r[1] - thr[role]) < 1e-9
                   and abs(r[2] - mar[role]) < 1e-9)
        ys = f"{row[4] * 100:>9.0f}%" if row[4] == row[4] else "         —"
        ncombo = len(THR_GRID) * len(MARGIN_GRID)
        print(f"  {role:<22}{thr[role]:>6.2f}{mar[role]:>6.2f}"
              f"{row[3] * 100:>7.0f}%{ys}{feas[role]:>7}/{ncombo}")

    print("\n[4/6] assigning…", flush=True)
    styles = assign_styles(T, Zb, thr, mar)
    fits, s2, f2 = fit_and_mix(T, Zb, styles)
    T = T.assign(style=styles, style_fit=fits, style_2=s2, style_2_fit=f2)

    print("\n=== occupancy by role (cohort only) ===")
    for role in ROLE_TENDENCY_MENU:
        g = T[cohort & (T['role'] == role)]
        if g.empty:
            continue
        vc = g['style'].value_counts(normalize=True)
        print(f"  {role} (n={len(g)}):")
        for nm, sh in vc.items():
            flag = ''
            if nm == conventional(role):
                if not (CONVENTIONAL_TARGET[0] <= sh <= CONVENTIONAL_TARGET[1]):
                    flag = '  *** outside 20-45%'
            elif sh > MAX_STYLE_SHARE:
                flag = '  *** > 60%'
            elif sh < MIN_STYLE_SHARE:
                flag = '  *** < 8%'
            print(f"     {nm:<28}{sh * 100:5.1f}%  ({int(round(sh * len(g)))}){flag}")

    print("\n[5/6] validation…", flush=True)
    y3, n3, k3 = _yoy(T[cohort], styles[cohort], T.loc[cohort, 'role'])
    ya, na, ka = _yoy(T[cohort], styles[cohort], T.loc[cohort, 'role'],
                      same_role=False)
    print(f"  v3 sticky YoY hard-label agreement (same role)   : "
          f"{y3 * 100:.1f}%  (n={n3})")
    print(f"  v3 sticky YoY incl. role changes:                  "
          f"{ya * 100:.1f}%  (n={na})")
    # the sticky label borrows last season's evidence, so its YoY is partly
    # construction. The unsmoothed run of the SAME config separates real
    # signal from smoothing: this is what the label churn would be with the
    # margin rule alone.
    st_flat = assign_styles(T, Z, thr, mar)
    yf, nf, kf = _yoy(T[cohort], st_flat[cohort], T.loc[cohort, 'role'])
    print(f"  same config, NO sticky blend (underlying signal): "
          f"{yf * 100:.1f}%  (n={nf})")
    v2_baselines()
    print("\n  --- per-role, chance-corrected (the fair comparison) ---")
    A = per_role_stability(*_v2_frame()) if _V2.exists() else None
    B = per_role_stability(T[cohort], styles[cohort], T.loc[cohort, 'role'])
    F = per_role_stability(T[cohort], st_flat[cohort], T.loc[cohort, 'role'])
    print(f"  {'role':<22}{'v2 obs':>8}{'v2 k':>7}{'v3.1 obs':>10}{'v3.1 k':>8}"
          f"{'flat k':>8}{'n':>6}")
    for role in ROLE_TENDENCY_MENU:
        b = B[B['role'] == role]
        f_ = F[F['role'] == role]
        a = A[A['role'] == role] if A is not None else None
        if b.empty:
            continue
        av = (f"{a['obs'].iloc[0] * 100:>7.1f}%{a['kappa'].iloc[0]:>7.3f}"
              if a is not None and not a.empty else f"{'—':>8}{'—':>7}")
        fk = f"{f_['kappa'].iloc[0]:>8.3f}" if not f_.empty else f"{'—':>8}"
        print(f"  {role:<22}{av}{b['obs'].iloc[0] * 100:>9.1f}%"
              f"{b['kappa'].iloc[0]:>8.3f}{fk}{b['n'].iloc[0]:>6}")
    if A is not None:
        print(f"  {'WEIGHTED MEAN':<22}"
              f"{np.average(A['obs'], weights=A['n']) * 100:>7.1f}%"
              f"{np.average(A['kappa'], weights=A['n']):>7.3f}"
              f"{np.average(B['obs'], weights=B['n']) * 100:>9.1f}%"
              f"{np.average(B['kappa'], weights=B['n']):>8.3f}"
              f"{np.average(F['kappa'], weights=F['n']):>8.3f}")

    print("\n  --- anchors ---")
    for nm in ['Balotelli', 'Yuk Jinyoung', 'Elias Franco', 'Tiago Morgado']:
        for _, r in T[T['name'] == nm].sort_values('seasonId').iterrows():
            fit = f"{r['style_fit']:.0f}" if r['style_fit'] == r['style_fit'] else '—'
            print(f"    {nm:<15} {r['seasonId']}  {r['role']:<20} "
                  f"{r['style']:<28} fit {fit}")

    if '--split-half' in sys.argv:
        split_half(T, stats, thr, mar, prior, cohort)
    else:
        print("  (split-half skipped — pass --split-half to run it)")

    print("\n[6/6] writing…", flush=True)
    ids = {}
    for role in sorted(T['role'].dropna().unique()):
        for i, nm in enumerate(sorted(T.loc[T['role'] == role, 'style'].unique())):
            ids[(role, nm)] = i
    T['style_id'] = [ids.get((r, s), -1) for r, s in zip(T['role'], T['style'])]
    T['primary_role_name'] = T['role']       # v2 column name
    T['style_name'] = T['style']             # v2 column name
    out = T[['playerId', 'seasonId', 'primary_role_name', 'style_id',
             'style_name', 'role', 'style', 'style_fit', 'style_2',
             'style_2_fit', 'mins_played', 'name', 'thin_sample']]
    out.to_parquet(_OUT, index=False)
    print(f"  saved {len(out):,} rows -> {_OUT.name}")

    prof = []
    for (role, nm), g in T[cohort].groupby(['role', 'style'], observed=True):
        dims = [f'z_{k}' for k in ROLE_TENDENCY_MENU.get(role, [])
                if f'z_{k}' in Z.columns]
        z = Z.loc[g.index, dims].mean()
        top = z.reindex(z.abs().sort_values(ascending=False).index)[:5]
        ex = (g.sort_values('mins_played', ascending=False)
                .drop_duplicates('playerId')['name'].dropna().head(6).tolist())
        prof.append({'role': role, 'style': nm, 'n': len(g),
                     'style_id': ids.get((role, nm), -1),
                     'top_z': '; '.join(f"{k[2:]}{v:+.2f}" for k, v in top.items()),
                     'exemplars': ', '.join(map(str, ex))})
    P = pd.DataFrame(prof).sort_values(['role', 'n'], ascending=[True, False])
    P.to_csv(_PROF, index=False)
    print("  saved style_profiles.csv")


if __name__ == '__main__':
    main()
