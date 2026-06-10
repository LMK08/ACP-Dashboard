#!/usr/bin/env python3
"""Duel grading system — Phase 1: contest builder.

Turns Wyscout's double-recorded duels (one row per participant, linked by
relatedDuelId) into single CONTEST rows for the Glicko ladders:

  aerial : aerialDuel rows, symmetric. Outcome = firstTouch.
  ground : defender row (groundDuel.duelType == defensive_duel) vs
            attacker row (duelType in {offensive_duel, dribble} — take-ons
            included per design). Defender win := stoppedProgress OR
            recoveredPossession; attacker win := keptPossession OR
            progressedWithBall. Conflicting/empty flags -> 0.5 (reported).

Every pairing step prints an audit (match rate, outcome consistency,
conflict rate) — the foundation must be trusted before the ratings are.

Output: models/duels/contests.parquet
    ladder, matchId, seasonId, date, t, playerA, playerB, scoreA (1/0/.5),
    att_kind (ground only: dribble|offensive), posA, posB, heightA, heightB

Run from the Dashboard dir: python models/duels/build_contests.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_DASH = _HERE.parent.parent
_GPA_DATA = _DASH.parent.parent / 'GPA Model Project v2' / 'parquet_data'
PERIOD_OFF = {'1H': 0, '2H': 45 * 60, 'E1': 90 * 60, 'E2': 105 * 60, 'P': 120 * 60}

print("[1/4] events…", flush=True)
cols = ['id', 'matchId', 'seasonId', 'matchPeriod', 'minute', 'second',
         'team.id', 'player.id', 'player.position', 'type.primary',
         'location.x', 'possession.types',
         'groundDuel.duelType', 'groundDuel.relatedDuelId',
         'groundDuel.opponent.id', 'groundDuel.stoppedProgress',
         'groundDuel.recoveredPossession', 'groundDuel.keptPossession',
         'groundDuel.progressedWithBall',
         'aerialDuel.firstTouch', 'aerialDuel.relatedDuelId',
         'aerialDuel.opponent.id', 'aerialDuel.height',
         'aerialDuel.opponent.height']
ev = pd.concat([pd.read_parquet(_GPA_DATA / f'{f}.parquet', columns=cols)
                  for f in ['liga3_portugal_events', 'campeonato_portugal_events']],
                 ignore_index=True)
ev = ev.dropna(subset=['player.id'])
ev['player.id'] = ev['player.id'].astype(int)
ev = ev[ev['player.id'] > 0]          # 0 = Wyscout's unattributed sentinel
# situational context: zone depth (4 bands, contest event's frame) + phase
import numpy as _np
ev['zx'] = _np.clip((pd.to_numeric(ev['location.x'], errors='coerce')
                       .fillna(50) / 25.0).astype(int), 0, 3)
_CTR = {'counterattack', 'transition_high'}
_TRN = {'transition_medium', 'transition_low'}
def _phase(t):
    if isinstance(t, (list, tuple, _np.ndarray)):
        ts = set(t)
        if ts & _CTR: return 'counter'
        if ts & _TRN: return 'transition'
    return 'settled'
ev['phase'] = [_phase(t) for t in ev['possession.types'].tolist()]
ev['_t'] = (ev['matchPeriod'].map(PERIOD_OFF).fillna(0)
              + pd.to_numeric(ev['minute'], errors='coerce').fillna(0) * 60
              + pd.to_numeric(ev['second'], errors='coerce').fillna(0))
dates = (pd.read_parquet(_DASH / 'matches_summary.parquet',
                           columns=['matchId', 'dateutc'])
           .set_index('matchId')['dateutc'].to_dict())

# ---------- aerial contests -------------------------------------------------
print("[2/4] aerial contests…", flush=True)
aer = ev[ev['aerialDuel.firstTouch'].notna()].copy()
aer['rdid'] = pd.to_numeric(aer['aerialDuel.relatedDuelId'], errors='coerce')
right = aer[['id', 'matchId', 'player.id', 'aerialDuel.firstTouch',
               'player.position', 'aerialDuel.height']].rename(
    columns={'id': 'id_B', 'player.id': 'pB', 'aerialDuel.firstTouch': 'ft_B',
              'player.position': 'posB', 'aerialDuel.height': 'hB'})
pairs = aer.merge(right, left_on=['matchId', 'rdid'],
                    right_on=['matchId', 'id_B'], how='inner')
print(f"  aerial rows {len(aer):,}; paired rows {len(pairs):,} "
       f"({len(pairs)/len(aer)*100:.1f}% match rate)")
# audit: opponent.id agreement + complementary outcomes
opp_ok = (pd.to_numeric(pairs['aerialDuel.opponent.id'], errors='coerce')
            == pairs['pB']).mean()
comp = (pairs['aerialDuel.firstTouch'].astype(bool)
          != pairs['ft_B'].astype(bool)).mean()
print(f"  opponent-id agreement {opp_ok*100:.1f}%; complementary outcomes "
       f"{comp*100:.1f}% (both should be ~100%)")
# dedupe: keep the lexicographically-first side of each pair
keep = pairs[pairs['id'] < pairs['id_B']].copy()
# scoring: clean win only when exactly one side has first touch; the 8.7%
# both-False pairs (ball ran through / keeper claimed) are draws (0.5).
_fa = keep['aerialDuel.firstTouch'].astype(bool)
_fb = keep['ft_B'].astype(bool)
_aer_score = np.where(_fa & ~_fb, 1.0, np.where(_fb & ~_fa, 0.0, 0.5))
aerial = pd.DataFrame({
    'ladder': 'aerial', 'matchId': keep['matchId'],
    'seasonId': keep['seasonId'], 't': keep['_t'],
    'playerA': keep['player.id'], 'playerB': keep['pB'],
    'scoreA': _aer_score,
    'att_kind': None, 'zx': keep['zx'], 'phase': keep['phase'],
    'posA': keep['player.position'], 'posB': keep['posB'],
    'heightA': pd.to_numeric(keep['aerialDuel.height'], errors='coerce'),
    'heightB': pd.to_numeric(keep['hB'], errors='coerce')})
print(f"  aerial contests: {len(aerial):,}")

# ---------- ground contests (tackle vs carry) --------------------------------
print("[3/4] ground contests…", flush=True)
gd = ev[ev['groundDuel.duelType'].notna()].copy()
gd['rdid'] = pd.to_numeric(gd['groundDuel.relatedDuelId'], errors='coerce')
dfn = gd[gd['groundDuel.duelType'] == 'defensive_duel']
att = gd[gd['groundDuel.duelType'].isin(['offensive_duel', 'dribble'])]
print(f"  defender rows {len(dfn):,}; attacker rows {len(att):,} "
       f"(offensive {int((att['groundDuel.duelType']=='offensive_duel').sum()):,}"
       f" + dribble {int((att['groundDuel.duelType']=='dribble').sum()):,})")
attr = att[['id', 'matchId', 'player.id', 'player.position',
              'groundDuel.duelType', 'groundDuel.keptPossession',
              'groundDuel.progressedWithBall']].rename(
    columns={'id': 'id_B', 'player.id': 'pB', 'player.position': 'posB',
              'groundDuel.duelType': 'att_kind',
              'groundDuel.keptPossession': 'kept',
              'groundDuel.progressedWithBall': 'prog'})
gp = dfn.merge(attr, left_on=['matchId', 'rdid'],
                 right_on=['matchId', 'id_B'], how='inner')
print(f"  paired via relatedDuelId: {len(gp):,} "
       f"({len(gp)/max(len(dfn),1)*100:.1f}% of defender rows)")
opp_ok = (pd.to_numeric(gp['groundDuel.opponent.id'], errors='coerce')
            == gp['pB']).mean()
print(f"  opponent-id agreement {opp_ok*100:.1f}%")
# Combo-audited scoring (see git history): defender clean win = stopped
# and/or recovered with attacker retaining nothing (31.1%); attacker clean
# win = kept/progressed with defender achieving nothing (37.5%); the
# stop-but-kept stalemate (29.5%) and rare contradictions (~2%) are draws.
_stop = gp['groundDuel.stoppedProgress'] == True
_rec = gp['groundDuel.recoveredPossession'] == True
_kept = gp['kept'] == True
_prog = gp['prog'] == True
def_clean = (_stop | _rec) & ~_kept & ~_prog
att_clean = (_kept | _prog) & ~_stop & ~_rec
scoreA = np.where(def_clean, 1.0, np.where(att_clean, 0.0, 0.5))
print(f"  scoring: defender win {def_clean.mean()*100:.1f}% | attacker win "
       f"{att_clean.mean()*100:.1f}% | draw {(~def_clean & ~att_clean).mean()*100:.1f}%")
ground = pd.DataFrame({
    'ladder': 'ground', 'matchId': gp['matchId'],
    'seasonId': gp['seasonId'], 't': gp['_t'],
    'playerA': gp['player.id'], 'playerB': gp['pB'],   # A = defender (tackle)
    'scoreA': scoreA, 'att_kind': gp['att_kind'],
    'zx': gp['zx'], 'phase': gp['phase'],
    'posA': gp['player.position'], 'posB': gp['posB'],
    'heightA': np.nan, 'heightB': np.nan})
print(f"  ground contests: {len(ground):,}")

print("[4/4] save…", flush=True)
contests = pd.concat([aerial, ground], ignore_index=True)
contests = contests[(contests['playerA'] > 0) & (contests['playerB'] > 0)]
contests['date'] = contests['matchId'].map(dates)
contests = contests.dropna(subset=['date'])
contests = contests.sort_values(['date', 'matchId', 't']).reset_index(drop=True)
contests.to_parquet(_HERE / 'contests.parquet')
print(f"  {len(contests):,} contests "
       f"({(contests['ladder']=='aerial').sum():,} aerial, "
       f"{(contests['ladder']=='ground').sum():,} ground) -> contests.parquet")
