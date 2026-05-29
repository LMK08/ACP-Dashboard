"""Tier-club matcher — determines whether a club name from TM (or
elsewhere) refers to a Liga 3 / Campeonato tier club in our data.

Key complications handled:
  - Noise tokens (FC, SC, CD, etc.) stripped from disambiguation
  - B-team markers ('II', 'B', '2') REQUIRED to match: 'Vit. Guimarães'
    (Primeira Liga senior team) does NOT match our 'Vitória Guimarães II'
    despite sharing the 'guimaraes' token
  - Bidirectional subset on identifying tokens, so 'Braga' (just one
    token in TM listing) correctly matches our 'Sporting Braga II'
    (after B status check)

Usage:
    from tier_matcher import build_tier_matcher, at_our_tier
    matcher = build_tier_matcher()  # cached at module load if you want
    if matcher('Sporting Covilhã'):
        ...
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd

# Generic noise tokens that don't disambiguate clubs
_NOISE = {
    'fc', 'sc', 'cd', 'cf', 'ud', 'sl', 'gd', 'ad', 'os', 'do', 'da',
    'de', 'sad', 'fca', 'sporting', 'futebol', 'clube', 'desportivo',
    'grupo', 'club',
    # Wait — sporting/uniao etc. ARE disambiguating. Keep them.
}
# Trim: only strip tokens that genuinely add nothing distinctive
_NOISE = {'fc', 'sc', 'cd', 'cf', 'ud', 'sl', 'gd', 'ad', 'os', 'do',
           'da', 'de', 'sad', 'fca', 'futebol', 'clube', 'desportivo',
           'grupo', 'club'}

# B-team / second-team markers — REQUIRED to match cross-club for
# B teams not to collide with senior teams of the same name
_B_MARKERS = {'ii', 'iii', 'b', 'b2', 'bteam', '2'}

# Known senior-team names that share a token with one of our tier
# clubs but are NOT our tier. Stops false-positives like "Real
# Valladolid" matching our "Real SC" via the shared 'real' token.
# Normalized to lowercase + no diacritics. Whole-name match.
_SENIOR_CLUB_BLOCKLIST = {
    'real madrid', 'real betis', 'real sociedad', 'real valladolid',
    'real oviedo', 'real zaragoza', 'real murcia', 'real mallorca',
    'real burgos', 'real avila', 'real jaen', 'real union',
    'real club deportivo', 'real club celta de vigo',
    'sporting gijon', 'sporting de gijon', 'sporting kansas city',
    'sporting clube de portugal',     # senior Sporting CP, not our II
    'vitoria sport clube',            # senior Vit Guimarães
    'vitoria sport clube b',          # if TM ever lists as B
    'fc atletico madrid', 'atletico madrid',
    'atletico de madrid', 'atletico bilbao',
}


def _ascii_lower(name):
    """ASCII-folded lowercase whole-name normalization (no tokenizing)."""
    if name is None:
        return ''
    if isinstance(name, float) and pd.isna(name):
        return ''
    folded = ''.join(c for c in unicodedata.normalize('NFKD', str(name))
                     if not unicodedata.combining(c)).lower()
    # Collapse runs of non-alnum to single space, trim
    return re.sub(r'[^a-z0-9 ]+', ' ', folded).strip()


def _normalize(name):
    """Return (identifying_tokens_frozenset, is_b_team_bool)."""
    if name is None:
        return (frozenset(), False)
    if isinstance(name, float) and pd.isna(name):
        return (frozenset(), False)
    folded = ''.join(c for c in unicodedata.normalize('NFKD', str(name))
                     if not unicodedata.combining(c)).lower()
    all_tokens = set(re.findall(r'[a-z0-9]+', folded))
    is_b = bool(all_tokens & _B_MARKERS)
    identifying = all_tokens - _NOISE - _B_MARKERS
    return (frozenset(identifying), is_b)


_OUR_TIER_CATALOG = None
_DISTINCTIVE_TOKENS = None   # tokens that appear in exactly ONE our-club


def build_tier_matcher(force_refresh=False):
    """Load + cache the our-tier club catalog from raw_events.parquet.
    Returns a callable `(other_club_name) -> bool`."""
    global _OUR_TIER_CATALOG, _DISTINCTIVE_TOKENS
    if _OUR_TIER_CATALOG is None or force_refresh:
        try:
            ev_path = Path(__file__).resolve().parent / 'raw_events.parquet'
            ev = pd.read_parquet(ev_path, columns=['team.name'])
            our_clubs = ev['team.name'].dropna().unique()
            _OUR_TIER_CATALOG = []
            for c in our_clubs:
                identity, is_b = _normalize(c)
                if identity:
                    _OUR_TIER_CATALOG.append((identity, is_b))
            # Pre-compute distinctive-token set: tokens appearing in
            # exactly ONE our-club's identity. Used to safely accept
            # bare single-token TM names like 'Felgueiras' (distinctive
            # → match our 'Felgueiras 1932') vs 'Sporting' (in 3 of
            # our clubs → ambiguous, must reject).
            token_counts = {}
            for identity, _ in _OUR_TIER_CATALOG:
                for tok in identity:
                    token_counts[tok] = token_counts.get(tok, 0) + 1
            _DISTINCTIVE_TOKENS = {t for t, n in token_counts.items() if n == 1}
        except Exception:
            _OUR_TIER_CATALOG = []
            _DISTINCTIVE_TOKENS = set()
    return at_our_tier


def at_our_tier(other_club_name):
    """True iff `other_club_name` refers to a Liga 3 / Camp tier club
    that appears in our data. Handles B-team distinctions strictly:
    'Vit. Guimarães' (senior, Primeira Liga) does NOT match our
    'Vitória Guimarães II' even though they share 'guimaraes'.

    Single-token our-tier clubs (like 'Real SC' → {real}) require
    EXACT identity-set match to avoid catching global senior teams
    that share the same token ('Real Madrid', 'Real Valladolid').
    Plus an explicit blocklist for known senior teams.
    """
    if _OUR_TIER_CATALOG is None:
        build_tier_matcher()
    # Explicit blocklist for whole-name matches against known senior teams
    if _ascii_lower(other_club_name) in _SENIOR_CLUB_BLOCKLIST:
        return False
    other_id, other_is_b = _normalize(other_club_name)
    if not other_id:
        return False
    for our_id, our_is_b in _OUR_TIER_CATALOG:
        # B-team status must match — Vit. Guimarães (senior) ≠ our B team
        if our_is_b != other_is_b:
            continue
        # OUR single-token: require exact equality so 'Real SC' {real}
        # doesn't match 'Real Madrid' {real, madrid} via subset.
        if len(our_id) == 1:
            if our_id == other_id:
                return True
            continue
        # OTHER single-token AND our multi: only accept if the bare
        # token is DISTINCTIVE to this our-club (appears in only one
        # our-tier club's identity). Lets TM's 'Felgueiras' match
        # our 'Felgueiras 1932' but blocks TM's 'Sporting' from
        # matching our 'Sporting Covilhã' (sporting also in 2 other
        # our-clubs).
        if len(other_id) == 1:
            tok = next(iter(other_id))
            if tok in our_id and tok in _DISTINCTIVE_TOKENS:
                return True
            continue
        # Both multi-token: bidirectional subset handles abbreviations
        # like TM's 'Vit. Setúbal' ↔ our 'Vitória Setúbal'.
        if our_id.issubset(other_id) or other_id.issubset(our_id):
            return True
    return False
