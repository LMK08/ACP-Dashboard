# app.py

import sys
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import logging
import yaml
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from YAML
def load_config():
    """Load configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("config.yaml not found, using default configuration")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config.yaml: {e}")
        return None
from mplsoccer import Pitch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.table import Table
import datetime # For Radar dates
import matplotlib.gridspec as gridspec # For Corner plots
import scipy.stats # For Radar stats percentile rank
import os # For checking logo file paths
import hashlib
import json

_TRANSFERRED_PLAYERS_PATH = os.path.join(os.path.dirname(__file__), 'transferred_players.json')

def load_transferred_players():
    """Load the list of players who transferred out."""
    try:
        with open(_TRANSFERRED_PLAYERS_PATH, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_transferred_players(players):
    """Persist the transferred players list to disk."""
    with open(_TRANSFERRED_PLAYERS_PATH, 'w') as f:
        json.dump(players, f, indent=2)

from PIL import Image # For scatter plot logos
from matplotlib.offsetbox import OffsetImage, AnnotationBbox # For scatter plot logos
from adjustText import adjust_text # For scatter plot logos
from math import pi # For player radar charts
import matplotlib.dates as mdates # <-- ADD THIS LINE
from matplotlib.gridspec import GridSpec # For player radar charts
from collections import defaultdict # For player radar calculations
import seaborn as sns # For player radar distributions
from collections import defaultdict # Make sure this is at the top with other imports
import plotly.graph_objects as go
from pitch_interactive import (plotly_shot_map, plotly_box_passes_map,
                                mpl_box_passes_map)
import io # For saving the in-memory image
# ... after your other imports ...
import base64
import pitch_visualizations as pv


# Metrics that need 3 decimal places (thousandths) instead of the default 2
THOUSANDTHS_METRICS = {'goalsPreventedPerSOT'}
WHOLE_NUMBER_METRICS = {'Defensive Area'}

def fmt_val(metric, value):
    """Format a stat value: 0 decimals for WHOLE_NUMBER_METRICS, 3 for THOUSANDTHS, 2 otherwise."""
    if metric in WHOLE_NUMBER_METRICS:
        return f"{value:.0f}"
    if metric in THOUSANDTHS_METRICS:
        return f"{value:.3f}"
    return f"{value:.2f}"

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Atlético CP Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# DARK THEME CSS INJECTION
# ==============================================================================
st.markdown("""
<style>
/* ============================================================================
   ATLÉTICO CP ANALYTICS — CREAM / TAN / BLACK THEME
   Accents: red, blue, yellow (used sparingly)
   ============================================================================ */

:root {
    --bg: #F8F5F0;
    --bg-warm: #F0ECE4;
    --card: #FFFFFF;
    --sidebar: #1a1a1a;
    --ink: #1a1a1a;
    --ink-2: #5c5650;
    --ink-3: #9a948d;
    --border: #DDD8D0;
    --border-light: #EBE7E1;
    --shadow-xs: 0 1px 3px rgba(0,0,0,0.04);
    --shadow-sm: 0 2px 6px rgba(0,0,0,0.06);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
    --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* === BASE === */
html, body, .stApp { background: var(--bg); color: var(--ink); font-family: var(--font); }
footer, footer::before { display: none !important; }

/* === SIDEBAR (dark) === */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div > div:first-child,
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--sidebar);
}
[data-testid="stSidebar"] { border-right: none; }
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }

/* Sidebar inputs & dropdowns */
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[role="combobox"],
[data-testid="stSidebar"] [data-testid="stMultiSelect"] div[role="combobox"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] input[type="text"],
[data-testid="stSidebar"] textarea {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(255,255,255,0.12) !important;
    color: #fff !important;
}
[data-testid="stSidebar"] button {
    background: rgba(255,255,255,0.12) !important;
    color: #fff !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
}
[data-testid="stSidebar"] button:hover {
    background: rgba(255,255,255,0.2) !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.08) !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] details,
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    background: transparent !important;
    color: rgba(255,255,255,0.85) !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    background: rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small {
    color: rgba(255,255,255,0.5) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: rgba(255,255,255,0.85) !important;
}

[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #fff !important; }
[data-testid="stSidebar"]::-webkit-scrollbar { width: 6px; }
[data-testid="stSidebar"]::-webkit-scrollbar-track { background: var(--sidebar); }
[data-testid="stSidebar"]::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }

/* === MAIN === */
[data-testid="stMainBlockContainer"] { background: var(--bg); padding-top: 2rem; }
.main { background: var(--bg); }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #c5bfb6; }

/* === TYPOGRAPHY === */
h1, h2, h3, h4, h5, h6 { color: var(--ink); font-weight: 600; letter-spacing: -0.3px; }
h1 { font-size: 2.2rem; font-weight: 700; }
h2 { font-size: 1.55rem; border-bottom: 1.5px solid var(--ink); padding-bottom: 0.4rem; margin-bottom: 1rem; }
h3 { font-size: 1.15rem; }
p { color: var(--ink-2); line-height: 1.65; }

[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 { color: var(--ink); }

/* === METRICS === */
[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border-light);
    border-radius: 8px;
    padding: 1.4rem;
    box-shadow: var(--shadow-xs);
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stMetric"]:hover {
    border-color: var(--border);
    box-shadow: var(--shadow-sm);
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--ink);
    font-weight: 700;
    font-size: 1.7rem;
}
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: var(--ink-3);
    font-weight: 500;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}

/* === TABLES === */
[data-testid="stDataFrame"] {
    background: var(--card) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 8px !important;
    box-shadow: var(--shadow-xs) !important;
}
[data-testid="stDataFrame"] table { background: var(--card) !important; color: var(--ink) !important; }
[data-testid="stDataFrame"] thead { background: var(--card) !important; border-bottom: 1.5px solid var(--ink) !important; }
[data-testid="stDataFrame"] thead th {
    color: var(--ink) !important;
    font-weight: 700 !important;
    text-align: center !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.3px !important;
}
[data-testid="stDataFrame"] tbody tr { background: var(--card) !important; border-bottom: 1px solid var(--border-light) !important; }
[data-testid="stDataFrame"] tbody tr:hover { background: #FDFBF8 !important; }
[data-testid="stDataFrame"] tbody td { color: var(--ink) !important; padding: 0.7rem !important; text-align: center !important; }

/* === BUTTONS === */
button {
    background: var(--ink) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.2rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s, box-shadow 0.2s !important;
}
button:hover { opacity: 0.85 !important; box-shadow: var(--shadow-sm) !important; }
button:active { opacity: 1 !important; }

[data-testid="stDownloadButton"] button { background: var(--ink) !important; }

/* === INPUTS === */
[data-testid="stSelectbox"] div[role="combobox"],
[data-testid="stMultiSelect"] div[role="combobox"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--ink) !important;
}
[data-testid="stSelectbox"] div[role="combobox"]:hover,
[data-testid="stMultiSelect"] div[role="combobox"]:hover {
    border-color: var(--ink-2) !important;
}

/* Force selected-value text dark in ALL selectboxes (baseweb select component) */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div *,
[data-testid="stSelectbox"] div[data-baseweb="select"] input,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div *,
[data-testid="stMultiSelect"] div[data-baseweb="select"] input,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div *,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] input,
[data-testid="stSidebar"] [data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-testid="stMultiSelect"] div[data-baseweb="select"] > div *,
[data-testid="stSidebar"] [data-testid="stMultiSelect"] div[data-baseweb="select"] input {
    color: var(--ink) !important;
}

/* Dropdown options / listbox popover */
[role="listbox"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
[role="option"] {
    color: var(--ink) !important;
    background: var(--card) !important;
}
[role="option"]:hover,
[role="option"][aria-selected="true"] {
    background: var(--bg-warm) !important;
    color: var(--ink) !important;
}

input[type="text"], input[type="number"], input[type="date"], textarea {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--ink) !important;
    padding: 0.55rem !important;
    font-family: var(--font) !important;
}
input:focus, textarea:focus {
    border-color: var(--ink) !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(26,26,26,0.06) !important;
}
input::placeholder, textarea::placeholder { color: var(--ink-3) !important; }

/* === RADIO / CHECKBOX === */
[data-testid="stRadio"] { padding: 0.4rem; }
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label { color: var(--ink) !important; font-weight: 500; }
[data-testid="stRadio"] input[type="radio"],
[data-testid="stCheckbox"] input[type="checkbox"] { accent-color: var(--ink) !important; cursor: pointer !important; }

/* === EXPANDERS === */
[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 8px !important;
    box-shadow: var(--shadow-xs) !important;
}
[data-testid="stExpander"] details { background: var(--card) !important; }
[data-testid="stExpander"] summary {
    color: var(--ink) !important;
    font-weight: 600 !important;
    padding: 0.9rem 1rem !important;
    background: var(--card) !important;
    border-radius: 6px !important;
}
[data-testid="stExpander"] summary:hover { background: #FDFBF8 !important; }
[data-testid="stExpander"] > div > div:nth-child(2) { padding: 1rem !important; }

/* === TABS === */
[role="tablist"] { border-bottom: 1px solid var(--border) !important; background: transparent !important; }
[role="tab"] {
    color: var(--ink-3) !important;
    background: transparent !important;
    border: none !important;
    padding: 0.7rem 1.4rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.2s !important;
}
[role="tab"]:hover { color: var(--ink) !important; }
[role="tab"][aria-selected="true"] { color: var(--ink) !important; border-bottom-color: var(--ink) !important; }

/* === DIVIDERS === */
[data-testid="stHorizontalBlock"] hr { border: none !important; border-top: 1px solid var(--border-light) !important; margin: 1.5rem 0 !important; }

/* === COLUMNS === */
[data-testid="stColumn"] { background: transparent !important; }
[data-testid="stHorizontalBlock"] { gap: 1.2rem !important; }

/* === SPINNER === */
[data-testid="stSpinner"] > div > div { border-color: var(--ink) !important; border-top-color: transparent !important; }

/* === ALERTS === */
[data-testid="stAlert"] {
    background: var(--card) !important;
    border: 1px solid var(--border-light) !important;
    border-left: 3px solid var(--ink-2) !important;
    border-radius: 6px !important;
    padding: 0.9rem 1rem !important;
}
[role="alert"],
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] { color: var(--ink) !important; }

/* === PLOTS === */
[data-testid="stPlotlyContainer"],
[data-testid="stPlotlyContainer"] > div,
canvas, .stPlotlyContainer svg { background: transparent !important; }

/* === LINKS === */
a { color: var(--ink) !important; text-decoration: none !important; font-weight: 500 !important; }
a:hover { text-decoration: underline !important; }

/* === CODE === */
code { background: var(--bg-warm) !important; color: var(--ink) !important; border-radius: 4px !important; padding: 0.2rem 0.5rem !important; font-family: 'Monaco','Courier New',monospace !important; }
pre { background: var(--bg-warm) !important; border: 1px solid var(--border-light) !important; border-radius: 8px !important; padding: 1rem !important; overflow-x: auto !important; }
pre code { background: transparent !important; padding: 0 !important; }

/* === MISC === */
::selection { background: var(--ink); color: #fff; }
[data-testid="stMarkdownContainer"] { color: var(--ink) !important; }
[role="dialog"] { background: var(--card) !important; border: 1px solid var(--border) !important; }
.stApp > header { background: var(--bg) !important; }

</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1B. SEASON CONSTANTS
# ==============================================================================
from league_config import COMPETITIONS, competition_for_season, all_season_id_map

# Build flat season map for backward compatibility
SEASON_ID_MAP = all_season_id_map()
CURRENT_SEASON_ID = 191782  # Liga 3 default
STATS_CACHE_DIR = 'stats_cache'
STATS_CACHE_VERSION = 'v14'  # Bump when stat COLUMNS or cached VALUES change (e.g. the
                             # 2026-06 minutes fixes: Camará alias dedup + Manuel Pedro
                             # override). v13 percentiles cache served stale minutes
                             # because the percentiles layer early-returns its disk cache
                             # and only this version key invalidates it.


def _stats_scope_key(season_id, frame):
    """Cache-key suffix: the season_id itself, or for All-Seasons
    (season_id is None) the league(s) present in the frame, so the
    two single-league All-Seasons views don't collide on one file."""
    if season_id is not None:
        return str(season_id)
    comps = []
    if frame is not None and 'competitionId' in getattr(frame, 'columns', []):
        comps = sorted(pd.to_numeric(frame['competitionId'], errors='coerce').dropna().astype(int).unique().tolist())
    return 'all_' + '_'.join(str(c) for c in comps) if comps else 'all'


def _parquet_safe(df):
    """Return a copy where object-dtype columns are coerced to string so the
    frame can always be serialized to parquet. The All-Seasons frame mixes
    str positions with int 0 (from fillna) in columns like primaryPosition,
    which pyarrow refuses to write; this normalizes them without touching the
    in-memory frame the caller keeps using."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str)
    return out

# ==============================================================================
# 2. DATA LOADING (with Caching)
# ==============================================================================
@st.cache_resource(ttl=86400)  # cache_resource avoids serializing large DataFrames
def load_data():
    """Load all pre-processed data files."""
    required_files = [
        'raw_events.parquet',
        'matches_summary.parquet',
        'all_match_data.pkl',
        'season_team_stats.pkl',
    ]

    # Check all files exist before loading
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        st.error(f"❌ Error: Missing data files: {', '.join(missing_files)}. Please run `process_data.py` first.")
        return None, None, None, None, None, None

    try:
        logger.info("Loading data files...")
        # Only load columns actually used in the app (reduces memory from 510MB to 250MB)
        events_columns = [
            'id', 'matchId', 'seasonId', 'competitionId', 'minute', 'second', 'matchTimestamp',
            'type.primary', 'type.secondary', 'player.id', 'player.name', 'player.position',
            'team.name', 'opponentTeam.name', 'location.x', 'location.y',
            'pass', 'pass.accurate', 'pass.endLocation.x', 'pass.endLocation.y', 'pass.length',
            'shot', 'shot.xg', 'shot.isGoal', 'shot.onTarget', 'shot.bodyPart', 'shot.postShotXg', 'shot.goalkeeper.id',
            'groundDuel.duelType', 'groundDuel.keptPossession', 'groundDuel.progressedWithBall',
            'groundDuel.recoveredPossession', 'groundDuel.stoppedProgress', 'groundDuel.takeOn',
            'aerialDuel.firstTouch', 'carry.endLocation.x', 'carry.endLocation.y',
            'possession.id', 'possession.eventIndex', 'possession.duration', 'possession.team.name', 'possession.types',
            'infraction', 'infraction.type', 'infraction.yellowCard', 'infraction.redCard',
            'is_dribble_attempt', 'is_custom_dribble_success', 'relatedEventId',
            'team.formation'
        ]
        # Load parquet, falling back to exclude competitionId if it doesn't exist yet
        try:
            raw_events_df = pd.read_parquet('raw_events.parquet', columns=events_columns)
        except Exception:
            cols_without_comp = [c for c in events_columns if c != 'competitionId']
            raw_events_df = pd.read_parquet('raw_events.parquet', columns=cols_without_comp)
        # Normalize direct free kick shots so all shot filters include them
        fk_shot_mask = (raw_events_df['type.primary'] == 'free_kick') & (raw_events_df['shot.xg'].notna())
        raw_events_df.loc[fk_shot_mask, 'type.primary'] = 'shot'
        # Apply PLAYER_ID_ALIASES so duplicate pids in raw_events flow
        # under the canonical pid. Defined below; we reference by name
        # which is fine because this load runs after the module body.
        try:
            if 'player.id' in raw_events_df.columns and PLAYER_ID_ALIASES:
                raw_events_df['player.id'] = raw_events_df['player.id'].map(
                    lambda p: PLAYER_ID_ALIASES.get(int(p), p)
                    if p is not None and not pd.isna(p) else p)
        except NameError:
            pass  # PLAYER_ID_ALIASES not yet defined at module load time
        # Backfill competitionId if not present (first run after upgrade)
        if 'competitionId' not in raw_events_df.columns:
            raw_events_df['competitionId'] = raw_events_df['seasonId'].map(competition_for_season)
        matches_summary_df = pd.read_parquet('matches_summary.parquet')
        if 'competitionId' not in matches_summary_df.columns:
            matches_summary_df['competitionId'] = matches_summary_df['seasonId'].map(competition_for_season)

        with open('all_match_data.pkl', 'rb') as f:
            all_match_data = pickle.load(f)

        with open('season_team_stats.pkl', 'rb') as f:
            season_team_stats = pickle.load(f)

        # Load pre-computed complete player minutes (all 3 sources merged)
        # Falls back to API-only minutes if pre-computed file doesn't exist
        if os.path.exists('complete_player_minutes.pkl'):
            with open('complete_player_minutes.pkl', 'rb') as f:
                player_minutes_data = pickle.load(f)
        elif os.path.exists('player_minutes_and_positions.pkl'):
            with open('player_minutes_and_positions.pkl', 'rb') as f:
                player_minutes_data = pickle.load(f)
        else:
            player_minutes_data = {}

        # Handle both old format (DataFrame) and new format (dict of DataFrames)
        if isinstance(player_minutes_data, pd.DataFrame):
            player_minutes_data = {CURRENT_SEASON_ID: player_minutes_data}

        # Apply PLAYER_ID_ALIASES to per-season minutes dataframes so a
        # duplicate pid's minutes flow under the canonical pid.
        #
        # The alias FROM pid is the stats-bearing record (that's why it's
        # aliased); the engine reads it directly. When BOTH the FROM and the
        # canonical TO pid have a minutes row in the SAME season, they are the
        # same stint duplicated by a Wyscout split (e.g. Mamadu Camará 25/26:
        # 71835=1144' + 1322978=1276'). Summing them double-counts — keep the
        # FROM row's minutes, DROP the TO row's, then remap, so the radar
        # matches the ACP index (1144'). Any leftover same-pid collisions
        # (genuine multi-position rows) still merge by summing.
        try:
            if PLAYER_ID_ALIASES:
                _alias_items = list(PLAYER_ID_ALIASES.items())
                for _sid, _mdf in list(player_minutes_data.items()):
                    if not isinstance(_mdf, pd.DataFrame) or _mdf.empty:
                        continue
                    if 'playerId' not in _mdf.columns:
                        continue
                    _mdf = _mdf.copy()
                    _pid = pd.to_numeric(_mdf['playerId'], errors='coerce')
                    _present = set(_pid.dropna().astype(int).tolist())
                    # drop canonical(TO) duplicate rows when the FROM row is also
                    # present this season — the FROM record is authoritative
                    _drop = pd.Series(False, index=_mdf.index)
                    for _from, _to in _alias_items:
                        if _from in _present and _to in _present:
                            _drop |= (_pid == _to)
                    if _drop.any():
                        _mdf = _mdf[~_drop]
                        _pid = pd.to_numeric(_mdf['playerId'], errors='coerce')
                    _mdf['playerId'] = _pid.map(
                        lambda p: PLAYER_ID_ALIASES.get(int(p), int(p))
                        if pd.notna(p) else p)
                    # merge any leftover genuine collisions per (playerId,
                    # optionally position) by summing numeric columns
                    group_cols = ['playerId']
                    if 'position' in _mdf.columns:
                        group_cols.append('position')
                    if _mdf.duplicated(subset=group_cols, keep=False).any():
                        num_cols = [c for c in _mdf.columns if c not in group_cols
                                      and pd.api.types.is_numeric_dtype(_mdf[c])]
                        non_num = [c for c in _mdf.columns if c not in group_cols
                                     and c not in num_cols]
                        agg_dict = {c: 'sum' for c in num_cols}
                        for c in non_num:
                            agg_dict[c] = 'first'
                        _mdf = (_mdf.groupby(group_cols, as_index=False)
                                     .agg(agg_dict))
                    player_minutes_data[_sid] = _mdf
        except NameError:
            pass  # PLAYER_ID_ALIASES not defined yet

        # --- Manual minutes overrides (after alias resolution) ---
        # Targeted fixes for Wyscout lineup-data errors a refresh can't correct.
        try:
            if MINUTES_OVERRIDE:
                for _sid, _mdf in list(player_minutes_data.items()):
                    if (not isinstance(_mdf, pd.DataFrame) or _mdf.empty
                            or 'playerId' not in _mdf.columns
                            or 'totalMinutes' not in _mdf.columns):
                        continue
                    _ov = {pid: mins for (pid, s), mins in MINUTES_OVERRIDE.items()
                           if s == _sid}
                    if not _ov:
                        continue
                    _mdf = _mdf.copy()
                    _pid = pd.to_numeric(_mdf['playerId'], errors='coerce')
                    for _pid_o, _mins_o in _ov.items():
                        _m = (_pid == _pid_o)
                        if _m.any():
                            _mdf.loc[_m, 'totalMinutes'] = float(_mins_o)
                    player_minutes_data[_sid] = _mdf
        except NameError:
            pass  # MINUTES_OVERRIDE not defined yet

        # --- Fill in "Unknown"/blank player names from player_details ---
        # complete_player_minutes.pkl lists ~785 players as "Unknown" while
        # player_details.pkl (the authoritative bio source the engine radar
        # reads) has their real names → names were inconsistent across the
        # dashboard. Resolve them here so the stats caches and every display
        # built on them match the radar. (Lucas 2026-06-30)
        try:
            _pdet = load_player_details()
            if _pdet is not None and not _pdet.empty:
                _nmap = {}
                for _pid_i, _r in _pdet.iterrows():
                    _nm = _r.get('shortName')
                    if not _nm or (isinstance(_nm, float) and pd.isna(_nm)):
                        _parts = [str(_r.get(_k)) for _k in ('firstName', 'lastName')
                                  if _r.get(_k) and not (isinstance(_r.get(_k), float)
                                                          and pd.isna(_r.get(_k)))]
                        _nm = ' '.join(_parts).strip()
                    if _nm:
                        _nmap[int(_pid_i)] = _nm
                _BAD_NAMES = {'unknown', 'n/a', 'nan', 'none', ''}
                for _sid, _mdf in list(player_minutes_data.items()):
                    if (not isinstance(_mdf, pd.DataFrame) or _mdf.empty
                            or 'playerId' not in _mdf.columns
                            or 'playerName' not in _mdf.columns):
                        continue
                    _bad = _mdf['playerName'].astype(str).str.strip().str.lower().isin(_BAD_NAMES)
                    if not _bad.any():
                        continue
                    _mdf = _mdf.copy()
                    _pidn = pd.to_numeric(_mdf['playerId'], errors='coerce')
                    _resolved = _pidn.map(lambda p: _nmap.get(int(p)) if pd.notna(p) else None)
                    _fillable = _bad & _resolved.notna()
                    _mdf.loc[_fillable, 'playerName'] = _resolved[_fillable]
                    player_minutes_data[_sid] = _mdf
        except Exception:
            pass

        # --- Fill in "Unknown"/blank team names from raw_events (per season) ---
        # The same incomplete complete_player_minutes records carry teamName
        # "Unknown". Every event has team.name + seasonId, so resolve the club the
        # player logged the most events for IN THAT SEASON. Restrict the (4.6M-row)
        # events groupby to only the players that need it. (Lucas 2026-06-30)
        try:
            _BAD_TM = {'unknown', 'n/a', 'nan', 'none', ''}
            _need_team = set()
            for _sid, _mdf in player_minutes_data.items():
                if (isinstance(_mdf, pd.DataFrame) and not _mdf.empty
                        and 'teamName' in _mdf.columns and 'playerId' in _mdf.columns):
                    _b = _mdf['teamName'].astype(str).str.strip().str.lower().isin(_BAD_TM)
                    if _b.any():
                        _need_team |= set(pd.to_numeric(
                            _mdf.loc[_b, 'playerId'], errors='coerce').dropna().astype(int))
            if (_need_team and raw_events_df is not None and not raw_events_df.empty
                    and {'player.id', 'team.name', 'seasonId'} <= set(raw_events_df.columns)):
                _ev = raw_events_df[['seasonId', 'player.id', 'team.name']].dropna().copy()
                _ev['player.id'] = pd.to_numeric(_ev['player.id'], errors='coerce')
                _ev = _ev[_ev['player.id'].isin(_need_team)]
                _ev['seasonId'] = pd.to_numeric(_ev['seasonId'], errors='coerce')
                _team_ev = (_ev.groupby(['seasonId', 'player.id'])['team.name']
                            .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None)
                            .to_dict())
                for _sid, _mdf in list(player_minutes_data.items()):
                    if (not isinstance(_mdf, pd.DataFrame) or _mdf.empty
                            or 'teamName' not in _mdf.columns or 'playerId' not in _mdf.columns):
                        continue
                    _b = _mdf['teamName'].astype(str).str.strip().str.lower().isin(_BAD_TM)
                    if not _b.any():
                        continue
                    _mdf = _mdf.copy()
                    _pidn = pd.to_numeric(_mdf['playerId'], errors='coerce')
                    _res = _pidn.map(lambda p: _team_ev.get((float(_sid), float(p)))
                                     if pd.notna(p) else None)
                    _fill = _b & _res.notna()
                    _mdf.loc[_fill, 'teamName'] = _res[_fill]
                    player_minutes_data[_sid] = _mdf
        except Exception:
            pass

        # Handle old season_team_stats format {team_name: stats} vs new {season_id: {team: stats}}
        # Old format has string keys (team names), new format has int keys (season IDs)
        if season_team_stats and isinstance(next(iter(season_team_stats.keys())), str):
            season_team_stats = {CURRENT_SEASON_ID: season_team_stats}

        # Load match lineups (lineup, bench, substitution data from Wyscout API)
        match_lineups = {}
        if os.path.exists('match_lineups.pkl'):
            with open('match_lineups.pkl', 'rb') as f:
                match_lineups = pickle.load(f)
            logger.info(f"Loaded lineup/substitution data for {len(match_lineups)} matches")

        logger.info(f"Loaded {len(raw_events_df)} events, {len(matches_summary_df)} matches")
        return raw_events_df, matches_summary_df, all_match_data, season_team_stats, player_minutes_data, match_lineups

    except FileNotFoundError as e:
        st.error(f"❌ Error: A data file was not found. Please run `process_data.py` first. Missing file: {e.filename}")
        logger.error(f"FileNotFoundError: {e}")
        return None, None, None, None, None, None
    except (pickle.UnpicklingError, pd.errors.ParserError) as e:
        st.error(f"❌ Error: Data file is corrupted. Please regenerate with `process_data.py`. Details: {e}")
        logger.error(f"Data corruption error: {e}")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred loading data: {e}")
        logger.exception("Unexpected error in load_data")
        return None, None, None, None, None, None


# ============================================================================
# PLAYER_ID_ALIASES — Wyscout sometimes splits one real-world player across
# two playerIds (different scrapes, different sources, mistyped name, etc.).
# Map FROM the duplicate pid → TO the canonical pid we want to keep.
#
# After alias resolution every downstream pipeline (GPA, raw_events,
# player_details, valuations, reported_fees) sees only the canonical pid.
# Player_details for the canonical pid wins the bio, so put the pid whose
# bio you want to keep on the RIGHT side of the mapping.
#
# Add a new entry by appending a line like:  <wrong_pid>: <canonical_pid>,
# with a comment naming the player + reason.
PLAYER_ID_ALIASES = {
    # Mamadu Camará at Brito (25-26 Camp). Wyscout has two records:
    # pid 71835 holds the GPA stats but DOB 1991-12-31 is the wrong
    # (older) profile; pid 1322978 has the correct DOB 2001-11-20 but
    # no stats. Remap 71835 → 1322978 so the GPA flows under the
    # correct younger bio.
    71835: 1322978,
}

# MINUTES_OVERRIDE — manual corrections for known Wyscout lineup-data errors a
# data refresh can't fix. Keyed (playerId, seasonId) -> totalMinutes. Used when
# Wyscout merged/changed a player's id on the backend so re-fetching by our
# cached id 404s (see Manuel Pedro below). Applied to player_minutes_data in
# load_data, so it survives a complete_player_minutes.pkl regeneration.
MINUTES_OVERRIDE = {
    # Manuel Pedro (AD Marco 09), Campeonato 23/24 (190230): Wyscout's lineup
    # feed dropped ~25 of his matches (read 96'); his id 273828 was later merged
    # on the backend, so a re-fetch by id 404s. His event record shows ~2,335'
    # (a near-full season of starts) — override to that. (Lucas 2026-06)
    (273828, 190230): 2335,
    # Camp 23/24 lineup-gap goalkeepers: same Wyscout gap as the 63-outfielder
    # fix (c7baa3a), but that sweep required >=500 event minutes, which these
    # GK records missed. Lineup feed shows one match (~90') while their event
    # record spans a full/partial season, so every per-90 GK metric (goals
    # conceded, saves…) was inflated 10-30x. Values are event-derived minutes
    # (same estimator as precompute_minutes.py). (2026-07)
    (135928, 190230): 2933,   # André Preto (Pevidém) — 32 matches, was 90'
    (593057, 190230): 2627,   # Diogo Figueiredo (Amarante) — 28 matches, was 96'
    (553006, 190230): 2278,   # Heitor Silva (Ribeirão) — 24 matches, was 92'
    (413326, 190230): 1971,   # Imerson Soares (Rabo Peixe) — 21 matches, was 93'
    (623567, 190230): 1774,   # Pedro Teixeira (Marítimo II) — 19 matches, was 92'
    (553543, 190230): 458,    # Pedro Palha (Vilar de Perdizes) — 5 matches, was 96'
    (968986, 190230): 280,    # Caio Mendonça (Fontinhas) — 3 matches, was 94'
}


def _resolve_pid(pid):
    """Translate any pid through PLAYER_ID_ALIASES. Pass-through for
    pids without an alias. Safe on None / NaN."""
    if pid is None:
        return pid
    try:
        if pd.isna(pid):
            return pid
    except (TypeError, ValueError):
        pass
    try:
        ipid = int(pid)
    except (TypeError, ValueError):
        return pid
    return PLAYER_ID_ALIASES.get(ipid, ipid)


@st.cache_data(ttl=86400)
def load_player_details():
    """Loads the player details (foot, height, etc.) from the pkl file."""
    try:
        with open('player_details.pkl', 'rb') as f:
            player_details_list = pickle.load(f)

        players_df = pd.DataFrame(player_details_list)
        players_df = players_df.dropna(subset=['playerId'])
        # Safe conversion with validation
        players_df['playerId'] = pd.to_numeric(players_df['playerId'], errors='coerce')
        invalid_ids = players_df['playerId'].isna().sum()
        if invalid_ids > 0:
            logger.warning(f"{invalid_ids} player IDs could not be converted to numeric")
        players_df = players_df.dropna(subset=['playerId'])
        players_df['playerId'] = players_df['playerId'].astype(int)
        # Drop alias-FROM rows so only the canonical bio remains.
        # (We never want to display the duplicate's wrong DOB/name.)
        if PLAYER_ID_ALIASES:
            n_before = len(players_df)
            players_df = players_df[~players_df['playerId'].isin(PLAYER_ID_ALIASES.keys())]
            n_dropped = n_before - len(players_df)
            if n_dropped:
                logger.info(f"PLAYER_ID_ALIASES dropped {n_dropped} duplicate bio(s)")
        players_df = players_df.set_index('playerId')
        logger.info(f"Loaded {len(players_df)} player details")
        return players_df
    except FileNotFoundError:
        st.error("❌ Error: `player_details.pkl` not found. Please run `get_player_details.py` locally and push the file.")
        logger.error("player_details.pkl not found")
        return pd.DataFrame()
    except (pickle.UnpicklingError, KeyError) as e:
        st.error(f"❌ Error loading player details: {e}")
        logger.error(f"Error loading player_details.pkl: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred loading player details: {e}")
        return pd.DataFrame()

# ==============================================================================
# 2A-GPA. GPA ACTION-VALUE DATA LOADER
# Per-player-season V breakdown from the GPA model project (LOSO-clean,
# out-of-sample). See the GPA repo's src/export_to_dashboard.py for the source.
# ==============================================================================
GPA_VALUE_CATEGORIES = [
    "Shooting", "Passing", "Receiving", "Dribbling",
    "Corner", "FreeKick", "ThrowIn", "SetPiece",
    "Interrupting", "Fouling",
    "GK_Shotstopping", "GK_Handling", "GK_Sweeping", "GK_Distribution",
    "Other", "total_v", "total_offensive_v", "gk_total_v",
]
# Map raw V category → display per-90 column name (used in radar + config.yaml).
# All 15 metrics use the "X Value" naming convention.
GPA_PER90_DISPLAY: dict[str, str] = {
    "Shooting":        "Shooting Value",
    "Passing":         "Passing Value",
    "Receiving":       "Receiving Value",
    "Dribbling":       "Dribbling Value",
    "SetPiece":        "Set Piece Value",
    "Corner":          "Corner Value",
    "FreeKick":        "Free Kick Value",
    "ThrowIn":         "Throw-In Value",
    "Interrupting":    "Interrupting Value",
    "Fouling":         "Fouling Value",
    "Other":           "Other Value",
    "GK_Shotstopping": "Shot-Stopping Value",
    "GK_Handling":     "Handling Value",
    "GK_Sweeping":     "Sweeping Value",
    "GK_Distribution": "GK Distribution Value",
    "total_v":         "Total Value",
    "total_offensive_v": "Total Offensive Value",
    "gk_total_v":        "GK Total Value",
}
GPA_PER90_COLS = [GPA_PER90_DISPLAY.get(c, f"{c}_per_90") for c in GPA_VALUE_CATEGORIES]

@st.cache_data(ttl=86400)
def load_gpa_values():
    """Load per-(playerId, seasonId) GPA action-value categories.

    Returns DataFrame with columns:
        playerId, seasonId, competitionId, name, position, position_group,
        mins_played, primary_share, <category>, <category>_per_90

    Returns empty DataFrame if the file is missing so the rest of the app
    still works unchanged.
    """
    path = os.path.join(os.path.dirname(__file__), 'gpa_player_season_values.parquet')
    if not os.path.exists(path):
        logger.info("gpa_player_season_values.parquet not found — GPA features disabled")
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        df['playerId'] = pd.to_numeric(df['playerId'], errors='coerce').astype('Int64')
        df['seasonId'] = pd.to_numeric(df['seasonId'], errors='coerce').astype('Int64')
        # Apply PLAYER_ID_ALIASES so duplicates' GPA flows to the canonical pid.
        # Then minutes-weighted-merge any (canonical_pid, seasonId) collisions
        # so two pids that BOTH have data for the same season combine correctly
        # (sum mins, weighted-average the per-90 V columns by mins).
        if PLAYER_ID_ALIASES:
            df['playerId'] = df['playerId'].map(
                lambda p: PLAYER_ID_ALIASES.get(int(p), int(p))
                if p is not None and not pd.isna(p) else p
            ).astype('Int64')
            # Identify which (pid, sid) groups have >1 row after the remap
            dup_mask = df.duplicated(subset=['playerId', 'seasonId'], keep=False)
            if dup_mask.any():
                dups = df[dup_mask].copy()
                singles = df[~dup_mask].copy()
                # Columns to handle: numeric value/per-90 columns get
                # minutes-weighted average; mins_played sums; everything else
                # keeps the first non-null.
                key = ['playerId', 'seasonId']
                num_cols = [c for c in dups.columns
                              if c not in key and pd.api.types.is_numeric_dtype(dups[c])]
                non_num = [c for c in dups.columns if c not in key and c not in num_cols]
                merged_rows = []
                for (pid, sid), grp in dups.groupby(key, dropna=False):
                    row = {'playerId': pid, 'seasonId': sid}
                    mins = grp['mins_played'].fillna(0).clip(lower=0)
                    total_mins = float(mins.sum())
                    for c in num_cols:
                        if c == 'mins_played':
                            row[c] = total_mins
                        elif total_mins > 0:
                            # minutes-weighted average for per-90 stats
                            vals = grp[c].fillna(0).astype(float).values
                            row[c] = float((vals * mins.values).sum() / total_mins)
                        else:
                            row[c] = float(grp[c].mean())
                    for c in non_num:
                        first_non_null = grp[c].dropna()
                        row[c] = first_non_null.iloc[0] if len(first_non_null) else None
                    merged_rows.append(row)
                df = pd.concat([singles, pd.DataFrame(merged_rows)],
                                 ignore_index=True)
                logger.info(f"PLAYER_ID_ALIASES: minutes-weighted-merged "
                              f"{len(merged_rows)} (pid, season) collisions")
        logger.info(f"Loaded GPA values: {len(df):,} rows × {df.shape[1]} cols")
        return df
    except Exception as e:
        logger.error(f"Failed to load GPA values parquet: {e}")
        return pd.DataFrame()


# ============================================================================
# Defensive Responsibility (DefR) — per-(player, season) per-type metrics.
# Loaded from models/defr/defr_per_player_season.parquet and merged into the
# stats DF as per-90 "DefR <action>" columns so they show in Overall Season
# Stats / Player Analysis and can be swapped onto radars.
# ============================================================================
# parquet count-column  ->  dashboard display metric name
DEFR_TYPE_TO_DISPLAY = {
    'defr_interception': 'DefR Interceptions',
    'defr_clearance':    'DefR Clearances',
    'defr_tackle':       'DefR Tackles',
    'defr_recovery':     'DefR Recoveries',
    'defr_def_aerial':   'DefR Aerials',
}
# DWAE — defensive QUALITY (wins above expectation on contested
# engagements, matchup-conditioned), complementing DefR's workload.
DWAE_TO_DISPLAY = {
    'defr_dwae':         'Def Wins Above Exp',
    'defr_dwae_tackle':  'Tackle Wins AE',
    'defr_dwae_aerial':  'Aerial Wins AE',
}
# DefR Value Conceded (né DefR OBV) — responsibility-weighted opposition
# on-ball value through the player's defensive domain, per 90. LOWER =
# more suppressive. Descriptive accountability (≈2/3 team exposure —
# measured); compare within position/league, never read as isolated skill.
DEFR_VALUE_TO_DISPLAY = {
    'obv_conceded':      'DefR Value Conceded',
}
DEFR_DISPLAY_METRICS = (list(DEFR_TYPE_TO_DISPLAY.values()) + ['DefR Total']
                          + list(DWAE_TO_DISPLAY.values())
                          + list(DEFR_VALUE_TO_DISPLAY.values()))
_DEFR_PCTL_INVERT = {'DefR Value Conceded'}   # lower = better in percentile mode
# Base defensive metric (as shown on radars / stat tables)  ->  DefR equivalent.
# Lets the radar "DefR mode" swap each defensive axis to its DefR value.
DEFR_RADAR_MAP = {
    'Interceptions': 'DefR Interceptions',
    'Clearances': 'DefR Clearances',
    'Defensive duels': 'DefR Tackles',
    'Defensive duels successful': 'DefR Tackles',
    'Defensive duels successful %': 'DefR Tackles',
    'Sliding tackles': 'DefR Tackles',
    'Sliding tackles successful': 'DefR Tackles',
    'Sliding tackles successful %': 'DefR Tackles',
    'Recoveries': 'DefR Recoveries',
    'Recoveries Opp Half': 'DefR Recoveries',
    'Counterpressing Recoveries': 'DefR Recoveries',
    'Aerial duels': 'DefR Aerials',
    'Aerial duels successful': 'DefR Aerials',
    'Aerial duels successful %': 'DefR Aerials',
}


def _season_id_list(active):
    """Normalize get_season_ids_for_selection output (None | int | list)
    to a list of ints. Empty list = All Seasons (no season filter)."""
    if active is None:
        return []
    if isinstance(active, (list, tuple, set)):
        return [int(s) for s in active if s is not None and pd.notna(s)]
    return [int(active)]


@st.cache_data(ttl=86400)
def load_box_passes():
    """Per-event passes into the attacking box with GPA pass values
    (models/creation/box_entry_passes.parquet). Empty DF if missing."""
    path = os.path.join(os.path.dirname(__file__),
                          'models', 'creation', 'box_entry_passes.parquet')
    if not os.path.exists(path):
        logger.info("box_entry_passes.parquet not found — creation chart disabled")
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        df['player.id'] = pd.to_numeric(df['player.id'], errors='coerce').astype('Int64')
        df['seasonId'] = pd.to_numeric(df['seasonId'], errors='coerce').astype('Int64')
        if PLAYER_ID_ALIASES:
            df['player.id'] = df['player.id'].map(
                lambda p: PLAYER_ID_ALIASES.get(int(p), int(p)) if pd.notna(p) else p)
        return df
    except Exception as e:
        logger.warning(f"load_box_passes failed: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=86400)
def load_defr_values():
    """Load per-(playerId, seasonId) DefR metrics. Returns empty DF if the
    file is missing so the rest of the app is unaffected."""
    path = os.path.join(os.path.dirname(__file__),
                          'models', 'defr', 'defr_per_player_season.parquet')
    if not os.path.exists(path):
        logger.info("defr_per_player_season.parquet not found — DefR disabled")
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        df['playerId'] = pd.to_numeric(df['playerId'], errors='coerce').astype('Int64')
        df['seasonId'] = pd.to_numeric(df['seasonId'], errors='coerce').astype('Int64')
        if PLAYER_ID_ALIASES:
            df['playerId'] = df['playerId'].map(
                lambda p: PLAYER_ID_ALIASES.get(int(p), int(p))
                if p is not None and not pd.isna(p) else p).astype('Int64')
            # sum the count columns on (pid, season) collisions
            cnt = [c for c in df.columns if c.startswith(('defr_', 'act_', 'exp_',
                     'gk_', 'obv_')) and not c.endswith('_p90') and c not in (
                     'defr_per90', 'defr_per90_vs_position', 'defr_adj', 'defr_career')]
            cnt = [c for c in cnt if pd.api.types.is_numeric_dtype(df[c])]
            cnt += ['mins_played']
            dup = df.duplicated(subset=['playerId', 'seasonId'], keep=False)
            if dup.any():
                singles = df[~dup]
                agg = (df[dup].groupby(['playerId', 'seasonId'], as_index=False)
                         [list(dict.fromkeys(cnt))].sum())
                # keep position from the longest-minutes row
                pos = (df[dup].sort_values('mins_played')
                         .drop_duplicates(['playerId', 'seasonId'], keep='last')
                         [['playerId', 'seasonId', 'position']])
                agg = agg.merge(pos, on=['playerId', 'seasonId'], how='left')
                df = pd.concat([singles, agg], ignore_index=True)
        logger.info(f"Loaded DefR values: {len(df):,} rows × {df.shape[1]} cols")
        return df
    except Exception as e:
        logger.error(f"Failed to load DefR parquet: {e}")
        return pd.DataFrame()


def merge_defr_values_into_stats(player_stats_df, season_ids=None, comp_ids=None):
    """Merge DefR per-90 columns (DefR Interceptions/Clearances/Tackles/
    Recoveries/Aerials/Total) into the stats DF. Outfield-only (no GKs).

    DefR counts are summed across the active seasons per player and per-90
    is recomputed from the totals (so multi-season views aggregate
    correctly). Merged on playerId; missing players get 0."""
    if player_stats_df is None or len(player_stats_df) == 0:
        return player_stats_df
    defr = load_defr_values()
    if defr is None or defr.empty:
        return player_stats_df
    if season_ids is not None:
        sids = [int(s) for s in (season_ids if isinstance(season_ids, (list, tuple, set))
                                   else [season_ids])]
        defr = defr[defr['seasonId'].isin(sids)]
    if defr.empty:
        return player_stats_df

    count_cols = (list(DEFR_TYPE_TO_DISPLAY.keys()) + ['defr']
                    + list(DWAE_TO_DISPLAY.keys())
                    + list(DEFR_VALUE_TO_DISPLAY.keys()))
    count_cols = [c for c in count_cols if c in defr.columns]
    g = defr.copy()
    g['playerId'] = pd.to_numeric(g['playerId'], errors='coerce').astype('Int64')
    g['_mins'] = pd.to_numeric(g['mins_played'], errors='coerce').fillna(0)
    agg = g.groupby('playerId', as_index=False)[count_cols + ['_mins']].sum()
    mins90 = (agg['_mins'] / 90.0).clip(lower=1e-9)
    new_cols = {}
    for raw, disp in DEFR_TYPE_TO_DISPLAY.items():
        if raw in agg.columns:
            new_cols[disp] = agg[raw] / mins90
    if 'defr' in agg.columns:
        new_cols['DefR Total'] = agg['defr'] / mins90
    for raw, disp in DWAE_TO_DISPLAY.items():
        if raw in agg.columns:
            new_cols[disp] = agg[raw] / mins90
    for raw, disp in DEFR_VALUE_TO_DISPLAY.items():
        if raw in agg.columns:
            new_cols[disp] = agg[raw] / mins90
    defr_sub = pd.DataFrame({'playerId': agg['playerId'], **new_cols})

    df = player_stats_df
    had_index = df.index.name == 'playerId'
    if had_index:
        df = df.reset_index()
    df = df.copy()
    df['playerId'] = pd.to_numeric(df['playerId'], errors='coerce').astype('Int64')
    drop = [c for c in defr_sub.columns if c != 'playerId' and c in df.columns]
    if drop:
        df = df.drop(columns=drop)
    merged = df.merge(defr_sub, on='playerId', how='left')
    for c in defr_sub.columns:
        if c != 'playerId':
            merged[c] = pd.to_numeric(merged[c], errors='coerce').fillna(0.0)
    merged = _add_defr_percentiles(merged)
    if had_index:
        merged = merged.set_index('playerId')
    return merged


def _defr_pos_bucket(pos):
    """Collapse a Wyscout position code to a broad bucket for DefR
    percentile ranking (peer group)."""
    if pos is None or (isinstance(pos, float) and pd.isna(pos)):
        return 'OTHER'
    p = str(pos).upper()
    if p == 'GK':
        return 'GK'
    if 'CB' in p:
        return 'CB'
    if p in ('LB', 'RB', 'LWB', 'RWB') or p.startswith(('LB', 'RB')):
        return 'FB'
    if p in ('CF', 'ST', 'SS'):
        return 'ST'
    if any(t in p for t in ('AMF', 'AM', 'LW', 'RW', 'LWF', 'RWF', 'LAMF', 'RAMF')):
        return 'AM_W'
    if 'M' in p:   # DMF/CMF/etc
        return 'CM'
    return 'OTHER'


def _add_defr_percentiles(df):
    """Add `<DefR metric>_percentile` columns, ranked within broad position
    bucket among 500+ minute players — so the radar's percentile mode can
    render DefR axes."""
    metrics = [m for m in DEFR_DISPLAY_METRICS if m in df.columns]
    if not metrics:
        return df
    pos_col = 'primaryPosition' if 'primaryPosition' in df.columns else (
        'position' if 'position' in df.columns else None)
    if pos_col is None:
        return df
    bucket = df[pos_col].map(_defr_pos_bucket)
    if 'totalMinutes' in df.columns:
        qual = pd.to_numeric(df['totalMinutes'], errors='coerce').fillna(0) >= 500
    else:
        qual = pd.Series(True, index=df.index)
    for m in metrics:
        pctl = pd.Series(np.nan, index=df.index)
        for b, idx in df.groupby(bucket).groups.items():
            sub = [i for i in idx if qual.get(i, False)]
            if len(sub) < 3:
                continue
            ranks = df.loc[sub, m].rank(pct=True)
            if m in _DEFR_PCTL_INVERT:      # lower = better (value conceded)
                ranks = 1.0 - ranks
            pctl.loc[sub] = ranks
        df[m + '_percentile'] = pctl.fillna(0.5)
    return df


# ============================================================================
# Cross-tier translation factors (Liga 3 ↔ Campeonato)
# ----------------------------------------------------------------------------
# Two sources for "translate this player's metric to the other tier":
#   1. Opta Power Rankings → uniform multiplier based on league-average strength
#      (data: opta_ratings.parquet, refreshed by fetch_opta_ratings.py)
#   2. Empirical median ratios → from players who actually played ≥500 min in
#      both leagues across our 7 covered seasons (data: derived on-demand from
#      raw_events + complete_player_minutes)
#
# Both return a scalar multiplier (applied uniformly across all metrics for
# the Opta source; per-metric for the empirical source).
# ============================================================================
@st.cache_data(ttl=86400)
def load_player_engine():
    """Unified ACP engine export (models/ratings/build_player_engine.py):
    one row per rated player-season with rating + abs, axis percentiles,
    role shares, projection + band + factor columns, duel ladders, team.

    Returns (DataFrame, meta_dict). Empty DF + {} if files are missing so
    the rest of the app works unchanged."""
    base = os.path.dirname(__file__)
    path = os.path.join(base, 'models', 'ratings', 'player_engine.parquet')
    meta_path = os.path.join(base, 'models', 'ratings', 'player_engine_meta.json')
    if not os.path.exists(path):
        logger.info("player_engine.parquet not found — engine card disabled")
        return pd.DataFrame(), {}
    try:
        df = pd.read_parquet(path)
        df['playerId'] = pd.to_numeric(df['playerId'], errors='coerce').astype('Int64')
        df['seasonId'] = pd.to_numeric(df['seasonId'], errors='coerce').astype('Int64')
        if PLAYER_ID_ALIASES:
            df['playerId'] = df['playerId'].map(
                lambda p: PLAYER_ID_ALIASES.get(int(p), int(p))
                if p is not None and not pd.isna(p) else p
            ).astype('Int64')
        # Engine projected value (EUR), computed ONCE here so the bio
        # headline and the analysis tables can never drift: perf =
        # percentile of projection_abs within the projection universe
        # (abs scale — Camp recruit discount already applied, so NO
        # extra Camp penalty) × career-NPV age multiplier, through the
        # fee-calibrated CVI→EUR curve. No reliability ramp: the
        # projection is already evidence-weighted internally.
        _ROLE2CVI = {'Striker': 'ST', 'Wide Attacker': 'AM_WG',
                     'Advanced Midfielder': 'AM_WG', 'Deep Midfielder': 'CM',
                     'Wide Defender': 'FB', 'Central Defender': 'CB'}
        _pool = df['projection_abs'].dropna()
        # global price temper (Lucas 2026-06-12): the engine values read
        # a touch rich for this market — scale the whole curve down 20%
        _ENGINE_VALUE_TEMPER = 0.8

        def _eng_eur(r):
            pa = r.get('projection_abs')
            if pa is None or pd.isna(pa) or len(_pool) == 0:
                return None
            perf = float((_pool < float(pa)).mean()
                          + 0.5 * (_pool == float(pa)).mean()) * 100.0
            grp = _ROLE2CVI.get(r.get('role'))
            am = _cvi_age_value_multiplier(r.get('age'), grp)
            v = cvi_to_projected_eur(perf * am, position_group=grp,
                                       competition_id=None)
            return None if v is None else v * _ENGINE_VALUE_TEMPER
        df['engine_value_eur'] = df.apply(_eng_eur, axis=1)
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        return df, meta
    except Exception:
        logger.exception("Failed to load player_engine.parquet")
        return pd.DataFrame(), {}


ENGINE_DISPLAY_METRICS = ['ACP Rating', 'ACP Rating (abs)', 'ACP Projection',
                           'ACP Projection (abs)', 'Projection Band',
                           'Evidence Weight', 'Offensive Value %',
                           'Def Quality Grade %', 'Aerial Grade %',
                           'Ground Def Grade %', 'Engine RAPM %',
                           'Def Volume Grade %', 'Off Duel Grade %',
                           'Engine Set Piece %', 'Shooting Grade %',
                           'Creating Grade %', 'Linking Grade %',
                           'Receiving Grade %', 'Dribbling Grade %',
                           'Aerial WOE /90', 'Ground WOE /90',
                           'Engine Value EUR']


# Shared "How these ratings work" copy — rendered in an expander on the Player
# Profile (above the engine card) and the Player Analysis role board.
RATINGS_EXPLAINER_MD = """
Both ratings sit on a **0–100 scale where 50 = league average and ±17 ≈ one
standard deviation** (≈85 is elite, ≈30 well below average), and both are
**role-fair** — a player is only compared to others in the same role, so a
defender is never docked for not scoring. **(abs)** puts Campeonato and Liga 3
on one scale (a Camp rating is shifted ~7 pts down to its Liga-3 equivalent) for
cross-league comparison.

**ACP Index — how good has this player been?** *(descriptive)*

A role-weighted blend of what a player actually did, each part scored as a
percentile among same-role players:
- **Attacking value** (largest share) — on-ball expected-goal value: shooting,
  creating, linking passes, receiving, carrying.
- **Defensive contribution** — how much defensive responsibility they take on,
  and how well they win their duels/situations versus expectation.
- **Plus-minus** — the team's xG swing while they're on the pitch (intangibles
  the ball-events miss).

The split shifts by role (strikers lean on attack, centre-backs on defence).
It's **minutes-shrunk** — thin-minutes players are pulled toward replacement
level; the **career** version (recency-weighted) is the steadier scouting
number. Set-pieces are shown separately; goalkeepers use the legacy keeper
system (the engine is outfield-only).

**ACP Projection — how good will they be next season?** *(predictive)*

A forecast of next season's Index — a recruitment view, not a summary of the
past. It blends the player's **own career form** (recency-weighted), **how much
we trust it** (an evidence weight from career minutes), and an **age curve**
(rising to a peak around 26, declining after).
- **Thin evidence regresses toward the league** — few minutes or one hot season
  gets pulled back toward replacement level; established players stay close to
  their own record.
- Every player with a **≥90-minute season** gets one, off their most recent
  season, with a **± band** (uncertainty) and a **"seasons ago"** marker — a
  player last seen a few seasons back keeps the same central estimate but a
  **wider band**, not a lower number.

*In short: the **Index** shows who's been best; the **Projection** shows who's
the best bet going forward, with the uncertainty made explicit.*
"""


def merge_engine_values_into_stats(player_stats_df, season_ids=None, comp_ids=None):
    """Merge ACP engine columns (rating / projection / axis percentiles)
    into the stats DF on playerId. Scope-aware: keeps engine rows from
    the active seasons; a winter mover keeps his latest-season row.
    Engine metrics are levels/rates — never season-totaled."""
    if player_stats_df is None or len(player_stats_df) == 0:
        return player_stats_df
    eng, _meta = load_player_engine()
    if eng is None or eng.empty:
        return player_stats_df
    e = eng.copy()
    if season_ids is not None:
        sids = [int(s) for s in (season_ids if isinstance(season_ids, (list, tuple, set))
                                   else [season_ids])]
        e = e[e['seasonId'].isin(sids)]
    if e.empty:
        return player_stats_df
    # Pick ONE engine row per player. seasonId is NOT chronological across
    # leagues (Camp 23/24=190230 > L3 24/25=190090), so a plain seasonId-max can
    # land on an older Campeonato row with no projection — e.g. M. Konaté in an
    # All-Seasons view (season_ids=None, so both leagues' rows are present).
    # Prefer the row that carries a projection (the player's chronological-
    # latest), then fall back to seasonId / minutes. (Lucas 2026-06-24)
    e = e.assign(_has_proj=e['projection'].notna())
    e = (e.sort_values(['playerId', '_has_proj', 'seasonId', 'mins_played'])
           .drop_duplicates('playerId', keep='last')
           .drop(columns='_has_proj'))
    out = pd.DataFrame({
        'playerId': e['playerId'],
        'ACP Rating': e['acp_rating'],
        'ACP Rating (abs)': e['acp_rating_abs'],
        'ACP Projection': e['projection'],
        'ACP Projection (abs)': e['projection_abs'],
        'Projection Band': e['band_sd'],
        'Evidence Weight': e['w_evidence'],
        'Offensive Value %': e['off_pct'] * 100.0,
        'Def Quality Grade %': e['qual_pct'] * 100.0,
        'Engine RAPM %': e['rapm_pct'] * 100.0,
        'Def Volume Grade %': e['defr_pct'] * 100.0,
        'Off Duel Grade %': e['datt_pct'] * 100.0,
        'Engine Set Piece %': e['setpiece_pct'] * 100.0,
        'Engine Value EUR': e['engine_value_eur'],
        'Aerial Grade %': e['aerial_grade_pct'] * 100.0,
        'Ground Def Grade %': e['ground_grade_pct'] * 100.0,
        'Shooting Grade %': e['Shooting_pct'] * 100.0,
        'Creating Grade %': e['Creating_pct'] * 100.0,
        'Linking Grade %': e['Linking_pct'] * 100.0,
        # raw pass-split VALUES (goals/90) — the Creating/Linking split of
        # passing value, surfaced as individual metrics (Lucas 2026-06-24)
        'Creating Value': e['raw_Creating90'],
        'Linking Value': e['raw_Linking90'],
        'Receiving Grade %': e['Receiving_pct'] * 100.0,
        'Dribbling Grade %': e['Dribbling_pct'] * 100.0,
        'Aerial WOE /90': e['woe_aerial_p90'],
        'Ground WOE /90': e['woe_ground_p90'],
    })
    df = player_stats_df
    had_index = df.index.name == 'playerId'
    if had_index:
        df = df.reset_index()
    df['playerId'] = pd.to_numeric(df['playerId'], errors='coerce').astype('Int64')
    df = df.merge(out, on='playerId', how='left')
    if had_index:
        df = df.set_index('playerId')
    return df


@st.cache_data(ttl=86400)
def load_opta_ratings():
    """Load opta_ratings.parquet. Returns empty DF if missing."""
    path = os.path.join(os.path.dirname(__file__), 'opta_ratings.parquet')
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.error(f"Failed to load opta_ratings.parquet: {e}")
        return pd.DataFrame()


# Map our internal competition IDs to Opta's domesticLeagueName strings.
OPTA_LEAGUE_NAME_BY_COMP = {
    43324: 'Liga 3',
    702:   'Campeonato de Portugal Prio',
}


def opta_league_strength(comp_id: int) -> float | None:
    """Mean current Opta rating of all clubs in the given league."""
    name = OPTA_LEAGUE_NAME_BY_COMP.get(int(comp_id))
    if not name:
        return None
    df = load_opta_ratings()
    if df.empty or 'domesticLeagueName' not in df.columns:
        return None
    sub = df[df['domesticLeagueName'] == name]
    if sub.empty:
        return None
    return float(sub['currentRating'].mean())


def _opta_norm_key(s):
    """Normalize a team name for fuzzy lookup against Opta clubName."""
    import re as _re
    return _re.sub(r'[^a-z0-9]+', '', str(s).lower().strip())


@st.cache_data(ttl=86400)
def build_opta_team_strength_map() -> dict:
    """Return a dict: {normalized_name OR raw_name → currentRating}.
    Cached; the caller wraps it in a lookup function via
    make_opta_team_strength_lookup() to keep the cached value picklable.
    """
    df = load_opta_ratings()
    if df.empty or 'clubName' not in df.columns or 'currentRating' not in df.columns:
        return {}
    out = {}
    for _, row in df.iterrows():
        nm = row.get('clubName')
        rt = row.get('currentRating')
        if pd.isna(nm) or pd.isna(rt):
            continue
        out[str(nm)] = float(rt)
        out[_opta_norm_key(nm)] = float(rt)
    return out


def make_opta_team_strength_lookup():
    """Return a callable team_name → float Opta rating (or None).
    Uses the cached map; tries raw + normalized name variants since
    Opta and Wyscout spell some clubs slightly differently."""
    m = build_opta_team_strength_map()
    def _lookup(team_name):
        if team_name is None:
            return None
        try:
            if pd.isna(team_name):
                return None
        except (TypeError, ValueError):
            pass
        v = m.get(str(team_name))
        if v is not None:
            return v
        try:
            return m.get(_opta_norm_key(team_name))
        except Exception:
            return None
    return _lookup


def opta_translation_multiplier(source_comp_id: int, target_comp_id: int) -> float | None:
    """Multiplier to apply to per-90 metrics when translating from source
    league to target league via Opta strength. None if data is missing.

    Interpretation: target_strength / source_strength. Above 1.0 means the
    target league is stronger; below 1.0 means weaker. We MULTIPLY a
    player's source-league per-90 by this to estimate target-league per-90
    — i.e. a player in a weaker league has their numbers SCALED DOWN when
    projected into a stronger one. (User-facing semantics: the projection
    asks "how would this player's output translate if the opposition were
    of target-league strength?")
    """
    src = opta_league_strength(source_comp_id)
    tgt = opta_league_strength(target_comp_id)
    if src is None or tgt is None or src == 0:
        return None
    return tgt / src


@st.cache_data(ttl=86400)
def compute_empirical_translation_factors(
    _raw_events_df, _player_minutes_data,
    source_comp_id: int, target_comp_id: int,
    min_minutes: int = 500,
) -> pd.DataFrame:
    """Empirical per-metric translation ratios from cross-tier movers.

    Strategy: for every player who has ≥min_minutes total minutes in BOTH
    source_comp_id and target_comp_id across all our covered seasons,
    compute their per-90 stats in each league, then take the median
    (target_per_90 / source_per_90) ratio per metric.

    Returns a DataFrame with columns:
        metric, n_movers, median_ratio
    indexed by metric name. Returns empty if not enough movers.
    """
    # Get season IDs per comp.
    src_seasons = list(COMPETITIONS.get(source_comp_id, {}).get('seasons', {}).keys())
    tgt_seasons = list(COMPETITIONS.get(target_comp_id, {}).get('seasons', {}).keys())
    if not src_seasons or not tgt_seasons:
        return pd.DataFrame(columns=['metric', 'n_movers', 'median_ratio'])

    # Aggregate per-90 stats per league.
    def _stats_for(season_ids, comp_id):
        from_seasons = get_season_player_minutes(
            _player_minutes_data, season_ids, comp_ids=[comp_id]
        )
        events = get_filtered_events(_raw_events_df, season_ids, [comp_id])
        if from_seasons.empty or events.empty:
            return pd.DataFrame()
        return calculate_all_player_stats(events, from_seasons, season_id=None,
                                           cache_version=STATS_CACHE_VERSION)

    src_stats = _stats_for(src_seasons, source_comp_id)
    tgt_stats = _stats_for(tgt_seasons, target_comp_id)
    if src_stats.empty or tgt_stats.empty:
        return pd.DataFrame(columns=['metric', 'n_movers', 'median_ratio'])

    # Filter to players with ≥min_minutes in BOTH leagues.
    src_eligible = src_stats[src_stats.get('totalMinutes', 0) >= min_minutes].copy()
    tgt_eligible = tgt_stats[tgt_stats.get('totalMinutes', 0) >= min_minutes].copy()
    common = set(src_eligible['playerId']) & set(tgt_eligible['playerId'])
    if len(common) < 30:
        # Too few movers to trust per-metric medians.
        return pd.DataFrame(columns=['metric', 'n_movers', 'median_ratio'])

    src_kept = src_eligible[src_eligible['playerId'].isin(common)].set_index('playerId')
    tgt_kept = tgt_eligible[tgt_eligible['playerId'].isin(common)].set_index('playerId')

    # For each numeric column present in BOTH frames, compute the
    # per-player ratio and median across movers.
    rows = []
    common_cols = set(src_kept.columns) & set(tgt_kept.columns)
    for col in common_cols:
        if not pd.api.types.is_numeric_dtype(src_kept[col]) \
                or not pd.api.types.is_numeric_dtype(tgt_kept[col]):
            continue
        if col in ('playerId', 'totalMinutes', 'posMinutes'):
            continue
        a = src_kept[col].astype(float)
        b = tgt_kept[col].astype(float)
        # Ratio per player, dropping zeros and NaNs in the denominator.
        ratios = (b / a.where(a > 0)).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratios) < 30:
            continue
        rows.append({
            'metric': col,
            'n_movers': int(len(ratios)),
            'median_ratio': float(ratios.median()),
        })
    out = pd.DataFrame(rows).set_index('metric').sort_index()
    return out


def get_gpa_values_filtered(gpa_df, season_ids=None, comp_ids=None):
    """Return GPA values filtered by active season + competition selection.

    Args:
        gpa_df: full GPA DataFrame from load_gpa_values()
        season_ids: list of season IDs; if None/empty → all seasons
        comp_ids: list of competition IDs; if None/empty → all competitions

    When the user has picked a single season, returns that season's rows directly.
    When multiple seasons are active (e.g. "All Seasons"), aggregates raw V sums
    across seasons per playerId and recomputes per-90 values using summed mins.
    """
    if gpa_df is None or gpa_df.empty:
        return pd.DataFrame()

    # Defensively coerce scalar inputs to lists. get_season_ids_for_selection
    # can return a single int for single-competition selections.
    if isinstance(season_ids, (int, np.integer)):
        season_ids = [int(season_ids)]
    if isinstance(comp_ids, (int, np.integer)):
        comp_ids = [int(comp_ids)]

    df = gpa_df.copy()
    if comp_ids:
        df = df[df['competitionId'].isin(list(comp_ids))]
    if season_ids:
        df = df[df['seasonId'].isin(list(season_ids))]
    if df.empty:
        return df

    # If only one season is active after filtering, no aggregation needed.
    n_seasons = df['seasonId'].nunique()
    if n_seasons <= 1:
        return df.reset_index(drop=True)

    # Aggregate raw V + mins across seasons per player, then recompute /90.
    raw_v_cols = [c for c in GPA_VALUE_CATEGORIES if c in df.columns]
    agg_spec = {c: 'sum' for c in raw_v_cols}
    agg_spec['mins_played'] = 'sum'
    # Keep most-recent metadata (name/position may change over seasons)
    agg_spec['name'] = 'last'
    agg_spec['position'] = 'last'
    agg_spec['position_group'] = 'last'
    # primary_share: minutes-weighted mean
    df = df.sort_values('seasonId')
    out = df.groupby('playerId', as_index=False).agg(agg_spec)
    # Minutes-weighted primary_share
    ps = (df.assign(_w=df['primary_share'] * df['mins_played'])
             .groupby('playerId').agg(_w=('_w','sum'), m=('mins_played','sum')))
    ps['primary_share'] = ps['_w'] / ps['m'].clip(lower=1)
    out = out.merge(ps[['primary_share']], on='playerId', how='left')

    # Recompute per-90 from the aggregated totals, using the display names
    # for the 9 core metrics and the _per_90 suffix for auxiliary ones.
    mins = out['mins_played'].clip(lower=1)
    for cat in raw_v_cols:
        col = GPA_PER90_DISPLAY.get(cat, f'{cat}_per_90')
        out[col] = out[cat] * 90.0 / mins

    # Preserve position-based masking for the totals (outfield vs GK).
    if 'position_group' in out.columns:
        is_gk_agg = out['position_group'].eq('GK')
        for col in ('Total Value', 'Total Offensive Value'):
            if col in out.columns:
                out.loc[is_gk_agg, col] = 0.0
        if 'GK Total Value' in out.columns:
            out.loc[~is_gk_agg, 'GK Total Value'] = 0.0

    # Virtual identifiers for the aggregated view
    out['seasonId'] = pd.NA
    out['competitionId'] = (comp_ids[0] if comp_ids and len(comp_ids) == 1 else pd.NA)
    return out.reset_index(drop=True)


def merge_gpa_values_into_stats(player_stats_df, season_ids=None, comp_ids=None):
    """Merge GPA "X Value" columns into player_stats_df.

    V data is filtered to the active (season × competition) selection and
    aggregated across seasons when multiple are active (sum raw V + mins,
    recompute /90). Merged on playerId.

    Missing GPA rows (players with events but no V data, e.g. under-500-min
    sample or pre-21/22) get 0 for every Value column so percentile math
    doesn't drop them entirely.

    Returns the stats DF unchanged if V data or stats DF is empty.
    """
    if player_stats_df is None or len(player_stats_df) == 0:
        return player_stats_df
    gpa_df = load_gpa_values()
    if gpa_df is None or gpa_df.empty:
        return player_stats_df

    v_filtered = get_gpa_values_filtered(gpa_df, season_ids=season_ids, comp_ids=comp_ids)
    if v_filtered.empty:
        return player_stats_df

    value_cols = [c for c in v_filtered.columns if c.endswith("Value")]
    if not value_cols:
        return player_stats_df

    v_sub = v_filtered[["playerId"] + value_cols].copy()
    v_sub["playerId"] = pd.to_numeric(v_sub["playerId"], errors="coerce").astype("Int64")
    v_sub = v_sub.dropna(subset=["playerId"]).drop_duplicates(subset=["playerId"], keep="last")

    df = player_stats_df
    had_index = df.index.name == "playerId"
    if had_index:
        df = df.reset_index()
    df = df.copy()
    df["playerId"] = pd.to_numeric(df["playerId"], errors="coerce").astype("Int64")

    # Drop any existing V columns before merge (idempotent)
    existing = [c for c in value_cols if c in df.columns]
    if existing:
        df = df.drop(columns=existing)

    merged = df.merge(v_sub, on="playerId", how="left")
    for col in value_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    if had_index:
        merged = merged.set_index("playerId")
    return merged


# ==============================================================================
# 2B. SEASON FILTERING HELPERS
# ==============================================================================
def get_season_events(raw_events_df, season_id):
    """Filter events by season_id(s), or return all if None."""
    if season_id is None:
        return raw_events_df
    if isinstance(season_id, list):
        return raw_events_df[raw_events_df['seasonId'].isin(season_id)]
    return raw_events_df[raw_events_df['seasonId'] == season_id]

@st.cache_resource(ttl=86400, show_spinner=False, max_entries=3)
def _get_filtered_events_cached(_events_df, season_key, comp_key):
    """Cache wrapper keyed on the hashable season/comp tuple. Uses
    cache_RESOURCE (returns the SAME object, no copy) — a season's events
    are ~657 MB, and cache_data was deserializing a full copy on EVERY
    rerun (the dominant per-interaction cost). Downstream consumers only
    read/slice/.copy() the frame (never mutate in place), so sharing one
    read-only instance across reruns is safe.

    max_entries=3 (LRU) is load-bearing: the master frame is ~10.5 GB deep
    and the per-scope filtered copies total ~21 GB across all 9 scopes —
    unbounded, the warm Space sat at ~31.5 GB on 32 GB hardware and
    segfaulted (exit 139) on any allocation spike (2026-07-14). A scope
    miss re-filters in a few seconds; the disk caches keep everything else
    fast. Keep the prewarm list (in _prewarm_scope_caches) no larger than
    this bound or the warm loop just churns the LRU."""
    return _filter_events_impl(_events_df, season_key, comp_key)


def get_filtered_events(events_df, season_ids, comp_ids):
    """Filter events by season_id(s) AND competition IDs in a single pass,
    cached on the scope so repeat navigation doesn't re-scan the frame."""
    season_key = (tuple(season_ids) if isinstance(season_ids, (list, tuple, set))
                   else season_ids)
    comp_key = tuple(comp_ids) if isinstance(comp_ids, (list, tuple, set)) else comp_ids
    return _get_filtered_events_cached(events_df, season_key, comp_key)


def _filter_events_impl(events_df, season_ids, comp_ids):
    """Single-pass boolean-mask filter. season_ids None = all seasons;
    comp_ids None (or covering all competitions) = all leagues."""
    mask = None
    if season_ids is not None:
        if isinstance(season_ids, (list, tuple, set)):
            mask = events_df['seasonId'].isin(list(season_ids))
        else:
            mask = events_df['seasonId'] == season_ids
    if comp_ids is not None and len(comp_ids) != len(COMPETITIONS):
        if 'competitionId' in events_df.columns:
            comp_mask = events_df['competitionId'].isin(comp_ids)
        elif 'seasonId' in events_df.columns:
            valid_seasons = set()
            for cid in comp_ids:
                if cid in COMPETITIONS:
                    valid_seasons.update(COMPETITIONS[cid]["seasons"].keys())
            comp_mask = events_df['seasonId'].isin(valid_seasons)
        else:
            comp_mask = None
        if comp_mask is not None:
            mask = comp_mask if mask is None else (mask & comp_mask)
    if mask is None:
        return events_df
    return events_df[mask]

def get_season_matches(matches_summary_df, season_id):
    """Filter matches by season_id(s), or return all if None."""
    if season_id is None:
        return matches_summary_df
    if isinstance(season_id, list):
        return matches_summary_df[matches_summary_df['seasonId'].isin(season_id)]
    return matches_summary_df[matches_summary_df['seasonId'] == season_id]

def _season_ids_for_comps(comp_ids):
    """Return set of season IDs belonging to the given competition IDs."""
    sids = set()
    if comp_ids is not None:
        for cid in comp_ids:
            if cid in COMPETITIONS:
                sids.update(COMPETITIONS[cid]["seasons"].keys())
    return sids

@st.cache_data(ttl=86400, show_spinner=False)
def _get_season_player_minutes_cached(_pmd, season_key, comp_key):
    return _season_player_minutes_impl(
        _pmd,
        list(season_key) if isinstance(season_key, tuple) else season_key,
        list(comp_key) if isinstance(comp_key, tuple) else comp_key)


def get_season_player_minutes(player_minutes_data, season_id, comp_ids=None):
    """Cached wrapper over the per-season minutes aggregation, keyed on the
    scope (the {season: df} dict is session-constant)."""
    if isinstance(player_minutes_data, pd.DataFrame):
        return player_minutes_data
    season_key = (tuple(season_id) if isinstance(season_id, (list, tuple, set))
                   else season_id)
    comp_key = tuple(comp_ids) if isinstance(comp_ids, (list, tuple, set)) else comp_ids
    return _get_season_player_minutes_cached(player_minutes_data, season_key, comp_key)


def _season_player_minutes_impl(player_minutes_data, season_id, comp_ids=None):
    """Get player minutes for a season. Returns DataFrame.
    player_minutes_data is {season_id: DataFrame}.
    If season_id is None, combine all seasons (filtered by comp_ids if provided).
    Accepts a single season_id (int) or a list of season_ids.
    """
    if isinstance(player_minutes_data, pd.DataFrame):
        return player_minutes_data
    if season_id is None:
        # Combine all seasons, but only those belonging to selected leagues
        valid_sids = _season_ids_for_comps(comp_ids) if comp_ids and len(comp_ids) < len(COMPETITIONS) else None
        all_dfs = []
        for sid, df in player_minutes_data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                if valid_sids is None or sid in valid_sids:
                    all_dfs.append(df)
        if not all_dfs:
            return pd.DataFrame()
        combined = pd.concat(all_dfs)
        return combined.groupby('playerId').agg({
            'playerName': 'first',
            'teamName': 'first',
            'primaryPosition': 'first',
            'totalMinutes': 'sum'
        }).reset_index()
    if isinstance(season_id, list):
        dfs = [player_minutes_data.get(sid, pd.DataFrame()) for sid in season_id]
        dfs = [df for df in dfs if isinstance(df, pd.DataFrame) and not df.empty]
        if not dfs:
            return pd.DataFrame()
        combined = pd.concat(dfs)
        return combined.groupby('playerId').agg({
            'playerName': 'first',
            'teamName': 'first',
            'primaryPosition': 'first',
            'totalMinutes': 'sum'
        }).reset_index()
    return player_minutes_data.get(season_id, pd.DataFrame())

def get_season_team_stats(season_team_stats, season_id, comp_ids=None):
    """Get team stats for a season. Returns {team: stats} dict.
    season_team_stats is {season_id: {team: stats}}.
    If season_id is None, merges all seasons (filtered by comp_ids if provided).
    """
    if season_id is None:
        valid_sids = _season_ids_for_comps(comp_ids) if comp_ids and len(comp_ids) < len(COMPETITIONS) else None
        merged = {}
        for sid in sorted(season_team_stats.keys()):
            if valid_sids is None or sid in valid_sids:
                merged.update(season_team_stats.get(sid, {}))
        return merged
    if isinstance(season_id, list):
        merged = {}
        for sid in sorted(season_id):
            merged.update(season_team_stats.get(sid, {}))
        return merged
    return season_team_stats.get(season_id, {})

def league_selector(section_key):
    """Render a league selector in the sidebar. Returns list of competition IDs."""
    options = ["Liga 3", "Campeonato"]
    # Guard against a stale "Both" (or any other invalid) value lingering in
    # session_state from a previous version of this selector, which would make
    # selectbox raise because the value is no longer a valid option.
    state_key = f"league_select_{section_key}"
    if st.session_state.get(state_key) not in options:
        st.session_state.pop(state_key, None)
    selected = st.sidebar.selectbox(
        "League",
        options,
        index=0,
        key=state_key
    )
    for comp_id, comp_config in COMPETITIONS.items():
        if comp_config["name"] == selected:
            return [comp_id]
    # Fallback: first competition (single-league list).
    return [next(iter(COMPETITIONS.keys()))]


def get_league_label(comp_ids):
    """Return a display label for the selected league(s)."""
    if len(comp_ids) == len(COMPETITIONS):
        return "Liga 3 + Campeonato"
    return " + ".join(COMPETITIONS[c]["name"] for c in comp_ids if c in COMPETITIONS)


def filter_by_league(df, comp_ids, matches_summary_df=None):
    """Filter a DataFrame by competition IDs.
    If df has 'competitionId', filter directly.
    Otherwise, join via matchId from matches_summary_df.
    If comp_ids covers all competitions, return unfiltered.
    """
    if comp_ids is None or len(comp_ids) == len(COMPETITIONS):
        return df
    if 'competitionId' in df.columns:
        return df[df['competitionId'].isin(comp_ids)]
    if 'matchId' in df.columns and matches_summary_df is not None:
        valid_matches = matches_summary_df[
            matches_summary_df['competitionId'].isin(comp_ids)
        ]['matchId']
        return df[df['matchId'].isin(valid_matches)]
    if 'seasonId' in df.columns:
        valid_seasons = set()
        for cid in comp_ids:
            if cid in COMPETITIONS:
                valid_seasons.update(COMPETITIONS[cid]["seasons"].keys())
        return df[df['seasonId'].isin(valid_seasons)]
    return df


def get_season_ids_for_selection(selected_season_id, comp_ids):
    """Given a selected season_id and comp_ids, return all season IDs that should be included.
    When 'Both' leagues are selected and they share the same season label (e.g. '2025/26'),
    return both league's season IDs for that year.
    """
    if selected_season_id is None:
        return None  # All seasons
    if comp_ids is None or len(comp_ids) <= 1:
        return selected_season_id
    # Find the label for the selected season
    target_label = SEASON_ID_MAP.get(selected_season_id, "")
    matching_ids = []
    for cid in comp_ids:
        if cid in COMPETITIONS:
            for sid, label in COMPETITIONS[cid]["seasons"].items():
                if label == target_label:
                    matching_ids.append(sid)
    if len(matching_ids) <= 1:
        return selected_season_id
    return matching_ids


def season_selector(section_key, include_all_seasons=False, comp_ids=None):
    """Render a season selector in the sidebar. Returns season_id (int) or None for 'All Seasons'.
    If comp_ids is provided, only shows seasons for those competitions.
    """
    if comp_ids is not None:
        available_seasons = {}
        for cid in comp_ids:
            if cid in COMPETITIONS:
                available_seasons.update(COMPETITIONS[cid]["seasons"])
    else:
        available_seasons = SEASON_ID_MAP

    # Deduplicate display names (e.g. both leagues have "2025/26")
    # Use an ordered dict to preserve season order (newest first)
    unique_labels = list(dict.fromkeys(available_seasons.values()))
    options = unique_labels
    if include_all_seasons:
        options = ["All Seasons"] + options

    session_key = f"season_select_{section_key}"
    default_idx = 1 if include_all_seasons else 0

    selected_label = st.sidebar.selectbox(
        "Season",
        options,
        index=default_idx,
        key=session_key
    )

    if selected_label == "All Seasons":
        return None
    # Reverse lookup: label -> season_id (return first match from available seasons)
    for sid, label in available_seasons.items():
        if label == selected_label:
            return sid
    # Fallback
    return list(available_seasons.keys())[0] if available_seasons else CURRENT_SEASON_ID


# ==============================================================================
# 2C. STAGE DETECTION + FILTERING
# ==============================================================================
# Liga 3 has Regular (1st stage) + Promotion + Maintenance (combined N+S).
# Campeonato has Regular (1st stage) + Promotion playoff.
# Detection is dynamic by roundId: the largest round per competition-season is
# the first stage. Remaining rounds are classified by team count (Liga 3) or
# by being any non-first-stage round (Campeonato).

STAGE_ALL = "All Stages"
STAGE_REGULAR = "Regular (1st stage)"
STAGE_PROMOTION = "Promotion (2nd stage)"
STAGE_MAINTENANCE = "Maintenance"
STAGE_PLAYOFF = "Promotion playoff"
_STAGE_ORDER = {
    STAGE_REGULAR: 0,
    STAGE_PROMOTION: 1,
    STAGE_PLAYOFF: 1,
    STAGE_MAINTENANCE: 2,
}


@st.cache_data
def compute_stage_map(matches_summary_df, comp_id, season_id):
    """For a single (competition, season), return {roundId: stage_label}.

    The first-stage round is the one with the most matches. Remaining rounds
    are classified by competition and team count:
      Liga 3 (43324): ≤ 9 teams → Promotion, more → Maintenance.
      Campeonato (702): anything not first-stage → Promotion playoff.
    """
    if matches_summary_df is None or matches_summary_df.empty:
        return {}
    sub = matches_summary_df[
        (matches_summary_df.get('competitionId') == comp_id)
        & (matches_summary_df.get('seasonId') == season_id)
    ]
    if sub.empty or 'roundId' not in sub.columns:
        return {}

    counts = sub.groupby('roundId').size()
    if len(counts) == 0:
        return {}
    first_stage = counts.idxmax()
    out: dict = {int(first_stage): STAGE_REGULAR}

    for rid in counts.index:
        if rid == first_stage:
            continue
        rid_matches = sub[sub['roundId'] == rid]
        teams = set(rid_matches['homeTeamName'].dropna().tolist()
                    + rid_matches['awayTeamName'].dropna().tolist())
        if comp_id == 43324:  # Liga 3
            if len(teams) <= 9:
                out[int(rid)] = STAGE_PROMOTION
            else:
                out[int(rid)] = STAGE_MAINTENANCE
        elif comp_id == 702:  # Campeonato
            out[int(rid)] = STAGE_PLAYOFF
        else:
            out[int(rid)] = f"Stage {rid}"
    return out


def _coerce_comp_season(comp_ids, season_ids):
    """Normalise selection inputs into lists of ints."""
    if comp_ids is None:
        cids = []
    elif isinstance(comp_ids, (list, tuple, set)):
        cids = [int(c) for c in comp_ids if c is not None]
    else:
        cids = [int(comp_ids)]
    if season_ids is None:
        sids = []
    elif isinstance(season_ids, (list, tuple, set)):
        sids = [int(s) for s in season_ids if s is not None]
    else:
        sids = [int(season_ids)]
    return cids, sids


def get_available_stages(matches_summary_df, comp_ids, season_ids):
    """Return an ordered list of stage labels available for the current
    competition/season selection. Used to populate the stage selector."""
    cids, sids = _coerce_comp_season(comp_ids, season_ids)
    seen: list[str] = []
    for cid in cids:
        for sid in sids:
            stage_map = compute_stage_map(matches_summary_df, cid, sid)
            for lbl in stage_map.values():
                if lbl not in seen:
                    seen.append(lbl)
    seen.sort(key=lambda s: _STAGE_ORDER.get(s, 99))
    return seen


def stage_selector(section_key, matches_summary_df, comp_ids, season_ids):
    """Sidebar stage selector. Returns the selected stage label, or
    STAGE_ALL when there's only one stage available (selector hidden)."""
    stages = get_available_stages(matches_summary_df, comp_ids, season_ids)
    if len(stages) <= 1:
        return STAGE_ALL
    options = [STAGE_ALL] + stages
    return st.sidebar.selectbox(
        "Stage:",
        options,
        index=0,
        key=f"{section_key}_stage_selector",
    )


def filter_by_stage(events_df, matches_df, matches_summary_df,
                     comp_ids, season_ids, stage_label):
    """Filter events and matches to the chosen stage.

    Returns (events_df, matches_df) unchanged when STAGE_ALL is selected.
    For a specific stage, aggregates target roundIds across every
    (comp_id, season_id) pair in the selection."""
    if stage_label in (STAGE_ALL, None) or events_df is None:
        return events_df, matches_df

    cids, sids = _coerce_comp_season(comp_ids, season_ids)
    target_rounds: set = set()
    for cid in cids:
        for sid in sids:
            stage_map = compute_stage_map(matches_summary_df, cid, sid)
            for rid, lbl in stage_map.items():
                if lbl == stage_label:
                    target_rounds.add(int(rid))
    if not target_rounds:
        return events_df, matches_df

    new_matches = matches_df[matches_df['roundId'].astype('Int64').isin(target_rounds)].copy() \
        if 'roundId' in matches_df.columns else matches_df
    if 'matchId' in events_df.columns and not new_matches.empty:
        new_events = events_df[events_df['matchId'].isin(new_matches['matchId'])].copy()
    else:
        new_events = events_df
    return new_events, new_matches


def _calculate_age(birth_date):
    """Helper function to calculate age from birth_date string."""
    if not birth_date or pd.isna(birth_date):
        return "N/A"
    try:
        today = datetime.date.today()
        birth = datetime.datetime.strptime(birth_date, '%Y-%m-%d').date()
        # Calculate age as a float
        age_in_days = (today - birth).days
        age = age_in_days / 365.25
        return age # Return the float
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse birth date '{birth_date}': {e}")
        return "N/A"

@st.cache_data
def get_player_minutes_by_position(_events_df, player_id, _player_match_log_df=None):
    """
    Calculate estimated minutes spent at each position for a player.
    Uses proportional allocation of events per position within each match,
    scaled to match-log totals when available.
    """
    player_events = _events_df[_events_df['player.id'] == player_id].copy()

    if player_events.empty or 'player.position' not in player_events.columns:
        return pd.DataFrame(columns=['Position', 'Minutes', 'Percentage'])

    player_events = player_events.dropna(subset=['player.position'])
    player_events = player_events[player_events['player.position'].astype(str).str.strip() != '']

    if player_events.empty:
        return pd.DataFrame(columns=['Position', 'Minutes', 'Percentage'])

    # Count events per (matchId, position)
    match_pos_counts = player_events.groupby(['matchId', 'player.position']).size().reset_index(name='event_count')
    match_totals = match_pos_counts.groupby('matchId')['event_count'].sum().reset_index(name='total_events')
    match_pos_counts = match_pos_counts.merge(match_totals, on='matchId')
    match_pos_counts['proportion'] = match_pos_counts['event_count'] / match_pos_counts['total_events']

    # Estimate minutes per match from events (last minute - first minute)
    match_minutes = player_events.groupby('matchId')['minute'].agg(['min', 'max']).reset_index()
    match_minutes['match_minutes'] = (match_minutes['max'] - match_minutes['min']).clip(lower=1)

    # Scale event-derived minutes to match the match-log total for better accuracy
    if _player_match_log_df is not None and not _player_match_log_df.empty and 'Minutes' in _player_match_log_df.columns:
        total_from_log = pd.to_numeric(_player_match_log_df['Minutes'], errors='coerce').sum()
        total_from_events = match_minutes['match_minutes'].sum()
        if total_from_events > 0 and total_from_log > 0:
            match_minutes['match_minutes'] = match_minutes['match_minutes'] * (total_from_log / total_from_events)

    match_pos_counts = match_pos_counts.merge(match_minutes[['matchId', 'match_minutes']], on='matchId')
    match_pos_counts['position_minutes'] = match_pos_counts['proportion'] * match_pos_counts['match_minutes']

    # Aggregate across all matches
    position_totals = match_pos_counts.groupby('player.position')['position_minutes'].sum().reset_index()
    position_totals.columns = ['Position', 'Minutes']

    total = position_totals['Minutes'].sum()
    position_totals['Percentage'] = (position_totals['Minutes'] / total * 100) if total > 0 else 0
    position_totals['Minutes'] = position_totals['Minutes'].round(0).astype(int)
    position_totals['Percentage'] = position_totals['Percentage'].round(1)
    position_totals = position_totals.sort_values('Minutes', ascending=False).reset_index(drop=True)

    return position_totals

@st.cache_data
def get_all_players_minutes_by_position(_events_df):
    """
    Batch version of get_player_minutes_by_position: computes estimated
    position-minutes for ALL players at once from event data.
    Returns a DataFrame with columns: playerId, Position, Minutes.
    """
    if _events_df.empty or 'player.position' not in _events_df.columns:
        return pd.DataFrame(columns=['playerId', 'Position', 'Minutes'])

    df = _events_df.dropna(subset=['player.id', 'player.position']).copy()
    df = df[df['player.position'].astype(str).str.strip() != '']
    if df.empty:
        return pd.DataFrame(columns=['playerId', 'Position', 'Minutes'])

    # Count events per (player, match, position)
    match_pos_counts = df.groupby(['player.id', 'matchId', 'player.position']).size().reset_index(name='event_count')
    match_totals = match_pos_counts.groupby(['player.id', 'matchId'])['event_count'].sum().reset_index(name='total_events')
    match_pos_counts = match_pos_counts.merge(match_totals, on=['player.id', 'matchId'])
    match_pos_counts['proportion'] = match_pos_counts['event_count'] / match_pos_counts['total_events']

    # Estimate minutes per (player, match) from events
    match_minutes = df.groupby(['player.id', 'matchId'])['minute'].agg(['min', 'max']).reset_index()
    match_minutes['match_minutes'] = (match_minutes['max'] - match_minutes['min']).clip(lower=1)

    match_pos_counts = match_pos_counts.merge(match_minutes[['player.id', 'matchId', 'match_minutes']], on=['player.id', 'matchId'])
    match_pos_counts['position_minutes'] = match_pos_counts['proportion'] * match_pos_counts['match_minutes']

    # Aggregate across all matches per player per position
    result = match_pos_counts.groupby(['player.id', 'player.position'])['position_minutes'].sum().reset_index()
    result.columns = ['playerId', 'Position', 'Minutes']
    result['Minutes'] = result['Minutes'].round(0).astype(int)
    result = result.sort_values(['playerId', 'Minutes'], ascending=[True, False]).reset_index(drop=True)

    return result

@st.cache_data(ttl=86400)
def get_player_match_stats(player_name, _all_match_data, _matches_summary_df, season_id=None):
    """
    Goes through all match data and extracts the individual match stats
    for a single selected player.
    """
    player_matches = []

    # Create a quick lookup map for match info
    match_info_map = _matches_summary_df.set_index('matchId').to_dict('index')
    
    for match_id, match_data in _all_match_data.items():
        if not match_data or 'player_stats' not in match_data:
            continue
            
        home_df = match_data['player_stats'].get('home')
        away_df = match_data['player_stats'].get('away')
        
        player_stats_series = None
        opponent = "N/A"
        
        match_info = match_info_map.get(match_id, {})
        home_team = match_info.get('homeTeamName', 'Home')
        away_team = match_info.get('awayTeamName', 'Away')
        
        if home_df is not None and player_name in home_df.index:
            player_stats_series = home_df.loc[player_name]
            opponent = f"vs. {away_team} (H)"
        elif away_df is not None and player_name in away_df.index:
            player_stats_series = away_df.loc[player_name]
            opponent = f"vs. {home_team} (A)"
            
        if player_stats_series is not None:
            # Convert series to a dictionary and add match info
            stats_dict = player_stats_series.to_dict()
            stats_dict['Match'] = opponent
            stats_dict['Date'] = match_info.get('dateutc', 'N/A')
            stats_dict['Score'] = match_info.get('score', 'N/A')
            player_matches.append(stats_dict)

    if not player_matches:
        return pd.DataFrame()
        
    # Create and format the DataFrame
    match_log_df = pd.DataFrame(player_matches)
    
    # Format the date
    if 'Date' in match_log_df.columns:
        match_log_df['Date'] = pd.to_datetime(match_log_df['Date'], errors='coerce')

    # Reorder columns to put match info first
    cols_to_front = ['Date', 'Match', 'Score', 'Minutes']
    all_cols = cols_to_front + [col for col in match_log_df.columns if col not in cols_to_front]
    match_log_df = match_log_df[all_cols]

    # Sort by date first (while still datetime), then convert to string
    match_log_df = match_log_df.sort_values(by='Date', ascending=False, na_position='last')
    if 'Date' in match_log_df.columns:
        match_log_df['Date'] = match_log_df['Date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else 'N/A')

    # Fill NaN in numeric columns only (after date is already a string)
    match_log_df = match_log_df.fillna(0)
    
    return match_log_df

@st.cache_data
def calculate_player_profile_stats(_raw_events_df, _player_minutes_df):
    """
    A new, streamlined function to calculate ONLY the key stats for player profiles.
    Calculates: npxG, xA, xT, Progressive Passes, Deep Completions, and GK Stats.
    """
    print("--- STARTING: Streamlined player profile stats ---")
    
    events_df = _raw_events_df.copy()
    # Start with our base player list
    combined_df = _player_minutes_df.copy()

    # Ensure player.id is int for all merging
    events_df['player.id'] = pd.to_numeric(events_df['player.id'], errors='coerce')
    events_df = events_df.dropna(subset=['player.id'])
    events_df['player.id'] = events_df['player.id'].astype(int)

    # --- 1. Calculate npxG, xAOP, xASP ---
    print("Step 1: Calculating npxG, xAOP, xASP...")
    try:
        shots_df = events_df[
            (events_df['shot.xg'].notna()) &
            (events_df['type.primary'] != 'penalty')
        ].copy()
        npxg_totals = shots_df.groupby('player.id')['shot.xg'].sum().reset_index().rename(columns={'shot.xg': 'npxG'})

        events_df['shot_event_id'] = np.where(events_df['shot.xg'].notna(), events_df['id'], np.nan)
        events_df['next_shot_id'] = events_df.groupby('matchId')['shot_event_id'].bfill()
        shot_xg_map = events_df[events_df['shot.xg'].notna()].set_index('id')['shot.xg'].to_dict()

        assists_df = events_df[events_df.get('type.secondary', pd.Series(dtype='object')).apply(lambda x: isinstance(x, (list, np.ndarray)) and 'shot_assist' in x)].copy()
        assists_df['xA'] = assists_df['next_shot_id'].map(shot_xg_map)
        set_piece_types = ['corner', 'free_kick', 'throw_in', 'goal_kick']
        assists_df['assist_type'] = np.where(assists_df['type.primary'].isin(set_piece_types), 'xASP', 'xAOP')
        
        xa_split_totals = assists_df.groupby(['player.id', 'assist_type'])['xA'].sum()
        xa_final_df = xa_split_totals.unstack(fill_value=0).reset_index()

        final_stats_df = pd.merge(npxg_totals, xa_final_df, on='player.id', how='outer')
        combined_df = pd.merge(combined_df, final_stats_df, left_on='playerId', right_on='player.id', how='left')
        if 'player.id' in combined_df.columns: combined_df = combined_df.drop(columns=['player.id'])
    except Exception as e:
        print(f"  -> ❌ ERROR (Step 1): {e}")

    # --- 2. Calculate Deep Completions and Progressive Passes ---
    print("Step 2: Calculating Deep Completions and Progressive Passes...")
    try:
        passes_df = events_df[
            (events_df['type.primary'] == 'pass') & (events_df.get('pass.accurate') == True)
        ].dropna(subset=['location.x', 'pass.endLocation.x', 'player.id']).copy()
        
        passes_df['end_x_m'] = passes_df['pass.endLocation.x'] * 1.05
        passes_df['end_y_m'] = passes_df['pass.endLocation.y'] * 0.68
        passes_df['dist_to_goal_center'] = np.sqrt((passes_df['end_x_m'] - 105)**2 + (passes_df['end_y_m'] - 34)**2)
        passes_df['is_cross'] = passes_df.get('type.secondary', pd.Series(dtype='object')).apply(lambda x: isinstance(x, (list, np.ndarray)) and 'cross' in x)
        passes_df['is_deep_completion'] = (passes_df['dist_to_goal_center'] <= 20) & (passes_df['is_cross'] == False)
        deep_completions = passes_df.groupby('player.id')['is_deep_completion'].sum().reset_index().rename(columns={'is_deep_completion': 'Deep Completions'})

        start_x = passes_df['location.x']; end_x = passes_df['pass.endLocation.x']
        cond1 = (start_x < 50) & (end_x < 50) & (end_x - start_x >= 30)
        cond2 = (start_x < 50) & (end_x >= 50) & (end_x - start_x >= 15)
        cond3 = (start_x >= 50) & (end_x >= 50) & (end_x - start_x >= 10)
        passes_df['is_progressive_pass'] = cond1 | cond2 | cond3
        progressive_passes = passes_df.groupby('player.id')['is_progressive_pass'].sum().reset_index().rename(columns={'is_progressive_pass': 'Progressive Passes'})

        new_metrics_df = pd.merge(deep_completions, progressive_passes, on='player.id', how='outer')
        combined_df = pd.merge(combined_df, new_metrics_df, left_on='playerId', right_on='player.id', how='left')
        if 'player.id' in combined_df.columns: combined_df = combined_df.drop(columns=['player.id'])
    except Exception as e:
        print(f"  -> ❌ ERROR (Step 2): {e}")

    # --- 3. Calculate Goalkeeper Stats ---
    print("Step 3: Calculating Goalkeeper stats...")
    try:
        gk_ids = events_df[events_df.get('player.position') == 'GK']['player.id'].dropna().unique().astype(int)
        gk_events_df = events_df[events_df['player.id'].isin(gk_ids)].copy()
        
        shots_faced_df = events_df[(events_df.get('type.primary') == 'shot') & (events_df.get('shot.onTarget') == True) & (events_df.get('shot.goalkeeper.id').notna())].copy()
        shots_faced_df['shot.goalkeeper.id'] = shots_faced_df['shot.goalkeeper.id'].astype(int)
        gk_shot_stopping_stats = shots_faced_df.groupby('shot.goalkeeper.id').agg(shotsOnTargetAgainst=('shot.isGoal', 'count'), goalsConceded=('shot.isGoal', 'sum'), psxG_faced=('shot.postShotXg', 'sum')).reset_index().rename(columns={'shot.goalkeeper.id': 'player.id'})
        if not gk_shot_stopping_stats.empty:
            gk_shot_stopping_stats['goalsPrevented'] = gk_shot_stopping_stats['psxG_faced'] - gk_shot_stopping_stats['goalsConceded']
            gk_shot_stopping_stats['goalsPreventedPerSOT'] = (gk_shot_stopping_stats['goalsPrevented'] / gk_shot_stopping_stats['shotsOnTargetAgainst']).fillna(0)
            gk_shot_stopping_stats['savePercentage'] = ((gk_shot_stopping_stats['shotsOnTargetAgainst'] - gk_shot_stopping_stats['goalsConceded']) / gk_shot_stopping_stats['shotsOnTargetAgainst'] * 100).fillna(0)
        else:
            gk_shot_stopping_stats = gk_shot_stopping_stats.reindex(columns=['player.id', 'shotsOnTargetAgainst', 'goalsConceded', 'psxG_faced', 'goalsPrevented', 'goalsPreventedPerSOT', 'savePercentage']).fillna(0)

        exits = gk_events_df[gk_events_df['type.primary'] == 'goalkeeper_exit'].groupby('player.id').size().reset_index(name='exits')
        recoveries_gk = gk_events_df[gk_events_df.get('type.secondary', pd.Series(dtype='object')).apply(lambda x: isinstance(x, (list, np.ndarray)) and 'recovery' in x)].groupby('player.id').size().reset_index(name='recoveries_gk')
        gk_passes = gk_events_df[gk_events_df['type.primary'] == 'pass']
        passes_total_gk = gk_passes.groupby('player.id').size().reset_index(name='passes_gk')
        passes_succ_gk = gk_passes[gk_passes['pass.accurate'] == True].groupby('player.id').size().reset_index(name='passesSuccessful_gk')
        long_passes_total_gk = gk_passes[gk_passes.get('type.secondary', pd.Series(dtype='object')).apply(lambda x: isinstance(x, (list, np.ndarray)) and 'long_pass' in x)].groupby('player.id').size().reset_index(name='longPasses_gk')
        long_passes_succ_gk = gk_passes[gk_passes.get('type.secondary', pd.Series(dtype='object')).apply(lambda x: isinstance(x, (list, np.ndarray)) and 'long_pass' in x) & (gk_passes['pass.accurate'] == True)].groupby('player.id').size().reset_index(name='longPassesSuccessful_gk')

        gk_report_df = pd.DataFrame({'player.id': gk_ids}); gk_report_df = pd.merge(gk_report_df, gk_shot_stopping_stats, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, exits, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, recoveries_gk, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, passes_total_gk, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, passes_succ_gk, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, long_passes_total_gk, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, long_passes_succ_gk, on='player.id', how='left')
        
        # Add % stats
        gk_report_df['Passes successful %'] = (gk_report_df['passesSuccessful_gk'] / gk_report_df['passes_gk'] * 100).fillna(0)
        gk_report_df['Long passes successful %'] = (gk_report_df['longPassesSuccessful_gk'] / gk_report_df['longPasses_gk'] * 100).fillna(0)

        combined_df = pd.merge(combined_df, gk_report_df, left_on='playerId', right_on='player.id', how='left')
        if 'player.id' in combined_df.columns: combined_df = combined_df.drop(columns=['player.id'])
    except Exception as e:
        print(f"  -> ❌ ERROR (Step 3): {e}")

    # --- 4. Calculate xT (Expected Threat) ---
    print("Step 4: Calculating Expected Threat (xT)...")
    try:
        xt_data_from_image = [[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04], [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.04, 0.05, 0.05], [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.05, 0.06, 0.06], [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.04, 0.11, 0.26, 0.26], [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.04, 0.11, 0.26, 0.26], [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.05, 0.06, 0.06], [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.04, 0.05, 0.05], [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04]]
        xt_grid = np.array(xt_data_from_image); rows, cols = xt_grid.shape
        _sp_types = ['corner', 'free_kick', 'throw_in']
        move_df = events_df[events_df['type.primary'].isin(['pass', 'touch', 'acceleration'] + _sp_types)].copy()
        successful_pass = (move_df['type.primary'] == 'pass') & (move_df.get('pass.accurate') == True)
        other_successful_moves = move_df['type.primary'].isin(['touch', 'acceleration'] + _sp_types)
        move_df = move_df[successful_pass | other_successful_moves]
        move_df['start_x'] = move_df['location.x']; move_df['start_y'] = move_df['location.y']
        _is_pass_like = move_df['type.primary'].isin(['pass'] + _sp_types)
        move_df['end_x'] = np.where(_is_pass_like, move_df.get('pass.endLocation.x'), move_df.get('carry.endLocation.x'))
        move_df['end_y'] = np.where(_is_pass_like, move_df.get('pass.endLocation.y'), move_df.get('carry.endLocation.y'))
        move_df = move_df.dropna(subset=['end_x', 'end_y', 'player.id'])

        # Vectorized xT zone calculation (much faster than apply)
        move_df['start_col'] = np.clip((move_df['start_x'] / 100 * cols).astype(float).fillna(0).astype(int), 0, cols - 1)
        move_df['start_row'] = np.clip((move_df['start_y'] / 100 * rows).astype(float).fillna(0).astype(int), 0, rows - 1)
        move_df['end_col'] = np.clip((move_df['end_x'] / 100 * cols).astype(float).fillna(0).astype(int), 0, cols - 1)
        move_df['end_row'] = np.clip((move_df['end_y'] / 100 * rows).astype(float).fillna(0).astype(int), 0, rows - 1)

        # Vectorized xT lookup using numpy advanced indexing
        move_df['xt_start'] = xt_grid[move_df['start_row'].values, move_df['start_col'].values]
        move_df['xt_end'] = xt_grid[move_df['end_row'].values, move_df['end_col'].values]
        move_df['xT'] = move_df['xt_end'] - move_df['xt_start']
        successful_threat = move_df[move_df['xT'] > 0]
        player_xt = successful_threat.groupby('player.id')['xT'].sum().reset_index()
        combined_df = pd.merge(combined_df, player_xt, left_on='playerId', right_on='player.id', how='left')
        if 'player.id' in combined_df.columns: combined_df = combined_df.drop(columns=['player.id'])

        # Split xT into open play (xTOP) and set piece (xTSP)
        # xTSP = xT from throw-ins, corners, free kicks only. Everything else = xTOP.
        successful_threat = successful_threat.copy()
        successful_threat['xt_type'] = np.where(
            successful_threat['type.primary'].isin(_sp_types), 'xTSP', 'xTOP'
        )
        xt_split = successful_threat.groupby(['player.id', 'xt_type'])['xT'].sum()
        xt_split_df = xt_split.unstack(fill_value=0).reset_index()
        combined_df = pd.merge(combined_df, xt_split_df, left_on='playerId', right_on='player.id', how='left')
        if 'player.id' in combined_df.columns: combined_df = combined_df.drop(columns=['player.id'])
    except Exception as e:
        print(f"  -> ❌ ERROR (Step 4): {e}")

    # --- 5. Normalize to Per 90 ---
    print("Step 5: Normalizing stats to per 90...")
    combined_df = combined_df.fillna(0)
    
    # Define only the metrics we just calculated
    metrics_to_normalize = [
        'npxG', 'xAOP', 'xASP', 'xT', 'xTOP', 'xTSP', 'Deep Completions', 'Progressive Passes',
        'shotsOnTargetAgainst', 'goalsConceded', 'psxG_faced', 'goalsPrevented', 'exits',
        'recoveries_gk', 'passes_gk', 'passesSuccessful_gk', 'longPasses_gk', 'longPassesSuccessful_gk'
    ]
    # Get only the metrics that actually exist in the df
    existing_metrics_to_normalize = [m for m in metrics_to_normalize if m in combined_df.columns]
    
    combined_df['totalMinutes'] = pd.to_numeric(combined_df['totalMinutes'], errors='coerce').fillna(0)
    minutes_gt_0 = combined_df['totalMinutes'] > 0
    
    for metric in existing_metrics_to_normalize:
        combined_df[metric] = pd.to_numeric(combined_df[metric], errors='coerce').fillna(0)
        combined_df[metric] = np.where(
            minutes_gt_0,
            (combined_df[metric].astype(float) / combined_df['totalMinutes']) * 90,
            0
        )

    print("--- FINISHED: Streamlined player stats ---")
    return combined_df.fillna(0)


@st.cache_data(ttl=600)
def _compute_peer_density_stack(events_hash, _events_df, position_codes,
                                _player_minutes_df=None,
                                include_recoveries=True):
    """Compute per-90 KDE density grids for all qualifying peers.

    Returns a numpy array of shape ``(100, 100, N_peers)`` sorted along the
    peer axis so that percentile lookups are fast.
    """
    from scipy.stats import gaussian_kde

    def_mask = pv._filter_defensive_actions(_events_df, include_recoveries=include_recoveries)
    def_events = _events_df[def_mask].copy()
    def_events['location.x'] = pd.to_numeric(def_events['location.x'], errors='coerce')
    def_events['location.y'] = pd.to_numeric(def_events['location.y'], errors='coerce')
    def_events = def_events.dropna(subset=['location.x', 'location.y'])
    def_events['player.id'] = pd.to_numeric(def_events['player.id'], errors='coerce')

    if 'player.position' in def_events.columns:
        def_events = def_events[def_events['player.position'].isin(position_codes)]

    # Build minutes lookup
    mins_lookup = {}
    if _player_minutes_df is not None and not _player_minutes_df.empty:
        for _, row in _player_minutes_df.iterrows():
            try:
                mins_lookup[int(row['playerId'])] = float(row['totalMinutes'])
            except (ValueError, TypeError):
                continue

    grid_x, grid_y = np.mgrid[0:100:100j, 0:100:100j]
    positions = np.vstack([grid_x.ravel(), grid_y.ravel()])

    grids = []
    for pid, grp in def_events.groupby('player.id'):
        n = len(grp)
        if n < 3:
            continue
        total_mins = mins_lookup.get(int(pid), 0)
        if total_mins < 300:
            continue
        p90_scale = 90.0 / total_mins

        values = np.vstack([grp['location.x'].values, grp['location.y'].values])
        try:
            kernel = gaussian_kde(values, bw_method='scott')
            density = np.reshape(kernel(positions), grid_x.shape) * p90_scale * n
            grids.append(density)
        except np.linalg.LinAlgError:
            continue

    if not grids:
        return np.zeros((100, 100, 0))

    stack = np.stack(grids, axis=-1)  # (100, 100, N_peers)
    stack.sort(axis=-1)
    return stack


@st.cache_data(ttl=86400)
def load_history_player_minutes():
    """Load historical player minutes from previous seasons."""
    if not os.path.exists('history_player_minutes.pkl'):
        return None
    try:
        with open('history_player_minutes.pkl', 'rb') as f:
            df = pickle.load(f)
        logger.info(f"Loaded {len(df)} players with historical minutes")
        return df
    except Exception as e:
        logger.error(f"Error loading history minutes: {e}")
        return None

def get_combined_career_minutes(_current_minutes_df, _history_minutes_df):
    """Combine current season and historical minutes for career totals."""
    if _history_minutes_df is None:
        return _current_minutes_df

    # Combine current + history
    combined = pd.concat([_current_minutes_df, _history_minutes_df], ignore_index=True)

    # Aggregate by player - sum minutes, keep most recent team/position
    career_minutes = combined.groupby('playerId').agg({
        'playerName': 'first',
        'teamName': 'first',  # Current team (first in concat order)
        'primaryPosition': 'first',
        'totalMinutes': 'sum'
    }).reset_index()

    logger.info(f"Combined career minutes: {len(career_minutes)} players")
    return career_minutes

# ==============================================================================
# 3. GLOBAL CONSTANTS FOR PLAYER RADARS
# ==============================================================================
# Hardcoded defaults (will be overridden by config.yaml if available)
POSITION_GROUPS = {
    'Shot Stopper': ['GK'], 'Cross Claimer': ['GK'], 'Ball-playing GK': ['GK'],
    'Mobile Striker': ['CF', 'SS'], 'Shadow Striker': ['CF', 'SS'], 'Poacher': ['CF', 'SS'], 'Target Man': ['CF', 'SS'], 'Pressing Forward': ['CF', 'SS'],
    'Box-to-Box': ['LCMF', 'RCMF', 'AMF', 'LCMF3', 'RCMF3', 'DMF', 'LDMF', 'RDMF'],
    'Ball-Winning Mid': ['LCMF', 'RCMF', 'LCMF3', 'RCMF3', 'DMF', 'LDMF', 'RDMF'],
    'Holding Mid': ['DMF', 'LDMF', 'RDMF'],
    'Deep-lying Playmaker': ['LCMF', 'RCMF', 'LCMF3', 'RCMF3', 'DMF', 'LDMF', 'RDMF'],
    'Advanced Playmaker': ['AMF', 'RAMF', 'LAMF', 'LW', 'RW'],
    'Wide Winger': ['LW', 'RW', 'LWF', 'RWF', 'LWB', 'RWB', 'RAMF', 'LAMF'],
    'Creative Winger': ['LW', 'RW', 'LWF', 'RWF', 'RAMF', 'LAMF'],
    'Inside Forward': ['LW', 'RW', 'LWF', 'RWF', 'RAMF', 'LAMF'],
    'Full Back': ['LB', 'RB', 'LB5', 'RB5', 'LWB', 'RWB'],
    'Wingback': ['LWB', 'RWB', 'LB5', 'RB5'],
    'Inverted Full Back': ['LB', 'RB', 'LWB', 'RWB', 'LB5', 'RB5'],
    'Ball-Playing Centerback': ['LCB', 'RCB', 'CB', 'LCB3', 'RCB3'],
    'Stopper': ['LCB', 'RCB', 'CB', 'LCB3', 'RCB3'],
    'Athletic Centerback': ['LCB', 'RCB', 'CB', 'LCB3', 'RCB3'],
}
WEIGHTS = {
    'Shot Stopper': {'goalsPrevented': 10.0, 'goalsPreventedPerSOT': 10.0, 'goalsConceded': 1.0, 'exits': 2.0, 'Passes successful %': 1.0, 'Long passes successful %': 1.0, 'passes_gk': 1.0, 'recoveries_gk': 2.0},
    'Cross Claimer': {'goalsPrevented': 10.0, 'goalsPreventedPerSOT': 10.0, 'goalsConceded': 1.0, 'exits': 20.0, 'Passes successful %': 1.0, 'longPassesSuccessful_gk': 1, 'passes_gk': 1.0, 'recoveries_gk': 10},
    'Ball-playing GK': {'goalsPrevented': 10.0, 'goalsPreventedPerSOT': 10.0, 'goalsConceded': 1.0, 'exits': 2.0, 'Passes successful %': 10.0, 'longPassesSuccessful_gk': 6.0, 'passes_gk': 4.0, 'recoveries_gk': 3.0},
    'Ball-Playing Centerback': {'npxG': 1.0, 'xAOP': 1.0, 'xTOP': 5.0, 'Passes': 20, 'Passes successful %': 10, 'Progressive Passes': 20, 'Progressive runs': 6, 'Aerial duels': 2, 'Aerial duels successful %': 6, 'Defensive duels': 2, 'Defensive duels successful %': 8, 'Interceptions': 6, 'Recoveries': 6, 'Clearances': 2},
    'Stopper': {'npxG': 3.0, 'xAOP': 1.0, 'xTOP': 1.0, 'Passes': 1.0, 'Passes successful %': 1.0, 'Progressive Passes': 1.0, 'Progressive runs': 1.0, 'Aerial duels': 8, 'Aerial duels successful %': 10, 'Defensive duels': 8, 'Defensive duels successful %': 10, 'Interceptions': 8, 'Recoveries': 8, 'Clearances': 6},
    'Athletic Centerback': {'npxG': 3.0, 'xAOP': 1.0, 'xTOP': 1.0, 'Passes': 1.0, 'Passes successful %': 1.0, 'Progressive Passes': 1.0, 'Progressive runs': 6, 'Aerial duels': 6, 'Aerial duels successful %': 10, 'Defensive duels': 8, 'Defensive duels successful %': 10, 'Interceptions': 10, 'Recoveries': 10, 'Clearances': 6},
    'Box-to-Box': {'Passes': 4, 'Passes successful %': 3, 'Progressive Passes': 2, 'xTOP': 4.0, 'Goals': 1.0, 'npxG': 4, 'Shots': 2, 'xG per Shot': 1.0, 'Assists': 1.0, 'xAOP': 4, 'Progressive runs': 5, 'Dribbles successful': 4, 'Aerial duels successful': 1.0, 'Defensive duels successful': 2, 'Interceptions': 3, 'Recoveries': 4},
    'Holding Mid': {'Passes': 6, 'Passes successful %': 6, 'Progressive Passes': 2, 'xTOP': 4.0, 'npxG': 1.0, 'xAOP': 1.0, 'Progressive runs': 1.0, 'Dribbles successful': 1.0, 'Aerial duels successful': 4, 'Defensive duels successful': 6, 'Interceptions': 6, 'Recoveries': 6},
    'Ball-Winning Mid': {'Passes': 4, 'Passes successful %': 6, 'Progressive Passes': 2, 'xTOP': 2.0, 'npxG': 1.0, 'xAOP': 1.0, 'Progressive runs': 1.0, 'Aerial duels': 4, 'Aerial duels successful %': 6, 'Defensive duels': 6, 'Defensive duels successful %': 10, 'Interceptions': 10, 'Recoveries': 10, 'Recoveries Opp Half': 4},
    'Deep-lying Playmaker': {'Passes': 10, 'Passes successful %': 6, 'Progressive Passes': 10, 'Passes to final third successful': 8, 'xTOP': 10,  'npxG': 1.0, 'xAOP': 8, 'Progressive runs': 2, 'Dribbles successful': 1.0, 'Aerial duels successful': 1.0, 'Defensive duels successful': 4, 'Interceptions': 4, 'Recoveries': 6},
    'Advanced Playmaker': {'Passes': 6, 'Passes successful %': 2, 'Progressive Passes': 4, 'xTOP': 8, 'Goals': 2, 'npxG': 8, 'Shots': 2, 'xG per Shot': 2, 'Assists': 2, 'xAOP': 8, 'Progressive runs': 2, 'Dribbles successful': 2, 'Aerial duels successful': 1.0, 'Defensive duels successful': 1.0, 'Interceptions': 1.0, 'Recoveries': 1.0, 'Counterpressing Recoveries': 1},
    'Full Back': {'npxG': 4, 'xAOP': 4, 'xTOP': 3, 'Passes': 2, 'Passes successful %': 2, 'Progressive Passes': 2, 'Progressive runs': 2, 'Aerial duels': 2, 'Aerial duels successful %': 8, 'Defensive duels': 4, 'Defensive duels successful %': 10, 'Interceptions': 8, 'Recoveries': 8, 'Clearances': 2},
    'Wingback': {'Goals': 2, 'npxG': 4, 'Shots': 2, 'xG per Shot': 1, 'Assists': 6, 'xAOP': 8, 'xTOP': 6, 'Passes': 1.0, 'Passes successful %': 1.0, 'Progressive Passes': 2, 'Crosses successful': 2, 'Progressive runs': 3, 'Aerial duels': 1.0, 'Aerial duels successful %': 1.0, 'Defensive duels': 1.0, 'Defensive duels successful %': 4, 'Interceptions': 4, 'Recoveries': 4, 'Clearances': 1.0},
    'Inverted Full Back': {'npxG': 1.0, 'xAOP': 1.0, 'xTOP': 12, 'Passes': 16, 'Passes successful %': 6, 'Progressive Passes': 8, 'Progressive runs': 2, 'Aerial duels': 1.0, 'Aerial duels successful %': 4, 'Defensive duels': 4, 'Defensive duels successful %': 6, 'Interceptions': 6, 'Recoveries': 4, 'Clearances': 1.0},
    'Wide Winger': {'Goals': 4, 'npxG': 8, 'Shots': 2, 'xG per Shot': 2, 'Assists': 4, 'xAOP': 8, 'xTOP': 6, 'Passes': 1.0, 'Passes successful %': 1.0, 'Progressive Passes': 1.0, 'Deep Completions': 2, 'Crosses successful': 2, 'Progressive runs': 2, 'Dribbles': 4, 'Dribbles successful %': 2, 'Loss index': 5, 'Aerial duels successful': 1.0, 'Defensive duels successful': 1.0, 'Interceptions': 1.0, 'Recoveries': 1.0, 'Counterpressing Recoveries': 1},
    'Creative Winger': {'Goals': 4, 'npxG': 8, 'Shots': 2, 'xG per Shot': 2, 'Assists': 6, 'xAOP': 12, 'xTOP': 10, 'Passes': 2, 'Passes successful %': 1.0, 'Progressive Passes': 2, 'Deep Completions': 3, 'Crosses successful': 2, 'Progressive runs': 2, 'Dribbles': 2, 'Dribbles successful %': 4, 'Loss index': 5, 'Aerial duels successful': 1.0, 'Defensive duels successful': 1.0, 'Interceptions': 1.0, 'Recoveries': 1.0, 'Counterpressing Recoveries': 1},
    'Inside Forward': {'Goals': 15, 'npxG': 30, 'Shots': 6, 'xG per Shot': 8, 'Assists': 10, 'xAOP': 20, 'xTOP': 2, 'Passes': 1.0, 'Passes successful %': 1.0, 'Progressive runs': 2, 'Dribbles': 4, 'Dribbles successful %': 4, 'Loss index': 5, 'Aerial duels successful': 4, 'Defensive duels successful': 1.0, 'Interceptions': 1.0, 'Recoveries': 1.0, 'Counterpressing Recoveries': 1},
    'Shadow Striker': {'Goals': 15, 'npxG': 30, 'Shots': 10, 'xG per Shot': 8, 'Assists': 10, 'xAOP': 20, 'xTOP': 4, 'Passes': 2, 'Passes successful %': 2, 'Progressive Passes': 3, 'Deep Completions': 3, 'Progressive runs': 2, 'Dribbles': 4, 'Dribbles successful %': 4, 'Loss index': 5, 'Aerial duels successful': 2, 'Defensive duels successful': 1.0, 'Interceptions': 1.0, 'Recoveries': 3},
    'Mobile Striker': {'Goals': 15, 'npxG': 30, 'Shots': 10, 'xG per Shot': 8, 'Assists': 10, 'xAOP': 20, 'xTOP': 4, 'Passes': 2, 'Passes successful %': 1.0, 'Progressive Passes': 1.0, 'Deep Completions': 1.0, 'Progressive runs': 8, 'Dribbles': 8, 'Dribbles successful %': 6, 'Loss index': 5, 'Aerial duels': 1.0, 'Aerial duels successful %': 1.0, 'Defensive duels successful': 1.0, 'Interceptions': 1.0, 'Recoveries': 6},
    'Poacher': {'Goals': 20, 'npxG': 40, 'Shots': 10, 'xG per Shot': 10, 'Assists': 10, 'xAOP': 20, 'Passes': 1.0, 'Passes successful %': 1.0, 'Progressive Passes': 1.0, 'Deep Completions': 1.0, 'Progressive runs': 1.0, 'Dribbles successful': 1.0, 'Loss index': 5, 'Aerial duels': 5, 'Aerial duels successful %': 5, 'Defensive duels successful': 1.0, 'Interceptions': 1.0, 'Recoveries': 1.0},
    'Target Man': {'Goals': 15, 'npxG': 30, 'Shots': 10, 'xG per Shot': 8, 'Assists': 10, 'xAOP': 20, 'xTOP': 2, 'Passes': 2, 'Passes successful %': 2, 'Progressive Passes': 1.0, 'Deep Completions': 1.0, 'Progressive runs': 1.0, 'Dribbles': 1.0, 'Dribbles successful %': 1.0, 'Loss index': 5, 'Aerial duels': 10, 'Aerial duels successful %': 10, 'Defensive duels successful': 1.0, 'Interceptions': 1.0, 'Recoveries': 1.0, 'Clearances': 10},
    'Pressing Forward': {'Goals': 15, 'npxG': 30, 'Shots': 10, 'xG per Shot': 8, 'Assists': 10, 'xAOP': 20, 'xTOP': 2, 'Passes': 2, 'Passes successful %': 1.0, 'Progressive Passes': 1.0, 'Deep Completions': 1.0, 'Progressive runs': 2, 'Dribbles': 2, 'Dribbles successful %': 2, 'Loss index': 5, 'Aerial duels': 1.0, 'Aerial duels successful %': 1.0, 'Defensive duels successful': 1.0, 'Interceptions': 8, 'Recoveries': 10, 'Counterpressing Recoveries': 4}
}
INVERT_METRICS = ['Loss index', 'goalsConceded', 'Dribbled past %', 'Dribbled past % (proj)', 'DefR Value Conceded']

# ==============================================================================
# Composite Value Index (CVI) — v1 parameters
# ==============================================================================
# CVI is a single 0–~150 score combining performance, age-value premium,
# sample reliability, and league strength. Calibrated against the
# 27 user-reported transfer fees (real + synthetic) via a CVI → EUR
# power curve in the bio "Projected value" cell.
#
# Formula:
#   CVI = PerformanceQuality
#       × AgeValueMultiplier
#       × ReliabilityWeight
#       × LeagueMultiplier
#
# Each component is documented inline below.

# Position-tuned blend of Role_Score percentile and Action V percentile
# for PerformanceQuality.
#
# v1.8 rebalance — the old 50-80% role weights underestimated the
# overlap between Role_Score and Action V. Role_Score's underlying
# metric weights already include heavy contributions from the same
# signals that build Total Value/90:
#   - npxG → Shooting Value           (r ≈ 0.95 per GPA explainer)
#   - xAOP → Passing/Receiving Value
#   - xTOP → Passing/Receiving Value
#   - Progressive Passes → Passing Value
#   - Dribbles successful → Dribbling Value
# So weighting Action V at 40-50% (the old ST/AM_WG weights) was
# largely double-counting the same chance-creation/progression signal.
# The new weights keep Action V as a sanity-check cross-validator but
# make Role_Score the dominant input — which is what role-fit
# scouting at the lower-division level actually rewards.
#
# GK still gets the most extreme tilt because the GPA explainer
# (Part VI) shows Action V is a particularly weak signal for keepers
# (Shot-Stopping Value within-pos r ≈ 0.03 within a single season).
CVI_PERF_WEIGHTS = {
    'GK':    (0.90, 0.10),   # (role, action_v)
    'CB':    (0.85, 0.15),
    'FB':    (0.80, 0.20),
    'CM':    (0.80, 0.20),
    'AM_WG': (0.75, 0.25),
    'ST':    (0.75, 0.25),
}

# AgeValueMultiplier(age, position) — NPV of remaining career value
# (v2.5).
#
# Model: a player's age multiplier = the sum of expected future
# performance years from their current age until career end. The
# multiplier strictly decreases with age because every year you age,
# you lose one year of remaining career.
#
# Captures the four mechanisms the user identified:
#   1. Projected rate of perf IMPROVEMENT  → youth_baseline → 1.0 by peak_age
#   2. Years REMAINING before decline      → peak_age → decline_start
#   3. Projected rate of perf DECLINE      → decline_start → career_end
#   4. Total career value                  → integral of the above
#
# Three-phase performance trajectory at any future age:
#   age < peak_age:        linear growth from youth_baseline at 16 to 1.0
#   peak_age ≤ age < decline_start: flat at 1.0
#   decline_start ≤ age < career_end: linear decline from 1.0 to 0
#   age ≥ career_end:      0
#
# Remaining career value at age A = ∫ perf(t) dt from t=A to career_end
# (approximated by sum across integer year boundaries, linearly
# interpolated for fractional ages).
#
# Final multiplier:
#   m(A) = old_floor + (max_mult − old_floor) × (rcv(A) / rcv(16))
#
# Anchored to rcv(16) so the multiplier hits max_mult at age 16 and
# old_floor at career_end. Strictly monotone non-increasing across
# all ages.
#
# Per-position trajectory parameters (best evidence from CIES +
# market analyses):
#   GK     peak 28, decline 33, end 39  — longest career, latest decline
#   CB     peak 27, decline 31, end 36
#   CM     peak 26, decline 30, end 35
#   FB     peak 25, decline 29, end 33
#   ST     peak 25, decline 28, end 33
#   AM_WG  peak 24, decline 27, end 32  — pace-dependent, earliest end
#
# Sample multipliers for ST (peak 25, decline 28, end 33):
#   16yo → 1.80  ~17 years of remaining perf; wonderkid premium
#   21yo → 1.38  Approaching peak, still 12 years
#   25yo → 0.92  At perf peak, 8 years left
#   28yo → 0.51  Decline starts now, 5 years
#   30yo → 0.26  Mid-decline, 3 years
#   33yo → 0.10  Floor
#
# Same perf=70:
#   16yo CVI 126 vs 25yo CVI 65   → 2× premium for the wonderkid
#   16yo CVI 126 vs 30yo CVI 18   → 7× premium
CVI_AGE_VALUE_PARAMS = {
    # v2.7 — further compressed. v2.6 still felt too age-heavy in
    # practice. Tightened the range from max~1.55 / floor~0.40
    # (~4× spread) to max~1.30 / floor~0.55 (~2.4× spread). Performance
    # now strongly dominates the final CVI; age is a meaningful but
    # subordinate modifier. Career-NPV shape unchanged.
    'GK':    {'peak_age': 28, 'decline_start': 33, 'career_end': 39,
              'max_mult': 1.20, 'old_floor': 0.60, 'youth_baseline': 0.55},
    'CB':    {'peak_age': 27, 'decline_start': 31, 'career_end': 36,
              'max_mult': 1.28, 'old_floor': 0.55, 'youth_baseline': 0.50},
    'CM':    {'peak_age': 26, 'decline_start': 30, 'career_end': 35,
              'max_mult': 1.28, 'old_floor': 0.55, 'youth_baseline': 0.50},
    'FB':    {'peak_age': 25, 'decline_start': 29, 'career_end': 33,
              'max_mult': 1.30, 'old_floor': 0.55, 'youth_baseline': 0.50},
    'ST':    {'peak_age': 25, 'decline_start': 28, 'career_end': 33,
              'max_mult': 1.32, 'old_floor': 0.55, 'youth_baseline': 0.50},
    'AM_WG': {'peak_age': 24, 'decline_start': 27, 'career_end': 32,
              'max_mult': 1.35, 'old_floor': 0.55, 'youth_baseline': 0.50},
}


def _cvi_expected_perf_at(age, params):
    """Expected normalized performance at given age (0..1).
    Three-phase: youth growth → peak plateau → linear decline → 0."""
    if age < 16:
        return params['youth_baseline']
    if age >= params['career_end']:
        return 0.0
    if age < params['peak_age']:
        yb = params['youth_baseline']
        return yb + (1.0 - yb) * (age - 16) / (params['peak_age'] - 16)
    if age < params['decline_start']:
        return 1.0
    decline_yrs = params['career_end'] - params['decline_start']
    if decline_yrs <= 0:
        return 0.0
    return max(1.0 - (age - params['decline_start']) / decline_yrs, 0.0)


def _cvi_cum_remaining_career(age_int, params):
    """Sum of expected perf from int(age) up to career_end−1
    (integer-year boundaries)."""
    ce = int(params['career_end'])
    if age_int >= ce:
        return 0.0
    start = max(int(age_int), 16)
    return sum(_cvi_expected_perf_at(t, params) for t in range(start, ce))


# Pre-compute max remaining career value at age 16 per position so
# the multiplier hits max_mult exactly at age 16. Computed at module
# load time; safe because CVI_AGE_VALUE_PARAMS is fixed.
_CVI_MAX_CAREER_VALUE = {
    pos: _cvi_cum_remaining_career(16, p)
    for pos, p in CVI_AGE_VALUE_PARAMS.items()
}

# ---- ReliabilityWeight ----
# Replaces the naive `min(mins/1800, 1.0)` ramp from CVI v1.
#
# Grounded in the empirical per-position stability table from the GPA
# v2 explainer (reports/gpa_explainer.pdf, Part VI; raw data at
# models/validation/stability_by_minutes.csv). The headline finding:
# within-position YoY r for Total Value differs by ~5× across positions
# at the same sample size:
#
#   pos      within-pos YoY r @ 900 min (Total Value)
#   ─────    ────────────────────────────────────────
#   CM       0.64    ← most stable outfield position
#   FB       0.41
#   STRIKER  0.32
#   AM_WG    ~0.20   (winger 0.13, attmid sparse sample)
#   CB       0.19
#   GK       0.12    ← V-metrics need ~2 seasons per explainer Part VII
#
# Two implications a single linear ramp can't capture:
#  1. ASYMPTOTIC CEILING differs by position. A CB rated on event-only V
#     metrics has a structural noise floor (defensive valuation is hard
#     per Visual 4). No amount of minutes makes CB Total Value as
#     reliable as CM Total Value. The ceiling reflects that.
#  2. TIME-TO-CEILING differs by position. GK needs ~2 full seasons
#     for shot-stopping / handling / sweeping to stabilize. Outfielders
#     reach near-ceiling around 1500-1800 min.
#
# These ceilings ARE NOT raw YoY r values — they're translated into
# 0-1 weights via the psychometric convention also used in the explainer:
#   r ≥ 0.7 ≈ trustworthy standalone (weight = 1.0)
#   r ≈ 0.5 ≈ useful as composite input (weight ≈ 0.7)
#   r < 0.3 ≈ noise floor (weight ≈ 0.2)
# Then bumped upward by the Spearman-Brown effect of CVI being a
# composite (Role_Score blends 15-25 weighted metrics; n_eff ≈ 3
# accounting for inter-metric correlation), which lifts a within-pos
# r=0.45 composite-input into an effective ~0.71 — putting a typical
# outfielder near ceiling=0.95 once they have full minutes.
CVI_RELIAB_CEILING_BY_POS = {
    'GK':    0.70,   # event-V can't fully measure shot-stopping; need 2 seasons
    'CB':    0.85,   # defensive valuation hard (explainer Visual 4 hybrid)
    'AM_WG': 0.90,
    'FB':    0.92,
    'ST':    0.92,
    'CM':    0.95,   # most metrics in the role composite are stable
}
CVI_RELIAB_CEILING_DEFAULT = 0.85

# Minutes at which we approximately reach the position's ceiling
# (≈95% there; the curve smoothly saturates above this).
CVI_RELIAB_MINS_TO_CEILING = {
    'GK':    3600,   # ~2 full seasons per explainer Part VII
    'CB':    2100,
    'AM_WG': 1800,
    'FB':    1800,
    'ST':    1800,
    'CM':    1500,
}
CVI_RELIAB_MINS_TO_CEILING_DEFAULT = 1800

# NOTE on the (now-removed) very-short-sample floor:
# v1 had a linear floor below 270 min so small-sample players didn't
# collapse to reliab=0 (which would have killed their CVI via
# multiplication). With v2.0's empirical-Bayes shrinkage the
# justification went away — even a 0-reliability player gets shrunk
# to a sensible prior (their career mean if known, 40 if not), never
# to 0. The floor was also creating a JUMP of ~25 percentage points
# at the 270-min boundary where the floor formula and the saturating
# curve didn't meet smoothly. Removed in v2.1.

# ---- Shrinkage prior ----
# When reliability is low, we don't shrink the rating to ZERO — we
# shrink it toward a "replacement-level" prior (the freely-available
# player a club could sign tomorrow). Statistically this is empirical
# Bayes: with low sample, weight the prior more; with high sample,
# weight the observation more.
#
#   shrunk_perf = reliab × raw_perf + (1 − reliab) × replacement_perf
#
# Why 40 on the 0-100 scale? PerformanceQuality is a within-position
# percentile blend, so "40" literally means "≈40th-percentile player
# within this position group" — a marginal starter / quality bench
# player at this tier. That's the conventional sabermetric definition
# of replacement level (the worst player a competitive team would
# field), borrowed from Baseball Prospectus's VORP and FanGraphs' WAR.
#
# Effect on the math:
#   • A 70-rated CB with 300 min (reliab=0.27):
#       shrunk = 0.27×70 + 0.73×40 = 48.1
#       (vs old: CVI_perf×reliab = 70×0.27 = 18.9 — punished too hard)
#   • A 70-rated CB with 1800 min (reliab=0.79):
#       shrunk = 0.79×70 + 0.21×40 = 63.7
#       (already close to the observed rating)
#   • A 25-rated player with 300 min (reliab=0.27):
#       shrunk = 0.27×25 + 0.73×40 = 35.9
#       (low ratings ALSO pulled toward replacement — we don't
#       overreact to a small sample of bad performances either)
CVI_REPLACEMENT_PERF = 40.0

# Pure tier-level league multipliers, anchored to the empirical
# mover-based cross-tier ratio from the cross-tier analysis (see
# git history around 2026-05: 41 Camp→L3 movers showed ~0.80×
# V/90 in L3 vs Camp; 271 L3→Camp movers showed ~0.89× L3/Camp;
# the all-movers median of 0.88 weighted by sample sizes lands
# at ~0.85 once selection bias on the upward-mover side is folded
# in). We deliberately do NOT layer team-strength-within-tier on
# top — GPA Total Value already encodes team context per action,
# so a team multiplier would double-count it.
#
# Reference frame: Liga 3 = 1.0 (baseline).
CVI_LEAGUE_MULTIPLIER = {
    43324: 1.00,    # Liga 3
    702:   0.85,    # Campeonato de Portugal
}
# Fallback for any competition not in the dict (won't normally fire
# since the dashboard's data scope is Liga 3 + Camp only).
CVI_LEAGUE_DEFAULT = 1.0


# v2.8 — position-specific multiplier applied to the CVI→EUR mapping
# (bio "Projected value" cell). Literature-grounded priors compressed
# toward 1.0 for Liga 3 reality.
#
# Sources informing the magnitudes:
#   • CIES (Poli, Besson & Ravenel 2022, Economies 10/1/4) — standardized
#     experience betas: forwards 0.934 > mid 0.793 > CB 0.749 > FB 0.606
#     > GK 0.407. Ratio GK/FW ≈ 0.44 in Big-5.
#   • Müller, Simons & Weinmann (2017, EJOR) — position random-effect SD
#     0.050 on log-MV (~±5% spread once age/perf/club/league controlled).
#   • Franceschi, Brocard, Follert & Gouguet (2024, JoES 38(3)) — review
#     of 29 papers / 111 specs: directional ordering ST > AM/WG > CM >
#     CB > FB > GK is robust; FB shows the most-negative coef vs CF.
#   • Frick (2007, SJPE) — GK pay penalty mechanism (low role flexibility).
#   • Garcia-del-Barrio & Pujol (2007, MDE) — attacker premium driven by
#     crowd-pulling capacity — a mechanism MUCH weaker in Liga 3, so we
#     deliberately compress the GK discount toward 0.70 (not Big-5's ~0.50).
POSITION_EUR_MULTIPLIER = {
    'ST':    1.30,
    'AM_WG': 1.25,
    'CM':    1.00,
    'CB':    0.90,
    'FB':    0.85,
    'GK':    0.70,
}

# v2.9 — extra Campeonato discount on the EUR side. CVI already uses
# league_factor 0.85 for Camp inside the score itself, but the user wants
# Camp projected prices nudged down further to reflect that even an
# "equivalent CVI" Camp player commands a lower real fee at sale (smaller
# scout footprint, less liquid market, lower buyer competition). Combined
# with the in-CVI 0.85, a Camp player at the same raw inputs as a Liga 3
# player ends up at ~0.85 × 0.85 ≈ 72% of the Liga 3 projected EUR.
CAMP_PROJECTED_EUR_PENALTY = 0.85

# v2.10 — steeper top + lower mid/bottom. User: "top line should stay
# similar while the middle and bottom drop off a little." Higher
# exponent (2.55 → 2.70) widens the spread between bottom and top;
# coefficient anchored (1.10) so CVI 110 stays at ~€355k (matches v2.9).
# Effect vs v2.9 (CM, Liga 3):
#   CVI 40   €27k → €23k   (−14%)
#   CVI 60   €76k → €69k   (−9%)
#   CVI 80   €159k → €151k (−5%)
#   CVI 100  €281k → €275k (−2%)
#   CVI 110  €358k → €355k (anchor)
#   CVI 120  €447k → €450k (+1%)
PROJECTED_EUR_COEF = 1.10
PROJECTED_EUR_EXP  = 2.70
PROJECTED_EUR_CAP  = None   # cap removed 2026-06-23 (Lucas) — was 500_000. Only
                            # 1 rated player was pinned at it (max uncapped ~€431k),
                            # so it was a near-inert safety rail. Set back to a number
                            # (e.g. 500_000) to re-enable the min() clamp.


def cvi_to_projected_eur(cvi, position_group=None, competition_id=None):
    """Convert a CVI score to a projected EUR figure: power curve +
    position multiplier + Camp penalty. Cap removed 2026-06-23 (the
    €500k clamp is now opt-in via PROJECTED_EUR_CAP). Returns None if
    cvi is None/<=0."""
    try:
        v = float(cvi)
    except (TypeError, ValueError):
        return None
    if not (v > 0):
        return None
    pos_mult = POSITION_EUR_MULTIPLIER.get(position_group, 1.00)
    camp_mult = (CAMP_PROJECTED_EUR_PENALTY
                  if competition_id is not None
                  and not (isinstance(competition_id, float) and pd.isna(competition_id))
                  and int(competition_id) == 702
                  else 1.00)
    val = (PROJECTED_EUR_COEF * (v ** PROJECTED_EUR_EXP)
            * pos_mult * camp_mult)
    return val if PROJECTED_EUR_CAP is None else min(val, PROJECTED_EUR_CAP)


def _cvi_position_group(primary_position):
    """Map Wyscout primaryPosition to a CVI position-group key
    (matches keys in CVI_PERF_WEIGHTS / CVI_AGE_VALUE_PARAMS)."""
    if primary_position is None:
        return None
    try:
        if pd.isna(primary_position):
            return None
    except (TypeError, ValueError):
        pass
    p = str(primary_position)
    if p == 'GK': return 'GK'
    if p in ('CB', 'LCB', 'RCB', 'LCB3', 'RCB3'): return 'CB'
    if p in ('LB', 'RB', 'LB5', 'RB5', 'LWB', 'RWB'): return 'FB'
    if p in ('CMF', 'LCMF', 'RCMF', 'LCMF3', 'RCMF3',
             'DMF', 'LDMF', 'RDMF'): return 'CM'
    if p in ('AMF', 'LAMF', 'RAMF', 'LMF', 'RMF',
             'LW', 'RW', 'LWF', 'RWF'): return 'AM_WG'
    if p in ('CF', 'SS'): return 'ST'
    return None


def _cvi_reliability_weight(mins, position_group):
    """Position-aware reliability weight grounded in within-position
    YoY r data from the GPA explainer (Part VI).

    Shape:
      mins < 270:        linear ramp 0 → FLOOR (0.15)
      mins ≥ 270:        ceiling(pos) × (1 − exp(−3 × mins / mins_to_ceiling(pos)))
                          which reaches ~95% of ceiling at mins_to_ceiling
                          and asymptotes toward ceiling above that

    Returns a tuple (weight, breakdown_dict) where breakdown_dict has the
    raw ceiling, sample_factor (the 0..1 saturating curve value before
    multiplying by ceiling), and mins_to_ceiling — for surfacing in the
    UI so users can audit why a player got 0.65 vs 1.0.
    """
    import math
    if mins is None:
        return 0.0, {'ceiling': None, 'sample_factor': 0.0, 'mins_to_ceiling': None}
    try:
        if pd.isna(mins):
            return 0.0, {'ceiling': None, 'sample_factor': 0.0, 'mins_to_ceiling': None}
        mins = float(mins)
    except Exception:
        return 0.0, {'ceiling': None, 'sample_factor': 0.0, 'mins_to_ceiling': None}
    if mins <= 0:
        return 0.0, {'ceiling': None, 'sample_factor': 0.0, 'mins_to_ceiling': None}
    ceiling = CVI_RELIAB_CEILING_BY_POS.get(position_group, CVI_RELIAB_CEILING_DEFAULT)
    mins_full = CVI_RELIAB_MINS_TO_CEILING.get(position_group,
                                                  CVI_RELIAB_MINS_TO_CEILING_DEFAULT)
    # Smooth saturating curve from 0 — no discontinuity. Combined with
    # the v2.0 empirical-Bayes prior, a very-low-sample player no longer
    # collapses to 0; they're shrunk toward their career prior (or 40
    # for a debutant). At mins=100 the weight is ~0.17, at 270 it's
    # ~0.40, at mins_full it's ~0.95 — continuous everywhere.
    sample_factor = 1.0 - math.exp(-3.0 * mins / mins_full)
    weight = ceiling * sample_factor
    return weight, {'ceiling': ceiling, 'sample_factor': sample_factor,
                     'mins_to_ceiling': mins_full}


def _cvi_age_value_multiplier(age, position_group):
    """NPV-of-remaining-career age multiplier. Returns 1.0 if inputs
    can't be evaluated (so missing age doesn't tank the CVI).

    Sums the player's expected future performance from current age to
    career_end, normalizes against the value at age 16. Result is
    strictly non-increasing in age — same raw_perf, younger always
    wins, with the magnitude reflecting how many productive years
    they have left.

    Captures: rate of perf improvement (youth → peak), years before
    decline, rate of perf decline, and total career horizon. See
    CVI_AGE_VALUE_PARAMS docstring for parameters.

    For fractional ages, linearly interpolates between integer-year
    cumulative values so the curve is smooth (no step jumps).
    """
    import math
    if age is None or position_group not in CVI_AGE_VALUE_PARAMS:
        return 1.0
    try:
        a = float(age)
        if pd.isna(a):
            return 1.0
    except (TypeError, ValueError):
        return 1.0
    p = CVI_AGE_VALUE_PARAMS[position_group]
    if a >= p['career_end']:
        return p['old_floor']
    lo = int(math.floor(a))
    hi = lo + 1
    f = a - lo
    rcv_lo = _cvi_cum_remaining_career(lo, p)
    rcv_hi = _cvi_cum_remaining_career(hi, p)
    rcv = rcv_lo * (1.0 - f) + rcv_hi * f
    max_rcv = _CVI_MAX_CAREER_VALUE.get(position_group, 1.0)
    if max_rcv <= 0:
        return p['old_floor']
    norm = max(0.0, min(rcv / max_rcv, 1.0))
    return p['old_floor'] + (p['max_mult'] - p['old_floor']) * norm


def compute_cvi_columns(player_stats_df, *, age_lookup,
                         comp_id_lookup=None,
                         opta_team_strength_lookup=None,   # deprecated, kept for compat
                         team_col='teamName',
                         prior_lookup=None):
    """Compute CVI + its components for every row in player_stats_df.

    Args:
        player_stats_df: DataFrame from calculate_player_percentiles_and_scores
            (must have primaryPosition, totalMinutes, all {role}_Score
            columns, and 'Total Value' from GPA merge).
        age_lookup: callable playerId -> age in years (or None).
        comp_id_lookup: callable playerId -> competitionId (43324 or 702).
            If None, falls back to the player_stats_df's competitionId
            column if present, else assumes Liga 3 (1.0×).
        opta_team_strength_lookup: deprecated. Earlier versions used team
            Opta strength to scale LeagueMultiplier within a tier; we
            dropped that to avoid double-counting team context which is
            already encoded in GPA Total Value per action. Argument
            kept so call sites don't break.
        team_col: column in player_stats_df with the team name (still
            used for joining / display, but not for CVI math anymore).

    Returns:
        DataFrame with the same index as input plus columns:
            _CVI            — final composite (0–~150 typical)
            _CVI_perf       — raw PerformanceQuality (0-100)
            _CVI_perf_shrunk — shrunk toward replacement-level prior
                              (this is what actually feeds into CVI)
            _CVI_age        — AgeValueMultiplier (0.4-1.6)
            _CVI_reliab     — ReliabilityWeight (0-1), position-aware;
                              this is the *shrinkage weight*, not a
                              multiplier on perf anymore
            _CVI_reliab_ceiling          — asymptotic max for this pos
            _CVI_reliab_sample_factor    — 0..1 sample-driven curve value
            _CVI_reliab_mins_to_ceiling  — min count for ~95% of ceiling
            _CVI_league     — LeagueMultiplier (0.85 or 1.0)
            _CVI_trajectory — shrunk_perf - same-age-position-median
                              (a "+30 flag" surfaced separately, NOT
                              applied to CVI)
    """
    if player_stats_df is None or player_stats_df.empty:
        return pd.DataFrame()

    df = player_stats_df.copy()

    # ---- BULLETPROOF DTYPE COERCION AT ENTRY ----
    # The dashboard's upstream pipelines (especially the cross-tier Liga 3
    # + Campeonato merges) can leak object-dtype columns containing
    # sentinel strings like '—', '-', 'N/A', or even mixed int/str
    # values. Any of those would later blow up a sort/rank/between/clip
    # comparison with the cryptic "'>=' not supported between str and
    # float" TypeError. Coerce every column we will compare/sort here.
    #
    # primaryPosition: leave as object (string-keyed map below) but force
    # to string so int values from a broken merge don't trip _cvi_position_group.
    if 'primaryPosition' in df.columns:
        df['primaryPosition'] = df['primaryPosition'].apply(
            lambda v: str(v) if v is not None and not pd.isna(v) else None
        )
    # Total Value (the V/90 column we rank): force numeric.
    if 'Total Value' in df.columns:
        df['Total Value'] = pd.to_numeric(df['Total Value'], errors='coerce')
    # totalMinutes drives reliability — same defensive coercion.
    if 'totalMinutes' in df.columns:
        df['totalMinutes'] = pd.to_numeric(df['totalMinutes'], errors='coerce')
    # All <Role>_Score columns we read in _best_role_score.
    for _sc in [c for c in df.columns if c.endswith('_Score')]:
        df[_sc] = pd.to_numeric(df[_sc], errors='coerce')

    # Map position to CVI group + age. Coerce age to numeric — for some
    # Campeonato player-seasons the age lookup may return a string
    # (e.g. an unparsed birthDate) which would later blow up the
    # df['_cvi_age'].between(a-2, a+2) call inside _expected_perf with
    # "TypeError: '>=' not supported between str and float".
    df['_cvi_group'] = df['primaryPosition'].map(_cvi_position_group)
    df['_cvi_age'] = pd.to_numeric(df['playerId'].map(age_lookup),
                                     errors='coerce')

    # ---- PerformanceQuality (Role component) ----
    # v1.9 — versatility-aware aggregation. Pure max threw away the
    # signal that a player good across multiple eligible roles is
    # more flexible (and hence more valuable in the transfer market)
    # than a one-role specialist at the same peak.
    #
    # Formula:
    #   role_score = α × max(eligible_role_scores)
    #              + (1 − α) × mean(eligible_role_scores)
    #
    # α = 0.6 → 60% best role + 40% mean across all eligible roles.
    # Worked examples (a CF/SS eligible for 5 striker roles):
    #
    #   Player type            scores              old (max)  new (0.6/0.4)
    #   ────────────────────── ──────────────────  ─────────  ──────────────
    #   Specialist Poacher     [80, 30, 30, 30, 30]   80        64
    #   Versatile #9           [70, 65, 60, 50, 40]   70        64.8
    #   Compleat striker       [70, 70, 70, 70, 70]   70        70
    #
    # The compleat striker now wins — what scouts intuit. The
    # specialist takes a bigger hit because their non-Poacher numbers
    # really are weak (and a Mourinho would pay for an all-rounder
    # over a one-trick pony at the same headline). α=0.6 is a starting
    # point; can be tuned off the reported transfer fees.
    CVI_ROLE_VERSATILITY_ALPHA = 0.6  # weight on max vs mean

    def _role_score_blend(row):
        pos = row.get('primaryPosition')
        eligible = [r for r in WEIGHTS if pos in POSITION_GROUPS.get(r, [])]
        vals = []
        for r in eligible:
            v = row.get(f"{r}_Score")
            try:
                if v is not None and not pd.isna(v):
                    vals.append(float(v))
            except Exception:
                pass
        if not vals:
            return None
        if len(vals) == 1:
            return vals[0]   # single-role case → no blending needed
        a = CVI_ROLE_VERSATILITY_ALPHA
        return a * max(vals) + (1.0 - a) * (sum(vals) / len(vals))

    df['_cvi_role_score'] = df.apply(_role_score_blend, axis=1)

    # Action V percentile within position group — rank Total Value
    # within same _cvi_group so a 0.05 V/90 striker isn't compared
    # against a 0.005 V/90 CB.
    # Defensive coercion: in cross-tier merges the 'Total Value' column
    # can come in as object dtype (string '—' for missing Camp rows
    # alongside floats for matched rows). rank() then raises
    # "'>=' not supported between str and float". Force numeric first.
    val_col = 'Total Value' if 'Total Value' in df.columns else None
    if val_col:
        df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
        df['_cvi_av_pct'] = (df.groupby('_cvi_group')[val_col]
                              .rank(pct=True, method='average') * 100.0)
    else:
        df['_cvi_av_pct'] = None

    # Same risk for totalMinutes (some pipelines stash it as object).
    if 'totalMinutes' in df.columns:
        df['totalMinutes'] = pd.to_numeric(df['totalMinutes'], errors='coerce')

    def _perf_quality(row):
        g = row.get('_cvi_group')
        if g not in CVI_PERF_WEIGHTS:
            return None
        w_role, w_av = CVI_PERF_WEIGHTS[g]
        role = row.get('_cvi_role_score')
        av = row.get('_cvi_av_pct')
        # Coerce both to float (or None) up-front so we can't return a
        # weird type that breaks downstream sorts/comparisons.
        try:
            role_f = (float(role) if role is not None
                       and not pd.isna(role) else None)
        except Exception:
            role_f = None
        try:
            av_f = (float(av) if av is not None
                     and not pd.isna(av) else None)
        except Exception:
            av_f = None
        # Fall back gracefully when one side is missing — re-weight
        # so the score still uses the other side at full weight.
        if av_f is None and role_f is None:
            return None
        if av_f is None:
            return role_f
        if role_f is None:
            return av_f
        return w_role * role_f + w_av * av_f

    df['_CVI_perf'] = pd.to_numeric(df.apply(_perf_quality, axis=1),
                                      errors='coerce')

    # ---- AgeValueMultiplier ----
    df['_CVI_age'] = df.apply(
        lambda r: _cvi_age_value_multiplier(r.get('_cvi_age'), r.get('_cvi_group')),
        axis=1,
    )

    # ---- ReliabilityWeight ----
    # Position-aware empirical curve (see CVI_RELIAB_* constants above).
    if 'totalMinutes' in df.columns:
        _reliab_results = df.apply(
            lambda r: _cvi_reliability_weight(r.get('totalMinutes'), r.get('_cvi_group')),
            axis=1,
        )
        df['_CVI_reliab'] = _reliab_results.apply(lambda t: t[0])
        df['_CVI_reliab_ceiling'] = _reliab_results.apply(
            lambda t: t[1].get('ceiling'))
        df['_CVI_reliab_sample_factor'] = _reliab_results.apply(
            lambda t: t[1].get('sample_factor'))
        df['_CVI_reliab_mins_to_ceiling'] = _reliab_results.apply(
            lambda t: t[1].get('mins_to_ceiling'))
    else:
        df['_CVI_reliab'] = 1.0
        df['_CVI_reliab_ceiling'] = None
        df['_CVI_reliab_sample_factor'] = None
        df['_CVI_reliab_mins_to_ceiling'] = None

    # ---- LeagueMultiplier ----
    # Pure tier-level: 1.0 for Liga 3, 0.85 for Campeonato. The
    # comp_id_lookup callable wins if provided; otherwise we use
    # competitionId from the player_stats_df if present; otherwise
    # the conservative default of 1.0 (we'd rather not penalize a
    # player if we can't classify their tier).
    if comp_id_lookup is not None and 'playerId' in df.columns:
        comps = df['playerId'].map(comp_id_lookup)
    elif 'competitionId' in df.columns:
        comps = df['competitionId']
    else:
        comps = pd.Series([None] * len(df), index=df.index)
    df['_CVI_league'] = comps.map(
        lambda c: CVI_LEAGUE_MULTIPLIER.get(int(c), CVI_LEAGUE_DEFAULT)
                   if c is not None and not pd.isna(c) else CVI_LEAGUE_DEFAULT
    )

    # ---- Empirical-Bayes shrinkage toward player-specific prior ----
    # v2.0 — instead of always shrinking toward the generic
    # replacement-level (40), shrink toward THIS PLAYER's career prior
    # when we have rich prior-season data. A 1350-min season from a
    # player with 2400 effective prior minutes shouldn't be discounted
    # toward generic replacement — we know who he is.
    #
    # Formula:
    #   prior_strength  = min(prior_mins_eff / 1500, 1.0)
    #   effective_prior = prior_strength × player_career_perf
    #                     + (1 − prior_strength) × CVI_REPLACEMENT_PERF
    #   shrunk_perf     = season_reliability × raw_perf
    #                     + (1 − season_reliability) × effective_prior
    #
    # With no prior data: effective_prior = 40 (falls back to v1.7
    # behavior — debutants get the generic replacement target).
    # With strong prior data: effective_prior = player's own career
    # mean → the shrinkage just regresses toward what we already
    # believe about the player, not toward a generic floor.
    def _shrink_perf(raw_perf, reliab, prior_info):
        if raw_perf is None or pd.isna(raw_perf):
            return None
        if reliab is None or pd.isna(reliab):
            return float(raw_perf)
        # Resolve effective prior using player-specific info if present.
        if prior_info is None:
            effective_prior = CVI_REPLACEMENT_PERF
        else:
            p_perf = prior_info.get('prior_perf')
            p_strength = prior_info.get('prior_strength', 0.0) or 0.0
            if p_perf is None or pd.isna(p_perf):
                effective_prior = CVI_REPLACEMENT_PERF
            else:
                effective_prior = (p_strength * float(p_perf)
                                    + (1.0 - p_strength) * CVI_REPLACEMENT_PERF)
        w = float(reliab)
        return w * float(raw_perf) + (1.0 - w) * effective_prior

    df['_CVI_perf_shrunk'] = df.apply(
        lambda r: _shrink_perf(
            r.get('_CVI_perf'),
            r.get('_CVI_reliab'),
            prior_lookup(r.get('playerId')) if callable(prior_lookup) else None,
        ),
        axis=1,
    )
    # Expose the prior used so the UI can surface "shrunk toward 70"
    # vs "shrunk toward 40 (debutant)" instead of always saying 40.
    if callable(prior_lookup):
        _prior_resolved = df['playerId'].apply(
            lambda pid: prior_lookup(pid) if pid is not None else None
        )
        df['_CVI_prior_perf'] = _prior_resolved.apply(
            lambda x: x.get('prior_perf') if isinstance(x, dict) else None
        )
        df['_CVI_prior_strength'] = _prior_resolved.apply(
            lambda x: x.get('prior_strength') if isinstance(x, dict) else None
        )
        df['_CVI_prior_mins_eff'] = _prior_resolved.apply(
            lambda x: x.get('prior_mins_eff') if isinstance(x, dict) else None
        )
        # Effective shrinkage target = blended prior actually used.
        def _effective_prior_for_row(info):
            if not isinstance(info, dict) or info.get('prior_perf') is None:
                return CVI_REPLACEMENT_PERF
            s = info.get('prior_strength', 0.0) or 0.0
            return s * float(info['prior_perf']) + (1 - s) * CVI_REPLACEMENT_PERF
        df['_CVI_effective_prior'] = _prior_resolved.apply(_effective_prior_for_row)
    else:
        df['_CVI_prior_perf'] = None
        df['_CVI_prior_strength'] = None
        df['_CVI_prior_mins_eff'] = None
        df['_CVI_effective_prior'] = CVI_REPLACEMENT_PERF

    # ---- Final composite ----
    # Note: _CVI_reliab is now BAKED INTO _CVI_perf_shrunk (it's the
    # shrinkage weight); we no longer multiply by it again.
    df['_CVI'] = (df['_CVI_perf_shrunk']
                   * df['_CVI_age']
                   * df['_CVI_league'])

    # ---- Trajectory flag (separate, NOT multiplied into CVI) ----
    # Median shrunk PerformanceQuality among same-position-group
    # same-age-band players. Using shrunk perf (not raw) keeps the
    # comparison apples-to-apples — both numerator and denominator
    # reflect the same sample-discount treatment.
    def _expected_perf(row):
        g = row.get('_cvi_group')
        a = row.get('_cvi_age')
        if g is None or a is None or pd.isna(a):
            return None
        peers = df[
            (df['_cvi_group'] == g)
            & (df['_cvi_age'].between(a - 2, a + 2))
            & df['_CVI_perf_shrunk'].notna()
        ]
        if len(peers) < 10:
            return None
        return float(peers['_CVI_perf_shrunk'].median())

    df['_cvi_expected_perf'] = df.apply(_expected_perf, axis=1)
    df['_CVI_trajectory'] = df['_CVI_perf_shrunk'] - df['_cvi_expected_perf']

    return df[['_CVI', '_CVI_perf', '_CVI_perf_shrunk', '_CVI_age',
                '_CVI_reliab', '_CVI_reliab_ceiling',
                '_CVI_reliab_sample_factor', '_CVI_reliab_mins_to_ceiling',
                '_CVI_prior_perf', '_CVI_prior_strength',
                '_CVI_prior_mins_eff', '_CVI_effective_prior',
                '_CVI_league', '_CVI_trajectory']]


# ==============================================================================
# Career CVI (cross-season + cross-league aggregation)
# ------------------------------------------------------------------------------
# Single-season CVI answers "how good was this player in 2024/25?".
# Career CVI answers "what's the durable estimate combining everything we
# know about this player up to and including season X?".
#
# Aggregation rules (anchored to chosen season; never uses future seasons):
#   1. For each prior season i (counting backwards from anchor):
#        decay_factor_i = CVI_CAREER_DECAY ** seasons_back_i
#        league_factor_i = CVI_LEAGUE_MULTIPLIER[comp_at_season_i]
#                          (translates Camp perf to Liga 3 equivalent;
#                           anchor-season league is applied AT THE END)
#        weight_i        = decay_factor_i × mins_i
#        contribution_i  = perf_i × league_factor_i × weight_i
#   2. career_perf_raw_l3 = Σ contribution_i / Σ weight_i
#   3. effective_mins     = Σ weight_i        (drives reliability shrinkage)
#   4. shrunk_perf        = reliab × career_perf_raw_l3
#                            + (1 − reliab) × CVI_REPLACEMENT_PERF
#      (reliab computed at the anchor season's position group + effective_mins)
#   5. age_at_anchor      = player_age at anchor season's start (Aug of year)
#   6. career_CVI = shrunk_perf × AgeValueMultiplier(age_at_anchor, pos)
#                                × CVI_LEAGUE_MULTIPLIER[league_at_anchor]
#
# Why "anchor at anchor season's league"?
#   The career_perf is now in Liga-3-equivalent units (we translated each
#   season's contribution). To finish in the right scale, we re-apply the
#   anchor season's league multiplier. So a career CVI anchored to a Camp
#   season gets the 0.85 final discount; anchored to a Liga 3 season does
#   not. This keeps Current CVI commensurate with the player's current
#   league context.
#
# Anchored never INCLUDES future seasons (we don't peek). When called for
# "Current CVI", anchor = the player's most recent season; for "Season
# CVI" inside a historical season's view, anchor = that selected season.
CVI_CAREER_DECAY = 0.5           # weighting per season back (steeper than v2.7's 0.6 — current season counts relatively more)
CVI_CAREER_MAX_LOOKBACK = 4      # seasons back included (0..4 = up to 5 seasons)
# v2.8 — current season gets an explicit bonus multiplier on top of decay.
# User: "weight the current season a little bit more". With CURRENT_BONUS=1.5
# and DECAY=0.5, the current season's recency weight is 3× the prior season's
# (1.5 vs 0.5). The per-season MINUTES weighting (mins_played × recency)
# already keeps small-sample seasons from dragging the avg down — this just
# tilts further toward "what they're doing RIGHT NOW".
CVI_CAREER_CURRENT_BONUS = 1.5


def _build_player_season_perf_table(gpa_values_df, player_minutes_df=None):
    """One row per (playerId, seasonId, competitionId) with:
       playerId, seasonId, competitionId, position_group, mins_played, perf_pct

    perf_pct is the player's Total Value /90 percentile WITHIN the same
    (seasonId × position_group) cohort — a sensible historical proxy for
    PerformanceQuality that doesn't require re-running the full role-score
    pipeline for every season.

    Returns empty DataFrame if GPA data is unavailable.
    """
    if gpa_values_df is None or gpa_values_df.empty:
        return pd.DataFrame()
    g = gpa_values_df.copy()
    # Pick the per-90 Total Value column (name varies between snapshots)
    val_col = next((c for c in ('Total Value', 'total_v_per_90',
                                  'Total Value_per_90')
                     if c in g.columns), None)
    if val_col is None:
        return pd.DataFrame()
    # Map raw position to CVI position group
    pos_col = next((c for c in ('position', 'primaryPosition')
                     if c in g.columns), None)
    if pos_col is None:
        return pd.DataFrame()
    g['_cvi_group'] = g[pos_col].map(_cvi_position_group)
    # Defensive numeric coercion (Camp/L3 cross-tier merges sometimes
    # leave val_col as object dtype which breaks rank() with the
    # str/float comparison error).
    g[val_col] = pd.to_numeric(g[val_col], errors='coerce')
    g = g.dropna(subset=['_cvi_group', val_col, 'seasonId', 'playerId'])
    # Within (seasonId, position_group), percentile-rank Total Value/90
    g['_perf_pct'] = (g.groupby(['seasonId', '_cvi_group'])[val_col]
                        .rank(pct=True, method='average') * 100.0)
    mins_col = next((c for c in ('mins_played', 'totalMinutes', 'Minutes')
                      if c in g.columns), None)
    if mins_col is None:
        # Fall back to player_minutes_df if provided
        if player_minutes_df is not None and not player_minutes_df.empty:
            pm = player_minutes_df[['playerId', 'totalMinutes']].rename(
                columns={'totalMinutes': '_mins_filled'}
            )
            g = g.merge(pm, on='playerId', how='left')
            mins_col = '_mins_filled'
        else:
            return pd.DataFrame()
    out = g[['playerId', 'seasonId', '_cvi_group', mins_col, '_perf_pct']].copy()
    out = out.rename(columns={mins_col: 'mins_played',
                                '_cvi_group': 'position_group',
                                '_perf_pct': 'perf_pct'})
    if 'competitionId' in g.columns:
        out['competitionId'] = g['competitionId'].values
    else:
        out['competitionId'] = out['seasonId'].map(competition_for_season)
    return out


def _season_year(season_id):
    """Numeric chronology key from SEASON_ID_MAP labels like '2024/25' → 2024.
    Used to order seasons and compute 'seasons back' from an anchor.
    """
    label = SEASON_ID_MAP.get(int(season_id)) if season_id is not None else None
    if not label:
        return None
    try:
        return int(str(label).split('/')[0])
    except (ValueError, IndexError):
        return None


def compute_career_cvi(player_id, anchor_season_id, *,
                        perf_table, dob_lookup,
                        decay=CVI_CAREER_DECAY,
                        max_lookback=CVI_CAREER_MAX_LOOKBACK):
    """Career-aggregated CVI anchored to anchor_season_id, including that
    season + up to `max_lookback` prior seasons (whichever the player has
    data for). Never peeks at seasons AFTER the anchor.

    Returns dict with:
        career_cvi, career_perf_raw (L3-equivalent), career_perf_shrunk,
        reliability, effective_mins, age_at_anchor, league_at_anchor,
        position_group, n_seasons_used, breakdown (list of per-season dicts)

    Returns None if the player has no GPA seasons at-or-before the anchor.
    """
    if perf_table is None or perf_table.empty or anchor_season_id is None:
        return None
    anchor_year = _season_year(anchor_season_id)
    if anchor_year is None:
        return None

    rows = perf_table[perf_table['playerId'] == player_id].copy()
    if rows.empty:
        return None
    rows['_season_year'] = rows['seasonId'].map(_season_year)
    rows = rows.dropna(subset=['_season_year'])
    rows['_season_year'] = rows['_season_year'].astype(int)
    # Only the anchor season + prior seasons, up to max_lookback back
    rows = rows[(rows['_season_year'] <= anchor_year)
                 & (rows['_season_year'] >= anchor_year - max_lookback)]
    if rows.empty:
        return None
    rows = rows.copy()   # slice of perf_table — write on a copy
    rows['_seasons_back'] = anchor_year - rows['_season_year']
    rows['_decay'] = decay ** rows['_seasons_back']
    # v2.8 — current season (seasons_back==0) gets an explicit recency bonus
    # on top of the decay-to-the-zero (which is 1.0). Prior seasons unaffected.
    rows['_recency'] = rows['_decay'].copy()
    rows.loc[rows['_seasons_back'] == 0, '_recency'] *= CVI_CAREER_CURRENT_BONUS
    rows['_league_factor'] = rows['competitionId'].map(
        lambda c: (CVI_LEAGUE_MULTIPLIER.get(int(c), CVI_LEAGUE_DEFAULT)
                    if c is not None and not pd.isna(c) else CVI_LEAGUE_DEFAULT)
    )
    rows['_weight'] = rows['_recency'] * rows['mins_played'].fillna(0).clip(lower=0)
    rows['_contribution'] = rows['perf_pct'] * rows['_league_factor'] * rows['_weight']

    total_w = float(rows['_weight'].sum())
    if total_w <= 0:
        return None
    career_perf_raw_l3 = float(rows['_contribution'].sum() / total_w)
    effective_mins = total_w   # decay-weighted effective minutes

    # Resolve the anchor row to lock down position group + league for the
    # final shrinkage/multiplier step. Use the most-recent matching season
    # ≤ anchor (handles the case where the player skipped the anchor year).
    anchor_row_candidates = rows.sort_values('_season_year', ascending=False)
    anchor_row = anchor_row_candidates.iloc[0]
    pos_group = anchor_row['position_group']
    league_at_anchor = (CVI_LEAGUE_MULTIPLIER.get(int(anchor_row['competitionId']),
                                                     CVI_LEAGUE_DEFAULT)
                         if anchor_row.get('competitionId') is not None
                         and not pd.isna(anchor_row.get('competitionId'))
                         else CVI_LEAGUE_DEFAULT)

    # Reliability from effective_mins under the anchor-position curve
    reliab, reliab_breakdown = _cvi_reliability_weight(effective_mins, pos_group)

    # Shrinkage toward replacement-level
    shrunk_perf = (reliab * career_perf_raw_l3
                    + (1 - reliab) * CVI_REPLACEMENT_PERF)

    # Age at anchor season (Aug 1 of anchor_year used as a reference date
    # so a player born in March looks "the right age" for that season)
    age_at_anchor = None
    try:
        dob = dob_lookup(player_id) if callable(dob_lookup) else None
        if dob is not None and not pd.isna(dob):
            from datetime import date as _date_cls
            anchor_ref = _date_cls(anchor_year, 8, 1)
            if hasattr(dob, 'date'):
                dob_d = dob.date()
            else:
                dob_d = dob
            age_at_anchor = (anchor_ref - dob_d).days / 365.25
    except Exception:
        age_at_anchor = None
    age_mult = _cvi_age_value_multiplier(age_at_anchor, pos_group)

    career_cvi = shrunk_perf * age_mult * league_at_anchor

    breakdown = (rows.sort_values('_season_year', ascending=False)
                       [['seasonId', '_season_year', '_seasons_back',
                          'competitionId', 'position_group', 'mins_played',
                          'perf_pct', '_league_factor', '_decay', '_weight']]
                       .rename(columns={'_season_year': 'season_year',
                                          '_seasons_back': 'seasons_back',
                                          '_league_factor': 'league_factor',
                                          '_decay': 'decay_factor',
                                          '_weight': 'weight'})
                       .to_dict('records'))

    return {
        'career_cvi': career_cvi,
        'career_perf_raw_l3': career_perf_raw_l3,
        'career_perf_shrunk': shrunk_perf,
        'reliability': reliab,
        'reliability_ceiling': reliab_breakdown.get('ceiling'),
        'reliability_sample_factor': reliab_breakdown.get('sample_factor'),
        'effective_mins': effective_mins,
        'age_at_anchor': age_at_anchor,
        'age_multiplier': age_mult,
        'league_at_anchor': league_at_anchor,
        'position_group': pos_group,
        'anchor_season_id': int(anchor_season_id),
        'n_seasons_used': int(len(rows)),
        'breakdown': breakdown,
    }


def build_player_priors_lookup(perf_table, anchor_season_id,
                                  decay=CVI_CAREER_DECAY,
                                  max_lookback=CVI_CAREER_MAX_LOOKBACK,
                                  full_strength_mins=1500):
    """Pre-compute the empirical-Bayes prior for every player relative
    to an anchor season. Returns a dict:
        {playerId: {'prior_perf': float, 'prior_strength': float (0..1),
                     'prior_mins_eff': float}}

    Used by compute_cvi_columns to shrink each player's season perf
    toward THEIR OWN career mean (when we have enough prior data)
    rather than the generic replacement-level (40). Implements the
    empirical-Bayes pattern: with rich prior data the shrinkage
    target IS the player's career; with no prior data we fall back
    to the league-replacement default.

    Strictly uses seasons PRIOR to anchor_season_id (excludes the
    anchor season itself) to avoid leakage: when judging Caleb's
    2024/25 perf, the prior is built from his 2021/22 + 2022/23 +
    2023/24 data only — never from 2024/25 itself or anything later.
    """
    if perf_table is None or perf_table.empty or anchor_season_id is None:
        return {}
    anchor_year = _season_year(anchor_season_id)
    if anchor_year is None:
        return {}
    anchor_comp = competition_for_season(anchor_season_id)
    pt = perf_table.copy()
    pt['_year'] = pt['seasonId'].map(_season_year)
    pt = pt.dropna(subset=['_year'])
    pt['_year'] = pt['_year'].astype(int)
    # Eligible prior rows:
    #   - STRICTLY prior years (year < anchor_year), up to lookback
    #   - SAME year + DIFFERENT competition — cross-league concurrent
    #     play (e.g. Santi Guzman 23/24 played for Leça in Camp AND
    #     for Atlético CP in Liga 3; Dedé 24/25 Dezembro/Camp +
    #     Sintrense/Liga 3). When rating the Liga 3 portion, the
    #     concurrent Camp portion is real evidence about current
    #     level and should inform the prior.
    pt = pt[
        ((pt['_year'] < anchor_year) & (pt['_year'] >= anchor_year - max_lookback))
        | ((pt['_year'] == anchor_year)
            & (pt['competitionId'].fillna(-1).astype(int) != (anchor_comp or -1)))
    ]
    if pt.empty:
        return {}
    # seasons_back ≥ 0; same-year cross-league gets decay=1.0 (full
    # weight) since it's contemporary evidence.
    pt['_seasons_back'] = (anchor_year - pt['_year']).clip(lower=0)
    pt['_decay'] = decay ** pt['_seasons_back']
    pt['_league_factor'] = pt['competitionId'].map(
        lambda c: (CVI_LEAGUE_MULTIPLIER.get(int(c), CVI_LEAGUE_DEFAULT)
                    if c is not None and not pd.isna(c) else CVI_LEAGUE_DEFAULT)
    )
    pt['_weight'] = pt['_decay'] * pt['mins_played'].fillna(0).clip(lower=0)
    pt['_contrib'] = pt['perf_pct'] * pt['_league_factor'] * pt['_weight']
    grouped = pt.groupby('playerId').agg(
        _sum_w=('_weight', 'sum'),
        _sum_c=('_contrib', 'sum'),
    )
    out = {}
    for pid, r in grouped.iterrows():
        w = float(r['_sum_w'])
        if w <= 0:
            continue
        prior_perf = float(r['_sum_c'] / w)
        # Prior strength ramps linearly from 0 (no prior) to 1.0 (at or
        # above full_strength_mins of decay-weighted prior minutes).
        strength = min(w / float(full_strength_mins), 1.0)
        out[int(pid)] = {
            'prior_perf': prior_perf,
            'prior_strength': strength,
            'prior_mins_eff': w,
        }
    return out


def most_recent_season_for_player(perf_table, player_id):
    """Return the most recent seasonId this player has GPA data for,
    or None if they have none. Used to anchor 'Current CVI' in the
    bio row.

    Tiebreaker for players with two same-year league rows (e.g. Santi
    Guzman 23/24 Leça-Camp + Atlético-CP-Liga-3): pick the seasonId
    where the player logged MORE MINUTES. The other league's data
    still contributes via Career CVI's same-year cross-league
    aggregation; this choice only affects which league_at_anchor
    multiplier is applied to the final composite (so a player who
    played mostly in Liga 3 gets a Liga-3-framed Current CVI).
    """
    if perf_table is None or perf_table.empty:
        return None
    rows = perf_table[perf_table['playerId'] == player_id].copy()
    if rows.empty:
        return None
    rows['_y'] = rows['seasonId'].map(_season_year)
    rows = rows.dropna(subset=['_y'])
    if rows.empty:
        return None
    rows['_mins'] = pd.to_numeric(rows.get('mins_played', 0),
                                     errors='coerce').fillna(0)
    rows = rows.sort_values(['_y', '_mins'], ascending=[False, False])
    return int(rows.iloc[0]['seasonId'])


# ==============================================================================
# Market-context features (consumed by the v2 EUR regression)
# ==============================================================================
# These are NOT inputs to CVI — they're signals that shift how the MARKET
# prices a player at a given quality level. Nationality drives sell-on
# premiums, team success drives visibility, xG over/underperformance
# captures finishing skill the market rewards/discounts. Computed
# per-(player, season) and surfaced alongside CVI in the Player Profile.
def compute_market_features(player_id, season_id, *,
                              raw_events_df, matches_summary_df,
                              player_details_df,
                              player_minutes_data,
                              team_name=None,
                              opta_team_lookup=None):
    """Per-(player, season) bundle of market-context features for the
    v2 transfer-value regression.

    Returns dict with:
        xg_residual_season        goals - xG, non-pen, this season
        xg_residual_career        goals - xG, non-pen, all seasons
        xg_residual_per90_season  same /90
        ass_residual_season       assists - xA proxy
        ass_residual_career       same career-cumulative
        passport_nationality      str (e.g. 'Portugal', 'Brazil')
        birth_nationality         str
        team_opta_rating          float (current team strength)
        team_ppm_season           team's points-per-match this season
        team_league_position      1-N rank within the season (NaN if unknown)
        positions_played_career   count of distinct primaryPositions across career
        seasons_played            count of distinct seasons in our data
    """
    out = {
        'xg_residual_season': None, 'xg_residual_career': None,
        'xg_residual_per90_season': None,
        'ass_residual_season': None, 'ass_residual_career': None,
        'passport_nationality': None, 'birth_nationality': None,
        'team_opta_rating': None, 'team_ppm_season': None,
        'team_league_position': None,
        'positions_played_career': None, 'seasons_played': None,
    }

    # ---- xG over/under (non-penalty) ----
    try:
        ev = raw_events_df[
            (raw_events_df['player.id'] == player_id)
            & raw_events_df['shot.xg'].notna()
            & (raw_events_df['type.primary'] != 'penalty')
        ]
        if not ev.empty:
            goals_c = ev['shot.isGoal'].fillna(False).astype(bool).sum()
            xg_c = float(ev['shot.xg'].sum())
            out['xg_residual_career'] = float(goals_c) - xg_c
            ev_s = ev[ev['seasonId'] == season_id] if 'seasonId' in ev.columns else ev
            if not ev_s.empty:
                goals_s = ev_s['shot.isGoal'].fillna(False).astype(bool).sum()
                xg_s = float(ev_s['shot.xg'].sum())
                out['xg_residual_season'] = float(goals_s) - xg_s
                # Use the player's totalMinutes for the season for /90
                pm = player_minutes_data.get(season_id) if isinstance(player_minutes_data, dict) else None
                mins = None
                if pm is not None and 'playerId' in pm.columns:
                    sub = pm[pm['playerId'] == player_id]
                    if not sub.empty:
                        mins = float(sub['totalMinutes'].sum())
                if mins and mins > 0:
                    out['xg_residual_per90_season'] = (out['xg_residual_season']
                                                         / mins * 90.0)
    except Exception:
        pass

    # ---- xA proxy: count of 'shot_assist'-tagged passes by player → xG of the shot they assisted ----
    try:
        # Player's shot-assist events
        sa_mask = (raw_events_df['player.id'] == player_id) & (
            raw_events_df.get('type.secondary', pd.Series(dtype='object'))
                          .apply(lambda x: isinstance(x, (list, np.ndarray))
                                  and 'shot_assist' in x)
        )
        sa = raw_events_df[sa_mask]
        # Approximate: next event in same match with non-null shot.xg = shot they assisted
        if not sa.empty and 'matchId' in raw_events_df.columns:
            # Lookup shot xG of the next event in the same match for each assist event
            ev_sorted = (raw_events_df[['matchId', 'matchTimestamp',
                                          'shot.xg', 'shot.isGoal', 'player.id']]
                          .sort_values(['matchId', 'matchTimestamp'])
                          .reset_index(drop=True))
            ev_sorted['next_xg'] = ev_sorted.groupby('matchId')['shot.xg'].shift(-1)
            ev_sorted['next_goal'] = ev_sorted.groupby('matchId')['shot.isGoal'].shift(-1)
            joined = sa.reset_index().merge(
                ev_sorted[['matchId', 'matchTimestamp', 'next_xg', 'next_goal']],
                on=['matchId', 'matchTimestamp'], how='left',
            )
            xa_c = joined['next_xg'].dropna().sum()
            assists_c = joined['next_goal'].fillna(False).sum()
            out['ass_residual_career'] = float(assists_c) - float(xa_c)
            jl_s = joined[joined.get('seasonId') == season_id] if 'seasonId' in joined.columns else joined
            if not jl_s.empty:
                xa_s = jl_s['next_xg'].dropna().sum()
                assists_s = jl_s['next_goal'].fillna(False).sum()
                out['ass_residual_season'] = float(assists_s) - float(xa_s)
    except Exception:
        pass

    # ---- Nationality ----
    try:
        if (player_details_df is not None and not player_details_df.empty
                and player_id in player_details_df.index):
            row = player_details_df.loc[player_id]
            out['passport_nationality'] = row.get('passportArea')
            out['birth_nationality'] = row.get('birthArea')
    except Exception:
        pass

    # ---- Team Opta rating ----
    try:
        if opta_team_lookup and team_name:
            out['team_opta_rating'] = opta_team_lookup(team_name)
    except Exception:
        pass

    # ---- Team PPM + league position this season ----
    try:
        if team_name and matches_summary_df is not None and season_id is not None:
            sm = matches_summary_df[matches_summary_df['seasonId'] == season_id].copy()
            # Parse scores
            def _parse_score(s):
                try:
                    if pd.isna(s) or '-' not in str(s): return (None, None)
                    h, a = str(s).split('-')
                    return (int(h.strip()), int(a.strip()))
                except Exception:
                    return (None, None)
            sm[['h_g','a_g']] = sm['score'].apply(_parse_score).apply(pd.Series)
            sm = sm.dropna(subset=['h_g','a_g'])
            # Points per team
            from collections import defaultdict
            pts = defaultdict(int); games = defaultdict(int)
            for _, m in sm.iterrows():
                h, a = m['homeTeamName'], m['awayTeamName']
                hg, ag = m['h_g'], m['a_g']
                games[h] += 1; games[a] += 1
                if hg > ag: pts[h] += 3
                elif ag > hg: pts[a] += 3
                else: pts[h] += 1; pts[a] += 1
            if team_name in games and games[team_name] > 0:
                out['team_ppm_season'] = pts[team_name] / games[team_name]
                # League position
                ppm_all = {t: pts[t]/games[t] for t in games if games[t] > 0}
                ranked = sorted(ppm_all.items(), key=lambda kv: -kv[1])
                for rank, (t, _) in enumerate(ranked, start=1):
                    if t == team_name:
                        out['team_league_position'] = rank
                        break
    except Exception:
        pass

    # ---- Position versatility + seasons played (career) ----
    try:
        if isinstance(player_minutes_data, dict):
            pos_set = set(); seasons_set = set()
            for sid, _pm in player_minutes_data.items():
                if not isinstance(_pm, pd.DataFrame) or 'playerId' not in _pm.columns:
                    continue
                sub = _pm[_pm['playerId'] == player_id]
                if sub.empty: continue
                seasons_set.add(sid)
                if 'primaryPosition' in sub.columns:
                    pos_set.update(p for p in sub['primaryPosition'].dropna().unique())
            out['positions_played_career'] = len(pos_set) if pos_set else None
            out['seasons_played'] = len(seasons_set) if seasons_set else None
    except Exception:
        pass

    return out


OUTPUT_METRICS = ['Goals', 'Assists', 'xG', 'npxG', 'xA', 'xAOP', 'xASP', 'xT', 'xTOP', 'xTSP', 'Second assists', 'Shots', 'xG per Shot']
PASSING_METRICS = ['Creating Value', 'Linking Value', 'Passes', 'Passes successful', 'Passes successful %', 'Long passes', 'Long passes successful', 'Long passes successful %', 'Crosses', 'Crosses successful', 'Crosses successful %', 'Through passes', 'Through passes successful', 'Progressive Passes', 'Passes to final third', 'Passes to final third successful', 'Forward passes', 'Forward passes successful', 'Back passes', 'Back passes successful', 'Passes to penalty area', 'Passes to penalty area successful', 'Deep Completions', 'Throw-ins', 'Avg max throw-in distance', 'Throw-ins into box', 'Avg max throw-in into box distance', 'Avg max throw-in into box aerial distance']
DEFENSIVE_METRICS = ['Possessions won', 'Interceptions', 'Aerial duels', 'Aerial duels successful', 'Aerial duels successful %', 'Sliding tackles', 'Sliding tackles successful', 'Sliding tackles successful %', 'Recoveries', 'Recoveries Opp Half', 'Counterpressing Recoveries', 'Defensive duels', 'Defensive duels successful', 'Defensive duels successful %', 'Defensive duels vs offensive duel', 'Defensive duels vs offensive duel successful %', 'Dribbles faced', 'Dribbled past %', 'Dribbled past % (proj)', 'Clearances', 'Fouls', 'Yellow cards', 'Red cards']
DRIBBLING_METRICS = ['Dribbles', 'Dribbles successful', 'Dribbles successful %', 'Touches in penalty area', 'Progressive runs', 'Fouls suffered']
GOALKEEPING_METRICS = ['shotsOnTargetAgainst', 'goalsConceded', 'exits', 'saves', 'goalsPrevented', 'goalsPreventedPerSOT', 'savePercentage', 'recoveries_gk', 'passes_gk', 'passesSuccessful_gk', 'Long passes successful %', 'longPasses_gk', 'longPassesSuccessful_gk']
OFF_BALL_DEFENDING_METRICS = ['Defensive Area', 'Territorial Dominance', 'Opp xT into Def Area OE', 'Opp xT from Def Area OE', 'Opp Pass Success % into Def Area']
DISTRIBUTION_METRICS_BY_POSITION = {
    'Shot Stopper': ['goalsPrevented', 'goalsPreventedPerSOT', 'exits', 'Long passes successful %', 'recoveries_gk'],
    'Cross Claimer': ['goalsPrevented', 'goalsPreventedPerSOT', 'exits', 'Long passes successful %', 'recoveries_gk'],
    'Ball-playing GK': ['goalsPrevented', 'goalsPreventedPerSOT', 'exits', 'recoveries_gk', 'passes_gk', 'Passes successful %', 'longPassesSuccessful_gk'],
    'Ball-Playing Centerback': ['xTOP', 'Passes', 'Passes successful %', 'Progressive Passes', 'Progressive runs'],
    'Stopper': ['Aerial duels', 'Aerial duels successful %', 'Defensive duels', 'Defensive duels successful %','Interceptions', 'Recoveries', 'Clearances'],
    'Athletic Centerback': ['npxG', 'Progressive runs', 'Aerial duels', 'Aerial duels successful %', 'Defensive duels', 'Defensive duels successful %','Interceptions', 'Recoveries', 'Clearances'],
    'Box-to-Box': ['Progressive Passes', 'npxG', 'Shots', 'xAOP', 'xTOP', 'Progressive runs', 'Dribbles successful', 'Aerial duels successful',  'Defensive duels successful', 'Interceptions', 'Recoveries'],
    'Holding Mid':['Passes', 'Passes successful %',  'Progressive Passes', 'xTOP', 'Aerial duels successful',  'Defensive duels successful', 'Interceptions', 'Recoveries'],
    'Ball-Winning Mid': ['Aerial duels', 'Aerial duels successful %',  'Defensive duels', 'Defensive duels successful %', 'Interceptions', 'Recoveries', 'Recoveries Opp Half'],
    'Deep-lying Playmaker': ['Passes', 'Passes successful %',  'Progressive Passes', 'xTOP','xAOP', 'Progressive runs'],
    'Advanced Playmaker': ['Goals', 'npxG', 'Shots', 'xG per Shot', 'Assists', 'xAOP', 'xTOP', 'Progressive runs', 'Dribbles successful'],
    'Full Back': ['Aerial duels', 'Aerial duels successful %', 'Defensive duels', 'Defensive duels successful %','Interceptions', 'Recoveries', 'Clearances'],
    'Wingback': ['Assists', 'xAOP', 'xTOP', 'Passes', 'Crosses successful', 'Progressive runs','Interceptions', 'Recoveries'],
    'Inverted Full Back': ['Progressive Passes', 'xTOP', 'Progressive runs', 'Defensive duels', 'Defensive duels successful %','Interceptions', 'Recoveries'],
    'Wide Winger': ['Goals', 'npxG', 'Shots', 'xG per Shot',  'Assists', 'xAOP', 'xTOP', 'Crosses successful', 'Progressive runs', 'Dribbles', 'Dribbles successful %',  'Loss index'],
    'Creative Winger': ['Goals', 'npxG', 'Shots', 'xG per Shot',  'Assists', 'xAOP', 'xTOP','Progressive runs', 'Dribbles', 'Dribbles successful %',  'Loss index'],
    'Inside Forward': ['Goals', 'npxG', 'Shots', 'xG per Shot',  'Assists', 'xAOP','Loss index'],
    'Shadow Striker': ['Goals', 'npxG', 'Shots', 'xG per Shot',  'Assists', 'xAOP', 'xTOP', 'Progressive runs', 'Dribbles', 'Dribbles successful %',  'Loss index'],
    'Mobile Striker': ['Goals', 'npxG', 'Shots', 'xG per Shot',  'Assists', 'xAOP', 'xTOP', 'Progressive runs', 'Dribbles', 'Dribbles successful %',  'Loss index'],
    'Poacher': ['Goals', 'npxG', 'Shots', 'xG per Shot',  'Assists', 'xAOP', 'Loss index'],
    'Target Man': ['Goals', 'npxG', 'Shots', 'xG per Shot',  'Assists', 'xAOP', 'Loss index', 'Aerial duels', 'Aerial duels successful %','Clearances'],
    'Pressing Forward': ['Goals', 'npxG', 'Shots', 'xG per Shot',  'Assists', 'xAOP', 'Loss index', 'Defensive duels successful', 'Interceptions', 'Recoveries', 'Counterpressing Recoveries']
}

# Override from config.yaml if available
_config = load_config()
if _config:
    POSITION_GROUPS = _config.get('position_groups', POSITION_GROUPS)
    WEIGHTS = _config.get('weights', WEIGHTS)
    INVERT_METRICS = _config.get('invert_metrics', INVERT_METRICS)
    OUTPUT_METRICS = _config.get('metric_categories', {}).get('output', OUTPUT_METRICS)
    PASSING_METRICS = _config.get('metric_categories', {}).get('passing', PASSING_METRICS)
    DEFENSIVE_METRICS = _config.get('metric_categories', {}).get('defensive', DEFENSIVE_METRICS)
    DRIBBLING_METRICS = _config.get('metric_categories', {}).get('dribbling', DRIBBLING_METRICS)
    GOALKEEPING_METRICS = _config.get('metric_categories', {}).get('goalkeeping', GOALKEEPING_METRICS)
    DISTRIBUTION_METRICS_BY_POSITION = _config.get('distribution_metrics_by_position', DISTRIBUTION_METRICS_BY_POSITION)
    # Metrics that are kept in the role weights (so they still contribute to the
    # composite role-fit score) but hidden from the radar chart axes.
    RADAR_HIDDEN_METRICS: set[str] = set(_config.get('radar_hidden_metrics', []) or [])
    logger.info("Configuration loaded from config.yaml")
else:
    RADAR_HIDDEN_METRICS = set()

# Formation coordinates for XI graphic (Opta 0-100 coordinate system)
# Note: Left positions use higher x values (right side of screen) to match broadcast view
FORMATION_COORDS = {
    '4-4-2': {
        'positions': ['GK', 'LB', 'LCB', 'RCB', 'RB', 'LM', 'LCM', 'RCM', 'RM', 'LST', 'RST'],
        'coords': [(50, 7), (85, 25), (62, 25), (38, 25), (15, 25),
                   (85, 50), (62, 50), (38, 50), (15, 50), (62, 75), (38, 75)]
    },
    '4-3-3': {
        'positions': ['GK', 'LB', 'LCB', 'RCB', 'RB', 'LCM', 'CDM', 'RCM', 'LW', 'CF', 'RW'],
        'coords': [(50, 7), (85, 25), (62, 25), (38, 25), (15, 25),
                   (70, 50), (50, 45), (30, 50), (82, 72), (50, 80), (18, 72)]
    },
    '4-2-3-1': {
        'positions': ['GK', 'LB', 'LCB', 'RCB', 'RB', 'LDM', 'RDM', 'LAM', 'CAM', 'RAM', 'CF'],
        'coords': [(50, 7), (85, 25), (62, 25), (38, 25), (15, 25),
                   (65, 42), (35, 42), (78, 62), (50, 62), (22, 62), (50, 80)]
    },
    '3-5-2': {
        'positions': ['GK', 'LCB', 'CB', 'RCB', 'LWB', 'LCM', 'CDM', 'RCM', 'RWB', 'LST', 'RST'],
        'coords': [(50, 7), (75, 25), (50, 25), (25, 25), (90, 50),
                   (65, 50), (50, 42), (35, 50), (10, 50), (62, 75), (38, 75)]
    },
    '3-4-3': {
        'positions': ['GK', 'LCB', 'CB', 'RCB', 'LM', 'LCM', 'RCM', 'RM', 'LW', 'CF', 'RW'],
        'coords': [(50, 7), (75, 25), (50, 25), (25, 25), (85, 50),
                   (62, 50), (38, 50), (15, 50), (80, 75), (50, 80), (20, 75)]
    },
    '4-1-4-1': {
        'positions': ['GK', 'LB', 'LCB', 'RCB', 'RB', 'CDM', 'LM', 'LCM', 'RCM', 'RM', 'CF'],
        'coords': [(50, 7), (85, 25), (62, 25), (38, 25), (15, 25),
                   (50, 38), (85, 55), (62, 55), (38, 55), (15, 55), (50, 78)]
    },
    '4-4-1-1': {
        'positions': ['GK', 'LB', 'LCB', 'RCB', 'RB', 'LM', 'LCM', 'RCM', 'RM', 'CAM', 'CF'],
        'coords': [(50, 7), (85, 25), (62, 25), (38, 25), (15, 25),
                   (85, 48), (62, 48), (38, 48), (15, 48), (50, 65), (50, 80)]
    },
    '5-3-2': {
        'positions': ['GK', 'LWB', 'LCB', 'CB', 'RCB', 'RWB', 'LCM', 'CDM', 'RCM', 'LST', 'RST'],
        'coords': [(50, 7), (90, 28), (72, 22), (50, 22), (28, 22), (10, 28),
                   (70, 50), (50, 45), (30, 50), (62, 75), (38, 75)]
    },
    '5-4-1': {
        'positions': ['GK', 'LWB', 'LCB', 'CB', 'RCB', 'RWB', 'LM', 'LCM', 'RCM', 'RM', 'CF'],
        'coords': [(50, 7), (90, 28), (72, 22), (50, 22), (28, 22), (10, 28),
                   (82, 52), (62, 52), (38, 52), (18, 52), (50, 78)]
    },
    '3-4-1-2': {
        'positions': ['GK', 'LCB', 'CB', 'RCB', 'LM', 'LCM', 'RCM', 'RM', 'CAM', 'LST', 'RST'],
        'coords': [(50, 7), (75, 25), (50, 25), (25, 25), (85, 50),
                   (62, 50), (38, 50), (15, 50), (50, 65), (62, 78), (38, 78)]
    },
    '4-4-1': {
        'positions': ['GK', 'LB', 'LCB', 'RCB', 'RB', 'LM', 'LCM', 'RCM', 'RM', 'CF', 'CF'],
        'coords': [(50, 7), (85, 25), (62, 25), (38, 25), (15, 25),
                   (85, 50), (62, 50), (38, 50), (15, 50), (50, 75), (50, 75)]
    },
    '3-4-2-1': {
        'positions': ['GK', 'LCB', 'CB', 'RCB', 'LM', 'LCM', 'RCM', 'RM', 'LAM', 'RAM', 'CF'],
        'coords': [(50, 7), (75, 25), (50, 25), (25, 25), (85, 48),
                   (62, 48), (38, 48), (15, 48), (70, 65), (30, 65), (50, 80)]
    },
    '4-1-3-2': {
        'positions': ['GK', 'LB', 'LCB', 'RCB', 'RB', 'CDM', 'LAM', 'CAM', 'RAM', 'LST', 'RST'],
        'coords': [(50, 7), (85, 25), (62, 25), (38, 25), (15, 25),
                   (50, 40), (75, 58), (50, 58), (25, 58), (62, 78), (38, 78)]
    },
    '4-2-1-3': {
        'positions': ['GK', 'LB', 'LCB', 'RCB', 'RB', 'LDM', 'RDM', 'CAM', 'LW', 'CF', 'RW'],
        'coords': [(50, 7), (85, 25), (62, 25), (38, 25), (15, 25),
                   (65, 42), (35, 42), (50, 58), (82, 72), (50, 80), (18, 72)]
    },
    '5-3-1': {
        'positions': ['GK', 'LWB', 'LCB', 'CB', 'RCB', 'RWB', 'LCM', 'CDM', 'RCM', 'CF', 'CF'],
        'coords': [(50, 7), (90, 28), (72, 22), (50, 22), (28, 22), (10, 28),
                   (70, 50), (50, 45), (30, 50), (50, 75), (50, 75)]
    },
    '4-3-2-1': {
        'positions': ['GK', 'LB', 'LCB', 'RCB', 'RB', 'LCM', 'CDM', 'RCM', 'LAM', 'RAM', 'CF'],
        'coords': [(50, 7), (85, 25), (62, 25), (38, 25), (15, 25),
                   (70, 45), (50, 42), (30, 45), (70, 62), (30, 62), (50, 80)]
    },
    '4-5-1': {
        'positions': ['GK', 'LB', 'LCB', 'RCB', 'RB', 'LM', 'LCM', 'CDM', 'RCM', 'RM', 'CF'],
        'coords': [(50, 7), (85, 25), (62, 25), (38, 25), (15, 25),
                   (85, 50), (65, 48), (50, 42), (35, 48), (15, 50), (50, 78)]
    },
    '3-5-1': {
        'positions': ['GK', 'LCB', 'CB', 'RCB', 'LWB', 'LCM', 'CDM', 'RCM', 'RWB', 'CF', 'CF'],
        'coords': [(50, 7), (75, 25), (50, 25), (25, 25), (90, 50),
                   (65, 50), (50, 42), (35, 50), (10, 50), (50, 75), (50, 75)]
    },
    '4-3-1': {
        'positions': ['GK', 'LB', 'LCB', 'RCB', 'RB', 'LCM', 'CDM', 'RCM', 'CF', 'CF', 'CF'],
        'coords': [(50, 7), (85, 25), (62, 25), (38, 25), (15, 25),
                   (70, 50), (50, 45), (30, 50), (50, 75), (50, 75), (50, 75)]
    },
    '4-3-2': {
        'positions': ['GK', 'LB', 'LCB', 'RCB', 'RB', 'LCM', 'CDM', 'RCM', 'LST', 'RST', 'RST'],
        'coords': [(50, 7), (85, 25), (62, 25), (38, 25), (15, 25),
                   (70, 50), (50, 45), (30, 50), (62, 75), (38, 75), (38, 75)]
    },
}

# Shadow Team tag categories with hex colors
SHADOW_TAG_CATEGORIES = {
    'A - No Brainer': '#2ecc71',
    'B - Possible Starter': '#3498db',
    'C - Quality Depth Squad': '#f1c40f',
    'D - Quality but Injury Prone': '#e74c3c',
    'E - Depth from Lisbon': '#9b59b6',
}

# Maps each formation slot to relevant role names whose _Score columns to display
POSITION_SLOT_TO_ROLES = {
    'GK': ['Shot Stopper', 'Cross Claimer', 'Ball-playing GK'],
    'LB': ['Full Back', 'Inverted Full Back', 'Wingback'],
    'RB': ['Full Back', 'Inverted Full Back', 'Wingback'],
    'LCB': ['Ball-Playing Centerback', 'Stopper', 'Athletic Centerback'],
    'RCB': ['Ball-Playing Centerback', 'Stopper', 'Athletic Centerback'],
    'CB': ['Ball-Playing Centerback', 'Stopper', 'Athletic Centerback'],
    'LWB': ['Wingback', 'Full Back', 'Wide Winger'],
    'RWB': ['Wingback', 'Full Back', 'Wide Winger'],
    'CDM': ['Holding Mid', 'Ball-Winning Mid', 'Deep-lying Playmaker'],
    'LDM': ['Holding Mid', 'Ball-Winning Mid', 'Deep-lying Playmaker'],
    'RDM': ['Holding Mid', 'Ball-Winning Mid', 'Deep-lying Playmaker'],
    'LCM': ['Box-to-Box', 'Ball-Winning Mid', 'Deep-lying Playmaker', 'Advanced Playmaker'],
    'RCM': ['Box-to-Box', 'Ball-Winning Mid', 'Deep-lying Playmaker', 'Advanced Playmaker'],
    'LM': ['Wide Winger', 'Creative Winger', 'Inside Forward'],
    'RM': ['Wide Winger', 'Creative Winger', 'Inside Forward'],
    'LAM': ['Advanced Playmaker', 'Creative Winger', 'Inside Forward'],
    'RAM': ['Advanced Playmaker', 'Creative Winger', 'Inside Forward'],
    'CAM': ['Advanced Playmaker', 'Shadow Striker', 'Creative Winger'],
    'LW': ['Wide Winger', 'Creative Winger', 'Inside Forward'],
    'RW': ['Wide Winger', 'Creative Winger', 'Inside Forward'],
    'CF': ['Mobile Striker', 'Target Man', 'Poacher', 'Pressing Forward', 'Shadow Striker'],
    'LST': ['Mobile Striker', 'Target Man', 'Poacher', 'Pressing Forward', 'Shadow Striker'],
    'RST': ['Mobile Striker', 'Target Man', 'Poacher', 'Pressing Forward', 'Shadow Striker'],
}


# ==============================================================================
# 4. HELPER & PLOTTING FUNCTIONS
# ==============================================================================

# --- Helper Functions for Team Formation XI Graphic ---
def get_team_primary_formation(events_df, team_name):
    """Get the most commonly used formation for a team."""
    team_events = events_df[events_df['team.name'] == team_name]
    if 'team.formation' not in team_events.columns:
        return '4-4-2'
    formation_counts = team_events['team.formation'].dropna().value_counts()
    if len(formation_counts) > 0:
        return formation_counts.index[0]
    return '4-4-2'  # Default fallback

def get_team_starting_xi(events_df, team_name):
    """Get the most frequent player at each position for a team."""
    team_events = events_df[events_df['team.name'] == team_name]

    position_players = {}
    # Include all position codes used in the data (Wyscout format)
    all_positions = ['GK',
                     'RB', 'RB5', 'RCB', 'RCB3', 'CB', 'LCB', 'LCB3', 'LB', 'LB5',
                     'RWB', 'LWB',
                     'RDMF', 'DMF', 'LDMF', 'RCMF', 'RCMF3', 'CMF', 'LCMF', 'LCMF3',
                     'RAMF', 'AMF', 'LAMF',
                     'RW', 'RWF', 'LW', 'LWF',
                     'RCF', 'CF', 'LCF', 'SS']

    for pos in all_positions:
        pos_events = team_events[team_events['player.position'] == pos]
        if len(pos_events) > 0:
            player_counts = pos_events.groupby(['player.id', 'player.name']).size()
            if len(player_counts) > 0:
                top_player = player_counts.idxmax()
                position_players[pos] = {'id': top_player[0], 'name': top_player[1]}

    return position_players

def map_players_to_formation(starting_xi, formation_slots):
    """Map actual player positions to formation display slots."""
    mapping = {}

    # Position equivalencies - which Wyscout positions can fill each formation slot
    equivalents = {
        'GK': ['GK'],
        # Defenders
        'LB': ['LB', 'LB5', 'LWB'], 'RB': ['RB', 'RB5', 'RWB'],
        'LCB': ['LCB', 'LCB3', 'CB'], 'RCB': ['RCB', 'RCB3', 'CB'], 'CB': ['CB', 'LCB3', 'RCB3', 'LCB', 'RCB'],
        'LWB': ['LWB', 'LB5', 'LB', 'LWF'], 'RWB': ['RWB', 'RB5', 'RB', 'RWF'],
        # Midfielders
        'LDM': ['LDMF', 'DMF', 'LCMF'], 'RDM': ['RDMF', 'DMF', 'RCMF'], 'CDM': ['DMF', 'LDMF', 'RDMF', 'LCMF', 'RCMF'],
        'LCM': ['LCMF', 'LCMF3', 'CMF', 'LDMF'], 'RCM': ['RCMF', 'RCMF3', 'CMF', 'RDMF'],
        'LM': ['LWF', 'LW', 'LAMF', 'LCMF'], 'RM': ['RWF', 'RW', 'RAMF', 'RCMF'],
        # Attacking Mids
        'LAM': ['LAMF', 'AMF', 'LWF', 'LW'], 'RAM': ['RAMF', 'AMF', 'RWF', 'RW'], 'CAM': ['AMF', 'LAMF', 'RAMF', 'SS'],
        # Wingers
        'LW': ['LW', 'LWF', 'LAMF'], 'RW': ['RW', 'RWF', 'RAMF'],
        # Forwards
        'CF': ['CF', 'LCF', 'RCF', 'SS'], 'LST': ['LCF', 'CF', 'SS'], 'RST': ['RCF', 'CF', 'SS'],
    }

    used_players = set()
    for slot in formation_slots:
        for pos in equivalents.get(slot, [slot]):
            if pos in starting_xi and starting_xi[pos]['name'] not in used_players:
                mapping[slot] = starting_xi[pos]
                used_players.add(starting_xi[pos]['name'])
                break

    # Second pass: assign remaining unmatched players to unfilled slots
    unfilled = [s for s in formation_slots if s not in mapping]
    remaining = [(pos, info) for pos, info in starting_xi.items()
                 if info['name'] not in used_players]

    # Coarse position tiers for proximity matching
    _tier = {
        'GK': 0,
        'LB': 1, 'RB': 1, 'LB5': 1, 'RB5': 1, 'LCB': 1, 'RCB': 1,
        'CB': 1, 'LCB3': 1, 'RCB3': 1, 'LWB': 1.5, 'RWB': 1.5,
        'LDMF': 2, 'RDMF': 2, 'DMF': 2,
        'LCMF': 2.5, 'RCMF': 2.5, 'LCMF3': 2.5, 'RCMF3': 2.5,
        'CMF': 2.5, 'AMF': 3, 'LAMF': 3, 'RAMF': 3,
        'LW': 3.5, 'RW': 3.5, 'LWF': 3.5, 'RWF': 3.5,
        'CF': 4, 'LCF': 4, 'RCF': 4, 'SS': 4,
    }
    _slot_tier = {
        'GK': 0,
        'LB': 1, 'RB': 1, 'LCB': 1, 'RCB': 1, 'CB': 1,
        'LWB': 1.5, 'RWB': 1.5,
        'LDM': 2, 'RDM': 2, 'CDM': 2,
        'LCM': 2.5, 'RCM': 2.5,
        'LM': 3, 'RM': 3, 'LAM': 3, 'RAM': 3, 'CAM': 3,
        'LW': 3.5, 'RW': 3.5,
        'CF': 4, 'LST': 4, 'RST': 4,
    }

    for slot in unfilled:
        st = _slot_tier.get(slot, 2.5)
        # Sort remaining by proximity to this slot's tier
        remaining.sort(key=lambda r: abs(_tier.get(r[0], 2.5) - st))
        if remaining:
            pos, info = remaining.pop(0)
            mapping[slot] = info
            used_players.add(info['name'])
        else:
            mapping[slot] = {'name': slot, 'id': None}

    return mapping

def create_formation_graphic(formation, starting_xi, team_name):
    """Create a pitch graphic showing the team formation with player names."""
    from mplsoccer import VerticalPitch

    pitch = VerticalPitch(pitch_type='opta', pitch_color='#1a472a', line_color='white',
                          linewidth=1, goal_type='box')
    fig, ax = pitch.draw(figsize=(6, 8))

    # Get formation coordinates (fallback to 4-4-2 if unknown)
    formation_key = formation if formation in FORMATION_COORDS else '4-4-2'
    formation_data = FORMATION_COORDS[formation_key]

    # Map actual player positions to formation slots
    position_mapping = map_players_to_formation(starting_xi, formation_data['positions'])

    for slot, coords in zip(formation_data['positions'], formation_data['coords']):
        x, y = coords
        # Draw player circle
        ax.scatter(x, y, s=800, c='white', edgecolors='#1a472a', linewidth=2, zorder=5)

        # Get player name for this slot
        player_info = position_mapping.get(slot, {'name': slot})
        player_name = player_info.get('name', slot)

        # Shorten to last name
        if ' ' in player_name:
            display_name = player_name.split()[-1][:12]
        else:
            display_name = player_name[:12]

        ax.text(x, y - 6, display_name, ha='center', va='top', fontsize=7,
                fontweight='bold', color='white')

    ax.set_title(f'{team_name}\n{formation}', fontsize=12, fontweight='bold',
                 color='white', pad=10)
    fig.patch.set_facecolor('#1a472a')

    return fig

# --- Helper Functions for Shadow Team ---
def create_shadow_team_graphic(formation_key, player_assignments, tag_assignments, shadow_team_name, player_stats_df=None):
    """Create a pitch graphic for the shadow team.

    player_assignments: {slot: [player_name, ...]} — list of players per slot
    tag_assignments: {slot: {player_name: {category, label}}} — per-player tags
    player_stats_df: DataFrame with playerName, teamName, totalMinutes columns
    """
    from mplsoccer import VerticalPitch

    pitch = VerticalPitch(pitch_type='opta', pitch_color='#1a472a', line_color='white',
                          linewidth=1, goal_type='box')
    fig, ax = pitch.draw(figsize=(8, 11))

    formation_data = FORMATION_COORDS.get(formation_key, FORMATION_COORDS['4-4-2'])
    used_tags = set()

    # Spread coords outward from center and shift up the pitch
    cx, cy = 50, 50  # pitch center in opta coords
    spread = 1.15     # 15% further from center
    y_shift = 12      # shift everything up the pitch
    spread_coords = []
    for (ox, oy) in formation_data['coords']:
        sx = cx + (ox - cx) * spread
        sy = cy + (oy - cy) * spread + y_shift
        # Clamp to pitch bounds
        sx = max(2, min(98, sx))
        sy = max(2, min(98, sy))
        spread_coords.append((sx, sy))

    # Build a quick lookup for team/minutes keyed by display_key "Name (Team)"
    player_info_map = {}
    if player_stats_df is not None and not player_stats_df.empty:
        for _, row in player_stats_df[['playerName', 'teamName', 'totalMinutes']].iterrows():
            dk = f"{row['playerName']} ({row['teamName']})"
            player_info_map[dk] = (row['playerName'], row['teamName'], int(row['totalMinutes']))

    for slot, coords in zip(formation_data['positions'], spread_coords):
        x, y = coords
        players = player_assignments.get(slot, [])

        # Always draw a white outline circle with the position label inside
        ax.scatter(x, y, s=900, c='none', edgecolors='white', linewidth=2, zorder=5)
        ax.text(x, y, slot, ha='center', va='center', fontsize=9,
                fontweight='bold', color='white', zorder=6)

        if players:
            # Stack player info below the circle
            y_offset = 2.8
            for p_display_key in players:
                p_tag_info = tag_assignments.get(slot, {}).get(p_display_key, {})
                p_category = p_tag_info.get('category', 'Current Starter')
                p_label = p_tag_info.get('label', '')
                p_color = SHADOW_TAG_CATEGORIES.get(p_category, '#ffffff')
                used_tags.add(p_category)

                # Extract real name and team from lookup (display_key -> (name, team, mins))
                info = player_info_map.get(p_display_key)
                if info:
                    real_name, team_name, mins = info
                else:
                    # Fallback: strip " (Team)" suffix
                    real_name = p_display_key.rsplit(' (', 1)[0] if ' (' in p_display_key else p_display_key
                    team_name, mins = '', 0

                # Player name
                ax.text(x, y - y_offset, real_name, ha='center', va='top', fontsize=8.5,
                        fontweight='bold', color=p_color, zorder=6)
                y_offset += 1.2

                # Team & minutes underneath in smaller italic
                if team_name:
                    detail_text = f"{team_name} | {mins:,}'"
                    ax.text(x, y - y_offset, detail_text, ha='center', va='top', fontsize=7,
                            fontstyle='italic', color='#cccccc', zorder=6)
                    y_offset += 1.8

                # Custom label
                if p_label:
                    ax.text(x, y - y_offset, p_label, ha='center', va='top', fontsize=6.5,
                            fontstyle='italic', color='#aaaaaa', zorder=6)
                    y_offset += 1.6

    # Legend for used tag categories
    if used_tags:
        legend_handles = []
        for tag in sorted(used_tags):
            color = SHADOW_TAG_CATEGORIES[tag]
            legend_handles.append(Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
                                         markeredgecolor='white', markersize=8, label=tag))
        ax.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                  ncol=min(len(legend_handles), 4), fontsize=7, facecolor='#1a472a',
                  edgecolor='white', labelcolor='white', framealpha=0.9)

    title = shadow_team_name if shadow_team_name else "Shadow Team"
    ax.set_title(f'{title}\n{formation_key}', fontsize=12, fontweight='bold',
                 color='white', pad=10)
    fig.patch.set_facecolor('#1a472a')

    return fig


def get_player_role_scores(player_row, slot_name):
    """Get relevant role scores for a player based on their formation slot."""
    roles = POSITION_SLOT_TO_ROLES.get(slot_name, [])
    scores = {}
    for role in roles:
        col = f'{role}_Score'
        if col in player_row.index:
            val = player_row[col]
            if pd.notna(val):
                scores[role] = round(float(val), 1)
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


# --- Helper Function for League Table ---
def calculate_league_table(matches_df, team_list):
    """Calculate league standings for a list of teams."""
    standings = {}

    for team in team_list:
        standings[team] = {
            'P': 0, 'W': 0, 'D': 0, 'L': 0,
            'GF': 0, 'GA': 0, 'GD': 0, 'Pts': 0
        }

    for _, match in matches_df.iterrows():
        home_team = match['homeTeamName']
        away_team = match['awayTeamName']
        score = match.get('score', '')

        # Only process matches between teams in our list
        if home_team not in team_list or away_team not in team_list:
            continue

        if not score or pd.isna(score) or '-' not in str(score):
            continue

        try:
            home_goals, away_goals = map(int, str(score).split('-'))
        except (ValueError, AttributeError):
            continue

        # Update home team stats
        standings[home_team]['P'] += 1
        standings[home_team]['GF'] += home_goals
        standings[home_team]['GA'] += away_goals

        # Update away team stats
        standings[away_team]['P'] += 1
        standings[away_team]['GF'] += away_goals
        standings[away_team]['GA'] += home_goals

        # Determine result
        if home_goals > away_goals:
            standings[home_team]['W'] += 1
            standings[home_team]['Pts'] += 3
            standings[away_team]['L'] += 1
        elif home_goals < away_goals:
            standings[away_team]['W'] += 1
            standings[away_team]['Pts'] += 3
            standings[home_team]['L'] += 1
        else:
            standings[home_team]['D'] += 1
            standings[home_team]['Pts'] += 1
            standings[away_team]['D'] += 1
            standings[away_team]['Pts'] += 1

    # Calculate goal difference
    for team in standings:
        standings[team]['GD'] = standings[team]['GF'] - standings[team]['GA']

    # Convert to DataFrame
    table_df = pd.DataFrame.from_dict(standings, orient='index')
    table_df.index.name = 'Team'
    table_df = table_df.reset_index()

    # Sort by Points, then GD, then GF
    table_df = table_df.sort_values(
        by=['Pts', 'GD', 'GF'],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    # Add position column and reorder columns (Pts after P)
    table_df.insert(0, 'Pos', range(1, len(table_df) + 1))
    table_df = table_df[['Pos', 'Team', 'P', 'Pts', 'W', 'D', 'L', 'GF', 'GA', 'GD']]

    return table_df

# --- Helper for Player Radars (from Cell 11) ---
def calculate_and_merge(base_df, events_df, stat_name, primary_type=None, bool_condition=None):
    """
    Helper function from notebook Cell 11.
    Calculates a stat based on a filter and merges it into the base DataFrame.
    (FIXED to accept primary_type and bool_condition)
    """
    # Start with a base condition (all True)
    filter_condition = pd.Series(True, index=events_df.index)
    
    if primary_type:
        filter_condition &= (events_df.get('type.primary') == primary_type)
    
    if bool_condition is not None:
        # Align indices before combining
        bool_condition_aligned = bool_condition.reindex(filter_condition.index, fill_value=False)
        filter_condition &= bool_condition_aligned
        
    # Ensure player.id is numeric for groupby
    events_df['player.id'] = pd.to_numeric(events_df['player.id'], errors='coerce')
    safe_condition = filter_condition & events_df['player.id'].notna()
    
    if safe_condition.empty or not safe_condition.any():
        stat_series = pd.Series(dtype='int').rename(stat_name)
    else:
        # Group by the integer version of player.id
        stat_series = events_df[safe_condition].groupby(events_df['player.id'].astype(int)).size()
        stat_series.name = stat_name
        
    base_df = base_df.merge(stat_series, left_index=True, right_index=True, how='left')
    return base_df

# --- NEW Helper for Robust List Checking ---
def calculate_and_merge_list(base_df, events_df, stat_name, tag_to_find, primary_type=None, and_condition=None):
    """
    Robust helper to count stats by checking for a tag in the 'type.secondary' list.
    """
    # Base condition: Check if the tag is in the list (if the list exists)
    condition = events_df.get('type.secondary', pd.Series(dtype='object')).apply(
        lambda x: isinstance(x, (list, np.ndarray)) and tag_to_find in x
    )
    
    if primary_type:
        condition &= (events_df.get('type.primary') == primary_type)
        
    if and_condition is not None:
        # Align indices before combining conditions
        and_condition_aligned = and_condition.reindex(condition.index, fill_value=False)
        condition = condition & and_condition_aligned
        
    # --- THIS IS THE FIX ---
    # We must pass 'primary_type' and the 'condition' as a 'bool_condition'
    # to the main helper function.
    return calculate_and_merge(
        base_df, 
        events_df, 
        stat_name, 
        primary_type=primary_type, # <-- Pass the primary_type
        bool_condition=condition   # <-- Pass the condition as a bool_condition
    )

@st.cache_data

def add_custom_dribble_success(events_df):
    """
    Applies custom logic to determine if a dribble was successful.
    
    IMPROVEMENT: Looks for the next event *by the same team* to skip 
    interleaved opponent defensive events.
    """
    # 1. Work on a sorted copy
    df = events_df.sort_values(by=['matchId', 'matchTimestamp']).copy()
    
    # --- FIX: Force timestamp to datetime object to avoid string errors ---
    df['matchTimestamp'] = pd.to_datetime(df['matchTimestamp'], errors='coerce')
    
    # 2. Calculate Distance to Goal for the CURRENT event
    # (100, 50) is the center of the opponent's goal
    df['dist_to_goal'] = np.sqrt((100 - df['location.x'].fillna(100))**2 + (50 - df['location.y'].fillna(50))**2)
    
    # 3. Get details of the NEXT event for the SAME TEAM
    # This skips opponent events (like defensive duels) that happen in between
    team_grouped = df.groupby(['matchId', 'team.name'])
    
    df['next_team_player_id'] = team_grouped['player.id'].shift(-1)
    df['next_team_type'] = team_grouped['type.primary'].shift(-1)
    df['next_team_accurate'] = team_grouped['pass.accurate'].shift(-1)
    df['next_team_start_x'] = team_grouped['location.x'].shift(-1)
    df['next_team_start_y'] = team_grouped['location.y'].shift(-1)
    df['next_team_timestamp'] = team_grouped['matchTimestamp'].shift(-1)

    # Calculate distance for the NEXT team event start
    df['next_team_start_dist'] = np.sqrt((100 - df['next_team_start_x'].fillna(100))**2 + (50 - df['next_team_start_y'].fillna(50))**2)
    
    # For Pass Success Logic: We need where the next pass ENDED
    df['next_team_end_x'] = team_grouped['pass.endLocation.x'].shift(-1)
    df['next_team_end_y'] = team_grouped['pass.endLocation.y'].shift(-1)
    df['next_team_end_dist'] = np.sqrt((100 - df['next_team_end_x'].fillna(100))**2 + (50 - df['next_team_end_y'].fillna(50))**2)

    # 4. Time Delta Check
    # If the next team event is > 4 seconds later, the chain is broken
    df['time_diff'] = (df['next_team_timestamp'] - df['matchTimestamp']).dt.total_seconds()
    is_chain_intact = df['time_diff'] <= 4.0

    # 5. Identify Dribble Attempts
    df['is_dribble_attempt'] = df.get('type.secondary', pd.Series(dtype='object')).apply(
        lambda x: isinstance(x, (list, np.ndarray)) and 'dribble' in x
    ) | (df.get('groundDuel.takeOn') == True)

    # 6. Evaluate Success Conditions
    
    # Condition A: Same player, next action is closer to goal (Carry, Shot, etc.)
    cond_a = (
        is_chain_intact &
        (df['next_team_player_id'] == df['player.id']) & 
        (df['next_team_type'] != 'pass') & 
        (df['next_team_start_dist'] < df['dist_to_goal'])
    )
    
    # Condition B: "Successful forward pass from a dribble"
    cond_b = (
        is_chain_intact &
        (df['next_team_player_id'] == df['player.id']) & 
        (df['next_team_type'] == 'pass') & 
        (df['next_team_accurate'] == True) & 
        (df['next_team_end_dist'] < df['dist_to_goal'])
    )
    
    # Condition C: "Touch of an attacking teammate closer to goal"
    cond_c = (
        is_chain_intact &
        (df['next_team_player_id'] != df['player.id']) & 
        (df['next_team_start_dist'] < df['dist_to_goal'])
    )
    
    # Condition D: Foul Suffered
    cond_d = df.get('type.secondary', pd.Series(dtype='object')).apply(
        lambda x: isinstance(x, (list, np.ndarray)) and 'foul_suffered' in x
    )

    # 7. Assign Result
    df['is_custom_dribble_success'] = df['is_dribble_attempt'] & (cond_a | cond_b | cond_c | cond_d)
    
    # Cleanup
    cols_to_drop = [c for c in df.columns if 'next_team_' in c] + ['dist_to_goal', 'time_diff']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    return df

@st.cache_data(ttl=86400)  # 24h — the old 1h expiry forced an hourly cold
# recompute from the 2.3M-row events frame; harmless now (32GB RAM,
# matplotlib leak fixed, ~8 small per-season frames cached at most).
def calculate_all_player_stats(_raw_events_df, _player_minutes_df, season_id=None, cache_version=STATS_CACHE_VERSION, cache_scope=None):
    # cache_scope: extra @st.cache_data key discriminator. season_id alone is
    # AMBIGUOUS for All-Seasons (season_id=None) — both leagues pass None, so
    # the in-memory cache collided (Campeonato All-Seasons returned Liga 3's
    # result in 1.1s and never computed/wrote all_702). load_and_score passes
    # the comp tuple here so each (None, league) scope is distinct. Not used
    # in the body; the disk-cache key still derives from season_id + events.
    """
    A new, streamlined, and correct function to calculate all player stats
    for the player profile page (Per 90 and Totals).
    season_id is used as a cache key so Streamlit recomputes when the season changes.
    """
    # Disk cache: load pre-computed results if available
    _REQUIRED_STAT_COLS = {'Throw-ins', 'Avg max throw-in distance', 'Throw-ins into box', 'Avg max throw-in into box distance', 'Avg max throw-in into box aerial distance', 'Defensive Area', 'Opp xT into Def Area', 'Opp Pass Success % into Def Area', 'Opp xT from Def Area', 'Territorial Dominance', 'Opp xT into Def Area OE', 'Opp xT from Def Area OE', 'Territorial Dominance OE', 'xTOP', 'xTSP'}
    _scope_key = _stats_scope_key(season_id, _raw_events_df)
    cache_path = os.path.join(STATS_CACHE_DIR, f'player_stats_{STATS_CACHE_VERSION}_{_scope_key}.parquet')
    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        if _REQUIRED_STAT_COLS.issubset(cached.columns):
            # Sanity check: goalsConceded should always be per-90 (typical
            # GK rate ≈ 0.5–1.5). Judge by the MEDIAN across keepers with
            # real minutes, not the max: a few data-gap keepers carry
            # broken tiny totalMinutes (e.g. a full season shown as 90')
            # whose legit per-90 division yields 20+, and the old max>=5
            # check invalidated five healthy caches on EVERY boot — the
            # resulting cold recompute loop is what segfaulted the HF
            # Space (2026-07-14). A cache of season TOTALS would put the
            # median regular keeper at 20+, so median>=5 still catches
            # the real failure this check exists for.
            if 'goalsConceded' in cached.columns and 'totalMinutes' in cached.columns:
                _gk_rows = cached[
                    (cached['goalsConceded'] > 0)
                    & (pd.to_numeric(cached['totalMinutes'],
                                      errors='coerce') >= 450)]
                if (not _gk_rows.empty
                        and float(_gk_rows['goalsConceded'].median()) >= 5):
                    print(f"Cache has goalsConceded as totals (median="
                          f"{_gk_rows['goalsConceded'].median():.1f}); "
                          f"invalidating and recomputing")
                    os.remove(cache_path)
                else:
                    print(f"Loading cached player stats for scope {_scope_key}")
                    if cached.index.name == 'playerId':
                        cached = cached.reset_index()
                    return cached
            else:
                print(f"Loading cached player stats for scope {_scope_key}")
                if cached.index.name == 'playerId':
                    cached = cached.reset_index()
                return cached
        else:
            print(f"Cache outdated (missing columns), recomputing stats for scope {_scope_key}")
            os.remove(cache_path)

    print("--- STARTING: New All-Player-Stats Calculation ---")
    
    events_df = _raw_events_df.copy()
    
    # --- NEW: Apply Custom Dribble Logic ---
    # Ensure this helper function is defined in app.py!
    events_df = add_custom_dribble_success(events_df)
    # ---------------------------------------

    base_df = _player_minutes_df.copy()
    base_df['totalMinutes'] = pd.to_numeric(base_df['totalMinutes'], errors='coerce').fillna(0)
    base_df = base_df.set_index('playerId')

    # Ensure player.id is int for all merging
    events_df['player.id'] = pd.to_numeric(events_df['player.id'], errors='coerce')
    events_df = events_df.dropna(subset=['player.id'])
    events_df['player.id'] = events_df['player.id'].astype(int)

    # --- Helper Function for Aggregation ---
    def count_and_merge(base_df, events_df, stat_name, filter_condition):
        """Groups events by player.id, counts them, and merges into base_df."""
        # Align index if bool_condition is passed
        if isinstance(filter_condition, pd.Series):
             filter_condition = filter_condition.reindex(events_df.index, fill_value=False)
        
        safe_condition = filter_condition & events_df['player.id'].notna()
        
        if safe_condition.empty or not safe_condition.any():
            stat_series = pd.Series(dtype='int', name=stat_name)
        else:
            stat_series = events_df[safe_condition].groupby(events_df['player.id']).size()
            stat_series.name = stat_name
            
        base_df = base_df.merge(stat_series, left_index=True, right_index=True, how='left')
        return base_df

    # --- Helper for list-checking (vectorized for performance) ---
    # Pre-compute secondary tags as sets for O(1) lookup
    _secondary_col = events_df.get('type.secondary', pd.Series(dtype='object'))
    _secondary_sets = _secondary_col.apply(
        lambda x: set(x) if isinstance(x, (list, np.ndarray)) else set()
    )

    def check_secondary_list(tag):
        """Returns a boolean Series if tag is in the 'type.secondary' list (vectorized)."""
        return _secondary_sets.apply(lambda s: tag in s)

    # --- Step 1: Calculate All Counting Stats (Totals) ---
    print("Step 1: Calculating counting stats...")
    
    # -- Basic Event Counts --
    base_df = count_and_merge(base_df, events_df, 'Goals', events_df.get('shot.isGoal') == True)
    base_df = count_and_merge(base_df, events_df, 'Assists', check_secondary_list('assist'))
    base_df = count_and_merge(base_df, events_df, 'Shots', events_df['type.primary'] == 'shot')
    base_df = count_and_merge(base_df, events_df, 'Shots on target', (events_df['type.primary'] == 'shot') & (events_df['shot.onTarget'] == True))
    base_df = count_and_merge(base_df, events_df, 'Interceptions', events_df['type.primary'] == 'interception')
    base_df = count_and_merge(base_df, events_df, 'Clearances', events_df['type.primary'] == 'clearance')
    base_df = count_and_merge(base_df, events_df, 'Fouls', events_df['type.primary'] == 'infraction')
    base_df = count_and_merge(base_df, events_df, 'Offsides', events_df['type.primary'] == 'offside')
    base_df = count_and_merge(base_df, events_df, 'Yellow cards', events_df.get('infraction.yellowCard') == True)
    base_df = count_and_merge(base_df, events_df, 'Red cards', events_df.get('infraction.redCard') == True)
    base_df = count_and_merge(base_df, events_df, 'Touches in penalty area', check_secondary_list('touch_in_box'))
    base_df = count_and_merge(base_df, events_df, 'Progressive runs', check_secondary_list('progressive_run'))
    base_df = count_and_merge(base_df, events_df, 'Fouls suffered', check_secondary_list('foul_suffered'))
    base_df = count_and_merge(base_df, events_df, 'Second assists', check_secondary_list('second_assist'))

    # -- Passing Metrics --
    pass_events = events_df[events_df['type.primary'] == 'pass']
    pass_accurate = pass_events.get('pass.accurate') == True
    base_df = count_and_merge(base_df, pass_events, 'Passes', pd.Series(True, index=pass_events.index))
    base_df = count_and_merge(base_df, pass_events, 'Passes successful', pass_accurate)
    base_df = count_and_merge(base_df, pass_events, 'Long passes', check_secondary_list('long_pass'))
    base_df = count_and_merge(base_df, pass_events, 'Long passes successful', check_secondary_list('long_pass') & pass_accurate)
    base_df = count_and_merge(base_df, pass_events, 'Crosses', check_secondary_list('cross'))
    base_df = count_and_merge(base_df, pass_events, 'Crosses successful', check_secondary_list('cross') & pass_accurate)
    base_df = count_and_merge(base_df, pass_events, 'Through passes', check_secondary_list('through_pass'))
    base_df = count_and_merge(base_df, pass_events, 'Through passes successful', check_secondary_list('through_pass') & pass_accurate)
    base_df = count_and_merge(base_df, pass_events, 'Passes to final third', check_secondary_list('pass_to_final_third'))
    base_df = count_and_merge(base_df, pass_events, 'Passes to final third successful', check_secondary_list('pass_to_final_third') & pass_accurate)
    base_df = count_and_merge(base_df, pass_events, 'Forward passes', check_secondary_list('forward_pass'))
    base_df = count_and_merge(base_df, pass_events, 'Forward passes successful', check_secondary_list('forward_pass') & pass_accurate)
    base_df = count_and_merge(base_df, pass_events, 'Back passes', check_secondary_list('back_pass'))
    base_df = count_and_merge(base_df, pass_events, 'Back passes successful', check_secondary_list('back_pass') & pass_accurate)
    base_df = count_and_merge(base_df, pass_events, 'Passes to penalty area', check_secondary_list('pass_to_penalty_area'))
    base_df = count_and_merge(base_df, pass_events, 'Passes to penalty area successful', check_secondary_list('pass_to_penalty_area') & pass_accurate)

    # -- Dueling & Defensive Metrics --
    duel_events = events_df[events_df['type.primary'] == 'duel']
    base_df = count_and_merge(base_df, duel_events, 'Duels', pd.Series(True, index=duel_events.index))
    
    # Success = Kept Poss OR Recovered Poss OR First Touch (Aerial) OR Stopped Progress (New Logic)
    base_df = count_and_merge(base_df, duel_events, 'Duels successful', 
                              (duel_events.get('groundDuel.keptPossession') == True) | 
                              (duel_events.get('groundDuel.recoveredPossession') == True) | 
                              (duel_events.get('groundDuel.stoppedProgress') == True) | 
                              (duel_events.get('aerialDuel.firstTouch') == True))
    
    base_df = count_and_merge(base_df, duel_events, 'Aerial duels', check_secondary_list('aerial_duel'))
    base_df = count_and_merge(base_df, duel_events, 'Aerial duels successful', check_secondary_list('aerial_duel') & (duel_events.get('aerialDuel.firstTouch') == True))

    # Aerial duels in defensive penalty box
    def_box = (check_secondary_list('aerial_duel') &
               (duel_events.get('location.x', 0) < 16) &
               (duel_events.get('location.y', 0) > 20) &
               (duel_events.get('location.y', 0) < 80))
    base_df = count_and_merge(base_df, duel_events, 'Aerial duels def box', def_box)
    base_df = count_and_merge(base_df, duel_events, 'Aerial duels def box successful',
                              def_box & (duel_events.get('aerialDuel.firstTouch') == True))

    # Aerial duels in attacking penalty box
    att_box = (check_secondary_list('aerial_duel') &
               (duel_events.get('location.x', 0) > 84) &
               (duel_events.get('location.y', 0) > 20) &
               (duel_events.get('location.y', 0) < 80))
    base_df = count_and_merge(base_df, duel_events, 'Aerial duels att box', att_box)
    base_df = count_and_merge(base_df, duel_events, 'Aerial duels att box successful',
                              att_box & (duel_events.get('aerialDuel.firstTouch') == True))

    base_df = count_and_merge(base_df, duel_events, 'Defensive duels', check_secondary_list('defensive_duel'))
    # --- FIX: Added stoppedProgress ---
    base_df = count_and_merge(base_df, duel_events, 'Defensive duels successful', 
                              check_secondary_list('defensive_duel') & 
                              ((duel_events.get('groundDuel.recoveredPossession') == True) | (duel_events.get('groundDuel.stoppedProgress') == True)))
    
    base_df = count_and_merge(base_df, duel_events, 'Offensive duels', check_secondary_list('offensive_duel'))
    base_df = count_and_merge(base_df, duel_events, 'Offensive duels successful', check_secondary_list('offensive_duel') & (duel_events.get('groundDuel.progressedWithBall') == True))

    # Dribbled past % inputs (Lucas 2026-06-30): defensive ground duels against a
    # DRIBBLE attempt (groundDuel.takeOn). "Dribbled past" = the defender neither
    # stopped the attacker's progress nor recovered possession — the attacker got
    # closer to goal. Stop-but-kept stalemates count as prevented, not past.
    _dribble_faced = (check_secondary_list('defensive_duel')
                      & (duel_events.get('groundDuel.takeOn') == True))
    base_df = count_and_merge(base_df, duel_events, 'Dribbles faced', _dribble_faced)
    base_df = count_and_merge(base_df, duel_events, 'Dribbled past',
                              _dribble_faced
                              & ~((duel_events.get('groundDuel.stoppedProgress') == True)
                                  | (duel_events.get('groundDuel.recoveredPossession') == True)))

    # Defensive duels split by CONTEST KIND (Lucas 2026-07): vs take-on (the
    # attacker tries to beat you off the dribble, groundDuel.takeOn) vs an
    # offensive duel (attacker shields/holds up/carries under challenge).
    # ~15%/85% of defensive duels league-wide. Success = same flags as the
    # pooled 'Defensive duels successful' (stoppedProgress OR recovered).
    _dd_win = ((duel_events.get('groundDuel.stoppedProgress') == True)
               | (duel_events.get('groundDuel.recoveredPossession') == True))
    _dd_vs_off = (check_secondary_list('defensive_duel')
                  & (duel_events.get('groundDuel.takeOn') == False))
    base_df = count_and_merge(base_df, duel_events, 'Defensive duels vs take-on', _dribble_faced)
    base_df = count_and_merge(base_df, duel_events, 'Defensive duels vs take-on successful',
                              _dribble_faced & _dd_win)
    base_df = count_and_merge(base_df, duel_events, 'Defensive duels vs offensive duel', _dd_vs_off)
    base_df = count_and_merge(base_df, duel_events, 'Defensive duels vs offensive duel successful',
                              _dd_vs_off & _dd_win)
    
    base_df = count_and_merge(base_df, duel_events, 'Sliding tackles', check_secondary_list('sliding_tackle'))
    # --- FIX: Added stoppedProgress ---
    base_df = count_and_merge(base_df, duel_events, 'Sliding tackles successful', 
                              check_secondary_list('sliding_tackle') & 
                              ((duel_events.get('groundDuel.recoveredPossession') == True) | (duel_events.get('groundDuel.stoppedProgress') == True)))

    # Average height of defensive actions (in metres)
    _def_primary = events_df['type.primary'].isin(['interception', 'clearance'])
    _def_secondary = (check_secondary_list('defensive_duel') |
                      check_secondary_list('sliding_tackle') |
                      check_secondary_list('aerial_duel'))
    def_actions = events_df[(_def_primary | _def_secondary) & events_df['location.x'].notna()]
    avg_def_height = (def_actions.groupby('player.id')['location.x'].mean() * 1.05)
    avg_def_height.name = 'Avg defensive action height'
    base_df = base_df.merge(avg_def_height, left_index=True, right_index=True, how='left')

    # -- Dribbles (Custom Logic) --
    # Attempt: is_dribble_attempt == True
    base_df = count_and_merge(base_df, events_df, 'Dribbles', events_df.get('is_dribble_attempt') == True)
    # Success: is_custom_dribble_success == True
    base_df = count_and_merge(base_df, events_df, 'Dribbles successful', events_df.get('is_custom_dribble_success') == True)

    # -- Losses & Recoveries --
    base_df = count_and_merge(base_df, events_df, 'Losses', check_secondary_list('loss'))
    base_df = count_and_merge(base_df, events_df, 'Losses Opp Half', check_secondary_list('loss') & (events_df.get('location.x', 0) >= 50))
    base_df = count_and_merge(base_df, events_df, 'Recoveries', check_secondary_list('recovery'))
    base_df = count_and_merge(base_df, events_df, 'Recoveries Opp Half', check_secondary_list('recovery') & (events_df.get('location.x', 0) >= 50))
    base_df = count_and_merge(base_df, events_df, 'Counterpressing Recoveries', check_secondary_list('counterpressing_recovery'))

    # Possessions won (Lucas 2026-06-30): a ball-winning defensive action that
    # STARTS a new possession for the player's team — an interception or duel
    # primary, or a recovery-tagged touch/pass, on the first event
    # (possession.eventIndex == 0) of a possession owned by the player's team.
    # Dead-ball restarts (throw-in / free-kick / corner / goal-kick primaries)
    # never match; clearances count only when Wyscout tags them as recoveries.
    base_df = count_and_merge(
        base_df, events_df, 'Possessions won',
        (events_df.get('possession.eventIndex') == 0)
        & (events_df.get('possession.team.name') == events_df.get('team.name'))
        & (events_df['type.primary'].isin(['interception', 'duel'])
           | check_secondary_list('recovery')))

    # -- Throw-In Metrics --
    try:
        throwin_events = events_df[events_df['type.primary'] == 'throw_in'].copy()
        print(f"  Throw-in events found: {len(throwin_events)}")
        if not throwin_events.empty:
            base_df = count_and_merge(base_df, throwin_events, 'Throw-ins', pd.Series(True, index=throwin_events.index))
            # Avg of top 10 longest throw-in distances per player (raw length, not per-90)
            if 'pass.length' in throwin_events.columns:
                ti_top10_avg = throwin_events.groupby('player.id')['pass.length'].apply(
                    lambda x: x.nlargest(min(10, len(x))).mean()
                )
                ti_top10_avg.name = 'Avg max throw-in distance'
                base_df = base_df.merge(ti_top10_avg, left_index=True, right_index=True, how='left')
            else:
                print("  WARNING: 'pass.length' column not found in throw-in events")
                base_df['Avg max throw-in distance'] = 0.0
            # Throw-ins into attacking penalty box (end x >= 84, 20 <= end y <= 80)
            if 'pass.endLocation.x' in throwin_events.columns and 'pass.endLocation.y' in throwin_events.columns:
                ti_into_box = throwin_events[
                    (throwin_events['pass.endLocation.x'] >= 84) &
                    (throwin_events['pass.endLocation.y'] >= 20) &
                    (throwin_events['pass.endLocation.y'] <= 80)
                ]
                base_df = count_and_merge(base_df, ti_into_box, 'Throw-ins into box', pd.Series(True, index=ti_into_box.index))
                if not ti_into_box.empty:
                    ti_box_top10_avg = ti_into_box.groupby('player.id')['pass.length'].apply(
                        lambda x: x.nlargest(min(10, len(x))).mean() if len(x) > 0 else 0.0
                    )
                    ti_box_top10_avg.name = 'Avg max throw-in into box distance'
                    base_df = base_df.merge(ti_box_top10_avg, left_index=True, right_index=True, how='left')

                    # Throw-ins into box where the next action is an aerial duel
                    sorted_events = events_df.sort_values(by=['matchId', 'minute', 'second']).reset_index(drop=True)
                    ti_box_aerial_indices = []
                    for idx in ti_into_box.index:
                        # Find this event's position in the sorted timeline
                        match_id = ti_into_box.loc[idx, 'matchId'] if 'matchId' in ti_into_box.columns else None
                        if match_id is None:
                            continue
                        match_events = sorted_events[sorted_events['matchId'] == match_id]
                        evt_minute = ti_into_box.loc[idx, 'minute']
                        evt_second = ti_into_box.loc[idx, 'second']
                        # Find the throw-in in the sorted match events
                        pos_mask = (match_events['minute'] == evt_minute) & (match_events['second'] == evt_second) & (match_events['type.primary'] == 'throw_in')
                        positions = match_events[pos_mask].index
                        if len(positions) == 0:
                            continue
                        pos = positions[0]
                        # Get the next event in the match
                        next_pos = pos + 1
                        if next_pos in match_events.index:
                            next_event = match_events.loc[next_pos]
                            next_secondary = next_event.get('type.secondary', '')
                            if isinstance(next_secondary, (list, set)):
                                is_aerial = 'aerial_duel' in next_secondary
                            else:
                                is_aerial = 'aerial_duel' in str(next_secondary)
                            if is_aerial:
                                ti_box_aerial_indices.append(idx)

                    ti_box_aerial = ti_into_box.loc[ti_into_box.index.isin(ti_box_aerial_indices)]
                    if not ti_box_aerial.empty and 'pass.length' in ti_box_aerial.columns:
                        ti_box_aerial_avg = ti_box_aerial.groupby('player.id')['pass.length'].apply(
                            lambda x: x.nlargest(min(10, len(x))).mean() if len(x) > 0 else 0.0
                        )
                        ti_box_aerial_avg.name = 'Avg max throw-in into box aerial distance'
                        base_df = base_df.merge(ti_box_aerial_avg, left_index=True, right_index=True, how='left')
                    else:
                        base_df['Avg max throw-in into box aerial distance'] = 0.0
                else:
                    base_df['Avg max throw-in into box distance'] = 0.0
                    base_df['Avg max throw-in into box aerial distance'] = 0.0
            else:
                base_df['Throw-ins into box'] = 0.0
                base_df['Avg max throw-in into box distance'] = 0.0
                base_df['Avg max throw-in into box aerial distance'] = 0.0
        else:
            print("  WARNING: No throw-in events found in data")
            base_df['Throw-ins'] = 0.0
            base_df['Avg max throw-in distance'] = 0.0
            base_df['Throw-ins into box'] = 0.0
            base_df['Avg max throw-in into box distance'] = 0.0
            base_df['Avg max throw-in into box aerial distance'] = 0.0
    except Exception as e:
        print(f"  ERROR computing throw-in metrics: {e}")
        import traceback; traceback.print_exc()
        base_df['Throw-ins'] = 0.0
        base_df['Avg max throw-in distance'] = 0.0
        base_df['Throw-ins into box'] = 0.0
        base_df['Avg max throw-in into box distance'] = 0.0
        base_df['Avg max throw-in into box aerial distance'] = 0.0

    # --- Step 1b: Defensive Area Metrics ---
    print("Step 1b: Calculating Defensive Area metrics...")
    try:
        from scipy.stats import chi2 as _chi2_dist
        CHI2_68_2DF = _chi2_dist(2).ppf(0.68)  # ~2.2788
        MIN_DEF_ACTIONS = 5

        # -- Open-play filter: set-piece = delivery + next 5 actions in possession --
        # possession.types tags ALL events in a SP possession, so we use eventIndex to limit
        _set_piece_tags = {'corner', 'free_kick', 'goal_kick', 'throw_in', 'penalty'}
        _poss_types_col = events_df.get('possession.types', pd.Series(dtype='object'))
        _in_sp_possession = _poss_types_col.apply(
            lambda x: bool(set(x) & _set_piece_tags) if isinstance(x, (list, np.ndarray, set)) else False
        )
        _event_idx = events_df.get('possession.eventIndex', pd.Series(dtype='float64')).fillna(99)
        _is_set_piece = _in_sp_possession & (_event_idx <= 5)
        _is_open_play = ~_is_set_piece

        # -- Open-play defensive actions per player --
        _def_mask_full = (
            events_df['type.primary'].isin(['interception', 'clearance'])
            | check_secondary_list('defensive_duel')
            | check_secondary_list('sliding_tackle')
            | check_secondary_list('aerial_duel')
            | check_secondary_list('recovery')
        )
        _open_def = events_df[
            _def_mask_full & _is_open_play
            & events_df['location.x'].notna()
            & events_df['location.y'].notna()
        ][['player.id', 'matchId', 'team.name', 'player.position', 'location.x', 'location.y']].copy()
        _open_def['x_m'] = _open_def['location.x'] * 1.05
        _open_def['y_m'] = _open_def['location.y'] * 0.68

        # -- Per-player ellipse parameters (filtered to primary position) --
        _ellipse_params = {}  # {player_id: (mean, cov_inv, area_sq_m)}
        for _pid, _grp in _open_def.groupby('player.id'):
            # Filter to primary position to avoid inflated areas for multi-position players
            _pos_counts = _grp['player.position'].dropna().value_counts()
            if not _pos_counts.empty:
                _primary_pos = _pos_counts.index[0]
                _grp = _grp[_grp['player.position'] == _primary_pos]
            if len(_grp) < MIN_DEF_ACTIONS:
                continue
            _coords = _grp[['x_m', 'y_m']].values
            _mean = _coords.mean(axis=0)
            _cov = np.cov(_coords.T)
            _det = np.linalg.det(_cov)
            if _det <= 1e-10:
                continue
            _area = np.pi * np.sqrt(_det) * CHI2_68_2DF
            _cov_inv = np.linalg.inv(_cov)
            _ellipse_params[_pid] = (_mean, _cov_inv, _area)

        # Merge Defensive Area
        _area_series = pd.Series(
            {pid: params[2] for pid, params in _ellipse_params.items()},
            name='Defensive Area'
        )
        base_df = base_df.merge(_area_series, left_index=True, right_index=True, how='left')

        # -- Build xT grid (used for both Expected xT and opposition xT calculations) --
        _xt_data = [[0.01,0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.03,0.03,0.04,0.04],[0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.03,0.04,0.05,0.05],[0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.03,0.05,0.06,0.06],[0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.04,0.11,0.26,0.26],[0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.04,0.11,0.26,0.26],[0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.03,0.05,0.06,0.06],[0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.03,0.04,0.05,0.05],[0.01,0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.03,0.03,0.04,0.04]]
        _xt_grid = np.array(_xt_data)
        _xt_rows, _xt_cols = _xt_grid.shape

        # Expected xT across the full defensive ellipse (opposition perspective)
        # Sample points on a 2m grid inside each player's 68% confidence ellipse,
        # convert to opposition Wyscout coords, look up xT grid values, and average.
        # This captures the threat level across the entire defensive zone, not just the center.
        _expected_xt = {}
        _sample_step = 2.0  # meters
        for _pid, (_mean, _cov_inv, _area) in _ellipse_params.items():
            _cov = np.linalg.inv(_cov_inv)
            _max_radius = np.sqrt(np.linalg.eigvalsh(_cov).max() * CHI2_68_2DF)
            _xs = np.arange(_mean[0] - _max_radius, _mean[0] + _max_radius + _sample_step, _sample_step)
            _ys = np.arange(_mean[1] - _max_radius, _mean[1] + _max_radius + _sample_step, _sample_step)
            _gx, _gy = np.meshgrid(_xs, _ys)
            _pts = np.column_stack([_gx.ravel(), _gy.ravel()])
            _d = _pts - _mean
            _inside = np.sum(_d @ _cov_inv * _d, axis=1) < CHI2_68_2DF
            _pts_in = _pts[_inside]
            if len(_pts_in) == 0:
                _expected_xt[_pid] = 0.0
                continue
            _opp_x = 100 - (_pts_in[:, 0] / 1.05)
            _opp_y = 100 - (_pts_in[:, 1] / 0.68)
            _r = np.clip((_opp_y / 100 * _xt_rows).astype(int), 0, _xt_rows - 1)
            _c = np.clip((_opp_x / 100 * _xt_cols).astype(int), 0, _xt_cols - 1)
            _expected_xt[_pid] = _xt_grid[_r, _c].mean()
        _expected_xt_series = pd.Series(_expected_xt, name='Expected xT at Center')
        base_df = base_df.merge(_expected_xt_series, left_index=True, right_index=True, how='left')

        # -- Build opposition xT dataset (open-play passes/touches/accelerations) --

        _opp_moves = events_df[
            events_df['type.primary'].isin(['pass', 'touch', 'acceleration']) & _is_open_play
        ].copy()
        _opp_pass_mask = (_opp_moves['type.primary'] == 'pass') & (_opp_moves.get('pass.accurate') == True)
        _opp_other = _opp_moves['type.primary'].isin(['touch', 'acceleration'])
        _opp_moves_successful = _opp_moves[_opp_pass_mask | _opp_other].copy()

        _opp_moves_successful['start_x'] = _opp_moves_successful['location.x']
        _opp_moves_successful['start_y'] = _opp_moves_successful['location.y']
        _opp_moves_successful['end_x'] = np.where(
            _opp_moves_successful['type.primary'] == 'pass',
            _opp_moves_successful.get('pass.endLocation.x'),
            _opp_moves_successful.get('carry.endLocation.x')
        )
        _opp_moves_successful['end_y'] = np.where(
            _opp_moves_successful['type.primary'] == 'pass',
            _opp_moves_successful.get('pass.endLocation.y'),
            _opp_moves_successful.get('carry.endLocation.y')
        )
        _opp_moves_successful = _opp_moves_successful.dropna(subset=['end_x', 'end_y'])

        # Compute xT for each action
        _opp_moves_successful['s_col'] = np.clip((_opp_moves_successful['start_x'] / 100 * _xt_cols).astype(float).fillna(0).astype(int), 0, _xt_cols - 1)
        _opp_moves_successful['s_row'] = np.clip((_opp_moves_successful['start_y'] / 100 * _xt_rows).astype(float).fillna(0).astype(int), 0, _xt_rows - 1)
        _opp_moves_successful['e_col'] = np.clip((_opp_moves_successful['end_x'] / 100 * _xt_cols).astype(float).fillna(0).astype(int), 0, _xt_cols - 1)
        _opp_moves_successful['e_row'] = np.clip((_opp_moves_successful['end_y'] / 100 * _xt_rows).astype(float).fillna(0).astype(int), 0, _xt_rows - 1)
        _opp_moves_successful['xT_val'] = _xt_grid[_opp_moves_successful['e_row'].values, _opp_moves_successful['e_col'].values] - _xt_grid[_opp_moves_successful['s_row'].values, _opp_moves_successful['s_col'].values]

        # Convert locations to meters — FLIP opposition coords to defending team's frame.
        # Wyscout uses team-relative coords (each team attacks toward x=100).
        # The same physical spot is x for defender vs (100-x) for opponent.
        _opp_moves_successful['start_x_m'] = (100 - _opp_moves_successful['start_x']) * 1.05
        _opp_moves_successful['start_y_m'] = (100 - _opp_moves_successful['start_y']) * 0.68
        _opp_moves_successful['end_x_m'] = (100 - _opp_moves_successful['end_x']) * 1.05
        _opp_moves_successful['end_y_m'] = (100 - _opp_moves_successful['end_y']) * 0.68

        # Also build ALL opposition passes (including inaccurate) for pass success %
        _opp_all_passes = events_df[
            (events_df['type.primary'] == 'pass') & _is_open_play
        ][['matchId', 'team.name', 'pass.accurate', 'pass.endLocation.x', 'pass.endLocation.y']].copy()
        _opp_all_passes = _opp_all_passes.dropna(subset=['pass.endLocation.x', 'pass.endLocation.y'])
        # Flip opposition pass end coords to defending team's frame
        _opp_all_passes['end_x_m'] = (100 - _opp_all_passes['pass.endLocation.x']) * 1.05
        _opp_all_passes['end_y_m'] = (100 - _opp_all_passes['pass.endLocation.y']) * 0.68

        # -- Match-based opposition matching (fast: iterate by match, not by player) --
        # Build player -> team mapping
        _player_teams = events_df.groupby('player.id')['team.name'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else None
        ).to_dict()

        # Build match -> set of player_ids with ellipses, keyed by team
        _match_players = {}  # {matchId: {team_name: [player_ids]}}
        for _pid in _ellipse_params:
            _p_team = _player_teams.get(_pid)
            if not _p_team:
                continue
            for _mid in _open_def[_open_def['player.id'] == _pid]['matchId'].unique():
                if _mid not in _match_players:
                    _match_players[_mid] = {}
                _match_players[_mid].setdefault(_p_team, []).append(_pid)

        # Pre-group by matchId using numpy arrays for speed
        _moves_grp = _opp_moves_successful.groupby('matchId')
        _passes_grp = _opp_all_passes.groupby('matchId')

        # Accumulators (defaultdict for fast summing)
        from collections import defaultdict
        _opp_xt_into = defaultdict(float)
        _opp_xt_from = defaultdict(float)
        _opp_pass_into_total = defaultdict(int)
        _opp_pass_into_succ = defaultdict(int)

        for _mid, _team_players in _match_players.items():
            # Get teams in this match
            _teams = list(_team_players.keys())

            # Process xT-carrying moves for this match
            try:
                _m_moves = _moves_grp.get_group(_mid)
            except KeyError:
                _m_moves = None

            # Process all passes for this match
            try:
                _m_passes = _passes_grp.get_group(_mid)
            except KeyError:
                _m_passes = None

            for _def_team in _teams:
                # Opposition events = events from teams != _def_team
                if _m_moves is not None:
                    _opp_m = _m_moves[_m_moves['team.name'] != _def_team]
                    if not _opp_m.empty:
                        _end_xy = _opp_m[['end_x_m', 'end_y_m']].values
                        _start_xy = _opp_m[['start_x_m', 'start_y_m']].values
                        _xt_vals = _opp_m['xT_val'].values

                        for _pid in _team_players[_def_team]:
                            _mean, _cov_inv, _ = _ellipse_params[_pid]
                            # Into: end location
                            _d = _end_xy - _mean
                            _m_sq = np.sum(_d @ _cov_inv * _d, axis=1)
                            _in_e = _m_sq < CHI2_68_2DF
                            _opp_xt_into[_pid] += _xt_vals[_in_e].sum()
                            # From: start location
                            _d2 = _start_xy - _mean
                            _m_sq2 = np.sum(_d2 @ _cov_inv * _d2, axis=1)
                            _in_s = _m_sq2 < CHI2_68_2DF
                            _opp_xt_from[_pid] += _xt_vals[_in_s].sum()

                # Pass success % into area
                if _m_passes is not None:
                    _opp_p = _m_passes[_m_passes['team.name'] != _def_team]
                    if not _opp_p.empty:
                        _p_end = _opp_p[['end_x_m', 'end_y_m']].values
                        _p_acc = _opp_p['pass.accurate'].values

                        for _pid in _team_players[_def_team]:
                            _mean, _cov_inv, _ = _ellipse_params[_pid]
                            _d3 = _p_end - _mean
                            _m_sq3 = np.sum(_d3 @ _cov_inv * _d3, axis=1)
                            _in_p = _m_sq3 < CHI2_68_2DF
                            _opp_pass_into_total[_pid] += int(_in_p.sum())
                            _opp_pass_into_succ[_pid] += int(_p_acc[_in_p].sum())

        # Merge opposition metrics
        for _name, _d in [('Opp xT into Def Area', _opp_xt_into), ('Opp xT from Def Area', _opp_xt_from)]:
            _s = pd.Series(_d, name=_name)
            base_df = base_df.merge(_s, left_index=True, right_index=True, how='left')
        # Store pass numerator/denominator for % calc in Step 4
        base_df = base_df.merge(pd.Series(_opp_pass_into_total, name='_opp_pass_into_total'), left_index=True, right_index=True, how='left')
        base_df = base_df.merge(pd.Series(_opp_pass_into_succ, name='_opp_pass_into_succ'), left_index=True, right_index=True, how='left')

        print(f"  Defensive Area computed for {len(_ellipse_params)} players")
    except Exception as e:
        print(f"  ERROR computing Defensive Area metrics: {e}")
        import traceback; traceback.print_exc()
        base_df['Defensive Area'] = 0.0
        base_df['Opp xT into Def Area'] = 0.0
        base_df['Opp xT from Def Area'] = 0.0
        base_df['Expected xT at Center'] = 0.0
        base_df['_opp_pass_into_total'] = 0.0
        base_df['_opp_pass_into_succ'] = 0.0

    # --- Step 2: Calculate xG, xA, xT, and special passing ---
    print("Step 2: Calculating xG, xA, xT...")
    
    # -- xG --
    xg_series = events_df.groupby('player.id')['shot.xg'].sum()
    xg_series.name = 'xG'
    base_df = base_df.merge(xg_series, left_index=True, right_index=True, how='left')

    # -- npxG, xAOP, xASP --
    shots_df = events_df[(events_df['shot.xg'].notna()) & (events_df['type.primary'] != 'penalty')].copy()
    npxg_totals = shots_df.groupby('player.id')['shot.xg'].sum().reset_index().rename(columns={'shot.xg': 'npxG'})
    events_df['shot_event_id'] = np.where(events_df['shot.xg'].notna(), events_df['id'], np.nan)
    events_df['next_shot_id'] = events_df.groupby('matchId')['shot_event_id'].bfill()
    shot_xg_map = events_df[events_df['shot.xg'].notna()].set_index('id')['shot.xg'].to_dict()
    assists_df = events_df[check_secondary_list('shot_assist')].copy()
    assists_df['xA'] = assists_df['next_shot_id'].map(shot_xg_map)
    set_piece_types = ['corner', 'free_kick', 'throw_in', 'goal_kick']
    assists_df['assist_type'] = np.where(assists_df['type.primary'].isin(set_piece_types), 'xASP', 'xAOP')
    xa_split_totals = assists_df.groupby(['player.id', 'assist_type'])['xA'].sum()
    xa_final_df = xa_split_totals.unstack(fill_value=0).reset_index()
    final_stats_df = pd.merge(npxg_totals, xa_final_df, on='player.id', how='outer')
    base_df = base_df.merge(final_stats_df.set_index('player.id'), left_index=True, right_index=True, how='left')

    # -- Deep Completions and Progressive Passes --
    passes_df = events_df[(events_df['type.primary'] == 'pass') & (events_df.get('pass.accurate') == True)].dropna(subset=['location.x', 'pass.endLocation.x', 'player.id']).copy()
    passes_df['end_x_m'] = passes_df['pass.endLocation.x'] * 1.05
    passes_df['end_y_m'] = passes_df['pass.endLocation.y'] * 0.68
    passes_df['dist_to_goal_center'] = np.sqrt((passes_df['end_x_m'] - 105)**2 + (passes_df['end_y_m'] - 34)**2)
    passes_df['is_cross'] = passes_df.get('type.secondary', pd.Series(dtype='object')).apply(lambda x: isinstance(x, (list, np.ndarray)) and 'cross' in x)
    passes_df['is_deep_completion'] = (passes_df['dist_to_goal_center'] <= 20) & (passes_df['is_cross'] == False)
    deep_completions = passes_df.groupby('player.id')['is_deep_completion'].sum().reset_index().rename(columns={'is_deep_completion': 'Deep Completions'})
    start_x = passes_df['location.x']; end_x = passes_df['pass.endLocation.x']
    cond1 = (start_x < 50) & (end_x < 50) & (end_x - start_x >= 30)
    cond2 = (start_x < 50) & (end_x >= 50) & (end_x - start_x >= 15)
    cond3 = (start_x >= 50) & (end_x >= 50) & (end_x - start_x >= 10)
    passes_df['is_progressive_pass'] = cond1 | cond2 | cond3
    progressive_passes = passes_df.groupby('player.id')['is_progressive_pass'].sum().reset_index().rename(columns={'is_progressive_pass': 'Progressive Passes'})
    new_metrics_df = pd.merge(deep_completions, progressive_passes, on='player.id', how='outer')
    base_df = base_df.merge(new_metrics_df.set_index('player.id'), left_index=True, right_index=True, how='left')
    
    # -- xT --
    xt_data_from_image = [[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04], [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.04, 0.05, 0.05], [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.05, 0.06, 0.06], [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.04, 0.11, 0.26, 0.26], [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.04, 0.11, 0.26, 0.26], [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.05, 0.06, 0.06], [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.04, 0.05, 0.05], [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04]]
    xt_grid = np.array(xt_data_from_image); rows, cols = xt_grid.shape
    _sp_types = ['corner', 'free_kick', 'throw_in']
    move_df = events_df[events_df['type.primary'].isin(['pass', 'touch', 'acceleration'] + _sp_types)].copy()
    successful_pass = (move_df['type.primary'] == 'pass') & (move_df.get('pass.accurate') == True)
    other_successful_moves = move_df['type.primary'].isin(['touch', 'acceleration'] + _sp_types)
    move_df = move_df[successful_pass | other_successful_moves]
    move_df['start_x'] = move_df['location.x']; move_df['start_y'] = move_df['location.y']
    # Set pieces (corners, free kicks, throw-ins) use pass.endLocation like passes
    _is_pass_like = move_df['type.primary'].isin(['pass'] + _sp_types)
    move_df['end_x'] = np.where(_is_pass_like, move_df.get('pass.endLocation.x'), move_df.get('carry.endLocation.x'))
    move_df['end_y'] = np.where(_is_pass_like, move_df.get('pass.endLocation.y'), move_df.get('carry.endLocation.y'))
    move_df = move_df.dropna(subset=['end_x', 'end_y', 'player.id'])

    # Vectorized xT zone calculation (much faster than apply)
    move_df['start_col'] = np.clip((move_df['start_x'] / 100 * cols).astype(float).fillna(0).astype(int), 0, cols - 1)
    move_df['start_row'] = np.clip((move_df['start_y'] / 100 * rows).astype(float).fillna(0).astype(int), 0, rows - 1)
    move_df['end_col'] = np.clip((move_df['end_x'] / 100 * cols).astype(float).fillna(0).astype(int), 0, cols - 1)
    move_df['end_row'] = np.clip((move_df['end_y'] / 100 * rows).astype(float).fillna(0).astype(int), 0, rows - 1)

    # Vectorized xT lookup using numpy advanced indexing
    move_df['xt_start'] = xt_grid[move_df['start_row'].values, move_df['start_col'].values]
    move_df['xt_end'] = xt_grid[move_df['end_row'].values, move_df['end_col'].values]
    move_df['xT'] = move_df['xt_end'] - move_df['xt_start']
    successful_threat = move_df[move_df['xT'] > 0]
    player_xt = successful_threat.groupby('player.id')['xT'].sum().reset_index()
    base_df = base_df.merge(player_xt.set_index('player.id'), left_index=True, right_index=True, how='left')

    # Split xT into open play (xTOP) and set piece (xTSP)
    # xTSP = xT from throw-ins, corners, free kicks only. Everything else = xTOP.
    successful_threat = successful_threat.copy()
    successful_threat['xt_type'] = np.where(
        successful_threat['type.primary'].isin(_sp_types), 'xTSP', 'xTOP'
    )
    xt_split = successful_threat.groupby(['player.id', 'xt_type'])['xT'].sum()
    xt_split_df = xt_split.unstack(fill_value=0).reset_index()
    base_df = base_df.merge(xt_split_df.set_index('player.id'), left_index=True, right_index=True, how='left')

    # --- Step 3: Calculate Goalkeeper Stats ---
    print("Step 3: Calculating Goalkeeper stats...")
    gk_ids = events_df[events_df.get('player.position') == 'GK']['player.id'].dropna().unique().astype(int)
    gk_events_df = events_df[events_df['player.id'].isin(gk_ids)].copy()
    shots_faced_df = events_df[(events_df.get('type.primary') == 'shot') & (events_df.get('shot.onTarget') == True) & (events_df.get('shot.goalkeeper.id').notna())].copy()
    shots_faced_df['shot.goalkeeper.id'] = shots_faced_df['shot.goalkeeper.id'].astype(int)
    gk_shot_stopping_stats = shots_faced_df.groupby('shot.goalkeeper.id').agg(shotsOnTargetAgainst=('shot.isGoal', 'count'), goalsConceded=('shot.isGoal', 'sum'), psxG_faced=('shot.postShotXg', 'sum')).reset_index().rename(columns={'shot.goalkeeper.id': 'player.id'})
    if not gk_shot_stopping_stats.empty:
        # shot.isGoal is object dtype in the events parquet, so its
        # groupby-sum can come out object (pandas-build dependent) — which
        # made the per-90 loop's is_numeric_dtype gate silently skip
        # goalsConceded so it stayed a season total. Coerce here so
        # normalization is deterministic and goalsPrevented/savePercentage
        # inherit float dtype.
        gk_shot_stopping_stats['goalsConceded'] = pd.to_numeric(
            gk_shot_stopping_stats['goalsConceded'], errors='coerce').fillna(0).astype('float64')
        gk_shot_stopping_stats['goalsPrevented'] = gk_shot_stopping_stats['psxG_faced'] - gk_shot_stopping_stats['goalsConceded']
        gk_shot_stopping_stats['goalsPreventedPerSOT'] = (gk_shot_stopping_stats['goalsPrevented'] / gk_shot_stopping_stats['shotsOnTargetAgainst']).fillna(0)
        gk_shot_stopping_stats['savePercentage'] = ((gk_shot_stopping_stats['shotsOnTargetAgainst'] - gk_shot_stopping_stats['goalsConceded']) / gk_shot_stopping_stats['shotsOnTargetAgainst'] * 100).fillna(0)
    else:
        gk_shot_stopping_stats = gk_shot_stopping_stats.reindex(columns=['player.id', 'shotsOnTargetAgainst', 'goalsConceded', 'psxG_faced', 'goalsPrevented', 'goalsPreventedPerSOT', 'savePercentage']).fillna(0)
    exits = gk_events_df[gk_events_df['type.primary'] == 'goalkeeper_exit'].groupby('player.id').size().reset_index(name='exits')
    recoveries_gk = gk_events_df[check_secondary_list('recovery')].groupby('player.id').size().reset_index(name='recoveries_gk')
    gk_passes = gk_events_df[gk_events_df['type.primary'] == 'pass']
    passes_total_gk = gk_passes.groupby('player.id').size().reset_index(name='passes_gk')
    passes_succ_gk = gk_passes[gk_passes['pass.accurate'] == True].groupby('player.id').size().reset_index(name='passesSuccessful_gk')
    long_passes_total_gk = gk_passes[check_secondary_list('long_pass')].groupby('player.id').size().reset_index(name='longPasses_gk')
    long_passes_succ_gk = gk_passes[check_secondary_list('long_pass') & (gk_passes['pass.accurate'] == True)].groupby('player.id').size().reset_index(name='longPassesSuccessful_gk')
    gk_report_df = pd.DataFrame({'player.id': gk_ids}); gk_report_df = pd.merge(gk_report_df, gk_shot_stopping_stats, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, exits, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, recoveries_gk, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, passes_total_gk, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, passes_succ_gk, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, long_passes_total_gk, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, long_passes_succ_gk, on='player.id', how='left')
    base_df = base_df.merge(gk_report_df.set_index('player.id'), left_index=True, right_index=True, how='left')

    # --- Step 4: Finalize, Calculate Percentages, and Normalize ---
    print("Step 4: Finalizing and normalizing...")
    base_df = base_df.fillna(0)
    
    # Calculate % stats from the totals
    def safe_divide_perc(n, d): return (base_df[n] / base_df[d] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    base_df['xG per Shot'] = (base_df['xG'] / base_df['Shots']).replace([np.inf, -np.inf], 0).fillna(0)
    base_df['Passes successful %'] = safe_divide_perc('Passes successful', 'Passes')
    base_df['Long passes successful %'] = safe_divide_perc('Long passes successful', 'Long passes')
    base_df['Crosses successful %'] = safe_divide_perc('Crosses successful', 'Crosses')
    base_df['Dribbles successful %'] = safe_divide_perc('Dribbles successful', 'Dribbles')
    base_df['Duels successful %'] = safe_divide_perc('Duels successful', 'Duels')
    base_df['Dribbled past %'] = safe_divide_perc('Dribbled past', 'Dribbles faced')
    base_df['Defensive duels vs take-on successful %'] = safe_divide_perc(
        'Defensive duels vs take-on successful', 'Defensive duels vs take-on')
    base_df['Defensive duels vs offensive duel successful %'] = safe_divide_perc(
        'Defensive duels vs offensive duel successful', 'Defensive duels vs offensive duel')
    # Dribbled past % (proj) — empirical-Bayes projection of the TRUE rate
    # (Lucas 2026-07): raw % stabilizes at k~54 dribbles faced (variance
    # decomposition on 57k contests; split-half + YoY agree) but a season
    # supplies a median of 8, so single-season raw % is mostly noise. Shrink
    # toward the scope's league mean with k=54: thin samples project to
    # average; only real volume moves you away. 0 faced -> exactly the mean.
    _k_dp = 54.0
    _dp_tot, _df_tot = base_df['Dribbled past'].sum(), base_df['Dribbles faced'].sum()
    _dp_mean = (_dp_tot / _df_tot) if _df_tot > 0 else 0.5
    base_df['Dribbled past % (proj)'] = ((base_df['Dribbled past'] + _k_dp * _dp_mean)
                                          / (base_df['Dribbles faced'] + _k_dp) * 100)
    base_df['Aerial duels successful %'] = safe_divide_perc('Aerial duels successful', 'Aerial duels')
    base_df['Offensive duels successful %'] = safe_divide_perc('Offensive duels successful', 'Offensive duels')
    base_df['Defensive duels successful %'] = safe_divide_perc('Defensive duels successful', 'Defensive duels')
    base_df['Sliding tackles successful %'] = safe_divide_perc('Sliding tackles successful', 'Sliding tackles')
    base_df['Aerial duels def box successful %'] = safe_divide_perc('Aerial duels def box successful', 'Aerial duels def box')
    base_df['Aerial duels att box successful %'] = safe_divide_perc('Aerial duels att box successful', 'Aerial duels att box')
    # Defensive Area: opposition pass success % into area
    base_df['Opp Pass Success % into Def Area'] = safe_divide_perc('_opp_pass_into_succ', '_opp_pass_into_total')
    base_df = base_df.drop(columns=['_opp_pass_into_total', '_opp_pass_into_succ'], errors='ignore')
    successful_attacking_actions = base_df['Shots on target'] + base_df['Crosses successful'] + base_df['Dribbles successful']
    base_df['Loss index'] = (base_df['Losses'] / successful_attacking_actions).replace([np.inf, -np.inf], 0).fillna(0)
    
    # GK % stats
    base_df['Passes successful %_gk'] = safe_divide_perc('passesSuccessful_gk', 'passes_gk')
    base_df['Long passes successful %_gk'] = safe_divide_perc('longPassesSuccessful_gk', 'longPasses_gk')
    # Rename for consistency
    base_df = base_df.rename(columns={
        'Passes successful %_gk': 'GK Passes successful %',
        'Long passes successful %_gk': 'GK Long passes successful %'
    })

    # Get a list of ALL calculated metrics
    all_calculated_metrics = list(base_df.columns)
    
    # Normalize to Per 90
    total_minutes = base_df['totalMinutes']
    minutes_gt_0 = total_minutes > 0
    
    # Define cols that should NOT be normalized
    rate_cols = [col for col in base_df.columns if '%' in col or 'per' in col.lower() or 'index' in col.lower() or 'Percentage' in col or 'Avg' in col]
    info_cols = ['playerName', 'teamName', 'totalMinutes', 'primaryPosition', 'secondaryPosition', 'tertiaryPosition', 'player.id', 'player.id_x', 'player.id_y', 'Defensive Area', 'Expected xT at Center']
    dont_normalize = rate_cols + info_cols

    # Track what actually got divided so the backstop below can normalize
    # skipped columns EXACTLY once — never by re-inspecting value shapes.
    _normalized_cols = set()
    for col in all_calculated_metrics:
        if col not in dont_normalize and pd.api.types.is_numeric_dtype(base_df[col]):
            base_df[col] = np.where(
                minutes_gt_0,
                (base_df[col].astype(float) / total_minutes) * 90,
                0
            )
            _normalized_cols.add(col)

    # ── Exactly-once backstop for the GK counting metrics ────────────────
    # goalsConceded historically reached the loop above as OBJECT dtype (the
    # groupby-sum of object-typed shot.isGoal), so the is_numeric_dtype gate
    # silently skipped it and it stayed a season total. The old backstop
    # (further down, now removed) rescaled whenever the MAX value "looked
    # like a total" — but data-gap keepers with broken tiny minutes (a full
    # season shown as ~90') have legit per-90 values above those thresholds,
    # so it also re-divided ALREADY-normalized columns a second time: the
    # 10-30x-low shotsOnTargetAgainst / psxG_faced / exits mis-scale that
    # flipped with unrelated MINUTES_OVERRIDE edits (see f683ed8). With the
    # dtype coerced at the source and this membership check, each column is
    # divided exactly once and no value-shape heuristic can misfire.
    for _col in ['goalsConceded', 'shotsOnTargetAgainst', 'psxG_faced',
                 'exits', 'goalsPrevented']:
        if _col in base_df.columns and _col not in _normalized_cols:
            print(f"  ⚠️ {_col} skipped the per-90 loop "
                  f"(dtype={base_df[_col].dtype}); normalizing it once now")
            _v = pd.to_numeric(base_df[_col], errors='coerce').fillna(0)
            base_df[_col] = np.where(minutes_gt_0, _v / total_minutes * 90, 0)

    # -- OE (Over-Expectation) metrics: normalize opposition metrics by Expected xT at Center --
    # This accounts for positional expectation: CBs near goal naturally face higher opposition xT.
    # OE = raw metric / Expected xT at Center — higher OE = more opposition threat than expected.
    _has_expected = (base_df['Expected xT at Center'] > 0)
    _has_area = (base_df['Defensive Area'] > 0)

    print(f"  DEBUG OE metric inputs:")
    print(f"    Opp xT into Def Area (after per90): non-zero={( base_df['Opp xT into Def Area'] != 0).sum()}")
    print(f"    Opp xT from Def Area (after per90): non-zero={( base_df['Opp xT from Def Area'] != 0).sum()}")
    print(f"    Expected xT at Center: non-zero={_has_expected.sum()}, min={base_df.loc[_has_expected, 'Expected xT at Center'].min():.4f}, max={base_df['Expected xT at Center'].max():.4f}")

    # Opp xT into Def Area OE = (Opp xT into Def Area per 90) / Expected xT at Center
    base_df['Opp xT into Def Area OE'] = np.where(
        _has_expected,
        base_df['Opp xT into Def Area'] / base_df['Expected xT at Center'],
        0
    )

    # Opp xT from Def Area OE = (Opp xT from Def Area per 90) / Expected xT at Center
    base_df['Opp xT from Def Area OE'] = np.where(
        _has_expected,
        base_df['Opp xT from Def Area'] / base_df['Expected xT at Center'],
        0
    )

    # Territorial Dominance OE = Opp xT into Def Area per 90 / (Defensive Area × Expected xT at Center) × 10000
    base_df['Territorial Dominance OE'] = np.where(
        _has_area & _has_expected,
        (base_df['Opp xT into Def Area'] / (base_df['Defensive Area'] * base_df['Expected xT at Center'])) * 10000,
        0
    )

    # Keep the original Territorial Dominance as well (raw, non-OE)
    base_df['Territorial Dominance'] = np.where(
        _has_area,
        (base_df['Opp xT into Def Area'] / base_df['Defensive Area']) * 100000,
        0
    )

    print(f"    Opp xT into Def Area OE: non-zero={(base_df['Opp xT into Def Area OE'] != 0).sum()}, min={base_df['Opp xT into Def Area OE'].min():.4f}, max={base_df['Opp xT into Def Area OE'].max():.4f}")
    print(f"    Opp xT from Def Area OE: non-zero={(base_df['Opp xT from Def Area OE'] != 0).sum()}, min={base_df['Opp xT from Def Area OE'].min():.4f}, max={base_df['Opp xT from Def Area OE'].max():.4f}")
    print(f"    Territorial Dominance OE: non-zero={(base_df['Territorial Dominance OE'] != 0).sum()}, min={base_df['Territorial Dominance OE'].min():.6f}, max={base_df['Territorial Dominance OE'].max():.6f}")

    # Clean up and return
    base_df = base_df.reset_index() # 'playerId' is now a column
    # Drop all the junk 'player.id' columns
    cols_to_drop = [col for col in base_df.columns if 'player.id' in str(col)]
    if 'playerId' in base_df.columns and 'player.id' in cols_to_drop:
        cols_to_drop.remove('player.id') # Keep the real one

    base_df = base_df.drop(columns=cols_to_drop, errors='ignore')

    # (The old "defensive per-90" max-threshold safety nets for goalsConceded /
    # shotsOnTargetAgainst / exits / psxG_faced lived here. They re-divided
    # already-normalized columns whenever an outlier keeper's per-90 crossed a
    # magnitude threshold — replaced by the exactly-once backstop right after
    # the main normalization loop, plus numeric coercion of goalsConceded at
    # its groupby-sum. See that block's comment for the full history.)

    print("--- FINISHED: New All-Player-Stats Calculation ---")
    result = base_df.fillna(0).reset_index()

    # Stamp competitionId per player from the source events so that (a) the
    # league-aware All-Seasons cache key in calculate_player_percentiles_and_scores
    # resolves to the league rather than a shared 'all' file, and (b) downstream
    # CVI tier logic can read it. Events carry competitionId; map player -> comp.
    if 'competitionId' in events_df.columns and 'playerId' in result.columns:
        _player_comp = (events_df.dropna(subset=['player.id'])
                                 .groupby('player.id')['competitionId']
                                 .agg(lambda s: pd.to_numeric(s, errors='coerce').dropna().iloc[0]
                                      if pd.to_numeric(s, errors='coerce').dropna().size else np.nan))
        result['competitionId'] = pd.to_numeric(
            result['playerId'].map(_player_comp), errors='coerce')

    # Save to disk cache for fast loading on restart
    os.makedirs(STATS_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(STATS_CACHE_DIR, f'player_stats_{STATS_CACHE_VERSION}_{_scope_key}.parquet')
    try:
        _parquet_safe(result).to_parquet(cache_path)
        print(f"  Cached player stats to {cache_path}")
    except Exception as e:
        print(f"  Warning: Could not cache player stats: {e}")

    return result

def calculate_career_player_stats(_current_events, _hist_events, _all_time_minutes):
    """
    Calculate career stats across all seasons by combining current and historical data.
    Returns per-90 normalized stats using all-time minutes.
    """
    if _hist_events is None or _all_time_minutes is None:
        return None

    print("--- STARTING: Career Stats Calculation ---")

    # Combine current and historical events
    combined_events = pd.concat([_current_events, _hist_events], ignore_index=True)

    # Remove duplicates based on event ID
    if 'id' in combined_events.columns:
        combined_events = combined_events.drop_duplicates(subset=['id'])

    print(f"Combined events: {len(combined_events)} total")
    print(f"All-time minutes: {len(_all_time_minutes)} players")

    # Use the existing stats calculation function with combined data
    career_stats = calculate_all_player_stats(combined_events, _all_time_minutes, season_id="career")

    print("--- FINISHED: Career Stats Calculation ---")
    return career_stats

@st.cache_data
def calculate_player_percentiles_and_scores(_player_data_df, _position_groups, _weights, _invert_metrics, min_minutes=500, season_id=None, cache_version=STATS_CACHE_VERSION, cache_scope=None):
    # cache_scope: league discriminator for the All-Seasons (season_id=None)
    # in-memory cache collision — see calculate_all_player_stats.
    """Calculates percentiles and scores for all players based on position.
    Players below min_minutes are kept but ranked against the min_minutes+ population
    (each low-minute player is temporarily added to the sample for their own percentile).
    season_id is used as a cache key so Streamlit recomputes when the season changes."""
    # Disk cache: load pre-computed results if available
    _REQUIRED_PCT_COLS = {'Throw-ins', 'Avg max throw-in distance', 'Throw-ins into box', 'Avg max throw-in into box distance', 'Avg max throw-in into box aerial distance', 'Defensive Area', 'Opp xT into Def Area', 'Opp Pass Success % into Def Area', 'Opp xT from Def Area', 'Territorial Dominance', 'Opp xT into Def Area OE', 'Opp xT from Def Area OE', 'Territorial Dominance OE', 'xTOP', 'xTSP'}
    _scope_key = _stats_scope_key(season_id, _player_data_df)
    cache_path = os.path.join(STATS_CACHE_DIR, f'player_percentiles_{STATS_CACHE_VERSION}_{_scope_key}.parquet')
    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        if _REQUIRED_PCT_COLS.issubset(cached.columns):
            print(f"Loading cached player percentiles for scope {_scope_key}")
            if cached.index.name == 'playerId':
                cached = cached.reset_index()
            return cached
        else:
            print(f"Percentiles cache outdated (missing columns), recomputing for scope {_scope_key}")
            os.remove(cache_path)

    print("Calculating player percentiles and scores...")
    data = _player_data_df.copy()

    data['totalMinutes'] = pd.to_numeric(data['totalMinutes'], errors='coerce')
    # Only include qualifying players (>= min_minutes) in percentile calculations
    _qualifying_mask = data['totalMinutes'] >= min_minutes
    if _qualifying_mask.sum() == 0:
        print(f"Warning: No players found with >= {min_minutes} minutes.")
        return pd.DataFrame()

    # Calculate percentiles — ONLY for qualifying players (500+ min)
    # Sub-threshold players are excluded from rankings entirely
    for position, group in _position_groups.items():
        metrics = list(_weights[position].keys())
        position_data_mask = data['primaryPosition'].isin(group)
        position_data_indices = data[position_data_mask].index

        if position_data_indices.empty: continue

        # Only qualifying players get percentiles
        _pos_qualifying = data.loc[position_data_indices][_qualifying_mask.reindex(position_data_indices, fill_value=False)]

        for metric in metrics:
            if metric in data.columns:
                data[metric] = pd.to_numeric(data[metric], errors='coerce').fillna(0)
                # Percentiles for qualifying players — ranked among themselves
                if not _pos_qualifying.empty:
                    percentiles_q = data.loc[_pos_qualifying.index, metric].rank(pct=True)
                    if metric in _invert_metrics:
                        percentiles_q = 1 - percentiles_q.fillna(0.5)
                    data.loc[_pos_qualifying.index, metric + '_percentile'] = percentiles_q
            
    # Calculate Scores
    for position, group in _position_groups.items():
        metrics = list(_weights[position].keys())
        position_data_mask = data['primaryPosition'].isin(group)
        position_data_indices = data[position_data_mask].index
        if position_data_indices.empty: continue

        total_score = pd.Series(0.0, index=position_data_indices, dtype='float64')
        for metric in metrics:
            percentile_col = metric + '_percentile'
            if percentile_col in data.columns:
                weight = _weights[position].get(metric, 0)
                total_score = total_score.add(data.loc[position_data_indices, percentile_col].fillna(0) * weight, fill_value=0)
        
        data.loc[position_data_indices, position + '_TotalScore'] = total_score
        
        min_score = total_score.min()
        max_score = total_score.max()
        if (max_score - min_score) != 0:
            data.loc[position_data_indices, position + '_Score'] = (total_score - min_score) / (max_score - min_score) * 100
        else:
            data.loc[position_data_indices, position + '_Score'] = 0.0

    print("✅ Player percentiles and scores calculated.")
    result = data.fillna(0)

    # Save to disk cache for fast loading on restart
    os.makedirs(STATS_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(STATS_CACHE_DIR, f'player_percentiles_{STATS_CACHE_VERSION}_{_scope_key}.parquet')
    try:
        _parquet_safe(result).to_parquet(cache_path)
        print(f"  Cached player percentiles to {cache_path}")
    except Exception as e:
        print(f"  Warning: Could not cache percentiles: {e}")

    return result


@st.cache_data(ttl=86400, show_spinner=False)  # 24h (was 1h — hourly expiry
# forced cold recomputes from the events frame on a non-sleeping Space)
def load_and_score_player_stats(_events_df, _minutes_df, season_id, active_season_ids, comp_ids):
    """Run the full player-stats pipeline: base stats, GPA/DefR/engine value
    merges, then percentiles + template scores.

    Cached on (season_id, active_season_ids, comp_ids) — the frame args are
    underscore-prefixed so Streamlit keys on the scope, not the 125 MB
    events content. This is the hot path: switching PLAYER (same scope)
    is now a pure cache hit, and revisiting a league/season hits cache
    too. cache_data returns a fresh copy each call, so downstream mutation
    is safe. Returns (player_stats_df, player_stats_with_scores_df).
    """
    events_df, minutes_df = _events_df, _minutes_df
    # league discriminator so the All-Seasons (season_id=None) in-memory cache
    # doesn't collide across leagues (Camp All-Seasons was getting L3's result)
    _scope = tuple(comp_ids) if isinstance(comp_ids, (list, tuple, set)) else comp_ids
    player_stats_df = calculate_all_player_stats(events_df, minutes_df, season_id=season_id, cache_scope=_scope)
    player_stats_df = merge_gpa_values_into_stats(player_stats_df, active_season_ids, comp_ids)
    player_stats_df = merge_defr_values_into_stats(player_stats_df, active_season_ids, comp_ids)
    player_stats_df = merge_engine_values_into_stats(player_stats_df, active_season_ids, comp_ids)
    player_stats_with_scores_df = calculate_player_percentiles_and_scores(
        player_stats_df, POSITION_GROUPS, WEIGHTS, INVERT_METRICS, min_minutes=500, season_id=season_id, cache_scope=_scope
    )
    return player_stats_df, player_stats_with_scores_df


@st.cache_resource(show_spinner=False)
def _prewarm_scope_caches(_raw_events_df, _player_minutes_data, _matches_summary_df):
    """Warm every (league, season) scope's IN-MEMORY caches once per process.

    After each deploy the Space restarts with empty in-memory caches. The disk
    caches (player stats / percentiles / team-strength) survive, but the per-
    scope events filter (`get_filtered_events`, cache_resource, NO disk backing),
    the minutes aggregation, the engine merges and the `load_and_score_player_stats`
    result do NOT — so the first visitor to each league/season eats a 20-50s cold
    build on the request path. This spawns a daemon thread that runs the exact
    page warm-path for every scope up front, off the request path, so interactive
    league/season switches are instant from the first click.

    Decorated with cache_resource so the thread is spawned exactly once per
    process (singleton; the frame args are underscore-prefixed and unhashed).

    Safe off the main ScriptRunContext: `load_and_score_player_stats` and its
    callees populate global caches (the cache store is process-global, not
    session-scoped) and perform no on-screen UI writes here. Because comp_ids is
    always a single league, the (season_id, active_season_ids) pair warmed below
    is byte-identical to what every page requests (active == selected for one
    league), so these are direct cache-key hits, not approximations.
    """
    import threading, time as _time, logging as _logging

    def _worker():
        # The cached functions emit a benign "missing ScriptRunContext" warning
        # when run off the main thread (the cache still populates correctly — only
        # the no-op spinner trips the warning). Quiet just those loggers.
        for _nm in ("streamlit.runtime.scriptrunner_utils.script_run_context",
                    "streamlit.runtime.scriptrunner.script_run_context"):
            try:
                _logging.getLogger(_nm).setLevel(_logging.ERROR)
            except Exception:
                pass
        # Politeness: this thread shares the process (and the GIL) with the
        # script runner. On the shared-CPU Space an unthrottled warm loop
        # starves every interaction for minutes after a deploy — users see
        # section toggles stuck on "Running" (2026-07-14). Let the first
        # page load settle, then yield between scopes so clicks preempt.
        _time.sleep(8)
        _t0 = _time.time(); _n = 0
        # Warm ONLY the current-season scopes (the default landing pages).
        # The filtered-events cache is bounded at max_entries=3 for memory
        # (see _get_filtered_events_cached) — warming all 9 scopes would
        # hold ~21 GB of frame copies and just churn the LRU anyway. Other
        # scopes lazy-build in a few seconds on first visit (disk caches
        # cover the expensive layers).
        _WARM_SCOPES = [(_cid, _sid) for _cid, _cfg in COMPETITIONS.items()
                        for _sid in _cfg.get('seasons', {})
                        if _sid in (CURRENT_SEASON_ID, 191779)]
        for _cid, _sid in _WARM_SCOPES:
                _time.sleep(2.0)
                try:
                    _ev = get_filtered_events(_raw_events_df, _sid, [_cid])
                    _mins = get_season_player_minutes(_player_minutes_data, _sid, comp_ids=[_cid])
                    if _ev is None or len(_ev) == 0 or _mins is None or len(_mins) == 0:
                        continue
                    load_and_score_player_stats(_ev, _mins, _sid, _sid, [_cid])
                    if _sid is not None:
                        try:
                            calculate_team_strength(_ev, _matches_summary_df, season_id=_sid)
                        except Exception:
                            pass
                    _n += 1
                except Exception as _e:
                    logger.warning(f"[prewarm] scope (comp={_cid}, season={_sid}) failed: {_e}")
        logger.info(f"[prewarm] warmed {_n} scope(s) in {_time.time()-_t0:.0f}s")

    _t = threading.Thread(target=_worker, name="acp-prewarm", daemon=True)
    _t.start()
    logger.info("[prewarm] background cache warm started")
    return True


def auto_column_config(df):
    """NumberColumn formats: floats %.2f, ints %d. Display-only polish."""
    import pandas.api.types as ptypes
    cfg = {}
    for c in df.columns:
        if ptypes.is_float_dtype(df[c]):
            cfg[c] = st.column_config.NumberColumn(format="%.2f")
        elif ptypes.is_integer_dtype(df[c]):
            cfg[c] = st.column_config.NumberColumn(format="%d")
    return cfg


def _create_base_radar_chart(ax, player_data, metrics, position, eligible_groups, full_df_for_ranking=None, season_label=None, radar_mode='percentile', population_data=None):
    """Helper function to create the base radar chart.
    radar_mode: 'percentile' (default) or 'raw' (raw per-90 values, mean ± 2σ scale).
    population_data: DataFrame of the position group population (required for 'raw' mode).
    """

    num_metrics = len(metrics)
    angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
    angles += angles[:1]

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])

    category_colors = {'output': 'green', 'passing': 'orange', 'defensive': 'red', 'dribbling': 'purple', 'goalkeeping': 'cyan', 'off_ball_defending': '#7ECE2B'}

    if radar_mode == 'raw' and population_data is not None:
        # --- RAW VALUE MODE: scale = mean ± 2σ ---
        # Filter population to 500+ minutes
        _pop = population_data.copy()
        if 'totalMinutes' in _pop.columns:
            _pop = _pop[pd.to_numeric(_pop['totalMinutes'], errors='coerce').fillna(0) >= 500]

        # Compute mean and std for each metric from the population
        _means = {}; _stds = {}
        for metric in metrics:
            if metric in _pop.columns:
                vals = pd.to_numeric(_pop[metric], errors='coerce').dropna()
                _means[metric] = vals.mean() if len(vals) > 0 else 0
                _stds[metric] = vals.std() if len(vals) > 1 else 1
            else:
                _means[metric] = 0; _stds[metric] = 1
            if _stds[metric] == 0: _stds[metric] = 1  # prevent division by zero

        # Map player raw values to 0-100 scale where 0 = mean - 2σ, 100 = mean + 2σ
        # 50 = mean
        values = []
        for metric in metrics:
            raw_val = float(player_data[metric].values[0]) if metric in player_data.columns else 0.0
            # For inverted metrics, flip direction: higher raw = worse = lower on chart
            if metric in INVERT_METRICS:
                mapped = 50 - ((raw_val - _means[metric]) / _stds[metric]) * 25
            else:
                mapped = 50 + ((raw_val - _means[metric]) / _stds[metric]) * 25
            mapped = np.clip(mapped, 0, 100)
            values.append(mapped)
        values += values[:1]

        ax.plot(angles, values, linewidth=2, linestyle='solid', color='#0077b6', zorder=3)
        ax.fill(angles, values, '#0077b6', alpha=0.25, zorder=2)

        # Gridlines at 0, 25, 50, 75, 100 (= mean-2σ, mean-1σ, mean, mean+1σ, mean+2σ)
        ax.set_rlabel_position(0)
        plt.yticks([25, 50, 75, 100], ["", "", "", ""], color="grey", size=7, zorder=1)
        plt.ylim(0, 100)

        # Spoke labels: show raw values at each gridline level (skip innermost = mean-2σ)
        _grid_levels = [25, 50, 75, 100]   # -1σ, mean, +1σ, +2σ
        _sigma_offsets = [-1, 0, 1, 2]      # corresponding σ offsets
        for i, metric in enumerate(metrics):
            angle_rad = angles[i]
            mean = _means[metric]; std = _stds[metric]
            for lvl, sig in zip(_grid_levels, _sigma_offsets):
                if metric in INVERT_METRICS:
                    grid_raw = mean - sig * std  # inverted: higher position = lower raw
                else:
                    grid_raw = mean + sig * std
                grid_label = fmt_val(metric, grid_raw)
                ax.text(angle_rad, lvl + 3, grid_label, size=7.5, ha='center', va='bottom', color='black')

    else:
        # --- PERCENTILE MODE (original behavior) ---
        values = []
        for metric in metrics:
            col = metric + '_percentile'
            if col in player_data.columns:
                val = float(player_data[col].values[0])
                values.append(np.nan_to_num(val, nan=0.0) * 100)
            else:
                values.append(0.0)
                print(f"Warning: Missing percentile column {col} for radar.")
        values += values[:1]

        ax.plot(angles, values, linewidth=2, linestyle='solid', color='#0077b6', zorder=3)
        ax.fill(angles, values, '#0077b6', alpha=0.25, zorder=2)

        ax.set_rlabel_position(0)
        plt.yticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"], color="grey", size=7, zorder=1)
        plt.ylim(0, 100)

        # Plot raw values near the shape
        for i, metric in enumerate(metrics):
            angle_rad = angles[i]
            label = fmt_val(metric, player_data[metric].values[0])
            ax.text(angle_rad, 85, label, size=8, ha='center', va='center', color='blue')

    # Plot metric names (shared by both modes)
    for i, metric in enumerate(metrics):
        angle_rad = angles[i]
        if metric in OFF_BALL_DEFENDING_METRICS: color = category_colors['off_ball_defending']
        elif metric in OUTPUT_METRICS: color = category_colors['output']
        elif metric in PASSING_METRICS: color = category_colors['passing']
        elif metric in DEFENSIVE_METRICS or metric in DEFR_DISPLAY_METRICS: color = category_colors['defensive']
        elif metric in DRIBBLING_METRICS: color = category_colors['dribbling']
        elif metric in GOALKEEPING_METRICS: color = category_colors['goalkeeping']
        else: color = 'grey'
        ax.text(angle_rad, 115, metric, size=8, ha='center', va='center', rotation=0, color=color, fontweight='bold')

    ax.set_rlabel_position(0)
    if radar_mode != 'raw':
        plt.yticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"], color="grey", size=7)
    plt.ylim(0, 100)

    player_name = player_data['playerName'].values[0]
    player_position = player_data['primaryPosition'].values[0]
    player_minutes = player_data['totalMinutes'].values[0]
    player_team = player_data['teamName'].values[0]

    ax.text(-0.1, 1.15, f"{player_name} | {player_team}", size=15, color='black', ha='left', va='top', transform=ax.transAxes, weight='bold')
    ax.text(-0.1, 1.11, f"{player_position} | {player_minutes:.0f} minutes played", horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, color='black', size=12)

    today = datetime.date.today()
    _season_str = season_label if season_label else SEASON_ID_MAP.get(CURRENT_SEASON_ID, '25-26')
    _mode_label = "Raw per 90 (mean ± 2σ)" if radar_mode == 'raw' else "Percentile"
    plt.figtext(0.90, 0.90, f'Stats are per 90 mins \n{_mode_label} \n{_season_str} \nData via Wyscout \n@lucaskimball\nDate: {today}', horizontalalignment='left', fontsize=10, color='black')
    # Build legend dynamically based on which categories are present in the metrics
    _has_off_ball = any(m in OFF_BALL_DEFENDING_METRICS for m in metrics)
    legend_labels = ['Output Metrics', 'Passing Metrics', 'Defensive Metrics', 'Dribbling Metrics', 'Goalkeeping Metrics']
    legend_colors = [category_colors['output'], category_colors['passing'], category_colors['defensive'], category_colors['dribbling'], category_colors['goalkeeping']]
    if _has_off_ball:
        legend_labels.append('Off-ball Defending Metrics')
        legend_colors.append(category_colors['off_ball_defending'])
    patches = [plt.Line2D([0], [0], color=color, lw=4) for color in legend_colors]
    ax.legend(patches, legend_labels, loc='lower right', bbox_to_anchor=(1.7, 1), frameon=False)

    score_text = "\n"
    for group in eligible_groups:
        score_col = group + '_Score'
        rank_col = group + '_Rank'
        if score_col in player_data.columns:
            player_score = player_data[score_col].values[0]
            player_rank_str = ""
            try:
                if full_df_for_ranking is not None and not full_df_for_ranking.empty:
                    group_players = full_df_for_ranking[full_df_for_ranking['primaryPosition'].isin(POSITION_GROUPS[group])]
                    if score_col in group_players.columns:
                        group_players[rank_col] = group_players[score_col].rank(ascending=False, method='dense').astype(int)
                        if player_data.index[0] in group_players.index:
                            player_rank = group_players.loc[player_data.index[0], rank_col]
                            player_rank_str = f" (Rank: {player_rank})"
            except Exception as e:
                print(f"Warning: Could not calculate rank for {group}. Error: {e}")
            score_text += f"{group}: {player_score:.2f}{player_rank_str}\n"

    outside_background_color = (0.95, 0.92, 0.87); inside_radar_color = (0.99, 0.98, 0.95); score_box_color = (1.0, 0.99, 0.97)
    ax.set_facecolor(inside_radar_color)
    if ax.figure: ax.figure.patch.set_facecolor(outside_background_color)
    plt.figtext(.55, 1, score_text, horizontalalignment='left', verticalalignment='top', fontsize=12, bbox=dict(facecolor=score_box_color, alpha=0.5))


def get_percentile_suffix(value):
    """Function to add the appropriate suffix for the percentile."""
    value = int(value)
    if 10 <= value % 100 <= 20: suffix = 'th'
    else: suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(value % 10, 'th')
    return suffix

def create_radar_with_distributions(player_data, metrics, position, eligible_groups, all_position_data, full_df_for_ranking=None, season_label=None, radar_mode='percentile'):
    """Creates the combined figure with radar and distribution plots."""

    player_name = player_data['playerName'].values[0]
    highest_scoring_group = None; highest_score = -1; scores_by_group = {}

    for group in eligible_groups:
        score_col = group + '_Score'
        if score_col in player_data.columns:
            player_score = player_data[score_col].values[0]
            scores_by_group[group] = player_score
            if player_score > highest_score:
                highest_score = player_score; highest_scoring_group = group

    if highest_scoring_group is None:
        print(f"No highest scoring group found for {player_name}. Using default.")
        highest_scoring_group = eligible_groups[0] if eligible_groups else "Default"

    relevant_metrics = DISTRIBUTION_METRICS_BY_POSITION.get(highest_scoring_group, metrics)
    relevant_metrics = [m for m in relevant_metrics if m in player_data.columns]

    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(1, 2, width_ratios=[2.5, 1.2], figure=fig)
    ax_radar = plt.subplot(gs[0], polar=True)

    # Get population for the best-matching position group (used for raw mode)
    _pop_group = POSITION_GROUPS.get(highest_scoring_group, [player_data['primaryPosition'].values[0]])
    _pop_for_radar = all_position_data[all_position_data['primaryPosition'].isin(_pop_group)]

    _create_base_radar_chart(ax_radar, player_data, metrics, position, eligible_groups, full_df_for_ranking=full_df_for_ranking, season_label=season_label, radar_mode=radar_mode, population_data=_pop_for_radar)
    
    ax_radar.text(-0.1, 1.065, f"{highest_scoring_group} Template",
                  horizontalalignment='left', verticalalignment='center', transform=ax_radar.transAxes,
                  fontsize=14, fontweight='bold', color='black')

    # --- Distribution Plots ---
    primary_pos_group = POSITION_GROUPS.get(eligible_groups[0], [player_data['primaryPosition'].values[0]])
    relevant_players_data = all_position_data[all_position_data['primaryPosition'].isin(primary_pos_group)]
    # Exclude sub-threshold players from distributions
    if 'totalMinutes' in relevant_players_data.columns:
        relevant_players_data = relevant_players_data[pd.to_numeric(relevant_players_data['totalMinutes'], errors='coerce').fillna(0) >= 500]
    
    if relevant_metrics and not relevant_players_data.empty:
        gs_distributions = GridSpec(len(relevant_metrics), 1, left=0.70, right=0.98, top=0.82, bottom=0.07, hspace=0.7, figure=fig)
        for i, metric in enumerate(relevant_metrics):
            ax_dist = plt.subplot(gs_distributions[i])
            if metric in OUTPUT_METRICS: color = 'green'
            elif metric in PASSING_METRICS: color = 'orange'
            elif metric in DEFENSIVE_METRICS or metric in DEFR_DISPLAY_METRICS: color = 'red'
            elif metric in DRIBBLING_METRICS: color = 'purple'
            elif metric in GOALKEEPING_METRICS: color = 'cyan'
            else: color = 'blue'
            
            valid_relevant_players = relevant_players_data[relevant_players_data[metric].notna()][metric]
            if len(valid_relevant_players) > 1: sns.kdeplot(valid_relevant_players, ax=ax_dist, fill=True, color=color, cut=0)
            elif len(valid_relevant_players) == 1: ax_dist.axvline(valid_relevant_players.iloc[0], color=color, linestyle='-')
            
            player_value = player_data[metric].values[0]
            
            percentile_rank = 0
            if len(valid_relevant_players) > 0: percentile_rank = scipy.stats.percentileofscore(valid_relevant_players, player_value, kind='strict')
            percentile_rank_int = int(percentile_rank); suffix = get_percentile_suffix(percentile_rank_int)
            min_value = valid_relevant_players.min(); max_value = valid_relevant_players.max()
            # Extend axis to include the player's value so their marker is always
            # visible — covers outliers above/below the qualifying-population range.
            if not pd.isna(player_value):
                if pd.isna(min_value) or player_value < min_value: min_value = player_value
                if pd.isna(max_value) or player_value > max_value: max_value = player_value
            if pd.isna(min_value) or pd.isna(max_value) or min_value == max_value: min_value = player_value - 0.1; max_value = player_value + 0.1
            if min_value == max_value: max_value = min_value + 1.0 # Handle 0 case
            
            ax_dist.set_xlim(min_value, max_value); ax_dist.set_xticks([min_value, max_value]); ax_dist.set_xticklabels([fmt_val(metric, min_value), fmt_val(metric, max_value)], fontsize=8)
            ax_dist.axvline(player_value, color='blue', linestyle='--')
            raw_value = fmt_val(metric, player_value)
            ax_dist.text(1.05, 0.5, f"%-tile: {percentile_rank_int}{suffix}\np/90 value: {raw_value}", transform=ax_dist.transAxes, fontsize=8, verticalalignment='center')
            ax_dist.set_yticks([]); ax_dist.set_ylabel(""); ax_dist.set_title(""); ax_dist.set_xlabel("");
            legend = ax_dist.get_legend();
            if legend is not None: legend.remove()
            ax_dist.text(-0.05, 0.5, metric, transform=ax_dist.transAxes, fontsize=9, fontweight='bold', va='center', ha='right')

    return fig


def bulk_export_radars(export_df, full_pop_df, radar_mode='percentile',
                        season_label='All Seasons', progress_cb=None,
                        output_path=None):
    """Render one radar PNG per player in `export_df`. Each player's
    radar uses their best-fit role template against the same-position-
    group population in `full_pop_df`.

    Args:
        export_df: DataFrame of players to export (must have playerId,
            playerName, primaryPosition, totalMinutes, and the per-90 metrics
            + role _Score columns produced by calculate_player_percentiles_and_scores).
        full_pop_df: full qualifying-population DataFrame used as the
            distribution sample (typically the same-position group of
            player_stats_with_scores_df).
        radar_mode: 'percentile' or 'raw'.
        season_label: string passed through to the chart for the header.
        progress_cb: optional callable(i, n, name) for progress updates.
        output_path: if given, treat as a DIRECTORY and write each PNG
            individually as it's rendered. This is the resilient path —
            every completed radar is committed before the next one
            starts, so a process kill leaves an immediately-usable set
            of PNGs on disk. The caller is responsible for ZIPing the
            directory at download time. When omitted, falls back to
            the legacy in-memory ZIP-bytes path.

    Returns:
        tuple: (result, rendered, skipped) where result is either the
        output_path directory (when output_path given) or the
        in-memory ZIP bytes (legacy path).
    """
    import io
    import re
    import gc
    import zipfile

    # Position-bucket ordering: GK → CB → FB → CM → AM → WG → ST. Used to
    # name files so the ZIP listing groups by position and ranks players
    # by best-fit role score within each group.
    _RAW_TO_BUCKET = {
        'GK': ('1_GK', 1),
        'CB': ('2_CB', 2), 'LCB': ('2_CB', 2), 'RCB': ('2_CB', 2),
        'LCB3': ('2_CB', 2), 'RCB3': ('2_CB', 2),
        'LB': ('3_FB', 3), 'RB': ('3_FB', 3), 'LB5': ('3_FB', 3), 'RB5': ('3_FB', 3),
        'LWB': ('3_FB', 3), 'RWB': ('3_FB', 3),
        'CMF': ('4_CM', 4), 'LCMF': ('4_CM', 4), 'RCMF': ('4_CM', 4),
        'LCMF3': ('4_CM', 4), 'RCMF3': ('4_CM', 4),
        'DMF': ('4_CM', 4), 'LDMF': ('4_CM', 4), 'RDMF': ('4_CM', 4),
        'AMF': ('5_AM', 5),
        'LW': ('6_WG', 6), 'RW': ('6_WG', 6), 'LWF': ('6_WG', 6), 'RWF': ('6_WG', 6),
        'LAMF': ('6_WG', 6), 'RAMF': ('6_WG', 6),
        'LMF': ('6_WG', 6), 'RMF': ('6_WG', 6),
        'CF': ('7_ST', 7), 'SS': ('7_ST', 7),
    }

    # In-memory ZIP target — only used when caller wants bytes back
    # directly (no streaming to disk).
    zip_target = io.BytesIO() if output_path is None else None
    rendered = 0
    skipped = []

    # ---- Pre-compute (bucket, score, best_role, eligible) per player ----
    plan = []  # list of (bucket_label, bucket_order, -score, player_row, primary_pos, best_role, eligible_roles)
    for _, player_row in export_df.iterrows():
        player_name = str(player_row.get('playerName', '?'))
        primary_pos = player_row.get('primaryPosition', None)
        if primary_pos is None or pd.isna(primary_pos) or str(primary_pos) in ('Unknown', 'N/A', ''):
            skipped.append((player_name, 'missing primary position'))
            continue

        eligible_roles = [r for r in WEIGHTS
                           if primary_pos in POSITION_GROUPS.get(r, [])]
        if not eligible_roles:
            skipped.append((player_name, f'no role template for {primary_pos}'))
            continue

        best_role = max(
            eligible_roles,
            key=lambda r: float(player_row.get(f'{r}_Score', 0) or 0)
        )
        score = float(player_row.get(f'{best_role}_Score', 0) or 0)
        bucket_label, bucket_order = _RAW_TO_BUCKET.get(
            str(primary_pos), (f'9_{primary_pos}', 9)
        )
        plan.append((bucket_label, bucket_order, -score, player_row,
                      primary_pos, best_role, eligible_roles))

    # Sort: position bucket ascending, then score descending (negated above).
    plan.sort(key=lambda t: (t[1], t[2], str(t[3].get('playerName', ''))))

    n = len(plan)
    bucket_rank: dict = {}  # bucket_label -> running rank counter

    # If a directory output is requested, render each PNG as an individual
    # file on disk. That avoids the ZIP-format crash-recovery problem
    # entirely: there's no central directory that can be truncated when
    # the process is killed — every successfully rendered radar is its
    # own committed PNG. The caller is responsible for ZIPing the
    # directory at download time.
    if output_path is not None:
        out_dir = output_path
        os.makedirs(out_dir, exist_ok=True)
        seen_fnames: set = set()
        # Pre-scan existing PNGs so we can resume an interrupted render:
        # the plan ordering is deterministic, so any PNG already on disk
        # corresponds to a player we'd otherwise be about to re-render.
        # We compute the same fname during the loop and short-circuit
        # the expensive matplotlib work for any that already exist.
        try:
            existing_pngs = {f for f in os.listdir(out_dir) if f.endswith('.png')}
        except Exception:
            existing_pngs = set()
        resumed_count = 0
    else:
        # In-memory ZIP path — used only by callers that want bytes
        # back directly (no streaming to disk).
        seen_fnames = set()
        existing_pngs = set()
        resumed_count = 0
        zf_mem = zipfile.ZipFile(zip_target, 'w', zipfile.ZIP_DEFLATED, compresslevel=1)

    try:
        for i, (bucket_label, _bucket_order, neg_score, player_row,
                 primary_pos, best_role, eligible_roles) in enumerate(plan):
            player_name = str(player_row.get('playerName', f'player_{i}'))
            try:
                bucket_rank[bucket_label] = bucket_rank.get(bucket_label, 0) + 1
                rank = bucket_rank[bucket_label]

                # Compute the target filename BEFORE doing any expensive
                # rendering, so resume can short-circuit instantly.
                safe_name = re.sub(r'[^\w-]+', '_', player_name).strip('_') or f'player_{i}'
                safe_role = re.sub(r'[^\w-]+', '_', str(best_role)).strip('_')
                # Filename prefix encodes bucket + rank within bucket so the
                # final ZIP sorts by position group then score-desc.
                # e.g. 1_GK_01_Júlio_Neiva__Shot_Stopper.png
                fname = f"{bucket_label}_{rank:02d}_{safe_name}__{safe_role}.png"
                if fname in seen_fnames:
                    pid = player_row.get('playerId', i)
                    fname = f"{bucket_label}_{rank:02d}_{safe_name}__{safe_role}__{pid}.png"

                # Resume short-circuit: PNG already on disk from a prior
                # interrupted run with the same filter set. Skip the
                # expensive matplotlib work but still register the
                # bucket rank + seen name so downstream collisions and
                # bucket ranks line up with a fresh run.
                if output_path is not None and fname in existing_pngs:
                    seen_fnames.add(fname)
                    rendered += 1
                    resumed_count += 1
                    continue

                # Population for distributions: same position group.
                pop_pos_group = POSITION_GROUPS.get(best_role, [primary_pos])
                final_population = full_pop_df[full_pop_df['primaryPosition'].isin(pop_pos_group)]
                if len(final_population) < 5:
                    final_population = full_pop_df

                # Metrics for the chart (filter out hidden + unavailable).
                metrics_to_plot = [m for m in WEIGHTS[best_role].keys()
                                    if m in player_row.index
                                    and m not in RADAR_HIDDEN_METRICS]
                if not metrics_to_plot:
                    skipped.append((player_name, 'no metrics resolved'))
                    continue

                player_data_row = pd.DataFrame([player_row])

                fig = create_radar_with_distributions(
                    player_data_row,
                    metrics_to_plot,
                    best_role,
                    eligible_roles,
                    all_position_data=final_population,
                    full_df_for_ranking=full_pop_df,
                    season_label=season_label,
                    radar_mode=radar_mode,
                )

                if output_path is not None:
                    # Write PNG directly to disk — fully committed
                    # before we move to the next player, so crash here
                    # leaves a valid PNG on disk and a recoverable run.
                    png_path = os.path.join(out_dir, fname)
                    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=100)
                else:
                    img_buf = io.BytesIO()
                    fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=100)
                    zf_mem.writestr(fname, img_buf.getvalue())

                plt.close(fig)
                # Aggressive cleanup — matplotlib leaks state across
                # savefig calls and the bulk export builds dozens of
                # figures in a row. On HF Spaces a slow accumulation
                # OOM-kills the process partway through.
                plt.close('all')
                if (i + 1) % 10 == 0:
                    gc.collect()

                seen_fnames.add(fname)
                rendered += 1
            except Exception as exc:
                skipped.append((player_name, f'render failed: {exc}'))
                try:
                    plt.close('all')
                except Exception:
                    pass

            if progress_cb is not None:
                try:
                    progress_cb(i + 1, n, player_name, resumed_count)
                except TypeError:
                    # Back-compat: callers with the old 3-arg signature.
                    try:
                        progress_cb(i + 1, n, player_name)
                    except Exception:
                        pass
                except Exception:
                    pass
    finally:
        # Close the in-memory ZIP cleanly. Directory output has no
        # finalization step — each PNG already committed.
        if output_path is None:
            try:
                zf_mem.close()
            except Exception:
                pass

    # Attach resume metadata as an attribute on the returned tuple so
    # the caller can show "resumed X / rendered Y new" status without
    # changing the return signature.
    if output_path is not None:
        return output_path, rendered, skipped, resumed_count
    return zip_target.getvalue(), rendered, skipped, resumed_count


def plot_comparison_radar(ax, player_a_data, player_b_data, metrics, position_template):
    """
    Creates a 2-player comparison radar styled to replicate the user-provided image.
    (Layout V15: Moving metric labels closer)
    """
    fig = ax.figure # Get the figure object for fig.text
    
    num_metrics = len(metrics)
    angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
    angles += angles[:1]
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])  

    # --- Player A Values ---
    values_a = []
    for metric in metrics:
        col = metric + '_percentile'
        val = float(player_a_data[col].values[0]) if col in player_a_data.columns else 0.0
        values_a.append(np.nan_to_num(val, nan=0.0) * 100)
    values_a += values_a[:1]

    # --- Player B Values ---
    values_b = []
    for metric in metrics:
        col = metric + '_percentile'
        val = float(player_b_data[col].values[0]) if col in player_b_data.columns else 0.0
        values_b.append(np.nan_to_num(val, nan=0.0) * 100)
    values_b += values_b[:1]

    # --- Plot Polygons ---
    color_a = '#0077b6' # Blue
    color_b = '#A67B5B' # Light Coffee
    
    player_a_name = player_a_data['playerName'].values[0]
    player_b_name = player_b_data['playerName'].values[0]

    ax.plot(angles, values_a, linewidth=2, linestyle='solid', color=color_a, zorder=3, label=player_a_name)  
    ax.fill(angles, values_a, color_a, alpha=0.2, zorder=2)
    
    ax.plot(angles, values_b, linewidth=2, linestyle='solid', color=color_b, zorder=3, label=player_b_name)  
    ax.fill(angles, values_b, color_b, alpha=0.2, zorder=2)

    # --- Plot Metric Names and Values (Radius-based) ---
    category_colors = {'output': 'green', 'passing': 'orange', 'defensive': 'red', 'dribbling': 'purple', 'goalkeeping': 'cyan', 'off_ball_defending': '#7ECE2B'}

    for i, metric in enumerate(metrics):
        angle_rad = angles[i]

        # Metric Names (Farthest out)
        if metric in OFF_BALL_DEFENDING_METRICS: color = category_colors['off_ball_defending']
        elif metric in OUTPUT_METRICS: color = category_colors['output']
        elif metric in PASSING_METRICS: color = category_colors['passing']
        elif metric in DEFENSIVE_METRICS or metric in DEFR_DISPLAY_METRICS: color = category_colors['defensive']
        elif metric in DRIBBLING_METRICS: color = category_colors['dribbling']
        elif metric in GOALKEEPING_METRICS: color = category_colors['goalkeeping']
        else: color = 'grey'
        # --- FIX: Moved radius to 116, kept size 11 ---
        ax.text(angle_rad, 117, metric, size=11, ha='center', va='center', rotation=0, color=color, fontweight='bold')

        # Player A stats (raw and percentile)
        val_a_raw = player_a_data.get(metric, 0).values[0]
        val_a_pct = player_a_data.get(metric + '_percentile', 0).values[0]
        label_a = f"{fmt_val(metric, val_a_raw)} ({int(val_a_pct*100)}th)"

        # Player B stats (raw and percentile)
        val_b_raw = player_b_data.get(metric, 0).values[0]
        val_b_pct = player_b_data.get(metric + '_percentile', 0).values[0]
        label_b = f"{fmt_val(metric, val_b_raw)} ({int(val_b_pct*100)}th)"
        
        # --- DYNAMIC PLACEMENT TO AVOID OVERLAP ---
        angle_deg = np.degrees(angle_rad) % 360
        
        # --- FIX: Tighter stack at 90 and 78 ---
        outer_radius = 89
        inner_radius = 70

        if (80 < angle_deg < 100) or (260 < angle_deg < 280): # Left side
            radius_a = inner_radius 
            radius_b = outer_radius 
        elif (0 <= angle_deg <= 20) or (340 <= angle_deg <= 360): # Right side
            radius_a = outer_radius 
            radius_b = inner_radius 
        else: # Top and Bottom (stack normally)
            radius_a = outer_radius
            radius_b = inner_radius

        ax.text(angle_rad, radius_a, label_a, size=9, ha='center', va='center', color=color_a, fontweight='bold')
        ax.text(angle_rad, radius_b, label_b, size=9, ha='center', va='center', color=color_b, fontweight='bold')

    ax.set_rlabel_position(0)
    plt.yticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"], color="grey", size=7, zorder=1)  
    plt.ylim(0, 100)  

    # --- Titles and Info (Using fig.transAxes for layout) ---
    player_a_team = player_a_data['teamName'].values[0]
    player_a_mins = player_a_data['totalMinutes'].values[0]
    player_a_pos = player_a_data['primaryPosition'].values[0]
    
    player_b_team = player_b_data['teamName'].values[0]
    player_b_mins = player_b_data['totalMinutes'].values[0]
    player_b_pos = player_b_data['primaryPosition'].values[0]
    
    # --- Score Box & Player Info (Top-Left) ---
    score_col = position_template + '_Score'
    score_a = player_a_data[score_col].values[0] if score_col in player_a_data.columns else 0.0
    score_b = player_b_data[score_col].values[0] if score_col in player_b_data.columns else 0.0
    
    outside_background_color = (0.95, 0.92, 0.87); inside_radar_color = (0.99, 0.98, 0.95); score_box_color = (1.0, 0.99, 0.97)
    ax.set_facecolor(inside_radar_color)
    fig.patch.set_facecolor(outside_background_color)
    
    # --- Build the info box line-by-line ---
    box_x = 0.01
    box_y_start = 0.98
    line_height = 0.025 
    font_size_large = 13 
    font_size_small = 11 
    
    # Player A (Blue)
    fig.text(box_x, box_y_start, f"{player_a_name} | {player_a_team}", 
             horizontalalignment='left', verticalalignment='top', fontsize=font_size_large, 
             fontweight='bold', color=color_a, transform=fig.transFigure)
    fig.text(box_x, box_y_start - (line_height*0.8), f"{player_a_pos} | {player_a_mins:.0f} minutes played", 
             horizontalalignment='left', verticalalignment='top', fontsize=font_size_small, 
             color='black', transform=fig.transFigure)

    # Player B (Pink)
    fig.text(box_x, box_y_start - (line_height*1.8), f"{player_b_name} | {player_b_team}", 
             horizontalalignment='left', verticalalignment='top', fontsize=font_size_large, 
             fontweight='bold', color=color_b, transform=fig.transFigure)
    fig.text(box_x, box_y_start - (line_height*2.6), f"{player_b_pos} | {player_b_mins:.0f} minutes played", 
             horizontalalignment='left', verticalalignment='top', fontsize=font_size_small, 
             color='black', transform=fig.transFigure)

    # Template & Scores
    fig.text(box_x, box_y_start - (line_height*3.8), f"Template: *{position_template}*", 
             horizontalalignment='left', verticalalignment='top', fontsize=font_size_small, 
             style='italic', color='black', transform=fig.transFigure)
    fig.text(box_x, box_y_start - (line_height*4.8), f"{player_a_name}: {score_a:.2f}", 
             horizontalalignment='left', verticalalignment='top', fontsize=font_size_small, 
             fontweight='bold', color=color_a, transform=fig.transFigure)
    fig.text(box_x, box_y_start - (line_height*5.6), f"{player_b_name}: {score_b:.2f}", 
             horizontalalignment='left', verticalalignment='top', fontsize=font_size_small, 
             fontweight='bold', color=color_b, transform=fig.transFigure)
    
    # Add the background box manually
    box_height = 0.16
    fig.patches.extend([plt.Rectangle((0.005, 0.985 - box_height), 0.18, box_height, # x, y, width, height
                                      facecolor=score_box_color, alpha=0.5,
                                      transform=fig.transFigure, zorder=-1)])
    
    # --- Metric Legend (Top-Right) ---
    _has_off_ball = any(m in OFF_BALL_DEFENDING_METRICS for m in metrics)
    legend_labels = ['Output Metrics', 'Passing Metrics', 'Defensive Metrics', 'Dribbling Metrics', 'Goalkeeping Metrics']
    legend_colors = [category_colors['output'], category_colors['passing'], category_colors['defensive'], category_colors['dribbling'], category_colors['goalkeeping']]
    if _has_off_ball:
        legend_labels.append('Off-ball Defending Metrics')
        legend_colors.append(category_colors['off_ball_defending'])
    patches = [plt.Line2D([0], [0], color=color, lw=4) for color in legend_colors]

    fig.legend(patches, legend_labels, loc='upper right', bbox_to_anchor=(0.98, 0.98),
               frameon=False)
    
    # --- General Info (Bottom-Left) ---
    today = datetime.date.today()
    info_text = f'Stats are per 90 mins\nData via Wyscout\n@lucaskimball\nDate: {today}'
    fig.text(0.01, 0.01, info_text, 
             horizontalalignment='left', verticalalignment='bottom', 
             fontsize=10, color='black', transform=fig.transFigure)

# --- Radar Stats Calculation ---
@st.cache_data
def load_team_advanced_stats():
    """Load Wyscout team advanced stats from parquet if available."""
    try:
        df = pd.read_parquet('team_advanced_stats.parquet')
        if df.empty:
            return None
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def calculate_all_team_radars_stats(season_events_df, matches_summary_df, season_id=None,
                                       force_events=False):
    """Calculates aggregated stats and percentiles for Offensive, Distribution, and Defensive radars.

    Source priority:
      1. Wyscout team advanced stats — when force_events is False (default).
      2. Event-based calculation against the (possibly stage-filtered)
         season_events_df — when force_events is True or Wyscout data
         doesn't cover enough teams.

    Pass force_events=True when a stage filter is active so the radars
    reflect that subset of matches; Wyscout's table is season-aggregated
    and would otherwise leak full-season numbers into a stage view."""

    # Count how many teams are in the events data for comparison
    event_teams = set()
    if 'team.name' in season_events_df.columns:
        event_teams = set(season_events_df['team.name'].dropna().unique())

    if not force_events:
        wyscout_stats = load_team_advanced_stats()
        if wyscout_stats is not None:
            raw_df, pct_df = _build_radars_from_wyscout(wyscout_stats, season_id=season_id)
            # Only use Wyscout stats if they cover at least half the teams in the events
            if not raw_df.empty and (not event_teams or len(raw_df) >= len(event_teams) * 0.5):
                print(f"Using Wyscout team advanced stats for radars ({len(raw_df)} teams)...")
                return raw_df, pct_df
            print(f"Wyscout stats insufficient ({len(raw_df)} of {len(event_teams)} teams), falling back to events...")
    else:
        print("Stage filter active — forcing events-based radar stats")

    print("Calculating radar stats from events...")
    return _calculate_radars_from_events(season_events_df, matches_summary_df)

def _build_radars_from_wyscout(df, season_id=None):
    """Build radar DataFrames from Wyscout team advanced stats.
    Filters by season_id so percentiles are compared within the selected season."""
    if season_id is not None and 'seasonId' in df.columns:
        if isinstance(season_id, list):
            df = df[df['seasonId'].isin(season_id)]
        else:
            df = df[df['seasonId'] == season_id]
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    all_teams_stats = {}

    for _, row in df.iterrows():
        team = row['team_name']
        shots = row.get('shots', 0)
        xg = row.get('xg', 0)
        shots_against = row.get('shots_against', 0)
        xg_against = row.get('xg_shot_against', 0)
        passes = row.get('passes', 0)
        forward_passes = row.get('forward_passes', 0)

        all_teams_stats[team] = {
            'Goals': row.get('goals', 0),
            'xG': xg,
            'xG per Shot': xg / shots if shots > 0 else 0,
            'Shots': shots,
            'Actions in Box': row.get('touch_in_box', 0),
            'Passes into Box': row.get('passes_to_final_third', 0),
            'Crosses': row.get('crosses', 0),
            'Dribbles': row.get('successful_dribbles', 0),
            'Passes': passes,
            'Progressive Passes': row.get('progressive_run', 0),
            'Directness': (forward_passes / passes * 100) if passes > 0 else 0,
            'Ball Possession': row.get('possession_percent', 0),
            'Losses': row.get('ball_losses', 0),
            'Goals Against': row.get('conceded_goals', 0),
            'xG Against': xg_against,
            'xG per Shot Against': xg_against / shots_against if shots_against > 0 else 0,
            'Shots Against': shots_against,
            'Aerial Duel Win %': row.get('aerial_duels_won_pct', 0),
            'Defensive Duel Win %': row.get('defensive_duels_won_pct', 0),
            'Interceptions': row.get('interceptions', 0),
            'Fouls': row.get('fouls', 0),
            'PPDA': row.get('ppda', 0),
        }

    stats_df_raw = pd.DataFrame.from_dict(all_teams_stats, orient='index').fillna(0).round(2)
    stats_df_raw.replace([np.inf, -np.inf], 999, inplace=True)
    stats_df_pct = stats_df_raw.copy()
    metrics_to_invert_pct = ['Goals Against', 'xG Against', 'xG per Shot Against', 'Shots Against', 'PPDA', 'Losses']
    valid_metrics_to_invert = [col for col in metrics_to_invert_pct if col in stats_df_pct.columns]
    stats_df_pct[valid_metrics_to_invert] = -stats_df_pct[valid_metrics_to_invert]
    for col in stats_df_pct.columns:
        stats_df_pct[col] = stats_df_pct[col].rank(pct=True) * 100
    return stats_df_raw, stats_df_pct

def _calculate_radars_from_events(season_events_df, matches_summary_df):
    """Fallback: calculate radar stats from raw events data."""

    all_teams_stats = {}

    if 'team.name' not in season_events_df.columns:
         print("Warning: 'team.name' column missing from events_df, cannot calculate radar stats.")
         return pd.DataFrame(), pd.DataFrame()

    teams = season_events_df['team.name'].unique()
    matches_played = season_events_df.groupby('team.name')['matchId'].nunique() if 'matchId' in season_events_df.columns else pd.Series(dtype='int')

    season_events_df['possession.duration_sec'] = pd.to_numeric(season_events_df.get('possession.duration', pd.Series(dtype='str')).str.replace('s', ''), errors='coerce')
    season_events_df['location.x'] = pd.to_numeric(season_events_df.get('location.x'), errors='coerce')
    season_events_df['location.y'] = pd.to_numeric(season_events_df.get('location.y'), errors='coerce')
    season_events_df['pass.endLocation.x'] = pd.to_numeric(season_events_df.get('pass.endLocation.x'), errors='coerce')
    season_events_df['pass.endLocation.y'] = pd.to_numeric(season_events_df.get('pass.endLocation.y'), errors='coerce')
    season_events_df['pass.length'] = pd.to_numeric(season_events_df.get('pass.length'), errors='coerce')
    season_events_df['shot.xg'] = pd.to_numeric(season_events_df.get('shot.xg'), errors='coerce')

    possessions_df = season_events_df.drop_duplicates(subset='possession.id')[['matchId', 'possession.team.name', 'possession.duration_sec']]
    match_team_duration = possessions_df.groupby(['matchId', 'possession.team.name'])['possession.duration_sec'].sum().reset_index()
    match_total_duration = match_team_duration.groupby('matchId')['possession.duration_sec'].sum().reset_index().rename(columns={'possession.duration_sec': 'match_total_duration'})
    possession_data = pd.merge(match_team_duration, match_total_duration, on='matchId')
    possession_data = possession_data[possession_data['match_total_duration'] > 0]
    possession_data['possession_pct'] = (possession_data['possession.duration_sec'] / possession_data['match_total_duration']) * 100
    avg_possession_per_team = possession_data.groupby('possession.team.name')['possession_pct'].mean()

    losses_df = pd.DataFrame()
    if 'possession.id' in season_events_df.columns:
        season_events_df['next_possession.id'] = season_events_df['possession.id'].shift(-1)
        possession_changes = season_events_df[season_events_df['possession.id'] != season_events_df['next_possession.id']]
        losses_df = possession_changes[possession_changes.get('infraction.type') != 'foul_suffered'].copy()

    if 'opponentTeam.name' not in season_events_df.columns and 'matchId' in season_events_df.columns:
         temp_summary = matches_summary_df[['matchId', 'homeTeamName', 'awayTeamName']].copy()
         temp_summary.rename(columns={'homeTeamName':'ht', 'awayTeamName':'at'}, inplace=True)
         season_events_df = season_events_df.merge(temp_summary, on='matchId', how='left')
         season_events_df['opponentTeam.name'] = np.where(season_events_df['team.name'] == season_events_df['ht'], season_events_df['at'], season_events_df['ht'])
         season_events_df.drop(columns=['ht', 'at'], inplace=True, errors='ignore')

    for team in teams:
        team_events = season_events_df[season_events_df.get('team.name') == team]
        opponent_events = season_events_df[season_events_df.get('opponentTeam.name') == team] if 'opponentTeam.name' in season_events_df.columns else pd.DataFrame()
        games = matches_played.get(team, 0)
        if games == 0: continue

        team_shots = team_events[team_events.get('type.primary') == 'shot']
        shots = team_shots.shape[0] / games
        goals = team_shots[team_shots.get('shot.isGoal') == True].shape[0] / games
        xg = team_shots['shot.xg'].sum() / games
        xg_per_shot = xg / shots if shots > 0 else 0
        PENALTY_AREA_X=83; PENALTY_AREA_Y1, PENALTY_AREA_Y2 = (21, 79)
        actions_in_box = team_events[(team_events['location.x'].fillna(0) >= PENALTY_AREA_X) & (team_events['location.y'].fillna(0).between(PENALTY_AREA_Y1, PENALTY_AREA_Y2))].shape[0] / games
        team_passes = team_events[team_events.get('type.primary') == 'pass']
        passes_into_box = team_passes[(team_passes['pass.endLocation.x'].fillna(0) >= PENALTY_AREA_X) & (team_passes['pass.endLocation.y'].fillna(0).between(PENALTY_AREA_Y1, PENALTY_AREA_Y2))].shape[0] / games
        crosses = team_passes[team_passes.get('type.secondary','').astype(str).str.contains('cross', na=False)].shape[0] / games
        team_duels_off = team_events[team_events.get('type.primary') == 'duel']
        dribbles = team_duels_off[team_duels_off.get('groundDuel.takeOn') == True].shape[0] / games

        passes_per_match = team_passes.shape[0] / games
        team_passes['start_dist_to_goal'] = np.sqrt((100 - team_passes['location.x'])**2 + (50 - team_passes['location.y'])**2)
        team_passes['end_dist_to_goal'] = np.sqrt((100 - team_passes['pass.endLocation.x'])**2 + (50 - team_passes['pass.endLocation.y'])**2)
        team_passes['progression'] = team_passes['start_dist_to_goal'] - team_passes['end_dist_to_goal']
        cond1 = (team_passes['location.x'] <= 50) & (team_passes['pass.endLocation.x'] <= 50) & (team_passes['progression'] >= 30)
        cond2 = (team_passes['location.x'] <= 50) & (team_passes['pass.endLocation.x'] > 50) & (team_passes['progression'] >= 15)
        cond3 = (team_passes['location.x'] > 50) & (team_passes['pass.endLocation.x'] > 50) & (team_passes['progression'] >= 10)
        progressive_passes = team_passes[cond1 | cond2 | cond3].shape[0] / games
        directness = team_passes['progression'].mean()

        ball_possession_pct = avg_possession_per_team.get(team, 0)

        final_third_entries = 0
        if 'possession.id' in team_events.columns and 'location.x' in team_events.columns:
            try:
                possessions_grouped = team_events.groupby('possession.id')[['location.x']]
                valid_groups = possessions_grouped.filter(lambda x: not x['location.x'].isna().all())
                if not valid_groups.empty:
                     final_third_entries_series = valid_groups.groupby('possession.id')['location.x'].transform(lambda x: x.min() < 66.6 and x.max() >= 66.6)
                     final_third_entries = final_third_entries_series[final_third_entries_series].index.get_level_values('possession.id').nunique() / games
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to calculate final third entries for team: {e}")
                final_third_entries = 0
        losses = losses_df[losses_df.get('team.name') == team].shape[0] / games if not losses_df.empty else 0

        goals_against=0; xg_against=0; shots_against=0; xg_per_shot_against=0;
        aerial_duel_win_pct=0; defensive_duel_win_pct=0; interceptions=0; fouls=0; ppda=np.inf;

        if not opponent_events.empty:
            opponent_shots = opponent_events[opponent_events.get('type.primary') == 'shot']
            goals_against = opponent_shots[opponent_shots.get('shot.isGoal') == True].shape[0] / games
            xg_against = opponent_shots['shot.xg'].sum() / games
            shots_against = opponent_shots.shape[0] / games
            xg_per_shot_against = xg_against / shots_against if shots_against > 0 else 0

        team_duels_def = team_events[team_events.get('type.primary') == 'duel']
        aerial_duels = team_duels_def[team_duels_def.get('type.secondary','').astype(str).str.contains('aerial', na=False)]
        total_aerial_duels = aerial_duels.shape[0]; won_aerial_duels_count = aerial_duels[aerial_duels.get('aerialDuel.firstTouch') == True].shape[0]
        aerial_duel_win_pct = (won_aerial_duels_count / total_aerial_duels) * 100 if total_aerial_duels > 0 else 0

        defensive_duels = team_duels_def[team_duels_def.get('groundDuel.duelType') == 'defensive_duel']
        total_defensive_duels = defensive_duels.shape[0]
        won_defensive_duels_count = defensive_duels[
            (defensive_duels.get('groundDuel.recoveredPossession') == True) |
            (defensive_duels.get('groundDuel.stoppedProgress') == True)
        ].shape[0]
        defensive_duel_win_pct = (won_defensive_duels_count / total_defensive_duels) * 100 if total_defensive_duels > 0 else 0

        interceptions = team_events[team_events.get('type.primary') == 'interception'].shape[0] / games
        fouls = team_events[team_events.get('type.primary') == 'infraction'].shape[0] / games

        PRESS_ZONE_X = 40
        opponent_passes_count = opponent_events[
            (opponent_events.get('type.primary') == 'pass') &
            (opponent_events.get('location.x', 0) >= PRESS_ZONE_X)
        ].shape[0]

        team_press_events = team_events[team_events.get('location.x', 0) >= PRESS_ZONE_X]

        if not team_press_events.empty:
            fouls_press = team_press_events[team_press_events.get('type.primary') == 'infraction'].shape[0]
            interceptions_press = team_press_events[team_press_events.get('type.primary') == 'interception'].shape[0]
            def_duels_press = team_press_events[
                (team_press_events.get('type.primary') == 'duel') &
                (team_press_events.get('groundDuel.duelType') == 'defensive_duel')
            ]
            won_def_duels_press = def_duels_press[
                (def_duels_press.get('groundDuel.recoveredPossession') == True) |
                (def_duels_press.get('groundDuel.stoppedProgress') == True)
            ].shape[0]
            sliding_tackles_press = team_press_events[
                (team_press_events.get('type.primary') == 'duel') &
                (team_press_events.get('groundDuel.duelType', '').astype(str).str.contains('sliding_tackle', na=False))
            ].shape[0]

            total_def_actions = fouls_press + interceptions_press + won_def_duels_press + sliding_tackles_press
            ppda = opponent_passes_count / total_def_actions if total_def_actions > 0 else np.inf
        else:
            ppda = np.inf

        all_teams_stats[team] = {
            'Goals': goals, 'xG': xg, 'xG per Shot': xg_per_shot, 'Shots': shots,
            'Actions in Box': actions_in_box, 'Passes into Box': passes_into_box,
            'Crosses': crosses, 'Dribbles': dribbles,
            'Passes': passes_per_match, 'Progressive Passes': progressive_passes,
            'Directness': directness, 'Ball Possession': ball_possession_pct,
            'Final 1/3 Entries': final_third_entries, 'Losses': losses,
            'Goals Against': goals_against, 'xG Against': xg_against,
            'xG per Shot Against': xg_per_shot_against, 'Shots Against': shots_against,
            'Aerial Duel Win %': aerial_duel_win_pct, 'Defensive Duel Win %': defensive_duel_win_pct,
            'Interceptions': interceptions, 'Fouls': fouls, 'PPDA': ppda,
        }

    stats_df_raw = pd.DataFrame.from_dict(all_teams_stats, orient='index').fillna(0).round(2)
    stats_df_raw.replace([np.inf, -np.inf], 999, inplace=True)
    stats_df_pct = stats_df_raw.copy()
    metrics_to_invert_pct = ['Goals Against', 'xG Against', 'xG per Shot Against', 'Shots Against', 'PPDA', 'Losses']
    valid_metrics_to_invert = [col for col in metrics_to_invert_pct if col in stats_df_pct.columns]
    stats_df_pct[valid_metrics_to_invert] = -stats_df_pct[valid_metrics_to_invert]
    for col in stats_df_pct.columns:
        stats_df_pct[col] = stats_df_pct[col].rank(pct=True) * 100
    return stats_df_raw, stats_df_pct

# --- Radar Plotting Function (Unchanged) ---
def plot_radar_chart(params, values_raw, values_pct, team_name, title_suffix, color, league="Liga 3", season="2025/26"):
    # (This is the full function from the previous step)
    num_params = len(params); angles = np.linspace(0, 2 * np.pi, num_params, endpoint=False).tolist(); angles += angles[:1]
    plot_values_pct = values_pct + values_pct[:1]; fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.set_facecolor('#f5f1e9'); ax.set_facecolor('#f5f1e9'); ax.set_xticks(angles[:-1]); ax.set_ylim(0, 100)
    ax.grid(color='gray', linestyle='--', linewidth=0.5); ax.spines['polar'].set_color('gray'); ax.set_yticks([25, 50, 75])
    ax.set_yticklabels(["25th", "50th", "75th"], color="grey", size=10); ax.set_rlabel_position(angles[0] * 180/np.pi + 10); ax.set_thetagrids([], [])
    LABEL_DISTANCES = {"xG per Shot": 106, "Crosses": 107, "Directness": 106, "Avg Out-of-Possession Action Height": 108, "Avg In-Possession Action Height": 122, "Final 1/3 Entries": 117, "Shots Against": 106, "xG per Shot Against": 108, "PPDA": 110, "Quick Recoveries": 110, "Goals per Corner": 108, "xG per Corner": 108, "Short Corner %": 108, "Long Throw %": 108, "xG per Long Throw": 110, "xG per FK Delivery": 110, "Penalties": 107, "Non-Pen SP Goals": 112, "Corners": 107, "Long Throws": 108, "First Contact %": 108, "DEFAULT": 115}
    for angle, param, percentile in zip(angles[:-1], params, values_pct):
        percentile_val = int(round(percentile, 0)); label_text = f"{param}\n({percentile_val}th %-tile)"; distance = LABEL_DISTANCES.get(param, LABEL_DISTANCES["DEFAULT"])
        ha_align = 'left' if (np.degrees(angle) > 100 and np.degrees(angle) < 260) else 'right'; ha_align = 'center' if (abs(np.degrees(angle) - 90) < 10 or abs(np.degrees(angle) - 270) < 10) else ha_align
        ax.text(angle, distance, label_text, ha=ha_align, va='center', size=10)
    ax.plot(angles, plot_values_pct, color=color, linewidth=2, linestyle='solid'); ax.fill(angles, plot_values_pct, color=color, alpha=0.6)
    for angle, value_raw, value_pct in zip(angles[:-1], values_raw, values_pct):
         raw_display = f'{value_raw}%' if '%' in str(value_raw) else f'{value_raw}'; ax.text(angle, 95, raw_display, ha='center', va='top', size=9, weight='bold', bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='none', alpha=0.7))
    footer_text = "@lucaskimball | Data via Wyscout | Values in parentheses are percentile rank vs. other teams in league"; fig.text(0.02, 0.02, footer_text, ha='left', va='bottom', fontsize=9, color='gray')
    report_date = datetime.date.today().strftime("%Y-%m-%d"); full_title = f"{team_name}\n{title_suffix} | {league} {season} (As of: {report_date})"; ax.set_title(full_title, size=18, weight='bold', pad=40)
    return fig

# --- Corner Analysis Plotting Function (Unchanged) ---
def plot_corner_analysis(season_events_df, team_to_analyze, side, league="Liga 3", season="2025/26"):
    # (This is the full function from the previous step)
    def categorize_corner(row, side):
        end_x = row.get('pass.endLocation.x'); end_y = row.get('pass.endLocation.y'); pass_len = row.get('pass.length')
        if pd.isna(pass_len) and pd.notna(row.get('location.x')): start_x = row.get('location.x', 0); start_y = row.get('location.y', 0); PITCH_LENGTH_M, PITCH_WIDTH_M = 105.0, 68.0; pass_len = np.sqrt(((end_x - start_x) * (PITCH_LENGTH_M / 100.0))**2 + ((end_y - start_y) * (PITCH_WIDTH_M / 100.0))**2)
        if pd.isna(end_x) or pd.isna(end_y): return 'Other'
        PENALTY_AREA_X = 83; SIX_YARD_BOX_Y1, SIX_YARD_BOX_Y2 = (36, 64); SHORT_CORNER_MAX_DIST_FROM_START = 20
        if end_x < PENALTY_AREA_X or (pd.notna(pass_len) and pass_len < SHORT_CORNER_MAX_DIST_FROM_START): return 'Short'
        third_of_box = (SIX_YARD_BOX_Y2 - SIX_YARD_BOX_Y1) / 3; near_thresh = SIX_YARD_BOX_Y1 + third_of_box; far_thresh = SIX_YARD_BOX_Y2 - third_of_box
        if side == 'left':
            if end_y < near_thresh: return 'Near Post'
            elif end_y > far_thresh: return 'Far Post'
            else: return 'Middle'
        elif side == 'right':
             if end_y > far_thresh: return 'Near Post'
             elif end_y < near_thresh: return 'Far Post'
             else: return 'Middle'
        return 'Other'
    if side == 'left': side_corners_df = season_events_df[(season_events_df.get('team.name') == team_to_analyze) & (season_events_df.get('type.primary') == 'corner') & (season_events_df.get('location.y', 101) < 50)].copy()
    else: side_corners_df = season_events_df[(season_events_df.get('team.name') == team_to_analyze) & (season_events_df.get('type.primary') == 'corner') & (season_events_df.get('location.y', -1) >= 50)].copy()
    if side_corners_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5)); fig.set_facecolor('#f5f1e9'); ax.set_facecolor('#f5f1e9'); ax.text(0.5, 0.5, f'No {side} corners found for {team_to_analyze}', ha='center', va='center', fontsize=14); ax.axis('off'); return fig
    side_corners_df['zone'] = side_corners_df.apply(categorize_corner, axis=1, side=side)
    if 'player.name' in side_corners_df.columns: corner_takers = side_corners_df.groupby('player.name').agg(Total=('id', 'count'), Short=('zone', lambda x: (x == 'Short').sum()), Near=('zone', lambda x: (x == 'Near Post').sum()), Middle=('zone', lambda x: (x == 'Middle').sum()), Far=('zone', lambda x: (x == 'Far Post').sum())).sort_values(by='Total', ascending=False).fillna(0).astype(int)
    else: corner_takers = pd.DataFrame(columns=['Total', 'Short', 'Near', 'Middle', 'Far'])
    fig = plt.figure(figsize=(16, 8)); fig.set_facecolor('#f5f1e9'); gs = gridspec.GridSpec(1, 2, width_ratios=[0.6, 0.4]); ax_pitch = fig.add_subplot(gs[0, 0]); ax_table = fig.add_subplot(gs[0, 1]); ax_table.axis('off')
    pitch = Pitch(pitch_type='wyscout', pitch_color='#f5f1e9', line_color='black', line_zorder=2); pitch.draw(ax=ax_pitch); zone_colors = {'Short': 'blue', 'Near Post': 'orange', 'Middle': 'red', 'Far Post': 'yellow', 'Other': 'grey'}
    for idx, corner in side_corners_df.iterrows():
         if pd.notna(corner.get('pass.endLocation.x')) and pd.notna(corner.get('pass.endLocation.y')): pitch.scatter(x=corner['pass.endLocation.x'], y=corner['pass.endLocation.y'], s=200, color=zone_colors.get(corner['zone'], 'gray'), edgecolor='black', ax=ax_pitch, zorder=3, alpha=0.7)
    ax_pitch.set_title(f"Corners from the {side.capitalize()} Side | {league} {season}", fontsize=14); legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Short'), Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Near Post'), Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Middle'), Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Far Post'), Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=10, label='Other/Outside PA')]; ax_pitch.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.01, 0.01), frameon=False, fontsize=10)
    ax_table.set_title("Corner Taker Summary", fontsize=14, weight='bold')
    if not corner_takers.empty:
        table = Table(ax_table, bbox=[0, 0, 1, 0.9], loc='center'); table.auto_set_font_size(False); table.set_fontsize(10)
        table_data = [['Player'] + list(corner_takers.columns)] + [[idx] + list(row) for idx, row in corner_takers.iterrows()]
        col_widths = [0.4] + [0.12] * 5
        for i, row_list in enumerate(table_data):
            for j, cell_text in enumerate(row_list):
                is_header = (i == 0); weight = 'bold' if is_header or j == 0 else 'normal'; facecolor = '#e0e0e0' if is_header else ['#fdfdfd', '#f0f0f0'][i % 2]; loc = 'left' if j == 0 else 'center'
                cell = table.add_cell(i, j, width=col_widths[j], height=1.0/len(table_data), text=cell_text, loc=loc, facecolor=facecolor, edgecolor='w', fontproperties={'weight': weight})
        ax_table.add_table(table)
    else: ax_table.text(0.5, 0.5, "No corner takers found.", ha='center', va='center')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); return fig

# --- (ORIGINAL MATPLOTLIB FUNCTION 1) ---
def create_match_shotmap(match_events_df, match_info, team_to_analyze):

    team_shots_df = match_events_df[(match_events_df.get('team.name') == team_to_analyze) & (match_events_df.get('type.primary').isin(['shot', 'penalty']))].copy().reset_index(drop=True)
    if team_shots_df.empty:
        fig, ax = plt.subplots(figsize=(12, 8)); fig.set_facecolor('#f5f1e9'); ax.set_facecolor('#f5f1e9'); ax.text(0.5, 0.5, 'No shots found for this team in this match.', ha='center', va='center', fontsize=12); ax.axis('off'); return fig

    home_team = match_info.get('homeTeamName', '?'); away_team = match_info.get('awayTeamName', '?'); opponent = away_team if team_to_analyze == home_team else home_team
    fig = plt.figure(figsize=(12, 8)); fig.set_facecolor('#f5f1e9')
    pitch = Pitch(pitch_type='wyscout', pitch_color='#f5f1e9', line_color='black', line_zorder=2, half=True)
    ax_pitch = fig.add_axes([0.02, 0.02, 0.96, 0.82])
    pitch.draw(ax=ax_pitch)

    XG_MAX = 0.8; colors = ["#03045e", "#ade8f4", "#fff3b0", "#ff8c00", "#e63946", "#800f2f"]; nodes = [0.0, 0.1 / XG_MAX, 0.2 / XG_MAX, 0.4 / XG_MAX, 0.6 / XG_MAX, 1.0]; cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))

    for index, shot in team_shots_df.iterrows():
        x = shot.get('location.x'); y = shot.get('location.y'); xg = pd.to_numeric(shot.get('shot.xg'), errors='coerce')
        if pd.isna(x) or pd.isna(y) or pd.isna(xg): continue
        is_goal = shot.get('shot.isGoal') == True; is_on_target = shot.get('shot.onTarget') == True

        color = cmap(min(xg / XG_MAX, 1.0)); edge_color = 'gray'; line_width = 1.5

        if is_goal: edge_color = 'green'; line_width = 2.5
        elif is_on_target: edge_color = 'black'; line_width = 2.5

        pitch.scatter(x, y, s=400, facecolor=color, edgecolor=edge_color, linewidth=line_width, ax=ax_pitch, zorder=3)
        pitch.text(x, y, str(index + 1), ax=ax_pitch, ha='center', va='center', fontsize=9, color='white', zorder=4)

    subtitle = f"vs. {opponent} | Score: {match_info.get('score', '?-?')} | xG: {pd.to_numeric(team_shots_df['shot.xg'], errors='coerce').sum():.2f}"
    ax_pitch.set_title(f"{team_to_analyze} Shot Map\n{subtitle}", fontsize=14, weight='bold')
    return fig

# --- (ORIGINAL MATPLOTLIB FUNCTION 2) ---
def create_season_shotmap(season_events_df, team_to_analyze):

    team_shots_df = season_events_df[(season_events_df.get('team.name') == team_to_analyze) & (season_events_df.get('type.primary') == 'shot')].copy().reset_index(drop=True)
    if team_shots_df.empty:
        fig, ax = plt.subplots(figsize=(12, 8)); fig.set_facecolor('#f5f1e9'); ax.set_facecolor('#f5f1e9'); ax.text(0.5, 0.5, 'No shots found for this team this season.', ha='center', va='center', fontsize=12); ax.axis('off'); return fig
    
    fig = plt.figure(figsize=(12, 8)); fig.set_facecolor('#f5f1e9')
    pitch = Pitch(pitch_type='wyscout', pitch_color='#f5f1e9', line_color='black', line_zorder=2, half=True)
    ax_pitch = fig.add_axes([0.02, 0.02, 0.96, 0.82])
    pitch.draw(ax=ax_pitch)

    XG_MAX = 0.8; colors = ["#03045e", "#ade8f4", "#fff3b0", "#ff8c00", "#e63946", "#800f2f"]; nodes = [0.0, 0.1 / XG_MAX, 0.2 / XG_MAX, 0.4 / XG_MAX, 0.6 / XG_MAX, 1.0]; cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))

    for index, shot in team_shots_df.iterrows():
        x = shot.get('location.x'); y = shot.get('location.y'); xg = pd.to_numeric(shot.get('shot.xg'), errors='coerce')
        if pd.isna(x) or pd.isna(y) or pd.isna(xg): continue
        is_goal = shot.get('shot.isGoal') == True; color = cmap(min(xg / XG_MAX, 1.0)); edge_color = 'green' if is_goal else 'black'
        pitch.scatter(x, y, s=150, facecolor=color, edgecolor=edge_color, linewidth=1.5, ax=ax_pitch, zorder=3, alpha=0.7)

    total_xg = pd.to_numeric(team_shots_df['shot.xg'], errors='coerce').sum(); goals = team_shots_df['shot.isGoal'].sum()
    subtitle = f"Total xG: {total_xg:.2f} | Goals: {goals}"
    ax_pitch.set_title(f"{team_to_analyze} — Shots For (Non-Pen)\n{subtitle}", fontsize=14, weight='bold')
    return fig

# --- (ORIGINAL MATPLOTLIB FUNCTION 3) ---
def create_season_shots_against_shotmap(season_events_df, matches_summary_df, team_to_analyze):

    team_match_ids = matches_summary_df[(matches_summary_df.get('homeTeamName') == team_to_analyze) | (matches_summary_df.get('awayTeamName') == team_to_analyze)]['matchId'].unique()
    relevant_events = season_events_df[season_events_df['matchId'].isin(team_match_ids)]
    opponent_shots_df = relevant_events[(relevant_events.get('type.primary') == 'shot') & (relevant_events.get('team.name') != team_to_analyze)].copy().reset_index(drop=True)
    if opponent_shots_df.empty:
        fig, ax = plt.subplots(figsize=(12, 8)); fig.set_facecolor('#f5f1e9'); ax.set_facecolor('#f5f1e9'); ax.text(0.5, 0.5, 'No shots against found for this team.', ha='center', va='center', fontsize=12); ax.axis('off'); return fig

    fig = plt.figure(figsize=(12, 8)); fig.set_facecolor('#f5f1e9')
    pitch = Pitch(pitch_type='wyscout', pitch_color='#f5f1e9', line_color='black', line_zorder=2, half=True)
    ax_pitch = fig.add_axes([0.02, 0.02, 0.96, 0.82])
    pitch.draw(ax=ax_pitch)

    XG_MAX = 0.8; colors = ["#03045e", "#ade8f4", "#fff3b0", "#ff8c00", "#e63946", "#800f2f"]; nodes = [0.0, 0.1 / XG_MAX, 0.2 / XG_MAX, 0.4 / XG_MAX, 0.6 / XG_MAX, 1.0]; cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))

    for index, shot in opponent_shots_df.iterrows():
        x = shot.get('location.x'); y = shot.get('location.y'); xg = pd.to_numeric(shot.get('shot.xg'), errors='coerce'); is_goal = shot.get('shot.isGoal') == True
        if pd.isna(x) or pd.isna(y) or pd.isna(xg): continue
        color = cmap(min(xg / XG_MAX, 1.0)); edge_color = 'green' if is_goal else 'black'; pitch.scatter(x, y, s=150, facecolor=color, edgecolor=edge_color, linewidth=1.5, ax=ax_pitch, zorder=3, alpha=0.7)

    total_xg_against = round(pd.to_numeric(opponent_shots_df.get('shot.xg'), errors='coerce').sum(), 2); goals_against = opponent_shots_df[opponent_shots_df.get('shot.isGoal') == True].shape[0]
    subtitle = f"Total xGA: {total_xg_against} | Goals Against: {goals_against}"
    ax_pitch.set_title(f"{team_to_analyze} — Shots Conceded (Non-Pen)\n{subtitle}", fontsize=14, weight='bold')
    return fig

def create_player_shotmap(player_shots_df, player_name):
    """
    Creates a static Matplotlib shotmap for a specific player (Non-Penalty).
    Includes shot numbering and an xG color scale.
    """
    if player_shots_df.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.set_facecolor('#f5f1e9'); ax.set_facecolor('#f5f1e9')
        ax.text(0.5, 0.5, 'No shots recorded for this player.', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # Setup Pitch
    fig = plt.figure(figsize=(12, 8))
    fig.set_facecolor('#f5f1e9')
    pitch = Pitch(pitch_type='wyscout', pitch_color='#f5f1e9', line_color='black', line_zorder=2, half=True)
    ax_pitch = fig.add_axes([0.02, 0.02, 0.96, 0.82])
    pitch.draw(ax=ax_pitch)
    
    # Colormap for xG
    XG_MAX = 0.8
    colors = ["#03045e", "#ade8f4", "#fff3b0", "#ff8c00", "#e63946", "#800f2f"]
    nodes = [0.0, 0.1 / XG_MAX, 0.2 / XG_MAX, 0.4 / XG_MAX, 0.6 / XG_MAX, 1.0]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))
    
    # Plot Shots
    for index, shot in player_shots_df.iterrows():
        x = shot.get('location.x')
        y = shot.get('location.y')
        xg = pd.to_numeric(shot.get('shot.xg'), errors='coerce')
        shot_num = shot.get('Shot Number', index + 1)
        
        if pd.isna(x) or pd.isna(y) or pd.isna(xg): continue
        
        is_goal = shot.get('shot.isGoal') == True
        color = cmap(min(xg / XG_MAX, 1.0))
        edge_color = 'green' if is_goal else 'black'
        z_order = 4 if is_goal else 3
        
        # Draw marker
        pitch.scatter(x, y, s=450, facecolor=color, edgecolor=edge_color, linewidth=2 if is_goal else 1, ax=ax_pitch, zorder=z_order, alpha=0.9)
        
        # Draw Number
        pitch.text(x, y, str(shot_num), ax=ax_pitch, ha='center', va='center', color='white', fontsize=10, fontweight='bold', zorder=z_order+1)

    # Add Colorbar Legend
    norm = plt.Normalize(vmin=0, vmax=XG_MAX)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_pitch, orientation='vertical', fraction=0.04, pad=0.04, shrink=0.6)
    cbar.set_label('Expected Goals (xG)', fontsize=12)
    cbar.outline.set_visible(False)
    
    # Title and Subtitle
    total_xg = pd.to_numeric(player_shots_df['shot.xg'], errors='coerce').sum()
    goals = player_shots_df['shot.isGoal'].sum()
    team_name = player_shots_df.iloc[0]['team.name'] if 'team.name' in player_shots_df.columns else "Unknown Team"
    
    ax_pitch.set_title(f"{player_name} | {team_name}\nSeason Shot Map (Non-Penalty)", fontsize=18, weight='bold', pad=15)
    ax_pitch.text(50, 95, f"Total Shots: {len(player_shots_df)} | Goals: {goals} | Total xG: {total_xg:.2f}", 
                  ha='center', va='center', fontsize=12, color='black')
    
    return fig

# --- NEW FUNCTION: Calculate Team Strength ---
@st.cache_data
def calculate_team_strength(season_events_df, matches_summary_df, season_id=None):
    """Calculates Attacking and Defending Strength metrics for all teams."""
    if season_id is not None:
        cache_path = os.path.join(STATS_CACHE_DIR, f'team_strength_{season_id}.parquet')
        if os.path.exists(cache_path):
            print(f"Loading cached team strength for season {season_id}")
            return pd.read_parquet(cache_path)
    print("Calculating team strength stats...") # Debug print
    team_stats = {}

    # Ensure necessary columns exist and are numeric
    all_shots = season_events_df[season_events_df.get('type.primary') == 'shot'].copy()
    all_shots['shot.xg'] = pd.to_numeric(all_shots.get('shot.xg'), errors='coerce').fillna(0)
    all_shots['shot.isGoal'] = all_shots.get('shot.isGoal') == True # Ensure boolean

    all_teams_in_data = season_events_df['team.name'].unique()
    # Ensure matchId exists before grouping
    matches_played = season_events_df.groupby('team.name')['matchId'].nunique() if 'matchId' in season_events_df.columns else pd.Series(dtype='int')

    # Add opponent name if missing (needed for GA/xGA)
    if 'opponentTeam.name' not in all_shots.columns and 'matchId' in all_shots.columns:
         if matches_summary_df is not None and not matches_summary_df.empty:
             # --- UPDATED Column Names ---
             temp_summary = matches_summary_df[['matchId', 'homeTeamName', 'awayTeamName']].copy()
             temp_summary.rename(columns={'homeTeamName':'ht', 'awayTeamName':'at'}, inplace=True)
             # ---
             all_shots = all_shots.merge(temp_summary, on='matchId', how='left')
             all_shots['opponentTeam.name'] = np.where(all_shots['team.name'] == all_shots['ht'], all_shots['at'], all_shots['ht'])
             all_shots.drop(columns=['ht', 'at'], inplace=True, errors='ignore')
         else:
             print("Warning: Cannot calculate GA/xGA reliably without opponent names.")
             all_shots['opponentTeam.name'] = "Unknown Opponent"


    for team in all_teams_in_data:
        team_shots = all_shots[all_shots.get('team.name') == team]
        goals_for = team_shots['shot.isGoal'].sum()
        xg_for = team_shots['shot.xg'].sum()

        opponent_shots = all_shots[all_shots.get('opponentTeam.name') == team]
        goals_against = opponent_shots['shot.isGoal'].sum()
        xg_against = opponent_shots['shot.xg'].sum()

        games = matches_played.get(team, 0)
        if games > 0:
            team_stats[team] = {
                'GF_per_match': goals_for / games,
                'GA_per_match': goals_against / games,
                'xGF_per_match': xg_for / games,
                'xGA_per_match': xg_against / games
            }

    stats_df = pd.DataFrame.from_dict(team_stats, orient='index').fillna(0)
    if stats_df.empty:
        return pd.DataFrame() # Return empty if no stats calculated

    # Calculate Strength Metrics
    stats_df['Attacking Strength'] = (stats_df['GF_per_match'] * 0.3) + (stats_df['xGF_per_match'] * 0.7)
    stats_df['Defending Strength'] = (stats_df['GA_per_match'] * 0.3) + (stats_df['xGA_per_match'] * 0.7)

    if season_id is not None:
        os.makedirs(STATS_CACHE_DIR, exist_ok=True)
        try:
            stats_df.to_parquet(os.path.join(STATS_CACHE_DIR, f'team_strength_{season_id}.parquet'))
        except Exception:
            pass

    return stats_df


# --- NEW FUNCTION: Rolling Per-Match Team Strength (No Data Leakage) ---
@st.cache_data
def calculate_rolling_team_strength(season_events_df, matches_summary_df, season_id=None):
    """Per-match rolling team strength with no data leakage.
    Returns DataFrame: matchId, team, matchDate, match_number,
                       att_strength, def_strength, cum_gf, cum_ga, cum_xgf, cum_xga, cum_matches
    """
    if season_id is not None:
        cache_path = os.path.join(STATS_CACHE_DIR, f'rolling_strength_{season_id}.parquet')
        if os.path.exists(cache_path):
            return pd.read_parquet(cache_path)

    # Pre-compute per-match-team shot aggregates (vectorized)
    shots = season_events_df[season_events_df['type.primary'] == 'shot'].copy()
    shots['shot.xg'] = pd.to_numeric(shots['shot.xg'], errors='coerce').fillna(0)
    shots['shot.isGoal'] = shots['shot.isGoal'] == True
    match_team_xg = shots.groupby(['matchId', 'team.name']).agg(
        goals=('shot.isGoal', 'sum'), xg=('shot.xg', 'sum')
    ).reset_index()

    # Get match info with dates and scores
    match_info = matches_summary_df[['matchId', 'dateutc', 'homeTeamName', 'awayTeamName', 'score']].copy()
    match_info = match_info.sort_values('dateutc')

    # Accumulators per team
    team_accum = {}  # {team: {gf, ga, xgf, xga, matches}}
    rows = []

    for _, match in match_info.iterrows():
        mid = match['matchId']
        home = match['homeTeamName']
        away = match['awayTeamName']
        match_date = match['dateutc']

        # Parse score
        score = match.get('score', '')
        if pd.isna(score) or '-' not in str(score):
            continue
        try:
            home_goals, away_goals = map(int, str(score).split('-'))
        except (ValueError, TypeError):
            continue

        # Get xG from pre-computed groupby
        home_xg_row = match_team_xg[(match_team_xg['matchId'] == mid) & (match_team_xg['team.name'] == home)]
        away_xg_row = match_team_xg[(match_team_xg['matchId'] == mid) & (match_team_xg['team.name'] == away)]
        home_xg = float(home_xg_row['xg'].iloc[0]) if len(home_xg_row) > 0 else 0.0
        away_xg = float(away_xg_row['xg'].iloc[0]) if len(away_xg_row) > 0 else 0.0

        # Record PRE-MATCH strength for both teams
        for team in [home, away]:
            if team not in team_accum:
                team_accum[team] = {'gf': 0, 'ga': 0, 'xgf': 0.0, 'xga': 0.0, 'matches': 0}
            acc = team_accum[team]
            m = acc['matches']
            rows.append({
                'matchId': mid, 'team': team, 'matchDate': match_date,
                'match_number': m,
                'att_strength': (0.3 * (acc['gf'] / m) + 0.7 * (acc['xgf'] / m)) if m > 0 else np.nan,
                'def_strength': (0.3 * (acc['ga'] / m) + 0.7 * (acc['xga'] / m)) if m > 0 else np.nan,
                'cum_gf': acc['gf'], 'cum_ga': acc['ga'],
                'cum_xgf': acc['xgf'], 'cum_xga': acc['xga'], 'cum_matches': m,
            })

        # Update accumulators AFTER recording pre-match strength
        team_accum[home]['gf'] += home_goals
        team_accum[home]['ga'] += away_goals
        team_accum[home]['xgf'] += home_xg
        team_accum[home]['xga'] += away_xg
        team_accum[home]['matches'] += 1

        team_accum[away]['gf'] += away_goals
        team_accum[away]['ga'] += home_goals
        team_accum[away]['xgf'] += away_xg
        team_accum[away]['xga'] += home_xg
        team_accum[away]['matches'] += 1

    result_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    if season_id is not None and not result_df.empty:
        os.makedirs(STATS_CACHE_DIR, exist_ok=True)
        try:
            result_df.to_parquet(os.path.join(STATS_CACHE_DIR, f'rolling_strength_{season_id}.parquet'))
        except Exception:
            pass

    return result_df


# --- NEW FUNCTION: SOS-Adjusted Team Strength ---
@st.cache_data
def calculate_sos_adjusted_strength(rolling_strength_df, team_strength_df, season_id=None):
    """SOS-adjust team strength ratings.
    Returns DataFrame (index=team): raw_att, raw_def, avg_opp_att, avg_opp_def,
                                     sos_att, sos_def, sos_factor
    """
    if season_id is not None:
        cache_path = os.path.join(STATS_CACHE_DIR, f'sos_strength_{season_id}.parquet')
        if os.path.exists(cache_path):
            return pd.read_parquet(cache_path)

    if rolling_strength_df.empty or team_strength_df.empty:
        return pd.DataFrame()

    # League averages from end-of-season team strength
    league_avg_att = max(team_strength_df['Attacking Strength'].mean(), 0.01)
    league_avg_def = max(team_strength_df['Defending Strength'].mean(), 0.01)

    # For each match, find opponent's pre-match strength
    # rolling_strength_df has one row per (matchId, team) — merge to find opponent
    match_teams = rolling_strength_df[['matchId', 'team', 'att_strength', 'def_strength']].copy()
    # Self-join: for each (matchId, team), find the other team in the same match
    opp = match_teams.merge(match_teams, on='matchId', suffixes=('', '_opp'))
    opp = opp[opp['team'] != opp['team_opp']]

    # Average opponent strength faced by each team (only where opponent had prior data)
    opp_valid = opp.dropna(subset=['att_strength_opp', 'def_strength_opp'])
    avg_opp = opp_valid.groupby('team').agg(
        avg_opp_att=('att_strength_opp', 'mean'),
        avg_opp_def=('def_strength_opp', 'mean'),
        matches_with_opp_data=('att_strength_opp', 'count')
    )

    # Build result
    result = team_strength_df[['Attacking Strength', 'Defending Strength']].copy()
    result.columns = ['raw_att', 'raw_def']
    result = result.join(avg_opp, how='left')

    # SOS adjustment — teams with < 3 matches with opponent data fall back to raw
    result['sos_att_factor'] = result['avg_opp_def'] / league_avg_def
    result['sos_def_factor'] = result['avg_opp_att'] / league_avg_att

    has_enough = result['matches_with_opp_data'].fillna(0) >= 3
    result['sos_att'] = np.where(has_enough, result['raw_att'] * result['sos_att_factor'], result['raw_att'])
    result['sos_def'] = np.where(has_enough, result['raw_def'] * result['sos_def_factor'], result['raw_def'])
    result['sos_factor'] = np.where(has_enough, (result['sos_att_factor'] + result['sos_def_factor']) / 2, np.nan)

    if season_id is not None:
        os.makedirs(STATS_CACHE_DIR, exist_ok=True)
        try:
            result.to_parquet(os.path.join(STATS_CACHE_DIR, f'sos_strength_{season_id}.parquet'))
        except Exception:
            pass

    return result


# --- NEW FUNCTION: Build Season Cumulative Stats ---
@st.cache_data
def build_season_cumulative_stats(raw_events_df, matches_summary_df, season_id):
    """Build end-of-season cumulative stats for all teams in a season.
    Returns: {team_name: {matches, wins, draws, losses, points, goals_for, goals_against,
              xG_for, xG_against, shots_for, sot_for, home_matches, home_wins, home_goals,
              away_matches, away_wins, away_goals, clean_sheets, last_5_results, last_5_xG,
              prior_stats}}
    """
    cache_path = os.path.join(STATS_CACHE_DIR, f'season_cum_stats_{season_id}.pkl')
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass

    # Filter to this season
    season_events = raw_events_df[raw_events_df['seasonId'] == season_id] if 'seasonId' in raw_events_df.columns else raw_events_df
    season_matches = matches_summary_df[matches_summary_df['seasonId'] == season_id] if 'seasonId' in matches_summary_df.columns else matches_summary_df

    # Pre-compute per-match xG and shots (vectorized)
    shots = season_events[season_events['type.primary'] == 'shot'].copy()
    shots['shot.xg'] = pd.to_numeric(shots['shot.xg'], errors='coerce').fillna(0)
    shots['shot.isGoal'] = shots['shot.isGoal'] == True
    match_team_xg = shots.groupby(['matchId', 'team.name']).agg(
        xg=('shot.xg', 'sum'), total_shots=('shot.isGoal', 'count'), goals=('shot.isGoal', 'sum')
    ).reset_index()

    # Sort matches chronologically
    matches = season_matches.sort_values('dateutc' if 'dateutc' in season_matches.columns else 'matchId').copy()

    def _init_stats():
        return {
            'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'points': 0,
            'goals_for': 0, 'goals_against': 0, 'xG_for': 0.0, 'xG_against': 0.0,
            'shots_for': 0, 'sot_for': 0, 'home_matches': 0, 'home_wins': 0,
            'home_goals': 0, 'away_matches': 0, 'away_wins': 0, 'away_goals': 0,
            'clean_sheets': 0, 'last_5_results': [], 'last_5_xG': []
        }

    team_stats = defaultdict(_init_stats)

    for _, match in matches.iterrows():
        mid = match['matchId']
        home = match['homeTeamName']
        away = match['awayTeamName']

        score = match.get('score', '')
        if pd.isna(score) or '-' not in str(score):
            continue
        try:
            home_goals, away_goals = map(int, str(score).split('-'))
        except (ValueError, TypeError):
            continue

        # Update match/goal stats
        team_stats[home]['matches'] += 1
        team_stats[home]['home_matches'] += 1
        team_stats[home]['goals_for'] += home_goals
        team_stats[home]['goals_against'] += away_goals
        team_stats[home]['home_goals'] += home_goals

        team_stats[away]['matches'] += 1
        team_stats[away]['away_matches'] += 1
        team_stats[away]['goals_for'] += away_goals
        team_stats[away]['goals_against'] += home_goals
        team_stats[away]['away_goals'] += away_goals

        # Results
        if home_goals > away_goals:
            team_stats[home]['wins'] += 1
            team_stats[home]['home_wins'] += 1
            team_stats[home]['points'] += 3
            team_stats[home]['last_5_results'].append(3)
            team_stats[away]['losses'] += 1
            team_stats[away]['last_5_results'].append(0)
        elif away_goals > home_goals:
            team_stats[away]['wins'] += 1
            team_stats[away]['away_wins'] += 1
            team_stats[away]['points'] += 3
            team_stats[away]['last_5_results'].append(3)
            team_stats[home]['losses'] += 1
            team_stats[home]['last_5_results'].append(0)
        else:
            team_stats[home]['draws'] += 1
            team_stats[home]['points'] += 1
            team_stats[home]['last_5_results'].append(1)
            team_stats[away]['draws'] += 1
            team_stats[away]['points'] += 1
            team_stats[away]['last_5_results'].append(1)

        # Clean sheets
        if away_goals == 0:
            team_stats[home]['clean_sheets'] += 1
        if home_goals == 0:
            team_stats[away]['clean_sheets'] += 1

        # xG and shots from pre-computed groupby
        home_xg_row = match_team_xg[(match_team_xg['matchId'] == mid) & (match_team_xg['team.name'] == home)]
        away_xg_row = match_team_xg[(match_team_xg['matchId'] == mid) & (match_team_xg['team.name'] == away)]

        home_xg = float(home_xg_row['xg'].iloc[0]) if len(home_xg_row) > 0 else 0.0
        away_xg = float(away_xg_row['xg'].iloc[0]) if len(away_xg_row) > 0 else 0.0
        home_shots = int(home_xg_row['total_shots'].iloc[0]) if len(home_xg_row) > 0 else 0
        away_shots = int(away_xg_row['total_shots'].iloc[0]) if len(away_xg_row) > 0 else 0
        home_shot_goals = int(home_xg_row['goals'].iloc[0]) if len(home_xg_row) > 0 else 0
        away_shot_goals = int(away_xg_row['goals'].iloc[0]) if len(away_xg_row) > 0 else 0

        team_stats[home]['xG_for'] += home_xg
        team_stats[home]['xG_against'] += away_xg
        team_stats[home]['last_5_xG'].append(home_xg)
        team_stats[away]['xG_for'] += away_xg
        team_stats[away]['xG_against'] += home_xg
        team_stats[away]['last_5_xG'].append(away_xg)

        team_stats[home]['shots_for'] += home_shots
        team_stats[home]['sot_for'] += min(home_shots, int(home_shot_goals + 0.3 * (home_shots - home_shot_goals)))
        team_stats[away]['shots_for'] += away_shots
        team_stats[away]['sot_for'] += min(away_shots, int(away_shot_goals + 0.3 * (away_shots - away_shot_goals)))

    team_stats = dict(team_stats)

    # Attach prior_stats from previous season
    sorted_sids = sorted(SEASON_ID_MAP.keys())
    sid_idx = sorted_sids.index(season_id) if season_id in sorted_sids else -1
    if sid_idx > 0:
        prior_sid = sorted_sids[sid_idx - 1]
        prior_cum = build_season_cumulative_stats(raw_events_df, matches_summary_df, prior_sid)
        for team in team_stats:
            team_stats[team]['prior_stats'] = prior_cum.get(team)
    else:
        for team in team_stats:
            team_stats[team]['prior_stats'] = None

    # Cache
    os.makedirs(STATS_CACHE_DIR, exist_ok=True)
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(team_stats, f)
    except Exception:
        pass

    return team_stats


# --- NEW FUNCTION: Plot Team Strength Scatter ---
def plot_team_strength(stats_df, teams_to_include=None, league="Liga 3", season="2025/26", icon_zoom=0.25): # <-- ADDED icon_zoom
    """Generates the Matplotlib figure for the team strength scatter plot."""

    if stats_df.empty or 'Attacking Strength' not in stats_df.columns or 'Defending Strength' not in stats_df.columns:
         fig, ax = plt.subplots(figsize=(10, 8)); fig.set_facecolor('#f5f1e9'); ax.set_facecolor('#f5f1e9')
         ax.text(0.5, 0.5, 'Team strength data unavailable.', ha='center', va='center', fontsize=14); ax.axis('off'); return fig

    fig, ax = plt.subplots(figsize=(16, 12)); fig.set_facecolor('#f5f1e9'); ax.set_facecolor('#f5f1e9'); ax.invert_yaxis()
    x_min, x_max = stats_df['Attacking Strength'].min(), stats_df['Attacking Strength'].max()
    y_min, y_max = stats_df['Defending Strength'].min(), stats_df['Defending Strength'].max()
    x_padding = (x_max - x_min) * 0.1; y_padding = (y_max - y_min) * 0.1
    ax.set_xlim(x_min - x_padding, x_max + x_padding); ax.set_ylim(y_max + y_padding, y_min - y_padding) # Inverted Y
    # --- (NEW) Diagonal Lines Logic ---
    # Get the final plot limits *after* setting them
    x_min, x_max = ax.get_xlim()
    y_max, y_min = ax.get_ylim() # Remember y-axis is inverted (y_max < y_min)

    # Calculate the 'c' value (c = x - y) for all four corners of the plot
    c_top_left = x_min - y_max
    c_top_right = x_max - y_max
    c_bottom_left = x_min - y_min
    c_bottom_right = x_max - y_min

    # Find the minimum and maximum 'c' values, rounded to nearest 0.1
    min_c = np.floor(min([c_top_left, c_top_right, c_bottom_left, c_bottom_right]) * 10) / 10
    max_c = np.ceil(max([c_top_left, c_top_right, c_bottom_left, c_bottom_right]) * 10) / 10

    # Draw lines for every 'c' value in the calculated range
    for c in np.arange(min_c, max_c + 0.1, 0.1):
        # Use axline to draw an infinite line with slope 1 passing through (0, -c)
        # Matplotlib will automatically clip it to the plot boundaries
        ax.axline((0, -c), slope=1, color='lightgray', linestyle=':', zorder=1, lw=1)
    # --- (END NEW) Diagonal Lines Logic ---


    stats_df_to_plot = stats_df
    if teams_to_include:
        valid_teams = [t for t in teams_to_include if t in stats_df.index]
        stats_df_to_plot = stats_df.loc[valid_teams]

    texts = []; logos_plotted = 0; base_icon_path = "icons" # ASSUMES 'icons' FOLDER
    for team_name, row in stats_df_to_plot.iterrows():
        safe_team_name = team_name.replace('/', '_').replace('\\', '_')
        logo_path = os.path.join(base_icon_path, f"{safe_team_name}.png")
        try:
            if os.path.exists(logo_path):
                 img = Image.open(logo_path); # --- USE THE NEW PARAMETER HERE ---
                 imagebox = OffsetImage(img, zoom=icon_zoom) # <-- USE icon_zoom
                 ab = AnnotationBbox(imagebox, (row['Attacking Strength'], row['Defending Strength']), frameon=False, zorder=2)
                 ax.add_artist(ab)
                 logos_plotted +=1
            else: texts.append(ax.text(row['Attacking Strength'], row['Defending Strength'], team_name, zorder=3, fontsize=9))
        except Exception as e: print(f"Error loading logo for {team_name}: {e}. Using text."); texts.append(ax.text(row['Attacking Strength'], row['Defending Strength'], team_name, zorder=3, fontsize=9))
    if texts: adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    if logos_plotted == 0 and not texts: ax.scatter(stats_df_to_plot['Attacking Strength'], stats_df_to_plot['Defending Strength'], s=50, zorder=2)

    report_date = datetime.date.today().strftime("%Y-%m-%d")
    ax.set_title(f'Team Strength Scatterplot | {league}, {season} (As of: {report_date})', fontsize=18, weight='bold')
    ax.set_xlabel('Attacking Strength (30% NP Goals, 70% NPxG)', fontsize=12)
    ax.set_ylabel('Defending Strength (30% NP Goals Against, 70% NPxG Against)', fontsize=12)
    #ax.grid(True, linestyle='--', alpha=0.5); plt.tight_layout(); return fig

# app.py (Add this new function)

# --- NEW FUNCTION: Plot Custom Scatter Plot ---
def plot_custom_scatter(stats_df, x_metric, y_metric, invert_x=False, invert_y=False, league="Liga 3", season="2025/26"):
    """Generates a dynamic Matplotlib scatter plot with logos."""

    # Ensure the selected metrics exist in the DataFrame
    if x_metric not in stats_df.columns or y_metric not in stats_df.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.set_facecolor('#f5f1e9'); ax.set_facecolor('#f5f1e9')
        ax.text(0.5, 0.5, f"Error: Metric not found.\nCheck data processing script.", ha='center', va='center', fontsize=12, color='red')
        ax.axis('off'); return fig

    fig, ax = plt.subplots(figsize=(16, 12))
    fig.set_facecolor('#f5f1e9')
    ax.set_facecolor('#f5f1e9')

    x_data = stats_df[x_metric]
    y_data = stats_df[y_metric]

    # --- 1. Set Axis Limits & Padding ---
    x_min, x_max = x_data.min(), x_data.max()
    y_min, y_max = y_data.min(), y_data.max()
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1

    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # --- 2. Invert Axis (if user checked the box) ---
    if invert_x:
        ax.set_xlim(x_max + x_padding, x_min - x_padding)
    if invert_y:
        ax.set_ylim(y_max + y_padding, y_min - y_padding)
        
    # --- 3. Add Mean Quadrant Lines for Context ---
    x_mean = x_data.mean()
    y_mean = y_data.mean()
    ax.axhline(y_mean, color='gray', linestyle='--', lw=1, zorder=1)
    ax.axvline(x_mean, color='gray', linestyle='--', lw=1, zorder=1)

    # --- 4. Plot Logos (re-using logic from plot_team_strength) ---
    stats_df_to_plot = stats_df.copy()
    texts = []; logos_plotted = 0; base_icon_path = "icons" 

    for team_name, row in stats_df_to_plot.iterrows():
        safe_team_name = team_name.replace('/', '_').replace('\\', '_')
        logo_path = os.path.join(base_icon_path, f"{safe_team_name}.png")
        try:
            if os.path.exists(logo_path):
                 img = Image.open(logo_path)
                 # Use the increased zoom factor
                 imagebox = OffsetImage(img, zoom=0.25) 
                 # Plot using the dynamic x_metric and y_metric
                 ab = AnnotationBbox(imagebox, (row[x_metric], row[y_metric]), frameon=False, zorder=2)
                 ax.add_artist(ab)
                 logos_plotted +=1
            else:
                 texts.append(ax.text(row[x_metric], row[y_metric], team_name, zorder=3, fontsize=9))
        except Exception as e:
            print(f"Error loading logo for {team_name}: {e}. Using text.")
            texts.append(ax.text(row[x_metric], row[y_metric], team_name, zorder=3, fontsize=9))
    
    if texts:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    if logos_plotted == 0 and not texts: 
         ax.scatter(stats_df_to_plot[x_metric], stats_df_to_plot[y_metric], s=50, zorder=2)

    # --- 5. Styling ---
    report_date = datetime.date.today().strftime("%Y-%m-%d")
    ax.set_title(f'League Scatterplot | {league}, {season} (As of: {report_date})', fontsize=18, weight='bold')
    ax.set_xlabel(x_metric, fontsize=12) # Dynamic X Label
    ax.set_ylabel(y_metric, fontsize=12) # Dynamic Y Label

    plt.tight_layout()
    return fig

# ... after plot_custom_scatter ...

def plot_xg_flowchart(match_events_df, match_info):
    """
    Generates a Matplotlib figure for the match xG flowchart.
    Includes markers for goals (regular, penalty, own) and red cards.
    """
    
    # 1. Get team names and set colors
    home_team = match_info.get('homeTeamName', 'Home')
    away_team = match_info.get('awayTeamName', 'Away')
    home_color = '#0077b6' # Blue
    away_color = '#e63946' # Red
    
    # 2. Filter for relevant events (shots, penalties, own goals, and red cards)
    # --- UPDATED: Include 'own_goal' type ---
    events_df = match_events_df[match_events_df['type.primary'].isin(['shot', 'penalty', 'own_goal'])].copy()
    events_df = events_df[['minute', 'team.name', 'shot.xg', 'shot.isGoal', 'type.primary']]
    
    reds_df = match_events_df[match_events_df.get('infraction.redCard') == True].copy()
    if not reds_df.empty:
        reds_df = reds_df[['minute', 'team.name']]
        reds_df['isRedCard'] = True
    else:
        reds_df = pd.DataFrame(columns=['minute', 'team.name', 'isRedCard'])
    
    # 3. Combine and sort
    # --- UPDATED: Use events_df ---
    df = pd.concat([events_df, reds_df]).sort_values(by='minute')
    
    # 4. Clean NaNs and identify event types
    df['shot.xg'] = pd.to_numeric(df['shot.xg'], errors='coerce').fillna(0)
    df['isRedCard'] = df['isRedCard'].fillna(False)
    
    # --- UPDATED: Identify Goal Types (using type.primary) ---
    df['shot.isGoal'] = df['shot.isGoal'].fillna(False) # For regular shots/penalties
    df['isPenalty'] = df['type.primary'] == 'penalty'
    df['isOwnGoal'] = df['type.primary'] == 'own_goal' # <-- THE FIX
    
    # A 'Regular Goal' is a scored shot, but not a penalty
    # (An own goal will have shot.isGoal=False, so it's excluded)
    df['isRegularGoal'] = df['shot.isGoal'] & (df['type.primary'] == 'shot')
    
    # A "Goal" for plotting is a scored shot/penalty OR an own goal
    df['isGoal'] = df['shot.isGoal'] | df['isOwnGoal']
    
    # 5. Create xG columns per team
    df['home_xG'] = np.where(df['team.name'] == home_team, df['shot.xg'], 0)
    df['away_xG'] = np.where(df['team.name'] == away_team, df['shot.xg'], 0)
    
    # 6. Add start row
    start_row = pd.DataFrame([{
        'minute': 0, 'home_xG': 0, 'away_xG': 0, 
        'shot.isGoal': False, 'isRedCard': False, 'team.name': 'Start',
        'isPenalty': False, 'isOwnGoal': False, 'isRegularGoal': False, 'isGoal': False
    }])
    df = pd.concat([start_row, df]).sort_values(by='minute')
    
    # 7. Calculate cumulative xG
    df['home_xG_cum'] = df['home_xG'].cumsum()
    df['away_xG_cum'] = df['away_xG'].cumsum()

    # 8. Get max minute for plot limit
    max_minute = df['minute'].max()
    if max_minute < 90: max_minute = 90
    
    # 9. Plotting
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.set_facecolor('#f5f1e9')
    ax.set_facecolor('#f5f1e9')
    
    ax.step(df['minute'], df['home_xG_cum'], label=home_team, color=home_color, where='post', linewidth=2.5)
    ax.step(df['minute'], df['away_xG_cum'], label=away_team, color=away_color, where='post', linewidth=2.5)
    
    # 10. Add Goal and Red Card Markers (This logic remains the same, but now uses the corrected types)
    goals_df = df[df['isGoal'] == True]
    for _, row in goals_df.iterrows():
        if row['isRegularGoal']:
            label = f"Goal ({row['minute']}')"
            marker_size = 250
            marker_shape = 'o'
        elif row['isPenalty'] and row['shot.isGoal']: # Check if penalty was scored
            label = f"Penalty Goal ({row['minute']}')"
            marker_size = 350
            marker_shape = 'X'
        elif row['isOwnGoal']:
            label = f"Own Goal ({row['minute']}')"
            marker_size = 250
            marker_shape = 's'
        else:
            continue # Don't plot missed penalties as "goals"
            
        if row['team.name'] == home_team:
            ax.scatter(row['minute'], row['home_xG_cum'], c=home_color, s=marker_size, marker=marker_shape, 
                       edgecolor='black', zorder=5, label=label)
        else:
            ax.scatter(row['minute'], row['away_xG_cum'], c=away_color, s=marker_size, marker=marker_shape, 
                       edgecolor='black', zorder=5, label=label)

    reds_df_plot = df[df['isRedCard'] == True]
    for _, row in reds_df_plot.iterrows():
        label = f"Red Card ({row['minute']}')"
        y_val = row['home_xG_cum'] if row['team.name'] == home_team else row['away_xG_cum']
        ax.scatter(row['minute'], y_val, c='red', s=150, marker='D', 
                   edgecolor='black', linewidth=1.5, zorder=5, label=label)

    # 11. Styling
    ax.set_xlabel('Minute', fontsize=12)
    ax.set_ylabel('Cumulative xG', fontsize=12)
    ax.set_title(f"xG Flowchart: {home_team} vs {away_team}\nScore: {match_info.get('score', '?-?')}", fontsize=16, weight='bold')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', frameon=False)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(0, max_minute + 2)
    ax.set_ylim(0)

    plt.tight_layout()
    return fig

@st.cache_data
def calculate_xg_history_data(_raw_events_df, _matches_summary_df):
    """
    Aggregates xG For and Against for every team for every match.
    (Previously named calculate_rolling_xg_data)
    """
    # ... (Logic is identical to your previous function, just renamed for clarity) ...
    print("Calculating xG history data...") 
    all_team_matches = []
    
    shots_df = _raw_events_df[_raw_events_df['type.primary'].isin(['shot', 'penalty'])].copy()
    shots_df['shot.xg'] = pd.to_numeric(shots_df['shot.xg'], errors='coerce').fillna(0)
    xg_by_match_team = shots_df.groupby(['matchId', 'team.name'])['shot.xg'].sum()

    for _, match in _matches_summary_df.iterrows():
        matchId = match.get('matchId')
        date = match.get('dateutc')
        season_marker = f"GW {match.get('gameweek', '1')}" 
        season_id = match.get('seasonId')
        round_id = match.get('roundId')
        home_team = match.get('homeTeamName')
        away_team = match.get('awayTeamName')

        if not all([matchId, date, home_team, away_team, season_id]):
            continue

        home_xg_for = xg_by_match_team.get((matchId, home_team), 0)
        away_xg_for = xg_by_match_team.get((matchId, away_team), 0)

        all_team_matches.append({
            'date': date,
            'season_marker': season_marker,
            'seasonId': season_id,
            'roundId': round_id,
            'teamName': home_team,
            'xG_For': home_xg_for,
            'xG_Against': away_xg_for
        })
        all_team_matches.append({
            'date': date,
            'season_marker': season_marker,
            'seasonId': season_id,
            'roundId': round_id,
            'teamName': away_team,
            'xG_For': away_xg_for,
            'xG_Against': home_xg_for
        })
    
    if not all_team_matches:
        print("Warning: No matches for xG history.")
        return pd.DataFrame()

    result_df = pd.DataFrame(all_team_matches)
    result_df['date'] = pd.to_datetime(result_df['date'])
    result_df = result_df.sort_values(by='date')
    
    return result_df

def plot_match_xg_history(all_matches_df, selected_team, rolling_window=5):
    """
    Plots 5-game rolling average xG with:
    1. Ordinal X-Axis (removes summer/off-season gaps visually).
    2. Conditional Fill (Blue for positive xG diff, Red for negative).
    """
    # 1. Filter for selected team
    team_df = all_matches_df[all_matches_df['teamName'] == selected_team].copy()
    if team_df.empty:
        fig, ax = plt.subplots(figsize=(14, 7)); ax.text(0.5, 0.5, 'No match data found.', ha='center'); return fig

    # 2. Filter for last 365 days
    today = pd.to_datetime(datetime.date.today())
    one_year_ago = today - pd.DateOffset(years=1)
    team_df = team_df[(team_df['date'] >= one_year_ago) & (team_df['date'] <= today)]

    # 3. Sort and Create "Ordinal" Axis (This removes the time gap)
    team_df = team_df.sort_values(by='date').reset_index(drop=True)
    team_df['match_seq'] = team_df.index

    if team_df.empty:
        fig, ax = plt.subplots(figsize=(14, 7)); ax.text(0.5, 0.5, 'No matches found in range.', ha='center'); return fig

    # 4. Compute rolling averages
    team_df = team_df.dropna(subset=['xG_For', 'xG_Against'])
    team_df['xG_For_Roll'] = team_df['xG_For'].rolling(window=rolling_window, min_periods=1).mean()
    team_df['xG_Against_Roll'] = team_df['xG_Against'].rolling(window=rolling_window, min_periods=1).mean()

    # 5. Plotting
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.set_facecolor('#f5f1e9')
    ax.set_facecolor('#f5f1e9')

    # Plot rolling average lines (smooth, no markers)
    ax.plot(team_df['match_seq'], team_df['xG_For_Roll'], label=f'{rolling_window}-Game Rolling xG For', color='#0077b6', lw=2.5, zorder=3)
    ax.plot(team_df['match_seq'], team_df['xG_Against_Roll'], label=f'{rolling_window}-Game Rolling xG Against', color='#e63946', lw=2.5, zorder=3)

    # Conditional shading on rolling averages
    ax.fill_between(
        team_df['match_seq'],
        team_df['xG_For_Roll'],
        team_df['xG_Against_Roll'],
        where=(team_df['xG_For_Roll'] >= team_df['xG_Against_Roll']),
        interpolate=True, color='#0077b6', alpha=0.2
    )
    ax.fill_between(
        team_df['match_seq'],
        team_df['xG_For_Roll'],
        team_df['xG_Against_Roll'],
        where=(team_df['xG_For_Roll'] < team_df['xG_Against_Roll']),
        interpolate=True, color='#e63946', alpha=0.2
    )

    # 6. Formatting the X-Axis to show DATES instead of numbers
    step = max(1, len(team_df) // 10)
    tick_indices = team_df['match_seq'][::step]
    tick_labels = team_df['date'].dt.strftime('%d/%m')[::step]

    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45)

    # Add Vertical Separator for New Season (if multiple seasons exist)
    if 'seasonId' in team_df.columns:
        season_changes = team_df['seasonId'].diff() != 0
        new_season_indices = team_df[season_changes].index[1:]

        ylim_top = ax.get_ylim()[1]
        for idx in new_season_indices:
            ax.axvline(idx - 0.5, color='gray', linestyle=':', lw=1.5, zorder=0)
            ax.text(idx - 0.5, ylim_top, ' New Season', ha='left', va='top', color='gray', rotation=90, fontsize=10)

    # Add Vertical Separator for Second Stage (Promotion / Maintenance)
    if 'roundId' in team_df.columns and 'seasonId' in team_df.columns:
        for sid in team_df['seasonId'].unique():
            season_mask = team_df['seasonId'] == sid
            season_slice = team_df[season_mask]
            if len(season_slice) < 2:
                continue
            season_all = all_matches_df[all_matches_df['seasonId'] == sid]
            round_counts = season_all.groupby('roundId').size()
            first_stage_round = round_counts.idxmax()
            stage_change = season_slice[season_slice['roundId'] != first_stage_round]
            if stage_change.empty:
                continue
            stage_idx = stage_change.index[0]
            second_round_id = stage_change.iloc[0]['roundId']
            second_stage_rounds = round_counts.drop(first_stage_round, errors='ignore')
            if len(second_stage_rounds) > 0:
                min_round = second_stage_rounds.idxmin()
                stage_label = 'Promotion Stage' if second_round_id == min_round else 'Maintenance Stage'
            else:
                stage_label = 'Second Stage'
            ylim_top = ax.get_ylim()[1]
            ax.axvline(stage_idx - 0.5, color='#6a0dad', linestyle='--', lw=1.5, zorder=0)
            ax.text(stage_idx - 0.5, ylim_top, f' {stage_label}', ha='left', va='top', color='#6a0dad', rotation=90, fontsize=10)

    ax.set_title(f"{selected_team} - {rolling_window}-Game Rolling xG", fontsize=16, weight='bold')
    ax.set_ylabel(f'{rolling_window}-Game Rolling Avg xG')
    ax.legend(loc='upper left', frameon=False)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    return fig

@st.cache_data
def calculate_expanded_team_stats(_all_match_data, _matches_summary_df, season_id=None):
    """
    Aggregates all per-match team stats from all_match_data into a
    season-long per-game average.
    season_id is used as a cache key so Streamlit recomputes when the season changes.
    """
    if season_id is not None:
        cache_path = os.path.join(STATS_CACHE_DIR, f'expanded_team_stats_{season_id}.parquet')
        if os.path.exists(cache_path):
            print(f"Loading cached expanded team stats for season {season_id}")
            return pd.read_parquet(cache_path)
    print("Calculating expanded team stats...") # Debug print
    
    # Use defaultdict for easy aggregation
    all_stats_agg = defaultdict(lambda: defaultdict(float))
    games_played = defaultdict(int)

    # Get team names from matches_summary
    team_name_map = {} # map matchId to (homeName, awayName)
    if 'homeTeamName' in _matches_summary_df.columns and 'awayTeamName' in _matches_summary_df.columns:
        for _, row in _matches_summary_df.iterrows():
            team_name_map[row['matchId']] = (row['homeTeamName'], row['awayTeamName'])
            
    if not team_name_map:
        st.error("Could not build team name map from matches_summary_df")
        return pd.DataFrame()

    # Loop through every match in the dataset
    for match_id, match_data in _all_match_data.items():
        if match_id not in team_name_map:
            continue # Skip if match not in summary
            
        home_team, away_team = team_name_map[match_id]
        
        # Increment games played
        games_played[home_team] += 1
        games_played[away_team] += 1
        
        # Check if 'team_stats' (the dict of DataFrames) exists
        if 'team_stats' in match_data and isinstance(match_data['team_stats'], dict):
            # Loop through each stat DataFrame (e.g., 'Passing', 'Defense')
            for category, df in match_data['team_stats'].items():
                
                # df format: index=metric names, columns=[homeTeamName, awayTeamName]
                if isinstance(df, pd.DataFrame) and not df.empty:
                    if home_team in df.columns and away_team in df.columns:
                        try:
                            # Convert all stats to numeric, coercing errors
                            df[home_team] = pd.to_numeric(df[home_team], errors='coerce').fillna(0)
                            df[away_team] = pd.to_numeric(df[away_team], errors='coerce').fillna(0)

                            # Loop through each metric row (metric name is the index)
                            for metric_name, row in df.iterrows():
                                # Add to the aggregate totals
                                all_stats_agg[home_team][metric_name] += row[home_team]
                                all_stats_agg[away_team][metric_name] += row[away_team]
                        except Exception as e:
                            print(f"Warning: Could not process df in match {match_id}. Error: {e}")
                    else:
                        print(f"Warning: Team columns '{home_team}' or '{away_team}' not in df for match {match_id}")

    # --- Convert aggregated totals to a DataFrame ---
    if not all_stats_agg:
        print("No stats aggregated.")
        return pd.DataFrame()
        
    stats_df = pd.DataFrame.from_dict(all_stats_agg, orient='index').fillna(0)
    
    # --- Normalize to Per-Game ---
    games_series = pd.Series(games_played, name='games')
    
    # Ensure we only divide teams that have games logged
    stats_df = stats_df.loc[games_series.index] 
    
    # Divide all stats by the number of games played
    stats_per_game_df = stats_df.div(games_series, axis=0)
    
    # Clean up any potential inf/-inf values
    stats_per_game_df.replace([np.inf, -np.inf], 0, inplace=True)
    
    print("✅ Expanded team stats calculated.")
    result = stats_per_game_df.fillna(0)
    if season_id is not None:
        os.makedirs(STATS_CACHE_DIR, exist_ok=True)
        try:
            result.to_parquet(os.path.join(STATS_CACHE_DIR, f'expanded_team_stats_{season_id}.parquet'))
        except Exception:
            pass
    return result

@st.cache_data(ttl=86400)
def calculate_set_piece_metrics(_events_df, season_id=None):
    """Calculate set piece xG and goals metrics for all teams.
    season_id is used as a cache key so Streamlit recomputes when the season changes."""
    _SP_CACHE_VERSION = 'v4'
    if season_id is not None:
        cache_path = os.path.join(STATS_CACHE_DIR, f'set_piece_metrics_{_SP_CACHE_VERSION}_{season_id}.parquet')
        if os.path.exists(cache_path):
            print(f"Loading cached set piece metrics for season {season_id}")
            return pd.read_parquet(cache_path)
    # Convert minute/second to total seconds
    events_df = _events_df.copy()
    events_df['total_seconds'] = events_df['minute'] * 60 + events_df['second']

    ATTACKING_THIRD = 66.67
    LONG_THROW_THRESHOLD = 25  # meters
    SHORT_CORNER_THRESHOLD = 20  # meters
    NEAR_BOX_X = 82  # Wyscout x-coord: penalty area starts ~84, this is "into or nearly into"
    AERIAL_DELIVERY_THRESHOLD = 20  # meters — passes shorter than this are "short" (ground play)
    PITCH_LENGTH_M, PITCH_WIDTH_M = 105.0, 68.0

    results = {}
    teams = events_df['team.name'].dropna().unique()

    for team in teams:
        results[team] = {
            'xG from Corners': 0, 'Goals from Corners': 0, 'Corners': 0, 'Short Corners': 0,
            'xG from Att Throw-ins': 0, 'Goals from Att Throw-ins': 0, 'Att Throw-ins': 0,
            'xG from Free Kicks': 0, 'Goals from Free Kicks': 0, 'Free Kicks Att Third': 0,
            'xG from Set Pieces': 0, 'Goals from Set Pieces': 0, 'Set Pieces Att Third': 0,
            'Long Throws': 0, 'xG from Long Throws': 0, 'Goals from Long Throws': 0,
            'Penalties': 0, 'Penalty Goals': 0,
            'First Contact Wins': 0, 'First Contact Deliveries': 0,
            'xG Conceded Corners': 0, 'Goals Conceded Corners': 0, 'Corners Against': 0,
            'xG Conceded Att Throw-ins': 0, 'Goals Conceded Att Throw-ins': 0,
            'xG Conceded Free Kicks': 0, 'Goals Conceded Free Kicks': 0,
            'xG Conceded Set Pieces': 0, 'Goals Conceded Set Pieces': 0,
        }

    # Process by match
    for match_id in events_df['matchId'].unique():
        match_events = events_df[events_df['matchId'] == match_id].sort_values('total_seconds').reset_index(drop=True)
        match_teams = match_events['team.name'].dropna().unique()
        shots = match_events[match_events['type.primary'] == 'shot'].copy()
        # Pre-extract team names array for fast first-contact lookups
        _me_teams = match_events['team.name'].values

        # Helper: check if delivering team wins first contact (next event with a team belongs to them)
        def _first_contact_won(event_idx, delivering_team):
            for j in range(event_idx + 1, min(event_idx + 6, len(match_events))):
                next_team = _me_teams[j]
                if pd.notna(next_team):
                    return next_team == delivering_team
            return False

        # Corners
        corners = match_events[match_events['type.primary'] == 'corner']
        for idx, corner in corners.iterrows():
            team = corner['team.name']
            if pd.isna(team) or team not in results:
                continue
            corner_time = corner['total_seconds']
            results[team]['Corners'] += 1

            # Detect short corners by pass distance
            pass_len = corner.get('pass.length', np.nan)
            if pd.isna(pass_len):
                sx, sy = corner.get('location.x', np.nan), corner.get('location.y', np.nan)
                ex, ey = corner.get('pass.endLocation.x', np.nan), corner.get('pass.endLocation.y', np.nan)
                if pd.notna(sx) and pd.notna(ex):
                    pass_len = np.sqrt(((ex - sx) * PITCH_LENGTH_M / 100.0)**2 + ((ey - sy) * PITCH_WIDTH_M / 100.0)**2)
            is_short = pd.notna(pass_len) and pass_len <= SHORT_CORNER_THRESHOLD
            if is_short:
                results[team]['Short Corners'] += 1

            # First contact tracking — only for aerial deliveries (not short corners)
            if not is_short:
                results[team]['First Contact Deliveries'] += 1
                if _first_contact_won(idx, team):
                    results[team]['First Contact Wins'] += 1

            team_shots = shots[(shots['team.name'] == team) &
                              (shots['total_seconds'] >= corner_time) &
                              (shots['total_seconds'] <= corner_time + 15)]

            xg = team_shots['shot.xg'].sum() if not team_shots.empty else 0
            goals = int(team_shots['shot.isGoal'].sum()) if not team_shots.empty else 0

            results[team]['xG from Corners'] += xg
            results[team]['Goals from Corners'] += goals
            results[team]['xG from Set Pieces'] += xg
            results[team]['Goals from Set Pieces'] += goals
            results[team]['Set Pieces Att Third'] += 1

            for opp_team in match_teams:
                if opp_team != team and opp_team in results:
                    results[opp_team]['Corners Against'] += 1
                    results[opp_team]['xG Conceded Corners'] += xg
                    results[opp_team]['Goals Conceded Corners'] += goals
                    results[opp_team]['xG Conceded Set Pieces'] += xg
                    results[opp_team]['Goals Conceded Set Pieces'] += goals

        # Attacking throw-ins
        throw_ins = match_events[match_events['type.primary'] == 'throw_in']
        att_throw_ins = throw_ins[throw_ins['location.x'] >= ATTACKING_THIRD]
        for _, throw in att_throw_ins.iterrows():
            team = throw['team.name']
            if pd.isna(team) or team not in results:
                continue
            throw_time = throw['total_seconds']
            results[team]['Att Throw-ins'] += 1
            results[team]['Set Pieces Att Third'] += 1

            team_shots = shots[(shots['team.name'] == team) &
                              (shots['total_seconds'] >= throw_time) &
                              (shots['total_seconds'] <= throw_time + 15)]

            xg = team_shots['shot.xg'].sum() if not team_shots.empty else 0
            goals = int(team_shots['shot.isGoal'].sum()) if not team_shots.empty else 0

            results[team]['xG from Att Throw-ins'] += xg
            results[team]['Goals from Att Throw-ins'] += goals
            results[team]['xG from Set Pieces'] += xg
            results[team]['Goals from Set Pieces'] += goals

            for opp_team in match_teams:
                if opp_team != team and opp_team in results:
                    results[opp_team]['xG Conceded Att Throw-ins'] += xg
                    results[opp_team]['Goals Conceded Att Throw-ins'] += goals
                    results[opp_team]['xG Conceded Set Pieces'] += xg
                    results[opp_team]['Goals Conceded Set Pieces'] += goals

        # Long throws: pass.length >= 25m AND landing into/near the penalty box
        long_throw_mask = (throw_ins['pass.length'] >= LONG_THROW_THRESHOLD)
        if 'pass.endLocation.x' in throw_ins.columns:
            long_throw_mask = long_throw_mask & (throw_ins['pass.endLocation.x'] >= NEAR_BOX_X)
        long_throws = throw_ins[long_throw_mask]
        for idx, throw in long_throws.iterrows():
            team = throw['team.name']
            if pd.isna(team) or team not in results:
                continue
            results[team]['Long Throws'] += 1
            # First contact tracking for long throws (always aerial)
            results[team]['First Contact Deliveries'] += 1
            if _first_contact_won(idx, team):
                results[team]['First Contact Wins'] += 1
            throw_time = throw['total_seconds']
            team_shots = shots[(shots['team.name'] == team) &
                              (shots['total_seconds'] >= throw_time) &
                              (shots['total_seconds'] <= throw_time + 15)]
            xg = team_shots['shot.xg'].sum() if not team_shots.empty else 0
            goals = int(team_shots['shot.isGoal'].sum()) if not team_shots.empty else 0
            results[team]['xG from Long Throws'] += xg
            results[team]['Goals from Long Throws'] += goals

        # Free kicks in attacking third
        free_kicks = match_events[match_events['type.primary'] == 'free_kick']
        att_free_kicks = free_kicks[free_kicks['location.x'] >= ATTACKING_THIRD]
        for idx, fk in att_free_kicks.iterrows():
            team = fk['team.name']
            if pd.isna(team) or team not in results:
                continue
            fk_time = fk['total_seconds']
            results[team]['Free Kicks Att Third'] += 1
            results[team]['Set Pieces Att Third'] += 1

            # First contact tracking — only for aerial FK deliveries (pass.length > threshold)
            fk_pass_len = fk.get('pass.length', np.nan)
            if pd.notna(fk_pass_len) and fk_pass_len > AERIAL_DELIVERY_THRESHOLD:
                results[team]['First Contact Deliveries'] += 1
                if _first_contact_won(idx, team):
                    results[team]['First Contact Wins'] += 1

            team_shots = shots[(shots['team.name'] == team) &
                              (shots['total_seconds'] >= fk_time) &
                              (shots['total_seconds'] <= fk_time + 15)]

            xg = team_shots['shot.xg'].sum() if not team_shots.empty else 0
            goals = int(team_shots['shot.isGoal'].sum()) if not team_shots.empty else 0

            results[team]['xG from Free Kicks'] += xg
            results[team]['Goals from Free Kicks'] += goals
            results[team]['xG from Set Pieces'] += xg
            results[team]['Goals from Set Pieces'] += goals

            for opp_team in match_teams:
                if opp_team != team and opp_team in results:
                    results[opp_team]['xG Conceded Free Kicks'] += xg
                    results[opp_team]['Goals Conceded Free Kicks'] += goals
                    results[opp_team]['xG Conceded Set Pieces'] += xg
                    results[opp_team]['Goals Conceded Set Pieces'] += goals

        # Penalties
        _sec_col = match_events.get('type.secondary')
        def _has_penalty_tag(s):
            if isinstance(s, (list, np.ndarray)):
                return 'penalty' in s
            return False
        penalty_shots = shots[shots['type.secondary'].apply(_has_penalty_tag)] if _sec_col is not None else pd.DataFrame()
        if penalty_shots.empty:
            penalty_shots = match_events[(match_events['type.primary'] == 'penalty')]
        for _, pen in penalty_shots.iterrows():
            team = pen['team.name']
            if pd.isna(team) or team not in results:
                continue
            results[team]['Penalties'] += 1
            if pen.get('shot.isGoal', False):
                results[team]['Penalty Goals'] += 1

    # Calculate per-event / rate metrics
    for team in results:
        r = results[team]
        r['xG per Corner'] = round(r['xG from Corners'] / r['Corners'], 3) if r['Corners'] > 0 else 0
        r['Goals per Corner'] = round(r['Goals from Corners'] / r['Corners'], 3) if r['Corners'] > 0 else 0
        r['Short Corner %'] = round(r['Short Corners'] / r['Corners'] * 100, 1) if r['Corners'] > 0 else 0
        r['Long Throw %'] = round(r['Long Throws'] / r['Att Throw-ins'] * 100, 1) if r['Att Throw-ins'] > 0 else 0
        r['xG per Long Throw'] = round(r['xG from Long Throws'] / r['Long Throws'], 3) if r['Long Throws'] > 0 else 0
        r['xG per FK Delivery'] = round(r['xG from Free Kicks'] / r['Free Kicks Att Third'], 3) if r['Free Kicks Att Third'] > 0 else 0
        r['Non-Pen SP Goals'] = r['Goals from Set Pieces']  # corners + throw-ins + FKs, excluding pens
        r['First Contact %'] = round(r['First Contact Wins'] / r['First Contact Deliveries'] * 100, 1) if r['First Contact Deliveries'] > 0 else 0
        r['xG per Att Set Piece'] = round(r['xG from Set Pieces'] / r['Set Pieces Att Third'], 3) if r['Set Pieces Att Third'] > 0 else 0
        r['xG Conceded per Corner'] = round(r['xG Conceded Corners'] / r['Corners Against'], 3) if r['Corners Against'] > 0 else 0
        r['Goals Conceded per Corner'] = round(r['Goals Conceded Corners'] / r['Corners Against'], 3) if r['Corners Against'] > 0 else 0

    result_df = pd.DataFrame.from_dict(results, orient='index')
    logger.info(f"✅ Set piece metrics calculated for {len(result_df)} teams")
    if season_id is not None:
        os.makedirs(STATS_CACHE_DIR, exist_ok=True)
        try:
            result_df.to_parquet(os.path.join(STATS_CACHE_DIR, f'set_piece_metrics_{_SP_CACHE_VERSION}_{season_id}.parquet'))
        except Exception:
            pass
    return result_df


# ==============================================================================
# 4B. SEASON REPORT — 7-dimension performance & style dot plots
# ==============================================================================
# Replicates the structure of Twelve.football's per-team season report:
# each dimension is a set of 5-8 metrics rendered as a horizontal dot-plot
# of every team in the league/stage, with the selected team highlighted.

# Per-dimension metric definitions. "direction" controls colour interpretation
# only; values are always plotted at their raw position. "label" is the
# display name; "key" matches a column produced by compute_team_season_metrics.
SEASON_REPORT_DIMENSIONS = {
    'Defence': {
        'subtitle': 'How well the team prevents opponents from creating chances',
        'metrics': [
            ('PPDA',                            'lower_better', 'PPDA'),
            ('def_intensity',                   'higher_better','Defensive intensity'),
            ('def_action_height',               'higher_better','Defensive action height (m)'),
            ('def_duels_won_pct',               'higher_better','Defensive duels won %'),
            ('opp_xg_per_match',                'lower_better', 'Opp. xG per match'),
            ('opp_goals_per_match',             'lower_better', 'Opp. Goals per match'),
        ],
    },
    'Defensive Transition': {
        'subtitle': 'How quickly the team re-organises after losing the ball',
        'metrics': [
            ('turnovers_per90',                 'lower_better', 'Turnovers / 90'),
            ('recoveries_per90',                'higher_better','Recoveries / 90'),
            ('final_third_recoveries_pct',      'higher_better','Final-third recoveries %'),
            ('recovery_line_height',            'higher_better','Recovery line height (m)'),
            ('counter_press_recoveries_per90',  'higher_better','Counter-press recoveries / 90'),
        ],
    },
    'Opposition Chance Creation': {
        'subtitle': 'Volume and quality of opposition chances faced',
        'metrics': [
            ('opp_shots_per90',                 'lower_better', 'Opp. Shots / 90'),
            ('opp_sot_per90',                   'lower_better', 'Opp. Shots on target / 90'),
            ('opp_xg_per90',                    'lower_better', 'Opp. xG / 90'),
            ('opp_xg_per_shot',                 'lower_better', 'Opp. xG per shot'),
            ('opp_box_touches_per90',           'lower_better', 'Opp. Box touches / 90'),
            ('opp_high_opp_shots_per90',        'lower_better', 'Opp. High-opportunity shots / 90'),
        ],
    },
    'Attacking Transition': {
        'subtitle': 'Threat generated immediately after winning the ball back',
        'metrics': [
            ('recoveries_per90',                'higher_better','Recoveries / 90'),
            ('recovery_line_height',            'higher_better','Recovery line height (m)'),
            ('final_third_recoveries_pct',      'higher_better','Final-third recoveries %'),
            ('counter_press_recoveries_per90',  'higher_better','Counter-press recoveries / 90'),
            ('long_passes_succ_per90',          'higher_better','Successful long passes / 90'),
        ],
    },
    'Attack': {
        'subtitle': 'Ball control and territorial dominance in the attacking half',
        'metrics': [
            ('ball_possession_pct',             'higher_better','Ball possession %'),
            ('field_tilt_pct',                  'higher_better','Field tilt %'),
            ('pass_tempo',                      'higher_better','Pass tempo (passes / min of poss.)'),
            ('long_ball_pct',                   'neutral',      'Long ball %'),
            ('passes_per90',                    'higher_better','Passes / 90'),
            ('progressive_passes_per90',        'higher_better','Progressive passes / 90'),
        ],
    },
    'Chance Creation': {
        'subtitle': 'Volume and quality of own chances generated',
        'metrics': [
            ('shots_per90',                     'higher_better','Shots / 90'),
            ('high_opp_shots_per90',            'higher_better','High-opportunity shots / 90'),
            ('xg_per90',                        'higher_better','xG / 90'),
            ('goals_per90',                     'higher_better','Goals / 90'),
            ('xg_per_shot',                     'higher_better','xG per shot'),
            ('box_touches_per90',               'higher_better','Box touches / 90'),
            ('sot_per90',                       'higher_better','Shots on target / 90'),
        ],
    },
    'Outcome': {
        'subtitle': 'Points won vs. expected; goal difference and field control',
        'metrics': [
            ('points_per_match',                'higher_better','Points / match'),
            ('xpoints_per_match',               'higher_better','xPoints / match'),
            ('points_minus_xpoints',            'higher_better','Points − xPoints'),
            ('goals_per_match',                 'higher_better','Goals / match'),
            ('opp_goals_per_match',             'lower_better', 'Opp. Goals / match'),
            ('xg_per_match',                    'higher_better','xG / match'),
            ('opp_xg_per_match',                'lower_better', 'Opp. xG / match'),
            ('field_tilt_pct',                  'higher_better','Field tilt %'),
        ],
    },
}


def _xpoints_from_xg(home_xg: float, away_xg: float, max_goals: int = 8) -> tuple:
    """Expected points for both sides from a single match xG pair, using a
    Poisson model. Returns (home_xPts, away_xPts).

    P(home_goals=h, away_goals=a) = Pois(home_xg, h) * Pois(away_xg, a).
    xPts = 3*P(win) + 1*P(draw)."""
    if home_xg < 0: home_xg = 0
    if away_xg < 0: away_xg = 0
    if home_xg == 0 and away_xg == 0:
        return (1.0, 1.0)
    from math import exp, factorial
    h_xpts = 0.0; a_xpts = 0.0
    for h in range(max_goals + 1):
        ph = (home_xg ** h) * exp(-home_xg) / factorial(h)
        for a in range(max_goals + 1):
            pa = (away_xg ** a) * exp(-away_xg) / factorial(a)
            p = ph * pa
            if h > a:
                h_xpts += 3 * p
            elif h < a:
                a_xpts += 3 * p
            else:
                h_xpts += p
                a_xpts += p
    return (h_xpts, a_xpts)


def _load_wyscout_overlay(season_ids):
    """Load Wyscout published per-match averages from team_advanced_stats
    for the given seasons. Returns a DataFrame indexed by team name, or
    None if Wyscout data is unavailable / matches no row.

    Values in team_advanced_stats are **per-match averages** for the
    season (e.g. shots=10.42 means 10.42 shots per match), which match the
    season-report's expected scale (per-90 ≈ per-match)."""
    df = load_team_advanced_stats()
    if df is None or df.empty:
        return None
    if season_ids is not None:
        sids = list(season_ids) if isinstance(season_ids, (list, tuple, set)) else [season_ids]
        df = df[df['seasonId'].isin(sids)]
    if df.empty:
        return None
    # For multi-season selections, average across seasons (each team plays
    # ~26 matches per season — close to equal weighting).
    agg = df.groupby('team_name').mean(numeric_only=True).reset_index()
    return agg.set_index('team_name')


# Map our season-report column → Wyscout's per-match column name.
# Values in team_advanced_stats are per-match averages, so per-match and
# per-90 columns map to the same Wyscout source.
_WYSCOUT_DIRECT_OVERLAY = {
    'PPDA':                  'ppda',
    'ball_possession_pct':   'possession_percent',
    'def_duels_won_pct':     'defensive_duels_won_pct',
    'goals_per_match':       'goals',
    'opp_goals_per_match':   'conceded_goals',
    'xg_per_match':          'xg',
    'opp_xg_per_match':      'xg_shot_against',
    'goals_per90':           'goals',
    'xg_per90':              'xg',
    'opp_xg_per90':          'xg_shot_against',
    'shots_per90':           'shots',
    'opp_shots_per90':       'shots_against',
    'passes_per90':          'passes',
    'box_touches_per90':     'touch_in_box',
}


@st.cache_data(ttl=86400, show_spinner=False)
def compute_team_season_metrics(_events_df, _matches_df, season_ids=None,
                                  use_wyscout=True, cache_key=None):
    """Build per-team aggregate metrics across all 7 season-report dimensions.

    Priority for each metric:
      1. Wyscout's published per-match value from team_advanced_stats
         (only when use_wyscout=True and the team-season exists there).
      2. Events-based formula matching Wyscout's documented method
         (PPDA's press-zone definition, etc.) — used when use_wyscout is
         off (e.g. stage filter active) or for metrics Wyscout doesn't
         publish (recovery line height, counter-press recoveries, …).

    cache_key: an arbitrary string that uniquely identifies the
    (competition, season, stage) combination — used to scope Streamlit's
    in-memory cache. The events/matches DataFrames are passed by reference;
    cache_key changes when the filter changes.

    Returns a DataFrame indexed by team name."""
    if _events_df is None or _events_df.empty or 'team.name' not in _events_df.columns:
        return pd.DataFrame()

    ev = _events_df.copy()
    matches = _matches_df.copy() if _matches_df is not None else pd.DataFrame()

    # ── Per-event flags ────────────────────────────────────────────────
    def _has_tag(row, tag):
        s = row
        return isinstance(s, (list, np.ndarray)) and tag in s

    sec = ev.get('type.secondary', pd.Series([[]]*len(ev)))
    def _tag(name):
        return sec.apply(lambda x: isinstance(x, (list, np.ndarray)) and name in x)
    has_recovery         = _tag('recovery')
    has_counter_press    = _tag('counterpressing_recovery')
    has_loss             = _tag('loss')
    has_long_pass        = _tag('long_pass')
    has_progressive_pass = _tag('progressive_pass')
    has_defensive_duel   = _tag('defensive_duel')
    has_sliding_tackle   = _tag('sliding_tackle')

    tp = ev['type.primary']
    is_pass        = tp == 'pass'
    is_shot        = tp == 'shot'  # excludes penalty (different type.primary)
    # Defensive actions (Wyscout-aligned): interceptions + clearances +
    # defensive duels + sliding tackles. Previously included ALL `duel`
    # events (offensive + aerial + loose-ball duels too) which inflated
    # the denominator and crashed PPDA to ~1.3.
    is_def_action  = (
        (tp == 'interception')
        | (tp == 'clearance')
        | has_defensive_duel
        | has_sliding_tackle
    )
    is_foul        = tp == 'infraction'
    is_goal        = ev.get('shot.isGoal', False).fillna(False).astype(bool)
    is_sot         = ev.get('shot.onTarget', False).fillna(False).astype(bool) & is_shot

    loc_x = pd.to_numeric(ev['location.x'], errors='coerce')
    loc_y = pd.to_numeric(ev['location.y'], errors='coerce')
    # In Wyscout coordinates, location.x = 0 at own goal, 100 at opponent
    # goal (longitudinal); location.y = lateral position.
    in_box = (loc_x >= 83) & (loc_y >= 21) & (loc_y <= 79)
    in_final_third = loc_x >= 66
    pass_acc = ev.get('pass.accurate', False).fillna(False).astype(bool)
    shot_xg = pd.to_numeric(ev.get('shot.xg'), errors='coerce').fillna(0)
    is_high_opp = is_shot & (shot_xg >= 0.25)

    # PPDA zone definitions (Wyscout / StatsBomb convention):
    #   numerator   = opp passes in the opponent's own defensive 60%
    #                 = opp events where their location.x ≤ 60
    #   denominator = our defensive actions in the opp's defensive 60%
    #                 = our events where our location.x ≥ 40
    # Both are encoded against the acting team's own coordinate system.
    is_press_pass = is_pass & (loc_x <= 60)         # used by opponent's view
    is_press_def  = is_def_action & (loc_x >= 40)   # used by our view

    # Defensive-duel won flag (reuses has_defensive_duel from above).
    _rec_pos = ev.get('groundDuel.recoveredPossession',
                       pd.Series(False, index=ev.index)).fillna(False).astype(bool) \
        if 'groundDuel.recoveredPossession' in ev.columns \
        else pd.Series(False, index=ev.index)
    _stp_prog = ev.get('groundDuel.stoppedProgress',
                        pd.Series(False, index=ev.index)).fillna(False).astype(bool) \
        if 'groundDuel.stoppedProgress' in ev.columns \
        else pd.Series(False, index=ev.index)
    has_def_duel_won = has_defensive_duel & (_rec_pos | _stp_prog)

    # Pre-compute conditional values for axis-filtered means so the agg
    # can use plain 'mean' rather than fragile lambdas.
    def_x_for_def_only      = loc_x.where(is_def_action)        # used for def action height
    recovery_x_for_rec_only = loc_x.where(has_recovery)         # used for recovery line height

    ev = ev.assign(
        _is_pass=is_pass, _is_shot=is_shot, _is_def=is_def_action, _is_foul=is_foul,
        _is_goal=is_goal, _is_sot=is_sot, _is_box_touch=in_box, _is_final_third=in_final_third,
        _is_recovery=has_recovery, _is_counterpress=has_counter_press,
        _is_loss=has_loss, _is_long_pass=has_long_pass, _is_progpass=has_progressive_pass,
        _is_long_pass_succ=has_long_pass & pass_acc,
        _is_high_opp=is_high_opp,
        _is_final_third_recovery=has_recovery & in_final_third,
        _is_press_pass=is_press_pass, _is_press_def=is_press_def,
        _is_def_duel=has_defensive_duel, _is_def_duel_won=has_def_duel_won,
        _pass_acc=pass_acc,
        _def_x_for_def=def_x_for_def_only,
        _rec_x_for_rec=recovery_x_for_rec_only,
        _loc_x=loc_x, _loc_y=loc_y, _shot_xg=shot_xg,
    )

    # Match minutes: max event timestamp per match, divided by 60.
    # Wyscout's `minute` is cumulative since match start (0 → 90+), so no
    # half-offset is needed. raw_events.parquet doesn't carry matchPeriod.
    ev['_t_s'] = (pd.to_numeric(ev['minute'], errors='coerce').fillna(0) * 60
                   + pd.to_numeric(ev['second'], errors='coerce').fillna(0))
    match_total_s = ev.groupby('matchId')['_t_s'].max().rename('match_total_s')

    # ── Per-(matchId, team) aggregates ────────────────────────────────
    g = ev.groupby(['matchId', 'team.name'], sort=False)
    per = g.agg(
        n_events=('_t_s', 'size'),
        n_passes=('_is_pass', 'sum'),
        n_pass_acc=('_pass_acc', 'sum'),
        n_long_pass=('_is_long_pass', 'sum'),
        n_long_pass_succ=('_is_long_pass_succ', 'sum'),
        n_progressive_pass=('_is_progpass', 'sum'),
        n_shots=('_is_shot', 'sum'),
        n_sot=('_is_sot', 'sum'),
        n_goals=('_is_goal', 'sum'),
        n_def_action=('_is_def', 'sum'),
        n_foul=('_is_foul', 'sum'),
        n_recovery=('_is_recovery', 'sum'),
        n_counter_press=('_is_counterpress', 'sum'),
        n_turnover=('_is_loss', 'sum'),
        n_box_touch=('_is_box_touch', 'sum'),
        n_final_third_touch=('_is_final_third', 'sum'),
        n_final_third_recovery=('_is_final_third_recovery', 'sum'),
        n_high_opp_shot=('_is_high_opp', 'sum'),
        sum_xg=('_shot_xg', 'sum'),
        # PPDA zone counts
        n_press_pass=('_is_press_pass', 'sum'),
        n_press_def=('_is_press_def', 'sum'),
        # Defensive duels (vectorised — no lambdas)
        n_dd=('_is_def_duel', 'sum'),
        n_dd_won=('_is_def_duel_won', 'sum'),
        # Avg longitudinal location of defensive actions / recoveries
        # (NaN-respecting mean over the pre-masked _loc_x series)
        avg_def_x=('_def_x_for_def', 'mean'),
        avg_recovery_x=('_rec_x_for_rec', 'mean'),
    ).reset_index()
    per = per.merge(match_total_s, on='matchId', how='left')
    # Approximate per-match minutes-played for each team (use match total).
    per['match_minutes'] = per['match_total_s'] / 60.0

    # Opponent stats for each row: self-join on matchId, swap team.
    # Drop match-level shared columns from per_opp so the merge doesn't
    # double them (else pandas suffixes them _x/_y and downstream lookups
    # by `match_minutes` fail).
    _shared_match_cols = ('matchId', 'team.name', 'match_total_s', 'match_minutes')
    per_opp = per.drop(columns=['match_total_s', 'match_minutes']).rename(columns={
        'team.name': '_opp_name',
        **{c: f'opp_{c}' for c in per.columns if c not in _shared_match_cols}
    })
    merged = per.merge(per_opp, left_on='matchId', right_on='matchId', how='left')
    merged = merged[merged['team.name'] != merged['_opp_name']].copy()

    # Points & xPoints per (matchId, team) — derived from final scores in matches_df.
    if not matches.empty and 'score' in matches.columns:
        score_split = matches['score'].fillna('').astype(str).str.split('-', expand=True)
        matches['_home_goals'] = pd.to_numeric(score_split.get(0), errors='coerce')
        matches['_away_goals'] = pd.to_numeric(score_split.get(1), errors='coerce')

        # xG totals per match-team for xPoints
        per_xg = per.groupby(['matchId', 'team.name'])['sum_xg'].sum().reset_index()
        match_xg = per_xg.merge(matches[['matchId', 'homeTeamName', 'awayTeamName',
                                          '_home_goals', '_away_goals']],
                                 on='matchId', how='left')
        per_team_points: dict = {}
        per_team_xpoints: dict = {}
        for mid, mrow in matches.iterrows():
            mid_val = mrow['matchId']; hg = mrow['_home_goals']; ag = mrow['_away_goals']
            h_name = mrow['homeTeamName']; a_name = mrow['awayTeamName']
            if pd.isna(hg) or pd.isna(ag) or pd.isna(h_name) or pd.isna(a_name):
                continue
            # Points
            if hg > ag:   pts_h, pts_a = 3, 0
            elif hg < ag: pts_h, pts_a = 0, 3
            else:         pts_h, pts_a = 1, 1
            per_team_points.setdefault(mid_val, {})[h_name] = pts_h
            per_team_points.setdefault(mid_val, {})[a_name] = pts_a
            # xPoints
            h_xg = float(match_xg[(match_xg['matchId']==mid_val) & (match_xg['team.name']==h_name)]['sum_xg'].sum())
            a_xg = float(match_xg[(match_xg['matchId']==mid_val) & (match_xg['team.name']==a_name)]['sum_xg'].sum())
            h_xp, a_xp = _xpoints_from_xg(h_xg, a_xg)
            per_team_xpoints.setdefault(mid_val, {})[h_name] = h_xp
            per_team_xpoints.setdefault(mid_val, {})[a_name] = a_xp

        merged['points']  = merged.apply(
            lambda r: per_team_points.get(r['matchId'], {}).get(r['team.name'], np.nan), axis=1)
        merged['xpoints'] = merged.apply(
            lambda r: per_team_xpoints.get(r['matchId'], {}).get(r['team.name'], np.nan), axis=1)
    else:
        merged['points'] = np.nan; merged['xpoints'] = np.nan

    # ── Per-team aggregation: mean across that team's matches ──────────
    teams = sorted(merged['team.name'].dropna().unique())
    out_rows: list = []
    for t in teams:
        sub = merged[merged['team.name'] == t]
        if sub.empty:
            continue
        n_matches = sub['matchId'].nunique()
        total_minutes = float(sub.drop_duplicates('matchId')['match_minutes'].sum()) or 1.0
        per90 = lambda col: float(sub[col].sum()) * 90.0 / total_minutes
        per90_opp = lambda col: float(sub[col].sum()) * 90.0 / total_minutes  # same denom (per match total mins)

        # PPDA (Wyscout / StatsBomb convention):
        #   numerator   = opp passes in opp's own defensive 60% (opp's loc.x ≤ 60)
        #   denominator = our defensive actions in opp's defensive 60% (our loc.x ≥ 40)
        # Both are pre-flagged at the event level; we aggregate from the
        # self-joined opp row for the numerator.
        our_press_def = float(sub['n_press_def'].sum())
        opp_press_passes = float(sub['opp_n_press_pass'].sum())
        ppda = (opp_press_passes / our_press_def) if our_press_def > 0 else np.nan

        # Defensive intensity: our defensive actions / 90.
        def_intensity = per90('n_def_action')

        # Defensive duels won %
        n_dd = float(sub['n_dd'].sum())
        n_dd_won = float(sub['n_dd_won'].sum())
        dd_pct = (n_dd_won / n_dd * 100.0) if n_dd > 0 else np.nan

        # Defensive action height (m): avg LONGITUDINAL location of defensive
        # actions across matches × pitch length (105 m / Wyscout's 0-100 scale).
        avg_def_x_match = sub['avg_def_x'].dropna()
        def_action_height = float(avg_def_x_match.mean()) * 1.05 if not avg_def_x_match.empty else np.nan

        # Recovery line height (m)
        avg_recovery_x_match = sub['avg_recovery_x'].dropna()
        recovery_line_height = float(avg_recovery_x_match.mean()) * 1.05 if not avg_recovery_x_match.empty else np.nan

        # Final-third recoveries %
        total_rec = float(sub['n_recovery'].sum())
        final_third_rec = float(sub['n_final_third_recovery'].sum())
        ft_rec_pct = (final_third_rec / total_rec * 100.0) if total_rec > 0 else np.nan

        # Pass tempo: passes per minute when we have possession. We don't track
        # possession-time per team here, so approximate by passes / ball-possession-minutes.
        # Without true possession minutes, use passes / (match_minutes * possession_pct).
        # Final fallback: passes / match_minutes.
        ball_possession_pct = (float(sub['n_passes'].sum())
                                / max(float(sub['n_passes'].sum() + sub['opp_n_passes'].sum()), 1)) * 100.0
        poss_minutes = total_minutes * (ball_possession_pct / 100.0) if ball_possession_pct > 0 else total_minutes
        pass_tempo = float(sub['n_passes'].sum()) / max(poss_minutes, 1)

        # Field tilt %: % of attacking-third touches that are ours.
        our_ft = float(sub['n_final_third_touch'].sum())
        opp_ft = float(sub['opp_n_final_third_touch'].sum())
        field_tilt_pct = (our_ft / (our_ft + opp_ft) * 100.0) if (our_ft + opp_ft) > 0 else np.nan

        # Long ball %
        n_passes = float(sub['n_passes'].sum())
        long_ball_pct = (float(sub['n_long_pass'].sum()) / n_passes * 100.0) if n_passes > 0 else np.nan

        # Outcome metrics from points/xpoints
        pts_series = sub['points'].dropna()
        xpts_series = sub['xpoints'].dropna()
        points_per_match = float(pts_series.mean()) if not pts_series.empty else np.nan
        xpoints_per_match = float(xpts_series.mean()) if not xpts_series.empty else np.nan
        points_minus_xp = points_per_match - xpoints_per_match if not (np.isnan(points_per_match) or np.isnan(xpoints_per_match)) else np.nan

        goals_per_match = float(sub['n_goals'].sum()) / max(n_matches, 1)
        opp_goals_per_match = float(sub['opp_n_goals'].sum()) / max(n_matches, 1)
        xg_per_match = float(sub['sum_xg'].sum()) / max(n_matches, 1)
        opp_xg_per_match = float(sub['opp_sum_xg'].sum()) / max(n_matches, 1)

        # xG per shot
        ts = float(sub['n_shots'].sum())
        xg_per_shot = (float(sub['sum_xg'].sum()) / ts) if ts > 0 else np.nan
        opp_ts = float(sub['opp_n_shots'].sum())
        opp_xg_per_shot = (float(sub['opp_sum_xg'].sum()) / opp_ts) if opp_ts > 0 else np.nan

        out_rows.append({
            'team_name': t,
            'n_matches': n_matches,
            'total_minutes': total_minutes,
            # Defence
            'PPDA': ppda,
            'def_intensity': def_intensity,
            'def_action_height': def_action_height,
            'def_duels_won_pct': dd_pct,
            'opp_xg_per_match': opp_xg_per_match,
            'opp_goals_per_match': opp_goals_per_match,
            # Defensive Transition
            'turnovers_per90': per90('n_turnover'),
            'recoveries_per90': per90('n_recovery'),
            'final_third_recoveries_pct': ft_rec_pct,
            'recovery_line_height': recovery_line_height,
            'counter_press_recoveries_per90': per90('n_counter_press'),
            # Opposition Chance Creation
            'opp_shots_per90': per90_opp('opp_n_shots'),
            'opp_sot_per90': per90_opp('opp_n_sot'),
            'opp_xg_per90': per90_opp('opp_sum_xg'),
            'opp_xg_per_shot': opp_xg_per_shot,
            'opp_box_touches_per90': per90_opp('opp_n_box_touch'),
            'opp_high_opp_shots_per90': per90_opp('opp_n_high_opp_shot'),
            # Attack
            'ball_possession_pct': ball_possession_pct,
            'field_tilt_pct': field_tilt_pct,
            'pass_tempo': pass_tempo,
            'long_ball_pct': long_ball_pct,
            'passes_per90': per90('n_passes'),
            'progressive_passes_per90': per90('n_progressive_pass'),
            'long_passes_succ_per90': per90('n_long_pass_succ'),
            # Chance Creation
            'shots_per90': per90('n_shots'),
            'high_opp_shots_per90': per90('n_high_opp_shot'),
            'xg_per90': per90('sum_xg'),
            'goals_per90': per90('n_goals'),
            'xg_per_shot': xg_per_shot,
            'box_touches_per90': per90('n_box_touch'),
            'sot_per90': per90('n_sot'),
            # Outcome
            'points_per_match': points_per_match,
            'xpoints_per_match': xpoints_per_match,
            'points_minus_xpoints': points_minus_xp,
            'goals_per_match': goals_per_match,
            'xg_per_match': xg_per_match,
        })
    result = pd.DataFrame(out_rows).set_index('team_name')

    # ── Overlay with Wyscout's published per-match values ──────────────
    # When the user hasn't applied a stage filter (i.e. use_wyscout=True),
    # take Wyscout's own season-aggregated numbers as the source of truth
    # for any metric where they publish a direct match. Events-based
    # computations stay as a fallback for stage-filtered views and for
    # metrics Wyscout doesn't expose.
    if use_wyscout:
        wyscout = _load_wyscout_overlay(season_ids)
        if wyscout is not None:
            for team in result.index:
                if team not in wyscout.index:
                    continue
                wrow = wyscout.loc[team]
                # Direct one-to-one overlays.
                for our_col, w_col in _WYSCOUT_DIRECT_OVERLAY.items():
                    if w_col in wrow.index and our_col in result.columns:
                        v = wrow[w_col]
                        if pd.notna(v):
                            result.loc[team, our_col] = float(v)
                # Derived: xG per shot, opp xG per shot.
                if 'shots' in wrow.index and 'xg' in wrow.index \
                        and pd.notna(wrow['shots']) and float(wrow['shots']) > 0:
                    result.loc[team, 'xg_per_shot'] = float(wrow['xg']) / float(wrow['shots'])
                if 'shots_against' in wrow.index and 'xg_shot_against' in wrow.index \
                        and pd.notna(wrow['shots_against']) and float(wrow['shots_against']) > 0:
                    result.loc[team, 'opp_xg_per_shot'] = (
                        float(wrow['xg_shot_against']) / float(wrow['shots_against'])
                    )
    return result


def _fmt_metric_value(key: str, value) -> str:
    """Format metric value for display in the dot-plot."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if key in ('PPDA', 'xg_per_shot', 'opp_xg_per_shot', 'pass_tempo'):
        return f"{value:.2f}"
    if key in ('points_minus_xpoints',):
        return f"{value:+.2f}"
    if 'pct' in key:
        return f"{value:.0f}%"
    if 'per_match' in key or 'per90' in key or 'xg' in key or 'xpoints' in key or 'points' in key:
        return f"{value:.2f}"
    return f"{value:.1f}"


def render_dimension_dot_plot(team_metrics_df: pd.DataFrame, team_name: str,
                                dimension_name: str):
    """Interactive Plotly dot-plot for one of the 7 dimensions.

    Each metric becomes one row: every team plotted as a small green dot at
    its value position, the selected team plotted as a white hexagon. Hover
    over any dot to see the team name and its raw value."""
    import plotly.graph_objects as go

    dim = SEASON_REPORT_DIMENSIONS.get(dimension_name)
    if dim is None:
        return None

    if team_metrics_df is None or team_metrics_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data for the current selection",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color='#888'),
        )
        fig.update_layout(height=220, xaxis=dict(visible=False),
                           yaxis=dict(visible=False),
                           margin=dict(l=20, r=20, t=20, b=20))
        return fig

    metrics = [m for m in dim['metrics'] if m[0] in team_metrics_df.columns]
    n_metrics = len(metrics)
    if n_metrics == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No metrics available for this dimension",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color='#888'),
        )
        fig.update_layout(height=220, xaxis=dict(visible=False),
                           yaxis=dict(visible=False))
        return fig

    fig = go.Figure()
    y_labels: list = []

    for i, (key, direction, label) in enumerate(metrics):
        col = pd.to_numeric(team_metrics_df[key], errors='coerce').dropna()
        if col.empty:
            y_labels.append(label)
            continue
        vmin, vmax = float(col.min()), float(col.max())
        rng = (vmax - vmin) if (vmax - vmin) > 1e-9 else 1.0
        normed = (col - vmin) / rng

        teams = col.index.tolist()
        values = col.values.tolist()
        hover_texts = [
            f"<b>{t}</b><br>{label}: {_fmt_metric_value(key, v)}"
            for t, v in zip(teams, values)
        ]

        # Baseline rail
        fig.add_shape(
            type="line", x0=0, x1=1, y0=i, y1=i,
            line=dict(color="#bbb", width=0.5), layer="below",
        )

        # All teams as green dots (excluding selected)
        other_idx = [j for j, t in enumerate(teams) if t != team_name]
        if other_idx:
            fig.add_trace(go.Scatter(
                x=[float(normed.iloc[j]) for j in other_idx],
                y=[i] * len(other_idx),
                mode='markers',
                marker=dict(size=11, color='#3a8a3a', opacity=0.65,
                             line=dict(width=1, color='#1f4f1f')),
                hovertext=[hover_texts[j] for j in other_idx],
                hoverinfo='text',
                showlegend=False,
                name='',
            ))

        # Selected team as white hexagon
        if team_name in col.index:
            sel_idx = teams.index(team_name)
            fig.add_trace(go.Scatter(
                x=[float(normed.iloc[sel_idx])],
                y=[i],
                mode='markers',
                marker=dict(size=18, color='white', symbol='hexagon',
                             line=dict(width=2, color='#0a0a0a')),
                hovertext=[hover_texts[sel_idx]],
                hoverinfo='text',
                showlegend=False,
                name='',
            ))

        # Right-side annotation: selected team's raw value
        team_val_str = "—" if team_name not in col.index \
            else _fmt_metric_value(key, col.loc[team_name])
        fig.add_annotation(
            xref="x", yref="y", x=1.05, y=i,
            text=f"<b>{team_val_str}</b>",
            showarrow=False, font=dict(size=11, color='#0077b6'),
            xanchor='left', yanchor='middle',
        )

        y_labels.append(label)

    # Direction arrows beneath the bottom row
    fig.add_annotation(xref="x", yref="paper", x=0, y=-0.04,
                       text="← Worse", showarrow=False,
                       font=dict(size=10, color='#666'),
                       xanchor='left', yanchor='top')
    fig.add_annotation(xref="x", yref="paper", x=1, y=-0.04,
                       text="Better →", showarrow=False,
                       font=dict(size=10, color='#666'),
                       xanchor='right', yanchor='top')

    fig.update_layout(
        title=dict(
            text=(f"<b style='font-size:14px'>{dimension_name}</b>"
                  f"<br><span style='font-size:11px; color:#666'>"
                  f"{dim.get('subtitle','')}</span>"),
            x=0.0, xanchor='left', y=0.97, yanchor='top',
        ),
        xaxis=dict(
            visible=False,
            range=[-0.05, 1.15],
            fixedrange=True,
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(n_metrics)),
            ticktext=y_labels,
            autorange='reversed',
            showgrid=False, zeroline=False,
            fixedrange=True,
            tickfont=dict(size=11),
        ),
        height=max(230, 55 * n_metrics + 90),
        margin=dict(l=200, r=70, t=70, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='closest',
    )
    return fig


def render_season_report_section(team_events_df, team_matches_df, team_name,
                                   season_ids=None, stage=None, cache_key=None):
    """Render the 7-dimension season report for one team.

    Uses Wyscout's published per-match averages as the metric source when
    no stage filter is active; falls back to events-based formulas
    (Wyscout-aligned) otherwise."""
    use_wyscout = stage in (STAGE_ALL, None)
    with st.spinner("Computing season-report metrics for every team…"):
        # Normalise season_ids → tuple for stable cache key
        sids_key = tuple(sorted(season_ids)) if isinstance(season_ids, (list, tuple, set)) \
            else (int(season_ids),) if season_ids is not None else None
        team_metrics_df = compute_team_season_metrics(
            team_events_df, team_matches_df,
            season_ids=sids_key,
            use_wyscout=use_wyscout,
            cache_key=cache_key,
        )
    if team_metrics_df is None or team_metrics_df.empty:
        st.info("Not enough data to build the season report for this stage / season.")
        return
    if use_wyscout:
        st.caption(
            "Metric source: **Wyscout team_advanced_stats** (season averages) "
            "for every metric Wyscout publishes; events-based fallback for the rest. "
            "Apply a stage filter to switch to events-only for that subset of matches."
        )
    else:
        st.caption(
            "Stage filter active — metrics computed from match events using "
            "Wyscout-aligned formulas. Numbers may differ from Wyscout's "
            "season-aggregated totals."
        )

    if team_name not in team_metrics_df.index:
        st.info(f"No matches found for {team_name} in the current stage / season.")
        return

    # Tabs instead of a 2-column grid: only one Plotly figure mounts at a
    # time, eliminating the layout jitter caused by all 7 charts loading
    # simultaneously inside an expander.
    dim_names = list(SEASON_REPORT_DIMENSIONS.keys())
    tabs = st.tabs(dim_names)
    for tab, dim_name in zip(tabs, dim_names):
        with tab:
            fig = render_dimension_dot_plot(team_metrics_df, team_name, dim_name)
            if fig is not None:
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={'displayModeBar': False, 'responsive': True},
                    key=f"sr_dim_{team_name}_{dim_name}",
                )


# ==============================================================================
# 5. STREAMLIT APP UI
# ==============================================================================
st.markdown('<h1 style="text-align: center; color: #1a1a1a; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 0;">Atlético CP Analytics</h1>', unsafe_allow_html=True)

# --- Load Data ---
with st.spinner("Loading match data..."):
    raw_events_df, matches_summary_df, all_match_data, season_team_stats, player_minutes_data, match_lineups = load_data()

# --- Declare player_stats_with_scores_df globally for the app session ---
# This ensures it's accessible inside the plotting function
player_stats_with_scores_df = pd.DataFrame()


# --- Main App Logic ---
if raw_events_df is not None and matches_summary_df is not None and player_minutes_data is not None:
    # --- Initialize Session State ---
    if 'selected_player_id' not in st.session_state:
        st.session_state.selected_player_id = None
    if 'nav_to_profile' not in st.session_state:
        st.session_state.nav_to_profile = False
    if 'nav_season_id' not in st.session_state:
        st.session_state.nav_season_id = None
    if 'nav_has_season' not in st.session_state:
        st.session_state.nav_has_season = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Match Analysis'
    if 'radio_key_version' not in st.session_state:
        st.session_state.radio_key_version = 0
    if 'shadow_teams' not in st.session_state:
        st.session_state.shadow_teams = {}
    if 'player_profile_current_id' not in st.session_state:
        st.session_state.player_profile_current_id = None
    if 'player_profile_last_season' not in st.session_state:
        st.session_state.player_profile_last_season = None

    # --- Sidebar for Navigation ---
    st.sidebar.markdown('<div style="text-align: center; padding: 1rem 0 0.5rem 0;"><h2 style="color: #ffffff; font-size: 1.3rem; font-weight: 600; margin: 0;">Navigation</h2></div>', unsafe_allow_html=True)

    # Check if we should navigate to Player Profile
    if st.session_state.nav_to_profile:
        st.session_state.current_page = 'Player Profile'
        # Set radio value directly on the existing key instead of creating a new one
        current_radio_key = f"analysis_type_radio_{st.session_state.radio_key_version}"
        st.session_state[current_radio_key] = 'Player Profile'
        st.session_state.nav_to_profile = False

    ANALYSIS_OPTIONS = ('Match Analysis', 'Team Analysis', 'League Analysis', 'Player Profile', 'Player Comparison', 'Player Analysis', 'Match Predictor', 'Shadow Team', 'Opposition Report')

    if '--precompute' in sys.argv:
        import time as _t
        import gc as _gc
        print('[precompute] warming per-season stats caches...', flush=True)
        for _cid, _cfg in COMPETITIONS.items():
            # Each league's individual seasons PLUS its All-Seasons scope
            # (_sid=None). The league-aware scope key keeps the two single-league
            # All-Seasons caches from colliding on one 'None' file.
            for _sid in (list(_cfg['seasons'].keys()) + [None]):
                _label = 'ALL' if _sid is None else _sid
                _t0 = _t.time()
                _ev = _mins = None
                try:
                    _ev = get_filtered_events(raw_events_df, _sid, [_cid])
                    _mins = get_season_player_minutes(player_minutes_data, _sid, comp_ids=[_cid])
                    if _ev is None or len(_ev) == 0 or _mins is None or len(_mins) == 0:
                        print(f'[precompute] season {_label}: no data, skipping', flush=True); continue
                    load_and_score_player_stats(_ev, _mins, _sid, _sid, [_cid])
                    print(f'[precompute] season {_label}: cached in {_t.time()-_t0:.1f}s', flush=True)
                    # Bonus: warm the team-strength disk cache for this scope.
                    # team_strength is keyed by season_id only and disk-caches
                    # solely for non-None seasons, so only warm per-season scopes
                    # (the None scope would neither persist nor be league-keyed).
                    if _sid is not None:
                        try:
                            calculate_team_strength(_ev, matches_summary_df, season_id=_sid)
                        except Exception as _te:
                            print(f'[precompute] team_strength {_label} FAILED: {_te}', flush=True)
                except Exception as _e:
                    print(f'[precompute] season {_label} FAILED: {_e}', flush=True)
                finally:
                    # Release this scope's in-memory frames before the next one.
                    # The per-scope parquet is already on disk (load_and_score_
                    # player_stats wrote it), so every cached frame for this scope
                    # is now dead weight. Without this teardown the per-scope
                    # caches accumulate across all 9 scopes — most importantly the
                    # @st.cache_resource filtered-events frame (hundreds of MB per
                    # scope, never evicted within its 24h TTL) — and the heavy
                    # Camp/L3 All-Seasons scopes near the end OOM-kill the 7 GB
                    # hosted runner (surfaces as "##[error]The operation was
                    # canceled."). Runs fine on a 16 GB local box, hence cold CI
                    # only. Drop local refs, flush the Streamlit caches, then GC.
                    # The base data frames (raw_events_df / player_minutes_data /
                    # matches_summary_df) survive — they're held by these locals,
                    # not the cache, so clear()+gc don't reload them from disk.
                    _ev = _mins = None
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    _gc.collect()
        print('[precompute] done.', flush=True)
        sys.exit(0)

    # Boot-time pre-warm: warm every scope's in-memory caches in the background
    # (once per process) so the first post-deploy visitor doesn't eat the per-
    # scope cold build. No-op after the first rerun (cache_resource singleton).
    # Opt out with ACP_DISABLE_PREWARM=1.
    if os.environ.get('ACP_DISABLE_PREWARM') != '1':
        try:
            _prewarm_scope_caches(raw_events_df, player_minutes_data, matches_summary_df)
        except Exception as _pe:
            logger.warning(f"[prewarm] could not start: {_pe}")

    analysis_type = st.sidebar.radio(
        "Choose Analysis Type",
        ANALYSIS_OPTIONS,
        index=ANALYSIS_OPTIONS.index(st.session_state.current_page),
        key=f"analysis_type_radio_{st.session_state.radio_key_version}"
    )
    st.session_state.current_page = analysis_type

    # Engine freshness stamp — visible on every page (lesson from the
    # April→June staleness: nobody could see the data was 2 months old)
    try:
        _, _eng_meta_sb = load_player_engine()
        if _eng_meta_sb:
            st.sidebar.caption(
                f"⚙️ Engine {_eng_meta_sb.get('rating_version', '?')} · "
                f"data through {_eng_meta_sb.get('data_through', '?')}")
    except Exception:
        pass

    if analysis_type == 'Match Analysis':
        # --- League & Season Selector ---
        selected_comp_ids = league_selector("match_analysis")
        selected_season_id = season_selector("match_analysis", comp_ids=selected_comp_ids)
        active_season_ids = get_season_ids_for_selection(selected_season_id, selected_comp_ids)
        season_matches_df = filter_by_league(get_season_matches(matches_summary_df, active_season_ids), selected_comp_ids).copy()

        # --- Match Selection (Using correct column names) ---
        if 'dateutc' in season_matches_df.columns:
            season_matches_df['display_date'] = pd.to_datetime(season_matches_df['dateutc']).dt.strftime('%Y-%m-%d')
        else: season_matches_df['display_date'] = 'Unknown Date'

        # Create a display-ready gameweek column
        season_matches_df['gw_display'] = "GW " + season_matches_df.get('gameweek', pd.Series(dtype='str')).fillna('?').astype(str)

        # --- Determine stage labels (Promotion League vs Maintenance Stage) ---
        if 'roundId' in season_matches_df.columns and len(season_matches_df) > 0:
            # Group matches by roundId to find the first stage (most matches)
            round_counts = season_matches_df.groupby('roundId').size()
            first_stage_round = round_counts.idxmax()

            # Determine second stage rounds and their labels
            second_stage_rounds = round_counts.drop(first_stage_round, errors='ignore')

            def get_stage_label(row):
                if row['roundId'] == first_stage_round:
                    return row['gw_display']  # Regular season: no prefix
                else:
                    # Second stage: determine if Promotion or Maintenance
                    if len(second_stage_rounds) > 0:
                        min_round = second_stage_rounds.idxmin()
                        is_promotion = row['roundId'] == min_round
                        prefix = "[P] " if is_promotion else "[M] "
                    else:
                        prefix = "[S2] "
                    return prefix + row['gw_display']

            season_matches_df['gw_display_with_stage'] = season_matches_df.apply(get_stage_label, axis=1)
        else:
            season_matches_df['gw_display_with_stage'] = season_matches_df['gw_display']

        # Build the full display name using the new columns (GW: Teams (Score) - Date)
        season_matches_df['display_name'] = season_matches_df['gw_display_with_stage'] + ": " + \
                                             season_matches_df.get('homeTeamName', '?').fillna('?') + " vs " + \
                                             season_matches_df.get('awayTeamName', '?').fillna('?') + \
                                             " (" + season_matches_df.get('score', '?-?').fillna('?-?') + ") - " + \
                                             season_matches_df['display_date']

        sort_key = 'dateutc' if 'dateutc' in season_matches_df.columns else 'matchId'
        # Sort descending to show newest matches first
        season_matches_df.sort_values(by=[sort_key, 'matchId'], inplace=True, ascending=False, na_position='last')

        selected_match_display = st.sidebar.selectbox("Select a Match", season_matches_df['display_name'])
        matching_matches = season_matches_df[season_matches_df['display_name'] == selected_match_display]
        if matching_matches.empty:
            st.error("Selected match not found. Please refresh the page and try again.")
            st.stop()
        selected_match_info = matching_matches.iloc[0]
        selected_match_id = selected_match_info['matchId']

        # --- Display stage badge for the selected match ---
        if 'roundId' in season_matches_df.columns and len(season_matches_df) > 0:
            round_counts = season_matches_df.groupby('roundId').size()
            first_stage_round = round_counts.idxmax()
            second_stage_rounds = round_counts.drop(first_stage_round, errors='ignore')

            current_round_id = selected_match_info.get('roundId')
            if current_round_id == first_stage_round:
                badge_text = "Regular Season"
                badge_bg = "rgba(255,255,255,0.08)"
                badge_fg = "rgba(255,255,255,0.45)"
                badge_border = "rgba(255,255,255,0.12)"
            else:
                if len(second_stage_rounds) > 0:
                    min_round = second_stage_rounds.idxmin()
                    is_promotion = current_round_id == min_round
                    if is_promotion:
                        badge_text = "Promotion League"
                        badge_bg = "rgba(255,255,255,0.12)"
                        badge_fg = "#fff"
                        badge_border = "rgba(255,255,255,0.2)"
                    else:
                        badge_text = "Maintenance Stage"
                        badge_bg = "rgba(255,255,255,0.08)"
                        badge_fg = "#fff"
                        badge_border = "rgba(255,255,255,0.25)"
                else:
                    badge_text = "Second Stage"
                    badge_bg = "rgba(255,255,255,0.08)"
                    badge_fg = "rgba(255,255,255,0.45)"
                    badge_border = "rgba(255,255,255,0.12)"

            st.sidebar.markdown(
                f'<div style="background:{badge_bg}; color:{badge_fg}; border:1px solid {badge_border}; padding:7px 12px; border-radius:6px; text-align:center; font-weight:600; font-size:0.8rem; margin-top:8px; letter-spacing:0.3px;">{badge_text}</div>',
                unsafe_allow_html=True
            )

        st.header(f"Match Report: {selected_match_info['homeTeamName']} vs {selected_match_info['awayTeamName']}")
        
    

        match_data = all_match_data.get(selected_match_id)
        if match_data:
            st.subheader("Shot Maps")
            col1, col2 = st.columns(2)
            
            # --- Get the match events ONCE ---
            match_events_df = raw_events_df[raw_events_df['matchId'] == selected_match_id]

            with col1:
                fig_sm_h = create_match_shotmap(match_events_df, selected_match_info, selected_match_info['homeTeamName'])
                st.pyplot(fig_sm_h, use_container_width=True)
                plt.close(fig_sm_h)
            with col2:
                fig_sm_a = create_match_shotmap(match_events_df, selected_match_info, selected_match_info['awayTeamName'])
                st.pyplot(fig_sm_a, use_container_width=True)
                plt.close(fig_sm_a)

            # --- NEW: Shot Details Tables ---
            st.markdown("---") # Add a separator
            st.subheader("Shot Details")
            
            def get_shot_table(df, team_name):
                """Helper function to create the shot detail table."""
                shots = df[
                    (df.get('team.name') == team_name) & 
                    (df.get('type.primary').isin(['shot', 'penalty']))
                ].copy()
                
                if shots.empty:
                    return pd.DataFrame(columns=["#", "Shooter", "Minute", "Body Part", "xG", "PSxG"])

                # Select and rename columns
                # Use .get() for safety, in case a column is missing
                shots_table = pd.DataFrame()
                shots_table['Shooter'] = shots.get('player.name', 'N/A')
                shots_table['Minute'] = shots.get('minute', 0).astype(int)
                # Use .get() on the dictionary-like column 'shot.bodyPart'
                shots_table['Body Part'] = shots.get('shot.bodyPart', {}).apply(lambda x: x.get('name', 'unknown') if isinstance(x, dict) else x)
                shots_table['xG'] = shots.get('shot.xg', 0).fillna(0).round(2)
                shots_table['PSxG'] = shots.get('shot.postShotXg', 0).fillna(0).round(2)
                
                # Add the shot number (#)
                shots_table.reset_index(drop=True, inplace=True)
                shots_table.index = shots_table.index + 1
                shots_table.reset_index(inplace=True)
                shots_table = shots_table.rename(columns={'index': '#'})
                
                return shots_table.set_index('#')

            col1_table, col2_table = st.columns(2)
            with col1_table:
                st.markdown(f"**{selected_match_info['homeTeamName']}**")
                home_shots_table = get_shot_table(match_events_df, selected_match_info['homeTeamName'])
                st.dataframe(home_shots_table, column_config=auto_column_config(home_shots_table))

            with col2_table:
                st.markdown(f"**{selected_match_info['awayTeamName']}**")
                away_shots_table = get_shot_table(match_events_df, selected_match_info['awayTeamName'])
                st.dataframe(away_shots_table, column_config=auto_column_config(away_shots_table))
            # --- END NEW SECTION ---
            
            # --- xG Flowchart ---
            st.subheader("xG Flowchart")
            match_events_df = raw_events_df[raw_events_df['matchId'] == selected_match_id]
            if not match_events_df.empty:
                try:
                    fig_flowchart = plot_xg_flowchart(match_events_df, selected_match_info)
                    st.pyplot(fig_flowchart, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate xG flowchart: {e}")
            else:
                st.info("No event data found for flowchart.")

            st.subheader("Team Stats")
            if 'team_stats' in match_data and isinstance(match_data['team_stats'], dict) and match_data['team_stats']:
                for stat_category, df in match_data['team_stats'].items():
                    st.markdown(f"**{stat_category}**")
                    if isinstance(df, pd.DataFrame): st.dataframe(df, column_config=auto_column_config(df))
                    else: st.warning(f"Data for '{stat_category}' is not a DataFrame.")
            else: st.warning("Team stats data not found.")

            st.subheader("Player Stats")
            if 'player_stats' in match_data and isinstance(match_data['player_stats'], dict) and 'home' in match_data['player_stats'] and 'away' in match_data['player_stats']:
                st.markdown(f"**{selected_match_info['homeTeamName']}**")
                if isinstance(match_data['player_stats']['home'], pd.DataFrame): st.dataframe(match_data['player_stats']['home'])
                else: st.warning("Home player stats data not a DataFrame.")
                st.markdown(f"**{selected_match_info['awayTeamName']}**")
                if isinstance(match_data['player_stats']['away'], pd.DataFrame): st.dataframe(match_data['player_stats']['away'])
                else: st.warning("Away player stats data not a DataFrame.")
            else: st.warning("Player stats data not found.")

            # =============================================================
            # Tactical Analysis (Wyscout-style pitch visualizations)
            # =============================================================
            st.subheader("Tactical Analysis")
            home_team = selected_match_info['homeTeamName']
            away_team = selected_match_info['awayTeamName']

            # Get lineup/substitution data for this match (if available)
            match_lineup_data = match_lineups.get(selected_match_id, {}) if match_lineups else {}
            home_lineup = match_lineup_data.get(home_team)
            away_lineup = match_lineup_data.get(away_team)

            # 1. Average Player Positions
            st.markdown("**Average Player Positions**")
            col_ap1, col_ap2 = st.columns(2)
            with col_ap1:
                try:
                    fig_ap_h = pv.plot_average_positions(match_events_df, home_team,
                                                         match_lineup=home_lineup)
                    st.pyplot(fig_ap_h, use_container_width=True)
                    plt.close(fig_ap_h)
                except Exception as e:
                    st.caption(f"Could not render: {e}")
            with col_ap2:
                try:
                    fig_ap_a = pv.plot_average_positions(match_events_df, away_team,
                                                         match_lineup=away_lineup)
                    st.pyplot(fig_ap_a, use_container_width=True)
                    plt.close(fig_ap_a)
                except Exception as e:
                    st.caption(f"Could not render: {e}")

            # 2. Average Positions by Substitution Phase
            st.markdown(f"**{home_team} — Avg Positions by Phase**")
            try:
                fig_sp_h = pv.plot_avg_positions_by_subs(match_events_df, home_team,
                                                          match_lineup=home_lineup)
                st.pyplot(fig_sp_h, use_container_width=True)
                plt.close(fig_sp_h)
            except Exception as e:
                st.caption(f"Could not render: {e}")

            st.markdown(f"**{away_team} — Avg Positions by Phase**")
            try:
                fig_sp_a = pv.plot_avg_positions_by_subs(match_events_df, away_team,
                                                          match_lineup=away_lineup)
                st.pyplot(fig_sp_a, use_container_width=True)
                plt.close(fig_sp_a)
            except Exception as e:
                st.caption(f"Could not render: {e}")

            # 3. Passing Network
            st.markdown("**Passing Network**")
            col_pn1, col_pn2 = st.columns(2)
            with col_pn1:
                try:
                    fig_pn_h = pv.plot_passing_network(match_events_df, home_team)
                    st.pyplot(fig_pn_h, use_container_width=True)
                    plt.close(fig_pn_h)
                except Exception as e:
                    st.caption(f"Could not render: {e}")
            with col_pn2:
                try:
                    fig_pn_a = pv.plot_passing_network(match_events_df, away_team)
                    st.pyplot(fig_pn_a, use_container_width=True)
                    plt.close(fig_pn_a)
                except Exception as e:
                    st.caption(f"Could not render: {e}")

            # 4. Ball Recoveries & Losses
            st.markdown("**Ball Recoveries & Losses**")
            tac_team = st.selectbox(
                "Select team for recovery/loss maps",
                [home_team, away_team],
                key="tac_recovery_team",
            )
            col_rl1, col_rl2 = st.columns(2)
            with col_rl1:
                try:
                    fig_rec = pv.plot_recovery_map(match_events_df, tac_team)
                    st.pyplot(fig_rec, use_container_width=True)
                    plt.close(fig_rec)
                except Exception as e:
                    st.caption(f"Could not render: {e}")
            with col_rl2:
                try:
                    fig_loss = pv.plot_loss_map(match_events_df, tac_team)
                    st.pyplot(fig_loss, use_container_width=True)
                    plt.close(fig_loss)
                except Exception as e:
                    st.caption(f"Could not render: {e}")

            # 5. Defensive Duels
            st.markdown("**Defensive Duels**")
            col_dd1, col_dd2 = st.columns(2)
            with col_dd1:
                try:
                    fig_dd_h = pv.plot_defensive_duels_map(match_events_df, home_team)
                    st.pyplot(fig_dd_h, use_container_width=True)
                    plt.close(fig_dd_h)
                except Exception as e:
                    st.caption(f"Could not render: {e}")
            with col_dd2:
                try:
                    fig_dd_a = pv.plot_defensive_duels_map(match_events_df, away_team)
                    st.pyplot(fig_dd_a, use_container_width=True)
                    plt.close(fig_dd_a)
                except Exception as e:
                    st.caption(f"Could not render: {e}")

            # 6. Shot Assists + Dribbles in Final Third
            st.markdown("**Shot Assists & Dribbles in Final Third**")
            col_sa1, col_sa2 = st.columns(2)
            with col_sa1:
                try:
                    fig_sa_h = pv.plot_shot_assists_and_dribbles(match_events_df, home_team)
                    st.pyplot(fig_sa_h, use_container_width=True)
                    plt.close(fig_sa_h)
                except Exception as e:
                    st.caption(f"Could not render: {e}")
            with col_sa2:
                try:
                    fig_sa_a = pv.plot_shot_assists_and_dribbles(match_events_df, away_team)
                    st.pyplot(fig_sa_a, use_container_width=True)
                    plt.close(fig_sa_a)
                except Exception as e:
                    st.caption(f"Could not render: {e}")

        else:
             st.warning(f"No detailed match data found for Match ID {selected_match_id}.")


    elif analysis_type == 'Team Analysis':

        # --- League & Season Selector ---
        selected_comp_ids = league_selector("team_analysis")
        selected_season_id = season_selector("team_analysis", comp_ids=selected_comp_ids)
        active_season_ids = get_season_ids_for_selection(selected_season_id, selected_comp_ids)
        season_label = SEASON_ID_MAP.get(selected_season_id, "Unknown") if isinstance(selected_season_id, int) else "Unknown"
        # Stage selector (Regular / Promotion / Maintenance / Promotion playoff)
        selected_stage = stage_selector(
            "team_analysis",
            matches_summary_df,
            selected_comp_ids,
            active_season_ids,
        )
        team_events_df = get_filtered_events(raw_events_df, active_season_ids, selected_comp_ids)
        team_matches_df = filter_by_league(get_season_matches(matches_summary_df, active_season_ids), selected_comp_ids)
        # Apply stage filter — narrows the working set to the chosen stage's matches.
        team_events_df, team_matches_df = filter_by_stage(
            team_events_df, team_matches_df, matches_summary_df,
            selected_comp_ids, active_season_ids, selected_stage,
        )
        team_player_minutes_df = get_season_player_minutes(player_minutes_data, active_season_ids, comp_ids=selected_comp_ids)
        team_season_stats = get_season_team_stats(season_team_stats, active_season_ids, comp_ids=selected_comp_ids)

        all_teams_t = sorted(pd.concat([team_matches_df.get('homeTeamName'), team_matches_df.get('awayTeamName')]).dropna().unique())
        selected_team_t = st.sidebar.selectbox("Select a Team", all_teams_t, key="team_select_tab")
        _stage_suffix = "" if selected_stage in (STAGE_ALL, None) else f" — {selected_stage}"
        st.header(f"Team Report: {selected_team_t}{_stage_suffix}")
        if selected_stage not in (STAGE_ALL, None):
            _n_team_matches = team_matches_df[
                (team_matches_df['homeTeamName'] == selected_team_t)
                | (team_matches_df['awayTeamName'] == selected_team_t)
            ].shape[0]
            st.caption(
                f"Filtered to **{selected_stage}** — {len(team_matches_df)} matches in scope, "
                f"{_n_team_matches} for {selected_team_t}."
            )

        # Load player details for roster table
        player_details_df = load_player_details()

        # When a stage filter is active, force events-based radars so they
        # reflect only the matches in that stage (Wyscout's table is
        # season-aggregated and would otherwise leak full-season numbers).
        _stage_active = selected_stage not in (STAGE_ALL, None)
        _radar_cache_key = (active_season_ids if isinstance(active_season_ids, list) else selected_season_id)
        if _stage_active:
            _radar_cache_key = f"{_radar_cache_key}_{selected_stage}"
        stats_df_raw, stats_df_pct = calculate_all_team_radars_stats(
            team_events_df, team_matches_df,
            season_id=_radar_cache_key,
            force_events=_stage_active,
        )

        # Compute set piece radar data (all rate metrics — higher = better, no inversions)
        sp_df_raw = None
        sp_df_pct = None
        try:
            sp_df_raw = calculate_set_piece_metrics(team_events_df, season_id=active_season_ids if isinstance(active_season_ids, list) else selected_season_id)
            if sp_df_raw is not None and not sp_df_raw.empty:
                sp_df_pct = sp_df_raw.copy()
                for col in sp_df_pct.columns:
                    sp_df_pct[col] = sp_df_pct[col].rank(pct=True) * 100
        except Exception:
            pass

        league_label = get_league_label(selected_comp_ids)
        st.subheader(f"Team Style Radars (Percentile Ranks vs {league_label})")
        if selected_team_t in stats_df_raw.index and selected_team_t in stats_df_pct.index:
            offensive_params = ['Goals', 'xG', 'xG per Shot', 'Shots', 'Actions in Box', 'Passes into Box', 'Crosses', 'Dribbles']
            distribution_params = ['Passes', 'Progressive Passes', 'Directness', 'Ball Possession', 'Losses']
            defensive_params = ['Goals Against', 'xG Against', 'xG per Shot Against', 'Shots Against', 'Aerial Duel Win %', 'Defensive Duel Win %', 'Interceptions', 'Fouls', 'PPDA']
            set_piece_params = [
                'Corners', 'xG per Corner', 'Goals per Corner', 'Short Corner %',  # corner cluster
                'Long Throws', 'Long Throw %', 'xG per Long Throw',  # throw-in cluster
                'First Contact %', 'xG per FK Delivery', 'Penalties', 'Non-Pen SP Goals',  # general
            ]
            team_stats_raw = stats_df_raw.loc[selected_team_t]
            team_stats_pct = stats_df_pct.loc[selected_team_t]
            current_league = get_league_label(selected_comp_ids); current_season = season_label

            # Row 1: Offensive + Distribution
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown("**Offensive Radar**")
                valid_offensive_params = [p for p in offensive_params if p in team_stats_raw.index]
                if valid_offensive_params:
                     fig_off = plot_radar_chart(valid_offensive_params, team_stats_raw[valid_offensive_params].tolist(), team_stats_pct[valid_offensive_params].tolist(), selected_team_t, "Offensive Radar", '#e60000', league=current_league, season=current_season)
                     st.pyplot(fig_off, use_container_width=True)
                     plt.close(fig_off)
            with col_r2:
                st.markdown("**Distribution Radar**")
                valid_distribution_params = [p for p in distribution_params if p in team_stats_raw.index]
                if valid_distribution_params:
                     raw_dist_values = team_stats_raw[valid_distribution_params].tolist()
                     try: poss_index = valid_distribution_params.index('Ball Possession'); raw_dist_values[poss_index] = f"{raw_dist_values[poss_index]:.0f}%"
                     except ValueError: pass
                     fig_dist = plot_radar_chart(valid_distribution_params, raw_dist_values, team_stats_pct[valid_distribution_params].tolist(), selected_team_t, "Distribution Radar", '#0077b6', league=current_league, season=current_season)
                     st.pyplot(fig_dist, use_container_width=True)
                     plt.close(fig_dist)

            # Row 2: Defensive + Set Piece
            col_r3, col_r4 = st.columns(2)
            with col_r3:
                st.markdown("**Defensive Radar**")
                valid_defensive_params = [p for p in defensive_params if p in team_stats_raw.index]
                if valid_defensive_params:
                     raw_def_values = team_stats_raw[valid_defensive_params].tolist()
                     try: aerial_idx = valid_defensive_params.index('Aerial Duel Win %'); raw_def_values[aerial_idx] = f"{raw_def_values[aerial_idx]:.0f}%"
                     except ValueError: pass
                     try: def_idx = valid_defensive_params.index('Defensive Duel Win %'); raw_def_values[def_idx] = f"{raw_def_values[def_idx]:.0f}%"
                     except ValueError: pass
                     fig_def = plot_radar_chart(valid_defensive_params, raw_def_values, team_stats_pct[valid_defensive_params].tolist(), selected_team_t, "Defensive Radar", '#52A736', league=current_league, season=current_season)
                     st.pyplot(fig_def, use_container_width=True)
                     plt.close(fig_def)
            with col_r4:
                st.markdown("**Set Piece Radar**")
                if sp_df_raw is not None and not sp_df_raw.empty and selected_team_t in sp_df_raw.index:
                    sp_team_raw = sp_df_raw.loc[selected_team_t]
                    sp_team_pct = sp_df_pct.loc[selected_team_t]
                    valid_sp_params = [p for p in set_piece_params if p in sp_team_raw.index]
                    if valid_sp_params:
                        raw_sp_values = sp_team_raw[valid_sp_params].tolist()
                        # Format percentage params with % suffix
                        for _pct_name in ['Short Corner %', 'Long Throw %', 'First Contact %']:
                            try:
                                _idx = valid_sp_params.index(_pct_name)
                                raw_sp_values[_idx] = f"{raw_sp_values[_idx]:.0f}%"
                            except ValueError:
                                pass
                        fig_sp = plot_radar_chart(valid_sp_params, raw_sp_values, sp_team_pct[valid_sp_params].tolist(), selected_team_t, "Set Piece Radar", '#ff8c00', league=current_league, season=current_season)
                        st.pyplot(fig_sp, use_container_width=True)
                        plt.close(fig_sp)
                    else:
                        st.info("Set piece data not available.")
                else:
                    st.info("Set piece data not available for this team.")
        else:
            st.warning(f"Could not find calculated radar statistics for {selected_team_t}.")

        # ── Season Report (7-dimension dot plots, replicates the Twelve format) ─
        st.divider()
        with st.expander("📊 Season Report — performance across 7 dimensions",
                          expanded=False):
            st.caption(
                "Each row shows every team in the current league/stage as a green dot, "
                f"with **{selected_team_t}** highlighted as a white hexagon. "
                "Values are the team's raw per-match / per-90 numbers."
            )
            _sr_cache_key = (
                f"sr_{','.join(map(str, selected_comp_ids))}"
                f"_{active_season_ids if not isinstance(active_season_ids, list) else ','.join(map(str, active_season_ids))}"
                f"_{selected_stage or 'all'}"
            )
            render_season_report_section(
                team_events_df, team_matches_df, selected_team_t,
                season_ids=active_season_ids,
                stage=selected_stage,
                cache_key=_sr_cache_key,
            )

        # Primary Formation XI Graphic
        st.subheader("Primary Formation")
        primary_formation = get_team_primary_formation(team_events_df, selected_team_t)
        starting_xi = get_team_starting_xi(team_events_df, selected_team_t)

        col_xi1, col_xi2 = st.columns([1, 1])

        with col_xi1:
            if primary_formation and starting_xi:
                fig_xi = create_formation_graphic(primary_formation, starting_xi, selected_team_t)
                st.pyplot(fig_xi, use_container_width=True)
                plt.close(fig_xi)
            else:
                st.info("Formation data not available for this team.")

        with col_xi2:
            st.write(f"**Formation:** {primary_formation}")

            # Build roster table with unique players
            if starting_xi:
                # Get unique players (same player may appear at multiple positions)
                unique_players = {}
                for pos, player in starting_xi.items():
                    pid = player['id']
                    if pid not in unique_players:
                        unique_players[pid] = {'name': player['name'], 'positions': [pos], 'id': pid}
                    else:
                        unique_players[pid]['positions'].append(pos)

                # Build table data
                roster_data = []
                player_id_list = []
                for pid, pinfo in unique_players.items():
                    row = {'Player': pinfo['name'], 'Position': pinfo['positions'][0]}

                    # Get age and nationality from player_details
                    if pid in player_details_df.index:
                        details = player_details_df.loc[pid]
                        age = _calculate_age(details.get('birthDate'))
                        row['Age'] = int(age) if isinstance(age, (int, float)) and age != "N/A" else "N/A"
                        row['Nationality'] = details.get('passportArea', 'N/A')
                    else:
                        row['Age'] = "N/A"
                        row['Nationality'] = "N/A"

                    # Get minutes from team_player_minutes_df
                    player_mins = team_player_minutes_df[team_player_minutes_df['playerId'] == pid] if not team_player_minutes_df.empty else pd.DataFrame()
                    if not player_mins.empty:
                        row['Minutes'] = int(player_mins['totalMinutes'].values[0])
                    else:
                        row['Minutes'] = 0

                    roster_data.append(row)
                    player_id_list.append(pid)

                roster_df = pd.DataFrame(roster_data)
                roster_df = roster_df.sort_values('Minutes', ascending=False)
                # Reorder player_id_list to match sorted dataframe
                player_id_list = [player_id_list[i] for i in roster_df.index] if len(roster_data) > 0 else []
                roster_df = roster_df.reset_index(drop=True)

                st.write("**Squad Roster** (click to view profile):")
                selection = st.dataframe(
                    roster_df,
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="team_roster_table",
                    hide_index=True,
                    column_config=auto_column_config(roster_df)
                )

                # Handle row selection for navigation to Player Profile
                if selection and selection.selection and selection.selection.rows:
                    selected_row_idx = selection.selection.rows[0]
                    if selected_row_idx < len(player_id_list):
                        selected_player_id = player_id_list[selected_row_idx]
                        st.session_state.selected_player_id = selected_player_id
                        st.session_state.nav_to_profile = True
                        st.session_state.nav_season_id = selected_season_id
                        st.session_state.nav_has_season = True
                        st.rerun()

        st.subheader("Season Shot Maps (Non-Penalty)")
        col1_shot, col2_shot = st.columns(2)
        with col1_shot:
            st.markdown(f"**Shots FOR {selected_team_t}**")
            fig_shots_for = create_season_shotmap(team_events_df, selected_team_t)
            st.pyplot(fig_shots_for, use_container_width=True)
            plt.close(fig_shots_for)
        with col2_shot:
            st.markdown(f"**Shots AGAINST {selected_team_t}**")
            fig_shots_against = create_season_shots_against_shotmap(team_events_df, team_matches_df, selected_team_t)
            st.pyplot(fig_shots_against, use_container_width=True)
            plt.close(fig_shots_against)

        # --- Rolling xG History ---
        with st.expander("Rolling xG (5-Game Average)", expanded=False):
            try:
                # Use the stage-filtered events/matches so the rolling
                # series only covers matches in the active stage.
                rolling_xg_data_for_plot = calculate_xg_history_data(team_events_df, team_matches_df)
                if not rolling_xg_data_for_plot.empty:
                    fig_rolling_xg = plot_match_xg_history(rolling_xg_data_for_plot, selected_team_t)
                    st.pyplot(fig_rolling_xg, use_container_width=True)
                    plt.close(fig_rolling_xg)
                else:
                    st.warning("No data available to calculate xG history.")
            except Exception as e:
                st.error(f"Error loading xG history: {e}")

        st.subheader("Corner Kick Analysis")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("**Corners from Left Side**")
            fig_corner_left = plot_corner_analysis(team_events_df, selected_team_t, 'left')
            st.pyplot(fig_corner_left, use_container_width=True)
        with col_c2:
            st.markdown("**Corners from Right Side**")
            fig_corner_right = plot_corner_analysis(team_events_df, selected_team_t, 'right')
            st.pyplot(fig_corner_right, use_container_width=True)

        st.subheader("Season-Long Stats")
        if selected_team_t in team_season_stats and 'corners' in team_season_stats[selected_team_t]:
            st.markdown("**Corner Kick Summary**")
            if selected_stage not in (STAGE_ALL, None):
                st.caption(
                    f"⚠️ Showing full-season aggregates; this table is pre-computed "
                    f"and doesn't filter to the **{selected_stage}** stage."
                )
            st.dataframe(team_season_stats[selected_team_t]['corners'])
        else:
            st.write("No season-long stats available for this team.")

        # =============================================================
        # Tactical Zone Analysis (Wyscout-style)
        # =============================================================
        st.subheader("Tactical Zone Analysis")

        # 1. Ball Recovery Zones (vs league average)
        st.markdown("**Ball Recovery Zones** (vs League Average)")
        try:
            fig_rec_z = pv.plot_zone_heatmap(
                team_events_df, selected_team_t, 'recovery',
                league_events_df=team_events_df,
            )
            st.pyplot(fig_rec_z, use_container_width=True)
            plt.close(fig_rec_z)
        except Exception as e:
            st.caption(f"Could not render recovery zones: {e}")

        # 2. Ball Loss Zones (vs league average)
        st.markdown("**Ball Loss Zones** (vs League Average)")
        try:
            fig_loss_z = pv.plot_zone_heatmap(
                team_events_df, selected_team_t, 'loss',
                league_events_df=team_events_df,
            )
            st.pyplot(fig_loss_z, use_container_width=True)
            plt.close(fig_loss_z)
        except Exception as e:
            st.caption(f"Could not render loss zones: {e}")

        # 3. Passing Network (Season)
        st.markdown("**Passing Network (Season)**")
        try:
            fig_pn = pv.plot_passing_network(team_events_df, selected_team_t)
            st.pyplot(fig_pn, use_container_width=True)
            plt.close(fig_pn)
        except Exception as e:
            st.caption(f"Could not render passing network: {e}")

        # 4. Defensive Structure
        st.markdown("**Defensive Structure**")
        try:
            fig_ds = pv.plot_defensive_structure(team_events_df, selected_team_t,
                                                   league_events_df=team_events_df)
            st.pyplot(fig_ds, use_container_width=True)
            plt.close(fig_ds)
        except Exception as e:
            st.caption(f"Could not render defensive structure: {e}")

    elif analysis_type == 'League Analysis':

        # --- League & Season Selector ---
        selected_comp_ids = league_selector("league_analysis")
        selected_season_id = season_selector("league_analysis", comp_ids=selected_comp_ids)
        active_season_ids = get_season_ids_for_selection(selected_season_id, selected_comp_ids)
        league_events_df = get_filtered_events(raw_events_df, active_season_ids, selected_comp_ids)
        league_matches_df = filter_by_league(get_season_matches(matches_summary_df, active_season_ids), selected_comp_ids)

        # --- 1. ALL DATA CALCS ---
        stats_df_raw, stats_df_pct = calculate_all_team_radars_stats(league_events_df, league_matches_df, season_id=active_season_ids if isinstance(active_season_ids, list) else selected_season_id)
        team_strength_df = calculate_team_strength(league_events_df, league_matches_df, season_id=active_season_ids if isinstance(active_season_ids, list) else selected_season_id).copy()

        # Filter all_match_data to only include matches from selected season
        season_match_ids = set(league_matches_df['matchId'].dropna().unique())
        season_match_data = {mid: data for mid, data in all_match_data.items() if mid in season_match_ids}

        try:
            expanded_stats_df = calculate_expanded_team_stats(season_match_data, league_matches_df, season_id=selected_season_id)
            combined_stats_df = pd.merge(stats_df_raw, expanded_stats_df, left_index=True, right_index=True, how='outer').fillna(0)
        except Exception as e:
            st.warning(f"Could not calculate expanded match stats: {e}")
            combined_stats_df = stats_df_raw.copy()

        # Calculate and merge set piece metrics
        try:
            set_piece_df = calculate_set_piece_metrics(league_events_df, season_id=selected_season_id)
            combined_stats_df = pd.merge(combined_stats_df, set_piece_df, left_index=True, right_index=True, how='outer').fillna(0)
        except Exception as e:
            st.warning(f"Could not calculate set piece metrics: {e}")

        # --- 2. Define Team Lists (Season-dependent groups) ---
        # Only 2025/26 has defined Group A/B; other seasons show all teams together
        SEASON_GROUPS = {
            191782: {
                'Group A': ['Fafe', 'Varzim', 'Paredes', 'Sanjoanense', 'São João Ver',
                            'Amarante', 'Vitória Guimarães II', 'Trofense', 'Sporting Braga II', 'AD Marco 09'],
                'Group B': ['1º Dezembro', 'Caldas', 'Sporting Covilhã', 'Mafra', 'União Santarém',
                            'Amora', 'Académica', 'CF Os Belenenses', 'Lusitano Évora 1911', 'Atlético CP'],
            }
        }

        has_groups = selected_season_id in SEASON_GROUPS
        all_season_teams = sorted(pd.concat([league_matches_df.get('homeTeamName'), league_matches_df.get('awayTeamName')]).dropna().unique())

        if has_groups:
            GROUP_A_TEAMS = SEASON_GROUPS[selected_season_id]['Group A']
            GROUP_B_TEAMS = SEASON_GROUPS[selected_season_id]['Group B']
            valid_group_a_teams = [t for t in GROUP_A_TEAMS if t in combined_stats_df.index]
            valid_group_b_teams = [t for t in GROUP_B_TEAMS if t in combined_stats_df.index]
            ALL_TEAMS_TO_HIGHLIGHT = list(set(GROUP_A_TEAMS + GROUP_B_TEAMS))
        else:
            ALL_TEAMS_TO_HIGHLIGHT = all_season_teams

        valid_all_teams = [t for t in ALL_TEAMS_TO_HIGHLIGHT if t in combined_stats_df.index]

        # --- 3. League Tables ---
        st.subheader("League Standings")

        league_table_config = {
            'Pos': st.column_config.NumberColumn('Pos', width='small'),
            'Team': st.column_config.TextColumn('Team', width='medium'),
            'P': st.column_config.NumberColumn('P', help='Played', width='small'),
            'W': st.column_config.NumberColumn('W', help='Won', width='small'),
            'D': st.column_config.NumberColumn('D', help='Drawn', width='small'),
            'L': st.column_config.NumberColumn('L', help='Lost', width='small'),
            'GF': st.column_config.NumberColumn('GF', help='Goals For', width='small'),
            'GA': st.column_config.NumberColumn('GA', help='Goals Against', width='small'),
            'GD': st.column_config.NumberColumn('GD', help='Goal Difference', width='small'),
            'Pts': st.column_config.NumberColumn('Pts', help='Points', width='small'),
        }

        if has_groups:
            col_table_a, col_table_b = st.columns(2)
            with col_table_a:
                st.markdown("**Group A**")
                table_a = calculate_league_table(league_matches_df, GROUP_A_TEAMS)
                st.dataframe(table_a, use_container_width=True, hide_index=True, column_config=league_table_config)
            with col_table_b:
                st.markdown("**Group B**")
                table_b = calculate_league_table(league_matches_df, GROUP_B_TEAMS)
                st.dataframe(table_b, use_container_width=True, hide_index=True, column_config=league_table_config)
        else:
            st.markdown("**All Teams**")
            table_all = calculate_league_table(league_matches_df, all_season_teams)
            st.dataframe(table_all, use_container_width=True, hide_index=True, column_config=league_table_config)

        # --- 4. Strength Charts ---
        if has_groups:
            st.subheader(f"Team Strength Scatterplot ({get_league_label(selected_comp_ids)} - Group B)")
            if not team_strength_df.empty:
                valid_group_b_strength_teams = [t for t in GROUP_B_TEAMS if t in team_strength_df.index]
                fig_group_b_strength = plot_team_strength(team_strength_df, teams_to_include=valid_group_b_strength_teams, icon_zoom=0.4)
                st.pyplot(fig_group_b_strength, use_container_width=True)
                with st.expander("View Group B Raw Strength Data"):
                    if valid_group_b_strength_teams:
                        st.dataframe(team_strength_df.loc[valid_group_b_strength_teams, ['Attacking Strength', 'Defending Strength']].round(2))
            else:
                st.warning("Could not calculate team strength data for Group B.")

            # Group B Custom Scatterplot
            st.subheader("Group B Custom Scatterplot")
            if not combined_stats_df.empty and valid_group_b_teams:
                group_b_stats_df = combined_stats_df.loc[valid_group_b_teams]
                metrics_to_exclude = ['teamName', 'matchId', 'seasonId', 'teamId']
                available_metrics_gb = sorted([col for col in group_b_stats_df.columns if col not in metrics_to_exclude])

                col_x_gb, col_y_gb = st.columns(2)
                with col_x_gb:
                    default_x_gb_index = available_metrics_gb.index('xG') if 'xG' in available_metrics_gb else 0
                    x_metric_gb = st.selectbox("Select X-Axis Metric:", available_metrics_gb, index=default_x_gb_index, key='x_metric_group_b')
                with col_y_gb:
                    default_y_gb_index = available_metrics_gb.index('xG Against') if 'xG Against' in available_metrics_gb else 1
                    y_metric_gb = st.selectbox("Select Y-Axis Metric:", available_metrics_gb, index=default_y_gb_index, key='y_metric_group_b')

                col_inv_x_gb, col_inv_y_gb = st.columns(2)
                with col_inv_x_gb:
                    invert_x_gb = st.checkbox("Invert X-Axis (Lower is Better)", key='invert_x_group_b')
                with col_inv_y_gb:
                    default_invert_y_gb = 'Against' in y_metric_gb or 'PPDA' in y_metric_gb or 'Losses' in y_metric_gb
                    invert_y_gb = st.checkbox("Invert Y-Axis (Lower is Better)", value=default_invert_y_gb, key='invert_y_group_b')

                if x_metric_gb and y_metric_gb:
                    fig_custom_gb = plot_custom_scatter(group_b_stats_df, x_metric_gb, y_metric_gb, invert_x_gb, invert_y_gb)
                    st.pyplot(fig_custom_gb, use_container_width=True)
            else:
                st.info("No data available for Group B custom plot.")

        # --- 5. All Teams Strength Chart ---
        st.subheader("Team Strength Scatterplot (All Teams)")

        # Multi-season comparison option (filtered to selected league)
        league_scatter_map = {}
        for cid in selected_comp_ids:
            if cid in COMPETITIONS:
                league_scatter_map.update(COMPETITIONS[cid]["seasons"])
        scatter_season_labels = list(league_scatter_map.values())
        scatter_default = league_scatter_map.get(selected_season_id)
        if not scatter_default and scatter_season_labels:
            scatter_default = scatter_season_labels[0]
        scatter_seasons = st.multiselect(
            "Compare seasons", scatter_season_labels,
            default=[scatter_default] if scatter_default else [],
            key="scatter_seasons"
        )
        season_name_to_id_scatter = {v: k for k, v in league_scatter_map.items()}

        if len(scatter_seasons) > 1:
            # Multi-season: combine team_strength_df from each season
            combined_strength_frames = []
            for sname in scatter_seasons:
                sid = season_name_to_id_scatter[sname]
                s_events = get_filtered_events(raw_events_df, sid, selected_comp_ids)
                s_matches = filter_by_league(get_season_matches(matches_summary_df, sid), selected_comp_ids)
                s_df = calculate_team_strength(s_events, s_matches, season_id=sid).copy()
                if not s_df.empty:
                    s_df.index = [f"{t} ({sname})" for t in s_df.index]
                    s_df['Season'] = sname
                    combined_strength_frames.append(s_df)
            if combined_strength_frames:
                multi_strength_df = pd.concat(combined_strength_frames)
                # Plot with text labels (no logos since same team appears multiple times)
                fig_multi = plot_team_strength(multi_strength_df, season="Multi-Season")
                st.pyplot(fig_multi, use_container_width=True)
                with st.expander("View Multi-Season Raw Strength Data"):
                    st.dataframe(multi_strength_df[['Attacking Strength', 'Defending Strength', 'Season']].round(2))
            else:
                st.warning("No team strength data for selected seasons.")
        else:
            # Single season (original behavior)
            if not team_strength_df.empty:
                valid_all_strength_teams = [t for t in ALL_TEAMS_TO_HIGHLIGHT if t in team_strength_df.index]
                fig_all_strength = plot_team_strength(team_strength_df, teams_to_include=valid_all_strength_teams)
                st.pyplot(fig_all_strength, use_container_width=True)
                with st.expander("View All Teams Raw Strength Data"):
                     st.dataframe(team_strength_df[['Attacking Strength', 'Defending Strength']].round(2))
            else:
                st.warning("Could not calculate team strength data.")

        # --- 6. All Teams Custom Scatterplot ---
        st.subheader("All Teams Custom Scatterplot")
        if not combined_stats_df.empty:
            metrics_to_exclude = ['teamName', 'matchId', 'seasonId', 'teamId']
            available_metrics_all = sorted([col for col in combined_stats_df.columns if col not in metrics_to_exclude])

            col_x_all, col_y_all = st.columns(2)
            with col_x_all:
                default_x_all_index = available_metrics_all.index('xG') if 'xG' in available_metrics_all else 0
                x_metric_all = st.selectbox("Select X-Axis Metric:", available_metrics_all, index=default_x_all_index, key='x_metric_all')
            with col_y_all:
                default_y_all_index = available_metrics_all.index('xG Against') if 'xG Against' in available_metrics_all else 1
                y_metric_all = st.selectbox("Select Y-Axis Metric:", available_metrics_all, index=default_y_all_index, key='y_metric_all')

            col_inv_x_all, col_inv_y_all = st.columns(2)
            with col_inv_x_all:
                invert_x_all = st.checkbox("Invert X-Axis (Lower is Better)", key='invert_x_all')
            with col_inv_y_all:
                default_invert_y_all = 'Against' in y_metric_all or 'PPDA' in y_metric_all or 'Losses' in y_metric_all
                invert_y_all = st.checkbox("Invert Y-Axis (Lower is Better)", value=default_invert_y_all, key='invert_y_all')

            if x_metric_all and y_metric_all:
                fig_custom_all = plot_custom_scatter(combined_stats_df, x_metric_all, y_metric_all, invert_x_all, invert_y_all)
                st.pyplot(fig_custom_all, use_container_width=True)

            with st.expander("View All Teams Raw Radar & Expanded Stats Data"):
                st.dataframe(combined_stats_df.round(2))
        else:
            st.warning("Could not calculate raw league stats for custom plot.")

    
    # --- UPDATED: Renamed to Player Profile ---
    elif analysis_type == 'Player Profile':

        # If navigating from another section, set the season selector to match
        if st.session_state.get('nav_has_season', False):
            nav_sid = st.session_state.nav_season_id
            if nav_sid is None:
                # "All Seasons" was selected
                st.session_state['season_select_player_profile'] = "All Seasons"
            else:
                nav_label = SEASON_ID_MAP.get(nav_sid)
                if nav_label:
                    st.session_state['season_select_player_profile'] = nav_label
            st.session_state.nav_season_id = None
            st.session_state.nav_has_season = False

        # --- League & Season Selector ---
        selected_comp_ids = league_selector("player_profile")
        selected_season_id = season_selector("player_profile", include_all_seasons=True, comp_ids=selected_comp_ids)
        active_season_ids = get_season_ids_for_selection(selected_season_id, selected_comp_ids)
        profile_season_changed = (selected_season_id != st.session_state.player_profile_last_season)
        st.session_state.player_profile_last_season = selected_season_id
        profile_events_df = get_filtered_events(raw_events_df, active_season_ids, selected_comp_ids)
        profile_matches_df = filter_by_league(get_season_matches(matches_summary_df, active_season_ids), selected_comp_ids)
        profile_player_minutes_df = get_season_player_minutes(player_minutes_data, active_season_ids, comp_ids=selected_comp_ids)

        # --- 1. Load All Necessary Data ---
        player_details_df = load_player_details()

        try:
            with st.spinner("Calculating player statistics (this may take a moment on first load)..."):
                player_stats_df, player_stats_with_scores_df = load_and_score_player_stats(
                    profile_events_df, profile_player_minutes_df, selected_season_id, active_season_ids, selected_comp_ids
                )
        except Exception as e:
            st.error(f"An error occurred calculating overall player stats: {e}")
            logger.exception("Error in calculate_all_player_stats")
            player_stats_df = pd.DataFrame()
            player_stats_with_scores_df = pd.DataFrame()
            
        if player_stats_df.empty or player_details_df.empty or player_stats_with_scores_df.empty:
            st.warning("Player data not available. Please ensure all processing scripts have run and data is loaded.")
            st.stop()

        # --- 2. Player Selector ---
        st.sidebar.subheader("Player Analysis Options")

        # FIX: Include 'playerId' in the list dataframe so we can grab it later
        # (Added 'playerId' to the list of columns below)
        player_list_df = player_stats_with_scores_df[['playerId', 'playerName', 'teamName', 'totalMinutes']].sort_values(by='totalMinutes', ascending=False)

        # Create unique display names
        player_list_df['display_name'] = player_list_df['playerName'].astype(str) + " (" + player_list_df['teamName'].astype(str) + ", " + pd.to_numeric(player_list_df['totalMinutes'], errors='coerce').fillna(0).astype(int).astype(str) + " min)"

        # If navigating from another section, set the player selectbox value directly
        if st.session_state.selected_player_id is not None:
            sorted_player_ids = player_list_df['playerId'].tolist()
            target_id = st.session_state.selected_player_id
            for i, pid in enumerate(sorted_player_ids):
                if int(pid) == int(target_id):
                    st.session_state['player_profile_selector'] = player_list_df['display_name'].iloc[i]
                    st.session_state.player_profile_current_id = int(target_id)
                    break
            st.session_state.selected_player_id = None
        # Persist player selection across season changes (only when season actually changed)
        elif profile_season_changed and st.session_state.player_profile_current_id is not None:
            target_id = st.session_state.player_profile_current_id
            sorted_player_ids = player_list_df['playerId'].tolist()
            for i, pid in enumerate(sorted_player_ids):
                if int(pid) == int(target_id):
                    st.session_state['player_profile_selector'] = player_list_df['display_name'].iloc[i]
                    break

        selected_player_display = st.sidebar.selectbox(
            "Select Player:",
            player_list_df['display_name'],
            key="player_profile_selector"
        )
        
        try:
            # FIX: Get the UNIQUE ID corresponding to the selected display name
            # (We use .values[0] to grab the actual integer ID)
            selected_player_id = player_list_df[player_list_df['display_name'] == selected_player_display]['playerId'].values[0]
            st.session_state.player_profile_current_id = int(selected_player_id)

            # FIX: Filter the main dataframe by ID, not by Name
            # This ensures we get the exact Miguel Lopes the user clicked on
            player_data_row = player_stats_with_scores_df[player_stats_with_scores_df['playerId'] == selected_player_id]
            
            # Extract stats series
            player_per_90_stats = player_data_row.iloc[0] 
            
            # Define player_id variable for use in other sections
            player_id = selected_player_id
            
            # Load Bio
            player_bio = player_details_df.loc[player_id] if player_id in player_details_df.index else pd.Series(dtype='object')
            total_minutes = player_per_90_stats.get('totalMinutes', 0)
            
            # Also update the 'selected_player_name' variable for the Match Log function
            selected_player_name = player_per_90_stats.get('playerName')

        except Exception as e:
            st.error(f"Could not load data for {selected_player_display}. Error: {e}")
            st.stop()
        
        # --- 3. Get Player's Match Log ---
        player_match_log_df = get_player_match_stats(selected_player_name, all_match_data, profile_matches_df, season_id=selected_season_id)

        # Enrich match log with per-match xA (xAOP/xASP) and xT (xTOP/xTSP) from raw events
        if not player_match_log_df.empty and not profile_events_df.empty:
            try:
                _p_events = profile_events_df[profile_events_df['player.name'] == selected_player_name].copy()
                if not _p_events.empty:
                    # Per-match xA: find shot assists, map to shot xG
                    _p_events['shot_event_id'] = np.where(_p_events['shot.xg'].notna(), _p_events['id'], np.nan)
                    _p_events['next_shot_id'] = _p_events.groupby('matchId')['shot_event_id'].bfill()
                    _shot_xg_map = _p_events[_p_events['shot.xg'].notna()].set_index('id')['shot.xg'].to_dict()
                    # Get all events in matches this player played (need all players for shot assists)
                    _player_match_ids = _p_events['matchId'].unique()
                    _match_events = profile_events_df[profile_events_df['matchId'].isin(_player_match_ids)].copy()
                    _match_events['shot_event_id'] = np.where(_match_events['shot.xg'].notna(), _match_events['id'], np.nan)
                    _match_events['next_shot_id'] = _match_events.groupby('matchId')['shot_event_id'].bfill()
                    _all_shot_xg = _match_events[_match_events['shot.xg'].notna()].set_index('id')['shot.xg'].to_dict()
                    _assists = _match_events[
                        (_match_events['player.name'] == selected_player_name) &
                        (_match_events.get('type.secondary', pd.Series(dtype='object')).apply(lambda x: isinstance(x, (list, np.ndarray)) and 'shot_assist' in x))
                    ].copy()
                    _assists['xA'] = _assists['next_shot_id'].map(_all_shot_xg)
                    _sp_types = ['corner', 'free_kick', 'throw_in', 'goal_kick']
                    _assists['_xa_type'] = np.where(_assists['type.primary'].isin(_sp_types), 'xASP', 'xAOP')
                    # Aggregate per match
                    _xa_per_match = _assists.groupby(['matchId', '_xa_type'])['xA'].sum().unstack(fill_value=0).reset_index()
                    for _c in ['xAOP', 'xASP']:
                        if _c not in _xa_per_match.columns:
                            _xa_per_match[_c] = 0.0
                    _xa_total = _assists.groupby('matchId')['xA'].sum().reset_index().rename(columns={'xA': 'xA_total'})
                    _xa_per_match = _xa_per_match.merge(_xa_total, on='matchId', how='left')

                    # Per-match xT: calculate from passes/touches/accelerations + set pieces
                    _xt_data = [[0.01,0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.03,0.03,0.04,0.04],[0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.03,0.04,0.05,0.05],[0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.03,0.05,0.06,0.06],[0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.04,0.11,0.26,0.26],[0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.04,0.11,0.26,0.26],[0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.03,0.05,0.06,0.06],[0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.03,0.04,0.05,0.05],[0.01,0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.03,0.03,0.04,0.04]]
                    _xt_grid = np.array(_xt_data); _r, _c = _xt_grid.shape
                    _sp_xt = ['corner', 'free_kick', 'throw_in']
                    _moves = _p_events[_p_events['type.primary'].isin(['pass', 'touch', 'acceleration'] + _sp_xt)].copy()
                    _suc_pass = (_moves['type.primary'] == 'pass') & (_moves.get('pass.accurate') == True)
                    _other_suc = _moves['type.primary'].isin(['touch', 'acceleration'] + _sp_xt)
                    _moves = _moves[_suc_pass | _other_suc]
                    _is_pass_like = _moves['type.primary'].isin(['pass'] + _sp_xt)
                    _moves['_ex'] = np.where(_is_pass_like, _moves.get('pass.endLocation.x'), _moves.get('carry.endLocation.x'))
                    _moves['_ey'] = np.where(_is_pass_like, _moves.get('pass.endLocation.y'), _moves.get('carry.endLocation.y'))
                    _moves = _moves.dropna(subset=['_ex', '_ey'])
                    _moves['_sc'] = np.clip((_moves['location.x'].astype(float).fillna(0) / 100 * _c).astype(int), 0, _c-1)
                    _moves['_sr'] = np.clip((_moves['location.y'].astype(float).fillna(0) / 100 * _r).astype(int), 0, _r-1)
                    _moves['_ec'] = np.clip((_moves['_ex'].astype(float).fillna(0) / 100 * _c).astype(int), 0, _c-1)
                    _moves['_er'] = np.clip((_moves['_ey'].astype(float).fillna(0) / 100 * _r).astype(int), 0, _r-1)
                    _moves['_xT'] = _xt_grid[_moves['_er'].values, _moves['_ec'].values] - _xt_grid[_moves['_sr'].values, _moves['_sc'].values]
                    _pos_xt = _moves[_moves['_xT'] > 0].copy()
                    _pos_xt['_xt_type'] = np.where(_pos_xt['type.primary'].isin(_sp_xt), 'xTSP', 'xTOP')
                    _xt_per_match = _pos_xt.groupby(['matchId', '_xt_type'])['_xT'].sum().unstack(fill_value=0).reset_index()
                    for _c_name in ['xTOP', 'xTSP']:
                        if _c_name not in _xt_per_match.columns:
                            _xt_per_match[_c_name] = 0.0
                    _xt_total = _pos_xt.groupby('matchId')['_xT'].sum().reset_index().rename(columns={'_xT': 'xT_total'})
                    _xt_per_match = _xt_per_match.merge(_xt_total, on='matchId', how='left')

                    # Map matchId to match log rows via Date + opponent lookup
                    _mid_map = profile_matches_df.set_index('matchId')['dateutc'].to_dict()
                    _xa_per_match['_date'] = _xa_per_match['matchId'].map(_mid_map)
                    _xa_per_match['_date'] = pd.to_datetime(_xa_per_match['_date'], errors='coerce').apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else 'N/A')
                    _xt_per_match['_date'] = _xt_per_match['matchId'].map(_mid_map)
                    _xt_per_match['_date'] = pd.to_datetime(_xt_per_match['_date'], errors='coerce').apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else 'N/A')

                    # Merge into match log
                    player_match_log_df = player_match_log_df.merge(
                        _xa_per_match[['_date', 'xAOP', 'xASP']].rename(columns={'_date': 'Date'}),
                        on='Date', how='left'
                    )
                    player_match_log_df = player_match_log_df.merge(
                        _xt_per_match[['_date', 'xTOP', 'xTSP']].rename(columns={'_date': 'Date'}),
                        on='Date', how='left'
                    )
                    # Round and fill
                    for _mc in ['xAOP', 'xASP', 'xTOP', 'xTSP']:
                        if _mc in player_match_log_df.columns:
                            player_match_log_df[_mc] = player_match_log_df[_mc].fillna(0).round(2)
            except Exception as e:
                print(f"Warning: Could not enrich match log with xA/xT: {e}")

        # --- 4. Display Player Bio ---
        current_team = player_per_90_stats.get('teamName', 'N/A')
        current_pos = player_per_90_stats.get('primaryPosition', 'N/A')

        # If 'Unknown', try to force a lookup in the minutes file
        if (current_team in ['Unknown', 'N/A'] or current_pos in ['Unknown', 'N/A']) and not profile_player_minutes_df.empty:
            try:
                pid_int = int(player_id)
                min_row = profile_player_minutes_df[profile_player_minutes_df['playerId'] == pid_int]
                if not min_row.empty:
                    current_team = min_row.iloc[0]['teamName']
                    current_pos = min_row.iloc[0]['primaryPosition']
            except Exception:
                pass # Fallback to original values if lookup fails

        st.header(f"{player_per_90_stats.get('playerName', 'N/A')}")

        # ---- Compute Transfer Value primitives ONCE per page load ----
        # Used twice: (a) compact projected / true / Δ metrics in the
        # Player Information bio row below, (b) full CVI breakdown +
        # Market Context detail section just above Career Trajectory.
        _tv_age = _calculate_age(player_bio.get('birthDate')) \
                   if isinstance(player_bio, pd.Series) else None
        _tv_comp_id = competition_for_season(selected_season_id) \
                       if selected_season_id else None
        _tv_player_row = player_data_row.iloc[0] if not player_data_row.empty else None
        _tv_single = (player_data_row.copy()
                       if not player_data_row.empty else None)
        if _tv_single is not None and 'Total Value' not in _tv_single.columns:
            # GPA merge may not be present in profile-mode stats; backfill.
            try:
                _gpa = load_gpa_values()
                if _gpa is not None and not _gpa.empty:
                    _r = _gpa[(_gpa['playerId'] == player_id)
                               & (_gpa['seasonId'] == selected_season_id)]
                    if not _r.empty:
                        _val_col = next((c for c in ('Total Value',
                                          'total_v_per_90')
                                          if c in _r.columns), None)
                        if _val_col:
                            _tv_single['Total Value'] = float(_r[_val_col].iloc[0])
            except Exception:
                pass

        _tv_cvi_block = pd.DataFrame()
        if _tv_single is not None and 'primaryPosition' in _tv_single.columns:
            try:
                # Build empirical-Bayes prior for THIS player based on
                # their strictly-prior-season career data. Pulls the
                # same perf_table used for Career CVI below so we only
                # pay the cost once.
                _tv_prior_lookup = None
                try:
                    _pt = _build_player_season_perf_table(
                        load_gpa_values(),
                        profile_player_minutes_df
                          if 'profile_player_minutes_df' in dir() else None,
                    )
                    _prior_map = build_player_priors_lookup(
                        _pt, selected_season_id,
                    ) if selected_season_id is not None else {}
                    _tv_prior_lookup = lambda pid: _prior_map.get(int(pid)) \
                                                    if pid is not None else None
                except Exception:
                    _tv_prior_lookup = None
                _tv_cvi_block = compute_cvi_columns(
                    _tv_single,
                    age_lookup=lambda pid: _tv_age,
                    comp_id_lookup=lambda pid: _tv_comp_id,
                    prior_lookup=_tv_prior_lookup,
                )
            except Exception:
                _tv_cvi_block = pd.DataFrame()

        # ---- Career CVI (cross-season, cross-league, decay-weighted) ----
        # Two anchors:
        #   • _tv_career_current — anchored to the player's most recent
        #     season → headline "Current CVI" in the bio row
        #   • _tv_career_season  — anchored to the selected season → shown
        #     in Transfer Value Detail with full per-season breakdown
        # Both apply 0.6 decay per season back and never peek at seasons
        # after the anchor.
        _tv_career_current = None
        _tv_career_season = None
        try:
            _gpa_for_career = load_gpa_values()
            _perf_table = _build_player_season_perf_table(
                _gpa_for_career,
                profile_player_minutes_df
                  if 'profile_player_minutes_df' in dir() else None,
            )
            _dob_for_career = None
            if isinstance(player_bio, pd.Series):
                _dob_for_career = player_bio.get('birthDate')
            _dob_lookup_for_career = (lambda pid:
                                         pd.to_datetime(_dob_for_career,
                                                          errors='coerce')
                                         if _dob_for_career is not None
                                         else None)
            # Current CVI: anchor at player's most recent season w/ GPA data
            _most_recent_sid = most_recent_season_for_player(_perf_table,
                                                                player_id)
            if _most_recent_sid is not None:
                _tv_career_current = compute_career_cvi(
                    player_id, _most_recent_sid,
                    perf_table=_perf_table,
                    dob_lookup=_dob_lookup_for_career,
                )
            # Season CVI: anchor at the user-selected season
            if selected_season_id is not None:
                _tv_career_season = compute_career_cvi(
                    player_id, selected_season_id,
                    perf_table=_perf_table,
                    dob_lookup=_dob_lookup_for_career,
                )
        except Exception as _car_exc:
            print(f"Warning: career CVI failed: "
                   f"{type(_car_exc).__name__}: {_car_exc}")

        _tv_valuations_rows = pd.DataFrame()
        try:
            from valuations.load_valuations import load_all_valuations
            _all_val = load_all_valuations()
            if _all_val is not None and not _all_val.empty:
                _tv_valuations_rows = _all_val[
                    _all_val['playerId'] == player_id
                ].sort_values('as_of_date', ascending=False)
        except Exception:
            pass

        # Headline projected / true / Δ values for the bio row.
        # v2.7+ — Projected value is now a direct CVI → EUR mapping,
        # calibrated against the 27 reported transfers (mostly €25k-€450k).
        # Power-curve fit: EUR ≈ 2.5 × CVI^2.5, capped at €500k. Anchors:
        #   CVI 40  → €25k    (replacement-level starter)
        #   CVI 60  → €70k
        #   CVI 80  → €150k
        #   CVI 100 → €260k
        #   CVI 120 → €400k
        #   CVI 135 → €500k   (cap — top of Yan Maranhão / Catarino tier)
        _tv_projected_eur = None
        _current_cvi_for_eur = None
        if _tv_career_current is not None:
            _v = _tv_career_current.get('career_cvi')
            if _v is not None and not pd.isna(_v):
                _current_cvi_for_eur = float(_v)
        if _current_cvi_for_eur is None and not _tv_cvi_block.empty:
            _v = _tv_cvi_block.iloc[0].get('_CVI')
            if _v is not None and not pd.isna(_v):
                _current_cvi_for_eur = float(_v)
        if _current_cvi_for_eur is not None and _current_cvi_for_eur > 0:
            # v2.9 — single helper handles power curve + position
            # multiplier + Camp penalty + cap. See cvi_to_projected_eur.
            _pos_grp_for_eur = _cvi_position_group(current_pos)
            _tv_projected_eur = cvi_to_projected_eur(
                _current_cvi_for_eur,
                position_group=_pos_grp_for_eur,
                competition_id=_tv_comp_id,
            )
        _tv_true_eur = None
        _tv_true_source = None
        if not _tv_valuations_rows.empty:
            _latest = _tv_valuations_rows.iloc[0]
            _tv_true_eur = _latest.get('value_eur')
            _tv_true_source = _latest.get('source')
        _tv_delta_eur = (_tv_projected_eur - _tv_true_eur
                          if _tv_projected_eur is not None and _tv_true_eur is not None
                          else None)

        # Engine projected value — computed centrally in
        # load_player_engine() (engine_value_eur); just look it up.
        _eng_proj_eur = None
        try:
            _eng_tv_df, _ = load_player_engine()
            if not _eng_tv_df.empty:
                _p_rows = _eng_tv_df[(_eng_tv_df['playerId'] == int(selected_player_id))
                                       ].dropna(subset=['engine_value_eur'])
                if not _p_rows.empty:
                    _p_row = (_p_rows[_p_rows['seasonId'] == _p_rows['seasonId'].max()]
                                .sort_values('mins_played').iloc[-1])
                    _eng_proj_eur = float(_p_row['engine_value_eur'])
        except Exception:
            logger.exception("engine projected value failed")

        col1_bio, col2_bio = st.columns([1, 3])
        with col1_bio:
            image_url = player_bio.get('imageDataURL', None)
            if image_url:
                st.image(image_url, width=150)
            else:
                st.image("https://t3.ftcdn.net/jpg/05/16/27/58/360_F_516275801_f3Fsp17x6HQK0xQgDQEELoGau0sJzEf4.jpg", width=150)

        with col2_bio:
            st.subheader("Player Information")
            bio_row1 = st.columns(4)
            bio_row1[0].metric("Team", current_team)
            bio_row1[1].metric("Position", current_pos)
            bio_row1[2].metric("Nationality", player_bio.get('passportArea', 'N/A'))
            
            age = _calculate_age(player_bio.get('birthDate'))
            age_display = f"{age:.1f}" if isinstance(age, float) else "N/A"
            bio_row1[3].metric("Age", age_display)
            
            bio_row2 = st.columns(4)
            foot_value = player_bio.get('foot')
            foot_display = "N/A"
            if foot_value and not pd.isna(foot_value):
                foot_display = foot_value.capitalize()
            bio_row2[0].metric("Foot", foot_display)
            bio_row2[1].metric("Height", f"{player_bio.get('height', 0)} cm")
            bio_row2[2].metric("Weight", f"{player_bio.get('weight', 0)} kg")
            bio_row2[3].metric("Birthplace", player_bio.get('birthArea', 'N/A'))

            # Headline value — ENGINE projected value for outfielders;
            # GOALKEEPERS keep the prior CVI→EUR value (Lucas) since the
            # outfield engine does not cover keepers.
            _is_gk = str(player_per_90_stats.get('primaryPosition', '')).upper().startswith('GK')
            bio_row3 = st.columns(3)
            if _is_gk:
                bio_row3[0].metric(
                    "Projected value",
                    ("—" if _tv_projected_eur is None else f"€{_tv_projected_eur:,.0f}"),
                    help="Goalkeeper — legacy CVI→EUR value (the prior "
                         "system, retained for keepers; the outfield ACP "
                         "engine does not rate goalkeepers).",
                )
            else:
                bio_row3[0].metric(
                    "Projected value",
                    ("—" if _eng_proj_eur is None else f"€{_eng_proj_eur:,.0f}"),
                    help="ACP engine projection → EUR. Perf = percentile of "
                         "the next-season projection (abs scale — Camp "
                         "recruit discount already applied, so no extra Camp "
                         "penalty) × career-NPV age multiplier, through the "
                         "fee-calibrated CVI→EUR curve (capped €500k). No "
                         "reliability ramp: the projection is already "
                         "evidence-weighted.",
                )

        st.divider()

        with st.expander("ℹ️ How these ratings work"):
            st.markdown(RATINGS_EXPLAINER_MD)

        # --- ACP Engine card: rating + projection + components ---------
        try:
            _eng_df, _eng_meta = load_player_engine()
            _erows = _eng_df[_eng_df['playerId'] == int(selected_player_id)] if not _eng_df.empty else pd.DataFrame()
            # active_season_ids can be None (All Seasons), an int, or a list
            if isinstance(active_season_ids, (list, tuple, set)):
                _e_sids = [int(s) for s in active_season_ids if s is not None]
            elif active_season_ids is not None:
                _e_sids = [int(active_season_ids)]
            else:
                _e_sids = None
            _career_view = False
            if _erows.empty:
                _escope = _erows
            elif _e_sids:
                _escope = _erows[_erows['seasonId'].isin(_e_sids)]
            else:   # All Seasons → CAREER view: aggregate every rated season
                _escope = _erows
                _career_view = len(_erows) > 1
            _eng_stale = False
            if _escope.empty and not _erows.empty:
                # not rated in the selected scope — fall back to last rated season
                _escope = _erows[_erows['seasonId'] == _erows['seasonId'].max()]
                _eng_stale = True
            _is_gk_card = str(player_per_90_stats.get('primaryPosition', '')).upper().startswith('GK')
            st.subheader("Goalkeeper Rating (legacy system)" if _is_gk_card
                          else "ACP Index")
            if _escope.empty:
                if _is_gk_card:
                    # Prior rating + value system, retained for keepers.
                    _gk_templates = ['Shot Stopper', 'Cross Claimer', 'Ball-playing GK']
                    _gk_scored = []
                    for _t in _gk_templates:
                        _s = player_per_90_stats.get(f'{_t}_Score')
                        if _s is not None and pd.notna(_s):
                            _gk_scored.append((_t, float(_s)))
                    if _gk_scored:
                        _gk_best = max(_gk_scored, key=lambda x: x[1])
                        _gkc = st.columns(3)
                        _gkc[0].metric(
                            "GK Rating (legacy)", f"{_gk_best[1]:.0f}",
                            help="Best-fit goalkeeper template score — the "
                                 "bespoke weighted-percentile system (the prior "
                                 "rating engine), retained for keepers since the "
                                 "outfield ACP engine does not cover them.")
                        _gkc[1].metric("Best-fit template", _gk_best[0])
                        _gkc[2].metric(
                            "Projected value",
                            "—" if _tv_projected_eur is None else f"€{_tv_projected_eur:,.0f}",
                            help="Legacy CVI→EUR value.")
                        st.caption("Template scores — " + " · ".join(
                            f"**{_t}** {_s:.0f}" for _t, _s in _gk_scored)
                            + "  ·  full goalkeeping metrics in the Player "
                              "Radar and Stats tabs below.")
                    else:
                        st.info("Goalkeeper — insufficient minutes for the "
                                "legacy GK rating in this scope.")
                else:
                    st.info("Not rated by the engine for this scope "
                            "(below the 90-minute floor, or no "
                            "role assignment yet).")
            else:
                _e = _escope.sort_values('mins_played').iloc[-1]
                if _career_view:
                    # All-seasons CAREER view: representative row = highest-
                    # minutes season (for role/league/shares); rating + every
                    # percentile/grade aggregated minutes-weighted across all
                    # the player's rated seasons.
                    _e = _e.copy()
                    _wts = pd.to_numeric(_escope['mins_played'], errors='coerce').fillna(0.0).to_numpy()
                    _pct_cols = ['off_pct', 'qual_pct', 'rapm_pct', 'defr_pct',
                                  'datt_pct', 'setpiece_pct', 'aerial_grade_pct',
                                  'ground_grade_pct', 'Shooting_pct', 'Creating_pct',
                                  'Linking_pct', 'Receiving_pct', 'Dribbling_pct',
                                  'career_asof', 'w_evidence']
                    for _c in _pct_cols:
                        if _c in _escope.columns and _wts.sum() > 0:
                            _v = pd.to_numeric(_escope[_c], errors='coerce').to_numpy()
                            _m = ~np.isnan(_v)
                            if _m.any():
                                _e[_c] = float(np.average(_v[_m], weights=_wts[_m]))
                    if pd.notna(_e.get('acp_rating_career')):
                        _e['acp_rating'] = float(_e['acp_rating_career'])
                    _e['mins_played'] = float(_wts.sum())
                    # projection = the LATEST season's forward look; lineup
                    # minutes = career total (for the radar header)
                    _latest_row = _escope.sort_values('seasonId').iloc[-1]
                    for _pc in ('projection', 'projection_abs', 'band_sd',
                                 'proj_delta', 'seasons_ago', 'age'):
                        if _pc in _latest_row:
                            _e[_pc] = _latest_row[_pc]
                    if 'mins_lineup' in _escope.columns:
                        _e['mins_lineup'] = float(pd.to_numeric(
                            _escope['mins_lineup'], errors='coerce').fillna(0.0).sum())
                    st.caption(f"📚 Career view — minutes-weighted across "
                               f"{len(_escope)} rated seasons "
                               f"({int(_wts.sum())} total minutes). Select a "
                               f"single season for that season's rating.")
                if _eng_stale:
                    st.caption(f"⏳ Not rated in the selected season — showing "
                               f"last rated season ({SEASON_ID_MAP.get(int(_e['seasonId']), _e['seasonId'])}).")
                # The projection is a single player-level forward-look anchored to
                # the player's most-recent season. seasonId is NOT chronological
                # across leagues (Camp 23/24 = 190230 > L3 24/25 = 190090), so the
                # in-scope "latest" row picked above can be the wrong one and carry
                # no projection (e.g. M. Konaté). Always source the projection from
                # the row that actually has one. (Lucas 2026-06-24)
                _proj_src = (_erows[_erows['projection'].notna()]
                             if not _erows.empty else _erows)
                if not _proj_src.empty:
                    _pr = _proj_src.sort_values('seasonId').iloc[-1]
                    for _pc in ('projection', 'projection_abs', 'band_sd',
                                 'proj_delta', 'seasons_ago'):
                        if _pc in _pr.index:
                            _e[_pc] = _pr[_pc]
                _ec1, _ec2, _ec3, _ec4 = st.columns(4)
                _abs_note = (f"abs {_e['acp_rating_abs']:.0f}"
                             if pd.notna(_e.get('acp_rating_abs'))
                             and _e.get('league') == 'CAMP' else None)
                _ec1.metric("ACP Rating", f"{_e['acp_rating']:.0f}", _abs_note,
                            delta_color="off",
                            help="Role-blended 5-axis rating, 50±17 scale, "
                                 "within-league. 'abs' = Liga-3-equivalent "
                                 "(descriptive league delta).")
                if pd.notna(_e.get('projection')):
                    _proj_note = (f"abs {_e['projection_abs']:.0f}"
                                  if pd.notna(_e.get('projection_abs'))
                                  and _e.get('league') == 'CAMP' else None)
                    _ec2.metric("Projection (next season)",
                                f"{_e['projection']:.0f} ± {_e['band_sd']:.0f}",
                                _proj_note, delta_color="off",
                                help="Career + evidence pull + role age curve. "
                                     "Band = role-specific 1 SD of realized "
                                     "error. 'abs' applies the stricter "
                                     "recruit league delta.")
                else:
                    _ec2.metric("Projection", "—",
                                help="No projection (age unknown or below floor).")
                _ec3.metric("Career (as-of)",
                            f"{_e['career_asof']:.0f}" if pd.notna(_e.get('career_asof')) else "—",
                            help="Recency-weighted career rating through this season.")
                _ec4.metric("Evidence",
                            f"{_e['w_evidence']:.0%}" if pd.notna(_e.get('w_evidence')) else "—",
                            help="Career evidence weight w = eff_mins/(eff_mins+K). "
                                 "Low = projection leans on the pull terms.")

                # badges
                _badges = []
                _shares = sorted(
                    [(c[3:], float(_e[c])) for c in _escope.columns
                     if c.startswith('sh_') and pd.notna(_e[c]) and float(_e[c]) > 0.15],
                    key=lambda x: -x[1])
                if _shares:
                    _badges.append(" · ".join(f"**{n}** {s:.0%}" for n, s in _shares[:3]))
                if pd.notna(_e.get('seasons_ago')) and int(_e['seasons_ago']) >= 1:
                    _badges.append("🕐 last rated 24/25 — projection carries "
                                   "extra age step + wider band")
                if float(_e['mins_played']) < 500:
                    _badges.append(f"⚠️ thin sample this league "
                                   f"({int(_e['mins_played'])}′ — admitted via "
                                   f"cross-competition season pooling)")
                if len(_escope) > 1:
                    _others = _escope[_escope.index != _e.name]
                    for _, _o in _others.iterrows():
                        _badges.append(f"also rated in {_o['league']} "
                                       f"({int(_o['mins_played'])}′, rating "
                                       f"{_o['acp_rating']:.0f})")
                if _badges:
                    st.markdown("  \n".join(_badges))

                # component bars
                _comp_cols = st.columns(6)
                for _i, (_lbl, _col) in enumerate([
                        ("Offensive Value", 'off_pct'),
                        ("Def Quality Grade", 'qual_pct'),
                        ("RAPM", 'rapm_pct'),
                        ("Def Volume Grade", 'defr_pct'),
                        ("Off Duel Grade", 'datt_pct'),
                        ("Set piece", 'setpiece_pct')]):
                    _v = _e.get(_col)
                    with _comp_cols[_i]:
                        if pd.notna(_v):
                            st.progress(min(max(float(_v), 0.0), 1.0))
                            st.caption(f"{_lbl} · {float(_v)*100:.0f}")
                        else:
                            st.caption(f"{_lbl} · —")
                # --- ACP Index radar: RAW per-90 values on a mean ± 2σ
                # scale vs the player's league × season × role cohort,
                # with per-axis cohort distributions on the right —
                # mirrors the traditional radar's raw mode + KDE panels.
                # Set piece removed (Lucas); Duel-att raw = n-weighted
                # take-on/shield Glicko; Def Quality raw = DWAE/90.
                _eng_rad_df = _eng_df.copy()
                _nt = _eng_rad_df['duel_takeon_n'].fillna(0.0)
                _ns = _eng_rad_df['duel_shield_n'].fillna(0.0)
                _eng_rad_df['_datt_glicko'] = (
                    (_eng_rad_df['duel_takeon'].fillna(0.0) * _nt
                     + _eng_rad_df['duel_shield'].fillna(0.0) * _ns)
                    / (_nt + _ns).replace(0.0, np.nan))
                if 'aerial_grade_pct' in _eng_rad_df.columns:
                    _eng_rad_df['_aer_grade100'] = _eng_rad_df['aerial_grade_pct'] * 100.0
                    _eng_rad_df['_grd_grade100'] = _eng_rad_df['ground_grade_pct'] * 100.0
                _RAD_AXES = [
                    ('Shooting', 'raw_Shooting90', 'output', '{:.2f}'),
                    ('Receiving', 'raw_Receiving90', 'output', '{:.2f}'),
                    ('Creating', 'raw_Creating90', 'passing', '{:.2f}'),
                    ('Linking', 'raw_Linking90', 'passing', '{:.2f}'),
                    ('Dribbling', 'raw_Dribbling90', 'dribbling', '{:.2f}'),
                    ('Off Duel Grade', '_datt_glicko', 'dribbling', '{:.0f}'),
                    ('Aerial Grade', '_aer_grade100', 'defensive', '{:.0f}'),
                    ('Ground Def Grade', '_grd_grade100', 'defensive', '{:.0f}'),
                    ('Def Volume', 'raw_resp', 'defensive', '{:.2f}'),
                    ('RAPM', 'raw_rapm', 'team', '{:.2f}'),
                ]
                _RAD_COLORS = {'output': 'green', 'passing': 'orange',
                                'defensive': 'red', 'dribbling': 'purple',
                                'team': '#0077b6'}
                _RAD_LEGEND = [('Output', 'green'),
                                ('Passing / Creation', 'orange'),
                                ('Ball Carrying', 'purple'),
                                ('Defending', 'red'),
                                ('Team Impact (RAPM)', '#0077b6')]
                if _career_view:
                    # cohort pooled across ALL seasons of this league+role;
                    # player point = minutes-weighted career per-90
                    _coh = _eng_rad_df[
                        (_eng_rad_df['league'] == _e['league'])
                        & (_eng_rad_df['role'] == _e['role'])
                        & (_eng_rad_df['mins_played'] >= 500)]
                    _pr = _eng_rad_df[_eng_rad_df['playerId'] == int(selected_player_id)]
                    _pw = pd.to_numeric(_pr['mins_played'], errors='coerce').fillna(0.0).to_numpy()
                    _pe = _pr.sort_values('mins_played').iloc[-1].copy()
                    for _rc in [c for _, c, _g, _fm in _RAD_AXES]:
                        if _rc in _pr.columns and _pw.sum() > 0:
                            _rv = pd.to_numeric(_pr[_rc], errors='coerce').to_numpy()
                            _rm = ~np.isnan(_rv)
                            if _rm.any():
                                _pe[_rc] = float(np.average(_rv[_rm], weights=_pw[_rm]))
                else:
                    _coh = _eng_rad_df[
                        (_eng_rad_df['league'] == _e['league'])
                        & (_eng_rad_df['seasonId'] == _e['seasonId'])
                        & (_eng_rad_df['role'] == _e['role'])
                        & (_eng_rad_df['mins_played'] >= 500)]
                    _pe = _eng_rad_df.loc[_e.name]
                _rad = []
                for _lbl, _col, _g, _f in _RAD_AXES:
                    if _col not in _eng_rad_df.columns:
                        continue
                    _pv = _pe.get(_col)
                    _pop = _coh[_col].dropna()
                    if _pv is None or pd.isna(_pv) or len(_pop) < 5:
                        continue
                    _mu = float(_pop.mean())
                    _sd = float(_pop.std()) or 1.0
                    _mapped = float(np.clip(
                        50.0 + (float(_pv) - _mu) / _sd * 25.0, 0.0, 100.0))
                    _rad.append((_lbl, _mapped, _g, float(_pv), _mu, _sd,
                                  _f, _pop))
                if len(_rad) >= 5:
                    from math import pi as _pi
                    _n = len(_rad)
                    _ang = [k / float(_n) * 2 * _pi for k in range(_n)]
                    _vals = [m for _, m, _g, _pv, _mu, _sd, _f, _pop in _rad]
                    _figr = plt.figure(figsize=(20, 10))
                    _figr.patch.set_facecolor((0.95, 0.92, 0.87))
                    _gsr = GridSpec(1, 2, width_ratios=[2.5, 1.2],
                                     figure=_figr)
                    _axr = plt.subplot(_gsr[0], polar=True)
                    _axr.set_facecolor((0.99, 0.98, 0.95))
                    _figr.subplots_adjust(top=0.80, bottom=0.08, left=0.03)
                    _axr.set_theta_offset(_pi / 2)    # first axis at 12 o'clock
                    _axr.set_theta_direction(-1)       # clockwise
                    _axr.set_xticks(_ang)
                    _axr.set_xticklabels([])
                    _axr.plot(_ang + _ang[:1], _vals + _vals[:1],
                               linewidth=2, linestyle='solid',
                               color='#0077b6', zorder=3)
                    _axr.fill(_ang + _ang[:1], _vals + _vals[:1],
                               '#0077b6', alpha=0.25, zorder=2)
                    _axr.set_rlabel_position(-180.0 / _n)
                    _axr.set_yticks([25, 50, 75, 100])
                    _axr.set_yticklabels(["", "", "", ""],
                                           color="grey", size=7)
                    _axr.set_ylim(0, 100)
                    # per-spoke gridline labels show RAW cohort values at
                    # -1σ / mean / +1σ / +2σ (traditional raw mode)
                    for _k, (_lbl, _m, _g, _pv, _mu, _sd, _f, _pop) in enumerate(_rad):
                        for _lvl, _sig in zip([25, 50, 75, 100],
                                                [-1, 0, 1, 2]):
                            _axr.text(_ang[_k], _lvl + 3,
                                       _f.format(_mu + _sig * _sd),
                                       size=7, ha='center', va='bottom',
                                       color='black')
                        _axr.text(_ang[_k], 116, _lbl, size=10,
                                   ha='center', va='center',
                                   color=_RAD_COLORS[_g], fontweight='bold')
                    _team_lbl = (str(_e.get('team'))
                                  if pd.notna(_e.get('team')) else '')
                    plt.figtext(0.04, 0.95,
                                 f"{_e['name']}"
                                 + (f" | {_team_lbl}" if _team_lbl else ''),
                                 fontsize=16, color='black', ha='left',
                                 weight='bold')
                    _disp_mins = (_e['mins_lineup']
                                   if pd.notna(_e.get('mins_lineup'))
                                   else _e['mins_played'])
                    plt.figtext(0.04, 0.905,
                                 f"{_e['role']} | {int(_disp_mins)} minutes"
                                 f" | ACP Index {_e['acp_rating']:.0f}"
                                 + (f" → projection {_e['projection']:.0f}"
                                    if pd.notna(_e.get('projection')) else '')
                                 + " | Raw per 90 vs role cohort (mean ± 2σ)",
                                 fontsize=12, color='black', ha='left')
                    _patches = [plt.Line2D([0], [0], color=c, lw=4)
                                for _, c in _RAD_LEGEND]
                    _figr.legend(_patches, [l for l, _ in _RAD_LEGEND],
                                  loc='upper right',
                                  bbox_to_anchor=(0.60, 0.99),
                                  frameon=False, fontsize=9)
                    # --- cohort distribution panels (right side) -------
                    _gsd = GridSpec(_n, 1, left=0.66, right=0.92,
                                     top=0.86, bottom=0.07, hspace=0.7,
                                     figure=_figr)
                    for _k, (_lbl, _m, _g, _pv, _mu, _sd, _f, _pop) in enumerate(_rad):
                        _axd = plt.subplot(_gsd[_k])
                        _axd.set_facecolor((0.99, 0.98, 0.95))
                        if len(_pop) > 1:
                            sns.kdeplot(_pop, ax=_axd, fill=True,
                                         color=_RAD_COLORS[_g], cut=0)
                        _pct = scipy.stats.percentileofscore(
                            _pop, _pv, kind='strict')
                        _lo = float(min(_pop.min(), _pv))
                        _hi = float(max(_pop.max(), _pv))
                        if _lo == _hi:
                            _lo, _hi = _lo - 0.1, _hi + 0.1
                        _axd.set_xlim(_lo, _hi)
                        _axd.set_xticks([_lo, _hi])
                        _axd.set_xticklabels([_f.format(_lo), _f.format(_hi)],
                                               fontsize=8)
                        _axd.axvline(_pv, color='blue', linestyle='--')
                        _sfx = get_percentile_suffix(int(_pct))
                        _axd.text(1.04, 0.5,
                                   f"%-tile: {int(_pct)}{_sfx}\n"
                                   f"value: {_f.format(_pv)}",
                                   transform=_axd.transAxes, fontsize=8,
                                   va='center')
                        _axd.set_yticks([])
                        _axd.set_ylabel('')
                        _axd.set_xlabel('')
                        _lgd = _axd.get_legend()
                        if _lgd is not None:
                            _lgd.remove()
                        _axd.text(-0.04, 0.5, _lbl, transform=_axd.transAxes,
                                   fontsize=9, fontweight='bold',
                                   va='center', ha='right')
                    plt.figtext(0.04, 0.035,
                                 f"Raw per-90 values vs {_e['role']} cohort "
                                 f"({_e['league']}, current season, 500+ mins) · "
                                 f"Aerial/Ground Def Grade = wins-above-expectation "
                                 f"+ opponent-adjusted duel ladder, 0-100 in cohort · "
                                 f"Off Duel Grade raw = take-on/shield Glicko · "
                                 f"RAPM = on-pitch xGD/90 · "
                                 f"Engine {_eng_meta.get('rating_version', '')} · "
                                 f"Data via Wyscout · @lucaskimball · "
                                 f"{datetime.date.today()}",
                                 ha='left', fontsize=9, color='black')
                    st.pyplot(_figr, use_container_width=True)
                    plt.close(_figr)
                # --- Projection outlook fan chart (prototype) ---
                try:
                    from pitch_interactive import plotly_projection_fan
                    _fan = plotly_projection_fan(
                        _erows, SEASON_ID_MAP, selected_player_name)
                    if _fan is not None:
                        st.plotly_chart(_fan, use_container_width=True,
                                        config={'displayModeBar': False})
                        st.caption(
                            "Career ratings by season (league and age on "
                            "each tick) flowing into next season's "
                            "projection. Shaded fan = ±1 SD of the "
                            "projection; a faded segment bridges seasons "
                            "missing from our data; green band = typical "
                            "peak ages for the role (literature-based "
                            "curve used by the projection model); "
                            "evidence % = how much data underwrites the "
                            "starting point. Cross-league careers plot on "
                            "the L3-equivalent scale (CAMP seasons "
                            "discounted by the league conversion — hover "
                            "a dot for the native rating).")
                except Exception:
                    logger.exception("Projection fan chart failed")
                st.caption(
                    f"Engine {_eng_meta.get('rating_version', '?')} · "
                    f"projection {_eng_meta.get('projection_version', '?')} · "
                    f"data through {_eng_meta.get('data_through', '?')} · "
                    f"percentiles within league × season × role cohort; "
                    f"set piece shown separately (not in the rating)")
        except Exception:
            logger.exception("Engine card failed")
        st.divider()

        # --- Exportable one-pager PDF ----------------------------------
        _op_cols = st.columns([1.4, 1.4, 3])
        with _op_cols[0]:
            _op_clicked = st.button("📄 Build one-pager PDF",
                                     key="onepager_build")
        if _op_clicked:
            try:
                with st.spinner("Composing one-pager…"):
                    from player_onepager import build_player_onepager
                    # 1) best-fit template radar (percentile mode)
                    _op_fig_radar = None
                    try:
                        _op_matches = player_stats_with_scores_df[
                            player_stats_with_scores_df['playerId']
                            == int(selected_player_id)]
                        if not _op_matches.empty:
                            _op_row = _op_matches.iloc[0]
                            _op_pos = _op_row.get('primaryPosition')
                            _op_elig = [r for r in WEIGHTS
                                        if _op_pos in POSITION_GROUPS.get(r, [])]
                            if _op_elig:
                                _op_role = max(_op_elig, key=lambda r: float(
                                    _op_row.get(f'{r}_Score', 0) or 0))
                                _op_pop = player_stats_with_scores_df[
                                    player_stats_with_scores_df['primaryPosition']
                                    .isin(POSITION_GROUPS.get(_op_role, [_op_pos]))]
                                if len(_op_pop) < 5:
                                    _op_pop = player_stats_with_scores_df
                                _op_metrics = [m for m in WEIGHTS[_op_role]
                                               if m in _op_row.index
                                               and m not in RADAR_HIDDEN_METRICS]
                                if _op_metrics:
                                    _op_seasons = _season_id_list(active_season_ids)
                                    _op_season_lbl = (
                                        SEASON_ID_MAP.get(_op_seasons[0], '')
                                        if len(_op_seasons) == 1 else 'All Seasons')
                                    _op_fig_radar = create_radar_with_distributions(
                                        pd.DataFrame([_op_row]), _op_metrics,
                                        _op_pos, _op_elig, _op_pop,
                                        full_df_for_ranking=player_stats_with_scores_df,
                                        season_label=_op_season_lbl,
                                        radar_mode='percentile')
                    except Exception:
                        logger.exception("one-pager radar failed")
                    # 2) shot map (light re-derivation of the shot log)
                    _op_fig_shots = None
                    try:
                        _op_shots = profile_events_df[
                            (profile_events_df['player.name'] == selected_player_name)
                            & (profile_events_df['type.primary'] == 'shot')].copy()
                        if not _op_shots.empty:
                            _op_shots = _op_shots.sort_values(
                                ['matchId', 'minute', 'second'])
                            _op_shots.reset_index(drop=True, inplace=True)
                            _op_shots['Shot Number'] = _op_shots.index + 1
                            _op_fig_shots = create_player_shotmap(
                                _op_shots, selected_player_name)
                    except Exception:
                        logger.exception("one-pager shotmap failed")
                    # 3) box-pass creativity map
                    _op_fig_passes = None
                    try:
                        _op_bp = load_box_passes()
                        if not _op_bp.empty:
                            _op_bp_seasons = _season_id_list(active_season_ids)
                            _op_bp = _op_bp[
                                _op_bp['player.id'] == int(selected_player_id)]
                            if _op_bp_seasons:
                                _op_bp = _op_bp[
                                    _op_bp['seasonId'].isin(_op_bp_seasons)]
                            if not _op_bp.empty:
                                _op_fig_passes = mpl_box_passes_map(
                                    _op_bp, selected_player_name)
                    except Exception:
                        logger.exception("one-pager box passes failed")
                    # 4) header tiles
                    _op_tiles = [("Team", current_team),
                                 ("Position", current_pos),
                                 ("Age", age_display)]
                    _op_footer = ""
                    try:
                        _op_prow = _erows[_erows['projection'].notna()] \
                            if not _erows.empty else pd.DataFrame()
                        if not _op_prow.empty:
                            _op_e = _op_prow.iloc[-1]
                            _op_tiles.append(
                                ("ACP Rating", f"{float(_op_e['acp_rating']):.0f}"))
                            _op_tiles.append(
                                ("Projection",
                                 f"{float(_op_e['projection']):.0f} "
                                 f"± {float(_op_e['band_sd']):.0f}"))
                        elif not _erows.empty:
                            _op_e = (_erows.sort_values('mins_played').iloc[-1])
                            _op_tiles.append(
                                ("ACP Rating", f"{float(_op_e['acp_rating']):.0f}"))
                        _op_footer = (
                            f"Engine {_eng_meta.get('rating_version', '?')} · "
                            f"projection {_eng_meta.get('projection_version', '?')} · "
                            f"generated {datetime.date.today().isoformat()}")
                    except Exception:
                        pass
                    _op_bytes = build_player_onepager(
                        selected_player_name,
                        f"{current_team} · {current_pos}",
                        _op_tiles, _op_fig_radar, _op_fig_shots,
                        _op_fig_passes, footer_note=_op_footer)
                    for _f in (_op_fig_radar, _op_fig_shots, _op_fig_passes):
                        if _f is not None:
                            plt.close(_f)
                    st.session_state['onepager_pdf'] = (
                        int(selected_player_id), _op_bytes)
            except Exception:
                logger.exception("one-pager build failed")
                st.error("Could not build the one-pager for this player.")
        _op_cached = st.session_state.get('onepager_pdf')
        if _op_cached and _op_cached[0] == int(selected_player_id):
            with _op_cols[1]:
                st.download_button(
                    "⬇️ Download one-pager",
                    data=_op_cached[1],
                    file_name=f"{selected_player_name.replace(' ', '_')}_onepager.pdf",
                    mime="application/pdf", key="onepager_dl")
        st.divider()

        # Hoisted so every lazy section can access it: the Stats section's
        # positional-peer population (radar_stats_df at ~10840) was previously
        # populated by the Player Radar tab body, which always ran under st.tabs.
        # With lazy if/elif rendering only one section runs, so compute the base
        # (player_stats_with_scores_df + DefR columns) up front. The Player Radar
        # section may still override radar_stats_df locally via its "Show Only
        # Position" filter for its own rendering.
        radar_stats_df = merge_defr_values_into_stats(
            player_stats_with_scores_df, active_season_ids, selected_comp_ids)

        _profile_sections = ["Player Radar", "Stats", "Value", "Shots & Creation", "Match Log"]
        _active_tab = st.radio("Profile section", _profile_sections,
                                horizontal=True, label_visibility="collapsed",
                                key="profile_active_tab")

        if _active_tab == "Player Radar":
            # --- Transfer Value Detail moved to just above Career Trajectory ---

           # --- 5. NEW: DISPLAY PLAYER RADAR ---
            st.subheader("Player Radar")

            # Gate: 300-minute minimum for radar charts
            _MIN_RADAR_MINUTES = 300
            _show_radar = total_minutes >= _MIN_RADAR_MINUTES
            if not _show_radar:
                st.info(f"⚠️ **Insufficient sample size** — {player_per_90_stats.get('playerName', 'This player')} has only played **{int(total_minutes)} minutes** this season.")

            # 1. Detect Raw Positions (What did they actually play?)
            try:
                player_events = profile_events_df[profile_events_df['player.id'] == player_id]
                if 'player.position' in player_events.columns:
                    raw_positions = player_events['player.position'].unique()
                elif 'position_name' in player_events.columns:
                    raw_positions = player_events['position_name'].unique()
                else:
                    raw_positions = []
            
                # Filter out None/Nan
                raw_positions = [x for x in raw_positions if x and str(x) != 'nan']

            except Exception:
                logger.exception("position extraction failed")
                raw_positions = []
            
            # Ensure we at least have the primary position from the bio
            if current_pos and current_pos not in raw_positions:
                raw_positions.append(current_pos)
            
            # Sort for the dropdown
            raw_positions = sorted([str(p) for p in raw_positions])

            # 2. Position Selector (Simple Raw Codes)
            col_rad_sel1, col_rad_sel2 = st.columns([1, 3])
            with col_rad_sel1:
                st.markdown("##### Show Radar For:")
            with col_rad_sel2:
                selected_raw_pos = st.selectbox(
                    "Select Position:",
                    raw_positions,
                    label_visibility="collapsed",
                    key="radar_pos_selector"
                )

            # --- Show Only Position toggle ---
            # When active, recompute stats using only events at the selected position
            profile_pos_filter = st.checkbox(
                "Show Only Position",
                key="profile_pos_played_filter",
                help=f"When checked, the radar uses only events where the player played as **{selected_raw_pos}** (selected above), instead of all events."
            )
            radar_stats_df = player_stats_with_scores_df
            radar_player_data_row = player_data_row
            if profile_pos_filter and 'player.position' in profile_events_df.columns:
                pos_filtered_events = profile_events_df[
                    profile_events_df['player.position'] == selected_raw_pos
                ]
                if not pos_filtered_events.empty:
                    # Build position-adjusted minutes: replace totalMinutes with minutes at this position
                    all_pos_minutes = get_all_players_minutes_by_position(profile_events_df)
                    pos_minutes = all_pos_minutes[all_pos_minutes['Position'] == selected_raw_pos][['playerId', 'Minutes']]
                    pos_player_minutes_df = profile_player_minutes_df.copy()
                    pos_player_minutes_df = pos_player_minutes_df.merge(pos_minutes, on='playerId', how='inner')
                    pos_player_minutes_df['totalMinutes'] = pos_player_minutes_df['Minutes']
                    pos_player_minutes_df = pos_player_minutes_df.drop(columns=['Minutes'])

                    # Use a distinct season_id so the cache differentiates from unfiltered stats
                    pos_cache_key = f"{selected_season_id}_pos_{selected_raw_pos}"
                    pos_filtered_stats = calculate_all_player_stats(
                        pos_filtered_events, pos_player_minutes_df, season_id=pos_cache_key
                    )
                    # Merge GPA Value columns (same season × competition scope as profile)
                    pos_filtered_stats = merge_gpa_values_into_stats(pos_filtered_stats, active_season_ids, selected_comp_ids)
                    pos_filtered_scores = calculate_player_percentiles_and_scores(
                        pos_filtered_stats, POSITION_GROUPS, WEIGHTS, INVERT_METRICS, min_minutes=500, season_id=pos_cache_key
                    )
                    if not pos_filtered_scores.empty:
                        radar_stats_df = pos_filtered_scores
                        pos_player_row = pos_filtered_scores[pos_filtered_scores['playerId'] == player_id]
                        if not pos_player_row.empty:
                            radar_player_data_row = pos_player_row

            # Ensure the radar population carries DefR columns (the percentile
            # function is disk-cached and may drop them) — used by the radar
            # population, the DefR-mode toggle, and Overall Season Stats coloring.
            radar_stats_df = merge_defr_values_into_stats(
                radar_stats_df, active_season_ids, selected_comp_ids)

            # 3. Find the "Best Fit" Template for this Raw Position
            # (e.g. If 'CF' is selected, check 'Target Man', 'Poacher', etc. and pick the best one)
        
            # Find all roles that include this raw position
            eligible_roles = []
            for role, valid_codes in POSITION_GROUPS.items():
                if selected_raw_pos in valid_codes:
                    eligible_roles.append(role)
        
            if not eligible_roles:
                st.warning(f"No radar templates defined for position '{selected_raw_pos}'.")
            else:
                # Calculate Scores to find the best fit
                best_role = None
                best_score = -1
            
                # We calculate a simple score (sum of percentiles) for each eligible role
                for role in eligible_roles:
                    # Get metrics and weights
                    role_weights = WEIGHTS.get(role, {})
                    if not role_weights: continue
                
                    # Define Population for this role (for percentile calculation)
                    role_codes = POSITION_GROUPS[role]
                    population = radar_stats_df[
                        radar_stats_df['primaryPosition'].isin(role_codes)
                    ]
                    if len(population) < 5: population = radar_stats_df # Fallback

                    # Calculate Score
                    role_score = 0
                    total_weight = 0

                    for metric, weight in role_weights.items():
                        if metric in radar_player_data_row.columns and metric in population.columns:
                            val = radar_player_data_row[metric].values[0]
                            pop_vals = population[metric].fillna(0)
                        
                            # Percentile
                            pct = (pop_vals < val).mean()
                            if metric in INVERT_METRICS: pct = 1.0 - pct
                        
                            role_score += (pct * weight)
                            total_weight += weight
                
                    final_score = (role_score / total_weight) if total_weight > 0 else 0
                
                    if final_score > best_score:
                        best_score = final_score
                        best_role = role
            
                # Handle case where no score could be calculated
                if best_role is None: best_role = eligible_roles[0]

                # 4. Generate Chart for the Winner
                st.caption(f"Best Template Match: **{best_role}**")
                _radar_style = st.radio("Radar Style", ["Percentile", "Raw Values (mean ± 2σ)"], horizontal=True, key=f"radar_style_{player_id}")
                _use_defr = st.toggle(
                    "Defensive metrics → DefR",
                    value=False, key=f"defr_mode_{player_id}",
                    help="Swap each defensive axis (tackles, interceptions, recoveries, "
                         "clearances, aerials) to its Defensive Responsibility value — "
                         "actions above/below what the player's role is expected to make.")

                # Prepare data for plotting
                metrics_to_plot = list(WEIGHTS[best_role].keys())
                metrics_to_plot = [m for m in metrics_to_plot
                                   if m in radar_player_data_row.columns
                                   and m not in RADAR_HIDDEN_METRICS]

                # Get Population for distribution
                final_population = radar_stats_df[
                    radar_stats_df['primaryPosition'].isin(POSITION_GROUPS[best_role])
                ]
                if len(final_population) < 5: final_population = radar_stats_df

                # --- DefR mode: swap each defensive axis to its DefR value ---
                if _use_defr:
                    final_population = merge_defr_values_into_stats(
                        final_population, active_season_ids, selected_comp_ids)
                    radar_player_data_row = merge_defr_values_into_stats(
                        radar_player_data_row, active_season_ids, selected_comp_ids)
                    _mapped, _seen = [], set()
                    for _m in metrics_to_plot:
                        _mm = DEFR_RADAR_MAP.get(_m, _m)
                        if _mm in radar_player_data_row.columns and _mm not in _seen:
                            _mapped.append(_mm); _seen.add(_mm)
                    if _mapped:
                        metrics_to_plot = _mapped
                    # Percentile for each DefR axis vs same-position peers
                    for _m in metrics_to_plot:
                        if _m in DEFR_DISPLAY_METRICS \
                                and _m in radar_player_data_row.columns \
                                and _m in final_population.columns:
                            _pv = final_population[_m].dropna()
                            if not _pv.empty:
                                _val = radar_player_data_row[_m].values[0]
                                radar_player_data_row[_m + '_percentile'] = (
                                    scipy.stats.percentileofscore(_pv, _val, kind='weak') / 100.0)
                    if _mapped:
                        st.caption("🛡️ Defensive axes show **DefR** (actions above/below role expectation).")

                # --- NEW: Recalculate percentiles and scores for ALL eligible roles ---
                # This ensures that if the user selects a raw position that maps to multiple templates
                # (e.g., 'CF' -> Mobile Striker, Poacher, etc.), ALL those scores are updated
                # based on the new comparison group.
                radar_player_data_row = radar_player_data_row.copy()

                for role in eligible_roles:
                    # 1. Get Population for this specific role
                    role_population = radar_stats_df[
                        radar_stats_df['primaryPosition'].isin(POSITION_GROUPS[role])
                    ]
                    if len(role_population) < 5: role_population = radar_stats_df

                    # 2. Get metrics and weights for this role
                    role_weights = WEIGHTS.get(role, {})
                    new_total_score = 0
                    total_weight = 0

                    # 3. Recalculate percentiles for all metrics used in this role
                    for metric, weight in role_weights.items():
                        if metric not in role_population.columns:
                            continue
                        # Get population values for this metric
                        pop_values = role_population[metric].dropna()

                        if metric in radar_player_data_row.columns:
                            player_val = radar_player_data_row[metric].values[0]
                        
                            if not pop_values.empty:
                                # Calculate percentile (0-100)
                                pct_score = scipy.stats.percentileofscore(pop_values, player_val, kind='weak')
                            
                                # Handle Inverted Metrics
                                if metric in INVERT_METRICS:
                                    pct_score = 100.0 - pct_score
                            
                                # Update the row's percentile column (0-1) used for plotting
                                # Note: This overwrites the column. If multiple roles use the same metric,
                                # the last one wins. This is generally acceptable as they are usually 
                                # compared against similar populations if they share a raw position.
                                # Ideally, we'd plot based on the 'best_role' metrics specifically.
                                radar_player_data_row[metric + '_percentile'] = pct_score / 100.0

                                # Add to weighted score
                                new_total_score += ((pct_score / 100.0) * weight)
                                total_weight += weight

                    # 4. Update the Role Score column
                    if total_weight > 0:
                        final_new_score = (new_total_score / total_weight) * 100
                        radar_player_data_row[role + '_Score'] = final_new_score

                # --- NEW: Update Position Label for Chart ---
                # This ensures the chart displays "CF" if we selected "CF", even if their bio says "RW"
                radar_player_data_row['primaryPosition'] = selected_raw_pos
                # -----------------------------------------------------------------------

                # Plot
                _radar_season_label = SEASON_ID_MAP.get(selected_season_id, 'All Seasons') if selected_season_id else 'All Seasons'
                _radar_mode = 'raw' if _radar_style == "Raw Values (mean ± 2σ)" else 'percentile'
                fig_radar = create_radar_with_distributions(
                    radar_player_data_row,
                    metrics_to_plot,
                    best_role,
                    eligible_roles,
                    all_position_data=final_population,
                    full_df_for_ranking=radar_stats_df,
                    season_label=_radar_season_label,
                    radar_mode=_radar_mode
                )
                st.pyplot(fig_radar, use_container_width=True)

            # Career radar section disabled to reduce memory usage
            # TODO: Re-enable when Streamlit Cloud resources are upgraded

            # --- Minutes by Position + Outlier Stats side-by-side ---
            _col_mins, _col_outliers = st.columns([1, 2])

            with _col_mins:
                minutes_by_pos = get_player_minutes_by_position(profile_events_df, player_id, player_match_log_df)
                if not minutes_by_pos.empty and len(minutes_by_pos) > 1:
                    st.caption("Minutes by Position")
                    st.dataframe(
                        minutes_by_pos,
                        column_config={
                            "Position": st.column_config.TextColumn("Position"),
                            "Minutes": st.column_config.NumberColumn("Minutes", format="%d"),
                            "Percentage": st.column_config.ProgressColumn(
                                "% of Minutes",
                                min_value=0,
                                max_value=100,
                                format="%.1f%%",
                            ),
                        },
                        hide_index=True,
                        use_container_width=False,
                    )

            with _col_outliers:
                # Compute outlier stats: metrics beyond 2σ / 3σ from positional mean
                try:
                    # Use the best role's position group as the population
                    _outlier_pop = radar_stats_df[
                        radar_stats_df['primaryPosition'].isin(POSITION_GROUPS.get(best_role, [selected_raw_pos]))
                    ].copy() if 'best_role' in dir() and best_role else radar_stats_df.copy()
                    if 'totalMinutes' in _outlier_pop.columns:
                        _outlier_pop = _outlier_pop[pd.to_numeric(_outlier_pop['totalMinutes'], errors='coerce').fillna(0) >= 500]

                    # Gather all numeric per-90 metrics (exclude info/intermediate columns)
                    _skip_cols = {'playerName', 'teamName', 'totalMinutes', 'primaryPosition', 'secondaryPosition',
                                  'tertiaryPosition', 'playerId', 'player.id', 'Defensive Area', 'Expected xT at Center',
                                  'competitionId', 'competitionId_per_90'}
                    _skip_suffixes = ('_percentile', '_Score', '_TotalScore', '_Rank')
                    _all_metrics = [c for c in radar_player_data_row.columns
                                    if pd.api.types.is_numeric_dtype(radar_player_data_row[c])
                                    and c not in _skip_cols
                                    and not c.endswith(_skip_suffixes)]

                    _outliers_2s = []  # (metric, value, z_score, direction)
                    _outliers_3s = []
                    for _m in _all_metrics:
                        if _m not in _outlier_pop.columns:
                            continue
                        _pop_vals = pd.to_numeric(_outlier_pop[_m], errors='coerce').dropna()
                        if len(_pop_vals) < 5:
                            continue
                        _mean = _pop_vals.mean()
                        _std = _pop_vals.std()
                        if _std == 0:
                            continue
                        _player_val = float(radar_player_data_row[_m].values[0])
                        _z = (_player_val - _mean) / _std
                        # For inverted metrics, negative z is "good" (below average = better)
                        if _m in INVERT_METRICS:
                            _direction = "⬇️" if _z < 0 else "⬆️"
                        else:
                            _direction = "⬆️" if _z > 0 else "⬇️"
                        if abs(_z) >= 3:
                            _outliers_3s.append((_m, _player_val, _z, _direction))
                        elif abs(_z) >= 2:
                            _outliers_2s.append((_m, _player_val, _z, _direction))

                    # Sort by absolute z-score descending
                    _outliers_3s.sort(key=lambda x: abs(x[2]), reverse=True)
                    _outliers_2s.sort(key=lambda x: abs(x[2]), reverse=True)

                    if _outliers_3s or _outliers_2s:
                        st.caption("Statistical Outliers (vs. positional avg)")
                        if _outliers_3s:
                            st.markdown("**🔴 Super-outliers (> 3σ)**")
                            for _m, _v, _z, _d in _outliers_3s:
                                st.markdown(f"&nbsp;&nbsp;{_d} **{_m}**: {fmt_val(_m, _v)} p90 &nbsp;({_z:+.1f}σ)")
                        if _outliers_2s:
                            st.markdown("**🟡 Outliers (> 2σ)**")
                            for _m, _v, _z, _d in _outliers_2s:
                                st.markdown(f"&nbsp;&nbsp;{_d} **{_m}**: {fmt_val(_m, _v)} p90 &nbsp;({_z:+.1f}σ)")
                    else:
                        st.caption("No statistical outliers (all metrics within 2σ of positional mean)")
                except Exception as e:
                    print(f"Warning: Could not compute outlier stats: {e}")



        elif _active_tab == "Stats":
            # --- 5b. Career Trajectory ----------------------------------------
            # Per-season summary of the player's appearances + per-season
            # strip plots of the chosen metric (Action V/90 or Best-fit
            # Rating) showing the full league-wide distribution with the
            # selected player highlighted. Lives just above Overall Season
            # Stats so the trajectory context flows into the per-season
            # breakdown right below it.
            st.subheader("Career Trajectory")

            from plotly.subplots import make_subplots as _make_subplots

            def _season_start_year(season_label: str) -> int | None:
                """Parse a season label like '2025/26' or '2025/2026' and
                return the start year. Used to sort seasons chronologically
                because raw seasonIds aren't comparable across competitions
                (e.g. Camp 23/24 seasonId 190230 sorts AFTER Liga 3 24/25
                seasonId 190090 by numeric value)."""
                try:
                    return int(str(season_label).split('/')[0])
                except (ValueError, AttributeError, IndexError):
                    return None

            def _age_at_season_label(birth, season_label) -> float | None:
                """Age at midpoint of the season (start year + 0.5)."""
                if not birth or pd.isna(birth):
                    return None
                try:
                    from datetime import datetime
                    bd = pd.to_datetime(birth, errors='coerce')
                    if pd.isna(bd):
                        return None
                    start_year = _season_start_year(season_label)
                    if start_year is None:
                        return None
                    mid_season = datetime(start_year, 12, 31)
                    return round((mid_season - bd.to_pydatetime()).days / 365.25, 1)
                except Exception:
                    return None

            # ---- 1) Walk player_minutes_data to build per-season rows ----
            career_rows = []
            birth = player_bio.get('birthDate') if isinstance(player_bio, pd.Series) else None
            for _sid, _pm_df in player_minutes_data.items():
                if not isinstance(_pm_df, pd.DataFrame) or _pm_df.empty:
                    continue
                if 'playerId' not in _pm_df.columns:
                    continue
                sub = _pm_df[_pm_df['playerId'] == player_id]
                if sub.empty:
                    continue
                for _, row in sub.iterrows():
                    _comp_id = competition_for_season(_sid)
                    _season_label = SEASON_ID_MAP.get(_sid, str(_sid))
                    _comp_name = (COMPETITIONS.get(_comp_id, {}).get('name')
                                   if _comp_id else 'Other')
                    career_rows.append({
                        'Season': _season_label,
                        '_seasonId': _sid,
                        '_startYear': _season_start_year(_season_label) or 0,
                        'Competition': _comp_name or 'Other',
                        '_compId': _comp_id,
                        'Team': row.get('teamName', 'N/A'),
                        'Position': row.get('primaryPosition', 'N/A'),
                        'Minutes': int(row.get('totalMinutes', 0) or 0),
                        'Age': _age_at_season_label(birth, _season_label),
                    })

            if not career_rows:
                st.caption("No multi-season history available for this player.")
            else:
                # Chronological sort: by parsed start year, then by competition
                # id as a stable tiebreaker (so Liga 3 23/24 + Camp 23/24
                # land next to each other deterministically).
                career_df = (pd.DataFrame(career_rows)
                              .sort_values(['_startYear', '_compId', '_seasonId']))

                # ---- 2) Merge in GPA Total Value per season for the player ----
                _gpa_full = None
                try:
                    _gpa_full = load_gpa_values()
                    if _gpa_full is not None and not _gpa_full.empty \
                            and 'playerId' in _gpa_full.columns:
                        _gpa_player = _gpa_full[_gpa_full['playerId'] == player_id].copy()
                        if not _gpa_player.empty:
                            _val_col = next((c for c in ('Total Value', 'total_v_per_90')
                                              if c in _gpa_player.columns), None)
                            if _val_col:
                                career_df = career_df.merge(
                                    _gpa_player[['seasonId', _val_col]]
                                        .rename(columns={'seasonId': '_seasonId',
                                                          _val_col: 'Action V/90'})
                                        .drop_duplicates('_seasonId'),
                                    on='_seasonId', how='left'
                                )
                except Exception:
                    pass

                # ---- 3) Per-season best-fit rating + per-season population ----
                # For each season the player has data in, also collect the
                # FULL league-wide distribution of the metric so we can plot
                # a violin per season with the player highlighted.
                _role_rating_by_season: dict[int, float] = {}
                _role_name_by_season:   dict[int, str]   = {}
                _rating_population: dict[int, np.ndarray] = {}  # sid -> array of all players' best-fit scores

                def _best_fit_score(row, _weights, _groups):
                    pos = row.get('primaryPosition')
                    if pd.isna(pos):
                        return None
                    eligible = [r for r in _weights if pos in _groups.get(r, [])]
                    vals = [row.get(f"{r}_Score") for r in eligible
                             if f"{r}_Score" in row.index]
                    vals = [float(v) for v in vals if v is not None and not pd.isna(v)]
                    return max(vals) if vals else None

                def _position_group_of(pos):
                    """Map a Wyscout primaryPosition to its top-level position
                    group (GK/CB/FB/CM/AM/WG/ST). Best-fit Rating is a
                    position-specific composite, so the per-season violin
                    should compare against same-position peers only — not
                    fullbacks vs centerbacks vs forwards."""
                    if pos is None or pd.isna(pos):
                        return None
                    p = str(pos)
                    if p == 'GK': return 'GK'
                    if p in ('CB', 'LCB', 'RCB', 'LCB3', 'RCB3'): return 'CB'
                    if p in ('LB', 'RB', 'LB5', 'RB5', 'LWB', 'RWB'): return 'FB'
                    if p in ('CMF', 'LCMF', 'RCMF', 'LCMF3', 'RCMF3',
                              'DMF', 'LDMF', 'RDMF'): return 'CM'
                    if p in ('AMF', 'LAMF', 'RAMF', 'LMF', 'RMF'): return 'AM'
                    if p in ('LW', 'RW', 'LWF', 'RWF'): return 'WG'
                    if p in ('CF', 'SS'): return 'ST'
                    return None

                try:
                    for _sid in career_df['_seasonId'].unique():
                        _events_sid = get_season_events(raw_events_df, [_sid])
                        _minutes_sid = player_minutes_data.get(_sid)
                        if _events_sid.empty or _minutes_sid is None or _minutes_sid.empty:
                            continue
                        _stats_sid = calculate_all_player_stats(
                            _events_sid, _minutes_sid, season_id=_sid
                        )
                        if _stats_sid.empty:
                            continue
                        _scores_sid = calculate_player_percentiles_and_scores(
                            _stats_sid, POSITION_GROUPS, WEIGHTS, INVERT_METRICS,
                            min_minutes=500, season_id=_sid
                        )
                        if _scores_sid.empty:
                            continue
                        # Per-player best-fit role score (vectorized via apply).
                        _scores_sid = _scores_sid.copy()
                        _scores_sid['_best_fit'] = _scores_sid.apply(
                            _best_fit_score, axis=1,
                            args=(WEIGHTS, POSITION_GROUPS),
                        )
                        # The selected player's row — needed up-front so we
                        # can filter the population to same-position peers.
                        _player_rows = _scores_sid[_scores_sid['playerId'] == player_id]
                        if _player_rows.empty:
                            continue
                        _player_row = _player_rows.iloc[0]
                        _pos = _player_row.get('primaryPosition')
                        _player_pos_group = _position_group_of(_pos)

                        # Population for violin = ≥500-min players in the
                        # SAME position group as the selected player this
                        # season. Best-fit Rating is position-specific so
                        # comparing a CB to wingers isn't meaningful.
                        if 'totalMinutes' in _scores_sid.columns:
                            _qualified = _scores_sid[
                                (_scores_sid['totalMinutes'].fillna(0) >= 500)
                                & _scores_sid['_best_fit'].notna()
                            ]
                        else:
                            _qualified = _scores_sid[_scores_sid['_best_fit'].notna()]
                        if _player_pos_group and 'primaryPosition' in _qualified.columns:
                            _qualified = _qualified[
                                _qualified['primaryPosition'].map(_position_group_of)
                                == _player_pos_group
                            ]
                        _pop = _qualified['_best_fit'].values
                        if len(_pop) > 0:
                            _rating_population[int(_sid)] = _pop
                        _eligible = [r for r in WEIGHTS
                                      if _pos in POSITION_GROUPS.get(r, [])]
                        _scored = [(r, float(_player_row.get(f"{r}_Score", 0) or 0))
                                    for r in _eligible
                                    if f"{r}_Score" in _player_row.index]
                        if not _scored:
                            continue
                        _best_role, _best_score = max(_scored, key=lambda t: t[1])
                        _role_rating_by_season[int(_sid)] = round(_best_score, 1)
                        _role_name_by_season[int(_sid)] = _best_role
                except Exception as _exc:
                    logger.warning(f"Could not compute per-season role ratings: {_exc}")

                if _role_rating_by_season:
                    career_df['Best-fit Rating'] = career_df['_seasonId'].map(_role_rating_by_season)
                    career_df['Best-fit Role']   = career_df['_seasonId'].map(_role_name_by_season)

                # ---- 4) Display per-season table ----
                _display_career = career_df.drop(columns=[c for c in career_df.columns if c.startswith('_')])
                if 'Action V/90' in _display_career.columns:
                    _display_career['Action V/90'] = _display_career['Action V/90'].round(3)
                st.dataframe(_display_career, use_container_width=True, hide_index=True, column_config=auto_column_config(_display_career))

                # ---- 5) Per-season strip plots with player highlighted ----
                _chart_y_candidates = [m for m in ('Action V/90', 'Best-fit Rating')
                                        if m in career_df.columns
                                        and career_df[m].notna().any()]
                if not _chart_y_candidates:
                    st.caption(
                        "No Action V/90 or Best-fit Rating data available across "
                        "this player's seasons — chart suppressed."
                    )
                else:
                    chart_y = st.radio(
                        "Trajectory metric:",
                        _chart_y_candidates,
                        horizontal=True,
                        key=f"_traj_metric_{player_id}",
                    )
                    # Build (season, label, population, player_value) list in
                    # chronological order. Skip seasons where the player has
                    # no value for the chosen metric — the violin without a
                    # highlighted dot is just visual noise.
                    _panels = []
                    _seen_sids = set()
                    for _, _row in career_df.iterrows():
                        _sid = int(_row['_seasonId'])
                        if _sid in _seen_sids:
                            continue  # de-dupe rows where player had multi teams
                        _seen_sids.add(_sid)
                        _pv = _row.get(chart_y)
                        if pd.isna(_pv):
                            continue
                        if chart_y == 'Action V/90' and _gpa_full is not None:
                            _val_col = next((c for c in ('Total Value', 'total_v_per_90')
                                              if c in _gpa_full.columns), None)
                            if _val_col:
                                # Filter to ≥500-min sample so the population
                                # represents proper regular-rotation players.
                                _gpa_season = _gpa_full.loc[
                                    (_gpa_full['seasonId'] == _sid)
                                    & (_gpa_full.get('mins_played', 0) >= 500)
                                ]
                                # Filter to same position group as the
                                # selected player in this season. Prefer
                                # GPA's own position_group column (source-
                                # of-truth for this dataset); fall back to
                                # deriving from the `position` column via
                                # _position_group_of if missing.
                                _player_pg = None
                                _gpa_player_row = _gpa_full[
                                    (_gpa_full['playerId'] == player_id)
                                    & (_gpa_full['seasonId'] == _sid)
                                ]
                                if not _gpa_player_row.empty:
                                    if 'position_group' in _gpa_player_row.columns:
                                        _v = _gpa_player_row['position_group'].iloc[0]
                                        if pd.notna(_v):
                                            _player_pg = str(_v)
                                    if _player_pg is None and 'position' in _gpa_player_row.columns:
                                        _player_pg = _position_group_of(
                                            _gpa_player_row['position'].iloc[0]
                                        )
                                if _player_pg:
                                    if 'position_group' in _gpa_season.columns:
                                        _gpa_season = _gpa_season[
                                            _gpa_season['position_group'] == _player_pg
                                        ]
                                    elif 'position' in _gpa_season.columns:
                                        _gpa_season = _gpa_season[
                                            _gpa_season['position'].map(_position_group_of)
                                            == _player_pg
                                        ]
                                _pop = _gpa_season[_val_col].dropna().values
                            else:
                                _pop = np.array([])
                        else:
                            _pop = _rating_population.get(_sid, np.array([]))
                        if len(_pop) < 5:
                            continue
                        _age_str = (f" · age {_row['Age']:.0f}"
                                     if pd.notna(_row.get('Age')) else "")
                        _comp_short = ('L3' if _row.get('_compId') == 43324
                                        else 'CP' if _row.get('_compId') == 702
                                        else (_row.get('Competition') or '')[:6])
                        _panels.append({
                            'sid': _sid,
                            'label': f"{_row['Season']}<br>{_comp_short}{_age_str}",
                            'population': _pop,
                            'player_value': float(_pv),
                            'team': _row.get('Team', ''),
                        })

                    if not _panels:
                        st.caption(
                            f"No seasons have both a {chart_y} value for this "
                            f"player AND a comparable population to plot."
                        )
                    else:
                        # One subplot per season, shared y-axis so the player's
                        # trajectory is easy to follow across seasons.
                        _fig = _make_subplots(
                            rows=1, cols=len(_panels),
                            shared_yaxes=True,
                            subplot_titles=[p['label'] for p in _panels],
                            horizontal_spacing=0.01,
                        )
                        # Common y-range — driven by the ≥500-min POPULATION
                        # only (per user request). If the highlighted player
                        # is outside that range we extend slightly so the gold
                        # dot is still visible, but the scale is anchored to
                        # the regular-rotation population.
                        _pop_concat = np.concatenate([p['population'] for p in _panels])
                        _y_lo = float(np.nanmin(_pop_concat))
                        _y_hi = float(np.nanmax(_pop_concat))
                        _y_pad = 0.05 * (_y_hi - _y_lo if _y_hi > _y_lo else 1.0)
                        # Allow the highlighted dot to bleed up to 1 pad
                        # outside the population range without rescaling the
                        # whole panel.
                        _player_vals = [p['player_value'] for p in _panels]
                        _y_lo = min(_y_lo, float(np.nanmin(_player_vals)) - _y_pad)
                        _y_hi = max(_y_hi, float(np.nanmax(_player_vals)) + _y_pad)
                        # Width scaling: seasons with more ≥500-min players get
                        # visibly wider violins so the user can tell apart a
                        # 200-player Liga 3 season from a 600-player Camp
                        # season. Use a power < 1 so small samples don't
                        # collapse to slivers.
                        _max_n = max(len(p['population']) for p in _panels) or 1
                        for _i, _p in enumerate(_panels, start=1):
                            _n = len(_p['population'])
                            _scaled_w = 0.85 * (_n / _max_n) ** 0.5
                            # 1) Violin: density shape, no built-in points (we
                            # render our own colored dots in the next trace).
                            # Explicit x=0 anchor — without this, plotly
                            # picks categorical mode and the highlight dot's
                            # numeric x positioning silently breaks.
                            _fig.add_trace(go.Violin(
                                x=np.zeros(len(_p['population'])),
                                y=_p['population'],
                                points=False,
                                box_visible=False,
                                meanline_visible=False,
                                side='both',
                                width=_scaled_w,
                                line_color='rgba(80,80,80,0.55)',
                                fillcolor='rgba(140,140,140,0.18)',
                                showlegend=False,
                                hoverinfo='skip',
                                name='',
                            ), row=1, col=_i)

                            # 2) Colored dots: deterministic jitter so the
                            # layout doesn't shift on rerun (seeded by sid),
                            # bounded so dots stay inside the violin.
                            _rng = np.random.default_rng(seed=int(_p['sid']) & 0xFFFFFFFF)
                            _half = max(0.05, _scaled_w / 2 - 0.04)
                            _jitter = _rng.uniform(-_half, _half, size=_n)
                            _fig.add_trace(go.Scatter(
                                x=_jitter,
                                y=_p['population'],
                                mode='markers',
                                marker=dict(
                                    size=5,
                                    color=_p['population'],
                                    colorscale='RdYlGn',
                                    cmin=_y_lo, cmax=_y_hi,
                                    opacity=0.6,
                                    line=dict(width=0),
                                    showscale=False,
                                ),
                                showlegend=False,
                                hoverinfo='y',
                                name='',
                            ), row=1, col=_i)

                            # 3) Highlight: the selected player's point on top.
                            # Bumped to a larger diamond with a thicker dark
                            # ring so it stays unmistakable against the
                            # dense violin shape.
                            _fig.add_trace(go.Scatter(
                                y=[_p['player_value']],
                                x=[0.0],
                                mode='markers',
                                marker=dict(
                                    size=18,
                                    color='#FFC400',
                                    line=dict(color='black', width=2),
                                    symbol='diamond',
                                ),
                                showlegend=False,
                                hovertemplate=(
                                    f"<b>{selected_player_name}</b><br>"
                                    f"Team: {_p['team']}<br>"
                                    f"{chart_y}: %{{y:.2f}}<extra></extra>"
                                ),
                                name='',
                            ), row=1, col=_i)
                            # Force the x-axis to linear — adding go.Violin
                            # without an explicit x array flips Plotly into
                            # categorical-axis mode, which silently clobbers
                            # the numeric x positions we use for jitter and
                            # the highlight dot (so they collapse onto a
                            # single phantom category and the gold dot got
                            # buried under the colored cloud).
                            _fig.update_xaxes(
                                type='linear',
                                showticklabels=False, zeroline=False,
                                range=[-0.5, 0.5], row=1, col=_i,
                            )
                        _fig.update_yaxes(range=[_y_lo - _y_pad, _y_hi + _y_pad])
                        _fig.update_layout(
                            title=f"{chart_y} by season (gold dot = {selected_player_name})",
                            height=420,
                            margin=dict(t=70, b=30, l=40, r=20),
                            showlegend=False,
                        )
                        # Make subplot titles smaller so they fit when there
                        # are 5+ panels.
                        for _ann in _fig['layout']['annotations']:
                            _ann['font'] = dict(size=10)
                        st.plotly_chart(_fig, use_container_width=True)

            st.divider()

            # --- 6. STATS TOGGLE ---
            st.subheader("Overall Season Stats")
            show_totals = st.toggle("Show Season Totals", value=False)
            stats_to_display = pd.Series(dtype='object')
        
            per_90_stats = player_per_90_stats.copy()
        
            if show_totals:
                st.text(f"Displaying TOTAL stats from {total_minutes:.0f} minutes played.")
                total_stats = per_90_stats.copy()
                rate_cols = [col for col in total_stats.index if '%' in col or 'per' in col.lower() or 'index' in col or 'Percentage' in col]
                # goalsConceded is conceptually a defensive rate (goals against per 90)
                # rather than a count — it stays per-90 even in season-totals mode,
                # the same way 'goalsPrevented' / xG family already do.
                if 'goalsConceded' in total_stats.index and 'goalsConceded' not in rate_cols:
                    rate_cols.append('goalsConceded')
                # engine metrics are levels, never season-totaled
                for _ec in ENGINE_DISPLAY_METRICS:
                    if _ec in total_stats.index and _ec not in rate_cols:
                        rate_cols.append(_ec)
            
                for col in total_stats.index:
                    if col not in rate_cols and pd.api.types.is_numeric_dtype(total_stats[col]):
                        total_val = (total_stats[col] * total_minutes) / 90
                        if col in ['xG', 'xA', 'xT', 'xTOP', 'xTSP', 'npxG', 'xAOP', 'xASP', 'psxG_faced', 'goalsPrevented']:
                             total_stats[col] = total_val
                        else:
                             total_stats[col] = np.round(total_val)
            
                for col in rate_cols:
                    if col in per_90_stats.index:
                        total_stats[col] = per_90_stats[col]
            
                stats_to_display = total_stats
            
            else: # Show Per 90
                st.text(f"Displaying PER 90 stats from {total_minutes:.0f} minutes played.")
                stats_to_display = per_90_stats

            # --- 7. Display Stats (Using all global groups) ---
            stat_groups = {
                "Output": OUTPUT_METRICS,
                "Passing": PASSING_METRICS,
                "Defensive": DEFENSIVE_METRICS,
                "Defensive Responsibility (DefR)": DEFR_DISPLAY_METRICS,
                "Dribbling": DRIBBLING_METRICS,
                "ACP Index": ENGINE_DISPLAY_METRICS,
                "Goalkeeping": GOALKEEPING_METRICS
            }

            player_is_gk = (per_90_stats.get('primaryPosition', 'N/A') == 'GK')

            # ── Build positional-peer population for percentile comparisons ──
            # Union of all raw positions that share at least one role template
            # with the player's primary position; filter to 500+ minutes.
            _primary_pos = player_per_90_stats.get('primaryPosition', None)
            _peer_pop = pd.DataFrame()
            if _primary_pos and _primary_pos not in ('N/A', 'Unknown', None, ''):
                _peer_positions = set()
                for _role, _positions in POSITION_GROUPS.items():
                    if _primary_pos in _positions:
                        _peer_positions.update(_positions)
                if not _peer_positions:
                    _peer_positions = {_primary_pos}
                _peer_pop = radar_stats_df[radar_stats_df['primaryPosition'].isin(_peer_positions)]
                if 'totalMinutes' in _peer_pop.columns:
                    _peer_pop = _peer_pop[
                        pd.to_numeric(_peer_pop['totalMinutes'], errors='coerce').fillna(0) >= 500
                    ]

            def _percentile_for(metric_name, p90_value):
                """Percentile of this player's per-90 value against same-position peers (≥500 min)."""
                if _peer_pop.empty or metric_name not in _peer_pop.columns:
                    return None
                if pd.isna(p90_value):
                    return None
                pop_vals = pd.to_numeric(_peer_pop[metric_name], errors='coerce').dropna()
                if len(pop_vals) < 5 or pop_vals.std() == 0:
                    return None
                pct = scipy.stats.percentileofscore(pop_vals, p90_value, kind='weak')
                if metric_name in INVERT_METRICS:
                    pct = 100.0 - pct
                return float(pct)

            def _percentile_color(p):
                """Red (0) → yellow (50) → green (100) HSL gradient."""
                if p is None or pd.isna(p):
                    return ''
                p_clamped = max(0.0, min(100.0, float(p)))
                hue = (p_clamped / 100.0) * 120.0   # 0=red, 60=yellow, 120=green
                return f'background-color: hsl({hue:.0f}, 65%, 72%); color: black;'

            for group_name, group_metrics in stat_groups.items():

                if player_is_gk and group_name != 'Goalkeeping':
                    continue
                if not player_is_gk and group_name == 'Goalkeeping':
                    continue

                if player_is_gk and group_name == 'Goalkeeping':
                    group_metrics = GOALKEEPING_METRICS + ['GK Passes successful %', 'GK Long passes successful %']

                metrics_to_show = [m for m in group_metrics if m in stats_to_display.index]

                if metrics_to_show:
                    default_expanded = (group_name == 'Output')
                    with st.expander(f"**{group_name} Stats**", expanded=default_expanded):

                        stats_subset_series = stats_to_display[metrics_to_show]
                        stats_subset_series = stats_subset_series[stats_subset_series != 0]

                        if stats_subset_series.empty:
                            st.text("No data for this category.")
                            continue

                        def _fmt_stat(metric_name, x):
                            if not isinstance(x, (int, float)):
                                return str(x)
                            if metric_name in THOUSANDTHS_METRICS:
                                return f"{x:.3f}"
                            if np.round(x) == x and '%' not in str(x):
                                return f"{x:.0f}"
                            return f"{x:.2f}"

                        # Build display DataFrame: Value (formatted str) + Percentile (numeric)
                        _rows = []
                        for _metric in stats_subset_series.index:
                            _disp_val = stats_subset_series[_metric]
                            _p90_val = per_90_stats.get(_metric, np.nan)
                            _pct = _percentile_for(_metric, _p90_val)
                            _rows.append({
                                'Metric': _metric,
                                'Value': _fmt_stat(_metric, _disp_val),
                                'Percentile': _pct,
                            })
                        stats_subset = pd.DataFrame(_rows).set_index('Metric')

                        _styled = (
                            stats_subset.style
                            .applymap(_percentile_color, subset=['Percentile'])
                            .format({'Percentile': lambda v: f"{int(round(v))}" if pd.notna(v) else '—'})
                        )
                        st.dataframe(_styled, use_container_width=True)
        

        elif _active_tab == "Value":
            st.divider()
        
            # --- Transfer Value Detail ----------------------------------------
            # Deep-dive on market value: Reported fees & manual entries
            # expander + the Market Context features block. The legacy CVI
            # breakdown display panels were retired (Lucas 2026-06-12) — the
            # ACP engine provides the headline Projected value in the bio
            # card. CVI computation helpers stay at module level for other
            # pages.
            st.subheader("Transfer Value Detail")
            st.caption("Projected value is computed by the ACP engine (see bio card). "
                       "Legacy CVI breakdown retired 2026-06-12.")
            try:
                with st.expander("Reported transfer fees & manual entries",
                                  expanded=False):
                    st.caption("**Market value sources**")
                    if _tv_valuations_rows.empty:
                        st.caption("No data yet. Populates from reported transfer "
                                    "fees + manual entries.")
                    else:
                        _src_view = (_tv_valuations_rows
                                      .groupby('source', as_index=False)
                                      .first()[['source', 'value_eur', 'as_of_date']])
                        _src_view['value_eur'] = _src_view['value_eur'].apply(
                            lambda v: f"€{v:,.0f}" if pd.notna(v) else "—"
                        )
                        st.dataframe(_src_view, use_container_width=True, hide_index=True, column_config=auto_column_config(_src_view))
                        if len(_tv_valuations_rows) > len(_src_view):
                            with st.expander(f"Full history ({len(_tv_valuations_rows)} entries)"):
                                _hist_view = _tv_valuations_rows[
                                    ['source', 'value_eur', 'as_of_date', 'notes']
                                ].copy()
                                _hist_view['value_eur'] = _hist_view['value_eur'].apply(
                                    lambda v: f"€{v:,.0f}" if pd.notna(v) else "—"
                                )
                                st.dataframe(_hist_view, use_container_width=True,
                                              hide_index=True,
                                              column_config=auto_column_config(_hist_view))

                    # ---- Manual valuation entry ----
                    # Add a hand-entered figure from club / agent conversations.
                    # Highest-authority source (weight 4.0 in the loader's blend).
                    with st.expander("➕ Add manual valuation", expanded=False):
                        with st.form(f"manual_val_{player_id}_{selected_season_id}",
                                      clear_on_submit=True):
                            _mv_col_a, _mv_col_b = st.columns(2)
                            _mv_eur = _mv_col_a.number_input(
                                "Value (EUR)", min_value=0, step=10_000,
                                value=0, help="Hand-entered figure from club "
                                              "or agent conversation. €0 = skip.",
                            )
                            from datetime import date as _date_cls
                            _mv_date = _mv_col_b.date_input(
                                "As-of date", value=_date_cls.today(),
                                help="When this valuation was given to you.",
                            )
                            _mv_notes = st.text_input(
                                "Notes (optional)",
                                placeholder="e.g. 'agent quote', 'club asking price', "
                                            "'rejected bid from X'",
                            )
                            _mv_submitted = st.form_submit_button("Save",
                                                                    type="primary")
                            if _mv_submitted:
                                if _mv_eur <= 0:
                                    st.warning("Value must be > €0 — skipping.")
                                else:
                                    try:
                                        import csv
                                        _man_path = (Path(__file__).resolve().parent
                                                      / 'valuations'
                                                      / 'manual_entries.csv')
                                        _man_path.parent.mkdir(exist_ok=True)
                                        _new_file = not _man_path.exists()
                                        with open(_man_path, 'a', newline='') as _f:
                                            _w = csv.writer(_f)
                                            if _new_file:
                                                _w.writerow(['playerId', 'value_eur',
                                                              'as_of_date', 'season_id',
                                                              'source_url', 'notes'])
                                            _w.writerow([
                                                int(player_id), int(_mv_eur),
                                                _mv_date.isoformat(),
                                                (int(selected_season_id)
                                                 if selected_season_id else ''),
                                                '',
                                                (f"{_mv_notes} | added via dashboard"
                                                 if _mv_notes else "added via dashboard"),
                                            ])
                                        st.success(
                                            f"Saved: €{_mv_eur:,} as of {_mv_date} "
                                            f"for {selected_player_name}. "
                                            f"Refresh the page to see it in the True value."
                                        )
                                    except Exception as _save_exc:
                                        st.error(f"Could not save: "
                                                  f"{type(_save_exc).__name__}: {_save_exc}")

                # ---- Market Context features ----
                st.markdown("##### Market Context")
                try:
                    _tv_team = (str(_tv_player_row.get('teamName'))
                                 if _tv_player_row is not None
                                 and pd.notna(_tv_player_row.get('teamName'))
                                 else None)
                    _opta_fn = (make_opta_team_strength_lookup()
                                 if 'make_opta_team_strength_lookup' in globals()
                                 else (lambda _t: None))
                    _mc = compute_market_features(
                        player_id=player_id,
                        season_id=selected_season_id,
                        raw_events_df=raw_events_df,
                        matches_summary_df=matches_summary_df,
                        player_details_df=player_details_df,
                        player_minutes_data=player_minutes_data,
                        team_name=_tv_team,
                        opta_team_lookup=_opta_fn,
                    )
                    _mc_c1, _mc_c2, _mc_c3, _mc_c4 = st.columns(4)
                    def _fmt_resid(v, n_dec=1):
                        if v is None or pd.isna(v): return "—"
                        return f"{v:+.{n_dec}f}"

                    _mc_c1.metric("xG O/U (season)",
                                    _fmt_resid(_mc['xg_residual_season']),
                                    help="Goals minus xG, non-penalty, this season. "
                                         "Positive = outperforming xG (clinical "
                                         "finishing or variance); negative = "
                                         "underperforming.")
                    _mc_c1.metric("xG O/U (career)",
                                    _fmt_resid(_mc['xg_residual_career']),
                                    help="Cumulative across all seasons in our "
                                         "data. More stable than single-season "
                                         "residuals.")
                    _mc_c2.metric("xA O/U (season)",
                                    _fmt_resid(_mc['ass_residual_season']),
                                    help="Assists minus xA proxy (sum of xG of "
                                         "shots the player set up).")
                    _mc_c2.metric("xA O/U (career)",
                                    _fmt_resid(_mc['ass_residual_career']))

                    _nat_p = _mc.get('passport_nationality') or '—'
                    _nat_b = _mc.get('birth_nationality') or '—'
                    _mc_c3.metric("Nationality (passport)", _nat_p)
                    if _nat_b != _nat_p:
                        _mc_c3.metric("Birthplace", _nat_b)

                    _team_opta = _mc.get('team_opta_rating')
                    _team_ppm = _mc.get('team_ppm_season')
                    _team_pos = _mc.get('team_league_position')
                    _mc_c4.metric(
                        "Team Opta",
                        f"{_team_opta:.1f}" if _team_opta is not None else "—",
                        help="Current team's Opta Power Ranking — proxy for "
                             "scouting visibility and tier-internal team strength.",
                    )
                    _mc_c4.metric(
                        "Team this season",
                        (f"{_team_ppm:.2f} PPM" if _team_ppm is not None else "—")
                        + (f" · {_team_pos}." if _team_pos is not None else ""),
                        help="Points per match + league position from parsed scores. "
                             "Successful-team players typically carry a market premium.",
                    )

                    _ver = _mc.get('positions_played_career')
                    _sea = _mc.get('seasons_played')
                    if _ver is not None or _sea is not None:
                        _bits = []
                        if _ver is not None:
                            _bits.append(f"{_ver} position{'s' if _ver != 1 else ''} played")
                        if _sea is not None:
                            _bits.append(f"{_sea} season{'s' if _sea != 1 else ''} in data")
                        st.caption("· ".join(_bits))
                    st.caption(
                        "📌 These features feed the v2 EUR regression "
                        "(currently pending). They don't change CVI itself."
                    )
                except Exception as _mc_exc:
                    st.caption(f"Market Context error: "
                                f"{type(_mc_exc).__name__}: {_mc_exc}")
            except Exception as _tv_exc:
                st.caption(f"Transfer Value Detail error: "
                            f"{type(_tv_exc).__name__}: {_tv_exc}")


        elif _active_tab == "Shots & Creation":
            st.divider()

            # --- 7. SHOT ANALYSIS (UPDATED) ---
            st.subheader("Shot Analysis")
        
            # 1. Get all player events
            player_events_all = profile_events_df[profile_events_df['player.name'] == selected_player_name].copy()
        
            # 2. Filter for shots (non-penalty) for the map/analysis
            shot_log = player_events_all[
                (player_events_all['type.primary'] == 'shot') &
                (player_events_all['type.primary'] != 'penalty')
            ].copy()
        
            if not shot_log.empty:
                # --- DATA PROCESSING START ---
            
                # Sort chronologically for numbering (oldest first)
                if 'dateutc' in shot_log.columns:
                    shot_log = shot_log.sort_values(by=['dateutc', 'minute', 'second'], ascending=True)
                else:
                    shot_log = shot_log.sort_values(by=['matchId', 'minute', 'second'], ascending=True)
                
                # Assign Shot Numbers (1 to N)
                shot_log.reset_index(drop=True, inplace=True)
                shot_log['Shot Number'] = shot_log.index + 1
            
                # Basic formatting
                shot_log['Date'] = pd.to_datetime(shot_log['dateutc']).dt.strftime('%Y-%m-%d') if 'dateutc' in shot_log.columns else "N/A"
                shot_log['Opponent'] = shot_log.get('opponentTeam.name', 'Unknown')
                shot_log['xG'] = pd.to_numeric(shot_log['shot.xg'], errors='coerce').fillna(0)
                shot_log['Result'] = np.where(shot_log['shot.isGoal'] == True, 'Goal', 
                                     np.where(shot_log['shot.onTarget'] == True, 'Saved', 'Off Target'))

                # Body Part Extraction
                if 'shot.bodyPart.name' in shot_log.columns:
                    shot_log['Body Part'] = shot_log['shot.bodyPart.name']
                elif 'shot.bodyPart' in shot_log.columns:
                    shot_log['Body Part'] = shot_log['shot.bodyPart'].apply(
                        lambda x: x.get('name', 'Unknown') if isinstance(x, dict) else str(x)
                    )
                    shot_log['Body Part'] = shot_log['Body Part'].str.replace('_', ' ').str.title()
                else:
                    shot_log['Body Part'] = 'Unknown'

                # Phase of Play
                def get_phase(possession_types):
                    if not isinstance(possession_types, (list, np.ndarray)): return "Open Play"
                    if 'counter_attack' in possession_types: return "Counter Attack"
                    if 'corner' in possession_types or 'free_kick' in possession_types or 'penalty' in possession_types: return "Set Piece"
                    if 'positional_attack' in possession_types: return "Positional Attack"
                    return "Open Play"
            
                if 'possession.types' in shot_log.columns:
                    shot_log['Phase'] = shot_log['possession.types'].apply(get_phase)
                else:
                    shot_log['Phase'] = "Unknown"

                # Shot Creating Action (SCA)
                relevant_match_ids = shot_log['matchId'].unique()
                context_events = profile_events_df[
                    (profile_events_df['matchId'].isin(relevant_match_ids)) &
                    (profile_events_df['team.name'] == shot_log.iloc[0]['team.name'])
                ].copy()
            
                shot_log['prev_event_idx'] = shot_log['possession.eventIndex'] - 1
            
                sca_merge = pd.merge(
                    shot_log[['id', 'matchId', 'possession.id', 'prev_event_idx']],
                    context_events[['matchId', 'possession.id', 'possession.eventIndex', 'type.primary', 'type.secondary']],
                    left_on=['matchId', 'possession.id', 'prev_event_idx'],
                    right_on=['matchId', 'possession.id', 'possession.eventIndex'],
                    how='left',
                    suffixes=('', '_prev')
                )
            
                def label_sca(row):
                    if pd.isna(row['type.primary']): return "Recovery/None"
                    sec_types = row['type.secondary'] if isinstance(row['type.secondary'], (list, np.ndarray)) else []
                    if 'cross' in sec_types: return "Cross"
                    if 'through_pass' in sec_types: return "Through Pass"
                    if 'deep_completion' in sec_types: return "Deep Completion"
                    prim = row['type.primary']
                    if prim == 'pass': return "Pass"
                    if prim == 'duel': return "Dribble/Duel"
                    if prim == 'acceleration' or prim == 'touch': return "Carry"
                    if prim == 'clearance': return "Clearance"
                    if prim == 'interception': return "Interception"
                    return prim.replace('_', ' ').title()

                sca_merge['SCA'] = sca_merge.apply(label_sca, axis=1)
                shot_log = shot_log.merge(sca_merge[['id', 'SCA']], on='id', how='left')
            
                # --- DATA PROCESSING END ---

                # --- VISUALIZATION: one combined visual — full-width
                # StatsBomb-style map (shape = creating action, color =
                # xG, ring = goal), shot log directly beneath. Static mpl
                # version kept for the PDF one-pager.
                st.plotly_chart(
                    plotly_shot_map(shot_log, selected_player_name,
                                    height=660),
                    use_container_width=True,
                    config={'displayModeBar': False})

                st.markdown("**Shot Log**")
                display_cols = ['Shot Number', 'Date', 'Opponent', 'Result', 'xG', 'Body Part', 'SCA']
                table_display = shot_log[display_cols].rename(columns={
                    'Shot Number': '#',
                    'SCA': 'Creating Action'
                }).sort_values(by='#', ascending=False) # Show newest first (highest number)

                st.dataframe(table_display, use_container_width=True, height=380, hide_index=True, column_config=auto_column_config(table_display))

                # --- NEW: SUMMARY TABLES ---
                st.markdown("---")
                col_sum1, col_sum2 = st.columns(2)
            
                with col_sum1:
                    st.markdown("**Stats by Body Part**")
                    body_summary = shot_log.groupby('Body Part').agg(
                        Shots=('id', 'count'),
                        Goals=('shot.isGoal', 'sum'),
                        Total_xG=('xG', 'sum')
                    ).sort_values(by='Total_xG', ascending=False)
                    body_summary['xG/Shot'] = (body_summary['Total_xG'] / body_summary['Shots']).round(2)
                    body_summary['Total_xG'] = body_summary['Total_xG'].round(2)
                    st.dataframe(body_summary, use_container_width=True, column_config=auto_column_config(body_summary))
                
                with col_sum2:
                    st.markdown("**Stats by Creating Action**")
                    sca_summary = shot_log.groupby('SCA').agg(
                        Shots=('id', 'count'),
                        Goals=('shot.isGoal', 'sum'),
                        Total_xG=('xG', 'sum')
                    ).sort_values(by='Total_xG', ascending=False)
                    sca_summary['xG/Shot'] = (sca_summary['Total_xG'] / sca_summary['Shots']).round(2)
                    sca_summary['Total_xG'] = sca_summary['Total_xG'].round(2)
                    st.dataframe(sca_summary, use_container_width=True, column_config=auto_column_config(sca_summary))

            else:
                st.info("No shots recorded for this player.")

            st.divider()

            # --- 7a-bis. CREATION — passes into the attacking box ---
            st.subheader("Creation — Passes into the Box")
            _bp_all = load_box_passes()
            if _bp_all.empty:
                st.caption("Box-pass data not available in this deployment.")
            else:
                _bp_seasons = _season_id_list(active_season_ids)
                _bp_p = _bp_all[
                    _bp_all['player.id'] == int(selected_player_id)].copy()
                if _bp_seasons:
                    _bp_p = _bp_p[_bp_p['seasonId'].isin(_bp_seasons)]
                if _bp_p.empty:
                    st.info("No passes into the box recorded for this player "
                            "in the selected season(s).")
                else:
                    _cc1, _cc2, _cc3 = st.columns([1, 1, 2])
                    with _cc1:
                        _bp_no_sp = st.checkbox(
                            "Open play only", value=False,
                            help="Exclude corner / free-kick / throw-in deliveries")
                    with _cc2:
                        _bp_acc_only = st.checkbox("Completed only", value=False)
                    _bp_view = _bp_p
                    if _bp_no_sp:
                        _bp_view = _bp_view[_bp_view['phase'] != 'set_piece']
                    if _bp_acc_only:
                        _bp_view = _bp_view[_bp_view['pass.accurate'] == True]  # noqa: E712
                    if _bp_view.empty:
                        st.info("No box passes match the current filters.")
                    else:
                        st.plotly_chart(
                            plotly_box_passes_map(_bp_view, selected_player_name),
                            use_container_width=True,
                            config={'displayModeBar': False})
                        _n_bp = len(_bp_view)
                        _mets = st.columns(4)
                        _mets[0].metric("Box passes", f"{_n_bp}")
                        _mets[1].metric(
                            "Completed",
                            f"{(_bp_view['pass.accurate'] == True).mean():.0%}")  # noqa: E712
                        _mets[2].metric(
                            "Total pass value",
                            f"{pd.to_numeric(_bp_view['action_value'], errors='coerce').sum():+.3f}",
                            help="Sum of GPA action values of these passes")
                        _mets[3].metric(
                            "Value / pass",
                            f"{pd.to_numeric(_bp_view['action_value'], errors='coerce').mean():+.4f}")
                        st.caption(
                            "Arrow color = GPA pass value (red = value created, "
                            "blue = negative value); arrow weight scales with "
                            "|value|. Hover an endpoint for pass details.")

            st.divider()

            # --- 7b. Shot Assists & Dribbles in Final Third ---
            st.subheader("Shot Assists & Dribbles in Final Third")
            try:
                fig_sa_player = pv.plot_shot_assists_and_dribbles(
                    profile_events_df, current_team,
                    player_name=selected_player_name,
                )
                st.pyplot(fig_sa_player, use_container_width=True)
                plt.close(fig_sa_player)
            except Exception as e:
                st.caption(f"Could not render shot assists & dribbles: {e}")

            st.divider()

            # --- 7c. Defensive Action Heatmap ---
            st.subheader("Defensive Action Heatmap")
            try:
                # Resolve positional peer group for defensive heatmap
                _DEFENSIVE_PEER_GROUPS = {
                    'GK': ['GK'],
                    'CB': ['CB', 'LCB', 'RCB', 'LCB3', 'RCB3'],
                    'FB': ['LB', 'RB', 'LB5', 'RB5', 'LWB', 'RWB'],
                    'CM': ['DMF', 'LDMF', 'RDMF', 'LCMF', 'RCMF', 'LCMF3', 'RCMF3'],
                    'AM/Wing': ['AMF', 'LAMF', 'RAMF', 'LW', 'RW', 'LWF', 'RWF'],
                    'ST': ['CF', 'SS'],
                }
                _heatmap_pos_codes = [current_pos]
                _heatmap_peer_label = current_pos
                for _grp_name, _grp_codes in _DEFENSIVE_PEER_GROUPS.items():
                    if current_pos in _grp_codes:
                        _heatmap_pos_codes = _grp_codes
                        _heatmap_peer_label = _grp_name
                        break

                # Compute peer density stack (cached)
                _events_hash = hashlib.md5(
                    f"{len(profile_events_df)}_{tuple(sorted(_heatmap_pos_codes))}".encode()
                ).hexdigest()

                _peer_stack = _compute_peer_density_stack(
                    _events_hash, profile_events_df,
                    tuple(sorted(_heatmap_pos_codes)),
                    _player_minutes_df=profile_player_minutes_df,
                    include_recoveries=True,
                )

                fig_def_heatmap = pv.plot_defensive_action_heatmap(
                    profile_events_df, player_id, selected_player_name,
                    position_codes=_heatmap_pos_codes,
                    player_minutes_df=profile_player_minutes_df,
                    peer_density_stack=_peer_stack,
                    include_recoveries=True,
                )
                st.pyplot(fig_def_heatmap, use_container_width=True)
                plt.close(fig_def_heatmap)
                st.caption(f"Colour intensity normalised across **{_heatmap_peer_label}** peers.")
            except Exception as e:
                st.caption(f"Could not render defensive action heatmap: {e}")

            st.divider()

            # --- 7a. Throw-In Analysis ---
            st.subheader("Throw-In Analysis")

            try:
                player_throwin_df = profile_events_df[
                    (profile_events_df['player.id'] == player_id) &
                    (profile_events_df['type.primary'] == 'throw_in')
                ].copy()

                if not player_throwin_df.empty and 'pass.length' in player_throwin_df.columns:
                    total_throwins = len(player_throwin_df)

                    # Avg of top 10 longest throw-ins (overall distance)
                    top_10_all = player_throwin_df.nlargest(min(10, total_throwins), 'pass.length')
                    avg_top10_length = top_10_all['pass.length'].mean()

                    # Throw-ins into the attacking penalty box (end x >= 84, 20 <= end y <= 80)
                    into_box = player_throwin_df[
                        (player_throwin_df['pass.endLocation.x'] >= 84) &
                        (player_throwin_df['pass.endLocation.y'] >= 20) &
                        (player_throwin_df['pass.endLocation.y'] <= 80)
                    ]
                    if not into_box.empty:
                        top_10_box = into_box.nlargest(min(10, len(into_box)), 'pass.length')
                        avg_top10_into_box = top_10_box['pass.length'].mean()
                    else:
                        avg_top10_into_box = 0.0

                    # Throw-ins into box where next action is an aerial duel
                    avg_top10_into_box_aerial = 0.0
                    if not into_box.empty:
                        sorted_match_events = profile_events_df.sort_values(by=['matchId', 'minute', 'second']).reset_index(drop=True)
                        aerial_box_throws = []
                        for _, ti_row in into_box.iterrows():
                            m_id = ti_row.get('matchId')
                            if m_id is None:
                                continue
                            m_events = sorted_match_events[sorted_match_events['matchId'] == m_id]
                            pos_mask = (m_events['minute'] == ti_row['minute']) & (m_events['second'] == ti_row['second']) & (m_events['type.primary'] == 'throw_in')
                            positions = m_events[pos_mask].index
                            if len(positions) == 0:
                                continue
                            next_pos = positions[0] + 1
                            if next_pos in m_events.index:
                                next_sec = m_events.loc[next_pos].get('type.secondary', '')
                                if isinstance(next_sec, (list, set)):
                                    is_aerial = 'aerial_duel' in next_sec
                                else:
                                    is_aerial = 'aerial_duel' in str(next_sec)
                                if is_aerial:
                                    aerial_box_throws.append(ti_row)
                        if aerial_box_throws:
                            aerial_df = pd.DataFrame(aerial_box_throws)
                            top_10_aerial = aerial_df.nlargest(min(10, len(aerial_df)), 'pass.length')
                            avg_top10_into_box_aerial = top_10_aerial['pass.length'].mean()

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Throw-Ins", int(total_throwins))
                    with col2:
                        st.metric("Avg Max Distance", f"{avg_top10_length:.1f}m")
                    with col3:
                        st.metric("Avg Max Into Box", f"{avg_top10_into_box:.1f}m")
                    with col4:
                        st.metric("Avg Max Into Box → Aerial", f"{avg_top10_into_box_aerial:.1f}m")
                else:
                    st.info(f"{selected_player_name} has no throw-ins in the selected period.")

            except Exception as e:
                st.caption(f"Could not render throw-in analysis: {e}")


        elif _active_tab == "Match Log":
            st.divider()

            # --- 8. Display Individual Match Stats (Unchanged) ---
            st.subheader("Individual Match Log")
        
            if player_match_log_df.empty:
                st.info("No individual match stats found for this player.")
            else:
                key_match_stats = ['Date', 'Match', 'Score', 'Minutes', 'Goals / xG', 'xAOP', 'xASP', 'xTOP', 'xTSP', 'Actions / successful', 'Passes / accurate', 'Duels / won']
                cols_to_show = [c for c in key_match_stats if c in player_match_log_df.columns]
                st.dataframe(player_match_log_df[cols_to_show].set_index('Date'))
                with st.expander("View Full Match Log (All Stats)"):
                    st.dataframe(player_match_log_df.set_index('Date'))

# --- NEW: Player Comparison Section ---
    elif analysis_type == 'Player Comparison':

        # --- League & Season Selector ---
        selected_comp_ids = league_selector("player_comparison")
        selected_season_id = season_selector("player_comparison", include_all_seasons=True, comp_ids=selected_comp_ids)
        active_season_ids = get_season_ids_for_selection(selected_season_id, selected_comp_ids)
        comp_events_df = get_filtered_events(raw_events_df, active_season_ids, selected_comp_ids)
        comp_player_minutes_df = get_season_player_minutes(player_minutes_data, active_season_ids, comp_ids=selected_comp_ids)

        # --- 1. Load Data ---
        try:
            with st.spinner("Loading player statistics..."):
                player_stats_df, player_stats_with_scores_df = load_and_score_player_stats(
                    comp_events_df, comp_player_minutes_df, selected_season_id, active_season_ids, selected_comp_ids
                )
        except Exception as e:
            st.error(f"An error occurred calculating player stats: {e}")
            logger.exception("Error in Player Comparison stats calculation")
            st.stop()
            
        if player_stats_with_scores_df.empty:
            st.warning("No players found with sufficient minutes for comparison.")
            st.stop()

        # --- 2. Player Selectors (NEW LOGIC) ---
        st.sidebar.subheader("Comparison Options")
        
        # --- Step A: Select Player A (from all players) ---
        # FIX: Include 'playerId' in the columns so we can use it for lookup
        player_list_df = player_stats_with_scores_df[['playerId', 'playerName', 'teamName', 'totalMinutes']].sort_values(by='totalMinutes', ascending=False)
        player_list_df['display_name'] = player_list_df['playerName'].astype(str) + " (" + player_list_df['teamName'].astype(str) + ", " + pd.to_numeric(player_list_df['totalMinutes'], errors='coerce').fillna(0).astype(int).astype(str) + " min)"
        
        selected_player_a_display = st.sidebar.selectbox(
            "Select Player A:", 
            player_list_df['display_name'], 
            index=0 # Default to first player
        )
        
        # FIX: Lookup by ID instead of Name
        selected_player_a_id = player_list_df[player_list_df['display_name'] == selected_player_a_display]['playerId'].values[0]
        player_a_data = player_stats_with_scores_df[player_stats_with_scores_df['playerId'] == selected_player_a_id]
        
        # Get the name safely from the ID-filtered data
        selected_player_a_name = player_a_data.iloc[0]['playerName']

        # --- Step B: Select Template ---
        all_templates = sorted(list(POSITION_GROUPS.keys()))
        
        # Find Player A's best-fit template as default
        primary_pos_a = player_a_data.iloc[0]['primaryPosition']
        eligible_groups_a = [pos_group for pos_group, pos_roles in POSITION_GROUPS.items() if primary_pos_a in pos_roles]
        highest_score = -1; default_template = all_templates[0]
        for group in eligible_groups_a:
            score_col = group + '_Score'
            if score_col in player_a_data.columns:
                player_score = player_a_data[score_col].values[0]
                if player_score > highest_score:
                    highest_score = player_score; default_template = group
        
        default_index = all_templates.index(default_template) if default_template in all_templates else 0
        
        selected_template = st.sidebar.selectbox(
            "Select Comparison Template:",
            all_templates,
            index=default_index
        )
        
        # --- Step C: Filter Player B list based on Template ---
        positions_in_group = POSITION_GROUPS.get(selected_template, [])
        
        filtered_player_df = player_stats_with_scores_df[
            player_stats_with_scores_df['primaryPosition'].isin(positions_in_group)
        ]
        
        # Create the display list for Player B from the filtered df
        # FIX: Include 'playerId' here too
        player_b_list_df = filtered_player_df[['playerId', 'playerName', 'teamName', 'totalMinutes']].sort_values(by='totalMinutes', ascending=False)
        player_b_list_df['display_name'] = player_b_list_df['playerName'].astype(str) + " (" + player_b_list_df['teamName'].astype(str) + ", " + pd.to_numeric(player_b_list_df['totalMinutes'], errors='coerce').fillna(0).astype(int).astype(str) + " min)"
        
        # Find a smart default index for Player B (e.g., the second player in the list)
        default_b_index = 0
        if len(player_b_list_df) > 1:
            default_b_index = 1
        
        # --- Step D: Select Player B (from filtered list) ---
        selected_player_b_display = st.sidebar.selectbox(
            "Select Player B (Same Position Group):", 
            player_b_list_df['display_name'],
            index=default_b_index 
        )
        
        # FIX: Lookup by ID instead of Name
        selected_player_b_id = player_b_list_df[player_b_list_df['display_name'] == selected_player_b_display]['playerId'].values[0]
        player_b_data = player_stats_with_scores_df[player_stats_with_scores_df['playerId'] == selected_player_b_id]
        
        # Get the name safely
        selected_player_b_name = player_b_data.iloc[0]['playerName']


        # --- 4. Plot Radar ---
        st.subheader(f"Comparing: {selected_player_a_name} vs. {selected_player_b_name}")

        _mins_a = pd.to_numeric(player_a_data.iloc[0].get('totalMinutes', 0), errors='coerce') or 0
        _mins_b = pd.to_numeric(player_b_data.iloc[0].get('totalMinutes', 0), errors='coerce') or 0
        _below_threshold = []
        if _mins_a < 300:
            _below_threshold.append(f"{selected_player_a_name} ({int(_mins_a)} min)")
        if _mins_b < 300:
            _below_threshold.append(f"{selected_player_b_name} ({int(_mins_b)} min)")

        if _below_threshold:
            st.info(f"⚠️ **Insufficient sample size** — {' and '.join(_below_threshold)}.")

        metrics_to_plot = list(WEIGHTS[selected_template].keys())
        metrics_to_plot = [m for m in metrics_to_plot
                           if m in player_stats_with_scores_df.columns
                           and m not in RADAR_HIDDEN_METRICS]

        # --- FIX: Use a square figure to prevent distortion ---
        fig = plt.figure(figsize=(15, 15))
        # [left, bottom, width, height] - This centers the radar
        ax_radar = fig.add_axes([0.15, 0.15, 0.7, 0.7], polar=True)

        plot_comparison_radar(
            ax_radar,
            player_a_data,
            player_b_data,
            metrics_to_plot,
            selected_template
        )

        st.pyplot(fig, use_container_width=True)

    # --- NEW: Player Analysis Section ---
    elif analysis_type == 'Player Analysis':

        # --- League & Season Selector ---
        selected_comp_ids = league_selector("player_analysis")
        selected_season_id = season_selector("player_analysis", include_all_seasons=True, comp_ids=selected_comp_ids)
        active_season_ids = get_season_ids_for_selection(selected_season_id, selected_comp_ids)
        analysis_events_df = get_filtered_events(raw_events_df, active_season_ids, selected_comp_ids)
        analysis_player_minutes_df = get_season_player_minutes(player_minutes_data, active_season_ids, comp_ids=selected_comp_ids)

        # --- 1. Load Data ---
        try:
            with st.spinner("Loading player statistics..."):
                player_stats_df, player_stats_with_scores_df = load_and_score_player_stats(
                    analysis_events_df, analysis_player_minutes_df, selected_season_id, active_season_ids, selected_comp_ids
                )
        except Exception as e:
            st.error(f"An error occurred calculating player stats: {e}")
            logger.exception("Error in Player Analysis stats calculation")
            st.stop()

        if player_stats_with_scores_df.empty:
            st.warning("No players found with sufficient minutes for analysis.")
            st.stop()

        # --- 2. Sidebar Controls ---
        st.sidebar.subheader("Analysis Options")

        # Template / mode selector — Overview is the default landing page
        _TEMPLATE_GROUPS = {
            'Goalkeepers': ['Shot Stopper', 'Cross Claimer', 'Ball-playing GK'],
            'Center Backs': ['Ball-Playing Centerback', 'Stopper', 'Athletic Centerback'],
            'Full Backs': ['Full Back', 'Wingback', 'Inverted Full Back'],
            'Central Midfielders': ['Box-to-Box', 'Holding Mid', 'Ball-Winning Mid', 'Deep-lying Playmaker'],
            'Attacking Mids / Wingers': ['Advanced Playmaker', 'Wide Winger', 'Creative Winger', 'Inside Forward'],
            'Forwards': ['Shadow Striker', 'Mobile Striker', 'Poacher', 'Target Man', 'Pressing Forward'],
        }
        # Build ordered template list from groups (preserving group order)
        _ordered_templates = []
        for _grp_templates in _TEMPLATE_GROUPS.values():
            _ordered_templates.extend([t for t in _grp_templates if t in POSITION_GROUPS])
        _selector_options = ["Overview"] + _ordered_templates + ["Individual Metric", "Peer Scatter"]
        _selected_view = st.sidebar.selectbox(
            "View:",
            _selector_options,
            index=0,
            key="player_analysis_view"
        )

        # Minimum minutes filter. Floor lowered 500->90 (Lucas 2026-06): the
        # ACP engine now rates players down to 90 min (scored against the
        # >=500 cohort, then minutes-shrunk toward replacement), so low-minute
        # players are penalised by the shrink rather than excluded. Default 90
        # surfaces them; the shrinkage keeps them off the top of the boards.
        max_minutes = int(player_stats_with_scores_df['totalMinutes'].max())
        min_minutes_filter = st.sidebar.slider(
            "Minimum Minutes Played:",
            min_value=90,
            max_value=max(max_minutes, 500),
            value=90,
            step=45,
            key="player_analysis_min_minutes",
            help="ACP ratings exist down to 90 min (heavily shrunk toward "
                 "replacement). Bespoke template scores still need 500+.",
        )

        # Number of players to display
        num_players = st.sidebar.slider(
            "Number of Players to Display:",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            key="player_analysis_num_players"
        )

        # --- Bulk Radar Export ---
        st.sidebar.markdown("---")
        # Per-radar PNG files in a render directory — each completed
        # radar is independently committed to disk, so a kill mid-render
        # leaves a directory of valid PNGs the user can still download
        # as a ZIP. The ZIP is built lazily at download time, never
        # streamed/written during render → no central-directory
        # truncation problem.
        import os as _os, io as _io, time as _time, pickle as _pickle
        import hashlib as _hashlib, json as _json, zipfile as _zipfile
        _BULK_CACHE_DIR = "/tmp/dashboard_bulk_export"
        _BULK_CACHE_ERROR = None
        try:
            _os.makedirs(_BULK_CACHE_DIR, exist_ok=True)
            _probe = _os.path.join(_BULK_CACHE_DIR, '.writable_probe')
            with open(_probe, 'w') as _pf:
                _pf.write('ok')
            _os.unlink(_probe)
        except Exception as _cache_dir_exc:
            _BULK_CACHE_ERROR = f"{type(_cache_dir_exc).__name__}: {_cache_dir_exc}"

        def _bulk_cache_key(season_lbl, groups, mode, min_mins):
            payload = _json.dumps({
                "season": str(season_lbl),
                "groups": sorted([str(g) for g in groups]),
                "mode": str(mode),
                "min_mins": int(min_mins),
            }, sort_keys=True)
            return _hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]

        def _bulk_render_dir(key):
            return _os.path.join(_BULK_CACHE_DIR, f"radars__{key}")

        def _bulk_meta_path(key):
            return _os.path.join(_bulk_render_dir(key), 'meta.pkl')

        def _list_cached_renders():
            """Return one entry per render directory under the cache
            dir, regardless of meta state. Each entry surfaces the PNG
            count, total bytes on disk, and last-modified time so
            partial/crashed runs are still visible and downloadable."""
            entries = []
            try:
                names = _os.listdir(_BULK_CACHE_DIR)
            except (FileNotFoundError, Exception):
                return entries
            for name in names:
                full = _os.path.join(_BULK_CACHE_DIR, name)
                if not _os.path.isdir(full):
                    continue
                try:
                    pngs = [f for f in _os.listdir(full) if f.endswith('.png')]
                except Exception:
                    continue
                if not pngs:
                    continue
                meta_path = _os.path.join(full, 'meta.pkl')
                meta = None
                if _os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'rb') as _f:
                            meta = _pickle.load(_f)
                    except Exception:
                        meta = None
                if meta is None:
                    meta = {'status': 'incomplete', 'label': name}
                try:
                    total_bytes = sum(_os.path.getsize(_os.path.join(full, f))
                                       for f in pngs)
                    mtime = max((_os.path.getmtime(_os.path.join(full, f))
                                  for f in pngs), default=0)
                except Exception:
                    total_bytes, mtime = 0, 0
                entries.append({
                    'path': full,
                    'meta': meta,
                    'mtime': mtime,
                    'size': total_bytes,
                    'png_count': len(pngs),
                })
            entries.sort(key=lambda e: e['mtime'], reverse=True)
            return entries

        def _build_zip_from_dir(render_dir):
            """Build a ZIP byte-string on the fly from every PNG in the
            render directory. Memory cost is proportional to ZIP size at
            click time only — not held during rendering."""
            buf = _io.BytesIO()
            with _zipfile.ZipFile(buf, 'w', _zipfile.ZIP_DEFLATED,
                                   compresslevel=1) as _zf:
                for fn in sorted(_os.listdir(render_dir)):
                    if not fn.endswith('.png'):
                        continue
                    _fp = _os.path.join(render_dir, fn)
                    with open(_fp, 'rb') as _ff:
                        _zf.writestr(fn, _ff.read())
            return buf.getvalue()

        # Apply minutes filter
        filtered_df = player_stats_with_scores_df[
            player_stats_with_scores_df['totalMinutes'] >= min_minutes_filter
        ].copy()

        if filtered_df.empty:
            st.warning(f"No players found with {min_minutes_filter}+ minutes. Try lowering the threshold.")
            st.stop()

        # --- Age filter ---
        analysis_player_details_df = load_player_details()
        if not analysis_player_details_df.empty and 'birthDate' in analysis_player_details_df.columns:
            _ages_series = analysis_player_details_df['birthDate'].apply(_calculate_age)
            _numeric_ages = pd.to_numeric(_ages_series, errors='coerce').dropna()
            if not _numeric_ages.empty:
                min_age_available = int(_numeric_ages.min())
                max_age_available = int(_numeric_ages.max()) + 1
                age_range = st.sidebar.slider(
                    "Age Range:",
                    min_value=min_age_available,
                    max_value=max_age_available,
                    value=(min_age_available, max_age_available),
                    key="player_analysis_age_range"
                )
                if age_range != (min_age_available, max_age_available):
                    valid_ids = _numeric_ages[
                        (_numeric_ages >= age_range[0]) & (_numeric_ages <= age_range[1])
                    ].index.tolist()
                    filtered_df = filtered_df[filtered_df['playerId'].isin(valid_ids)]
                    if filtered_df.empty:
                        st.warning(f"No players found in age range {age_range[0]}-{age_range[1]}.")
                        st.stop()

        # --- Show Only Position toggle (HIDDEN per Lucas 2026-06) ---
        # Toggle removed from the sidebar; pos-played filtering stays off.
        analysis_pos_played_filter = False
        analysis_pos_played_active = False
        analysis_selected_positions = []
        if analysis_pos_played_filter:
            all_pos_minutes = get_all_players_minutes_by_position(analysis_events_df)
            if not all_pos_minutes.empty:
                available_positions = sorted(all_pos_minutes['Position'].unique().tolist())
                analysis_selected_positions = st.sidebar.multiselect(
                    "Position(s):",
                    available_positions,
                    default=available_positions[:1],
                    key="analysis_pos_played_position"
                )
                if analysis_selected_positions:
                    pos_min_for_positions = all_pos_minutes[all_pos_minutes['Position'].isin(analysis_selected_positions)].groupby('playerId')['Minutes'].sum().reset_index().rename(columns={'Minutes': 'posMinutes'})
                    filtered_df = filtered_df.merge(pos_min_for_positions, on='playerId', how='inner')
                    analysis_pos_played_active = not filtered_df.empty
                    if filtered_df.empty:
                        pos_label = "/".join(analysis_selected_positions)
                        st.warning(f"No players found who played at {pos_label} with current filters.")
                        st.stop()

        # --- Global rating-adjustment toggles ----------------------------
        # These adjustments operate on the Rating column (Role_Score) used
        # in Overview + per-template views. They are intentionally NOT
        # offered on Individual Metric (per user — they only make sense
        # for overall profile ratings, not per-metric leaderboards).
        st.sidebar.markdown("---")
        # Same-age-peers + cross-tier toggles REMOVED (Lucas 2026-06-12):
        # the engine's age curve handles age context in the projection,
        # and the abs columns handle cross-league translation. Flags stay
        # pinned off; their dormant downstream code paths were deleted.
        age_adjusted = False

        # Cross-tier translation — Opta strength multiplier only. The
        # empirical-median variant is per-metric, so it doesn't apply
        # to a composite Role_Score.
        cross_tier_mode = 'Off'
        _trans_src_comp = _trans_tgt_comp = None
        _rating_multiplier = None
        _rating_caption = None

        # --- CVI (Composite Value Index) toggle -----------------------
        # When ON, a CVI column is appended to the right of the Rating
        # column in Overview + per-template tables, and (optionally) the
        # ranking re-sorts by CVI. Position-tuned age curve
        # (see CVI_AGE_VALUE_PARAMS) calibrated off the 27 reported transfers.
        show_cvi = st.sidebar.checkbox(
            "Show Projected value",
            value=False,
            key="player_analysis_show_cvi",
            help="ACP engine projection → EUR: percentile of the "
                 "next-season projection (abs scale, recruit-discounted "
                 "for Camp) × career-NPV age multiplier (ST 1.30, AM/WG "
                 "1.25, CM 1.00, CB 0.90, FB 0.85), through the "
                 "fee-calibrated CVI→EUR curve, capped at €500k.",
        )
        sort_by_cvi = False
        if show_cvi:
            sort_by_cvi = st.sidebar.checkbox(
                "Sort by Projected value",
                value=False,
                key="player_analysis_sort_by_cvi",
                help="Replace the Rating-based sort with a Projected-value sort.",
            )

        # Pre-compute age column for the full filtered pool — used by
        # the same-age peer computation inside _build_template_table
        # AND by CVI's age-value lookup.
        _has_age = (not analysis_player_details_df.empty
                     and 'birthDate' in analysis_player_details_df.columns)
        if show_cvi and _has_age:
            _filtered_age = filtered_df['playerId'].map(
                lambda pid: _calculate_age(analysis_player_details_df.loc[pid, 'birthDate'])
                if pid in analysis_player_details_df.index else None
            )
            filtered_df = filtered_df.assign(_age=_filtered_age.values)

        # Pre-compute CVI columns once for the full filtered_df. The
        # helper does its own position-grouped percentile internally;
        # we slice the result per-template inside _build_template_table.
        if show_cvi:
            try:
                _age_map = (filtered_df.set_index('playerId')['_age'].to_dict()
                             if '_age' in filtered_df.columns else {})
                # Map player → competitionId. selected_comp_ids is the
                # league filter active in the sidebar; when one league
                # is selected, every visible player belongs to it.
                # Otherwise fall back to competition_for_season.
                if selected_comp_ids and len(selected_comp_ids) == 1:
                    _the_comp = int(selected_comp_ids[0])
                    _comp_lookup = lambda _pid: _the_comp
                elif 'seasonId' in filtered_df.columns:
                    _season_to_comp = {
                        sid: competition_for_season(sid)
                        for sid in filtered_df['seasonId'].dropna().unique()
                    }
                    _ssid_map = filtered_df.set_index('playerId')['seasonId'].to_dict()
                    _comp_lookup = lambda pid: _season_to_comp.get(_ssid_map.get(pid))
                else:
                    _comp_lookup = lambda _pid: None
                # Build empirical-Bayes prior lookup so each player's
                # season perf is shrunk toward THEIR OWN career prior
                # (not generic 40). Single perf_table build covers
                # every player in filtered_df → cheap bulk lookup.
                _bulk_prior_lookup = None
                try:
                    _bulk_pt = _build_player_season_perf_table(
                        load_gpa_values(), None,
                    )
                    if (_bulk_pt is not None and not _bulk_pt.empty
                            and selected_season_id is not None):
                        _bulk_prior_map = build_player_priors_lookup(
                            _bulk_pt, selected_season_id,
                        )
                        _bulk_prior_lookup = (
                            lambda pid: _bulk_prior_map.get(int(pid))
                                          if pid is not None else None
                        )
                except Exception as _prior_exc:
                    print(f"[CVI prior] bulk lookup build failed: "
                           f"{type(_prior_exc).__name__}: {_prior_exc}")
                _cvi_block = compute_cvi_columns(
                    filtered_df,
                    age_lookup=lambda pid: _age_map.get(pid),
                    comp_id_lookup=_comp_lookup,
                    prior_lookup=_bulk_prior_lookup,
                )
                if not _cvi_block.empty:
                    filtered_df = pd.concat(
                        [filtered_df.reset_index(drop=True),
                         _cvi_block.reset_index(drop=True)],
                        axis=1,
                    )
            except Exception as _cvi_exc:
                import traceback as _tb
                _tb_str = _tb.format_exc()
                # Dtype diagnostic for the input frame — the most likely
                # source of comparison errors is a mixed-type column.
                _dtype_lines = []
                try:
                    for _c in ('primaryPosition', 'totalMinutes',
                                'Total Value', 'playerId', 'seasonId',
                                'competitionId', '_age'):
                        if _c in filtered_df.columns:
                            _samp = filtered_df[_c].dropna().head(3).tolist()
                            _dtype_lines.append(
                                f"  {_c:<18} dtype={filtered_df[_c].dtype} "
                                f"sample={_samp}")
                    _score_cols = [c for c in filtered_df.columns
                                    if c.endswith('_Score')][:5]
                    for _c in _score_cols:
                        _samp = filtered_df[_c].dropna().head(3).tolist()
                        _dtype_lines.append(
                            f"  {_c:<18} dtype={filtered_df[_c].dtype} "
                            f"sample={_samp}")
                except Exception:
                    pass
                _diag = "\n".join(_dtype_lines)
                st.sidebar.warning(f"CVI compute failed: "
                                    f"{type(_cvi_exc).__name__}: {_cvi_exc}")
                # Full traceback + dtype diagnostic to BOTH sidebar
                # (always visible) and server-side stdout. Don't gate
                # on an expander — Streamlit's expander_state can hide
                # the message on some deploys.
                _full_diag = (f"{_tb_str}\n"
                                f"---- input column diagnostics ----\n"
                                f"{_diag}")
                print(f"[CVI ERROR] {type(_cvi_exc).__name__}: {_cvi_exc}\n"
                       f"{_full_diag}")
                st.sidebar.code(_full_diag, language='python')
                show_cvi = False
                sort_by_cvi = False

        # --- Helper: build a display table for a given template ---
        def _build_template_table(template_name, source_df, n_players, compact=False):
            """Build a display DataFrame for a template. Returns (display_df, player_ids) or (None, [])."""
            positions_in_group = POSITION_GROUPS.get(template_name, [])
            score_col = f"{template_name}_Score"
            if score_col not in source_df.columns:
                return None, []

            if analysis_pos_played_active:
                tdf = source_df
            else:
                tdf = source_df[source_df['primaryPosition'].isin(positions_in_group)]

            if tdf.empty:
                return None, []

            _sort_col = score_col

            # Overview defaults to ACP PROJECTION order (Lucas) —
            # bespoke template score still drives the per-template views
            # and remains sortable via Individual Metric mode. GK
            # templates (no engine coverage) keep the score sort.
            if (compact and 'ACP Projection (abs)' in tdf.columns
                    and tdf['ACP Projection (abs)'].notna().any()):
                _sort_col = 'ACP Projection (abs)'

            # Projected-value sort override.
            if show_cvi and sort_by_cvi and 'Engine Value EUR' in tdf.columns:
                _sort_col = 'Engine Value EUR'

            sorted_tdf = tdf.sort_values(by=_sort_col, ascending=False, na_position='last').head(n_players)

            if compact:
                # Overview mode (Lucas): ACP Projection first, then ACP
                # Rating, REPLACING the bespoke template score. GK
                # templates fall back to the bespoke score.
                _eng_over = [c for c in ('ACP Projection (abs)', 'ACP Rating')
                             if c in sorted_tdf.columns
                             and sorted_tdf[c].notna().any()]
                if _eng_over:
                    cols = ['playerName', 'teamName', 'primaryPosition',
                            'totalMinutes'] + _eng_over
                else:
                    cols = ['playerName', 'teamName', 'primaryPosition',
                            'totalMinutes', score_col]
            else:
                # Template-specific mode: include all weighted metrics (weight > 0) sorted by weight desc
                template_weights = WEIGHTS.get(template_name, {})
                weighted_metrics = sorted(
                    [(m, w) for m, w in template_weights.items() if w > 0],
                    key=lambda x: x[1], reverse=True
                )
                metric_cols = [m for m, _ in weighted_metrics if m in sorted_tdf.columns]
                _eng_cols = [c for c in ('ACP Rating', 'ACP Projection (abs)')
                             if c in sorted_tdf.columns]
                cols = (['playerName', 'teamName', 'primaryPosition', 'totalMinutes',
                          score_col] + _eng_cols + metric_cols)

            cols = [c for c in cols if c in sorted_tdf.columns]
            display = sorted_tdf[cols].copy()
            _ren = {
                'playerName': 'Player',
                'teamName': 'Team',
                'primaryPosition': 'Position',
                'totalMinutes': 'Minutes',
                score_col: 'Rating'
            }
            if compact and 'ACP Rating' in cols:
                # engine columns take the headline names in Overview
                _ren.update({'ACP Projection (abs)': 'Projection',
                              'ACP Rating': 'Rating'})
            display = display.rename(columns=_ren)
            if 'Rating' in display.columns:
                display['Rating'] = pd.to_numeric(display['Rating'], errors='coerce').round(1)
            if 'Projection' in display.columns:
                display['Projection'] = pd.to_numeric(display['Projection'], errors='coerce').round(1)
            display['Minutes'] = display['Minutes'].astype(int)

            # Projected value: insert next to Rating in both compact
            # and full modes. EUR computed from CVI × position mult ×
            # Camp penalty, capped at €500k. In full mode also surface
            # the Trajectory flag (perf - same-age-position median).
            # Projected Value column — ENGINE value (legacy CVI→EUR
            # removed per Lucas 2026-06-12). Computed centrally in
            # load_player_engine(); merged in as 'Engine Value EUR'.
            if show_cvi and 'Engine Value EUR' in sorted_tdf.columns:
                pv_vals = pd.Series(sorted_tdf['Engine Value EUR'].values,
                                     index=display.index)
                pv_display = [
                    (f"€{int(v):,}" if v is not None and pd.notna(v) else '')
                    for v in pv_vals
                ]
                _r_idx = display.columns.get_loc('Rating')
                display.insert(_r_idx + 1, 'Projected Value', pv_display)

            # Add Pos. Minutes
            if analysis_pos_played_active and 'posMinutes' in sorted_tdf.columns:
                display.insert(display.columns.get_loc('Minutes') + 1, 'Pos. Minutes', sorted_tdf['posMinutes'].astype(int).values)

            # Add Age
            if not analysis_player_details_df.empty and 'birthDate' in analysis_player_details_df.columns:
                age_pos = display.columns.get_loc('Pos. Minutes') + 1 if 'Pos. Minutes' in display.columns else display.columns.get_loc('Minutes') + 1
                display.insert(age_pos, 'Age', sorted_tdf['playerId'].map(
                    lambda pid: _calculate_age(analysis_player_details_df.loc[pid, 'birthDate']) if pid in analysis_player_details_df.index else None
                ).apply(lambda x: round(x, 1) if isinstance(x, float) else None))

            # Round metric columns
            for col in display.columns:
                if pd.api.types.is_numeric_dtype(display[col]) and col not in ['Minutes', 'Rating', 'Pos. Minutes', 'Age', 'Rank']:
                    decimals = 3 if col in THOUSANDTHS_METRICS else (0 if col in WHOLE_NUMBER_METRICS else 2)
                    display[col] = display[col].round(decimals)

            display.insert(0, 'Rank', range(1, len(display) + 1))
            return display, sorted_tdf['playerId'].tolist()

        # --- Helper: handle row selection from a dataframe ---
        def _handle_row_selection(selection, player_ids):
            if selection and selection.selection and selection.selection.rows:
                selected_row_idx = selection.selection.rows[0]
                if selected_row_idx < len(player_ids):
                    st.session_state.selected_player_id = player_ids[selected_row_idx]
                    st.session_state.nav_to_profile = True
                    st.session_state.nav_season_id = selected_season_id
                    st.session_state.nav_has_season = True
                    st.rerun()

        # --- 3. Display based on selected view ---
        if _selected_view == "Overview":
            # --- Engine-role board (shown ABOVE the template Overview) ---
            # Best players per OBSERVED engine role (the 6 data-derived
            # clusters), ranked by ACP Projection, with Minutes + Age.
            #
            # CONSISTENCY: the engine ROLE is the ONLY thing taken from
            # player_engine. Age, Min and Proj are read from the SAME scoped
            # stats frame (filtered_df) + player_details that the template table
            # below uses — so every shared column is identical across the two
            # tables. In particular: Age = age TODAY (_calculate_age), Min =
            # totalMinutes (TRUE lineup minutes). We deliberately do NOT use the
            # engine's `age` (as-of-season, ~5.5 mo stale) or `mins_played`
            # (event-derived undercount) for display.
            _eng_role_df, _ = load_player_engine()
            if (_eng_role_df is not None and not _eng_role_df.empty
                    and filtered_df is not None and not filtered_df.empty):
                # role per player = their MOST COMMON ACP engine role across ALL
                # seasons and BOTH leagues (Lucas 2026-06-24): sum minutes-in-each-
                # role (mins_lineup × per-season role share sh_<role>) over the
                # player's ENTIRE engine history and take the argmax. Uses the full
                # _eng_role_df — NOT an in-scope slice — so a player is bucketed by
                # the role they've actually played most, regardless of the selected
                # league (and immune to the non-chronological seasonId ordering).
                _sh_cols = [c for c in _eng_role_df.columns if c.startswith('sh_')]
                _gm = _eng_role_df[['playerId'] + _sh_cols].copy()
                _w_role = pd.to_numeric(_eng_role_df.get('mins_lineup'), errors='coerce')
                _w_role = _w_role.fillna(
                    pd.to_numeric(_eng_role_df.get('mins_played'), errors='coerce')).fillna(0.0)
                for _c in _sh_cols:
                    _gm[_c] = pd.to_numeric(_gm[_c], errors='coerce').fillna(0.0) * _w_role.values
                _role_tot = _gm.groupby('playerId')[_sh_cols].sum()
                _role_tot = _role_tot[_role_tot.sum(axis=1) > 0]
                _role_map = (_role_tot.idxmax(axis=1).str[3:]
                                       .rename('role').reset_index())
                # join role onto the scoped, minutes-filtered stats pool — the
                # board now inherits filtered_df's totalMinutes + ACP columns
                _rb = filtered_df.merge(_role_map, on='playerId', how='inner')
                # canonical age TODAY (same lookup the template table uses)
                if (not analysis_player_details_df.empty
                        and 'birthDate' in analysis_player_details_df.columns):
                    _rb_age = _rb['playerId'].map(
                        lambda pid: _calculate_age(analysis_player_details_df.loc[pid, 'birthDate'])
                        if pid in analysis_player_details_df.index else None)
                    _rb = _rb.assign(_age_disp=_rb_age.values)
                else:
                    _rb = _rb.assign(_age_disp=None)
                # Projection is the headline. Rank by it and show only players
                # who HAVE one (current + recent-lapsed), so inactive historical
                # players — whose face-value rating would otherwise outrank the
                # mean-shrunk projections — don't pollute the board. A purely
                # past-season scope carries no projections, so there we fall
                # back to rating and relabel the column honestly.
                _proj_num = pd.to_numeric(_rb.get('ACP Projection (abs)'), errors='coerce')
                if _proj_num.notna().any():
                    _rb = _rb[_proj_num.notna()].copy()
                    _metric_label, _metric_col = 'Proj', 'ACP Projection (abs)'
                else:
                    _metric_label, _metric_col = 'Rating', 'ACP Rating (abs)'
                _rb['_rankval'] = pd.to_numeric(_rb[_metric_col], errors='coerce')

                _ENGINE_ROLE_ORDER = ['Striker', 'Wide Attacker', 'Advanced Midfielder',
                                       'Deep Midfielder', 'Wide Defender', 'Central Defender']
                _role_sub = ['Player', _metric_label, 'Min', 'Age']
                _role_cols = {}
                for _role in _ENGINE_ROLE_ORDER:
                    _rsub = _rb[_rb['role'] == _role].sort_values('_rankval', ascending=False).head(num_players)
                    if _rsub.empty:
                        continue
                    _rows = []
                    for _, _r in _rsub.iterrows():
                        _mval = pd.to_numeric(_r.get(_metric_col), errors='coerce')
                        _age = _r.get('_age_disp')
                        _age = float(_age) if isinstance(_age, (int, float)) and pd.notna(_age) else None
                        _mins = pd.to_numeric(_r.get('totalMinutes'), errors='coerce')
                        _rows.append((
                            _r.get('playerName', ''),
                            (round(float(_mval), 1) if pd.notna(_mval) else ''),
                            (int(_mins) if pd.notna(_mins) else 0),
                            (round(_age, 1) if _age is not None else ''),
                        ))
                    _role_cols[_role] = _rows

                if _role_cols:
                    _max_r = max(len(v) for v in _role_cols.values())
                    _ctups, _cdata = [], {}
                    _present_roles = [r for r in _ENGINE_ROLE_ORDER if r in _role_cols]
                    for _role in _present_roles:
                        for _s in _role_sub:
                            _ctups.append((_role, _s)); _cdata[(_role, _s)] = []
                    for _i in range(_max_r):
                        for _role in _present_roles:
                            _rws = _role_cols[_role]
                            if _i < len(_rws):
                                for _s, _v in zip(_role_sub, _rws[_i]):
                                    _cdata[(_role, _s)].append(_v)
                            else:
                                for _s in _role_sub:
                                    _cdata[(_role, _s)].append('')
                    _erole_df = pd.DataFrame(_cdata, columns=pd.MultiIndex.from_tuples(_ctups))
                    _erole_df.index = range(1, len(_erole_df) + 1)
                    _erole_df.index.name = 'Rank'
                    st.subheader("Best Players by Role")
                    with st.expander("ℹ️ How these ratings work"):
                        st.markdown(RATINGS_EXPLAINER_MD)
                    if _metric_label == 'Proj':
                        st.caption(
                            "Observed ACP engine roles (data-derived from playing patterns), "
                            "ranked by ACP Projection. **Proj** = projected level next season "
                            "(absolute / cross-league scale). **Age** (current) and **Min** "
                            "(total minutes in scope) match the table below. Only players with "
                            "a live projection appear."
                        )
                    else:
                        st.caption(
                            "Observed ACP engine roles (data-derived from playing patterns), "
                            "ranked by ACP **Rating** (absolute scale) — the selected season is "
                            "historical, so no forward projection exists. **Age** is current; "
                            "**Min** is total minutes in scope."
                        )
                    st.dataframe(_erole_df, use_container_width=True)
                    st.markdown("---")

            # Build wide pivot table: each template is a column group with Player, Team, Minutes, Rating
            _OVERVIEW_ORDER = [
                'Forwards', 'Attacking Mids / Wingers', 'Central Midfielders',
                'Full Backs', 'Center Backs', 'Goalkeepers',
            ]
            # Collect per-template data as lists aligned by rank.
            # When Projected Value is on, append it as a 5th sub-column
            # per template (already EUR-formatted by _build_template_table).
            _sub_cols = ['Player', 'Team', 'Min', 'Proj', 'Rating']
            if show_cvi:
                _sub_cols.append('Proj. Value')
            template_columns = {}
            for group_name in _OVERVIEW_ORDER:
                group_templates = _TEMPLATE_GROUPS.get(group_name, [])
                for tmpl in [t for t in group_templates if t in POSITION_GROUPS]:
                    display_df, _ = _build_template_table(tmpl, filtered_df, num_players, compact=True)
                    if display_df is not None and not display_df.empty:
                        rows = []
                        for _, row in display_df.iterrows():
                            _proj_v = row.get('Projection')
                            _rat_v = row.get('Rating')
                            tup = [
                                row.get('Player', ''),
                                row.get('Team', ''),
                                int(row.get('Minutes', 0)),
                                (round(float(_proj_v), 1)
                                 if _proj_v is not None and pd.notna(_proj_v) else ''),
                                (round(float(_rat_v), 1)
                                 if _rat_v is not None and pd.notna(_rat_v) else ''),
                            ]
                            if show_cvi:
                                _pv = row.get('Projected Value', '')
                                tup.append(_pv if _pv else '')
                            rows.append(tuple(tup))
                        template_columns[tmpl] = rows

            if template_columns:
                max_rows = max(len(v) for v in template_columns.values())
                # Build MultiIndex columns DataFrame
                col_tuples = []
                data_dict = {}
                for tmpl in template_columns:
                    for sub in _sub_cols:
                        col_tuples.append((tmpl, sub))
                        data_dict[(tmpl, sub)] = []

                for rank_idx in range(max_rows):
                    for tmpl in template_columns:
                        rows = template_columns[tmpl]
                        if rank_idx < len(rows):
                            tup = rows[rank_idx]
                            for sub, val in zip(_sub_cols, tup):
                                data_dict[(tmpl, sub)].append(val)
                        else:
                            for sub in _sub_cols:
                                data_dict[(tmpl, sub)].append('')

                multi_idx = pd.MultiIndex.from_tuples(col_tuples)
                overview_df = pd.DataFrame(data_dict, columns=multi_idx)
                overview_df.index = range(1, len(overview_df) + 1)
                overview_df.index.name = 'Rank'

                st.subheader("Player Overview — Template Roles")
                st.caption(
                    "Bespoke scouting templates (Shadow Striker, Mobile Striker, "
                    "Creative Winger, …) — same players, scored against each role's "
                    "weighting. Proj/Rating columns are the ACP engine values."
                )
                if show_cvi:
                    st.caption(
                        "🟩 Projected Value = CVI → EUR mapping "
                        "(2.5 × CVI^2.5 × position multiplier, capped at €500k). "
                        "Calibrated against the 27 reported transfer fees."
                        + (" Sort is by Projected Value." if sort_by_cvi else "")
                    )
                st.dataframe(overview_df, use_container_width=True)
            else:
                st.warning("No players match current filters.")

        elif _selected_view == "Peer Scatter":
            # --- Peer Scatter: any metric vs any metric, full peer cloud ---
            _SC_SET_PIECE = ['Set Piece Value', 'Corner Value',
                             'Free Kick Value', 'Throw-In Value',
                             'xASP', 'xTSP']
            _sc_categories = {
                "Output": OUTPUT_METRICS,
                "Passing": PASSING_METRICS,
                "Defensive": DEFENSIVE_METRICS,
                "Defensive Responsibility (DefR)": DEFR_DISPLAY_METRICS,
                "Dribbling": DRIBBLING_METRICS,
                "Goalkeeping": GOALKEEPING_METRICS,
                "Set Pieces": _SC_SET_PIECE,
                "ACP Index": ENGINE_DISPLAY_METRICS,
                "Template Ratings": sorted(
                    [c for c in filtered_df.columns if c.endswith('_Score')]),
            }

            def _sc_metric_picker(axis_label, default_cat, default_metric):
                cat = st.sidebar.selectbox(
                    f"{axis_label} category:", list(_sc_categories.keys()),
                    index=list(_sc_categories.keys()).index(default_cat),
                    key=f"peer_scatter_cat_{axis_label}")
                opts = [m for m in _sc_categories[cat]
                        if m in filtered_df.columns]
                if not opts:
                    return None
                idx = opts.index(default_metric) if default_metric in opts else 0
                return st.sidebar.selectbox(
                    f"{axis_label} metric:", opts, index=idx,
                    key=f"peer_scatter_metric_{axis_label}")

            _sc_x = _sc_metric_picker("X", "Defensive", "Interceptions")
            _sc_y = _sc_metric_picker("Y", "Output", "npxG")

            _sc_positions = sorted(
                filtered_df['primaryPosition'].dropna().unique().tolist())
            _sc_pos_filter = st.sidebar.multiselect(
                "Filter by Position (optional):", _sc_positions, default=[],
                key="peer_scatter_positions")

            if _sc_x is None or _sc_y is None:
                st.warning("No metrics available for the selected categories.")
                st.stop()
            if _sc_x == _sc_y:
                st.info("Pick two different metrics to compare.")
                st.stop()

            _sc_df = filtered_df
            if _sc_pos_filter:
                _sc_df = _sc_df[_sc_df['primaryPosition'].isin(_sc_pos_filter)]
            _sc_df = _sc_df.dropna(subset=[_sc_x, _sc_y])
            if len(_sc_df) < 5:
                st.warning("Not enough players with both metrics under the "
                           "current filters.")
                st.stop()

            st.subheader(f"{_sc_y} vs {_sc_x} (per 90)")
            _sc_c1, _sc_c2 = st.columns([2, 1])
            with _sc_c1:
                _sc_hl = st.selectbox(
                    "Highlight player:",
                    ["(none)"] + sorted(_sc_df['playerName'].dropna().unique().tolist()),
                    key="peer_scatter_highlight")
            with _sc_c2:
                _sc_fit = st.checkbox("Show line of best fit", value=True,
                                      key="peer_scatter_fit")

            _sx = pd.to_numeric(_sc_df[_sc_x], errors='coerce')
            _sy = pd.to_numeric(_sc_df[_sc_y], errors='coerce')
            _sc_fig = go.Figure()
            _sc_hover = [
                f"<b>{r.playerName}</b> · {r.teamName}<br>"
                f"{r.primaryPosition} · {int(r.totalMinutes)}'<br>"
                f"{_sc_x}: {x:.3f}<br>{_sc_y}: {y:.3f}"
                for r, x, y in zip(_sc_df.itertuples(), _sx, _sy)]
            _sc_fig.add_trace(go.Scatter(
                x=_sx, y=_sy, mode='markers',
                marker=dict(size=8, color='rgba(110,125,118,0.45)',
                            line=dict(color='rgba(255,255,255,0.6)', width=0.5)),
                text=_sc_hover, hovertemplate='%{text}<extra></extra>',
                name='peers'))
            _sc_fig.add_vline(x=float(_sx.median()), line_dash='dot',
                              line_color='rgba(128,128,128,0.5)')
            _sc_fig.add_hline(y=float(_sy.median()), line_dash='dot',
                              line_color='rgba(128,128,128,0.5)')
            if _sc_fit and len(_sc_df) >= 3:
                _b, _a = np.polyfit(_sx, _sy, 1)
                _xline = np.linspace(_sx.min(), _sx.max(), 50)
                _r = float(np.corrcoef(_sx, _sy)[0, 1])
                _sc_fig.add_trace(go.Scatter(
                    x=_xline, y=_b * _xline + _a, mode='lines',
                    line=dict(color='#3987e5', width=2, dash='dash'),
                    hoverinfo='skip', name=f'fit (r = {_r:+.2f})'))
            if _sc_hl != "(none)":
                _hrow = _sc_df[_sc_df['playerName'] == _sc_hl]
                _sc_fig.add_trace(go.Scatter(
                    x=pd.to_numeric(_hrow[_sc_x], errors='coerce'),
                    y=pd.to_numeric(_hrow[_sc_y], errors='coerce'),
                    mode='markers+text',
                    marker=dict(size=14, color='#2aa876',
                                line=dict(color='white', width=2)),
                    text=[_sc_hl] * len(_hrow), textposition='top center',
                    textfont=dict(size=12),
                    hoverinfo='skip', name=_sc_hl))
            _sc_fig.update_layout(
                height=640, showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.01),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title=_sc_x, gridcolor='rgba(128,128,128,0.15)',
                           zeroline=False),
                yaxis=dict(title=_sc_y, gridcolor='rgba(128,128,128,0.15)',
                           zeroline=False))
            st.plotly_chart(_sc_fig, use_container_width=True)
            _sc_notes = []
            if _sc_x in INVERT_METRICS or _sc_y in INVERT_METRICS:
                _inv = [m for m in (_sc_x, _sc_y) if m in INVERT_METRICS]
                _sc_notes.append(f"lower is better for: {', '.join(_inv)}")
            _sc_notes.append("dotted lines = peer medians")
            _sc_notes.append(f"{len(_sc_df)} players shown "
                             "(sidebar minutes/age filters apply)")
            st.caption(" · ".join(_sc_notes))

        elif _selected_view == "Individual Metric":
            # --- Individual Metric mode (preserved from original) ---
            # Set-piece metrics — the four GPA action-value columns
            # ("Set Piece Value" = sum of the other three) plus the
            # set-piece-only flavors of xA and xT. xASP/xTSP also live
            # in OUTPUT_METRICS but are duplicated here so users can
            # find every set-piece metric in one place.
            SET_PIECE_METRICS = [
                'Set Piece Value', 'Corner Value', 'Free Kick Value',
                'Throw-In Value', 'xASP', 'xTSP',
            ]
            metric_categories = {
                "Output": OUTPUT_METRICS,
                "Passing": PASSING_METRICS,
                "Defensive": DEFENSIVE_METRICS,
                "Defensive Responsibility (DefR)": DEFR_DISPLAY_METRICS,
                "Dribbling": DRIBBLING_METRICS,
                "Goalkeeping": GOALKEEPING_METRICS,
                "Set Pieces": SET_PIECE_METRICS,
                "ACP Index": ENGINE_DISPLAY_METRICS,
                # bespoke template ratings stay sortable here (Lucas) —
                # they left the Overview headline but remain a metric
                "Template Ratings": sorted(
                    [c for c in filtered_df.columns if c.endswith('_Score')]),
            }

            selected_category = st.sidebar.selectbox(
                "Metric Category:",
                list(metric_categories.keys()),
                key="player_analysis_metric_category"
            )

            available_metrics = [m for m in metric_categories[selected_category]
                               if m in filtered_df.columns]
            if not available_metrics:
                st.warning(f"No metrics available for {selected_category} category.")
                st.stop()

            selected_metric = st.sidebar.selectbox(
                "Select Metric:",
                available_metrics,
                key="player_analysis_metric_v2"
            )

            all_positions = sorted(filtered_df['primaryPosition'].dropna().unique().tolist())
            position_filter = st.sidebar.multiselect(
                "Filter by Position (optional):",
                all_positions,
                default=[],
                key="player_analysis_position_filter"
            )

            if position_filter:
                metric_filtered_df = filtered_df[filtered_df['primaryPosition'].isin(position_filter)]
            else:
                metric_filtered_df = filtered_df

            if metric_filtered_df.empty:
                st.warning("No players found with current filters.")
                st.stop()

            # Individual Metric mode intentionally does NOT apply the
            # age-adjusted / cross-tier-translation toggles — those are
            # for overall profile ratings only (Overview + per-template).
            _sort_ascending = selected_metric in INVERT_METRICS
            sorted_df = metric_filtered_df.sort_values(by=selected_metric, ascending=_sort_ascending).head(num_players)

            st.subheader(f"Top Players by {selected_metric} (per 90)")

            related_metrics = []
            if selected_metric in OUTPUT_METRICS:
                related_metrics = ['Goals', 'xG', 'npxG', 'Shots', 'Assists', 'xAOP']
            elif selected_metric in PASSING_METRICS:
                related_metrics = ['Passes', 'Passes successful %', 'Progressive Passes', 'xTOP']
            elif selected_metric in DEFENSIVE_METRICS:
                related_metrics = ['Interceptions', 'Recoveries', 'Defensive duels', 'Aerial duels']
            elif selected_metric in DRIBBLING_METRICS:
                related_metrics = ['Dribbles', 'Dribbles successful %', 'Progressive runs']
            elif selected_metric in GOALKEEPING_METRICS:
                related_metrics = ['goalsPrevented', 'savePercentage', 'exits']
            elif selected_metric in SET_PIECE_METRICS:
                # Show the other set-piece value flavors + the set-
                # piece xA/xT pair so you can see, e.g., which corner
                # specialists are also creating high-xT deliveries.
                related_metrics = ['Set Piece Value', 'Corner Value',
                                    'Free Kick Value', 'Throw-In Value',
                                    'xASP', 'xTSP']

            related_metrics = [m for m in related_metrics if m in sorted_df.columns and m != selected_metric][:4]

            display_cols = ['playerName', 'teamName', 'primaryPosition', 'totalMinutes', selected_metric] + related_metrics
            display_cols = [c for c in display_cols if c in sorted_df.columns]

            display_df = sorted_df[display_cols].copy()
            display_df = display_df.rename(columns={
                'playerName': 'Player',
                'teamName': 'Team',
                'primaryPosition': 'Position',
                'totalMinutes': 'Minutes'
            })
            display_df['Minutes'] = display_df['Minutes'].astype(int)

            if analysis_pos_played_active and 'posMinutes' in sorted_df.columns:
                display_df.insert(display_df.columns.get_loc('Minutes') + 1, 'Pos. Minutes', sorted_df['posMinutes'].astype(int).values)

            if not analysis_player_details_df.empty and 'birthDate' in analysis_player_details_df.columns:
                age_col_pos = display_df.columns.get_loc('Pos. Minutes') + 1 if 'Pos. Minutes' in display_df.columns else display_df.columns.get_loc('Minutes') + 1
                display_df.insert(age_col_pos, 'Age', sorted_df['playerId'].map(
                    lambda pid: _calculate_age(analysis_player_details_df.loc[pid, 'birthDate']) if pid in analysis_player_details_df.index else None
                ).apply(lambda x: round(x, 1) if isinstance(x, float) else None))

            # Individual Metric no longer applies the cross-tier / age
            # toggles — those are now scoped to overall ratings only.
            for col in display_df.columns:
                if pd.api.types.is_numeric_dtype(display_df[col]) and col not in ['Minutes', 'Pos. Minutes', 'Age']:
                    decimals = 3 if col in THOUSANDTHS_METRICS else 2
                    display_df[col] = display_df[col].round(decimals)

            display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
            player_ids = sorted_df['playerId'].tolist()

            st.caption("Click on a row to view that player's profile")
            selection = st.dataframe(
                display_df.set_index('Rank'),
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row",
                key="individual_metric_table"
            )
            _handle_row_selection(selection, player_ids)

        else:
            # --- Template-specific view ---
            selected_template = _selected_view
            st.subheader(f"Top {selected_template}s by Rating")

            display_df, player_ids = _build_template_table(selected_template, filtered_df, num_players, compact=False)

            if display_df is not None and not display_df.empty:
                if show_cvi:
                    st.caption(
                        "🟩 CVI = composite scout-facing value · 'Traj vs age' = "
                        "performance vs same-position-same-age median "
                        "(e.g. '+25' = 25pt ahead of age peer median). "
                        "Currently uses placeholder parameters; will be calibrated "
                        "against scraped market values."
                        + (" Sort is by CVI." if sort_by_cvi else "")
                    )
                st.caption("Click on a row to view that player's profile")
                selection = st.dataframe(
                    display_df.set_index('Rank'),
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="template_detail_table"
                )
                _handle_row_selection(selection, player_ids)

                # Display weight reference table below
                template_weights = WEIGHTS.get(selected_template, {})
                weighted_items = sorted(
                    [(m, w) for m, w in template_weights.items() if w > 0],
                    key=lambda x: x[1], reverse=True
                )
                if weighted_items:
                    with st.expander("Template Weights", expanded=False):
                        weight_df = pd.DataFrame(weighted_items, columns=['Metric', 'Weight'])
                        weight_df['Weight'] = weight_df['Weight'].apply(lambda w: f"{w:.1f}")
                        st.dataframe(weight_df, use_container_width=True, hide_index=True, column_config=auto_column_config(weight_df))
            else:
                st.warning(f"No players found for {selected_template} template with current filters.")

        # ===== Distribution violins (appended below either Overview or
        # template table). Skips Individual Metric because that view is
        # already a per-metric leaderboard. =====
        if _selected_view not in ("Individual Metric", "Peer Scatter"):
            st.markdown("---")

            from plotly.subplots import make_subplots as _make_subplots

            def _pos_to_group_full(pos):
                """primaryPosition → top-level position-group label
                (matches _TEMPLATE_GROUPS keys)."""
                if pos is None or pd.isna(pos):
                    return None
                p = str(pos)
                if p == 'GK': return 'Goalkeepers'
                if p in ('CB','LCB','RCB','LCB3','RCB3'): return 'Center Backs'
                if p in ('LB','RB','LB5','RB5','LWB','RWB'): return 'Full Backs'
                if p in ('CMF','LCMF','RCMF','LCMF3','RCMF3',
                         'DMF','LDMF','RDMF'): return 'Central Midfielders'
                if p in ('AMF','LAMF','RAMF','LMF','RMF',
                         'LW','RW','LWF','RWF'): return 'Attacking Mids / Wingers'
                if p in ('CF','SS'): return 'Forwards'
                return None

            def _best_fit_in_group(row, group_roles):
                """Best Role_Score among the templates in this group."""
                vals = []
                for r in group_roles:
                    col = f"{r}_Score"
                    if col in row.index:
                        v = row.get(col)
                        if v is not None and not pd.isna(v):
                            vals.append(float(v))
                return max(vals) if vals else None

            def _add_strip(fig, row_idx, col_idx, values, names, teams,
                            metric_label, y_lo, y_hi, scaled_w, sid):
                """Append a violin + jittered dots to the (row, col)
                subplot. Hover on a dot shows player + team + value."""
                if len(values) < 3:
                    return
                # 1) Violin shape
                fig.add_trace(go.Violin(
                    x=np.zeros(len(values)),
                    y=values,
                    points=False,
                    box_visible=False,
                    meanline_visible=False,
                    side='both',
                    width=scaled_w,
                    line_color='rgba(80,80,80,0.55)',
                    fillcolor='rgba(140,140,140,0.18)',
                    showlegend=False,
                    hoverinfo='skip',
                    name='',
                ), row=row_idx, col=col_idx)
                # 2) Colored jittered dots with rich hover text
                _rng = np.random.default_rng(seed=int(sid) & 0xFFFFFFFF)
                _half = max(0.06, scaled_w / 2 - 0.04)
                _jit = _rng.uniform(-_half, _half, size=len(values))
                _custom = np.array(list(zip(names, teams)), dtype=object)
                fig.add_trace(go.Scatter(
                    x=_jit, y=values,
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=values,
                        colorscale='RdYlGn',
                        cmin=y_lo, cmax=y_hi,
                        opacity=0.7,
                        line=dict(width=0),
                        showscale=False,
                    ),
                    customdata=_custom,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Team: %{customdata[1]}<br>"
                        f"{metric_label}: " + "%{y:.3f}<extra></extra>"
                    ),
                    showlegend=False,
                    name='',
                ), row=row_idx, col=col_idx)
                fig.update_xaxes(
                    type='linear', showticklabels=False, zeroline=False,
                    range=[-0.5, 0.5], row=row_idx, col=col_idx,
                )

            def _build_panel_for_group(group_name, source_df, metric_key):
                """Return (values, names, teams) for the panel at this
                position group. metric_key ∈ {'action_v', 'best_fit',
                'acp_rating', 'acp_proj'}."""
                grp_pop = source_df[
                    source_df['primaryPosition'].map(_pos_to_group_full)
                    == group_name
                ]
                if grp_pop.empty:
                    return np.array([]), [], []
                if metric_key in ('action_v', 'acp_rating', 'acp_proj'):
                    if metric_key == 'action_v':
                        col = next((c for c in ('Total Value', 'total_v_per_90')
                                      if c in grp_pop.columns), None)
                    elif metric_key == 'acp_rating':
                        col = 'ACP Rating' if 'ACP Rating' in grp_pop.columns else None
                    else:
                        col = ('ACP Projection (abs)'
                               if 'ACP Projection (abs)' in grp_pop.columns else None)
                    if col is None:
                        return np.array([]), [], []
                    sub = grp_pop[grp_pop[col].notna()]
                    return (sub[col].astype(float).values,
                             sub.get('playerName', sub.index).astype(str).tolist(),
                             sub.get('teamName', pd.Series([''] * len(sub))).fillna('').astype(str).tolist())
                # best_fit
                group_roles = _TEMPLATE_GROUPS.get(group_name, [])
                if not group_roles:
                    return np.array([]), [], []
                _scores = grp_pop.apply(_best_fit_in_group, axis=1,
                                          args=(group_roles,))
                mask = _scores.notna()
                vals = _scores[mask].astype(float).values
                names = grp_pop.loc[mask].get('playerName',
                            grp_pop.loc[mask].index).astype(str).tolist()
                teams = grp_pop.loc[mask].get(
                    'teamName', pd.Series([''] * mask.sum())
                ).fillna('').astype(str).tolist()
                return vals, names, teams

            # --- View-dependent rendering ---
            if _selected_view == "Overview":
                st.subheader("Distribution by Position Group")
                _viz_metric = st.radio(
                    "Distribution metric:",
                    ["ACP Projection", "ACP Rating", "Action V/90", "Best-fit Rating"],
                    horizontal=True,
                    key="player_analysis_viz_metric_overview",
                )
                _metric_key = {'Action V/90': 'action_v',
                                'Best-fit Rating': 'best_fit',
                                'ACP Rating': 'acp_rating',
                                'ACP Projection': 'acp_proj'}[_viz_metric]

                _groups = ['Goalkeepers', 'Center Backs', 'Full Backs',
                            'Central Midfielders',
                            'Attacking Mids / Wingers', 'Forwards']
                _panels = []
                for g in _groups:
                    vals, names, teams = _build_panel_for_group(
                        g, filtered_df, _metric_key
                    )
                    if len(vals) >= 5:
                        _panels.append({
                            'label': f"{g}<br><span style='font-size:0.85em;color:#777'>n={len(vals)}</span>",
                            'group': g,
                            'values': vals,
                            'names': names,
                            'teams': teams,
                        })

                if not _panels:
                    st.caption(
                        f"No {_viz_metric} data available for any position "
                        f"group in the current selection."
                    )
                else:
                    _pop_concat = np.concatenate([p['values'] for p in _panels])
                    _y_lo = float(np.nanmin(_pop_concat))
                    _y_hi = float(np.nanmax(_pop_concat))
                    _y_pad = 0.05 * (_y_hi - _y_lo or 1.0)
                    _max_n = max(len(p['values']) for p in _panels) or 1
                    _fig = _make_subplots(
                        rows=1, cols=len(_panels),
                        shared_yaxes=True,
                        subplot_titles=[p['label'] for p in _panels],
                        horizontal_spacing=0.01,
                    )
                    for _i, _p in enumerate(_panels, start=1):
                        _scaled_w = 0.85 * (len(_p['values']) / _max_n) ** 0.5
                        # Use a stable per-group seed for jitter (so layout
                        # doesn't dance on rerun).
                        _seed = abs(hash(_p['group'])) & 0xFFFFFFFF
                        _add_strip(
                            _fig, 1, _i,
                            _p['values'], _p['names'], _p['teams'],
                            _viz_metric, _y_lo, _y_hi, _scaled_w, _seed,
                        )
                    _fig.update_yaxes(range=[_y_lo - _y_pad, _y_hi + _y_pad])
                    _fig.update_layout(
                        title=(f"{_viz_metric} distribution by position "
                                f"group · ≥{min_minutes_filter:.0f} min"),
                        height=460,
                        margin=dict(t=70, b=30, l=40, r=20),
                        showlegend=False,
                    )
                    for _ann in _fig['layout']['annotations']:
                        _ann['font'] = dict(size=11)
                    st.plotly_chart(_fig, use_container_width=True)

            else:
                # Template view — one panel per season for that template's
                # position group. We compute role scores for each season
                # (cached), filter to the template's eligible positions,
                # and use THAT template's Role_Score as the y-value.
                _template = _selected_view
                _eligible_positions = POSITION_GROUPS.get(_template, [])
                if not _eligible_positions:
                    st.caption(f"No position list defined for template '{_template}'.")
                else:
                    st.subheader(f"{_template} distribution across seasons")
                    _viz_metric = st.radio(
                        "Distribution metric:",
                        ["ACP Projection", "ACP Rating", "Action V/90", "Best-fit Rating"],
                        horizontal=True,
                        key="player_analysis_viz_metric_template",
                    )

                    # Iterate over every season we have data for, sorted
                    # chronologically by parsed start year.
                    def _season_start_year(label):
                        try: return int(str(label).split('/')[0])
                        except (ValueError, AttributeError, IndexError): return None
                    _season_panels = []
                    _all_sids = sorted(
                        [int(s) for s in SEASON_ID_MAP.keys()],
                        key=lambda s: (
                            _season_start_year(SEASON_ID_MAP.get(s, '')) or 0,
                            competition_for_season(s) or 0,
                            s,
                        ),
                    )

                    with st.spinner(f"Computing {_template} distributions across "
                                     f"{len(_all_sids)} seasons…"):
                        for _sid in _all_sids:
                            _evs = get_season_events(raw_events_df, [_sid])
                            _mins = player_minutes_data.get(_sid)
                            if _evs.empty or _mins is None or _mins.empty:
                                continue
                            if _viz_metric == 'Action V/90':
                                # GPA path: filter by season + position
                                if 'load_gpa_values' in globals():
                                    _gpa_all = load_gpa_values()
                                else:
                                    _gpa_all = None
                                if _gpa_all is None or _gpa_all.empty:
                                    continue
                                _val_col = next((c for c in ('Total Value',
                                                  'total_v_per_90')
                                                  if c in _gpa_all.columns), None)
                                if _val_col is None:
                                    continue
                                _sub = _gpa_all[
                                    (_gpa_all['seasonId'] == _sid)
                                    & (_gpa_all.get('mins_played', 0)
                                       >= min_minutes_filter)
                                    & (_gpa_all.get('position', '').astype(str)
                                       .isin(_eligible_positions))
                                ]
                                _sub = _sub[_sub[_val_col].notna()]
                                vals = _sub[_val_col].astype(float).values
                                names = _sub.get('name', pd.Series([''] * len(_sub))).astype(str).tolist()
                                # GPA doesn't carry teamName — leave blank
                                teams = [''] * len(_sub)
                            elif _viz_metric in ('ACP Rating', 'ACP Projection'):
                                # Engine path: per-season engine values,
                                # position-filtered via the GPA table
                                # (engine rows don't carry raw position
                                # codes). Projections exist only for the
                                # current + lapsed seasons, so older
                                # panels skip naturally (<5 rows).
                                _eng_all, _ = load_player_engine()
                                _gpa_all = (load_gpa_values()
                                             if 'load_gpa_values' in globals() else None)
                                if _eng_all.empty or _gpa_all is None or _gpa_all.empty:
                                    continue
                                _ecol = ('acp_rating' if _viz_metric == 'ACP Rating'
                                          else 'projection_abs')
                                _sub = _gpa_all[
                                    (_gpa_all['seasonId'] == _sid)
                                    & (_gpa_all.get('mins_played', 0)
                                       >= min_minutes_filter)
                                    & (_gpa_all.get('position', '').astype(str)
                                       .isin(_eligible_positions))
                                ][['playerId', 'name']].merge(
                                    _eng_all[_eng_all['seasonId'] == _sid][
                                        ['playerId', _ecol]],
                                    on='playerId', how='inner')
                                _sub = _sub[_sub[_ecol].notna()]
                                vals = _sub[_ecol].astype(float).values
                                names = _sub['name'].astype(str).tolist()
                                teams = [''] * len(_sub)
                            else:
                                # Best-fit (this template's specific Role_Score)
                                _stats = calculate_all_player_stats(
                                    _evs, _mins, season_id=_sid
                                )
                                if _stats.empty:
                                    continue
                                _scored = calculate_player_percentiles_and_scores(
                                    _stats, POSITION_GROUPS, WEIGHTS, INVERT_METRICS,
                                    min_minutes=int(min_minutes_filter),
                                    season_id=_sid,
                                )
                                if _scored.empty:
                                    continue
                                _score_col = f"{_template}_Score"
                                if _score_col not in _scored.columns:
                                    continue
                                _sub = _scored[
                                    _scored['primaryPosition'].isin(_eligible_positions)
                                    & _scored[_score_col].notna()
                                    & (_scored.get('totalMinutes', 0)
                                       >= min_minutes_filter)
                                ]
                                vals = _sub[_score_col].astype(float).values
                                names = _sub.get('playerName',
                                            pd.Series([''] * len(_sub))).astype(str).tolist()
                                teams = _sub.get('teamName',
                                            pd.Series([''] * len(_sub))).fillna('').astype(str).tolist()

                            if len(vals) < 5:
                                continue
                            _comp = competition_for_season(_sid)
                            _comp_short = ('L3' if _comp == 43324
                                            else 'CP' if _comp == 702
                                            else (COMPETITIONS.get(_comp, {}).get('name', '') or '')[:6])
                            _season_panels.append({
                                'sid': _sid,
                                'label': (f"{SEASON_ID_MAP.get(_sid, str(_sid))}<br>"
                                           f"<span style='font-size:0.85em;color:#777'>"
                                           f"{_comp_short} · n={len(vals)}</span>"),
                                'values': vals,
                                'names': names,
                                'teams': teams,
                            })

                    if not _season_panels:
                        st.caption(
                            f"No {_viz_metric} data available across seasons "
                            f"for the {_template} template."
                        )
                    else:
                        _pop_concat = np.concatenate([p['values'] for p in _season_panels])
                        _y_lo = float(np.nanmin(_pop_concat))
                        _y_hi = float(np.nanmax(_pop_concat))
                        _y_pad = 0.05 * (_y_hi - _y_lo or 1.0)
                        _max_n = max(len(p['values']) for p in _season_panels) or 1
                        _fig = _make_subplots(
                            rows=1, cols=len(_season_panels),
                            shared_yaxes=True,
                            subplot_titles=[p['label'] for p in _season_panels],
                            horizontal_spacing=0.01,
                        )
                        for _i, _p in enumerate(_season_panels, start=1):
                            _scaled_w = 0.85 * (len(_p['values']) / _max_n) ** 0.5
                            _add_strip(
                                _fig, 1, _i,
                                _p['values'], _p['names'], _p['teams'],
                                _viz_metric, _y_lo, _y_hi, _scaled_w, _p['sid'],
                            )
                        _fig.update_yaxes(range=[_y_lo - _y_pad, _y_hi + _y_pad])
                        _fig.update_layout(
                            title=(f"{_template} · {_viz_metric} distribution "
                                    f"by season · ≥{min_minutes_filter:.0f} min"),
                            height=460,
                            margin=dict(t=70, b=30, l=40, r=20),
                            showlegend=False,
                        )
                        for _ann in _fig['layout']['annotations']:
                            _ann['font'] = dict(size=11)
                        st.plotly_chart(_fig, use_container_width=True)

        # Always keep the expander open on the Player Analysis page so the
        # Cached ZIPs section, debug info, and any errors are unmissable.
        _bulk_expander_open = True
        with st.sidebar.expander("📥 Bulk Export Radars", expanded=_bulk_expander_open):
            _bulk_groups_default = list(_TEMPLATE_GROUPS.keys())
            _bulk_groups = st.multiselect(
                "Position groups:",
                _bulk_groups_default,
                default=_bulk_groups_default,
                key="bulk_export_groups",
            )
            _bulk_mode_label = st.radio(
                "Radar style:",
                ["Percentile", "Raw (mean ± 2σ)"],
                key="bulk_export_mode",
            )
            _bulk_min_mins = st.number_input(
                "Min minutes:",
                min_value=0,
                max_value=int(max_minutes) if max_minutes else 5000,
                value=int(min_minutes_filter),
                step=45,
                key="bulk_export_min_mins",
                help="Default uses the Minimum Minutes Played slider above."
            )
            _bulk_generate = st.button("Generate ZIP", key="bulk_export_btn", use_container_width=True)

            if _bulk_generate:
                # Resolve the multi-select group labels to raw position codes.
                _bulk_raw_codes = set()
                for _grp in _bulk_groups:
                    for _role in _TEMPLATE_GROUPS.get(_grp, []):
                        if _role in POSITION_GROUPS:
                            _bulk_raw_codes.update(POSITION_GROUPS[_role])

                _export_df = player_stats_with_scores_df[
                    (pd.to_numeric(player_stats_with_scores_df['totalMinutes'], errors='coerce').fillna(0) >= _bulk_min_mins) &
                    (player_stats_with_scores_df['primaryPosition'].isin(_bulk_raw_codes))
                ].copy()

                if _export_df.empty:
                    st.warning("No players match the selection.")
                else:
                    _n_total = len(_export_df)
                    _progress = st.progress(0.0, text=f"Rendering 0/{_n_total} radars…")

                    def _on_progress(i, n, name, resumed=0):
                        if resumed:
                            _progress.progress(
                                i / max(n, 1),
                                text=f"Rendering {i}/{n} (resumed {resumed}): {name}"
                            )
                        else:
                            _progress.progress(
                                i / max(n, 1),
                                text=f"Rendering {i}/{n}: {name}"
                            )

                    _radar_mode = 'raw' if _bulk_mode_label.startswith("Raw") else 'percentile'
                    _season_lbl = SEASON_ID_MAP.get(selected_season_id, 'All Seasons') if selected_season_id else 'All Seasons'

                    # Write each PNG into a per-render directory; the
                    # download ZIP is built lazily at click time. Sentinel
                    # meta.pkl is written first so even crashed runs
                    # appear in the Cached list.
                    _cache_key = _bulk_cache_key(_season_lbl, _bulk_groups, _radar_mode, _bulk_min_mins)
                    _render_dir = _bulk_render_dir(_cache_key)
                    _meta_path = _bulk_meta_path(_cache_key)
                    try:
                        _os.makedirs(_render_dir, exist_ok=True)
                        with open(_meta_path, 'wb') as _f:
                            _pickle.dump({
                                'status': 'running',
                                'rendered': 0,
                                'skipped': [],
                                'label': f"{_season_lbl}__{_radar_mode}",
                                'season': _season_lbl,
                                'mode': _radar_mode,
                                'groups': list(_bulk_groups),
                                'min_mins': int(_bulk_min_mins),
                                'started_at': _time.time(),
                            }, _f)
                    except Exception as _sentinel_exc:
                        st.warning(f"⚠️ Could not write sentinel meta: "
                                   f"{type(_sentinel_exc).__name__}: {_sentinel_exc}")

                    try:
                        _result_path, _rendered, _skipped, _resumed = bulk_export_radars(
                            _export_df,
                            player_stats_with_scores_df,
                            radar_mode=_radar_mode,
                            season_label=_season_lbl,
                            progress_cb=_on_progress,
                            output_path=_render_dir,
                        )
                        try:
                            with open(_meta_path, 'wb') as _f:
                                _pickle.dump({
                                    'status': 'complete',
                                    'rendered': _rendered,
                                    'skipped': _skipped,
                                    'resumed': _resumed,
                                    'label': f"{_season_lbl}__{_radar_mode}",
                                    'season': _season_lbl,
                                    'mode': _radar_mode,
                                    'groups': list(_bulk_groups),
                                    'min_mins': int(_bulk_min_mins),
                                }, _f)
                        except Exception as _meta_exc:
                            st.warning(f"⚠️ Render finished but completion-meta "
                                       f"write failed: {type(_meta_exc).__name__}: "
                                       f"{_meta_exc}")
                        _progress.empty()
                        _new_count = _rendered - _resumed
                        _resume_note = (f" (resumed {_resumed}, rendered {_new_count} new)"
                                         if _resumed else "")
                        st.success(
                            f"Rendered {_rendered} radars to disk{_resume_note}"
                            + (f" · {len(_skipped)} skipped" if _skipped else "")
                            + ". Use the Prepare ZIP button below."
                        )
                    except Exception as _gen_exc:
                        _progress.empty()
                        import traceback as _tb
                        st.error(f"Render failed: {type(_gen_exc).__name__}: {_gen_exc}")
                        with st.popover("Traceback (for debugging)", use_container_width=True):
                            st.code(_tb.format_exc())

            # --- Cached Renders section — always shown. ---
            st.markdown("---")
            _cached = _list_cached_renders()
            if _BULK_CACHE_ERROR:
                st.error(f"⚠️ Cache directory unusable: {_BULK_CACHE_ERROR}. "
                         f"Generated renders will not survive the page render. "
                         f"This usually means /tmp/ is not writable in this runtime.")
            if not _cached:
                st.caption("💾 No cached renders yet. Run Generate ZIP to create one.")
            else:
                st.caption(f"💾 Cached renders ({len(_cached)})")
            _now = _time.time()
            for _idx, _entry in enumerate(_cached):
                _meta = _entry['meta']
                _rd = _entry['path']
                _age = _now - _entry['mtime']
                _age_str = (f"{int(_age)}s ago" if _age < 60 else
                            f"{int(_age/60)} min ago" if _age < 3600 else
                            f"{int(_age/3600)} h ago" if _age < 86400 else
                            f"{int(_age/86400)} d ago")
                _size_mb = _entry['size'] / (1024 * 1024)
                _status = _meta.get('status', 'complete')
                _png_count = _entry['png_count']
                _season = _meta.get('season', '?')
                _mode = _meta.get('mode', '?')
                _mm = _meta.get('min_mins', '?')
                _ngroups = len(_meta.get('groups', []) or [])
                _badge = ""
                if _status == 'running':
                    _badge = " · 🟡 interrupted (partial download still works)"
                elif _status == 'incomplete':
                    _badge = " · 🟠 metadata missing (download still works)"
                st.markdown(
                    f"**{_season}** · {_mode} · {_ngroups} groups · ≥{_mm} min{_badge}  \n"
                    f"<span style='color:#888;font-size:0.85em'>{_png_count} radars · "
                    f"{_size_mb:.1f} MB on disk · {_age_str}</span>",
                    unsafe_allow_html=True,
                )
                # Build ZIP lazily from the directory contents. This is
                # the moment we pay the in-memory cost for the ZIP, not
                # during render. Even if the render was interrupted,
                # every PNG that made it to disk gets bundled cleanly.
                _zip_btn_key = f"bulk_export_dl_{_idx}"
                _prep_key = f"bulk_export_prep_{_idx}"
                _zip_bytes_key = f"bulk_export_bytes_{_idx}"
                _zip_fp_key = f"bulk_export_fp_{_idx}"
                # Fingerprint the on-disk state so a previously-built
                # ZIP gets invalidated when the directory has grown
                # (e.g. after a resume run). Without this the download
                # button would happily serve stale bytes.
                _current_fp = f"{_png_count}_{int(_entry['size'])}"
                _cached_fp = st.session_state.get(_zip_fp_key)
                if (_zip_bytes_key in st.session_state
                        and _cached_fp == _current_fp):
                    _cached_size_mb = len(st.session_state[_zip_bytes_key]) / (1024*1024)
                    st.download_button(
                        label=f"⬇️ Download ({_cached_size_mb:.0f} MB)",
                        data=st.session_state[_zip_bytes_key],
                        file_name=f"radars__{_meta.get('label', 'export')}.zip",
                        mime="application/zip",
                        key=_zip_btn_key,
                        use_container_width=True,
                    )
                else:
                    # If we have stale cached bytes, surface that so the
                    # user understands why the prepare button is back.
                    if _zip_bytes_key in st.session_state and _cached_fp:
                        try:
                            _stale_count = int(_cached_fp.split('_')[0])
                            st.caption(f"⚠️ Cached ZIP is stale "
                                       f"({_stale_count} files vs {_png_count} on disk). "
                                       f"Re-prepare to refresh.")
                        except Exception:
                            pass
                    if st.button(f"📦 Prepare ZIP ({_png_count} files, ~{_size_mb:.0f} MB)",
                                  key=_prep_key, use_container_width=True):
                        try:
                            with st.spinner("Building ZIP from rendered PNGs…"):
                                st.session_state[_zip_bytes_key] = _build_zip_from_dir(_rd)
                                st.session_state[_zip_fp_key] = _current_fp
                            st.rerun()
                        except Exception as _zip_exc:
                            st.error(f"ZIP build failed: "
                                     f"{type(_zip_exc).__name__}: {_zip_exc}")
                if _meta.get('skipped'):
                    _sk = _meta['skipped']
                    with st.popover(f"View skipped ({len(_sk)})", use_container_width=True):
                        for _name, _reason in _sk[:50]:
                            st.caption(f"• **{_name}** — {_reason}")
                        if len(_sk) > 50:
                            st.caption(f"…and {len(_sk) - 50} more")

            # --- Diagnostics: surfaces the actual on-disk state ---
            # popover instead of expander — Streamlit forbids nested expanders.
            with st.popover("🔍 Diagnostics", use_container_width=True):
                st.caption(f"Cache dir: `{_BULK_CACHE_DIR}`")
                if _BULK_CACHE_ERROR:
                    st.error(f"Cache dir setup error: {_BULK_CACHE_ERROR}")
                try:
                    _entries = sorted(_os.listdir(_BULK_CACHE_DIR))
                    if not _entries:
                        st.caption("Directory is empty.")
                    else:
                        _rows = []
                        for _en in _entries:
                            _ep = _os.path.join(_BULK_CACHE_DIR, _en)
                            try:
                                if _os.path.isdir(_ep):
                                    _png = [f for f in _os.listdir(_ep)
                                             if f.endswith('.png')]
                                    _sz = sum(_os.path.getsize(_os.path.join(_ep, f))
                                               for f in _os.listdir(_ep))
                                    _mt = _os.path.getmtime(_ep)
                                    _rows.append({
                                        'entry': _en + '/',
                                        'type': 'dir',
                                        'pngs': len(_png),
                                        'size_MB': f"{_sz/(1024*1024):.2f}",
                                        'mtime': _time.strftime('%Y-%m-%d %H:%M:%S',
                                                                _time.localtime(_mt)),
                                    })
                                else:
                                    _sz = _os.path.getsize(_ep)
                                    _mt = _os.path.getmtime(_ep)
                                    _rows.append({
                                        'entry': _en,
                                        'type': 'file',
                                        'pngs': 0,
                                        'size_MB': f"{_sz/(1024*1024):.2f}",
                                        'mtime': _time.strftime('%Y-%m-%d %H:%M:%S',
                                                                _time.localtime(_mt)),
                                    })
                            except Exception:
                                _rows.append({'entry': _en, 'type': '?',
                                              'pngs': 0, 'size_MB': '?', 'mtime': '?'})
                        st.dataframe(pd.DataFrame(_rows),
                                     use_container_width=True, hide_index=True)
                except Exception as _diag_exc:
                    st.error(f"Cannot list cache dir: "
                             f"{type(_diag_exc).__name__}: {_diag_exc}")
                # Disk usage info — useful if /tmp/ is filling up
                try:
                    import shutil as _shutil
                    _du = _shutil.disk_usage(_BULK_CACHE_DIR if _os.path.exists(_BULK_CACHE_DIR) else '/tmp')
                    st.caption(
                        f"`/tmp/` disk: total {_du.total/(1024**3):.1f} GB · "
                        f"used {_du.used/(1024**3):.1f} GB · "
                        f"free {_du.free/(1024**3):.1f} GB"
                    )
                except Exception as _du_exc:
                    st.caption(f"disk_usage error: {_du_exc}")
                # Process RAM, if psutil is available — early-warning for OOM
                try:
                    import psutil as _psutil
                    _proc = _psutil.Process()
                    _rss = _proc.memory_info().rss / (1024**3)
                    _vmem = _psutil.virtual_memory()
                    st.caption(
                        f"Process RSS: {_rss:.2f} GB · "
                        f"system RAM: {_vmem.used/(1024**3):.1f} GB used / "
                        f"{_vmem.total/(1024**3):.1f} GB total "
                        f"({_vmem.percent:.0f}%)"
                    )
                except Exception:
                    pass


    elif analysis_type == 'Match Predictor':
        selected_comp_ids = league_selector("match_predictor")
        st.markdown("Predict the outcome of upcoming matches based on team performance data with season-specific priors.")

        # Load prediction model
        @st.cache_resource
        def load_prediction_model():
            try:
                with open('match_predictor_model.pkl', 'rb') as f:
                    return pickle.load(f)
            except FileNotFoundError:
                return None

        # Helper functions for decay priors
        def get_decay_weight(matches_played, decay_rate=0.15):
            """Exponential decay weight for prior season stats"""
            return np.exp(-decay_rate * matches_played)

        def get_blended_stat(current_value, current_matches, prior_per_game, decay_rate=0.15, default_prior=None):
            """Blend current season stat with decaying prior"""
            if current_matches == 0:
                return prior_per_game if prior_per_game is not None else (default_prior if default_prior else 0.0)
            current_per_game = current_value / current_matches
            if prior_per_game is None:
                return current_per_game
            prior_weight = get_decay_weight(current_matches, decay_rate)
            current_weight = 1 - prior_weight
            return current_weight * current_per_game + prior_weight * prior_per_game

        def calculate_prediction_features(team_stats, prior_stats, league_avg, is_home):
            """Calculate features for a team with decaying priors"""
            curr = team_stats
            m = curr['matches']

            # Get prior per-game stats
            if prior_stats and prior_stats.get('matches', 0) > 0:
                pm = prior_stats['matches']
                prior_ppg = prior_stats['points'] / pm
                prior_gpg = prior_stats['goals_for'] / pm
                prior_gapg = prior_stats['goals_against'] / pm
                prior_xgpg = prior_stats['xG_for'] / pm if prior_stats['xG_for'] > 0 else 1.0
                prior_xgapg = prior_stats['xG_against'] / pm if prior_stats['xG_against'] > 0 else 1.0
                prior_winrate = prior_stats['wins'] / pm
                prior_csrate = prior_stats['clean_sheets'] / pm
                prior_shot_conv = prior_stats['goals_for'] / max(prior_stats['shots_for'], 1)
                prior_sot_rate = prior_stats['sot_for'] / max(prior_stats['shots_for'], 1)
                if is_home and prior_stats['home_matches'] > 0:
                    prior_venue_wr = prior_stats['home_wins'] / prior_stats['home_matches']
                    prior_venue_gpg = prior_stats['home_goals'] / prior_stats['home_matches']
                elif not is_home and prior_stats['away_matches'] > 0:
                    prior_venue_wr = prior_stats['away_wins'] / prior_stats['away_matches']
                    prior_venue_gpg = prior_stats['away_goals'] / prior_stats['away_matches']
                else:
                    prior_venue_wr = prior_winrate
                    prior_venue_gpg = prior_gpg
            else:
                # Promoted team: use league average (slightly below)
                prior_ppg = league_avg.get('ppg', 1.0) * 0.85
                prior_gpg = league_avg.get('gpg', 1.0) * 0.85
                prior_gapg = league_avg.get('gapg', 1.0) * 1.15
                prior_xgpg = league_avg.get('xgpg', 1.0) * 0.85
                prior_xgapg = league_avg.get('xgapg', 1.0) * 1.15
                prior_winrate = 0.28
                prior_csrate = league_avg.get('csrate', 0.25) * 0.85
                prior_shot_conv = league_avg.get('shot_conv', 0.1) * 0.9
                prior_sot_rate = league_avg.get('sot_rate', 0.35) * 0.95
                prior_venue_wr = 0.28
                prior_venue_gpg = prior_gpg

            decay_rate = 0.15
            # Blend current and prior stats
            ppg = get_blended_stat(curr['points'], m, prior_ppg, decay_rate)
            gpg = get_blended_stat(curr['goals_for'], m, prior_gpg, decay_rate)
            gapg = get_blended_stat(curr['goals_against'], m, prior_gapg, decay_rate)
            xgpg = get_blended_stat(curr['xG_for'], m, prior_xgpg, decay_rate)
            xgapg = get_blended_stat(curr['xG_against'], m, prior_xgapg, decay_rate)
            win_rate = get_blended_stat(curr['wins'], m, prior_winrate, decay_rate)
            cs_rate = get_blended_stat(curr['clean_sheets'], m, prior_csrate, decay_rate)

            curr_shot_conv = curr['goals_for'] / max(curr['shots_for'], 1) if m > 0 else 0
            curr_sot_rate = curr['sot_for'] / max(curr['shots_for'], 1) if m > 0 else 0
            shot_conv = curr_shot_conv if m > 3 else prior_shot_conv
            sot_rate = curr_sot_rate if m > 3 else prior_sot_rate

            venue_key = 'home' if is_home else 'away'
            venue_wr = get_blended_stat(
                curr[f'{venue_key}_wins'], curr[f'{venue_key}_matches'],
                prior_venue_wr, decay_rate
            )
            venue_gpg = get_blended_stat(
                curr[f'{venue_key}_goals'], curr[f'{venue_key}_matches'],
                prior_venue_gpg, decay_rate
            )

            gd = gpg - gapg
            xg_diff = xgpg - xgapg
            form = np.mean(curr['last_5_results'][-5:]) if curr['last_5_results'] else 1.0
            xg_form = np.mean(curr['last_5_xG'][-5:]) if curr['last_5_xG'] else prior_xgpg

            return {
                'ppg': ppg, 'gpg': gpg, 'gapg': gapg, 'xgpg': xgpg, 'xgapg': xgapg,
                'win_rate': win_rate, 'cs_rate': cs_rate, 'shot_conv': shot_conv,
                'sot_rate': sot_rate, 'gd': gd, 'xg_diff': xg_diff, 'form': form,
                'xg_form': xg_form, 'venue_wr': venue_wr, 'venue_gpg': venue_gpg
            }

        model_data = load_prediction_model()

        if model_data is None:
            st.error("Prediction model not found. Please ensure 'match_predictor_model.pkl' exists.")
        else:
            model = model_data['model']
            scaler = model_data['scaler']
            team_stats = model_data['team_stats']
            prior_season_stats = model_data.get('prior_season_stats', {})
            league_avg_stats = model_data.get('league_avg_stats', {'ppg': 1.0, 'gpg': 1.19, 'gapg': 1.19, 'xgpg': 1.0, 'xgapg': 1.0, 'csrate': 0.25, 'shot_conv': 0.1, 'sot_rate': 0.35})
            team_ratings = model_data.get('team_ratings', {})

            # Display Team Strength Ratings (Multi-Season with SOS)
            with st.expander("Team Strength Ratings", expanded=False):
                # Build season options filtered to selected league(s)
                league_season_map = {}
                for cid in selected_comp_ids:
                    if cid in COMPETITIONS:
                        league_season_map.update(COMPETITIONS[cid]["seasons"])
                league_season_labels = list(league_season_map.values())
                # Default to current season for selected league
                default_label = league_season_map.get(
                    COMPETITIONS[selected_comp_ids[0]].get("current_season") if selected_comp_ids else CURRENT_SEASON_ID,
                    league_season_labels[0] if league_season_labels else None
                )
                rating_seasons = st.multiselect(
                    "Seasons", league_season_labels,
                    default=[default_label] if default_label else [],
                    key="rating_seasons"
                )
                # Reverse-lookup season IDs from display names (league-filtered)
                season_name_to_id = {v: k for k, v in league_season_map.items()}
                rating_rows = []
                for season_name in rating_seasons:
                    sid = season_name_to_id[season_name]
                    s_events = get_filtered_events(raw_events_df, sid, selected_comp_ids)
                    s_matches = filter_by_league(get_season_matches(matches_summary_df, sid), selected_comp_ids)
                    ts_df = calculate_team_strength(s_events, s_matches, season_id=sid)
                    if ts_df.empty:
                        continue
                    rolling_df = calculate_rolling_team_strength(s_events, s_matches, season_id=sid)
                    sos_df = calculate_sos_adjusted_strength(rolling_df, ts_df, season_id=sid)
                    for team in ts_df.index:
                        raw_att = ts_df.loc[team, 'Attacking Strength']
                        raw_def = ts_df.loc[team, 'Defending Strength']
                        sos_att = sos_df.loc[team, 'sos_att'] if team in sos_df.index else raw_att
                        sos_def = sos_df.loc[team, 'sos_def'] if team in sos_df.index else raw_def
                        sos_factor = sos_df.loc[team, 'sos_factor'] if team in sos_df.index else np.nan
                        # Count matches from rolling data
                        team_rolling = rolling_df[rolling_df['team'] == team] if not rolling_df.empty else pd.DataFrame()
                        n_matches = int(team_rolling['match_number'].max()) + 1 if not team_rolling.empty else 0
                        rating_rows.append({
                            'Rank': 0,
                            'Team': team,
                            'Season': season_name,
                            'Att Strength': round(raw_att, 3),
                            'Def Strength': round(raw_def, 3),
                            'SOS Att': round(float(sos_att), 3),
                            'SOS Def': round(float(sos_def), 3),
                            'SOS Factor': round(float(sos_factor), 3) if not np.isnan(float(sos_factor)) else None,
                            'Matches': n_matches,
                        })
                if rating_rows:
                    ratings_combined = pd.DataFrame(rating_rows)
                    # Overall = att - def, rescaled to 0-100
                    raw_overall = ratings_combined['SOS Att'] - ratings_combined['SOS Def']
                    ov_min = raw_overall.min()
                    ov_max = raw_overall.max()
                    if ov_max > ov_min:
                        ratings_combined['Overall'] = round(((raw_overall - ov_min) / (ov_max - ov_min)) * 100, 1)
                    else:
                        ratings_combined['Overall'] = 50.0
                    ratings_combined = ratings_combined.sort_values('Overall', ascending=False).reset_index(drop=True)
                    ratings_combined['Rank'] = range(1, len(ratings_combined) + 1)
                    st.dataframe(ratings_combined, use_container_width=True, hide_index=True, column_config=auto_column_config(ratings_combined))
                else:
                    st.info("No team strength data available for selected seasons.")

            # Season Simulation - Promotion/Relegation Probabilities
            @st.cache_data
            def load_simulation_data():
                try:
                    with open('season_simulation.pkl', 'rb') as f:
                        return pickle.load(f)
                except FileNotFoundError:
                    return None

            sim_data = load_simulation_data()
            if sim_data is None:
                st.info("Season simulation not yet available. Run simulate_season.py to generate probabilities.")
            else:
                st.subheader("Promotion & Relegation Probabilities")
                sim_ts = sim_data.get('timestamp', '')
                n_sims = sim_data.get('n_simulations', 0)
                st.caption(f"Based on {n_sims:,} Monte Carlo simulations | Updated: {sim_ts[:16].replace('T', ' ')}")

                def render_probability_table(group_name, prob_df, matches_remaining, bonus_points=None, expanded=False, current_standings=None, playoff_pct=None, promotion_pct=None):
                    """Render a color-coded probability table for a second-stage group."""
                    n_teams = len(prob_df)
                    pos_cols = [str(i+1) for i in range(n_teams)]
                    is_serie = group_name.startswith('Série')
                    is_playoff_group = group_name.startswith('Promotion Playoff')

                    # Build lookup for points and matches played from current standings
                    standings_lookup = {}
                    if current_standings is not None:
                        for _, row in current_standings.iterrows():
                            standings_lookup[row['Team']] = {'P': row['P'], 'Pts': row['Pts']}

                    with st.expander(f"{group_name} ({matches_remaining} matches remaining)", expanded=expanded):
                        # Build HTML table
                        html = '<table style="width:100%;border-collapse:collapse;font-size:0.85em;text-align:center;">'

                        # Header row
                        html += '<tr style="border-bottom:2px solid #444;">'
                        html += '<th style="text-align:left;padding:6px 10px;">Team</th>'
                        html += '<th style="padding:6px 8px;">P</th>'
                        html += '<th style="padding:6px 8px;">Pts</th>'
                        for p in pos_cols:
                            html += f'<th style="padding:6px 8px;">{p}</th>'

                        # Summary column header
                        if group_name == 'Promotion':
                            html += '<th style="padding:6px 8px;border-left:2px solid #444;">Promo %</th>'
                            html += '<th style="padding:6px 8px;">Playoff %</th>'
                        elif is_playoff_group:
                            html += '<th style="padding:6px 8px;border-left:2px solid #444;">Promo %</th>'
                        elif is_serie:
                            html += '<th style="padding:6px 8px;border-left:2px solid #444;">Playoff %</th>'
                            html += '<th style="padding:6px 8px;">Promo %</th>'
                            html += '<th style="padding:6px 8px;border-left:2px solid #444;">Releg %</th>'
                        else:
                            html += '<th style="padding:6px 8px;border-left:2px solid #444;">Releg %</th>'
                        html += '</tr>'

                        # Data rows
                        for team in prob_df.index:
                            html += '<tr style="border-bottom:1px solid #ddd;">'
                            html += f'<td style="text-align:left;padding:6px 10px;font-weight:bold;white-space:nowrap;">{team}</td>'
                            team_info = standings_lookup.get(team, {'P': 0, 'Pts': 0})
                            html += f'<td style="padding:6px 8px;color:#888;">{team_info["P"]}</td>'
                            total_pts = team_info["Pts"] + (bonus_points.get(team, 0) if bonus_points else 0)
                            html += f'<td style="padding:6px 8px;font-weight:bold;">{total_pts}</td>'

                            for p in pos_cols:
                                val = prob_df.loc[team, p]
                                pos_num = int(p)

                                # Determine cell color
                                bg = ''
                                if group_name == 'Promotion':
                                    if pos_num <= 2:
                                        intensity = min(val * 1.2, 1.0)
                                        bg = f'background-color:rgba(46,204,113,{intensity:.2f});'
                                    elif pos_num == 3:
                                        intensity = min(val * 1.2, 1.0)
                                        bg = f'background-color:rgba(241,196,15,{intensity:.2f});'
                                elif is_playoff_group:
                                    if pos_num <= 2:
                                        intensity = min(val * 1.2, 1.0)
                                        bg = f'background-color:rgba(46,204,113,{intensity:.2f});'
                                elif is_serie:
                                    if pos_num <= 2:
                                        # Green for playoff qualification positions
                                        intensity = min(val * 1.2, 1.0)
                                        bg = f'background-color:rgba(46,204,113,{intensity:.2f});'
                                    elif pos_num >= n_teams - 4:
                                        # Red for relegation positions (bottom 5)
                                        intensity = min(val * 1.2, 1.0)
                                        bg = f'background-color:rgba(231,76,60,{intensity:.2f});'
                                else:
                                    if pos_num >= n_teams - 1:
                                        intensity = min(val * 1.2, 1.0)
                                        bg = f'background-color:rgba(231,76,60,{intensity:.2f});'

                                cell_text = f'{val:.1%}' if val >= 0.005 else ''
                                html += f'<td style="padding:6px 8px;{bg}">{cell_text}</td>'

                            # Summary columns
                            if group_name == 'Promotion':
                                promo_pct = prob_df.loc[team, '1'] + prob_df.loc[team, '2']
                                po_pct = prob_df.loc[team, '3']
                                promo_bg = f'background-color:rgba(46,204,113,{min(promo_pct * 1.2, 1.0):.2f});'
                                po_bg = f'background-color:rgba(241,196,15,{min(po_pct * 1.2, 1.0):.2f});'
                                html += f'<td style="padding:6px 8px;border-left:2px solid #444;font-weight:bold;{promo_bg}">{promo_pct:.1%}</td>'
                                html += f'<td style="padding:6px 8px;font-weight:bold;{po_bg}">{po_pct:.1%}</td>'
                            elif is_playoff_group:
                                team_promo = promotion_pct.get(team, 0) if promotion_pct else 0
                                promo_bg = f'background-color:rgba(46,204,113,{min(team_promo * 1.2, 1.0):.2f});'
                                html += f'<td style="padding:6px 8px;border-left:2px solid #444;font-weight:bold;{promo_bg}">{team_promo:.1%}</td>'
                            elif is_serie:
                                # Playoff % = chance of finishing top 2 in série
                                team_playoff = playoff_pct.get(team, 0) if playoff_pct else 0
                                # Promotion % = chance of top 2 in série AND top 2 in playoff group
                                team_promo = promotion_pct.get(team, 0) if promotion_pct else 0
                                # Relegation % = bottom 5 positions
                                releg_positions = [str(i) for i in range(n_teams - 4, n_teams + 1)]
                                team_releg = sum(prob_df.loc[team, p] for p in releg_positions if p in prob_df.columns)

                                playoff_bg = f'background-color:rgba(46,204,113,{min(team_playoff * 1.2, 1.0):.2f});'
                                promo_bg = f'background-color:rgba(46,204,113,{min(team_promo * 2.0, 1.0):.2f});'
                                releg_bg = f'background-color:rgba(231,76,60,{min(team_releg * 1.2, 1.0):.2f});'
                                html += f'<td style="padding:6px 8px;border-left:2px solid #444;font-weight:bold;{playoff_bg}">{team_playoff:.1%}</td>'
                                html += f'<td style="padding:6px 8px;font-weight:bold;{promo_bg}">{team_promo:.1%}</td>'
                                html += f'<td style="padding:6px 8px;border-left:2px solid #444;font-weight:bold;{releg_bg}">{team_releg:.1%}</td>'
                            else:
                                releg_pct = prob_df.loc[team, str(n_teams - 1)] + prob_df.loc[team, str(n_teams)]
                                releg_bg = f'background-color:rgba(231,76,60,{min(releg_pct * 1.2, 1.0):.2f});'
                                html += f'<td style="padding:6px 8px;border-left:2px solid #444;font-weight:bold;{releg_bg}">{releg_pct:.1%}</td>'

                            html += '</tr>'

                        html += '</table>'
                        st.markdown(html, unsafe_allow_html=True)

                # Display simulation results for selected competition(s)
                competitions_data = sim_data.get('competitions', {})
                if not competitions_data:
                    # Backward compat: old format has 'groups' at top level (Liga 3 only)
                    competitions_data = {43324: {'competition_name': 'Liga 3', 'groups': sim_data.get('groups', {})}}

                for comp_id in selected_comp_ids:
                    comp_sim = competitions_data.get(comp_id, {})
                    sim_groups = comp_sim.get('groups', {})
                    if not sim_groups:
                        continue

                    comp_name = comp_sim.get('competition_name', COMPETITIONS.get(comp_id, {}).get('name', ''))
                    if len(selected_comp_ids) > 1:
                        st.markdown(f"#### {comp_name}")

                    for group_name, g in sim_groups.items():
                        expanded = (
                            group_name == 'Promotion'
                            or group_name.startswith('Promotion Playoff')
                            or (comp_id == 702 and group_name == list(sim_groups.keys())[0])
                        )
                        render_probability_table(
                            group_name, g['position_probabilities'], g['matches_remaining'],
                            bonus_points=g.get('bonus_points'), expanded=expanded,
                            current_standings=g.get('current_standings'),
                            playoff_pct=g.get('playoff_pct'),
                            promotion_pct=g.get('promotion_pct'),
                        )

            # Team selection — cross-season team-season combos
            all_season_options = {}  # {"Team (Season)": (team_name, season_id)}
            # Filter seasons by selected league(s)
            available_sids = set()
            for cid in selected_comp_ids:
                if cid in COMPETITIONS:
                    available_sids.update(COMPETITIONS[cid]["seasons"].keys())
            pred_events = filter_by_league(raw_events_df, selected_comp_ids)
            pred_matches = filter_by_league(matches_summary_df, selected_comp_ids)
            for sid in sorted(available_sids, reverse=True):
                season_cum = build_season_cumulative_stats(pred_events, pred_matches, sid)
                for team_name, stats in season_cum.items():
                    if stats['matches'] >= 3:
                        label = f"{team_name} ({SEASON_ID_MAP[sid]})"
                        all_season_options[label] = (team_name, sid)

            sorted_options = sorted(all_season_options.keys())
            col1, col2 = st.columns(2)
            with col1:
                home_label = st.selectbox("Home Team", sorted_options, key="pred_home")
            with col2:
                away_options = [t for t in sorted_options if t != home_label]
                away_label = st.selectbox("Away Team", away_options, key="pred_away")

            if st.button("Predict Match Outcome", type="primary"):
                home_team_name, home_sid = all_season_options[home_label]
                away_team_name, away_sid = all_season_options[away_label]
                home_cum = build_season_cumulative_stats(raw_events_df, matches_summary_df, home_sid)[home_team_name]
                away_cum = build_season_cumulative_stats(raw_events_df, matches_summary_df, away_sid)[away_team_name]

                home_prior = home_cum.get('prior_stats')
                away_prior = away_cum.get('prior_stats')

                # Calculate features with decay priors
                home_feats = calculate_prediction_features(home_cum, home_prior, league_avg_stats, is_home=True)
                away_feats = calculate_prediction_features(away_cum, away_prior, league_avg_stats, is_home=False)

                feature_vector = [
                    home_feats['ppg'], away_feats['ppg'], home_feats['ppg'] - away_feats['ppg'],
                    home_feats['form'], away_feats['form'], home_feats['form'] - away_feats['form'],
                    home_feats['gpg'], away_feats['gpg'], home_feats['gpg'] - away_feats['gpg'],
                    home_feats['gd'], away_feats['gd'], home_feats['gd'] - away_feats['gd'],
                    home_feats['xgpg'], away_feats['xgpg'], home_feats['xgpg'] - away_feats['xgpg'],
                    home_feats['xgapg'], away_feats['xgapg'],
                    home_feats['xg_diff'], away_feats['xg_diff'], home_feats['xg_diff'] - away_feats['xg_diff'],
                    home_feats['xg_form'], away_feats['xg_form'],
                    home_feats['win_rate'], away_feats['win_rate'], home_feats['win_rate'] - away_feats['win_rate'],
                    home_feats['venue_wr'], away_feats['venue_wr'],
                    home_feats['shot_conv'], away_feats['shot_conv'],
                    home_feats['sot_rate'], away_feats['sot_rate'],
                    home_feats['cs_rate'], away_feats['cs_rate'],
                    home_feats['venue_gpg'], away_feats['venue_gpg'],
                ]

                X = scaler.transform([feature_vector])
                proba = model.predict_proba(X)[0]
                pred = model.predict(X)[0]

                # Display results
                st.subheader(f"{home_label} vs {away_label}")

                # Show team strength ratings
                if team_ratings:
                    home_rating = team_ratings.get(home_team_name, {})
                    away_rating = team_ratings.get(away_team_name, {})
                    if home_rating and away_rating:
                        rcol1, rcol2 = st.columns(2)
                        with rcol1:
                            st.caption(f"**{home_label}** Rating: {home_rating['overall']:.1f}")
                        with rcol2:
                            st.caption(f"**{away_label}** Rating: {away_rating['overall']:.1f}")

                # Probability bars
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Home Win", f"{proba[1]:.1%}")
                    st.progress(proba[1])
                with col2:
                    st.metric("Draw", f"{proba[0]:.1%}")
                    st.progress(proba[0])
                with col3:
                    st.metric("Away Win", f"{proba[2]:.1%}")
                    st.progress(proba[2])

                # Prediction
                if pred == 1:
                    st.success(f"**Predicted Outcome: {home_label} Win**")
                elif pred == 2:
                    st.success(f"**Predicted Outcome: {away_label} Win**")
                else:
                    st.info(f"**Predicted Outcome: Draw**")

                # Team stats comparison (using blended stats)
                st.subheader("Team Comparison (Season Stats with Priors)")
                comparison_data = {
                    'Metric': ['Points/Game', 'Goals/Game', 'xG/Game', 'xG Against/Game', 'Win Rate', 'Form (Last 5)', 'Clean Sheet Rate'],
                    home_label: [f"{home_feats['ppg']:.2f}", f"{home_feats['gpg']:.2f}", f"{home_feats['xgpg']:.2f}", f"{home_feats['xgapg']:.2f}", f"{home_feats['win_rate']:.1%}", f"{home_feats['form']:.2f}", f"{home_feats['cs_rate']:.1%}"],
                    away_label: [f"{away_feats['ppg']:.2f}", f"{away_feats['gpg']:.2f}", f"{away_feats['xgpg']:.2f}", f"{away_feats['xgapg']:.2f}", f"{away_feats['win_rate']:.1%}", f"{away_feats['form']:.2f}", f"{away_feats['cs_rate']:.1%}"]
                }
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True, column_config=auto_column_config(comparison_df))

                # Show matches played
                home_matches = home_cum['matches']
                away_matches = away_cum['matches']
                st.caption(f"Based on {home_matches} matches for {home_label} and {away_matches} matches for {away_label}")

    # ==========================================================================
    # SHADOW TEAM BUILDER
    # ==========================================================================
    elif analysis_type == 'Shadow Team':

        # --- League & Season Selector ---
        selected_comp_ids = league_selector("shadow_team")
        selected_season_id = season_selector("shadow_team", include_all_seasons=True, comp_ids=selected_comp_ids)
        active_season_ids = get_season_ids_for_selection(selected_season_id, selected_comp_ids)
        shadow_events_df = get_filtered_events(raw_events_df, active_season_ids, selected_comp_ids)
        shadow_player_minutes_df = get_season_player_minutes(player_minutes_data, active_season_ids, comp_ids=selected_comp_ids)

        # --- Load Data ---
        player_details_df = load_player_details()

        try:
            with st.spinner("Loading player statistics..."):
                player_stats_df, player_stats_with_scores_df = load_and_score_player_stats(
                    shadow_events_df, shadow_player_minutes_df, selected_season_id, active_season_ids, selected_comp_ids
                )
        except Exception as e:
            st.error(f"An error occurred calculating player stats: {e}")
            logger.exception("Error in Shadow Team stats calculation")
            st.stop()

        if player_stats_with_scores_df.empty:
            st.warning("No players found with sufficient minutes for analysis.")
            st.stop()

        # Build player options sorted by minutes desc, using "Name (Team)" for uniqueness
        player_list_df = player_stats_with_scores_df[['playerId', 'playerName', 'teamName', 'totalMinutes']].copy()
        player_list_df = player_list_df.sort_values('totalMinutes', ascending=False)
        player_list_df['display_key'] = player_list_df['playerName'] + ' (' + player_list_df['teamName'] + ')'
        player_display_names = player_list_df['display_key'].tolist()
        # Map display_key -> index in stats df for exact lookups
        display_key_to_idx = {}
        for idx, row in player_list_df.iterrows():
            display_key_to_idx[row['display_key']] = idx

        # --- Sidebar Controls ---
        st.sidebar.subheader("Formation")
        formation_key = st.sidebar.selectbox("Select Formation", list(FORMATION_COORDS.keys()), key="shadow_formation")
        formation_data = FORMATION_COORDS[formation_key]

        st.sidebar.markdown("---")
        st.sidebar.subheader("Save / Load")
        team_name_input = st.sidebar.text_input("Team Name", value="My Shadow Team", key="shadow_team_name_input")

        if st.sidebar.button("Save Team", key="shadow_save_btn"):
            players = {}
            tags = {}
            for slot in formation_data['positions']:
                p_key = f"shadow_players_{slot}"
                selected = st.session_state.get(p_key, [])
                players[slot] = selected
                slot_tags = {}
                for p_name in selected:
                    t_key = f"shadow_tag_{slot}_{p_name}"
                    l_key = f"shadow_label_{slot}_{p_name}"
                    slot_tags[p_name] = {
                        'category': st.session_state.get(t_key, 'Current Starter'),
                        'label': st.session_state.get(l_key, ''),
                    }
                tags[slot] = slot_tags
            st.session_state.shadow_teams[team_name_input] = {
                'formation': formation_key,
                'players': players,
                'tags': tags,
            }
            st.sidebar.success(f"Saved '{team_name_input}'!")

        saved_names = list(st.session_state.shadow_teams.keys())
        if saved_names:
            load_name = st.sidebar.selectbox("Load Saved Team", saved_names, key="shadow_load_select")
            if st.sidebar.button("Load Team", key="shadow_load_btn"):
                saved = st.session_state.shadow_teams[load_name]
                st.session_state['shadow_formation'] = saved['formation']
                for slot, player_list in saved['players'].items():
                    st.session_state[f"shadow_players_{slot}"] = player_list
                for slot, slot_tags in saved['tags'].items():
                    for p_name, tag_info in slot_tags.items():
                        st.session_state[f"shadow_tag_{slot}_{p_name}"] = tag_info.get('category', 'Current Starter')
                        st.session_state[f"shadow_label_{slot}_{p_name}"] = tag_info.get('label', '')
                st.rerun()

        # --- Tag Legend (sidebar) ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("Tag Legend")
        for tag_name, tag_color in SHADOW_TAG_CATEGORIES.items():
            st.sidebar.markdown(
                f'<span style="display:inline-block;width:12px;height:12px;'
                f'background-color:{tag_color};border-radius:50%;margin-right:6px;'
                f'vertical-align:middle;"></span>'
                f'<span style="vertical-align:middle;">{tag_name}</span>',
                unsafe_allow_html=True
            )

        # --- Main Content: Two Columns ---
        left_col, right_col = st.columns([3, 2])

        # Gather current assignments from widget state
        player_assignments = {}  # {slot: [player_name, ...]}
        tag_assignments = {}     # {slot: {player_name: {category, label}}}
        tag_categories_list = list(SHADOW_TAG_CATEGORIES.keys())

        with right_col:
            st.subheader("Assign Players")
            for slot in formation_data['positions']:
                with st.expander(f"{slot}", expanded=False):
                    selected_players = st.multiselect(
                        "Players", player_display_names,
                        key=f"shadow_players_{slot}"
                    )
                    player_assignments[slot] = selected_players
                    slot_tags = {}
                    for p_name in selected_players:
                        st.markdown(f"**{p_name}**")
                        c1, c2 = st.columns(2)
                        with c1:
                            selected_tag = st.selectbox(
                                "Tag", tag_categories_list,
                                key=f"shadow_tag_{slot}_{p_name}"
                            )
                        with c2:
                            custom_label = st.text_input(
                                "Label", value="",
                                key=f"shadow_label_{slot}_{p_name}"
                            )
                        slot_tags[p_name] = {
                            'category': selected_tag,
                            'label': custom_label,
                        }
                    tag_assignments[slot] = slot_tags

        with left_col:
            st.subheader("Formation View")
            fig = create_shadow_team_graphic(formation_key, player_assignments, tag_assignments, team_name_input, player_stats_with_scores_df)
            st.pyplot(fig)
            plt.close(fig)

            # Export PNG
            buf = io.BytesIO()
            fig_export = create_shadow_team_graphic(formation_key, player_assignments, tag_assignments, team_name_input, player_stats_with_scores_df)
            fig_export.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='#1a472a')
            plt.close(fig_export)
            buf.seek(0)
            st.download_button(
                label="Download PNG",
                data=buf,
                file_name=f"shadow_team_{team_name_input.replace(' ', '_')}.png",
                mime="image/png",
                key="shadow_download_png"
            )

        # --- Player Details Panel ---
        st.markdown("---")
        st.subheader("Player Details")

        for slot in formation_data['positions']:
            slot_players = player_assignments.get(slot, [])
            if not slot_players:
                continue

            for p_display_key in slot_players:
                # Find player row via display_key index lookup
                row_idx = display_key_to_idx.get(p_display_key)
                if row_idx is None:
                    continue
                player_row = player_stats_with_scores_df.loc[row_idx]
                player_id = player_row.get('playerId', None)
                p_name = player_row.get('playerName', p_display_key)
                team = player_row.get('teamName', 'N/A')
                minutes = player_row.get('totalMinutes', 0)
                primary_pos = player_row.get('primaryPosition', 'N/A')

                # Get tag info for display
                p_tag_info = tag_assignments.get(slot, {}).get(p_display_key, {})
                p_tag = p_tag_info.get('category', '')
                p_tag_color = SHADOW_TAG_CATEGORIES.get(p_tag, '#ffffff')

                # Get age from player_details
                age_str = "N/A"
                if player_id is not None and not player_details_df.empty:
                    pid = int(player_id) if pd.notna(player_id) else None
                    if pid is not None and pid in player_details_df.index:
                        birth_date = player_details_df.loc[pid].get('birthDate', None)
                        age_val = _calculate_age(birth_date)
                        if age_val != "N/A":
                            age_str = f"{age_val:.1f}"

                with st.expander(f"{slot}: {p_name} ({team})", expanded=False):
                    # Tag badge
                    if p_tag:
                        st.markdown(
                            f'<span style="background-color:{p_tag_color};color:#fff;'
                            f'padding:2px 8px;border-radius:10px;font-size:0.8em;">'
                            f'{p_tag}</span>',
                            unsafe_allow_html=True
                        )
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Team", team)
                    m2.metric("Age", age_str)
                    m3.metric("Minutes", f"{int(minutes):,}")
                    m4.metric("Position", primary_pos)

                    # Role scores table
                    role_scores = get_player_role_scores(player_row, slot)
                    if role_scores:
                        scores_df = pd.DataFrame(list(role_scores.items()), columns=['Role', 'Score'])
                        st.dataframe(scores_df, use_container_width=True, hide_index=True, column_config=auto_column_config(scores_df))
                    else:
                        st.caption("No role scores available for this slot.")

    elif analysis_type == 'Opposition Report':
        selected_comp_ids = league_selector("opposition_report")
        from opposition_report import render_opposition_report
        opp_events = filter_by_league(raw_events_df, selected_comp_ids)
        opp_matches = filter_by_league(matches_summary_df, selected_comp_ids)
        # Build season map for selected leagues
        opp_season_map = {}
        for cid in selected_comp_ids:
            if cid in COMPETITIONS:
                opp_season_map.update(COMPETITIONS[cid]["seasons"])
        opp_current_sid = COMPETITIONS[selected_comp_ids[0]]["current_season"] if selected_comp_ids else CURRENT_SEASON_ID
        render_opposition_report(
            opp_events, opp_matches, all_match_data,
            season_team_stats, player_minutes_data,
            opp_current_sid, opp_season_map,
        )


else:
    st.error("Data files not loaded. Please run `process_data.py` locally and ensure all artifacts are pushed to GitHub.")

# Free every matplotlib figure created during this rerun. st.pyplot has
# already rasterized them to PNG; without this, the 41 figure call sites
# accumulate across reruns and leak memory on the HF Space.
plt.close('all')
