# app.py

# Dump a C-level + Python traceback to stderr on SIGSEGV/SIGFPE/SIGABRT.
# The Space has been exiting 139 with nothing in the logs to act on; HF captures
# stderr, so the next fault lands in the container log instead of being inferred.
# Must come before anything that loads native code (numpy/pyarrow/matplotlib).
import faulthandler
faulthandler.enable()

import sys
import streamlit as st
import pandas as pd
from event_tags import TagIndex
import numpy as np
import pickle
import logging
import yaml
# Selects the Agg backend before pyplot is imported, and owns MPL_LOCK, the
# process-wide lock serialising all matplotlib work (see mpl_safety.py).
from mpl_safety import MPL_LOCK, mpl_locked
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
from pathlib import Path  # used by the Player Profile manual-entries path (was an unimported name)
# ... after your other imports ...
import base64
import pitch_visualizations as pv
import obv_viz
import theme  # colours + figure conventions shared with every plotter
import navigation
import context_bar
import views.home
import views.opposition
import views.shadow_team
import views.match_predictor
import views.player_analysis
import views.player_comparison
import views.player_profile
import views.league_analysis
import views.team_analysis
import views.match_analysis


# ------------------------------------------------------------------------------
# groupby(observed=True) shim — REQUIRED while raw_events string columns are
# stored as pandas category dtype (see load_data). With categorical keys,
# pandas' observed=False default emits a row for EVERY category (all 109 teams
# in a 2-team match groupby, cartesian blowups on multi-key groupbys). Every
# groupby in this codebase was written for object-dtype keys, i.e. observed
# semantics, so default observed=True process-wide. Explicit observed=... at a
# callsite still wins; non-categorical keys are unaffected. pandas 3.0 makes
# observed=True the default, at which point this shim becomes a no-op.
_orig_df_groupby = pd.DataFrame.groupby
_orig_ser_groupby = pd.Series.groupby

def _df_groupby_observed(self, *args, **kwargs):
    kwargs.setdefault('observed', True)
    return _orig_df_groupby(self, *args, **kwargs)

def _ser_groupby_observed(self, *args, **kwargs):
    kwargs.setdefault('observed', True)
    return _orig_ser_groupby(self, *args, **kwargs)

pd.DataFrame.groupby = _df_groupby_observed
pd.Series.groupby = _ser_groupby_observed


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
CURRENT_SEASON_ID = 192831  # Liga 3 default (2026/27)
STATS_CACHE_DIR = 'stats_cache'
STATS_CACHE_VERSION = 'v14'  # Bump when stat COLUMNS or cached VALUES change (e.g. the
                             # 2026-06 minutes fixes: Camará alias dedup + Manuel Pedro
                             # override). v13 percentiles cache served stale minutes
                             # because the percentiles layer early-returns its disk cache
                             # and only this version key invalidates it.
FIGURE_CACHE_VERSION = 'v4'  # Bump when any DRAWING code behind the cached-PNG figure
                             # renderers changes (_render_match_figure_png /
                             # _render_team_figure_png / _render_league_figure_png /
                             # opposition_report._render_opp_figure_png, and
                             # everything they call:
                             # create_match_shotmap, plot_xg_flowchart, plot_radar_chart,
                             # create_season_shotmap, plot_corner_analysis, the
                             # pitch_visualizations plotters, ...). Those renderers key on
                             # the DATA scope only, so a pure drawing change (colour,
                             # label, marker size) is invisible to the key and would keep
                             # serving the old PNG until the 24 h TTL expires. Same role
                             # STATS_CACHE_VERSION plays for the stat caches.


def _cache_meta_path(cache_path):
    return cache_path + '.meta.json'


def _write_cache_meta(cache_path, fingerprint):
    """Sidecar stamp recording what data built a stats cache."""
    try:
        with open(_cache_meta_path(cache_path), 'w') as f:
            json.dump(fingerprint, f)
    except Exception:
        pass


def _cache_is_stale(cache_path, fingerprint):
    """True when the sidecar stamp exists and disagrees with current data —
    i.e. a data refresh landed but the cache predates it (on-demand recompute
    covers the window until the engine rebuild redeploys warmed caches).
    Legacy caches without a stamp are trusted, so old deploys can't trigger
    recompute storms (the 2026-07 segfault failure mode)."""
    try:
        with open(_cache_meta_path(cache_path)) as f:
            meta = json.load(f)
    except Exception:
        return False
    return meta != fingerprint


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
        # Only load columns actually used in the app. Audited 2026-07-14: every
        # column below has a real read site in app.py / pitch_visualizations /
        # pitch_interactive / opposition_report / player_onepager. Dropped as
        # dead weight (never read, ~150 MB deep): 'pass', 'shot', 'infraction'
        # (all-null struct markers) and 'relatedEventId'.
        events_columns = [
            'id', 'matchId', 'seasonId', 'competitionId', 'minute', 'second', 'matchTimestamp',
            'type.primary', 'type.secondary', 'player.id', 'player.name', 'player.position',
            # team.id: read by the OBV visuals (momentum team mapping, stats-table
            # augmentation) added 2026-08 — the engine aggregates key on Wyscout ids
            'team.id', 'team.name', 'opponentTeam.name', 'location.x', 'location.y',
            'pass.accurate', 'pass.endLocation.x', 'pass.endLocation.y', 'pass.length',
            'shot.xg', 'shot.isGoal', 'shot.onTarget', 'shot.bodyPart', 'shot.postShotXg', 'shot.goalkeeper.id',
            'groundDuel.duelType', 'groundDuel.keptPossession', 'groundDuel.progressedWithBall',
            'groundDuel.recoveredPossession', 'groundDuel.stoppedProgress', 'groundDuel.takeOn',
            'aerialDuel.firstTouch', 'carry.endLocation.x', 'carry.endLocation.y',
            'possession.id', 'possession.eventIndex', 'possession.duration', 'possession.team.name', 'possession.types',
            'infraction.type', 'infraction.yellowCard', 'infraction.redCard',
            'is_dribble_attempt', 'is_custom_dribble_success',
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

        # ---- Memory slim (2026-07-14): 6.15 GB -> ~3.5 GB deep -------------
        # High-repetition string columns become category dtype (nunique 3..3.5k
        # over 4.7M rows). Scoped copies cut by _get_filtered_events_cached
        # inherit the dtype, so the per-scope cache entries shrink too.
        # REQUIRES the pd.*.groupby(observed=True) shim at the top of this
        # module — without it, groupbys keyed on these columns emit zero-rows
        # for every out-of-scope category. type.secondary / possession.types
        # are list-typed and stay object. Must run AFTER the free-kick
        # type.primary rewrite above ('shot' has to exist as a value first).
        _cat_cols = [
            'type.primary', 'player.name', 'player.position', 'team.name',
            'opponentTeam.name', 'possession.team.name', 'team.formation',
            'groundDuel.duelType', 'infraction.type', 'shot.bodyPart',
        ]
        for _cc in _cat_cols:
            if _cc in raw_events_df.columns:
                raw_events_df[_cc] = raw_events_df[_cc].astype('category')
        # Numeric-string column ("14.118731", 661k uniques — not category
        # material). Parsed once here; the one read site
        # (_calculate_radars_from_events) handles both str and numeric.
        if 'possession.duration' in raw_events_df.columns:
            raw_events_df['possession.duration'] = pd.to_numeric(
                raw_events_df['possession.duration'].astype(str).str.replace('s', '', regex=False),
                errors='coerce').astype('float32')

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
def load_obv_viz_data():
    """Engine OBV/phases aggregates (exported by GPA engine rebuild).

    Returns {key: DataFrame or None}. Missing files (e.g. before the first
    engine rebuild ships them) simply disable the OBV visuals.
    """
    files = {
        'minute': 'obv_match_minute.parquet',
        'pairs': 'obv_pass_pairs.parquet',
        'players': 'obv_match_player.parquet',
        'team_season': 'obv_team_season.parquet',
        'phase_profile': 'team_phase_profile.parquet',
    }
    out = {}
    for key, fname in files.items():
        try:
            out[key] = pd.read_parquet(fname) if os.path.exists(fname) else None
        except Exception:
            out[key] = None
    return out


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
        # × career-NPV age multiplier, through the fee-calibrated
        # CVI→EUR curve. No reliability ramp: the projection is
        # already evidence-weighted internally.
        #
        # Camp EUR penalty (Lucas 2026-07-26): earlier versions skipped
        # the extra Camp discount here on the theory that projection_abs
        # already carries the recruit discount. The fee calibration pass
        # (14 real permanent sales) said otherwise: L3 sales realize at
        # a median 1.09× engine value (well centred) but Camp sales at
        # only 0.67×. Applying the existing CAMP_PROJECTED_EUR_PENALTY
        # (0.85) closes most of that gap without a new constant. Camp
        # membership derived from seasonId (no competitionId in the
        # engine parquet).
        _ROLE2CVI = {'Striker': 'ST', 'Wide Attacker': 'AM_WG',
                     'Advanced Midfielder': 'AM_WG', 'Deep Midfielder': 'CM',
                     'Wide Defender': 'FB', 'Central Defender': 'CB'}
        _CAMP_SEASON_IDS = {190230, 191779, 192925}
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
            try:
                _comp = (702 if int(r.get('seasonId')) in _CAMP_SEASON_IDS
                          else 43324)
            except (TypeError, ValueError):
                _comp = None
            v = cvi_to_projected_eur(perf * am, position_group=grp,
                                       competition_id=_comp)
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


# ============================================================================
# Player TENDENCIES + STYLES (models/roles/) — the descriptive stylistic layer.
# Tendencies = within-role percentiles of attempt composition (50 = role-
# typical); styles = tendency-derived archetypes with a Conventional centre.
# Both are DISPLAY-ONLY and never touch the rating/projection. GKs are excluded
# upstream (no role assignment), so keepers simply get nothing here.
# ============================================================================
@st.cache_data(ttl=86400)
def load_tendencies():
    """tendencies_season.parquet -> one row per (playerId, seasonId, role) with
    t_<key> raw values, p_<key> within-role percentiles, thin_sample flag.
    Empty DF if the file is missing (feature degrades to hidden)."""
    path = os.path.join(os.path.dirname(__file__), 'models', 'roles',
                        'tendencies_season.parquet')
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        df['playerId'] = pd.to_numeric(df['playerId'], errors='coerce').astype('Int64')
        df['seasonId'] = pd.to_numeric(df['seasonId'], errors='coerce').astype('Int64')
        if PLAYER_ID_ALIASES:
            df['playerId'] = df['playerId'].map(
                lambda p: PLAYER_ID_ALIASES.get(int(p), int(p))
                if p is not None and not pd.isna(p) else p).astype('Int64')
        return df
    except Exception:
        logger.exception("Failed to load tendencies_season.parquet")
        return pd.DataFrame()


@st.cache_data(ttl=86400)
def load_styles():
    """style_assignments_season.parquet -> style + style_fit + top-2 mix per
    (playerId, seasonId). Empty DF if missing."""
    path = os.path.join(os.path.dirname(__file__), 'models', 'roles',
                        'style_assignments_season.parquet')
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        df['playerId'] = pd.to_numeric(df['playerId'], errors='coerce').astype('Int64')
        df['seasonId'] = pd.to_numeric(df['seasonId'], errors='coerce').astype('Int64')
        if PLAYER_ID_ALIASES:
            df['playerId'] = df['playerId'].map(
                lambda p: PLAYER_ID_ALIASES.get(int(p), int(p))
                if p is not None and not pd.isna(p) else p).astype('Int64')
        return df
    except Exception:
        logger.exception("Failed to load style_assignments_season.parquet")
        return pd.DataFrame()


@st.cache_data(ttl=86400)
def get_career_engine_role_map():
    """playerId -> observed engine role, by Lucas's 2026-06-24 rule: the
    minutes-weighted argmax over per-season role SHARES across the player's
    ENTIRE engine history (both leagues) — mins_lineup x sh_<role> summed,
    then argmax. The single source of truth for "what role is he" wherever a
    player gets ONE role (role board buckets, analysis role filter/columns).
    Deliberately NOT scope-sliced: a player is bucketed by the role he has
    actually played most, immune to non-chronological seasonId ordering."""
    try:
        eng, _ = load_player_engine()
    except Exception:
        return {}
    if eng is None or eng.empty or 'playerId' not in eng.columns:
        return {}
    sh_cols = [c for c in eng.columns if c.startswith('sh_')]
    if not sh_cols:
        return {}
    gm = eng[['playerId'] + sh_cols].copy()
    w = pd.to_numeric(eng.get('mins_lineup'), errors='coerce')
    w = w.fillna(pd.to_numeric(eng.get('mins_played'), errors='coerce')).fillna(0.0)
    for c in sh_cols:
        gm[c] = pd.to_numeric(gm[c], errors='coerce').fillna(0.0) * w.values
    tot = gm.groupby('playerId')[sh_cols].sum()
    tot = tot[tot.sum(axis=1) > 0]
    return {int(p): r[3:] for p, r in tot.idxmax(axis=1).items()}


# The tendency metadata + role menus live in the builder so there is one source
# of truth. Import them; if the builder is unavailable (e.g. a code-only deploy
# that dropped models/roles), fall back to empty and the panel hides itself.
try:
    import sys as _sys
    _roles_dir = os.path.join(os.path.dirname(__file__), 'models', 'roles')
    if _roles_dir not in _sys.path:
        _sys.path.insert(0, _roles_dir)
    from build_tendencies import (TENDENCY_META as TENDENCY_META,
                                   ROLE_TENDENCY_MENU as ROLE_TENDENCY_MENU,
                                   ALL_TENDENCIES as ALL_TENDENCIES,
                                   poles as tendency_poles)
except Exception:
    logger.exception("tendency metadata import failed — tendencies panel off")
    TENDENCY_META, ROLE_TENDENCY_MENU, ALL_TENDENCIES = {}, {}, []

    def tendency_poles(role, key):
        m = TENDENCY_META.get(key, {})
        return m.get('pole_low', ''), m.get('pole_high', '')


def get_scoped_tendencies(player_id, season_ids):
    """Season-scope-aware tendency row for a player.

    season_ids None (All Seasons) -> minutes-weighted average of the player's
    season tendency percentiles (matching the ACP card's career aggregation);
    a single/list scope -> that scope's rows, minutes-weighted if more than one.
    Returns (Series-like dict of p_<key>/t_<key>, role, mins, thin_sample) or
    None when the player has no tendencies (GK, or below the 300' floor)."""
    tend = load_tendencies()
    if tend is None or tend.empty:
        return None
    rows = tend[tend['playerId'] == int(player_id)]
    if rows.empty:
        return None
    if season_ids is not None:
        sids = [int(s) for s in (season_ids if isinstance(season_ids, (list, tuple, set))
                                 else [season_ids])]
        rows = rows[rows['seasonId'].isin(sids)]
    if rows.empty:
        return None
    # role = the player's most-played role across the scoped rows
    w = pd.to_numeric(rows['mins_played'], errors='coerce').fillna(0.0)
    role = (rows.assign(_w=w).groupby('role')['_w'].sum().idxmax())
    rr = rows[rows['role'] == role]
    ww = pd.to_numeric(rr['mins_played'], errors='coerce').fillna(0.0).to_numpy()
    agg = {}
    for k in ALL_TENDENCIES:
        for pfx in ('p_', 't_'):
            col = f'{pfx}{k}'
            if col not in rr.columns:
                continue
            v = pd.to_numeric(rr[col], errors='coerce').to_numpy()
            m = ~np.isnan(v)
            agg[col] = (float(np.average(v[m], weights=ww[m]))
                        if m.any() and ww[m].sum() > 0 else np.nan)
    mins = float(ww.sum())
    return {'values': agg, 'role': role, 'mins': mins,
            'thin_sample': mins < 900}


def get_scoped_style(player_id, season_ids):
    """Season-scope-aware style label for a player: the style of his highest-
    minutes row in scope (styles are per-season; All Seasons -> most-played
    season's style). Returns dict or None."""
    styles = load_styles()
    if styles is None or styles.empty:
        return None
    rows = styles[styles['playerId'] == int(player_id)]
    if rows.empty:
        return None
    if season_ids is not None:
        sids = [int(s) for s in (season_ids if isinstance(season_ids, (list, tuple, set))
                                 else [season_ids])]
        rows = rows[rows['seasonId'].isin(sids)]
    if rows.empty:
        return None
    r = rows.sort_values('mins_played').iloc[-1]
    return {'style': r.get('style'), 'style_fit': r.get('style_fit'),
            'style_2': r.get('style_2'), 'style_2_fit': r.get('style_2_fit'),
            'role': r.get('role'), 'thin_sample': bool(r.get('thin_sample', False))}


def render_tendencies_panel(player_id, season_ids, st_container=None):
    """futi-style bipolar tendency sliders for the player's role menu.

    Each slider is the within-role percentile (50 = role-typical). The dominant
    side is coloured, the other greyed. Low-confidence pairs (near/far post) are
    tucked into an expander. Renders nothing for players without tendencies."""
    tgt = st_container if st_container is not None else st
    scoped = get_scoped_tendencies(player_id, season_ids)
    if scoped is None:
        return False
    role = scoped['role']
    menu = ROLE_TENDENCY_MENU.get(role, [])
    if not menu:
        return False
    vals = scoped['values']

    def _slider(key):
        p = vals.get(f'p_{key}')
        if p is None or pd.isna(p):
            return
        lo, hi = tendency_poles(role, key)
        p = float(p)
        # colour the dominant side, grey the other; 50 is role-typical. Use
        # HTML (not markdown **) for the bold — markdown isn't parsed inside a
        # raw-HTML block, so ** would render as literal asterisks.
        dom_hi = p >= 50
        pct_hi = p
        _dom = "font-weight:700;color:#1a1a1a"
        _sub = "color:#8a8a8a"
        # hover tooltip = what this pair actually measures (eye-test aid)
        _desc = str(TENDENCY_META.get(key, {}).get('desc', '')).replace("'", "&#39;")
        left_lbl = (f"<span title='{_desc}' "
                    f"style='{_sub if dom_hi else _dom}'>{lo}</span>")
        right_lbl = (f"<span title='{_desc}' "
                     f"style='{_dom if dom_hi else _sub}'>{hi}</span>")
        c1, c2, c3 = tgt.columns([2.2, 5, 2.2])
        c1.markdown(f"<div style='text-align:right'>{left_lbl}</div>",
                    unsafe_allow_html=True)
        # diverging bar centred at 50
        fill = int(round(pct_hi))
        bar = (f"<div style='background:#e9ecef;border-radius:5px;height:12px;"
               f"width:100%;position:relative'>"
               f"<div style='position:absolute;left:50%;top:0;height:12px;"
               f"width:1px;background:#adb5bd'></div>"
               f"<div style='background:{'#2f6feb' if dom_hi else '#e8590c'};"
               f"height:12px;border-radius:5px;"
               f"{'left:50%;width:'+str(max(fill-50,0))+'%' if dom_hi else 'right:50%;width:'+str(max(50-fill,0))+'%'};"
               f"position:absolute;top:0'></div></div>")
        c2.markdown(bar, unsafe_allow_html=True)
        c3.markdown(f"{right_lbl} <span style='color:#adb5bd;font-size:0.8em'>"
                    f"{p:.0f}</span>", unsafe_allow_html=True)

    hi_conf = [k for k in menu if TENDENCY_META.get(k, {}).get('confidence') != 'low']
    lo_conf = [k for k in menu if TENDENCY_META.get(k, {}).get('confidence') == 'low']
    for k in hi_conf:
        _slider(k)
    if scoped['thin_sample']:
        tgt.caption(f"⚠️ thin sample ({int(scoped['mins'])}′) — scored against "
                    f"the {role} cohort but read with caution.")
    if lo_conf:
        with tgt.expander("Low-confidence tendencies (thin event support)"):
            st.caption("Shot-side pairs rest on few events; futi hides these — "
                       "shown here for completeness, not for judgement.")
            for k in lo_conf:
                _slider(k)
    return True


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

**Roles, Tendencies & Styles — *what kind* of player, not how good** *(descriptive, never in the rating)*

Separate from the quality question, we describe *how* a player plays:
- **Role** — where he actually operates, learned from where his events happen
  match by match (a Striker, Wide Defender, Deep Midfielder…), not the
  lineup-card position. A season role is a blend of his per-match roles.
- **Tendencies** — within his role, which way he leans on the things he
  *attempts*: carry vs pass, cross vs combine, come short vs run behind, and so
  on. Each is a **within-role percentile** (50 = typical for the role, so a
  full bar means "does far more of this than his peers"). It's **attempt
  composition, not quality** — a high bar is never "better", just "more of
  that".
- **Style** — a single archetype read off the strongest tendencies (Wide
  Arriver, Deep-Lying Playmaker, Ball-Playing Defender…). Players without a
  pronounced lean — or with two leans too close to call — are
  **"Conventional"** for their role: a real centre, not a gap. To keep labels
  from flipping on a few noisy months, the style reads a recency-weighted view
  of the player's tendencies (this season blended with his last season in the
  same role); the tendency bars themselves always show the selected season
  as-is. Styles describe; they **never enter the rating or projection**.
  Goalkeepers are outside this system for now.
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

@st.cache_resource(ttl=86400, show_spinner=False, max_entries=5)
def _get_filtered_events_cached(_events_df, season_key, comp_key):
    """Cache wrapper keyed on the hashable season/comp tuple. Uses
    cache_RESOURCE (returns the SAME object, no copy) — cache_data was
    deserializing a full copy on EVERY rerun (the dominant per-interaction
    cost). Downstream consumers only read/slice/.copy() the frame (never
    mutate in place), so sharing one read-only instance across reruns is
    safe.

    max_entries (LRU) is load-bearing: unbounded, the warm Space sat at
    ~31.5 GB on 32 GB hardware and segfaulted (exit 139) on any allocation
    spike (2026-07-14, when the master frame was still ~10.5 GB deep).
    After the load_data memory slim (category dtypes + dead-column drop,
    same date) the arithmetic is: master ~3.5 GB + worst-case scope copy
    1.9 GB deep (Liga 3 all-seasons; single seasons ≤0.9 GB) →
    3.5 + 5 × 1.9 + ~4 GB app/runtime overhead ≈ 17 GB, well under 32 GB
    even with every slot holding the biggest scope. Hence max_entries=5
    (was 3 pre-slim). A scope miss re-filters in a few seconds; the disk
    caches keep everything else fast. Keep the prewarm list (in
    _prewarm_scope_caches) no larger than this bound or the warm loop just
    churns the LRU."""
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


# Canonical minutes schema — empty results must still carry these columns or
# downstream stats code KeyErrors on 'totalMinutes' (seen at the 2026/27 season
# start, when the current season has matches but no minutes data yet).
_EMPTY_MINUTES_COLS = ['playerId', 'playerName', 'teamName', 'primaryPosition', 'totalMinutes']


def _empty_minutes_df():
    return pd.DataFrame(columns=_EMPTY_MINUTES_COLS)


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
            return _empty_minutes_df()
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
            return _empty_minutes_df()
        combined = pd.concat(dfs)
        return combined.groupby('playerId').agg({
            'playerName': 'first',
            'teamName': 'first',
            'primaryPosition': 'first',
            'totalMinutes': 'sum'
        }).reset_index()
    result = player_minutes_data.get(season_id)
    if not isinstance(result, pd.DataFrame) or result.empty:
        return _empty_minutes_df()
    return result

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

# ------------------------------------------------------------------------------
# Club-first defaults and cross-page selector persistence
# ------------------------------------------------------------------------------
OUR_TEAM = 'Atlético CP'
# League / season are chosen ONCE in the sidebar context bar (context_bar.py);
# every page resolves its scope from these two session keys.
GLOBAL_LEAGUE_KEY = 'ctx_league'          # = context_bar.LEAGUE_KEY (the bar's widget key)
GLOBAL_SEASON_KEY = 'ctx_season_memory'   # = context_bar.SEASON_MEMORY (last chosen season label)
# A season needs this many matches WITH EVENTS before it is the default —
# the newest season in the fixture list has none for its first weeks.
MIN_MATCHES_FOR_DEFAULT_SEASON = 5
# {season_id: n matches with events}; filled in the main body after load_data.
SEASON_MATCHES_WITH_EVENTS = {}


@st.cache_data(ttl=86400, show_spinner=False)
def _season_match_counts_cached(n_events, _events_df):
    counts = _events_df.groupby('seasonId')['matchId'].nunique()
    return {int(k): int(v) for k, v in counts.items()}


def _season_match_counts(events_df):
    """{season_id: matches with events} for the loaded events, or {} if unknown."""
    if events_df is None or events_df.empty or 'seasonId' not in events_df.columns:
        return {}
    try:
        return _season_match_counts_cached(len(events_df), events_df)
    except Exception as e:
        logger.warning(f"[seasons] could not count matches with events: {e}")
        return {}


def _default_season_label(available_seasons, options):
    """Newest season label (options are newest-first) with event data for at
    least MIN_MATCHES_FOR_DEFAULT_SEASON matches; else the newest label."""
    for label in options:
        if label == "All Seasons":
            continue
        sids = [sid for sid, lab in available_seasons.items() if lab == label]
        if any(SEASON_MATCHES_WITH_EVENTS.get(int(sid), 0) >= MIN_MATCHES_FOR_DEFAULT_SEASON
               for sid in sids):
            return label
    return next((o for o in options if o != "All Seasons"), options[0] if options else None)


def _team_fixtures(matches_df, season_id, team):
    season_matches = matches_df[matches_df['seasonId'] == season_id]
    return season_matches[
        (season_matches['homeTeamName'] == team) | (season_matches['awayTeamName'] == team)
    ].copy()


def _fixture_record(row, team, team_matches):
    opponent = row['awayTeamName'] if row['homeTeamName'] == team else row['homeTeamName']
    home_away = 'Home' if row['homeTeamName'] == team else 'Away'
    gw = row.get('gameweek', '?')
    # Gameweek missing: derive it from the team's position in its own fixture list
    if pd.isna(gw) or str(gw).strip() in ('', '?', 'nan', 'None'):
        ordered = team_matches.assign(
            _d=pd.to_datetime(team_matches['dateutc'], errors='coerce')).sort_values('_d')
        try:
            gw = ordered['matchId'].tolist().index(row.get('matchId')) + 1
        except ValueError:
            gw = '?'
    return {
        'opponent': opponent,
        'date': pd.to_datetime(row.get('dateutc'), errors='coerce'),
        'gameweek': gw,
        'home_away': home_away,
        'matchId': row.get('matchId'),
    }


def _is_played(score):
    return pd.notna(score) and '-' in str(score)


def next_fixture_for_team(matches_df, season_id, team=None):
    """Next unplayed fixture for `team` (default OUR_TEAM) in `season_id`, or None.
    Returns {'opponent', 'date', 'gameweek', 'home_away', 'matchId'}."""
    team = team or OUR_TEAM
    team_matches = _team_fixtures(matches_df, season_id, team)
    if team_matches.empty:
        return None
    unplayed = team_matches[~team_matches['score'].apply(_is_played)].copy()
    if unplayed.empty:
        return None
    unplayed['_d'] = pd.to_datetime(unplayed['dateutc'], errors='coerce')
    return _fixture_record(unplayed.sort_values('_d').iloc[0], team, team_matches)


def last_fixture_for_team(matches_df, season_id, team=None):
    """Most recently PLAYED fixture for `team` in `season_id`, or None."""
    team = team or OUR_TEAM
    team_matches = _team_fixtures(matches_df, season_id, team)
    if team_matches.empty:
        return None
    played = team_matches[team_matches['score'].apply(_is_played)].copy()
    if played.empty:
        return None
    played['_d'] = pd.to_datetime(played['dateutc'], errors='coerce')
    return _fixture_record(played.sort_values('_d').iloc[-1], team, team_matches)


def _predictor_default_labels(all_season_options, season_ids_desc, matches_df):
    """(home_label, away_label) defaults for the Match Predictor selectors.
    Home: OUR_TEAM in the newest season it is available in. Away: that season's
    next fixture opponent, else its most recent opponent, else None."""
    for sid in season_ids_desc:
        season_label = SEASON_ID_MAP.get(sid)
        home = f"{OUR_TEAM} ({season_label})"
        if home not in all_season_options:
            continue
        fixture = (next_fixture_for_team(matches_df, sid, OUR_TEAM)
                   or last_fixture_for_team(matches_df, sid, OUR_TEAM))
        away = f"{fixture['opponent']} ({season_label})" if fixture else None
        return home, (away if away in all_season_options else None)
    return None, None


def league_selector(section_key):
    """Competition ids for the league in the global context bar (context_bar.py).
    `section_key` is kept for the call sites; the bar is drawn once in the
    sidebar by app.py and every page reads the same choice."""
    return context_bar.current_comp_ids()


def season_selector(section_key, include_all_seasons=False, comp_ids=None):
    """Season id for this page from the global context bar: None for
    'All Seasons' (only when the page accepts it), else the chosen season,
    else the newest season with event data (see _default_season_label)."""
    return context_bar.current_season_id(comp_ids if comp_ids is not None else context_bar.current_comp_ids(),
                                         include_all_seasons)


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
def _next_shot_id_by_match(events_df):
    """Series aligned to events_df.index: the id of the next shot in the same
    match (a backfill of 'shot_event_id'). Safe on an EMPTY frame — there the
    grouped bfill returns a Series that pandas does NOT turn into a column on
    assignment, which surfaced as KeyError 'next_shot_id' on every player page
    for a season whose events hadn't been ingested yet."""
    if events_df.empty:
        return pd.Series(index=events_df.index, dtype='float64')
    return events_df.groupby('matchId')['shot_event_id'].bfill().reindex(events_df.index)


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
        events_df['next_shot_id'] = _next_shot_id_by_match(events_df)
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

# --- CVI / projected-value model: extracted to models/value/cvi.py (2026-09) ---
# Every public and _private name the pages use is re-exported via __all__ so
# the ~12 call sites below are unchanged. Keep model logic THERE, not here.
from models.value.cvi import *  # noqa: F401,F403


def _role_key_stats(stats_row, template_role, cap=12):
    """Return [(metric, 'value p90', pct_0_100 or None), ...] for the metrics
    that matter most to `template_role`, ordered by the role's own weights.

    WEIGHTS[role] maps metric -> weight, so 'most important' is not an
    arbitrary pick — it is the role model's own emphasis. Percentiles come
    from the {metric}_percentile columns (0-1, inversion already applied);
    scaled to 0-100 here for the colour wash.
    """
    weights = WEIGHTS.get(template_role, {})
    if not weights or stats_row is None:
        return []
    ordered = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for metric, _w in ordered:
        if metric in RADAR_HIDDEN_METRICS or metric not in stats_row.index:
            continue
        raw = stats_row.get(metric)
        if raw is None or pd.isna(raw):
            continue
        pct = stats_row.get(f'{metric}_percentile')
        pct100 = None if pct is None or pd.isna(pct) else float(pct) * 100.0
        out.append((metric, f'{fmt_val(metric, float(raw))}', pct100))
        if len(out) >= cap:
            break
    return out


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

# Display-name overrides for radar / distribution axis labels. The underlying
# data column keeps its name; only the printed label changes. 'Progressive
# Passes' is counted from accurate passes only (pass.accurate == True), so
# label it as successful.
METRIC_DISPLAY_NAMES = {'Progressive Passes': 'Progressive passes successful'}

def _metric_display(metric):
    return METRIC_DISPLAY_NAMES.get(metric, metric)

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
SHADOW_TAG_CATEGORIES = theme.SHADOW_TAG_COLORS

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
    # team.formation is categorical: value_counts lists every category, so
    # drop zero-count rows or an absent formation could win the fallback.
    formation_counts = formation_counts[formation_counts > 0]
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
    if _raw_events_df is None or _raw_events_df.empty:
        # No events for this scope (a season whose fixtures exist but whose
        # events haven't been ingested yet). Every step below assumes rows —
        # the xA backfill was the first to fail — so return the empty frame
        # the callers already handle ("Player data not available ...").
        logger.warning("[player-stats] no events for season_id=%s — returning empty stats", season_id)
        return pd.DataFrame()
    # Disk cache: load pre-computed results if available
    _REQUIRED_STAT_COLS = {'Throw-ins', 'Avg max throw-in distance', 'Throw-ins into box', 'Avg max throw-in into box distance', 'Avg max throw-in into box aerial distance', 'Defensive Area', 'Opp xT into Def Area', 'Opp Pass Success % into Def Area', 'Opp xT from Def Area', 'Territorial Dominance', 'Opp xT into Def Area OE', 'Opp xT from Def Area OE', 'Territorial Dominance OE', 'xTOP', 'xTSP'}
    _scope_key = _stats_scope_key(season_id, _raw_events_df)
    cache_path = os.path.join(STATS_CACHE_DIR, f'player_stats_{STATS_CACHE_VERSION}_{_scope_key}.parquet')
    _data_fp = {'n_matches': int(_raw_events_df['matchId'].nunique()),
                'n_events': int(len(_raw_events_df))}
    if os.path.exists(cache_path) and _cache_is_stale(cache_path, _data_fp):
        print(f"Player-stats cache for scope {_scope_key} predates current data — recomputing")
        os.remove(cache_path)
    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        # An empty cache must not be served (see percentiles cache) — recompute.
        if not cached.empty and _REQUIRED_STAT_COLS.issubset(cached.columns):
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
            # (player.position is categorical — drop the zero-count categories
            # value_counts now reports, else all-NaN players pick a bogus one)
            _pos_counts = _grp['player.position'].dropna().value_counts()
            _pos_counts = _pos_counts[_pos_counts > 0]
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
    events_df['next_shot_id'] = _next_shot_id_by_match(events_df)
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
        _write_cache_meta(cache_path, _data_fp)
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
    _REQUIRED_PCT_COLS = {'Throw-ins', 'Avg max throw-in distance', 'Throw-ins into box', 'Avg max throw-in into box distance', 'Avg max throw-in into box aerial distance', 'Defensive Area', 'Opp xT into Def Area', 'Opp Pass Success % into Def Area', 'Opp xT from Def Area', 'Territorial Dominance', 'Opp xT into Def Area OE', 'Opp xT from Def Area OE', 'Territorial Dominance OE', 'xTOP', 'xTSP', 'Touches in penalty area_percentile'}
    _scope_key = _stats_scope_key(season_id, _player_data_df)
    cache_path = os.path.join(STATS_CACHE_DIR, f'player_percentiles_{STATS_CACHE_VERSION}_{_scope_key}.parquet')
    _pct_fp = {'n_rows': int(len(_player_data_df)),
               'minutes_sum': int(pd.to_numeric(
                   _player_data_df.get('totalMinutes', pd.Series(dtype=float)),
                   errors='coerce').fillna(0).sum()),
               # Role scores are cached VALUES derived from the weights, so a
               # weight tweak in config.yaml must invalidate this cache too.
               'weights_hash': hashlib.md5(json.dumps(
                   _weights, sort_keys=True, default=str).encode()).hexdigest()}
    if os.path.exists(cache_path) and _cache_is_stale(cache_path, _pct_fp):
        print(f"Percentiles cache for scope {_scope_key} predates current data — recomputing")
        os.remove(cache_path)
    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        # An empty cache (written early-season when nobody met the minutes
        # floor) must not be served — treat as invalid and recompute.
        if not cached.empty and _REQUIRED_PCT_COLS.issubset(cached.columns):
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
    # Early-season clamp: after 1-2 matchweeks nobody can reach the mid-season
    # 500' floor, the qualifying population is empty, and the whole scores
    # frame came back blank (2026/27 matchweek 1). Only when NO player meets
    # the floor, drop it to half the current max so scores exist from day one;
    # mid-season this never triggers and the floor stays as passed.
    _max_min = data['totalMinutes'].max()
    if pd.notna(_max_min) and _max_min < min_minutes:
        min_minutes = max(1, int(_max_min * 0.5))
        print(f"Early-season minutes floor: no player at requested floor; using {min_minutes}'")
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
        _write_cache_meta(cache_path, _pct_fp)
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
    if player_stats_df.empty:
        # No events for this scope: nothing to merge or rank. Callers show
        # their "Player data not available" state on an empty frame.
        return player_stats_df, pd.DataFrame()
    player_stats_df = merge_gpa_values_into_stats(player_stats_df, active_season_ids, comp_ids)
    player_stats_df = merge_defr_values_into_stats(player_stats_df, active_season_ids, comp_ids)
    player_stats_df = merge_engine_values_into_stats(player_stats_df, active_season_ids, comp_ids)
    player_stats_with_scores_df = calculate_player_percentiles_and_scores(
        player_stats_df, POSITION_GROUPS, WEIGHTS, INVERT_METRICS, min_minutes=500, season_id=season_id, cache_scope=_scope
    )
    return player_stats_df, player_stats_with_scores_df


def _log_rss(tag: str) -> None:
    """One grep-able container-log line: 'RSS x.x GB — <tag>'."""
    try:
        import psutil
        _rss = psutil.Process().memory_info().rss / (1024 ** 3)
        logger.info(f"RSS {_rss:.1f} GB — {tag}")
    except Exception:
        pass  # psutil missing/failing must never take the app down


@st.cache_resource(show_spinner=False)
def _start_rss_telemetry():
    """Log RSS at boot and every ~5 min from a daemon thread.

    cache_resource makes this a once-per-process singleton (app.py re-executes
    on every rerun; without the guard each rerun would spawn a new thread).
    Keeps the next memory incident diagnosable from container logs alone.
    """
    import threading, time as _time

    _log_rss('boot')

    def _worker():
        while True:
            _time.sleep(300)
            _log_rss('periodic')

    _t = threading.Thread(target=_worker, name='acp-rss-telemetry', daemon=True)
    _t.start()
    return True


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
        # Warm ONLY the landing scopes: per league, the season the context
        # bar defaults to — the newest one with event data for at least
        # MIN_MATCHES_FOR_DEFAULT_SEASON matches (_default_season_label).
        # It used to warm `current_season`, which early in a season is
        # fixtures-only (no events yet): the loop found empty frames and
        # warmed NOTHING, so every first visit to a team page was cold
        # (2026-09). The filtered-events cache is LRU-bounded at
        # max_entries=5 (see _get_filtered_events_cached for the memory
        # arithmetic) — keep this warm list within that bound or the loop
        # just churns the LRU. Other scopes lazy-build in a few seconds on
        # first visit (disk caches cover the expensive layers).
        def _landing_season(_cfg):
            _by_newest = sorted(_cfg.get('seasons', {}).items(),
                                key=lambda kv: str(kv[1]), reverse=True)
            for _sid, _label in _by_newest:
                if SEASON_MATCHES_WITH_EVENTS.get(int(_sid), 0) >= MIN_MATCHES_FOR_DEFAULT_SEASON:
                    return _sid
            return _cfg.get('current_season')
        _WARM_SCOPES = [(_cid, _landing_season(_cfg)) for _cid, _cfg in COMPETITIONS.items()]
        _WARM_SCOPES = [(_c, _s) for _c, _s in _WARM_SCOPES if _s is not None]
        logger.info(f"[prewarm] landing scopes: {_WARM_SCOPES}")
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
                        # Season-report metrics: the single biggest cold cost
                        # of BOTH team pages (~6 s), computed per scope for
                        # every team — one warm serves any team either page
                        # opens. Args mirror views/team_analysis.py exactly
                        # (stage 'all', single-season scope) so this is a
                        # direct key hit, not an approximation.
                        try:
                            _m = filter_by_league(get_season_matches(_matches_summary_df, _sid), [_cid])
                            compute_team_season_metrics(
                                _ev, _m, season_ids=(int(_sid),), use_wyscout=True,
                                cache_key=season_report_cache_key([_cid], _sid))
                        except Exception as _e:
                            logger.warning(f"[prewarm] season report (comp={_cid}, season={_sid}) failed: {_e}")
                    _n += 1
                except Exception as _e:
                    logger.warning(f"[prewarm] scope (comp={_cid}, season={_sid}) failed: {_e}")
        logger.info(f"[prewarm] warmed {_n} scope(s) in {_time.time()-_t0:.0f}s")
        _log_rss('post-prewarm')

    _t = threading.Thread(target=_worker, name="acp-prewarm", daemon=True)
    _t.start()
    logger.info("[prewarm] background cache warm started")
    return True


# mpl_locked sits INSIDE cache_data so a cache hit never touches the lock;
# only an actual build serialises. Self-contained: savefig + close before it
# returns PNG bytes, so no live figure escapes the critical section.
@st.cache_data(ttl=86400, show_spinner=False, max_entries=64)
@mpl_locked
def _render_acp_index_card_png(player_id, season_scope_key, stats_cache_ver,
                               engine_ver, career_view,
                               _eng_df, _e, _eng_meta):
    """ACP Index radar + cohort-KDE card, rendered once to PNG bytes.

    This always-on matplotlib figure used to rebuild on every rerun (~2-3 s
    per profile section toggle). Cached on (playerId, season scope,
    STATS_CACHE_VERSION, engine rating_version, career_view) — the
    underscore-prefixed frame/row/meta args are NOT hashed, they are the
    render inputs. Returns PNG bytes, or None when fewer than 5 radar axes
    have data (the card is skipped, matching the old inline behavior).

    Radar: RAW per-90 values on a mean ± 2σ scale vs the player's
    league × season × role cohort, with per-axis cohort distributions on the
    right — mirrors the traditional radar's raw mode + KDE panels.
    Set piece removed (Lucas); Duel-att raw = n-weighted take-on/shield
    Glicko; Def Quality raw = DWAE/90.
    """
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
    if career_view:
        # cohort pooled across ALL seasons of this league+role;
        # player point = minutes-weighted career per-90
        _coh = _eng_rad_df[
            (_eng_rad_df['league'] == _e['league'])
            & (_eng_rad_df['role'] == _e['role'])
            & (_eng_rad_df['mins_played'] >= 500)]
        _pr = _eng_rad_df[_eng_rad_df['playerId'] == int(player_id)]
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
    if len(_rad) < 5:
        return None
    from math import pi as _pi
    _n = len(_rad)
    _ang = [k / float(_n) * 2 * _pi for k in range(_n)]
    _vals = [m for _, m, _g, _pv, _mu, _sd, _f, _pop in _rad]
    _figr = plt.figure(figsize=(20, 10))
    try:
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
        _buf = io.BytesIO()
        _figr.savefig(_buf, format='png', dpi=110,
                      facecolor=_figr.get_facecolor())
        return _buf.getvalue()
    finally:
        plt.close(_figr)


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
        ax.text(angle_rad, 115, _metric_display(metric), size=8, ha='center', va='center', rotation=0, color=color, fontweight='bold')

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
            ax_dist.text(-0.05, 0.5, _metric_display(metric), transform=ax_dist.transAxes, fontsize=9, fontweight='bold', va='center', ha='right')

    return fig


# Locked for the WHOLE export, not per player: the loop calls plt.close('all')
# after each render (see the anti-OOM note inside), which would free other
# sessions' in-flight figures if the lock were dropped between iterations.
# Cost: a bulk export of dozens of players stalls other sessions' charts for
# its duration. Accepted -- it is a deliberate, infrequent admin action, and it
# already saturates the GIL while it runs.
@mpl_locked
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

    # possession.duration arrives as float32 since the load_data memory slim;
    # keep a str-path for stray frames that still carry the raw "12.3s" form.
    _pd_dur = season_events_df.get('possession.duration', pd.Series(dtype='float'))
    if _pd_dur.dtype == object:
        _pd_dur = _pd_dur.str.replace('s', '', regex=False)
    season_events_df['possession.duration_sec'] = pd.to_numeric(_pd_dur, errors='coerce')
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
def _scope_label(league=None, season=None, sep=" "):
    """'Liga 3 2025/26', 'Liga 3', '2025/26' or '' — only what the caller
    actually passed, so a figure never claims a scope it wasn't given (the
    old defaults stamped 'Liga 3, 2025/26' on Campeonato and 2026/27 charts)."""
    return sep.join(str(p) for p in (league, season) if p)


def plot_radar_chart(params, values_raw, values_pct, team_name, title_suffix, color, league=None, season=None):
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
    report_date = datetime.date.today().strftime("%Y-%m-%d"); _scope = _scope_label(league, season); full_title = f"{team_name}\n{title_suffix}{(' | ' + _scope) if _scope else ''} (As of: {report_date})"; ax.set_title(full_title, size=18, weight='bold', pad=40)
    return fig

# --- Corner Analysis Plotting Function (Unchanged) ---
def plot_corner_analysis(season_events_df, team_to_analyze, side, league=None, season=None):
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
    _scope = _scope_label(league, season); ax_pitch.set_title(f"Corners from the {side.capitalize()} Side{(' | ' + _scope) if _scope else ''}", fontsize=14); legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Short'), Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Near Post'), Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Middle'), Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Far Post'), Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=10, label='Other/Outside PA')]; ax_pitch.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.01, 0.01), frameon=False, fontsize=10)
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

    XG_MAX = theme.XG_MAX; colors = theme.XG_COLORS; nodes = theme.XG_NODES; cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))

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

    XG_MAX = theme.XG_MAX; colors = theme.XG_COLORS; nodes = theme.XG_NODES; cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))

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

    XG_MAX = theme.XG_MAX; colors = theme.XG_COLORS; nodes = theme.XG_NODES; cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))

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
    colors = theme.XG_COLORS
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
    # Phase-2 opponent adjustment (dormant during the first phase): once a
    # second stage exists, unbalanced schedules bias raw totals — overwrite
    # the aggregate goal/xG totals with SOS-adjusted ones (see schedule_adjust)
    try:
        from schedule_adjust import phase2_adjusted_totals
        _rows = []
        for _, _m in season_matches.iterrows():
            _sc = str(_m.get('score', ''))
            if '-' not in _sc:
                continue
            try:
                _hg, _ag = map(int, _sc.split('-'))
            except Exception:
                continue
            _rows.append({'matchId': _m['matchId'], 'roundId': _m.get('roundId'),
                          'home': _m['homeTeamName'], 'away': _m['awayTeamName'],
                          'hg': _hg, 'ag': _ag})
        _sh = season_events[(season_events['type.primary'] == 'shot')].dropna(
            subset=['shot.xg', 'team.name'])
        _xg = {(int(m), t): float(v) for (m, t), v in
               _sh.groupby(['matchId', 'team.name'])['shot.xg'].sum().items()}
        _adj = phase2_adjusted_totals(_rows, _xg)
        if _adj:
            for _t, _vals in _adj.items():
                if _t in team_stats:
                    team_stats[_t].update(_vals)
    except Exception as _e:
        print(f"phase-2 adjustment skipped: {_e}")

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
def plot_team_strength(stats_df, teams_to_include=None, league=None, season=None, icon_zoom=0.25): # <-- ADDED icon_zoom
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
    _scope = _scope_label(league, season, sep=', ')
    ax.set_title(f'Team Strength Scatterplot{(" | " + _scope) if _scope else ""} (As of: {report_date})', fontsize=18, weight='bold')
    ax.set_xlabel('Attacking Strength (30% NP Goals, 70% NPxG)', fontsize=12)
    ax.set_ylabel('Defending Strength (30% NP Goals Against, 70% NPxG Against)', fontsize=12)
    # The grid and tight_layout that used to share this line stay disabled: the
    # whole line was commented out, so neither has ever run against these
    # charts, and enabling them now would change every one of them. Only the
    # return is restored -- without it the callers' st.pyplot(None) silently
    # rendered pyplot's global current figure instead.
    return fig

# app.py (Add this new function)

# --- NEW FUNCTION: Plot Custom Scatter Plot ---
def plot_custom_scatter(stats_df, x_metric, y_metric, invert_x=False, invert_y=False, league=None, season=None):
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
    _scope = _scope_label(league, season, sep=', ')
    ax.set_title(f'League Scatterplot{(" | " + _scope) if _scope else ""} (As of: {report_date})', fontsize=18, weight='bold')
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
def calculate_xg_history_data(_raw_events_df, _matches_summary_df, scope_key=None):
    """
    Aggregates xG For and Against for every team for every match.
    (Previously named calculate_rolling_xg_data)

    scope_key is the cache key, and it is the ONLY one: both frames are
    underscore-prefixed, so Streamlit ignores them when hashing. Without it
    this function has no key components at all — it computes once per process
    and returns that first frame for every season/competition/stage after.

    Pass (season_key, comp_key, stage_key) — the same triple that keys
    _render_team_figure_png — because the caller hands us the SCOPED frames
    (get_filtered_events + filter_by_stage), so the scope has to be in the
    key for the result to follow it.
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
def calculate_set_piece_metrics(_events_df, season_id=None, stage=None):
    """Calculate set piece xG and goals metrics for all teams.

    season_id and stage are cache keys only — they never filter anything;
    _events_df arrives already scoped by the caller. Both must be named here:
    _events_df is unhashed (leading underscore), so a scope the key omits is
    invisible to Streamlit AND to the parquet below, and a stage-filtered call
    gets served the All-Stages numbers. Pass stage whenever the caller narrowed
    _events_df with filter_by_stage()."""
    import re
    _SP_CACHE_VERSION = 'v5'
    cache_path = None
    if season_id is not None:
        _stage_tag = ''
        if stage is not None:
            _stage_tag = '_' + (re.sub(r'[^\w-]+', '_', str(stage)).strip('_') or 'stage')
        cache_path = os.path.join(
            STATS_CACHE_DIR,
            f'set_piece_metrics_{_SP_CACHE_VERSION}_{season_id}{_stage_tag}.parquet')
        if os.path.exists(cache_path):
            print(f"Loading cached set piece metrics for season {season_id}{_stage_tag}")
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
    if cache_path is not None:
        os.makedirs(STATS_CACHE_DIR, exist_ok=True)
        try:
            result_df.to_parquet(cache_path)
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
    _tags = TagIndex(sec)  # one explode; the per-row lambda cost ~5.5 s here
    def _tag(name):
        return _tags.has(name)
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
                customdata=[[teams[j]] for j in other_idx],  # click -> that team
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
                customdata=[[team_name]],
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
        clickmode='event+select',
    )
    return fig


# ==============================================================================
# 6Y. INTERACTIVE TEAM VISUALS — shared plumbing for Team Analysis / Opposition
# ==============================================================================
@st.cache_data(ttl=86400, show_spinner=False)
def cached_passing_network(season_key, comp_key, stage_key, team_name, fig_ver,
                           _events_df, _obv_pairs):
    """pitch_visualizations.compute_passing_network for one scope, cached on
    the same key triple the PNG renderers use (the frames are unhashed)."""
    return pv.compute_passing_network(_events_df, team_name, obv_pairs=_obv_pairs)


def open_match_from_selection(event, season_label=None):
    """A click on a shot / rolling-xG point opens that match in Match Analysis
    (customdata[-2] on shots, [-1] on rolling points carries the matchId)."""
    import team_interactive
    mid = team_interactive.match_id_from_rows(team_interactive.selected_customdata(event))
    if mid is not None:
        navigation.go_to('Match Analysis', nav_match_id=mid)


def open_profile_from_selection(event, season_id):
    """A click on a passing-network node opens that player's profile via the
    same bridge the roster tables use (customdata = [playerId, name])."""
    import team_interactive
    pid = team_interactive.player_id_from_rows(team_interactive.selected_customdata(event))
    if pid is not None:
        st.session_state.selected_player_id = pid
        st.session_state.nav_to_profile = True
        st.session_state.nav_season_id = season_id
        st.session_state.nav_has_season = True
        st.rerun()


_PLOTLY_CFG = {'displayModeBar': False, 'responsive': True}


# ==============================================================================
# 6Z. CACHED FIGURE RENDERERS (Match Analysis / Team Analysis)
# ==============================================================================
# Both pages rebuilt ~10 matplotlib figures from scratch on EVERY rerun —
# nothing was cached, so a page switch, a selectbox change, even an expander
# toggle paid the full cost. Measured 2026-07-15 on a dev Mac: Team Analysis
# 7.1 s, Match Analysis 4.2-4.8 s per rerun, against 0.2-0.9 s for Player
# Profile. The Space runs ~3x slower per core, which is the 10-20 s page
# switch users actually saw. Each figure is ~0.4 s: ~0.1 s to build it and
# ~0.3 s for st.pyplot to serialise it to PNG.
#
# Fix is the one proven on the ACP Index card in f1c5a6f: build the figure
# once, cache the PNG BYTES, and st.image() them. A cache hit skips both the
# build and the serialise.
#
# CACHE-KEY CONTRACT — read this before adding a `kind`:
#   * The hashed args must name EVERY input that changes the picture. A key
#     that's missing a component doesn't render slowly, it renders the WRONG
#     TEAM'S MAP under the right heading. That is the only way this change can
#     hurt, so the keys are documented per-renderer below.
#   * Underscore-prefixed args are NOT hashed (Streamlit convention, same as
#     the stat caches in this file). They are the render inputs; the plain
#     scalars alongside them are the actual key and must pin them down
#     completely.
#   * FIGURE_CACHE_VERSION is in every key so a drawing-code change (colour,
#     label, marker size) invalidates the cache — the keys describe the DATA,
#     so nothing else would notice.


def _fig_png_bytes(fig):
    """Serialise a matplotlib figure to PNG bytes, then close it.

    dpi=200 + bbox_inches='tight' are exactly what st.pyplot hands to savefig
    (streamlit/elements/pyplot.py: `options = {"bbox_inches": "tight",
    "dpi": 200, "format": "png"}`), so st.image(_fig_png_bytes(fig)) is
    pixel-identical to the st.pyplot(fig) it replaces. facecolor is
    deliberately not passed: st.pyplot doesn't pass it either, so both paths
    fall through to rcParams['savefig.facecolor'] ('auto' = the figure's own
    facecolor) and the radars keep their '#f5f1e9' background.

    The close is in a finally so a savefig raise can't leak the figure.
    """
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
        return buf.getvalue()
    finally:
        plt.close(fig)


@st.cache_data(ttl=86400, show_spinner=False, max_entries=192)
# mpl_locked INSIDE cache_data: a cache hit returns PNG bytes without
# touching matplotlib or the lock; only an actual (re)build serialises.
# The build + savefig + close all happen inside _fig_png_bytes, so the
# whole figure lifecycle is one critical section (see mpl_safety).
@mpl_locked
def _render_match_figure_png(kind, match_id, team_name, fig_ver,
                              _match_events_df, _match_info, _match_lineup,
                              _obv=None):
    """One Match Analysis figure -> PNG bytes.

    KEY: (kind, match_id, team_name, FIGURE_CACHE_VERSION).

    Why that key is complete — every render input is a function of match_id
    and team_name:
      * _match_events_df is always raw_events_df[matchId == match_id].
        raw_events_df is the UNSCOPED master frame (assigned once in
        load_data, never reassigned), and Match Analysis never scopes it —
        it slices by matchId directly. A matchId belongs to exactly one
        match in exactly one season and competition, so the league/season
        selectors cannot change these events. season/comp are therefore
        deliberately absent from this key; adding them would only split the
        cache into identical entries.
      * _match_info is season_matches_df's row for match_id.
      * _match_lineup is match_lineups[match_id][team_name].
      * team_name is what separates home from away — the two calls are
        otherwise identical (same kind, same frame). If anything here is
        cross-wired, it is this, so it is the one component to check first.

    Returns PNG bytes, or None when the plotter produced no figure.
    """
    _obv = _obv or {}
    if kind == 'shotmap':
        fig = create_match_shotmap(_match_events_df, _match_info, team_name)
    elif kind == 'xg_flowchart':
        # Whole-match figure: both teams on one axes, so team_name is None.
        fig = plot_xg_flowchart(_match_events_df, _match_info)
    elif kind == 'obv_momentum':
        # Whole-match figure; team ids derived from the events slice because
        # matches_summary home/awayTeamId are unpopulated.
        minute_df = _obv.get('minute')
        if minute_df is None or minute_df.empty:
            return None
        if 'team.id' not in _match_events_df.columns:
            return None
        _named = _match_events_df.dropna(subset=['team.id', 'team.name'])
        name_to_id = (_named.groupby('team.name')['team.id']
                      .first().astype(int).to_dict())
        home_nm = _match_info.get('homeTeamName')
        away_nm = _match_info.get('awayTeamName')
        if home_nm not in name_to_id or away_nm not in name_to_id:
            return None
        _goal_rows = _match_events_df[
            (_match_events_df['type.primary'] == 'shot')
            & (_match_events_df.get('shot.isGoal') == True)]
        goals = [{'minute': r['minute'], 'teamId': int(r['team.id'])}
                 for _, r in _goal_rows.iterrows() if pd.notna(r.get('team.id'))]
        fig = obv_viz.plot_obv_momentum(
            minute_df, name_to_id[home_nm], name_to_id[away_nm],
            home_nm, away_nm, goals)
    elif kind == 'avg_positions':
        fig = pv.plot_average_positions(_match_events_df, team_name,
                                         match_lineup=_match_lineup)
    elif kind == 'avg_positions_by_subs':
        fig = pv.plot_avg_positions_by_subs(_match_events_df, team_name,
                                             match_lineup=_match_lineup)
    elif kind == 'passing_network':
        fig = pv.plot_passing_network(_match_events_df, team_name,
                                      obv_pairs=_obv.get('pairs'))
    elif kind == 'recovery_map':
        fig = pv.plot_recovery_map(_match_events_df, team_name)
    elif kind == 'loss_map':
        fig = pv.plot_loss_map(_match_events_df, team_name)
    elif kind == 'defensive_duels':
        fig = pv.plot_defensive_duels_map(_match_events_df, team_name)
    elif kind == 'shot_assists':
        fig = pv.plot_shot_assists_and_dribbles(_match_events_df, team_name)
    else:
        raise ValueError(f"unknown match figure kind: {kind!r}")
    return _fig_png_bytes(fig) if fig is not None else None


@st.cache_data(ttl=86400, show_spinner=False, max_entries=192)
# mpl_locked INSIDE cache_data: a cache hit returns PNG bytes without
# touching matplotlib or the lock; only an actual (re)build serialises.
# The build + savefig + close all happen inside _fig_png_bytes, so the
# whole figure lifecycle is one critical section (see mpl_safety).
@mpl_locked
def _render_team_figure_png(kind, team_name, season_key, comp_key, stage_key,
                             extra, fig_ver,
                             _team_events_df, _team_matches_df, _payload):
    """One Team Analysis figure -> PNG bytes.

    KEY: (kind, team_name, season_key, comp_key, stage_key, extra,
          FIGURE_CACHE_VERSION).

    Why that key is complete — unlike Match Analysis, these figures read a
    SCOPED frame, so the scope has to be in the key. Both frames are built by
    exactly two steps at the top of the Team Analysis block:
        team_events_df  = get_filtered_events(raw_events_df,
                                              active_season_ids,
                                              selected_comp_ids)
        team_events_df, team_matches_df = filter_by_stage(..., selected_stage)
    so (season_key, comp_key, stage_key) pins both frames down, and team_name
    picks the team out of them.

    season_key: _season_id_list(active_season_ids) normalised to a SORTED
      TUPLE. active_season_ids is None | int | list, and the None case ('All
      Seasons' — no season filter) must stay distinct from any specific
      season; the empty tuple keeps it distinct. Sorting means [a, b] and
      [b, a] share one entry, which is correct — filter_by_stage/
      get_filtered_events use `.isin`, so order can't change the result.
    comp_key: sorted tuple of selected_comp_ids (Liga 3 / Campeonato / both).
    stage_key: the stage label, or '' for STAGE_ALL/None.
    extra: the remaining per-figure inputs, as a hashable tuple. Anything
      DRAWN AS TEXT must live here — the radars print league and season
      labels onto the image, and season_label comes from selected_season_id
      rather than active_season_ids, so it is passed explicitly instead of
      assumed to follow from season_key.

    For the radars, `extra` also carries the plotted VALUES themselves. They
    are only ~10 floats and they make the key airtight: the picture is then a
    pure function of the key, independent of whether the upstream stat cache
    (calculate_all_team_radars_stats / calculate_set_piece_metrics) keyed
    itself correctly. The pitch maps can't do that — their input is a
    multi-hundred-MB event frame — so they rely on the scope key above.

    Returns PNG bytes, or None when the plotter produced no figure.
    """
    if kind == 'radar':
        _title, _params, _color, _league, _season = extra
        _values_raw, _values_pct = _payload
        fig = plot_radar_chart(list(_params), list(_values_raw),
                                list(_values_pct), team_name, _title, _color,
                                league=_league, season=_season)
    elif kind == 'formation_xi':
        _formation = extra[0]
        fig = create_formation_graphic(_formation, _payload, team_name)
    elif kind == 'season_shotmap_for':
        fig = create_season_shotmap(_team_events_df, team_name)
    elif kind == 'season_shotmap_against':
        fig = create_season_shots_against_shotmap(_team_events_df,
                                                   _team_matches_df, team_name)
    elif kind == 'rolling_xg':
        fig = plot_match_xg_history(_payload, team_name)
    elif kind == 'corner_analysis':
        _side = extra[0]
        fig = plot_corner_analysis(_team_events_df, team_name, _side)
    elif kind == 'zone_heatmap':
        _tag = extra[0]
        # league_events_df is the same scoped frame — the plotter derives the
        # league average from it, so it needs no key of its own.
        fig = pv.plot_zone_heatmap(_team_events_df, team_name, _tag,
                                    league_events_df=_team_events_df)
    elif kind == 'passing_network':
        _p = _payload if isinstance(_payload, dict) else {}
        fig = pv.plot_passing_network(_team_events_df, team_name,
                                      obv_pairs=_p.get('pairs'))
    elif kind == 'defensive_structure':
        fig = pv.plot_defensive_structure(_team_events_df, team_name,
                                           league_events_df=_team_events_df)
    elif kind == 'phase_profile':
        _p = _payload if isinstance(_payload, dict) else {}
        fig = obv_viz.plot_phase_profile(_p.get('profile'), team_name)
    elif kind == 'obv_categories':
        _p = _payload if isinstance(_payload, dict) else {}
        if _p.get('team_season') is None or _p.get('team_id') is None:
            return None
        fig = obv_viz.plot_team_obv_categories(
            _p['team_season'], _p['team_id'], team_name)
    elif kind == 'avg_positions':
        # Same visual as the Opposition Report's kind of this name. extra[0]
        # is the sorted tuple of XI names (or None) — it restricts which
        # players are drawn, so it is a real picture input.
        _xi_names = extra[0]
        fig = pv.plot_average_positions(
            _team_events_df, team_name,
            player_names=set(_xi_names) if _xi_names else None)
    elif kind == 'shot_assists':
        fig = pv.plot_shot_assists_and_dribbles(_team_events_df, team_name)
    else:
        raise ValueError(f"unknown team figure kind: {kind!r}")
    return _fig_png_bytes(fig) if fig is not None else None


_STRENGTH_COLS = ('Attacking Strength', 'Defending Strength')


def _plot_cell_key(v):
    """One frame cell, normalised for a cache key. Rounded so float noise
    can't split two entries that would draw the same picture."""
    try:
        return round(float(v), 6)
    except (TypeError, ValueError):
        return str(v)


def _plot_values_key(df, cols):
    """The numbers a strength/scatter figure draws, as a hashable tuple.

    plot_team_strength and plot_custom_scatter read NOTHING from their frame
    except its index and the columns named here — the points, the axis limits,
    plot_custom_scatter's mean quadrant lines and the per-team logo lookup all
    come out of that slice. So this tuple describes the picture's data
    completely, which is what lets _render_league_figure_png key on the DATA
    rather than on the scope.

    Row order is preserved, not sorted: it is the draw order, so it decides
    which logo lands on top where two overlap.

    A missing column returns a sentinel instead — the plotters draw a 'metric
    not found' card in that case, which is its own picture.
    """
    missing = tuple(c for c in cols if c not in df.columns)
    if missing:
        return ('__missing__', missing)
    return tuple(
        (str(idx), *(_plot_cell_key(v) for v in vals))
        for idx, *vals in df[list(cols)].itertuples(name=None)
    )


@st.cache_data(ttl=86400, show_spinner=False, max_entries=64)
# mpl_locked INSIDE cache_data: a cache hit returns PNG bytes without
# touching matplotlib or the lock; only an actual (re)build serialises.
# The build + savefig + close all happen inside _fig_png_bytes, so the
# whole figure lifecycle is one critical section (see mpl_safety).
@mpl_locked
def _render_league_figure_png(kind, values_key, extra, day_key, fig_ver, _stats_df):
    """One League Analysis figure -> PNG bytes.

    KEY: (kind, values_key, extra, day_key, FIGURE_CACHE_VERSION).

    Unlike the Match/Team renderers this one does NOT key on the data scope.
    Its two plotters touch only the frame's index and the columns they plot,
    so values_key (see _plot_values_key) IS the data half of the key. That is
    deliberately stronger than a scope key would be: it holds however
    league_events_df was filtered, and regardless of whether the stat caches
    upstream — calculate_team_strength, calculate_expanded_team_stats,
    calculate_set_piece_metrics — keyed themselves correctly. A scope change
    either moves these numbers, and misses, or it doesn't, in which case the
    picture is genuinely identical and one entry is right.

    `extra` carries what is drawn but is NOT a number in the frame. This is
    the half no scope key could supply, because it is WIDGET STATE:
      team_strength:  (teams_to_include, icon_zoom, season_label, league_label)
        teams_to_include is the subset actually plotted; the axis limits still
        come from every row, which is why values_key covers the whole frame.
        season_label / league_label are stamped into the title (the
        multi-season chart passes "Multi-Season"), so they must be in the key.
      custom_scatter: (x_metric, y_metric, invert_x, invert_y,
                       league_label, season_label)
        The metrics are the axis LABELS as well as the columns, and the invert
        flags flip the limits without moving a single value — so neither is
        implied by values_key. The two labels are title text, as above.

    day_key is today's date. Both plotters stamp 'As of: {date}' into the
    title, so without it a figure built at 23:59 would keep serving
    yesterday's date for the rest of the 24 h TTL.

    icons/: adding a team's logo changes the picture without moving any key
    component, so that is a FIGURE_CACHE_VERSION bump.

    Returns PNG bytes, or None when the plotter produced no figure.
    """
    if kind == 'team_strength':
        _teams, _icon_zoom, _season_label, _league_label = extra
        fig = plot_team_strength(_stats_df,
                                  teams_to_include=list(_teams) if _teams else None,
                                  icon_zoom=_icon_zoom,
                                  league=_league_label, season=_season_label)
    elif kind == 'custom_scatter':
        _x_metric, _y_metric, _invert_x, _invert_y, _league_label, _season_label = extra
        fig = plot_custom_scatter(_stats_df, _x_metric, _y_metric,
                                   _invert_x, _invert_y,
                                   league=_league_label, season=_season_label)
    else:
        raise ValueError(f"unknown league figure kind: {kind!r}")
    return _fig_png_bytes(fig) if fig is not None else None


def season_report_cache_key(comp_ids, season_ids, stage=None):
    """Cache key for compute_team_season_metrics — ONE format for Team
    Analysis, the Opposition Report and the boot prewarm, so one
    league+season computes once and serves every team on both pages."""
    sids = (','.join(map(str, season_ids)) if isinstance(season_ids, (list, tuple))
            else season_ids)
    stage_key = 'all' if stage in (STAGE_ALL, None) else stage
    return f"sr_{','.join(map(str, comp_ids or []))}_{sids}_{stage_key}"


def render_season_report_section(team_events_df, team_matches_df, team_name,
                                   season_ids=None, stage=None, cache_key=None,
                                   on_team_select=None):
    """Render the 7-dimension season report for one team.

    on_team_select(team_name): when given, clicking any team's dot calls it
    (the pages use it to open that team's report).

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
                event = st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={'displayModeBar': False, 'responsive': True},
                    key=f"sr_dim_{team_name}_{dim_name}",
                    on_select='rerun' if on_team_select else 'ignore',
                    selection_mode='points',
                )
                if on_team_select:
                    import team_interactive
                    picked = [row[0] for row in team_interactive.selected_customdata(event) if row]
                    if picked and picked[0] != team_name:
                        on_team_select(picked[0])


# ==============================================================================
# 5. STREAMLIT APP UI
# ==============================================================================
st.markdown('<h1 style="text-align: center; color: #1a1a1a; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 0;">Atlético CP Analytics</h1>', unsafe_allow_html=True)

# --- Load Data ---
with st.spinner("Loading match data..."):
    raw_events_df, matches_summary_df, all_match_data, season_team_stats, player_minutes_data, match_lineups = load_data()
    # Which seasons actually have events — drives the default season selection.
    SEASON_MATCHES_WITH_EVENTS = _season_match_counts(raw_events_df)

# RSS telemetry: boot line + 5-min daemon (singleton via cache_resource)
try:
    _start_rss_telemetry()
except Exception as _rss_e:
    logger.warning(f"[rss] telemetry could not start: {_rss_e}")

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
        st.session_state.current_page = navigation.HOME
    if 'radio_key_version' not in st.session_state:
        st.session_state.radio_key_version = 0
    if 'shadow_teams' not in st.session_state:
        st.session_state.shadow_teams = {}
    if 'player_profile_current_id' not in st.session_state:
        st.session_state.player_profile_current_id = None
    if 'player_profile_last_season' not in st.session_state:
        st.session_state.player_profile_last_season = None

    # --- Sidebar: context bar (filled once the page is known), then navigation ---
    _context_slot = st.sidebar.container()
    st.sidebar.markdown('<div style="text-align: center; padding: 1rem 0 0.5rem 0;"><h2 style="color: #ffffff; font-size: 1.3rem; font-weight: 600; margin: 0;">Navigation</h2></div>', unsafe_allow_html=True)

    # Cross-page bridge: a view that selected a player asks for the profile,
    # optionally carrying the season it was looking at. The season goes into
    # the context bar's key HERE, before the bar draws its widget (Streamlit
    # rejects writes to a widget key after the widget exists in a run).
    if st.session_state.nav_to_profile:
        st.session_state.current_page = 'Player Profile'
        st.session_state.nav_to_profile = False
        if st.session_state.get('nav_has_season', False):
            _nav_sid = st.session_state.get('nav_season_id')
            context_bar.set_context(season=(
                context_bar.ALL_SEASONS if _nav_sid is None else SEASON_ID_MAP.get(_nav_sid, context_bar.ALL_SEASONS)))
            st.session_state.nav_season_id = None
            st.session_state.nav_has_season = False

    ANALYSIS_OPTIONS = navigation.ALL_PAGES

    if '--precompute' in sys.argv:
        import time as _t
        import gc as _gc
        print('[precompute] warming per-season stats caches...', flush=True)
        # Current-season (and All-Seasons, which include it) disk caches go
        # stale as matches accrue — the cache key carries no data fingerprint,
        # so a cache written at matchweek 1 would be served all season. CI is
        # the cache factory: delete those scopes first to force fresh computes.
        _current_sids = {cfg['current_season'] for cfg in COMPETITIONS.values()}
        for _cid, _cfg in COMPETITIONS.items():
            # Each league's individual seasons PLUS its All-Seasons scope
            # (_sid=None). The league-aware scope key keeps the two single-league
            # All-Seasons caches from colliding on one 'None' file.
            for _sid in (list(_cfg['seasons'].keys()) + [None]):
                _label = 'ALL' if _sid is None else _sid
                _t0 = _t.time()
                _ev = _mins = None
                if _sid in _current_sids or _sid is None:
                    _scope = f'all_{_cid}' if _sid is None else str(_sid)
                    _stale = [f'player_stats_{STATS_CACHE_VERSION}_{_scope}.parquet',
                              f'player_percentiles_{STATS_CACHE_VERSION}_{_scope}.parquet']
                    if _sid is not None:
                        _stale.append(f'team_strength_{_sid}.parquet')
                    for _fname in _stale:
                        _fp = os.path.join(STATS_CACHE_DIR, _fname)
                        if os.path.exists(_fp):
                            os.remove(_fp)
                            print(f'[precompute] dropped stale cache {_fname}', flush=True)
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

    # Grouped navigation (Club / Opposition / Players / Recruitment); the
    # single source of truth is st.session_state.current_page — see navigation.py.
    analysis_type = navigation.render_sidebar_nav()
    with _context_slot:
        context_bar.render(analysis_type)
    navigation.scroll_to_top_on_page_change(analysis_type)

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

    if analysis_type == navigation.HOME:
        views.home.render()

    elif analysis_type == 'Match Analysis':
        views.match_analysis.render()



    elif analysis_type == 'Team Analysis':
        views.team_analysis.render()


    elif analysis_type == 'League Analysis':
        views.league_analysis.render()


    
    # --- UPDATED: Renamed to Player Profile ---
    elif analysis_type == 'Player Profile':
        views.player_profile.render()


# --- NEW: Player Comparison Section ---
    elif analysis_type == 'Player Comparison':
        views.player_comparison.render()


    # --- NEW: Player Analysis Section ---
    elif analysis_type == 'Player Analysis':
        views.player_analysis.render()



    elif analysis_type == 'Match Predictor':
        views.match_predictor.render()


    # ==========================================================================
    # SHADOW TEAM BUILDER
    # ==========================================================================
    elif analysis_type == 'Shadow Team':
        views.shadow_team.render()


    elif analysis_type == 'Opposition Report':
        views.opposition.render()



else:
    st.error("Data files not loaded. Please run `process_data.py` locally and ensure all artifacts are pushed to GitHub.")

# Free every matplotlib figure created during this rerun. st.pyplot has
# already rasterized them to PNG; without this, the 41 figure call sites
# accumulate across reruns and leak memory on the HF Space.
#
# close('all') is process-global, not session-scoped: it frees figures owned by
# every other browser session too, not just this rerun's. That is only safe
# because it takes MPL_LOCK and because every build+render elsewhere holds that
# same lock across the WHOLE span -- so no other thread can be holding a figure
# that is built but not yet rasterized when this runs. Dropping the lock here,
# or splitting a build/render span, reintroduces a use-after-free in Agg.
with MPL_LOCK:
    plt.close('all')
