# app.py

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
STATS_CACHE_VERSION = 'v11'  # Bump this when adding/removing stat columns to invalidate old caches

# ==============================================================================
# 2. DATA LOADING (with Caching)
# ==============================================================================
@st.cache_resource(ttl=3600)  # cache_resource avoids serializing large DataFrames
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

@st.cache_data(ttl=3600)
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

@st.cache_data(ttl=3600)
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
        logger.info(f"Loaded GPA values: {len(df):,} rows × {df.shape[1]} cols")
        return df
    except Exception as e:
        logger.error(f"Failed to load GPA values parquet: {e}")
        return pd.DataFrame()


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

def get_season_player_minutes(player_minutes_data, season_id, comp_ids=None):
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
    options = ["Liga 3", "Campeonato", "Both"]
    selected = st.sidebar.selectbox(
        "League",
        options,
        index=0,
        key=f"league_select_{section_key}"
    )
    if selected == "Both":
        return list(COMPETITIONS.keys())
    for comp_id, comp_config in COMPETITIONS.items():
        if comp_config["name"] == selected:
            return [comp_id]
    return list(COMPETITIONS.keys())


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

@st.cache_data(ttl=3600)
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


@st.cache_data(ttl=3600)
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
INVERT_METRICS = ['Loss index', 'goalsConceded']
OUTPUT_METRICS = ['Goals', 'Assists', 'xG', 'npxG', 'xA', 'xAOP', 'xASP', 'xT', 'xTOP', 'xTSP', 'Second assists', 'Shots', 'xG per Shot']
PASSING_METRICS = ['Passes', 'Passes successful', 'Passes successful %', 'Long passes', 'Long passes successful', 'Long passes successful %', 'Crosses', 'Crosses successful', 'Crosses successful %', 'Through passes', 'Through passes successful', 'Progressive Passes', 'Passes to final third', 'Passes to final third successful', 'Forward passes', 'Forward passes successful', 'Back passes', 'Back passes successful', 'Passes to penalty area', 'Passes to penalty area successful', 'Deep Completions', 'Throw-ins', 'Avg max throw-in distance', 'Throw-ins into box', 'Avg max throw-in into box distance', 'Avg max throw-in into box aerial distance']
DEFENSIVE_METRICS = ['Interceptions', 'Aerial duels', 'Aerial duels successful', 'Aerial duels successful %', 'Sliding tackles', 'Sliding tackles successful', 'Sliding tackles successful %', 'Recoveries', 'Recoveries Opp Half', 'Counterpressing Recoveries', 'Defensive duels', 'Defensive duels successful', 'Defensive duels successful %', 'Clearances', 'Fouls', 'Yellow cards', 'Red cards']
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

@st.cache_data(ttl=3600)  # Cache expires after 1 hour to prevent memory leaks
def calculate_all_player_stats(_raw_events_df, _player_minutes_df, season_id=None, cache_version=STATS_CACHE_VERSION):
    """
    A new, streamlined, and correct function to calculate all player stats
    for the player profile page (Per 90 and Totals).
    season_id is used as a cache key so Streamlit recomputes when the season changes.
    """
    # Disk cache: load pre-computed results if available
    _REQUIRED_STAT_COLS = {'Throw-ins', 'Avg max throw-in distance', 'Throw-ins into box', 'Avg max throw-in into box distance', 'Avg max throw-in into box aerial distance', 'Defensive Area', 'Opp xT into Def Area', 'Opp Pass Success % into Def Area', 'Opp xT from Def Area', 'Territorial Dominance', 'Opp xT into Def Area OE', 'Opp xT from Def Area OE', 'Territorial Dominance OE', 'xTOP', 'xTSP'}
    if season_id is not None:
        cache_path = os.path.join(STATS_CACHE_DIR, f'player_stats_{STATS_CACHE_VERSION}_{season_id}.parquet')
        if os.path.exists(cache_path):
            cached = pd.read_parquet(cache_path)
            if _REQUIRED_STAT_COLS.issubset(cached.columns):
                print(f"Loading cached player stats for season {season_id}")
                if cached.index.name == 'playerId':
                    cached = cached.reset_index()
                return cached
            else:
                print(f"Cache outdated (missing columns), recomputing stats for season {season_id}")
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

    # DEBUG: Log goalsConceded before normalization
    if 'goalsConceded' in base_df.columns:
        gk_mask = base_df['goalsConceded'] > 0
        if gk_mask.any():
            print(f"  DEBUG goalsConceded BEFORE normalization: {base_df.loc[gk_mask, 'goalsConceded'].head(3).tolist()}")
            print(f"  DEBUG totalMinutes for those GKs: {base_df.loc[gk_mask, 'totalMinutes'].head(3).tolist()}")

    for col in all_calculated_metrics:
        if col not in dont_normalize and pd.api.types.is_numeric_dtype(base_df[col]):
            if col == 'goalsConceded':
                print(f"  DEBUG: Normalizing goalsConceded (in dont_normalize={col in dont_normalize})")
            base_df[col] = np.where(
                minutes_gt_0,
                (base_df[col].astype(float) / total_minutes) * 90,
                0
            )

    # DEBUG: Log goalsConceded after normalization
    if 'goalsConceded' in base_df.columns:
        gk_mask = base_df['goalsConceded'] > 0
        if gk_mask.any():
            print(f"  DEBUG goalsConceded AFTER normalization: {base_df.loc[gk_mask, 'goalsConceded'].head(3).tolist()}")
            
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
    
    print("--- FINISHED: New All-Player-Stats Calculation ---")
    result = base_df.fillna(0).reset_index()

    # DEBUG: Verify goalsConceded in final result for Diogo Figueiredo
    if 'goalsConceded' in result.columns and 'playerId' in result.columns:
        _dbg = result[result['playerId'] == 593057]
        if not _dbg.empty:
            print(f"  DEBUG FINAL RESULT: Diogo Figueiredo goalsConceded={_dbg['goalsConceded'].values[0]:.4f}, totalMinutes={_dbg['totalMinutes'].values[0]}")

    # Save to disk cache for fast loading on restart
    if season_id is not None:
        os.makedirs(STATS_CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(STATS_CACHE_DIR, f'player_stats_{STATS_CACHE_VERSION}_{season_id}.parquet')
        try:
            result.to_parquet(cache_path)
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
def calculate_player_percentiles_and_scores(_player_data_df, _position_groups, _weights, _invert_metrics, min_minutes=500, season_id=None, cache_version=STATS_CACHE_VERSION):
    """Calculates percentiles and scores for all players based on position.
    Players below min_minutes are kept but ranked against the min_minutes+ population
    (each low-minute player is temporarily added to the sample for their own percentile).
    season_id is used as a cache key so Streamlit recomputes when the season changes."""
    # Disk cache: load pre-computed results if available
    _REQUIRED_PCT_COLS = {'Throw-ins', 'Avg max throw-in distance', 'Throw-ins into box', 'Avg max throw-in into box distance', 'Avg max throw-in into box aerial distance', 'Defensive Area', 'Opp xT into Def Area', 'Opp Pass Success % into Def Area', 'Opp xT from Def Area', 'Territorial Dominance', 'Opp xT into Def Area OE', 'Opp xT from Def Area OE', 'Territorial Dominance OE', 'xTOP', 'xTSP'}
    if season_id is not None:
        cache_path = os.path.join(STATS_CACHE_DIR, f'player_percentiles_{STATS_CACHE_VERSION}_{season_id}.parquet')
        if os.path.exists(cache_path):
            cached = pd.read_parquet(cache_path)
            if _REQUIRED_PCT_COLS.issubset(cached.columns):
                print(f"Loading cached player percentiles for season {season_id}")
                if cached.index.name == 'playerId':
                    cached = cached.reset_index()
                return cached
            else:
                print(f"Percentiles cache outdated (missing columns), recomputing for season {season_id}")
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
    if season_id is not None:
        os.makedirs(STATS_CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(STATS_CACHE_DIR, f'player_percentiles_{STATS_CACHE_VERSION}_{season_id}.parquet')
        try:
            result.to_parquet(cache_path)
            print(f"  Cached player percentiles to {cache_path}")
        except Exception as e:
            print(f"  Warning: Could not cache percentiles: {e}")

    return result


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
        elif metric in DEFENSIVE_METRICS: color = category_colors['defensive']
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
            elif metric in DEFENSIVE_METRICS: color = 'red'
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
        elif metric in DEFENSIVE_METRICS: color = category_colors['defensive']
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
def calculate_all_team_radars_stats(season_events_df, matches_summary_df, season_id=None):
    """Calculates aggregated stats and percentiles for Offensive, Distribution, and Defensive radars.
    Uses Wyscout team advanced stats if available, otherwise falls back to event-based calculation.
    season_id filters the Wyscout stats to compare within a single season."""

    # Count how many teams are in the events data for comparison
    event_teams = set()
    if 'team.name' in season_events_df.columns:
        event_teams = set(season_events_df['team.name'].dropna().unique())

    wyscout_stats = load_team_advanced_stats()
    if wyscout_stats is not None:
        raw_df, pct_df = _build_radars_from_wyscout(wyscout_stats, season_id=season_id)
        # Only use Wyscout stats if they cover at least half the teams in the events
        if not raw_df.empty and (not event_teams or len(raw_df) >= len(event_teams) * 0.5):
            print(f"Using Wyscout team advanced stats for radars ({len(raw_df)} teams)...")
            return raw_df, pct_df
        print(f"Wyscout stats insufficient ({len(raw_df)} of {len(event_teams)} teams), falling back to events...")

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

@st.cache_data(ttl=3600)
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

    analysis_type = st.sidebar.radio(
        "Choose Analysis Type",
        ANALYSIS_OPTIONS,
        index=ANALYSIS_OPTIONS.index(st.session_state.current_page),
        key=f"analysis_type_radio_{st.session_state.radio_key_version}"
    )
    st.session_state.current_page = analysis_type

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
                st.dataframe(home_shots_table)

            with col2_table:
                st.markdown(f"**{selected_match_info['awayTeamName']}**")
                away_shots_table = get_shot_table(match_events_df, selected_match_info['awayTeamName'])
                st.dataframe(away_shots_table)
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
                    if isinstance(df, pd.DataFrame): st.dataframe(df)
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
        team_events_df = filter_by_league(get_season_events(raw_events_df, active_season_ids), selected_comp_ids)
        team_matches_df = filter_by_league(get_season_matches(matches_summary_df, active_season_ids), selected_comp_ids)
        team_player_minutes_df = get_season_player_minutes(player_minutes_data, active_season_ids, comp_ids=selected_comp_ids)
        team_season_stats = get_season_team_stats(season_team_stats, active_season_ids, comp_ids=selected_comp_ids)

        all_teams_t = sorted(pd.concat([team_matches_df.get('homeTeamName'), team_matches_df.get('awayTeamName')]).dropna().unique())
        selected_team_t = st.sidebar.selectbox("Select a Team", all_teams_t, key="team_select_tab")
        st.header(f"Team Report: {selected_team_t}")

        # Load player details for roster table
        player_details_df = load_player_details()

        stats_df_raw, stats_df_pct = calculate_all_team_radars_stats(team_events_df, team_matches_df, season_id=active_season_ids if isinstance(active_season_ids, list) else selected_season_id)

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
                    hide_index=True
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
                rolling_xg_data_for_plot = calculate_xg_history_data(raw_events_df, matches_summary_df)
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
        league_events_df = filter_by_league(get_season_events(raw_events_df, active_season_ids), selected_comp_ids)
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
                s_events = filter_by_league(get_season_events(raw_events_df, sid), selected_comp_ids)
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
        profile_events_df = filter_by_league(get_season_events(raw_events_df, active_season_ids), selected_comp_ids)
        profile_matches_df = filter_by_league(get_season_matches(matches_summary_df, active_season_ids), selected_comp_ids)
        profile_player_minutes_df = get_season_player_minutes(player_minutes_data, active_season_ids, comp_ids=selected_comp_ids)

        # --- 1. Load All Necessary Data ---
        player_details_df = load_player_details()

        try:
            with st.spinner("Calculating player statistics (this may take a moment on first load)..."):
                player_stats_df = calculate_all_player_stats(profile_events_df, profile_player_minutes_df, season_id=selected_season_id)
                # Merge GPA Value columns (scoped to active season × competition)
                player_stats_df = merge_gpa_values_into_stats(player_stats_df, active_season_ids, selected_comp_ids)
                # --- NEW: Calculate percentiles ---
                player_stats_with_scores_df = calculate_player_percentiles_and_scores(
                    player_stats_df, POSITION_GROUPS, WEIGHTS, INVERT_METRICS, min_minutes=500, season_id=selected_season_id
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

        st.divider()

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
            
        except:
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
            "Dribbling": DRIBBLING_METRICS,
            "Goalkeeping": GOALKEEPING_METRICS
        }

        player_is_gk = (per_90_stats.get('primaryPosition', 'N/A') == 'GK')

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
                        
                    stats_subset = stats_subset_series.to_frame(name='Value')
                    def _fmt_stat(metric_name, x):
                        if not isinstance(x, (int, float)):
                            return str(x)
                        if metric_name in THOUSANDTHS_METRICS:
                            return f"{x:.3f}"
                        if np.round(x) == x and '%' not in str(x):
                            return f"{x:.0f}"
                        return f"{x:.2f}"
                    stats_subset['Value'] = [_fmt_stat(m, v) for m, v in zip(stats_subset.index, stats_subset['Value'])]
                    st.dataframe(stats_subset, use_container_width=True)
        
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

            # --- VISUALIZATION ---
            col_shot_map, col_shot_table = st.columns([1, 1.4])
            
            with col_shot_map:
                st.markdown("**Season Shot Map**")
                # Pass the fully processed shot_log which has 'Shot Number'
                fig_player_shots = create_player_shotmap(shot_log, selected_player_name)
                st.pyplot(fig_player_shots, use_container_width=True)
                plt.close(fig_player_shots)
                
            with col_shot_table:
                st.markdown("**Shot Log**")
                
                # Prepare display table
                display_cols = ['Shot Number', 'Date', 'Opponent', 'Result', 'xG', 'Body Part', 'SCA']
                table_display = shot_log[display_cols].rename(columns={
                    'Shot Number': '#',
                    'SCA': 'Creating Action'
                }).sort_values(by='#', ascending=False) # Show newest first (highest number)
                
                st.dataframe(table_display, use_container_width=True, height=500, hide_index=True)

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
                st.dataframe(body_summary, use_container_width=True)
                
            with col_sum2:
                st.markdown("**Stats by Creating Action**")
                sca_summary = shot_log.groupby('SCA').agg(
                    Shots=('id', 'count'),
                    Goals=('shot.isGoal', 'sum'),
                    Total_xG=('xG', 'sum')
                ).sort_values(by='Total_xG', ascending=False)
                sca_summary['xG/Shot'] = (sca_summary['Total_xG'] / sca_summary['Shots']).round(2)
                sca_summary['Total_xG'] = sca_summary['Total_xG'].round(2)
                st.dataframe(sca_summary, use_container_width=True)

        else:
            st.info("No shots recorded for this player.")

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
        comp_events_df = filter_by_league(get_season_events(raw_events_df, active_season_ids), selected_comp_ids)
        comp_player_minutes_df = get_season_player_minutes(player_minutes_data, active_season_ids, comp_ids=selected_comp_ids)

        # --- 1. Load Data ---
        try:
            with st.spinner("Loading player statistics..."):
                player_stats_df = calculate_all_player_stats(comp_events_df, comp_player_minutes_df, season_id=selected_season_id)
                player_stats_df = merge_gpa_values_into_stats(player_stats_df, active_season_ids, selected_comp_ids)
                player_stats_with_scores_df = calculate_player_percentiles_and_scores(
                    player_stats_df, POSITION_GROUPS, WEIGHTS, INVERT_METRICS, min_minutes=500, season_id=selected_season_id
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
        analysis_events_df = filter_by_league(get_season_events(raw_events_df, active_season_ids), selected_comp_ids)
        analysis_player_minutes_df = get_season_player_minutes(player_minutes_data, active_season_ids, comp_ids=selected_comp_ids)

        # --- 1. Load Data ---
        try:
            with st.spinner("Loading player statistics..."):
                player_stats_df = calculate_all_player_stats(analysis_events_df, analysis_player_minutes_df, season_id=selected_season_id)
                player_stats_df = merge_gpa_values_into_stats(player_stats_df, active_season_ids, selected_comp_ids)
                player_stats_with_scores_df = calculate_player_percentiles_and_scores(
                    player_stats_df, POSITION_GROUPS, WEIGHTS, INVERT_METRICS, min_minutes=500, season_id=selected_season_id
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
        _selector_options = ["Overview"] + _ordered_templates + ["Individual Metric"]
        _selected_view = st.sidebar.selectbox(
            "View:",
            _selector_options,
            index=0,
            key="player_analysis_view"
        )

        # Minimum minutes filter
        max_minutes = int(player_stats_with_scores_df['totalMinutes'].max())
        min_minutes_filter = st.sidebar.slider(
            "Minimum Minutes Played:",
            min_value=500,
            max_value=max(max_minutes, 500),
            value=500,
            step=45,
            key="player_analysis_min_minutes"
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

        # --- Show Only Position toggle ---
        analysis_pos_played_filter = st.sidebar.checkbox("Show Only Position", key="analysis_pos_played_filter")
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

            sorted_tdf = tdf.sort_values(by=score_col, ascending=False).head(n_players)

            if compact:
                # Overview mode: Rank, Player, Team, Position, Minutes, Age, Rating
                cols = ['playerName', 'teamName', 'primaryPosition', 'totalMinutes', score_col]
            else:
                # Template-specific mode: include all weighted metrics (weight > 0) sorted by weight desc
                template_weights = WEIGHTS.get(template_name, {})
                weighted_metrics = sorted(
                    [(m, w) for m, w in template_weights.items() if w > 0],
                    key=lambda x: x[1], reverse=True
                )
                metric_cols = [m for m, _ in weighted_metrics if m in sorted_tdf.columns]
                cols = ['playerName', 'teamName', 'primaryPosition', 'totalMinutes', score_col] + metric_cols

            cols = [c for c in cols if c in sorted_tdf.columns]
            display = sorted_tdf[cols].copy()
            display = display.rename(columns={
                'playerName': 'Player',
                'teamName': 'Team',
                'primaryPosition': 'Position',
                'totalMinutes': 'Minutes',
                score_col: 'Rating'
            })
            display['Rating'] = display['Rating'].round(1)
            display['Minutes'] = display['Minutes'].astype(int)

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
            # Build wide pivot table: each template is a column group with Player, Team, Minutes, Rating
            _OVERVIEW_ORDER = [
                'Forwards', 'Attacking Mids / Wingers', 'Central Midfielders',
                'Full Backs', 'Center Backs', 'Goalkeepers',
            ]
            # Collect per-template data as lists aligned by rank
            template_columns = {}  # {template_name: [(player, team, minutes, rating), ...]}
            for group_name in _OVERVIEW_ORDER:
                group_templates = _TEMPLATE_GROUPS.get(group_name, [])
                for tmpl in [t for t in group_templates if t in POSITION_GROUPS]:
                    display_df, _ = _build_template_table(tmpl, filtered_df, num_players, compact=True)
                    if display_df is not None and not display_df.empty:
                        rows = []
                        for _, row in display_df.iterrows():
                            rows.append((row.get('Player', ''), row.get('Team', ''), int(row.get('Minutes', 0)), round(float(row.get('Rating', 0)), 1)))
                        template_columns[tmpl] = rows

            if template_columns:
                max_rows = max(len(v) for v in template_columns.values())
                # Build MultiIndex columns DataFrame
                col_tuples = []
                data_dict = {}
                for tmpl in template_columns:
                    for sub in ['Player', 'Team', 'Min', 'Rating']:
                        col_tuples.append((tmpl, sub))
                        data_dict[(tmpl, sub)] = []

                for rank_idx in range(max_rows):
                    for tmpl in template_columns:
                        rows = template_columns[tmpl]
                        if rank_idx < len(rows):
                            player, team, mins, rating = rows[rank_idx]
                            data_dict[(tmpl, 'Player')].append(player)
                            data_dict[(tmpl, 'Team')].append(team)
                            data_dict[(tmpl, 'Min')].append(mins)
                            data_dict[(tmpl, 'Rating')].append(rating)
                        else:
                            data_dict[(tmpl, 'Player')].append('')
                            data_dict[(tmpl, 'Team')].append('')
                            data_dict[(tmpl, 'Min')].append('')
                            data_dict[(tmpl, 'Rating')].append('')

                multi_idx = pd.MultiIndex.from_tuples(col_tuples)
                overview_df = pd.DataFrame(data_dict, columns=multi_idx)
                overview_df.index = range(1, len(overview_df) + 1)
                overview_df.index.name = 'Rank'

                st.subheader("Player Overview")
                st.dataframe(overview_df, use_container_width=True)
            else:
                st.warning("No players match current filters.")

        elif _selected_view == "Individual Metric":
            # --- Individual Metric mode (preserved from original) ---
            metric_categories = {
                "Output": OUTPUT_METRICS,
                "Passing": PASSING_METRICS,
                "Defensive": DEFENSIVE_METRICS,
                "Dribbling": DRIBBLING_METRICS,
                "Goalkeeping": GOALKEEPING_METRICS
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
                        st.dataframe(weight_df, use_container_width=True, hide_index=True)
            else:
                st.warning(f"No players found for {selected_template} template with current filters.")

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
                    s_events = filter_by_league(get_season_events(raw_events_df, sid), selected_comp_ids)
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
                    st.dataframe(ratings_combined, use_container_width=True, hide_index=True)
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
                        expanded = group_name == 'Promotion' or (comp_id == 702 and group_name == list(sim_groups.keys())[0])
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
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

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
        shadow_events_df = filter_by_league(get_season_events(raw_events_df, active_season_ids), selected_comp_ids)
        shadow_player_minutes_df = get_season_player_minutes(player_minutes_data, active_season_ids, comp_ids=selected_comp_ids)

        # --- Load Data ---
        player_details_df = load_player_details()

        try:
            with st.spinner("Loading player statistics..."):
                player_stats_df = calculate_all_player_stats(shadow_events_df, shadow_player_minutes_df, season_id=selected_season_id)
                player_stats_df = merge_gpa_values_into_stats(player_stats_df, active_season_ids, selected_comp_ids)
                player_stats_with_scores_df = calculate_player_percentiles_and_scores(
                    player_stats_df, POSITION_GROUPS, WEIGHTS, INVERT_METRICS, min_minutes=500, season_id=selected_season_id
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
                        st.dataframe(scores_df, use_container_width=True, hide_index=True)
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

    # --- Transferred Players Manager (Bottom of Sidebar) ---
    st.sidebar.markdown("---")
    with st.sidebar.expander("Transferred Out Players"):
        if 'transferred_players' not in st.session_state:
            st.session_state.transferred_players = load_transferred_players()

        # Build a list of all player names across all seasons for autocomplete
        _all_names = set()
        if player_minutes_data:
            for _sid, _pm in player_minutes_data.items():
                if isinstance(_pm, pd.DataFrame) and 'playerName' in _pm.columns:
                    _all_names.update(_pm['playerName'].dropna().unique())
        all_player_names = sorted(_all_names)

        new_player = st.selectbox(
            "Add player",
            options=[""] + [n for n in all_player_names
                            if n not in st.session_state.transferred_players],
            key="transfer_add_select",
        )
        if st.button("Add", key="transfer_add_btn") and new_player:
            if new_player not in st.session_state.transferred_players:
                st.session_state.transferred_players.append(new_player)
                save_transferred_players(st.session_state.transferred_players)
                st.rerun()

        if st.session_state.transferred_players:
            st.caption("Current list:")
            for pname in list(st.session_state.transferred_players):
                col_name, col_btn = st.columns([3, 1])
                col_name.write(pname)
                if col_btn.button("X", key=f"transfer_rm_{pname}"):
                    st.session_state.transferred_players.remove(pname)
                    save_transferred_players(st.session_state.transferred_players)
                    st.rerun()
        else:
            st.caption("No players added yet.")

else:
    st.error("Data files not loaded. Please run `process_data.py` locally and ensure all artifacts are pushed to GitHub.")