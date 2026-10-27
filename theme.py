"""theme.py — the one place the dashboard's colours and figure conventions live.

Every plotter (app.py, opposition_report.py, obv_viz.py, pitch_visualizations.py,
pitch_interactive.py) and the CSS block read from here. Before 2026-09 the
same hexes were re-typed per module (the four radar colours in two files, the
cream pitch ground in four, the xG colour ramp in five places), which is why
the Team Analysis <-> Opposition Report parity rule had to police colours by
hand. Change a colour HERE; bump app.FIGURE_CACHE_VERSION so cached PNGs
re-render.

Two creams and two inks are deliberate: the page ground / text (CSS) and the
figure ground / ink (matplotlib) were tuned separately and are kept as-is.
"""
import re

# ---------------------------------------------------------------------------
# Grounds and ink
# ---------------------------------------------------------------------------
FIGURE_BG = '#f5f1e9'      # matplotlib / plotly figure and pitch ground (cream)
FIGURE_INK = '#1c2321'     # figure text and marks
PAGE_BG = '#F8F5F0'        # CSS --bg
PAGE_BG_WARM = '#F0ECE4'   # CSS --bg-warm
PAGE_CARD = '#FFFFFF'      # CSS --card
PAGE_SIDEBAR = '#1a1a1a'   # CSS --sidebar
PAGE_INK = '#1a1a1a'       # CSS --ink
PAGE_INK_2 = '#5c5650'     # CSS --ink-2
PAGE_INK_3 = '#9a948d'     # CSS --ink-3
PAGE_BORDER = '#DDD8D0'    # CSS --border
PAGE_BORDER_LIGHT = '#EBE7E1'
FONT_STACK = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

# Pitch drawing
PITCH_LINE_MPL = 'black'     # mplsoccer line colour
PITCH_LINE_PLOTLY = '#5a5a5a'

# ---------------------------------------------------------------------------
# Team semantics (who is who on a chart)
# ---------------------------------------------------------------------------
FOCUS = '#1a472a'          # the team the chart is about (dark green)
OPPONENT = '#a63a46'       # the other side / "against"
CONTEXT = '#8d968e'        # league average, other teams
GRID = '#ddd8c9'           # gridlines on the cream ground
HOME_BLUE = '#0077b6'      # match charts: home / team A
AWAY_RED = '#e63946'       # match charts: away / team B, negatives

# ---------------------------------------------------------------------------
# The four team radars — SAME colours on Team Analysis, Opposition Report, PDF
# ---------------------------------------------------------------------------
RADAR_OFFENSIVE = '#e60000'
RADAR_DISTRIBUTION = '#0077b6'
RADAR_DEFENSIVE = '#52A736'
RADAR_SET_PIECE = '#ff8c00'
RADAR_COLORS = {
    'Offensive': RADAR_OFFENSIVE,
    'Distribution': RADAR_DISTRIBUTION,
    'Defensive': RADAR_DEFENSIVE,
    'Set Piece': RADAR_SET_PIECE,
}

# Player Comparison: Player A / Player B on the overlaid radar, and the same
# two colours on the page's player cards and stats tables
COMPARE_A = '#0077b6'      # blue
COMPARE_B = '#A67B5B'      # light coffee

# ---------------------------------------------------------------------------
# Colour ramps
# ---------------------------------------------------------------------------
# xG of a shot, anchored at XG_MAX (0.8): the same six stops drive the
# matplotlib shot maps (app.py) and the plotly shot map (pitch_interactive).
XG_MAX = 0.8
XG_COLORS = ['#03045e', '#ade8f4', '#fff3b0', '#ff8c00', '#e63946', '#800f2f']
XG_NODES = [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]   # = xG 0 / .1 / .2 / .4 / .6 / .8 over XG_MAX
XG_COLORSCALE = [[n, c] for n, c in zip(XG_NODES, XG_COLORS)]   # plotly form

# GPA / OBV action value: diverging, negative <- grey -> positive
VALUE_NEG = '#2166ac'
VALUE_MID = '#b8b2a7'
VALUE_POS = '#b2182b'
VALUE_COLORSCALE = [[0.0, VALUE_NEG], [0.5, VALUE_MID], [1.0, VALUE_POS]]

# ---------------------------------------------------------------------------
# Shadow-team scouting grades
# ---------------------------------------------------------------------------
SHADOW_TAG_COLORS = {
    'A - No Brainer': '#2ecc71',
    'B - Possible Starter': '#3498db',
    'C - Quality Depth Squad': '#f1c40f',
    'D - Quality but Injury Prone': '#e74c3c',
    'E - Depth from Lisbon': '#9b59b6',
}

# ---------------------------------------------------------------------------
# Figure conventions
# ---------------------------------------------------------------------------
DPI = 200


# ---------------------------------------------------------------------------
# Metric cards: word values fit whole
# ---------------------------------------------------------------------------
# A st.metric value is bold Source Sans at up to 1.7rem and wraps at word
# breaks (app.py CSS). A row of cards inside a container keyed
# metric_fit_key(...) caps that font at card width / its longest word's
# width in em (one CSS rule per half-em bucket, metric_fit_css), so
# 'Conventional' or 'Mozambique' never breaks mid-letter in a narrow card
# while a row of short values keeps the full size. Letter widths measured
# from the font (2026-10): i / l / t / r 0.28-0.40 em, capitals up to
# 0.68, m / w / M / W / % 0.76-0.86 — each class is rounded up, so the
# estimate runs ~7-12% wide ('Conventional' 6.3 em vs 5.9 measured).
_FIT_NARROW = set("fijlrtI\u00ed.,-\u00b7'\u2019:")
_FIT_WIDE = set("mwMW%")
METRIC_FIT_TENTHS = range(30, 85, 5)     # 3.0 ... 8.0 em, half-em steps


def _word_em(word):
    return sum(0.34 if ch in _FIT_NARROW else 0.86 if ch in _FIT_WIDE
               else 0.68 if ch.isupper() else 0.58 for ch in word)


def metric_fit_key(name, *values):
    """Container key ('mfit<tenths of an em>_<name>') sizing the metric
    values inside it so the longest word among `values` fits its card.
    A line breaks after a hyphen, so 'Guinea-Bissau' counts as two words."""
    words = [w for v in values for w in re.split(r'(?<=-)|\s+', str(v)) if w]
    em = max((_word_em(w) for w in words), default=0.0)
    tenths = next((t for t in METRIC_FIT_TENTHS if em * 10 <= t),
                  METRIC_FIT_TENTHS[-1])
    return f"mfit{tenths}_{name}"


def metric_fit_css():
    """The bucket rules metric_fit_key's keys select (cqi = 1% of the
    card's width; a key's class is 'st-key-<key>')."""
    return "\n".join(
        f'[class*="st-key-mfit{t}_"] [data-testid="stMetric"] '
        f'[data-testid="stMetricValue"] > div '
        f'{{ font-size: clamp(1rem, {1000 / t:.2f}cqi, 1.7rem); }}'
        for t in METRIC_FIT_TENTHS)
