# pitch_visualizations.py
"""Wyscout-style pitch visualizations for the ACP Dashboard.

All functions expect Wyscout-normalised coordinates (0-100 on both axes).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mplsoccer import Pitch, VerticalPitch
from adjustText import adjust_text

# ---------------------------------------------------------------------------
# Pitch setup defaults
# ---------------------------------------------------------------------------
PITCH_COLOR = '#f5f1e9'
LINE_COLOR = 'black'

# Formation coordinates (mirrored from app.py to avoid circular imports)
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


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _check_secondary(secondary_col, tag):
    """Return boolean Series where *tag* is in the type.secondary list."""
    return secondary_col.apply(
        lambda x: tag in x if isinstance(x, (list, np.ndarray, set)) else False
    )


def _make_pitch(figsize=(12, 8)):
    """Standard horizontal Wyscout pitch."""
    pitch = Pitch(pitch_type='wyscout', pitch_color=PITCH_COLOR, line_color=LINE_COLOR)
    fig, ax = pitch.draw(figsize=figsize)
    return pitch, fig, ax


def _add_attack_direction_arrow(ax):
    """Draw a small arrow + label showing the attacking direction (left → right)."""
    ax.annotate('', xy=(30, 104), xytext=(12, 104),
                xycoords='data', textcoords='data',
                arrowprops=dict(arrowstyle='->', color='#333333',
                                lw=2.0, mutation_scale=15),
                annotation_clip=False, zorder=10)
    ax.text(21, 107, 'Direction of Attack', ha='center', va='bottom',
            fontsize=8, color='#333333', fontstyle='italic',
            clip_on=False, zorder=10)


def _short_name(full_name):
    """Shorten to last name."""
    if pd.isna(full_name):
        return ''
    parts = str(full_name).split()
    return parts[-1][:12] if parts else ''


# =========================================================================
# 1. Average Player Positions
# =========================================================================
def plot_average_positions(events_df, team_name, title=None, player_names=None,
                           match_lineup=None):
    """Average x, y per player on a pitch outline, with sub timing for single matches.

    Parameters
    ----------
    player_names : set or list, optional
        If provided, only show these players (e.g. projected XI names).
    match_lineup : dict, optional
        Lineup data from Wyscout API: {'lineup': [...], 'substitutions': [...]}
        Used for exact substitution minute labeling.
    """
    te = events_df[events_df['team.name'] == team_name].copy()
    te['location.x'] = pd.to_numeric(te['location.x'], errors='coerce')
    te['location.y'] = pd.to_numeric(te['location.y'], errors='coerce')
    te['minute'] = pd.to_numeric(te.get('minute'), errors='coerce')
    te = te.dropna(subset=['location.x', 'location.y', 'player.name'])

    avg = te.groupby('player.name').agg(
        x=('location.x', 'mean'),
        y=('location.y', 'mean'),
        count=('location.x', 'size'),
        first_min=('minute', 'min'),
        last_min=('minute', 'max'),
    ).reset_index()

    # Build player_id -> name map and name -> player_id map
    id_to_name = {}
    name_to_id = {}
    if 'player.id' in te.columns:
        for _, row in te[['player.id', 'player.name']].dropna().drop_duplicates().iterrows():
            try:
                pid = int(float(row['player.id']))
                id_to_name[pid] = row['player.name']
                name_to_id[row['player.name']] = pid
            except (ValueError, TypeError):
                pass

    # Build exact sub timing from API data
    sub_in_map = {}   # player_id -> minute subbed on
    sub_off_map = {}  # player_id -> minute subbed off
    starter_ids = set()
    has_lineup = (match_lineup is not None and
                  'lineup' in match_lineup and
                  len(match_lineup.get('lineup', [])) > 0)
    if has_lineup:
        for p in match_lineup['lineup']:
            pid = p.get('playerId')
            if pid:
                starter_ids.add(pid)
        for sub in match_lineup.get('substitutions', []):
            sub_min = sub.get('minute', 0)
            p_out = sub.get('playerOut')
            p_in = sub.get('playerIn')
            if p_out:
                sub_off_map[p_out] = sub_min
            if p_in:
                sub_in_map[p_in] = sub_min

    if player_names:
        avg = avg[avg['player.name'].isin(player_names)]
    else:
        avg = avg.nlargest(14, 'count')

    is_single_match = te['matchId'].nunique() == 1
    match_max = te['minute'].max() if is_single_match else 0

    pitch, fig, ax = _make_pitch()

    for _, row in avg.iterrows():
        pid = name_to_id.get(row['player.name'])
        if has_lineup and pid:
            is_sub = pid not in starter_ids
        else:
            is_sub = is_single_match and row['first_min'] > 5
        color = '#5a9a6a' if is_sub else '#1a472a'
        pitch.scatter(row['x'], row['y'], s=300, color=color,
                      edgecolors='white', linewidth=2, zorder=5, ax=ax)

    texts = []
    for _, row in avg.iterrows():
        pid = name_to_id.get(row['player.name'])
        if has_lineup and pid:
            is_sub = pid not in starter_ids
        else:
            is_sub = is_single_match and row['first_min'] > 5
        color = '#5a9a6a' if is_sub else '#1a472a'

        label = _short_name(row['player.name'])
        if is_single_match:
            # Use exact sub minutes from API when available
            if has_lineup and pid:
                if pid in sub_in_map:
                    label += f" In {sub_in_map[pid]}'"
                if pid in sub_off_map:
                    label += f" Off {sub_off_map[pid]}'"
            else:
                if row['first_min'] > 5:
                    label += f" In {int(row['first_min'])}'"
                if row['last_min'] < match_max - 5:
                    label += f" Off {int(row['last_min'])}'"

        t = ax.text(row['x'], row['y'] - 3.5, label,
                    ha='center', va='top', fontsize=8, fontweight='bold',
                    color=color, zorder=6,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none',
                              pad=1, boxstyle='round,pad=0.2'))
        texts.append(t)

    try:
        adjust_text(texts, ax=ax,
                    only_move={'text': 'xy'},
                    lim=50,
                    force_text=(0.3, 0.5),
                    force_static=(0.5, 0.8),
                    expand=(1.2, 1.4))
    except Exception:
        pass

    ax.set_title(title or f'{team_name} — Average Positions',
                 fontsize=13, fontweight='bold', pad=8)
    fig.tight_layout()
    return fig


# =========================================================================
# 2. Formation Timeline (formation per time segment)
# =========================================================================
POSITION_EQUIVALENTS = {
    'GK': ['GK'],
    'LB': ['LB', 'LB5', 'LWB'], 'RB': ['RB', 'RB5', 'RWB'],
    'LCB': ['LCB', 'LCB3', 'CB'], 'RCB': ['RCB', 'RCB3', 'CB'],
    'CB': ['CB', 'LCB3', 'RCB3', 'LCB', 'RCB'],
    'LWB': ['LWB', 'LB5', 'LB', 'LWF'], 'RWB': ['RWB', 'RB5', 'RB', 'RWF'],
    'LDM': ['LDMF', 'DMF', 'LCMF'], 'RDM': ['RDMF', 'DMF', 'RCMF'],
    'CDM': ['DMF', 'LDMF', 'RDMF', 'LCMF', 'RCMF'],
    'LCM': ['LCMF', 'LCMF3', 'CMF', 'LDMF', 'DMF', 'AMF'],
    'RCM': ['RCMF', 'RCMF3', 'CMF', 'RDMF', 'DMF', 'AMF'],
    'LM': ['LWF', 'LW', 'LAMF', 'LCMF'], 'RM': ['RWF', 'RW', 'RAMF', 'RCMF'],
    'LAM': ['LAMF', 'AMF', 'LWF', 'LW'], 'RAM': ['RAMF', 'AMF', 'RWF', 'RW'],
    'CAM': ['AMF', 'LAMF', 'RAMF', 'SS'],
    'LW': ['LW', 'LWF', 'LAMF'], 'RW': ['RW', 'RWF', 'RAMF'],
    'CF': ['CF', 'LCF', 'RCF', 'SS'], 'LST': ['LCF', 'CF', 'SS'],
    'RST': ['RCF', 'CF', 'SS'],
    'SS': ['SS', 'CF', 'AMF'],
}


def _map_players_to_slots(seg_events, formation_slots):
    """Map players from segment events to formation slots using position equivalents.

    Returns dict {slot: player_name} with short names.
    """
    if 'player.position' not in seg_events.columns:
        return {slot: slot for slot in formation_slots}

    # Build position -> best player mapping from segment events
    pos_players = {}
    for pos in seg_events['player.position'].dropna().unique():
        candidates = seg_events[seg_events['player.position'] == pos]
        if not candidates.empty:
            best = candidates['player.name'].value_counts().index[0]
            pos_players[pos] = best

    mapping = {}
    used_players = set()
    for slot in formation_slots:
        for pos in POSITION_EQUIVALENTS.get(slot, [slot]):
            if pos in pos_players and pos_players[pos] not in used_players:
                mapping[slot] = _short_name(pos_players[pos])
                used_players.add(pos_players[pos])
                break

    # Second pass: assign any remaining unmatched players to unfilled slots
    unfilled = [s for s in formation_slots if s not in mapping]
    remaining = [name for name in pos_players.values() if name not in used_players]
    for slot, name in zip(unfilled, remaining):
        mapping[slot] = _short_name(name)
        used_players.add(name)

    # Any still-unfilled slots get the slot abbreviation
    for slot in formation_slots:
        if slot not in mapping:
            mapping[slot] = slot

    return mapping


def _build_active_players(lineup_data, substitutions, seg_start, seg_end, id_to_name):
    """Given API lineup data and substitution list, return the set of player IDs
    that were on the pitch during the segment [seg_start, seg_end].

    Parameters
    ----------
    lineup_data : list of dicts with 'playerId' keys (starting XI from API)
    substitutions : list of dicts with 'minute', 'playerIn', 'playerOut'
    seg_start, seg_end : float, minute boundaries of the segment
    id_to_name : dict {player_id: player_name}

    Returns set of player IDs active during this segment.
    """
    # Start with starting XI
    on_pitch = set()
    for p in lineup_data:
        pid = p.get('playerId')
        if pid:
            on_pitch.add(pid)

    # Sort substitutions by minute
    subs_sorted = sorted(substitutions, key=lambda s: s.get('minute', 999))

    # Apply all subs that happen BEFORE or AT the segment start
    for sub in subs_sorted:
        sub_min = sub.get('minute', 999)
        if sub_min <= seg_start:
            p_out = sub.get('playerOut')
            p_in = sub.get('playerIn')
            if p_out:
                on_pitch.discard(p_out)
            if p_in:
                on_pitch.add(p_in)
        elif sub_min <= seg_end:
            # Sub happens during the segment — include both players
            p_in = sub.get('playerIn')
            if p_in:
                on_pitch.add(p_in)
        else:
            break

    return on_pitch


def _map_active_players_to_slots(active_ids, formation_slots, events_in_seg, id_to_name):
    """Map a known set of active player IDs to formation slots using their
    positions from the events data.

    Returns dict {slot: short_name}.
    """
    if events_in_seg.empty or 'player.position' not in events_in_seg.columns:
        return {slot: slot for slot in formation_slots}

    # Build player_id -> most common position from events in this segment
    pid_col = events_in_seg['player.id'].dropna()
    if pid_col.empty:
        return {slot: slot for slot in formation_slots}

    events_in_seg = events_in_seg.copy()
    events_in_seg['_pid'] = pd.to_numeric(events_in_seg['player.id'], errors='coerce')

    pid_pos = {}
    for pid in active_ids:
        player_events = events_in_seg[events_in_seg['_pid'] == pid]
        if not player_events.empty:
            positions = player_events['player.position'].dropna()
            if not positions.empty:
                pid_pos[pid] = positions.value_counts().index[0]

    mapping = {}
    used = set()
    for slot in formation_slots:
        for equiv_pos in POSITION_EQUIVALENTS.get(slot, [slot]):
            for pid, pos in pid_pos.items():
                if pos == equiv_pos and pid not in used:
                    name = id_to_name.get(pid, f'ID:{pid}')
                    mapping[slot] = _short_name(name)
                    used.add(pid)
                    break
            if slot in mapping:
                break

    # Second pass: assign unmatched active players to unfilled slots
    unfilled = [s for s in formation_slots if s not in mapping]
    remaining = [pid for pid in active_ids if pid not in used and pid in id_to_name]
    for slot, pid in zip(unfilled, remaining):
        mapping[slot] = _short_name(id_to_name[pid])
        used.add(pid)

    for slot in formation_slots:
        if slot not in mapping:
            mapping[slot] = slot

    return mapping


def plot_formation_timeline(events_df, team_name, title=None, match_lineup=None):
    """Small pitch subplots showing formation per time segment with player names.

    Segments split at BOTH formation changes AND substitution minutes so each
    panel shows the actual XI on the pitch at that time.

    Parameters
    ----------
    match_lineup : dict, optional
        Lineup data for this team from the Wyscout API match endpoint:
        {'lineup': [...], 'bench': [...], 'substitutions': [...]}
        If provided, uses exact substitution data for accurate player mapping.
    """
    te = events_df[
        (events_df['team.name'] == team_name) &
        (events_df['team.formation'].notna())
    ].sort_values('minute')

    if te.empty:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, 'No formation data', ha='center', va='center')
        ax.axis('off')
        return fig

    has_lineup = (match_lineup is not None and
                  'lineup' in match_lineup and
                  len(match_lineup.get('lineup', [])) > 0)

    # Build formation-change segments first
    formation_segments = []
    current_formation = None
    start_min = 0
    for _, row in te.iterrows():
        f = row['team.formation']
        m = row['minute']
        if f != current_formation:
            if current_formation is not None:
                formation_segments.append((current_formation, start_min, m))
            current_formation = f
            start_min = m
    if current_formation is not None:
        formation_segments.append((current_formation, start_min, te['minute'].max()))

    if not formation_segments:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, 'No formation data', ha='center', va='center')
        ax.axis('off')
        return fig

    # Collect substitution minutes to use as additional split points
    sub_minutes = set()
    if has_lineup:
        for sub in match_lineup.get('substitutions', []):
            sm = sub.get('minute')
            if sm is not None:
                sub_minutes.add(sm)

    # Split formation segments at substitution minutes
    segments = []  # (formation, start, end, subs_in_this_seg)
    for formation, seg_s, seg_e in formation_segments:
        # Find sub minutes that fall within this segment
        splits = sorted([m for m in sub_minutes if seg_s < m < seg_e])
        boundaries = [seg_s] + splits + [seg_e]
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            # Collect subs that happen at this boundary
            subs_at_boundary = []
            if has_lineup:
                for sub in match_lineup.get('substitutions', []):
                    if sub.get('minute') == s and s != seg_s:
                        subs_at_boundary.append(sub)
            segments.append((formation, s, e, subs_at_boundary))

    # Merge consecutive segments with same formation AND same players to avoid
    # too many panels. Only keep a split if the XI actually changes.
    # Limit to 6 segments max for readability
    segments = segments[:6]
    n = len(segments)
    if n == 0:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, 'No formation data', ha='center', va='center')
        ax.axis('off')
        return fig

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 6))
    if n == 1:
        axes = [axes]

    # Get ALL team events for player mapping
    all_team_events = events_df[events_df['team.name'] == team_name].copy()

    # Build player ID -> name mapping from events
    id_to_name = {}
    if 'player.id' in all_team_events.columns:
        for _, row in all_team_events[['player.id', 'player.name']].dropna().iterrows():
            try:
                pid = int(float(row['player.id']))
                id_to_name[pid] = row['player.name']
            except (ValueError, TypeError):
                pass

    for idx, (formation, smin, emin, subs_here) in enumerate(segments):
        ax = axes[idx]
        vpitch = VerticalPitch(pitch_type='opta', pitch_color='#1a472a',
                               line_color='white', linewidth=1, goal_type='box')
        vpitch.draw(ax=ax)

        fkey = formation if formation in FORMATION_COORDS else '4-4-2'
        fdata = FORMATION_COORDS[fkey]

        seg_events = all_team_events[
            (all_team_events['minute'] >= smin) &
            (all_team_events['minute'] <= emin)
        ]

        if has_lineup:
            # Use exact substitution data — get who's on pitch AFTER subs at smin
            active_ids = _build_active_players(
                match_lineup['lineup'],
                match_lineup.get('substitutions', []),
                smin, emin, id_to_name
            )
            slot_names = _map_active_players_to_slots(
                active_ids, fdata['positions'], seg_events, id_to_name
            )
        else:
            slot_names = _map_players_to_slots(seg_events, fdata['positions'])

        # Build sub annotations for this segment's boundary subs
        sub_in_names = set()
        sub_off_names = set()
        for sub in subs_here:
            p_in = sub.get('playerIn')
            p_out = sub.get('playerOut')
            if p_in and p_in in id_to_name:
                sub_in_names.add(_short_name(id_to_name[p_in]))
            if p_out and p_out in id_to_name:
                sub_off_names.add(_short_name(id_to_name[p_out]))

        for slot, coords in zip(fdata['positions'], fdata['coords']):
            x, y = coords
            label = slot_names.get(slot, slot)

            # Color nodes for players who just came on
            if label in sub_in_names:
                node_color = '#90EE90'
            else:
                node_color = 'white'

            ax.scatter(x, y, s=500, c=node_color, edgecolors='#1a472a',
                       linewidth=2, zorder=5)

            ax.text(x, y - 5, label, ha='center', va='top', fontsize=6,
                    fontweight='bold', color='white')

        # Build title with sub info
        sub_info = ""
        if subs_here:
            sub_parts = []
            for sub in subs_here:
                p_in = sub.get('playerIn')
                p_out = sub.get('playerOut')
                in_name = _short_name(id_to_name.get(p_in, '')) if p_in else ''
                out_name = _short_name(id_to_name.get(p_out, '')) if p_out else ''
                if in_name and out_name:
                    sub_parts.append(f"{in_name} for {out_name}")
            if sub_parts:
                sub_info = "\n" + ", ".join(sub_parts)

        ax.set_title(f"{formation}\n{int(smin)}'-{int(emin)}'{sub_info}",
                     fontsize=9, fontweight='bold', color='white', pad=5)
        ax.set_facecolor('#1a472a')

    fig.patch.set_facecolor('#1a472a')
    fig.suptitle(title or f'{team_name} — Formation Timeline',
                 fontsize=13, fontweight='bold', color='white', y=1.02)
    fig.tight_layout()
    return fig


# =========================================================================
# 2b. Average Positions by Substitution Windows
# =========================================================================
def plot_avg_positions_by_subs(events_df, team_name, title=None, match_lineup=None):
    """Side-by-side average-position pitches, one per substitution window.

    Segments split at unique substitution minutes (grouping subs at the same
    minute together).  Falls back to a single panel when no lineup data exists.

    Parameters
    ----------
    match_lineup : dict, optional
        {'lineup': [...], 'bench': [...], 'substitutions': [...]}
    """
    te = events_df[events_df['team.name'] == team_name].copy()
    te['location.x'] = pd.to_numeric(te['location.x'], errors='coerce')
    te['location.y'] = pd.to_numeric(te['location.y'], errors='coerce')
    te['minute'] = pd.to_numeric(te.get('minute'), errors='coerce')
    te = te.dropna(subset=['location.x', 'location.y', 'player.name'])

    if te.empty:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, 'No event data', ha='center', va='center')
        ax.axis('off')
        return fig

    # Build player ID <-> name maps
    id_to_name = {}
    name_to_id = {}
    if 'player.id' in te.columns:
        for _, row in te[['player.id', 'player.name']].dropna().drop_duplicates().iterrows():
            try:
                pid = int(float(row['player.id']))
                id_to_name[pid] = row['player.name']
                name_to_id[row['player.name']] = pid
            except (ValueError, TypeError):
                pass

    has_lineup = (match_lineup is not None and
                  'lineup' in match_lineup and
                  len(match_lineup.get('lineup', [])) > 0)

    match_end = te['minute'].max()

    # Build segments from unique sub minutes
    if has_lineup:
        sub_minutes = sorted(set(
            s.get('minute') for s in match_lineup.get('substitutions', [])
            if s.get('minute') is not None
        ))
    else:
        sub_minutes = []

    boundaries = [0] + sub_minutes + [match_end]
    # Remove duplicates and sort
    boundaries = sorted(set(boundaries))

    # Build segments: list of (start, end, subs_at_start)
    segments = []
    all_subs = match_lineup.get('substitutions', []) if has_lineup else []
    for i in range(len(boundaries) - 1):
        seg_s, seg_e = boundaries[i], boundaries[i + 1]
        if seg_e <= seg_s:
            continue
        # Subs that triggered this segment
        subs_here = [s for s in all_subs if s.get('minute') == seg_s and seg_s > 0]
        segments.append((seg_s, seg_e, subs_here))

    if not segments:
        segments = [(0, match_end, [])]

    # Cap at 5 panels
    segments = segments[:5]
    n = len(segments)

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 10))
    if n == 1:
        axes = [axes]

    for idx, (smin, emin, subs_here) in enumerate(segments):
        pitch = Pitch(pitch_type='wyscout', pitch_color=PITCH_COLOR,
                      line_color=LINE_COLOR)
        pitch.draw(ax=axes[idx])
        ax = axes[idx]

        # Filter events to this time window
        seg_events = te[(te['minute'] >= smin) & (te['minute'] < emin)]
        if seg_events.empty:
            seg_events = te[(te['minute'] >= smin) & (te['minute'] <= emin)]

        # Determine who was on pitch during this segment
        if has_lineup:
            active_ids = _build_active_players(
                match_lineup['lineup'],
                all_subs,
                smin, emin, id_to_name,
            )
            active_names = {id_to_name.get(pid) for pid in active_ids
                           if pid in id_to_name}
            # Filter events to active players only
            seg_events = seg_events[seg_events['player.name'].isin(active_names)]

        # Compute average positions
        avg = seg_events.groupby('player.name').agg(
            x=('location.x', 'mean'),
            y=('location.y', 'mean'),
            count=('location.x', 'size'),
        ).reset_index()

        # Keep top 11 by event count
        avg = avg.nlargest(11, 'count')

        # Identify subs coming in at this window
        sub_in_names = set()
        for sub in subs_here:
            p_in = sub.get('playerIn')
            if p_in and p_in in id_to_name:
                sub_in_names.add(id_to_name[p_in])

        # Draw dots
        for _, row in avg.iterrows():
            is_new_sub = row['player.name'] in sub_in_names
            color = '#5a9a6a' if is_new_sub else '#1a472a'
            pitch.scatter(row['x'], row['y'], s=300, color=color,
                          edgecolors='white', linewidth=2, zorder=5, ax=ax)

        # Draw labels
        texts = []
        for _, row in avg.iterrows():
            is_new_sub = row['player.name'] in sub_in_names
            color = '#5a9a6a' if is_new_sub else '#1a472a'
            label = _short_name(row['player.name'])
            t = ax.text(row['x'], row['y'] - 3.5, label,
                        ha='center', va='top', fontsize=8, fontweight='bold',
                        color=color, zorder=6,
                        bbox=dict(facecolor='white', alpha=0.8,
                                  edgecolor='none', pad=1,
                                  boxstyle='round,pad=0.2'))
            texts.append(t)

        try:
            adjust_text(texts, ax=ax,
                        only_move={'text': 'xy'}, lim=50,
                        force_text=(0.3, 0.5), force_static=(0.5, 0.8),
                        expand=(1.2, 1.4))
        except Exception:
            pass

        # Build title
        sub_info = ""
        if subs_here:
            parts = []
            for sub in subs_here:
                p_in = sub.get('playerIn')
                p_out = sub.get('playerOut')
                in_n = _short_name(id_to_name.get(p_in, '')) if p_in else ''
                out_n = _short_name(id_to_name.get(p_out, '')) if p_out else ''
                if in_n and out_n:
                    parts.append(f"{in_n} for {out_n}")
            if parts:
                sub_info = "\n" + ", ".join(parts)

        ax.set_title(f"{int(smin)}' – {int(emin)}'{sub_info}",
                     fontsize=10, fontweight='bold', pad=8)

    fig.suptitle(title or f'{team_name} — Average Positions by Phase',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig


# =========================================================================
# 3. Passing Network
# =========================================================================
def plot_passing_network(events_df, team_name, title=None):
    """Pass connections between players (using possession chain)."""
    te = events_df[events_df['team.name'] == team_name].copy()
    passes = te[te['type.primary'] == 'pass'].copy()

    passes['location.x'] = pd.to_numeric(passes['location.x'], errors='coerce')
    passes['location.y'] = pd.to_numeric(passes['location.y'], errors='coerce')

    # Detect receiver via possession chain
    passes['possession.eventIndex'] = pd.to_numeric(
        passes['possession.eventIndex'], errors='coerce')
    passes = passes.dropna(subset=['player.id', 'possession.id',
                                    'possession.eventIndex',
                                    'location.x', 'location.y'])

    # Build next-event mapping within possessions
    te_sorted = te.dropna(subset=['player.id', 'possession.id',
                                   'possession.eventIndex']).copy()
    te_sorted['possession.eventIndex'] = pd.to_numeric(
        te_sorted['possession.eventIndex'], errors='coerce')

    # Map: for each pass, the next event in the same possession has the receiver
    receiver_lookup = te_sorted[['possession.id', 'possession.eventIndex',
                                  'player.id']].copy()
    receiver_lookup.columns = ['possession.id', 'recv_event_idx', 'receiver_id']

    passes = passes.copy()
    passes['_next_idx'] = passes['possession.eventIndex'] + 1
    passes = passes.merge(
        receiver_lookup,
        left_on=['possession.id', '_next_idx'],
        right_on=['possession.id', 'recv_event_idx'],
        how='left',
    )

    passes['passer'] = passes['player.id'].astype(int)
    passes = passes.dropna(subset=['receiver_id'])
    passes['receiver'] = passes['receiver_id'].astype(int)

    # Average positions
    player_pos = te.dropna(subset=['player.id', 'location.x', 'location.y']).copy()
    player_pos['player.id'] = player_pos['player.id'].astype(int)
    avg_pos = player_pos.groupby('player.id').agg(
        x=('location.x', 'mean'),
        y=('location.y', 'mean'),
        count=('location.x', 'size'),
        name=('player.name', 'first'),
        position=('player.position', lambda s: s.mode().iloc[0] if not s.mode().empty else ''),
    ).reset_index()

    # Count passes per player to ensure we only include players who actually passed
    passer_counts = passes.groupby('passer').size()

    # Ensure exactly 1 GK is included (the one with most events)
    gk_all = avg_pos[avg_pos['position'] == 'GK']
    if not gk_all.empty:
        gk_row = gk_all.nlargest(1, 'count')
        gk_ids = set(gk_row['player.id'])
    else:
        gk_ids = set()

    # Filter to players who made at least 1 pass (or are the primary GK)
    avg_pos = avg_pos[
        (avg_pos['player.id'].isin(passer_counts.index)) |
        (avg_pos['player.id'].isin(gk_ids))
    ]

    # Top 11 by involvement (primary GK always kept, max 1)
    non_gk = avg_pos[~avg_pos['player.id'].isin(gk_ids)].nlargest(10, 'count')
    gk_rows = avg_pos[avg_pos['player.id'].isin(gk_ids)]
    avg_pos = pd.concat([gk_rows, non_gk]).head(11)
    top_ids = set(avg_pos['player.id'])

    passes = passes[passes['passer'].isin(top_ids) & passes['receiver'].isin(top_ids)]
    pair_counts = passes.groupby(['passer', 'receiver']).size().reset_index(name='count')

    # Minimum threshold
    min_passes = max(pair_counts['count'].quantile(0.3), 2) if not pair_counts.empty else 2
    pair_counts = pair_counts[pair_counts['count'] >= min_passes]

    # Count total passes per player (as passer)
    player_pass_counts = passes.groupby('passer').size().to_dict()

    pitch, fig, ax = _make_pitch()

    # Lines
    pos_dict = {int(r['player.id']): (r['x'], r['y'])
                for _, r in avg_pos.iterrows()}
    max_cnt = pair_counts['count'].max() if not pair_counts.empty else 1

    for _, row in pair_counts.iterrows():
        p = int(row['passer'])
        r = int(row['receiver'])
        if p in pos_dict and r in pos_dict:
            sx, sy = pos_dict[p]
            ex, ey = pos_dict[r]
            lw = 0.5 + 4 * (row['count'] / max_cnt)
            ax.plot([sx, ex], [sy, ey], color='#457b9d', linewidth=lw,
                    alpha=0.6, zorder=2)

    # Nodes
    max_involvement = avg_pos['count'].max() if not avg_pos.empty else 1
    sizes = 100 + 500 * (avg_pos['count'] / max_involvement)
    pitch.scatter(avg_pos['x'], avg_pos['y'], s=sizes,
                  color='#1d3557', edgecolors='white', linewidth=2,
                  zorder=5, ax=ax)

    # Pass count inside each node
    for _, row in avg_pos.iterrows():
        pid = int(row['player.id'])
        pc = player_pass_counts.get(pid, 0)
        ax.text(row['x'], row['y'], str(pc), ha='center', va='center',
                fontsize=7, fontweight='bold', color='white', zorder=6)

    # Player name labels below nodes — use ax.text so adjust_text works properly
    texts = []
    for _, row in avg_pos.iterrows():
        t = ax.text(row['x'], row['y'] - 3.5, _short_name(row['name']),
                    ha='center', va='top', fontsize=8, fontweight='bold',
                    color='#1d3557', zorder=6,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none',
                              pad=1, boxstyle='round,pad=0.2'))
        texts.append(t)

    try:
        adjust_text(texts, ax=ax,
                    only_move={'text': 'xy'},
                    lim=50,
                    force_text=(0.3, 0.5),
                    force_static=(0.5, 0.8),
                    expand=(1.2, 1.4))
    except Exception:
        pass

    ax.set_title(title or f'{team_name} — Passing Network',
                 fontsize=13, fontweight='bold', pad=8)
    fig.tight_layout()
    return fig


# =========================================================================
# 4. Ball Recovery Map
# =========================================================================
def plot_recovery_map(events_df, team_name, title=None):
    """Scatter of ball recoveries on the pitch."""
    te = events_df[events_df['team.name'] == team_name].copy()
    is_recovery = _check_secondary(te.get('type.secondary', pd.Series(dtype='object')),
                                   'recovery')
    recoveries = te[is_recovery].copy()
    recoveries['location.x'] = pd.to_numeric(recoveries['location.x'], errors='coerce')
    recoveries['location.y'] = pd.to_numeric(recoveries['location.y'], errors='coerce')
    recoveries = recoveries.dropna(subset=['location.x', 'location.y'])

    pitch, fig, ax = _make_pitch()

    if not recoveries.empty:
        pitch.kdeplot(recoveries['location.x'], recoveries['location.y'],
                      ax=ax, fill=True, cmap='Greens', alpha=0.5, levels=50,
                      zorder=1)
        pitch.scatter(recoveries['location.x'], recoveries['location.y'],
                      s=25, color='#2d6a4f', edgecolors='white', linewidth=0.5,
                      zorder=3, ax=ax, alpha=0.7)

    _add_attack_direction_arrow(ax)
    ax.set_title(title or f'{team_name} — Ball Recoveries ({len(recoveries)})',
                 fontsize=13, fontweight='bold', pad=8)
    fig.tight_layout()
    return fig


# =========================================================================
# 5. Ball Loss Map
# =========================================================================
def plot_loss_map(events_df, team_name, title=None):
    """Scatter of ball losses on the pitch."""
    te = events_df[events_df['team.name'] == team_name].copy()
    is_loss = _check_secondary(te.get('type.secondary', pd.Series(dtype='object')),
                               'loss')
    losses = te[is_loss].copy()
    losses['location.x'] = pd.to_numeric(losses['location.x'], errors='coerce')
    losses['location.y'] = pd.to_numeric(losses['location.y'], errors='coerce')
    losses = losses.dropna(subset=['location.x', 'location.y'])

    pitch, fig, ax = _make_pitch()

    if not losses.empty:
        pitch.kdeplot(losses['location.x'], losses['location.y'],
                      ax=ax, fill=True, cmap='Reds', alpha=0.5, levels=50,
                      zorder=1)
        pitch.scatter(losses['location.x'], losses['location.y'],
                      s=25, color='#9d0208', edgecolors='white', linewidth=0.5,
                      zorder=3, ax=ax, alpha=0.7)

    _add_attack_direction_arrow(ax)
    ax.set_title(title or f'{team_name} — Ball Losses ({len(losses)})',
                 fontsize=13, fontweight='bold', pad=8)
    fig.tight_layout()
    return fig


# =========================================================================
# 6. Defensive Duels Map
# =========================================================================
def plot_defensive_duels_map(events_df, team_name, title=None):
    """Won (green) and lost (red) defensive duels on pitch."""
    te = events_df[events_df['team.name'] == team_name].copy()
    is_def_duel = _check_secondary(
        te.get('type.secondary', pd.Series(dtype='object')), 'defensive_duel')
    duels = te[is_def_duel].copy()

    duels['location.x'] = pd.to_numeric(duels['location.x'], errors='coerce')
    duels['location.y'] = pd.to_numeric(duels['location.y'], errors='coerce')
    duels = duels.dropna(subset=['location.x', 'location.y'])

    won = duels[
        (duels.get('groundDuel.recoveredPossession') == True) |
        (duels.get('groundDuel.stoppedProgress') == True)
    ]
    lost = duels[~duels.index.isin(won.index)]

    pitch, fig, ax = _make_pitch()

    if not lost.empty:
        pitch.scatter(lost['location.x'], lost['location.y'],
                      s=50, color='#e63946', edgecolors='white', linewidth=0.5,
                      zorder=3, ax=ax, alpha=0.7, label=f'Lost ({len(lost)})')
    if not won.empty:
        pitch.scatter(won['location.x'], won['location.y'],
                      s=50, color='#2a9d8f', edgecolors='white', linewidth=0.5,
                      zorder=4, ax=ax, alpha=0.7, label=f'Won ({len(won)})')

    _add_attack_direction_arrow(ax)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title(title or f'{team_name} — Defensive Duels',
                 fontsize=13, fontweight='bold', pad=8)
    fig.tight_layout()
    return fig


# =========================================================================
# 7. Zone Heatmap (4x3 grid, Benfica page-15 style)
# =========================================================================
def plot_zone_heatmap(events_df, team_name, tag, league_events_df=None,
                      title=None):
    """Grid-based zone map colored green/orange vs league average.

    Parameters
    ----------
    tag : str
        Secondary tag to count, e.g. 'recovery' or 'loss'.
    league_events_df : DataFrame, optional
        Full league events for computing league averages.
        If None, only team percentages are shown.
    """
    x_bounds = [0, 25, 50, 75, 100]
    y_bounds = [0, 33.33, 66.67, 100]

    def _zone_pcts(df, team):
        te = df[df['team.name'] == team].copy()
        secondary_col = te.get('type.secondary', pd.Series(dtype='object'))
        mask = _check_secondary(secondary_col, tag)
        ev = te[mask].copy()
        ev['location.x'] = pd.to_numeric(ev['location.x'], errors='coerce')
        ev['location.y'] = pd.to_numeric(ev['location.y'], errors='coerce')
        ev = ev.dropna(subset=['location.x', 'location.y'])

        total = len(ev)
        grid = np.zeros((len(x_bounds) - 1, len(y_bounds) - 1))
        for i in range(len(x_bounds) - 1):
            for j in range(len(y_bounds) - 1):
                cnt = ev[
                    (ev['location.x'] >= x_bounds[i]) &
                    (ev['location.x'] < x_bounds[i + 1]) &
                    (ev['location.y'] >= y_bounds[j]) &
                    (ev['location.y'] < y_bounds[j + 1])
                ].shape[0]
                grid[i, j] = (cnt / total * 100) if total > 0 else 0
        return grid

    team_grid = _zone_pcts(events_df, team_name)

    # League average
    league_grid = None
    if league_events_df is not None:
        teams = league_events_df['team.name'].unique()
        all_grids = []
        for t in teams:
            g = _zone_pcts(league_events_df, t)
            all_grids.append(g)
        if all_grids:
            league_grid = np.mean(all_grids, axis=0)

    pitch, fig, ax = _make_pitch()

    for i in range(len(x_bounds) - 1):
        for j in range(len(y_bounds) - 1):
            x0, x1 = x_bounds[i], x_bounds[i + 1]
            y0, y1 = y_bounds[j], y_bounds[j + 1]

            pct = team_grid[i, j]
            diff = 0
            if league_grid is not None:
                diff = pct - league_grid[i, j]

            # Color: green above avg, orange below
            if diff > 0:
                alpha = min(0.2 + abs(diff) / 15, 0.7)
                color = mcolors.to_rgba('#2a9d8f', alpha)
            elif diff < 0:
                alpha = min(0.2 + abs(diff) / 15, 0.7)
                color = mcolors.to_rgba('#e76f51', alpha)
            else:
                color = mcolors.to_rgba('#cccccc', 0.15)

            rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                 facecolor=color, edgecolor='grey',
                                 linewidth=0.5, zorder=2)
            ax.add_patch(rect)

            # Text
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(cx, cy, f'{pct:.1f}%', ha='center', va='center',
                    fontsize=11, fontweight='bold', color='#222', zorder=3)
            if league_grid is not None:
                sign = '+' if diff > 0 else ''
                ax.text(cx, cy + 5, f'({sign}{diff:.1f}%)', ha='center',
                        va='center', fontsize=8, color='#555', zorder=3)

    label = tag.replace('_', ' ').title()
    ax.set_title(title or f'{team_name} — {label} Zones vs League Avg',
                 fontsize=13, fontweight='bold', pad=8)
    fig.tight_layout()
    return fig


# =========================================================================
# 8. Through Passes Map
# =========================================================================
def plot_through_passes(events_df, team_name, title=None):
    """Arrows showing through passes on the pitch."""
    te = events_df[events_df['team.name'] == team_name].copy()
    is_through = _check_secondary(
        te.get('type.secondary', pd.Series(dtype='object')), 'through_pass')
    tp = te[(te['type.primary'] == 'pass') & is_through].copy()

    for col in ['location.x', 'location.y', 'pass.endLocation.x', 'pass.endLocation.y']:
        tp[col] = pd.to_numeric(tp[col], errors='coerce')
    tp = tp.dropna(subset=['location.x', 'location.y',
                           'pass.endLocation.x', 'pass.endLocation.y'])

    pitch, fig, ax = _make_pitch()

    if not tp.empty:
        accurate = tp[tp.get('pass.accurate') == True]
        inaccurate = tp[tp.get('pass.accurate') != True]

        if not inaccurate.empty:
            pitch.arrows(inaccurate['location.x'], inaccurate['location.y'],
                         inaccurate['pass.endLocation.x'],
                         inaccurate['pass.endLocation.y'],
                         width=2, headwidth=5, headlength=4,
                         color='#adb5bd', alpha=0.6, zorder=2, ax=ax,
                         label=f'Inaccurate ({len(inaccurate)})')

        if not accurate.empty:
            pitch.arrows(accurate['location.x'], accurate['location.y'],
                         accurate['pass.endLocation.x'],
                         accurate['pass.endLocation.y'],
                         width=2, headwidth=5, headlength=4,
                         color='#e63946', alpha=0.8, zorder=3, ax=ax,
                         label=f'Accurate ({len(accurate)})')

    ax.legend(loc='upper left', fontsize=9)
    ax.set_title(title or f'{team_name} — Through Passes ({len(tp)})',
                 fontsize=13, fontweight='bold', pad=8)
    fig.tight_layout()
    return fig


# =========================================================================
# 9. Successful Dribbles in Final Third
# =========================================================================
def plot_final_third_dribbles(events_df, team_name, title=None):
    """Successful dribbles in the attacking third (x >= 66.7)."""
    te = events_df[events_df['team.name'] == team_name].copy()
    te['location.x'] = pd.to_numeric(te['location.x'], errors='coerce')
    te['location.y'] = pd.to_numeric(te['location.y'], errors='coerce')

    dribbles = te[
        (te.get('is_custom_dribble_success') == True) &
        (te['location.x'] >= 66.7)
    ].dropna(subset=['location.x', 'location.y'])

    pitch, fig, ax = _make_pitch()

    # Shade the final third
    rect = plt.Rectangle((66.7, 0), 33.3, 100, facecolor='#e9ecef',
                          edgecolor='none', alpha=0.4, zorder=1)
    ax.add_patch(rect)

    if not dribbles.empty:
        # If carry end location available, draw arrows
        has_end = (dribbles['carry.endLocation.x'].notna() &
                   dribbles['carry.endLocation.y'].notna())
        arrows_df = dribbles[has_end]
        dots_df = dribbles[~has_end]

        if not arrows_df.empty:
            pitch.arrows(arrows_df['location.x'], arrows_df['location.y'],
                         pd.to_numeric(arrows_df['carry.endLocation.x'], errors='coerce'),
                         pd.to_numeric(arrows_df['carry.endLocation.y'], errors='coerce'),
                         width=2, headwidth=5, headlength=4,
                         color='#e63946', alpha=0.7, zorder=3, ax=ax)

        if not dots_df.empty:
            pitch.scatter(dots_df['location.x'], dots_df['location.y'],
                          s=50, color='#e63946', edgecolors='white',
                          linewidth=0.5, zorder=3, ax=ax, alpha=0.8)

    ax.set_title(title or f'{team_name} — Dribbles in Final Third ({len(dribbles)})',
                 fontsize=13, fontweight='bold', pad=8)
    fig.tight_layout()
    return fig


# =========================================================================
# 10. Shot Assists + Final Third Dribbles (side-by-side)
# =========================================================================
def plot_shot_assists_and_dribbles(events_df, team_name, player_name=None,
                                   title=None):
    """Side-by-side: shot assists (left) and successful dribbles in final third (right)."""
    te = events_df[events_df['team.name'] == team_name].copy()
    if player_name:
        te = te[te['player.name'] == player_name]

    # --- Left: Shot assists ---
    has_secondary = 'type.secondary' in te.columns
    if has_secondary:
        is_shot_assist = _check_secondary(te['type.secondary'], 'shot_assist')
        sa = te[is_shot_assist].copy()
    else:
        sa = pd.DataFrame()

    for col in ['location.x', 'location.y', 'pass.endLocation.x', 'pass.endLocation.y']:
        if col in sa.columns:
            sa[col] = pd.to_numeric(sa[col], errors='coerce')
    if not sa.empty:
        sa = sa.dropna(subset=['location.x', 'location.y'])

    # --- Right: Dribbles in final third (successful + failed) ---
    te['location.x'] = pd.to_numeric(te['location.x'], errors='coerce')
    te['location.y'] = pd.to_numeric(te['location.y'], errors='coerce')

    all_dribble_attempts = te[
        (te.get('is_dribble_attempt') == True) &
        (te['location.x'] >= 66.7)
    ].dropna(subset=['location.x', 'location.y'])

    dribbles_success = all_dribble_attempts[
        all_dribble_attempts.get('is_custom_dribble_success') == True
    ]
    dribbles_failed = all_dribble_attempts[
        all_dribble_attempts.get('is_custom_dribble_success') != True
    ]

    # --- Figure ---
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(20, 8))

    # Left pitch: shot assists
    pitch_l = Pitch(pitch_type='wyscout', pitch_color=PITCH_COLOR, line_color=LINE_COLOR)
    pitch_l.draw(ax=ax_l)

    if not sa.empty:
        has_end = (sa['pass.endLocation.x'].notna() & sa['pass.endLocation.y'].notna())
        arrows_sa = sa[has_end]
        dots_sa = sa[~has_end]

        if not arrows_sa.empty:
            pitch_l.arrows(arrows_sa['location.x'], arrows_sa['location.y'],
                           arrows_sa['pass.endLocation.x'],
                           arrows_sa['pass.endLocation.y'],
                           width=2, headwidth=5, headlength=4,
                           color='#e63946', alpha=0.7, zorder=3, ax=ax_l)
        if not dots_sa.empty:
            pitch_l.scatter(dots_sa['location.x'], dots_sa['location.y'],
                            s=50, color='#e63946', edgecolors='white',
                            linewidth=0.5, zorder=3, ax=ax_l, alpha=0.8)

    who = _short_name(player_name) if player_name else team_name
    ax_l.set_title(f'{who} — Shot Assists ({len(sa)})',
                   fontsize=13, fontweight='bold', pad=8)

    # Right pitch: dribbles in final third
    pitch_r = Pitch(pitch_type='wyscout', pitch_color=PITCH_COLOR, line_color=LINE_COLOR)
    pitch_r.draw(ax=ax_r)

    # Shade final third
    rect = plt.Rectangle((66.7, 0), 33.3, 100, facecolor='#e9ecef',
                          edgecolor='none', alpha=0.4, zorder=1)
    ax_r.add_patch(rect)

    # Successful dribbles as arrows
    if not dribbles_success.empty:
        has_end_d = (dribbles_success['carry.endLocation.x'].notna() &
                     dribbles_success['carry.endLocation.y'].notna())
        arrows_d = dribbles_success[has_end_d]
        dots_d = dribbles_success[~has_end_d]

        if not arrows_d.empty:
            pitch_r.arrows(arrows_d['location.x'], arrows_d['location.y'],
                           pd.to_numeric(arrows_d['carry.endLocation.x'], errors='coerce'),
                           pd.to_numeric(arrows_d['carry.endLocation.y'], errors='coerce'),
                           width=2, headwidth=5, headlength=4,
                           color='#2a9d8f', alpha=0.7, zorder=3, ax=ax_r,
                           label=f'Successful ({len(dribbles_success)})')
        if not dots_d.empty:
            pitch_r.scatter(dots_d['location.x'], dots_d['location.y'],
                            s=50, color='#2a9d8f', edgecolors='white',
                            linewidth=0.5, zorder=3, ax=ax_r, alpha=0.8)

    # Failed dribbles as 'x' markers
    if not dribbles_failed.empty:
        ax_r.scatter(dribbles_failed['location.x'], dribbles_failed['location.y'],
                     s=60, color='#e63946', marker='x', linewidths=2,
                     zorder=4, alpha=0.8,
                     label=f'Failed ({len(dribbles_failed)})')

    if not all_dribble_attempts.empty:
        ax_r.legend(loc='upper left', fontsize=9)

    ax_r.set_title(f'{who} — Dribbles in Final Third ({len(all_dribble_attempts)})',
                   fontsize=13, fontweight='bold', pad=8)

    fig.tight_layout()
    return fig
