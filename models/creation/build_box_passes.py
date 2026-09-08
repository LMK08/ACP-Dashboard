"""Build box_entry_passes.parquet — every pass whose end location lands in
the attacking penalty box, joined with its GPA action value.

The dashboard can't ship the 212MB events_with_action_values.parquet, so
this extracts the slim slice the creativity chart needs (~1-2% of events).

Box in Wyscout 100x100 coords: end_x >= 84, 19 <= end_y <= 81.

Run from the Dashboard directory:
    python3 models/creation/build_box_passes.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DASH = HERE.parent.parent
GPA_PROJECT = DASH.parent.parent / 'GPA Model Project v2'

# Keep in sync with PLAYER_ID_ALIASES in league_config.py (canonical pid mapping)
PLAYER_ID_ALIASES = {71835: 1322978}

BOX_X = 84.0
BOX_Y_LO, BOX_Y_HI = 19.0, 81.0

TAG_COLS = ['cross', 'through_pass', 'smart_pass', 'key_pass', 'assist',
            'progressive_pass', 'deep_completion', 'head_pass']


def main():
    ev_cols = ['id', 'matchId', 'seasonId', 'competitionId',
               'minute', 'second', 'player.id', 'player.name', 'team.name',
               'opponentTeam.name', 'type.primary', 'type.secondary',
               'pass.accurate', 'pass.recipient.name', 'pass.height',
               'location.x', 'location.y',
               'pass.endLocation.x', 'pass.endLocation.y',
               'possession.types']
    raw = pd.read_parquet(DASH / 'raw_events.parquet', columns=ev_cols)
    passes = raw[raw['type.primary'] == 'pass'].copy()
    del raw
    dates = pd.read_parquet(DASH / 'matches_summary.parquet',
                            columns=['matchId', 'dateutc'])
    passes = passes.merge(dates.drop_duplicates('matchId'),
                          on='matchId', how='left')

    ex = pd.to_numeric(passes['pass.endLocation.x'], errors='coerce')
    ey = pd.to_numeric(passes['pass.endLocation.y'], errors='coerce')
    sx = pd.to_numeric(passes['location.x'], errors='coerce')
    in_box = (ex >= BOX_X) & ey.between(BOX_Y_LO, BOX_Y_HI)
    # exclude passes that both start and end in the box (recycling inside
    # the area isn't a box ENTRY) but keep starts exactly on the line
    starts_in_box = (sx >= BOX_X) & pd.to_numeric(
        passes['location.y'], errors='coerce').between(BOX_Y_LO, BOX_Y_HI)
    box = passes[in_box & ~starts_in_box].copy()
    del passes
    print(f"box-entry passes: {len(box):,}")

    def tags(sec):
        s = set(sec) if isinstance(sec, (list, np.ndarray)) else set()
        return pd.Series({t: (t in s) for t in TAG_COLS})

    box = pd.concat([box, box['type.secondary'].apply(tags)], axis=1)

    def phase(pt):
        s = set(pt) if isinstance(pt, (list, np.ndarray)) else set()
        if s & {'corner', 'free_kick', 'throw_in', 'penalty'}:
            return 'set_piece'
        if 'counter_attack' in s:
            return 'counter'
        return 'open_play'

    box['phase'] = box['possession.types'].apply(phase)

    vals = pd.read_parquet(
        GPA_PROJECT / 'parquet_data' / 'events_with_action_values.parquet',
        columns=['id', 'action_value'])
    box = box.merge(vals, on='id', how='left')
    del vals
    print(f"action_value coverage: {box['action_value'].notna().mean():.1%}")

    box['player.id'] = box['player.id'].replace(PLAYER_ID_ALIASES)

    out_cols = (['id', 'matchId', 'seasonId', 'competitionId', 'dateutc',
                 'minute', 'player.id', 'player.name', 'team.name',
                 'opponentTeam.name', 'pass.accurate', 'pass.recipient.name',
                 'pass.height', 'location.x', 'location.y',
                 'pass.endLocation.x', 'pass.endLocation.y', 'phase',
                 'action_value'] + TAG_COLS)
    out = box[out_cols]
    out_path = HERE / 'box_entry_passes.parquet'
    out.to_parquet(out_path, index=False)
    print(f"wrote {out_path} ({len(out):,} rows, "
          f"{out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
