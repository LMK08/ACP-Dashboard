"""Match-report PDF for the Match Analysis page.

Assembles the page's own cached figure PNGs (shot maps, xG flowchart,
tactical pitch maps) plus the team/player stats tables into a single
downloadable PDF. Reuses the styling/building blocks of the opposition
report (generate_pdf.OppositionReportPDF).
"""
import datetime
import io
import os

import pandas as pd
from fpdf import FPDF
from PIL import Image

from generate_pdf import OppositionReportPDF

# (section title, home figure key, away figure key) — one landscape page each
_TACTICAL_PAGES = [
    ("Average Player Positions", 'avg_positions_home', 'avg_positions_away'),
    ("Passing Networks", 'passing_network_home', 'passing_network_away'),
    ("Defensive Duels", 'defensive_duels_home', 'defensive_duels_away'),
    ("Shot Assists & Dribbles", 'shot_assists_home', 'shot_assists_away'),
    ("Ball Recoveries", 'recovery_map_home', 'recovery_map_away'),
    ("Ball Losses", 'loss_map_home', 'loss_map_away'),
]

_MAX_TABLE_COLS = 14  # landscape A4 readability limit for player stats


class MatchReportPDF(OppositionReportPDF):
    """Opposition-report styling with a match-report header."""

    def __init__(self, title, subtitle):
        # Skip OppositionReportPDF.__init__ (opponent fields) but keep the
        # same page setup it uses.
        FPDF.__init__(self, orientation='L', unit='mm', format='A4')
        self.title_text = self._sanitize(title)
        self.subtitle_text = self._sanitize(subtitle)
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(40, 40, 40)
        self.cell(0, 8, f"Match Report: {self.title_text}", ln=False)
        self.set_font('Helvetica', '', 9)
        self.cell(0, 8, self.subtitle_text, ln=True, align='R')
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.ln(4)


def _table_ready(df, max_cols=None):
    """Make a DataFrame printable: index becomes a column when meaningful,
    column count capped for page width."""
    out = df.copy()
    if not isinstance(out.index, pd.RangeIndex):
        out.insert(0, out.index.name or '', out.index.astype(str))
        out = out.reset_index(drop=True)
    truncated = False
    if max_cols and len(out.columns) > max_cols:
        out = out.iloc[:, :max_cols]
        truncated = True
    return out, truncated


def _fit_figure(pdf, tmp_files, png, max_w=None):
    """Place a figure centered, scaled to fit the space left on the page
    (so a section title is never orphaned by an auto page break)."""
    with Image.open(io.BytesIO(png)) as im:
        aspect = im.height / im.width
    avail_h = pdf.h - 20 - pdf.get_y() - 2
    w = min(max_w or (pdf.w - 40), avail_h / aspect)
    tmp_files.append(pdf.add_figure(png, w=w, x=(pdf.w - w) / 2))


def _side_by_side(pdf, tmp_files, left_png, right_png, left_label=None, right_label=None):
    """Place up to two figures on one row; returns figure height used."""
    w_half = (pdf.w - 20) / 2
    y = pdf.get_y()
    if left_label or right_label:
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(w_half, 6, pdf._sanitize(left_label or ''), align='C')
        pdf.cell(w_half, 6, pdf._sanitize(right_label or ''), align='C')
        pdf.ln()
        y = pdf.get_y()
    placed = False
    for i, png in enumerate((left_png, right_png)):
        if png:
            tmp_files.append(pdf.add_figure(png, w=w_half - 4, x=12 + i * w_half, y=y))
            placed = True
    if placed:
        # pitch figures are ~4:3; step past the taller possible render
        pdf.set_y(y + (w_half - 4) * 0.75 + 4)


def generate_match_report_pdf(match_info, figures, team_stats=None, player_stats=None):
    """Build the match report and return raw PDF bytes.

    match_info: row from season_matches_df (homeTeamName/awayTeamName/score/...)
    figures:    {key: png_bytes or None} — see _TACTICAL_PAGES plus
                'shotmap_home', 'shotmap_away', 'xg_flowchart'
    team_stats: {category: DataFrame} from all_match_data[mid]['team_stats']
    player_stats: {'home': DataFrame, 'away': DataFrame}
    """
    home = str(match_info.get('homeTeamName', 'Home'))
    away = str(match_info.get('awayTeamName', 'Away'))
    score = str(match_info.get('score', '') or '')
    date = str(match_info.get('display_date', '') or match_info.get('dateutc', ''))[:10]
    gw = match_info.get('gameweek', '')
    gw_txt = f"GW {int(gw)}  |  " if pd.notna(gw) and str(gw) not in ('', '?') else ''

    pdf = MatchReportPDF(f"{home} {score} {away}".replace('  ', ' '),
                         f"{gw_txt}{date}")
    pdf.alias_nb_pages()
    tmp_files = []
    try:
        # ---- Page 1: shot maps + xG flowchart ----
        pdf.add_page()
        if figures.get('shotmap_home') or figures.get('shotmap_away'):
            pdf.add_section_title("Shot Maps")
            _side_by_side(pdf, tmp_files,
                          figures.get('shotmap_home'), figures.get('shotmap_away'),
                          home, away)
        if figures.get('xg_flowchart'):
            pdf.add_page()
            pdf.add_section_title("xG Flowchart")
            _fit_figure(pdf, tmp_files, figures['xg_flowchart'])

        # ---- Team stats ----
        if team_stats:
            pdf.add_page()
            pdf.add_section_title("Team Stats")
            for category, df in team_stats.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                tbl, _ = _table_ready(df)
                # keep category + table together: new page if near the bottom
                needed = 8 + 7 + 6 * len(tbl)
                if pdf.get_y() + needed > pdf.h - 20:
                    pdf.add_page()
                pdf.add_subtitle(str(category))
                pdf.add_stats_table(tbl, max_w=pdf.w - 20)
                pdf.ln(4)

        # ---- Player stats ----
        if isinstance(player_stats, dict):
            for side, team in (('home', home), ('away', away)):
                df = player_stats.get(side)
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                pdf.add_page()
                pdf.add_section_title(f"Player Stats — {team}")
                tbl, truncated = _table_ready(df, max_cols=_MAX_TABLE_COLS)
                pdf.add_stats_table(tbl, max_w=pdf.w - 20)
                if truncated:
                    pdf.ln(2)
                    pdf.set_font('Helvetica', 'I', 8)
                    pdf.set_text_color(120, 120, 120)
                    pdf.cell(0, 5, pdf._sanitize(
                        f"Showing first {_MAX_TABLE_COLS} columns — full stats in the dashboard."), ln=True)

        # ---- Tactical pages ----
        for title, hk, ak in _TACTICAL_PAGES:
            if figures.get(hk) or figures.get(ak):
                pdf.add_page()
                pdf.add_section_title(title)
                _side_by_side(pdf, tmp_files, figures.get(hk), figures.get(ak),
                              home, away)

        return bytes(pdf.output())
    finally:
        for tmp in tmp_files:
            try:
                os.unlink(tmp)
            except OSError:
                pass
