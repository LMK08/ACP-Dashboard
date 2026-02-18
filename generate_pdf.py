# generate_pdf.py
"""PDF generation for the Opposition Report.

Uses fpdf2 (FPDF) to produce a landscape A4 report with embedded charts,
tables, and textual analysis.
"""

import io
import datetime
import tempfile
import os

from fpdf import FPDF

# A4 landscape: 297mm wide x 210mm tall
# Usable area after margins (10mm each side): 277mm wide
# Usable height after header/footer (~25mm top, 20mm bottom): ~165mm


class OppositionReportPDF(FPDF):
    """Custom FPDF subclass for opposition reports."""

    # Unicode -> ASCII replacements for the built-in Helvetica font
    _UNICODE_MAP = {
        '\u2014': '-',   # em dash
        '\u2013': '-',   # en dash
        '\u2022': '-',   # bullet
        '\u2018': "'",   # left single quote
        '\u2019': "'",   # right single quote
        '\u201c': '"',   # left double quote
        '\u201d': '"',   # right double quote
        '\u2026': '...', # ellipsis
        '\u00e9': 'e',   # e-acute
        '\u00e1': 'a',   # a-acute
        '\u00e3': 'a',   # a-tilde
        '\u00f3': 'o',   # o-acute
        '\u00fa': 'u',   # u-acute
        '\u00e7': 'c',   # c-cedilla
        '\u00ba': 'o',   # masculine ordinal (1º)
        '\u00c9': 'E',   # E-acute
        '\u00c1': 'A',   # A-acute
        '\u2212': '-',   # minus sign
    }

    @classmethod
    def _sanitize(cls, text):
        """Replace Unicode characters unsupported by Helvetica."""
        text = str(text)
        for char, replacement in cls._UNICODE_MAP.items():
            text = text.replace(char, replacement)
        return text.encode('latin-1', errors='replace').decode('latin-1')

    def __init__(self, opponent_name, match_date, gameweek):
        super().__init__(orientation='L', unit='mm', format='A4')
        self.opponent_name = self._sanitize(opponent_name)
        self.match_date = self._sanitize(match_date)
        self.gameweek = self._sanitize(gameweek)
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(40, 40, 40)
        self.cell(0, 8, f"Opposition Report: {self.opponent_name}", ln=False)
        self.set_font('Helvetica', '', 9)
        self.cell(0, 8, f"GW {self.gameweek}  |  {self.match_date}",
                  ln=True, align='R')
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align='C')
        self.cell(
            0, 10,
            f"Atletico CP  |  @lucaskimball  |  "
            f"{datetime.date.today().isoformat()}",
            align='R', new_x="LMARGIN", new_y="TOP",
        )

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------
    def add_section_title(self, title):
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(30, 30, 30)
        self.cell(0, 10, self._sanitize(title), ln=True)
        self.ln(2)

    def add_subtitle(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(60, 60, 60)
        self.cell(0, 8, self._sanitize(title), ln=True)
        self.ln(1)

    def add_figure(self, png_bytes, w=None, h=None, x=None, y=None):
        """Embed a PNG image. Returns temp file path for cleanup."""
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp.write(png_bytes)
        tmp.close()
        kw = {}
        if x is not None:
            kw['x'] = x
        if y is not None:
            kw['y'] = y
        if w:
            kw['w'] = w
        if h:
            kw['h'] = h
        self.image(tmp.name, **kw)
        return tmp.name

    def add_bullet_list(self, items, font_size=10):
        self.set_font('Helvetica', '', font_size)
        self.set_text_color(50, 50, 50)
        for item in items:
            clean = (item.replace('**', '').replace(':green[', '')
                     .replace(':red[', '').replace(':orange[', '')
                     .replace(']', ''))
            self.multi_cell(0, 6, self._sanitize(f"  -  {clean}"))
            self.ln(1)

    def add_stats_table(self, df, col_widths=None, max_w=None,
                        x_offset=None):
        """Render a DataFrame as a table.

        Parameters
        ----------
        max_w : float, optional
            Maximum total width for all columns.
        x_offset : float, optional
            If provided, each row starts at this x position instead of the
            left margin.  Needed when the table is placed beside an image.
        """
        cols = list(df.columns)
        n = len(cols)
        available = (max_w or self.w - 20)
        if col_widths is None:
            col_widths = [available / n] * n

        if x_offset is not None:
            self.set_x(x_offset)
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(220, 220, 220)
        for i, col in enumerate(cols):
            self.cell(col_widths[i], 7, self._sanitize(str(col)),
                      border=1, fill=True, align='C')
        self.ln()

        self.set_font('Helvetica', '', 9)
        self.set_fill_color(245, 245, 245)
        for row_idx, (_, row) in enumerate(df.iterrows()):
            if x_offset is not None:
                self.set_x(x_offset)
            fill = row_idx % 2 == 1
            for i, col in enumerate(cols):
                val = row[col]
                txt = f"{val:.2f}" if isinstance(val, float) else str(val)
                self.cell(col_widths[i], 6, self._sanitize(txt),
                          border=1, fill=fill, align='C')
            self.ln()

    def add_player_card(self, name, position, minutes,
                        strengths_text, weaknesses_text, png_bytes=None):
        """One player card — radar image left, text right.
        Returns temp file path (or None)."""
        self.add_subtitle(f"{name}  ({position}, {int(minutes)} mins)")
        start_y = self.get_y()

        if png_bytes:
            tmp = self.add_figure(png_bytes, w=150, x=10, y=start_y)
            text_x = 165
        else:
            text_x = 10
            tmp = None

        self.set_xy(text_x, start_y)
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 6, "Strengths:", ln=True)
        self.set_font('Helvetica', '', 9)
        for line in strengths_text[:5]:
            self.set_x(text_x)
            self.cell(0, 5, self._sanitize(f"  + {line}"), ln=True)

        self.ln(2)
        self.set_x(text_x)
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 6, "Weaknesses:", ln=True)
        self.set_font('Helvetica', '', 9)
        for line in weaknesses_text[:5]:
            self.set_x(text_x)
            self.cell(0, 5, self._sanitize(f"  - {line}"), ln=True)

        # Move below the image (radar is ~75mm tall at w=150)
        if png_bytes:
            self.set_y(max(self.get_y(), start_y + 75))

        self.ln(4)
        return tmp


def generate_opposition_report_pdf(opponent_name, match_date, gameweek,
                                   figures, texts):
    """Build the full PDF and return raw bytes."""
    pdf = OppositionReportPDF(opponent_name, str(match_date), str(gameweek))
    pdf.alias_nb_pages()
    tmp_files = []
    san = pdf._sanitize  # shorthand

    # ==================================================================
    # Page 1: Team Overview (3 radars)
    # ==================================================================
    pdf.add_page()
    pdf.add_section_title(f"{opponent_name} - Team Overview")

    radar_keys = ['radar_offensive', 'radar_distribution', 'radar_defensive']
    radar_present = [k for k in radar_keys if k in figures]
    if radar_present:
        n = len(radar_present)
        w = (pdf.w - 20) / n
        x = 10
        y = pdf.get_y()
        for key in radar_present:
            tmp = pdf.add_figure(figures[key], w=w - 2, x=x, y=y)
            tmp_files.append(tmp)
            x += w
        # Radars are square-ish (10x10 figsize) so height ~ width
        pdf.set_y(y + (w - 2) * 0.75)

    # ==================================================================
    # Page 2: Projected Lineup + Subs
    # ==================================================================
    pdf.add_page()
    pdf.add_section_title(f"{opponent_name} - Projected Lineup")

    content_y = pdf.get_y()

    if 'formation' in figures:
        # Formation is 6x8 (portrait pitch) — fit within left half of page
        # At w=80, height ~107mm, fits comfortably in usable area
        tmp = pdf.add_figure(figures['formation'], w=80, x=10, y=content_y)
        tmp_files.append(tmp)

    subs_df = texts.get('subs')
    if subs_df is not None and hasattr(subs_df, 'iterrows'):
        # Place subs table to the right of the formation
        subs_x = 100
        subs_max_w = pdf.w - subs_x - 10
        pdf.set_xy(subs_x, content_y)
        pdf.add_subtitle("Projected Substitutes")
        pdf.add_stats_table(subs_df.head(8), max_w=subs_max_w,
                            x_offset=subs_x)

    # ==================================================================
    # Pages 3+: Key Players  (1 per page for clean layout)
    # ==================================================================
    key_players = texts.get('key_players', [])
    for i, kp in enumerate(key_players):
        pdf.add_page()
        if i == 0:
            pdf.add_section_title(f"{opponent_name} - Key Players")
        else:
            pdf.add_section_title(f"{opponent_name} - Key Players (cont.)")

        fig_key = f'player_{i}'
        png = figures.get(fig_key)

        # Use pre-computed role-specific strengths/weaknesses from Streamlit
        s_lines = kp.get('strengths_lines', [])
        w_lines = kp.get('weaknesses_lines', [])

        if not s_lines and not w_lines:
            s_lines = ["No standout strengths"]
            w_lines = ["No clear weaknesses"]

        tmp = pdf.add_player_card(
            kp['name'], kp['position'], kp['minutes'],
            s_lines or ["No standout strengths"],
            w_lines or ["No clear weaknesses"],
            png_bytes=png,
        )
        if tmp:
            tmp_files.append(tmp)

    # ==================================================================
    # Strengths & Weaknesses Synopsis (side-by-side to avoid overflow)
    # ==================================================================
    pdf.add_page()
    pdf.add_section_title(f"{opponent_name} - Strengths & Weaknesses")

    profiles = texts.get('profiles', [])
    if profiles:
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 8, san(f"Tactical Profile: {', '.join(profiles)}"),
                 ln=True)
        pdf.ln(3)

    strengths = texts.get('strengths', [])
    weaknesses = texts.get('weaknesses', [])

    col_top_y = pdf.get_y()
    col_w = (pdf.w - 20) / 2  # two columns

    # Left column: Strengths
    if strengths:
        pdf.set_xy(10, col_top_y)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(col_w, 8, san("Strengths (65th+ percentile)"), ln=True)
        pdf.ln(1)
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(50, 50, 50)
        for m, v in strengths:
            pdf.set_x(10)
            line = f"  -  {m}: {v:.0f}th pct ({'Elite' if v >= 80 else 'Above Avg'})"
            pdf.multi_cell(col_w, 6, san(line))
            pdf.ln(1)
    left_bottom_y = pdf.get_y()

    # Right column: Weaknesses
    if weaknesses:
        right_x = 10 + col_w + 5
        pdf.set_xy(right_x, col_top_y)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(col_w - 5, 8, san("Weaknesses (35th- percentile)"), ln=True)
        pdf.ln(1)
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(50, 50, 50)
        for m, v in weaknesses:
            pdf.set_x(right_x)
            line = f"  -  {m}: {v:.0f}th pct ({'Poor' if v <= 20 else 'Below Avg'})"
            pdf.multi_cell(col_w - 5, 6, san(line))
            pdf.ln(1)
    right_bottom_y = pdf.get_y()

    pdf.set_y(max(left_bottom_y, right_bottom_y))

    # ==================================================================
    # Set Piece Analysis
    # ==================================================================
    pdf.add_page()
    pdf.add_section_title(f"{opponent_name} - Set Piece Analysis")

    sp_table = texts.get('set_piece_table')
    if sp_table is not None and hasattr(sp_table, 'iterrows'):
        pdf.add_stats_table(sp_table)
        pdf.ln(5)

    corner_keys = ['corner_left', 'corner_right']
    corner_present = [k for k in corner_keys if k in figures]
    if corner_present:
        w = (pdf.w - 20) / len(corner_present)
        x = 10
        y = pdf.get_y()
        for key in corner_present:
            tmp = pdf.add_figure(figures[key], w=w - 2, x=x, y=y)
            tmp_files.append(tmp)
            x += w

    # Set piece scatter plots (2 per page, side by side)
    sp_scatter_keys = [k for k in sorted(figures.keys())
                       if k.startswith('sp_scatter_')]
    for page_start in range(0, len(sp_scatter_keys), 2):
        page_keys = sp_scatter_keys[page_start:page_start + 2]
        if page_keys:
            pdf.add_page()
            pdf.add_section_title(
                f"{opponent_name} - Set Piece Efficiency")
            w = (pdf.w - 20) / len(page_keys)
            x = 10
            y = pdf.get_y()
            for key in page_keys:
                tmp = pdf.add_figure(figures[key], w=w - 2, x=x, y=y)
                tmp_files.append(tmp)
                x += w

    # ==================================================================
    # Season Form + Shot Maps
    # ==================================================================
    pdf.add_page()
    pdf.add_section_title(f"{opponent_name} - Season Form & Shot Maps")

    form_results = texts.get('form_results', [])
    if form_results:
        form_str = '  '.join(
            f"{r['result']} ({r['score']} vs {r['opponent']})"
            for r in form_results
        )
        pdf.set_font('Helvetica', '', 10)
        pdf.multi_cell(0, 6, san(f"Recent form: {form_str}"))
        pdf.ln(3)

    if 'xg_history' in figures:
        tmp = pdf.add_figure(figures['xg_history'], w=260)
        tmp_files.append(tmp)
        pdf.ln(3)

    # Shot maps on a new page (they're large square images)
    shot_keys = ['shotmap_for', 'shotmap_against']
    shot_present = [k for k in shot_keys if k in figures]
    if shot_present:
        pdf.add_page()
        w = (pdf.w - 20) / len(shot_present)
        x = 10
        y = pdf.get_y()
        for key in shot_present:
            tmp = pdf.add_figure(figures[key], w=w - 2, x=x, y=y)
            tmp_files.append(tmp)
            x += w

    # ==================================================================
    # Tactical Zone Analysis
    # ==================================================================
    tac_keys = ['avg_positions', 'zone_recovery', 'zone_loss',
                'shot_assists_dribbles']
    if any(k in figures for k in tac_keys):
        # Page 1: Average Positions
        if 'avg_positions' in figures:
            pdf.add_page()
            pdf.add_section_title(f"{opponent_name} - Tactical Zone Analysis")
            # avg_positions is ~12:8 ratio; w=200 → h≈133mm, fits after title
            tmp = pdf.add_figure(figures['avg_positions'], w=200)
            tmp_files.append(tmp)

        # Page 2: Recovery / Loss zones + shot assists
        zone_keys = ['zone_recovery', 'zone_loss']
        zone_present = [k for k in zone_keys if k in figures]
        if zone_present or 'shot_assists_dribbles' in figures:
            pdf.add_page()
            if not ('avg_positions' in figures):
                pdf.add_section_title(
                    f"{opponent_name} - Tactical Zone Analysis")
            else:
                pdf.add_section_title(
                    f"{opponent_name} - Tactical Zones (cont.)")

            if zone_present:
                w = (pdf.w - 20) / len(zone_present)
                x = 10
                y = pdf.get_y()
                for key in zone_present:
                    tmp = pdf.add_figure(figures[key], w=w - 2, x=x, y=y)
                    tmp_files.append(tmp)
                    x += w
                # Move past the zone images (~height is w * 0.6 for landscape)
                pdf.set_y(y + w * 0.55)

            if 'shot_assists_dribbles' in figures:
                pdf.ln(3)
                tmp = pdf.add_figure(
                    figures['shot_assists_dribbles'], w=240)
                tmp_files.append(tmp)

    # ==================================================================
    # Key Takeaways
    # ==================================================================
    pdf.add_page()
    pdf.add_section_title("Key Takeaways")

    takeaways = texts.get('takeaways', [])
    if takeaways:
        pdf.add_bullet_list(takeaways, font_size=11)
    else:
        pdf.set_font('Helvetica', '', 11)
        pdf.cell(0, 8, "No takeaways generated.", ln=True)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    pdf_bytes = pdf.output()

    for path in tmp_files:
        try:
            os.unlink(path)
        except OSError:
            pass

    return bytes(pdf_bytes)
