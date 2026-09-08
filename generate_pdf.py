# generate_pdf.py
"""PDF generation for the Opposition Report.

Uses fpdf2 (FPDF) to produce a landscape A4 report with embedded charts,
tables, and textual analysis.

LAYOUT RULE — every image goes through add_figure / add_figure_row, which
measure the PNG's REAL pixel aspect ratio (Pillow, an fpdf2 dependency) and
either page-break or scale so nothing crosses the bottom margin. Never place
an image with an assumed aspect ratio: the figures come from matplotlib with
bbox_inches='tight', so their proportions are not knowable in advance, and a
wrong guess doesn't error — it silently clips the chart off the page (the
old layout lost the bottom radars and the defensive-structure pitch this way).
"""

import io
import datetime
import tempfile
import os

from fpdf import FPDF
from PIL import Image

# A4 landscape: 297mm wide x 210mm tall
# Usable area after margins (10mm each side): 277mm wide
# Bottom limit: page height - 20mm auto-page-break margin = 190mm


class OppositionReportPDF(FPDF):
    """Custom FPDF subclass for opposition reports."""

    # Unicode -> ASCII replacements for the built-in Helvetica font
    _UNICODE_MAP = {
        '—': '-',   # em dash
        '–': '-',   # en dash
        '•': '-',   # bullet
        '‘': "'",   # left single quote
        '’': "'",   # right single quote
        '“': '"',   # left double quote
        '”': '"',   # right double quote
        '…': '...', # ellipsis
        'é': 'e',   # e-acute
        'á': 'a',   # a-acute
        'ã': 'a',   # a-tilde
        'ó': 'o',   # o-acute
        'ú': 'u',   # u-acute
        'ç': 'c',   # c-cedilla
        'º': 'o',   # masculine ordinal (1º)
        'É': 'E',   # E-acute
        'Á': 'A',   # A-acute
        '−': '-',   # minus sign
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
    # Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _png_aspect(png_bytes):
        """height / width of the actual PNG."""
        with Image.open(io.BytesIO(png_bytes)) as im:
            iw, ih = im.size
        return ih / float(iw)

    def _bottom_limit(self):
        """Lowest y content may reach (matches the auto-page-break margin)."""
        return self.h - 20

    def _avail_w(self):
        return self.w - 20

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

    def add_figure(self, png_bytes, w=None, h=None, x=None, y=None,
                   section_title=None):
        """Embed a PNG at its true aspect ratio, guaranteed to fit the page.

        The drawn height always comes from the PNG's real pixel dimensions.
        For flowing placements (no explicit y) the image page-breaks when it
        doesn't fit the space left and a fresh page offers more room
        (repeating section_title on the new page when given); whatever space
        it ends up with, it is scaled down to fit rather than clipped. The
        cursor advances below the image. Returns the temp file path for
        cleanup.
        """
        aspect = self._png_aspect(png_bytes)
        if w and not h:
            h = w * aspect
        elif h and not w:
            w = h / aspect
        elif not w and not h:
            w = self._avail_w()
            h = w * aspect

        bottom = self._bottom_limit()
        y0 = y if y is not None else self.get_y()

        if y is None and y0 + h > bottom:
            # Break when the image would FULLY fit a fresh page and the move
            # gains real room. An image too tall for ANY page is scaled in
            # place instead (breaking would strand the title on its own page
            # and still scale on the next one) — unless less than half a page
            # is left here: then the fresh page wins, because scaling into a
            # sliver, or into NOTHING when the cursor already sits on the
            # bottom margin (avail_h <= 0 gave w = h = 0 and the figure
            # silently vanished — caught by tests/test_pdf_layout.py), is
            # worse than a stranded title.
            fresh_avail = bottom - 30  # ~content top after header
            remaining = bottom - y0
            gains_room = fresh_avail - remaining > 5
            if gains_room and (h <= fresh_avail or remaining < fresh_avail / 2):
                self.add_page()
                if section_title:
                    self.add_section_title(section_title)
                y0 = self.get_y()

        # Whatever space we ended with: shrink to fit, keeping the aspect.
        avail_h = bottom - y0
        if h > avail_h:
            scale = avail_h / h
            h *= scale
            w *= scale

        x0 = x if x is not None else 10

        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp.write(png_bytes)
        tmp.close()
        self.image(tmp.name, x=x0, y=y0, w=w, h=h)
        if y is None:
            self.set_y(y0 + h)
        return tmp.name

    def add_figure_row(self, png_list, gap=4, section_title=None):
        """Place PNGs side by side at equal widths, top-aligned.

        Row height is the tallest image at its true aspect. Page-breaks first
        when the row doesn't fit below the cursor (repeating section_title on
        the new page when given), and scales the whole row down when even a
        fresh page can't hold it. Advances the cursor below the row. Returns
        temp file paths.
        """
        pngs = [p for p in png_list if p]
        if not pngs:
            return []
        n = len(pngs)
        aspects = [self._png_aspect(p) for p in pngs]
        w_each = (self._avail_w() - gap * (n - 1)) / n
        row_h = max(w_each * a for a in aspects)

        bottom = self._bottom_limit()
        if self.get_y() + row_h > bottom:
            self.add_page()
            if section_title:
                self.add_section_title(section_title)
        avail_h = bottom - self.get_y()
        if row_h > avail_h:
            scale = avail_h / row_h
            w_each *= scale
            row_h = avail_h

        y0 = self.get_y()
        x = 10
        tmps = []
        for p in pngs:
            tmps.append(self.add_figure(p, w=w_each, x=x, y=y0))
            x += w_each + gap
        self.set_y(y0 + row_h)
        return tmps

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
            # Cap the radar to the space left on the page — add_figure
            # derives the height from the real PNG, so the card's text
            # placement below can trust it.
            radar_w = 150
            radar_h = radar_w * self._png_aspect(png_bytes)
            avail_h = self._bottom_limit() - start_y
            if radar_h > avail_h:
                radar_w *= avail_h / radar_h
                radar_h = avail_h
            tmp = self.add_figure(png_bytes, w=radar_w, x=10, y=start_y)
            text_x = 10 + radar_w + 5
        else:
            text_x = 10
            radar_h = 0
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

        # Move below whichever is taller: the text block or the radar.
        if png_bytes:
            self.set_y(max(self.get_y(), start_y + radar_h))

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
    # Page 1: Team Overview (4 radars, ~square — each row of two fills a
    # page, so the second row flows onto its own page automatically)
    # ==================================================================
    pdf.add_page()
    title = f"{opponent_name} - Team Overview"
    pdf.add_section_title(title)

    radar_row1 = [figures[k] for k in ['radar_offensive', 'radar_distribution']
                  if k in figures]
    radar_row2 = [figures[k] for k in ['radar_defensive', 'radar_set_piece']
                  if k in figures]
    tmp_files += pdf.add_figure_row(radar_row1, section_title=title)
    tmp_files += pdf.add_figure_row(radar_row2,
                                    section_title=f"{title} (cont.)")

    # ==================================================================
    # On-Ball Value & Phases (parity with Team Analysis)
    # ==================================================================
    if 'obv_categories' in figures or 'phase_profile' in figures:
        pdf.add_page()
        title = f"{opponent_name} - On-Ball Value & Phases"
        pdf.add_section_title(title)
        if 'obv_categories' in figures:
            tmp_files.append(pdf.add_figure(figures['obv_categories'], w=230,
                                            section_title=title))
            pdf.ln(3)
        if 'phase_profile' in figures:
            tmp_files.append(pdf.add_figure(figures['phase_profile'], w=230,
                                            section_title=f"{title} (cont.)"))

    # ==================================================================
    # Projected Lineup + Subs
    # ==================================================================
    pdf.add_page()
    pdf.add_section_title(f"{opponent_name} - Projected Lineup")

    content_y = pdf.get_y()

    if 'formation' in figures:
        # Portrait pitch — fit within the left half of the page.
        tmp_files.append(pdf.add_figure(figures['formation'], w=80,
                                        x=10, y=content_y))

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
    # Key Players  (1 per page for clean layout)
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
    title = f"{opponent_name} - Set Piece Analysis"
    pdf.add_section_title(title)

    sp_table = texts.get('set_piece_table')
    if sp_table is not None and hasattr(sp_table, 'iterrows'):
        pdf.add_stats_table(sp_table)
        pdf.ln(5)

    corner_row = [figures[k] for k in ['corner_left', 'corner_right']
                  if k in figures]
    tmp_files += pdf.add_figure_row(corner_row,
                                    section_title=f"{title} (cont.)")

    # Set piece scatter plots (2 per page, side by side)
    sp_scatter_keys = [k for k in sorted(figures.keys())
                       if k.startswith('sp_scatter_')]
    for page_start in range(0, len(sp_scatter_keys), 2):
        page_keys = sp_scatter_keys[page_start:page_start + 2]
        if page_keys:
            pdf.add_page()
            title = f"{opponent_name} - Set Piece Efficiency"
            pdf.add_section_title(title)
            tmp_files += pdf.add_figure_row(
                [figures[k] for k in page_keys],
                section_title=f"{title} (cont.)")

    # ==================================================================
    # Season Form + Shot Maps
    # ==================================================================
    pdf.add_page()
    title = f"{opponent_name} - Season Form & Shot Maps"
    pdf.add_section_title(title)

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
        tmp_files.append(pdf.add_figure(figures['xg_history'], w=260,
                                        section_title=f"{title} (cont.)"))
        pdf.ln(3)

    # Shot maps on a new page (large images)
    shot_row = [figures[k] for k in ['shotmap_for', 'shotmap_against']
                if k in figures]
    if shot_row:
        pdf.add_page()
        tmp_files += pdf.add_figure_row(shot_row)

    # ==================================================================
    # Tactical Zone Analysis
    # ==================================================================
    tac_keys = ['avg_positions', 'defensive_structure', 'zone_recovery',
                'zone_loss', 'passing_network', 'shot_assists_dribbles']
    if any(k in figures for k in tac_keys):
        # Page 1: Average Positions
        if 'avg_positions' in figures:
            pdf.add_page()
            pdf.add_section_title(f"{opponent_name} - Tactical Zone Analysis")
            tmp_files.append(pdf.add_figure(figures['avg_positions'], w=200))

        # Defensive Structure (tall vertical pitch — add_figure scales it
        # to whatever fits under the title; a fixed width would run off
        # the bottom of the page)
        if 'defensive_structure' in figures:
            pdf.add_page()
            pdf.add_section_title(
                f"{opponent_name} - Defensive Structure")
            tmp_files.append(pdf.add_figure(figures['defensive_structure']))

        # Recovery / Loss zones side by side
        zone_row = [figures[k] for k in ['zone_recovery', 'zone_loss']
                    if k in figures]
        if zone_row or 'passing_network' in figures \
                or 'shot_assists_dribbles' in figures:
            pdf.add_page()
            if 'avg_positions' in figures:
                title = f"{opponent_name} - Tactical Zones (cont.)"
            else:
                title = f"{opponent_name} - Tactical Zone Analysis"
            pdf.add_section_title(title)

            tmp_files += pdf.add_figure_row(zone_row,
                                            section_title=f"{title} (cont.)")

            if 'passing_network' in figures:
                pdf.ln(3)
                tmp_files.append(pdf.add_figure(
                    figures['passing_network'], w=200,
                    section_title=f"{opponent_name} - Passing Network"))

            if 'shot_assists_dribbles' in figures:
                pdf.ln(3)
                tmp_files.append(pdf.add_figure(
                    figures['shot_assists_dribbles'], w=240,
                    section_title=f"{opponent_name} - Shot Assists & Dribbles"))

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
