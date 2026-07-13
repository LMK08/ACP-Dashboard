"""One-page player report PDF (portrait A4).

Composes the profile's core visuals — template radar, shot map, box-pass
creativity map — under a bio/ratings header. Figures are passed in as
matplotlib Figure objects; this module only lays out and embeds them.
"""
import datetime
import os
import tempfile

from fpdf import FPDF

from generate_pdf import OppositionReportPDF

_sanitize = OppositionReportPDF._sanitize

ACCENT = (29, 111, 78)     # pitch green
INK = (22, 33, 26)
MUTED = (110, 120, 113)
TILE_BG = (243, 245, 241)


class PlayerOnePagerPDF(FPDF):
    def __init__(self, player_name, subtitle):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.player_name = _sanitize(player_name)
        self.subtitle = _sanitize(subtitle)
        self.set_auto_page_break(auto=False)

    def header(self):
        self.set_fill_color(*ACCENT)
        self.rect(0, 0, 210, 20, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 17)
        self.set_xy(10, 4)
        self.cell(130, 8, self.player_name)
        self.set_font('Helvetica', '', 9.5)
        self.set_xy(10, 12)
        self.cell(130, 5, self.subtitle)
        self.set_font('Helvetica', 'B', 9)
        self.set_xy(140, 7)
        self.cell(60, 6, 'ACP Analytics - Player One-Pager', align='R')
        self.set_text_color(*INK)

    def footer(self):
        self.set_y(-12)
        self.set_font('Helvetica', 'I', 7.5)
        self.set_text_color(*MUTED)
        self.cell(0, 5, _sanitize(self._footer_note), align='C')

    _footer_note = ''


def _embed_fig(pdf, fig, x, y, w):
    """Save a matplotlib figure to a temp PNG and place it. Returns the
    rendered height in mm."""
    fw, fh = fig.get_size_inches()
    h = w * fh / fw
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    try:
        fig.savefig(tmp.name, dpi=110, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        pdf.image(tmp.name, x=x, y=y, w=w)
    finally:
        tmp.close()
        os.unlink(tmp.name)
    return h


def build_player_onepager(player_name, subtitle, tiles, fig_radar,
                          fig_shots, fig_passes, footer_note=''):
    """Build the PDF and return its bytes.

    tiles: list of (label, value) shown as header stat boxes (max 5).
    Any of the three figures may be None — layout skips them.
    """
    pdf = PlayerOnePagerPDF(player_name, subtitle)
    pdf._footer_note = (footer_note or
                        f"Generated {datetime.date.today().isoformat()}")
    pdf.add_page()

    y = 25
    # --- stat tiles row ---
    tiles = [t for t in (tiles or []) if t][:5]
    if tiles:
        tile_w = (190 - 4 * (len(tiles) - 1)) / len(tiles)
        x = 10
        for label, value in tiles:
            pdf.set_fill_color(*TILE_BG)
            pdf.rect(x, y, tile_w, 14, 'F')
            pdf.set_xy(x + 2, y + 1.5)
            pdf.set_font('Helvetica', '', 6.5)
            pdf.set_text_color(*MUTED)
            pdf.cell(tile_w - 4, 3.5, _sanitize(str(label).upper()))
            pdf.set_xy(x + 2, y + 6)
            pdf.set_font('Helvetica', 'B', 12)
            pdf.set_text_color(*INK)
            pdf.cell(tile_w - 4, 6, _sanitize(str(value)))
            x += tile_w + 4
        y += 18

    # --- radar (full width) ---
    if fig_radar is not None:
        h = _embed_fig(pdf, fig_radar, 10, y, 190)
        y += h + 4

    # --- shot map + box passes (two-up) ---
    row_h = 0
    if fig_shots is not None:
        row_h = max(row_h, _embed_fig(pdf, fig_shots, 10, y, 93))
    if fig_passes is not None:
        row_h = max(row_h, _embed_fig(pdf, fig_passes, 107, y, 93))
    y += row_h

    return bytes(pdf.output())
