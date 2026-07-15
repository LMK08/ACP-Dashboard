"""Multi-page player report PDF (A4).

Composes the Player Profile's core content into a printable dossier:

  page 1 (portrait)   identity header, bio strip, rating tiles, the
                      template radar, and the role's most important stats
                      with a percentile wash
  page 2 (portrait)   projection outlook, then the position-appropriate
                      maps — shot + creation for attackers, defensive
                      action heatmap for everyone else
  page 3 (landscape)  the ACP Index engine card

Figures arrive as matplotlib Figure objects (or, for the engine card,
PNG bytes); this module only lays out and embeds them. It never builds a
figure, so it needs no matplotlib lock of its own — the caller holds
MPL_LOCK across build + this + close (see mpl_safety).

Named "one-pager" for continuity with the button that triggers it; it has
not been one page since the profile outgrew it.
"""
import colorsys
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
RULE = (214, 220, 215)

A4_W, A4_H = 210.0, 297.0
MARGIN = 10.0
HEADER_H = 20.0


class PlayerOnePagerPDF(FPDF):
    def __init__(self, player_name, subtitle):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.player_name = _sanitize(player_name)
        self.subtitle = _sanitize(subtitle)
        self.set_auto_page_break(auto=False)

    def header(self):
        # self.w, not a literal: page 3 is landscape and the banner has to
        # span it too.
        self.set_fill_color(*ACCENT)
        self.rect(0, 0, self.w, HEADER_H, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 17)
        self.set_xy(MARGIN, 4)
        self.cell(self.w - 80, 8, self.player_name)
        self.set_font('Helvetica', '', 9.5)
        self.set_xy(MARGIN, 12)
        self.cell(self.w - 80, 5, self.subtitle)
        self.set_font('Helvetica', 'B', 9)
        self.set_xy(self.w - 70, 7)
        self.cell(60, 6, 'ACP Analytics - Player Report', align='R')
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


def _embed_png(pdf, png_bytes, x, y, w, aspect):
    """Place raw PNG bytes (the engine card is already rendered). `aspect`
    is width/height of the source image. Returns height in mm."""
    h = w / aspect
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    try:
        tmp.write(png_bytes)
        tmp.close()
        pdf.image(tmp.name, x=x, y=y, w=w)
    finally:
        os.unlink(tmp.name)
    return h


def _percentile_rgb(pct):
    """Red (0) -> yellow (50) -> green (100), matching the Stats tab's wash.

    The tab emits `hsl(hue, 65%, 72%)` with hue = pct/100*120; fpdf wants
    RGB, so reproduce that exact colour rather than eyeballing a new one.
    Note colorsys takes H,L,S — not H,S,L.
    """
    p = max(0.0, min(100.0, float(pct)))
    hue = (p / 100.0) * 120.0
    r, g, b = colorsys.hls_to_rgb(hue / 360.0, 0.72, 0.65)
    return int(r * 255), int(g * 255), int(b * 255)


def _rule(pdf, y, w):
    pdf.set_draw_color(*RULE)
    pdf.set_line_width(0.3)
    pdf.line(MARGIN, y, MARGIN + w, y)


def _bio_strip(pdf, bio, y, w):
    """One-line 'Nationality · Right · 189 cm · ...' strip under the tiles."""
    if not bio:
        return y
    txt = '   ·   '.join(f'{k} {v}' for k, v in bio if v not in (None, '', '-'))
    if not txt.strip():
        return y
    pdf.set_xy(MARGIN, y)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(*MUTED)
    pdf.cell(w, 4, _sanitize(txt))
    pdf.set_text_color(*INK)
    return y + 6


def _tiles_grid(pdf, tiles, y, w, per_row=6):
    """Stat tiles, wrapping onto extra rows. The old build capped at 5 and
    silently dropped the rest; the header now carries more than that."""
    tiles = [t for t in (tiles or []) if t]
    if not tiles:
        return y
    for i in range(0, len(tiles), per_row):
        row = tiles[i:i + per_row]
        tile_w = (w - 4 * (len(row) - 1)) / len(row) if len(row) > 1 else w
        x = MARGIN
        for label, value in row:
            pdf.set_fill_color(*TILE_BG)
            pdf.rect(x, y, tile_w, 14, 'F')
            pdf.set_xy(x + 2, y + 1.5)
            pdf.set_font('Helvetica', '', 6.5)
            pdf.set_text_color(*MUTED)
            pdf.cell(tile_w - 4, 3.5, _sanitize(str(label).upper()))
            pdf.set_xy(x + 2, y + 6)
            pdf.set_font('Helvetica', 'B', 11)
            pdf.set_text_color(*INK)
            pdf.cell(tile_w - 4, 6, _sanitize(str(value)))
            x += tile_w + 4
        y += 16
    return y + 2


def _percentile_table(pdf, rows, y, x, w, title=None):
    """Metric | Value | Percentile, with the Stats tab's colour wash.

    rows: list of (metric, value_str, pct_0_100 or None), already ordered
    and capped by the caller. Written here rather than reusing
    OppositionReportPDF.add_stats_table because that one has equal-width
    columns and no per-cell fill, which is the whole point of this table.
    """
    if not rows:
        return y
    if title:
        pdf.set_xy(x, y)
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_text_color(*INK)
        pdf.cell(w, 5, _sanitize(title))
        y += 6

    w_pct, w_val = 22.0, 24.0
    w_met = w - w_pct - w_val

    pdf.set_xy(x, y)
    pdf.set_font('Helvetica', 'B', 7.5)
    pdf.set_fill_color(228, 232, 228)
    pdf.set_text_color(*MUTED)
    pdf.cell(w_met, 5.5, ' Metric', border=0, fill=True)
    pdf.cell(w_val, 5.5, 'per 90', border=0, fill=True, align='C')
    pdf.cell(w_pct, 5.5, 'pct', border=0, fill=True, align='C')
    y += 5.5

    pdf.set_font('Helvetica', '', 7.5)
    for metric, val, pct in rows:
        pdf.set_xy(x, y)
        pdf.set_fill_color(250, 250, 249)
        pdf.set_text_color(*INK)
        pdf.cell(w_met, 5, _sanitize(f' {metric}'), border=0, fill=True)
        pdf.cell(w_val, 5, _sanitize(str(val)), border=0, fill=True,
                 align='C')
        if pct is None:
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(w_pct, 5, '-', border=0, fill=True, align='C')
        else:
            pdf.set_fill_color(*_percentile_rgb(pct))
            pdf.set_text_color(0, 0, 0)
            pdf.cell(w_pct, 5, f'{float(pct):.0f}', border=0, fill=True,
                     align='C')
        y += 5
    pdf.set_text_color(*INK)
    return y + 2


def build_player_onepager(player_name, subtitle, tiles, fig_radar,
                          fig_shots, fig_passes, footer_note='',
                          bio=None, role_stats=None, role_label='',
                          fig_projection=None, fig_defensive=None,
                          engine_card_png=None, engine_card_aspect=2.0):
    """Build the PDF and return its bytes.

    tiles         list of (label, value) header stat boxes; wraps past 6.
    bio           list of (label, value) for the one-line bio strip.
    role_stats    list of (metric, value_str, pct_0_100) for the role's
                  most important metrics, already ordered/capped.
    role_label    e.g. 'Pressing Forward' — titles the stats table.
    fig_*         matplotlib Figures; any may be None and are skipped.
    engine_card_png  PNG bytes from _render_acp_index_card_png.

    Pages 2 and 3 are only added when they have content, so a player with
    no engine row (keepers) still gets a clean page 1.
    """
    pdf = PlayerOnePagerPDF(player_name, subtitle)
    pdf._footer_note = (footer_note or
                        f"Generated {datetime.date.today().isoformat()}")

    # ---------------- page 1: identity, radar, role stats ----------------
    pdf.add_page()
    w = A4_W - 2 * MARGIN
    y = HEADER_H + 5
    y = _tiles_grid(pdf, tiles, y, w)
    y = _bio_strip(pdf, bio, y, w)
    _rule(pdf, y, w)
    y += 3

    if fig_radar is not None:
        y += _embed_fig(pdf, fig_radar, MARGIN, y, w) + 3

    if role_stats:
        title = (f'Key metrics - {role_label}' if role_label
                 else 'Key metrics')
        # Two columns when the list is long enough to warrant it; the page
        # has ~80mm here and a 1-column list of 14 would overrun it.
        if len(role_stats) > 7:
            half = (len(role_stats) + 1) // 2
            col_w = (w - 6) / 2
            y_l = _percentile_table(pdf, role_stats[:half], y, MARGIN,
                                    col_w, title=title)
            y_r = _percentile_table(pdf, role_stats[half:], y,
                                    MARGIN + col_w + 6, col_w, title=' ')
            y = max(y_l, y_r)
        else:
            y = _percentile_table(pdf, role_stats, y, MARGIN, w, title=title)

    # ---------------- page 2: projection + position maps ----------------
    if fig_projection is not None or fig_shots is not None \
            or fig_passes is not None or fig_defensive is not None:
        pdf.add_page()
        y = HEADER_H + 5
        if fig_projection is not None:
            y += _embed_fig(pdf, fig_projection, MARGIN, y, w) + 4
        if fig_defensive is not None:
            # Non-attacking roles: one wide heatmap instead of the two-up.
            y += _embed_fig(pdf, fig_defensive, MARGIN + w * 0.15, y,
                            w * 0.70) + 4
        row_h = 0
        if fig_shots is not None:
            row_h = max(row_h, _embed_fig(pdf, fig_shots, MARGIN, y, 93))
        if fig_passes is not None:
            row_h = max(row_h, _embed_fig(pdf, fig_passes, 107, y, 93))
        y += row_h

    # ---------------- page 3: engine card (landscape) ----------------
    if engine_card_png:
        # Landscape on purpose: the card is a 2:1 wide-screen figure whose
        # KDE panel labels are ~8pt at 20in. At portrait width they shrink
        # past legibility; across a landscape page it lands ~200dpi.
        pdf.add_page(orientation='L')
        _embed_png(pdf, engine_card_png, MARGIN, HEADER_H + 5,
                   A4_H - 2 * MARGIN, engine_card_aspect)

    return bytes(pdf.output())
