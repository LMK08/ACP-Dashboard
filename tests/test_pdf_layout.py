"""Geometry test for the Opposition Report PDF (see CLAUDE.md, PDF layout rule).

Spies on FPDF.image and asserts that every figure placement made through
add_figure / add_figure_row stays inside the printable area of an A4
landscape page — x in [10, 287], y + h <= 190 — whatever the PNG's aspect
ratio, and that the aspect ratio is preserved (scaled, never stretched or
clipped). Pure-Python: no data files, runs in well under a second.

Run:  python -m pytest tests/test_pdf_layout.py -v
"""
import io
import os
import sys

import pytest
from PIL import Image

DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DASHBOARD_DIR)

from generate_pdf import OppositionReportPDF  # noqa: E402

X_MIN, X_MAX, Y_MAX = 10.0, 287.0, 190.0   # mm; A4 landscape is 297 x 210
EPS = 0.05                                  # float slack in mm


def _png(width, height):
    buf = io.BytesIO()
    Image.new('RGB', (width, height), (180, 180, 180)).save(buf, format='PNG')
    return buf.getvalue()


TALL, WIDE, SQUARE, HUGE = _png(600, 1800), _png(1800, 600), _png(900, 900), _png(2000, 7000)


@pytest.fixture
def pdf_and_placements(monkeypatch):
    """A fresh report whose image() calls are recorded as (x, y, w, h, page)."""
    placements = []
    original = OppositionReportPDF.image

    def spy(self, name, x=None, y=None, w=0, h=0, *args, **kwargs):
        placements.append((float(x), float(y), float(w), float(h), self.page_no()))
        return original(self, name, x=x, y=y, w=w, h=h, *args, **kwargs)

    monkeypatch.setattr(OppositionReportPDF, 'image', spy)
    pdf = OppositionReportPDF('Test FC', '2026-09-05', 1)
    pdf.add_page()
    yield pdf, placements


def _assert_inside(placements):
    assert placements, "no image was placed"
    for x, y, w, h, page in placements:
        assert w > 0 and h > 0, (x, y, w, h, page)
        assert x >= X_MIN - EPS, f"x={x} left of margin on page {page}"
        assert x + w <= X_MAX + EPS, f"x+w={x + w} past right margin on page {page}"
        assert y + h <= Y_MAX + EPS, f"y+h={y + h} crosses bottom margin on page {page}"


def _aspect(png_bytes):
    with Image.open(io.BytesIO(png_bytes)) as im:
        return im.size[1] / float(im.size[0])


def test_flowing_figures_never_cross_bottom_margin(pdf_and_placements):
    pdf, placements = pdf_and_placements
    tmps = []
    # Enough mixed-aspect figures to force several page breaks.
    for i, png in enumerate([TALL, WIDE, SQUARE, TALL, WIDE, WIDE, SQUARE, TALL] * 2):
        tmps.append(pdf.add_figure(png, w=120 if i % 3 else None,
                                   section_title=f"Section {i}" if i % 4 == 0 else None))
    _assert_inside(placements)
    assert max(p[4] for p in placements) > 1, "expected at least one page break"
    for t in tmps:
        os.remove(t)


def test_aspect_ratio_is_preserved_when_scaled(pdf_and_placements):
    pdf, placements = pdf_and_placements
    for png in (TALL, WIDE, SQUARE, HUGE):
        os.remove(pdf.add_figure(png))
    for (x, y, w, h, page), png in zip(placements, (TALL, WIDE, SQUARE, HUGE)):
        assert abs(h / w - _aspect(png)) < 0.01, f"aspect drifted for placement on page {page}"
    _assert_inside(placements)


def test_oversized_figure_is_scaled_not_clipped(pdf_and_placements):
    pdf, placements = pdf_and_placements
    os.remove(pdf.add_figure(HUGE))
    x, y, w, h, page = placements[-1]
    assert y + h <= Y_MAX + EPS
    assert h < 7000 / 2000 * (X_MAX - X_MIN), "a 3.5:1 image at full width cannot fit; it must shrink"
    assert pdf.get_y() >= y + h - EPS, "cursor must advance below the image"


@pytest.mark.parametrize('n', [2, 3, 4])
def test_figure_rows_fit_side_by_side(pdf_and_placements, n):
    pdf, placements = pdf_and_placements
    pngs = [TALL, WIDE, SQUARE, HUGE][:n]
    for _ in range(4):  # repeat to force a break inside the row sequence
        for t in pdf.add_figure_row(pngs, section_title="Row"):
            os.remove(t)
    _assert_inside(placements)
    rows = {}
    for x, y, w, h, page in placements:
        rows.setdefault((page, round(y, 2)), []).append((x, w))
    for (page, y), items in rows.items():
        assert len(items) == n, f"row on page {page} at y={y} placed {len(items)} images, expected {n}"
        xs = sorted(items)
        for (x1, w1), (x2, _) in zip(xs, xs[1:]):
            assert x2 >= x1 + w1 - EPS, "images in a row overlap"
