"""One section drawn at a time on the two team pages (2026-11 UX pass).

Team Analysis and the Opposition Report were 14 and 17 screens of every
visual at once. Both now carry the SAME section row (parity rule): the
sections are the pages' existing blocks, in their old order, nothing re-cut.
The Opposition Report adds its page-specific "Summary" in front.

Two ways to honour the row, and each page uses the one it needs:

* `Sections.is_open(name)` — plain `if`: a closed block does not run at all.
  Team Analysis uses this. Anything a LATER block needs (the starting XI,
  the OBV frames) is computed above the blocks, not inside one.

* `Sections.block(name)` — the block's code still RUNS, in its old order, but
  its output goes to a hidden container unless the section is open, and
  `Sections.drawing` is False inside it. The Opposition Report uses this,
  because its takeaways read what the Players / Set pieces / Form blocks
  compute and its PDF needs every block's tables and figures: there the
  figure helper queues a figure for the PDF instead of building it when
  `drawing` is False, and the Plotly charts are skipped. What still lands
  in the hidden container is text and small tables.

The section radio is keyed and its options are a constant tuple, so the
choice survives a change of team or opponent (widget-seeding rule).
"""
import contextlib

import streamlit as st

TEAM_SECTIONS = ("Overview", "Players", "Set pieces", "Form & shots", "Tactics")
OPP_SECTIONS = ("Summary",) + TEAM_SECTIONS

# The hidden container closed blocks write into (see Sections.block). The
# rule is injected once by app.py with the rest of the theme CSS.
SINK_KEY = "closed_sections_sink"
SINK_CSS = f'[class*="st-key-{SINK_KEY}"] {{ display: none !important; }}'


class Sections:
    """The section row of a page, and where each block's output goes.

    `side=True` leaves a column beside the row (`self.side`) for a control
    that must be on the page whatever the section, such as a PDF button.
    """

    def __init__(self, options, key, label, side=False):
        if side:
            row, self.side = st.columns([3, 1], vertical_alignment="center")
        else:
            row, self.side = st.container(), None
        with row:
            self.current = st.radio(label, options, horizontal=True, key=key,
                                    label_visibility="collapsed")
        self.body = st.container()                 # the open section
        self.sink = st.container(key=SINK_KEY)     # everything else (hidden)
        self.drawing = True                        # False inside a closed block

    def is_open(self, name):
        return self.current == name

    @contextlib.contextmanager
    def block(self, name, into=None):
        """Run a block as part of section `name`. Its output goes to `into`
        (default: the page body) when that section is open, else to the
        hidden sink with `self.drawing` False. Yields whether it is open."""
        is_open = self.current == name
        previous, self.drawing = self.drawing, is_open
        try:
            with (into if into is not None else self.body) if is_open else self.sink:
                yield is_open
        finally:
            self.drawing = previous
