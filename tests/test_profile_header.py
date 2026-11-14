"""views/player_profile.py — the header's one line of facts (no app boot)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from views import player_profile  # noqa: E402

facts_line = player_profile.facts_line


def test_every_known_fact_is_on_the_line_in_one_order():
    assert facts_line(team='Vianense', position='LCB', age=33, foot='Left', height=184, weight=78,
                      nationality='Portugal', birthplace='Brazil') == (
        '**Vianense** · LCB · 33 years · Left foot · 184 cm · 78 kg · Nationality: Portugal · Born: Brazil')


def test_a_fact_we_do_not_have_is_left_out():
    # every way the bio table says "not known"
    for missing in (None, float('nan'), np.nan, pd.NA, pd.NaT, '', '  ', 'N/A', 'Unknown', 0, 0.0, '0'):
        line = facts_line(team='Clube A', position='GK', age=missing, foot=missing, height=missing,
                          weight=missing, nationality=missing, birthplace=missing)
        assert line == '**Clube A** · GK', (missing, line)
    assert facts_line() == ''
    assert facts_line(position='CF') == 'CF'                 # no club: nothing bold, nothing dangling
    assert facts_line(team=['a', 'b'], position='CF') == 'CF'    # not a scalar: not a fact


def test_numbers_print_whole_whatever_type_the_column_has():
    # one null in the bio table makes the whole column float
    assert facts_line(age=24.0, height=184.0, weight=78.0) == '24 years · 184 cm · 78 kg'
    assert facts_line(age=np.float64(24.7), height=np.int64(184), weight='78.4') == '24 years · 184 cm · 78 kg'
    assert facts_line(height='tall', weight=-1, age=float('inf')) == ''


def test_names_are_literal_text():
    # a trailing space left the asterisks on the page; Markdown characters are escaped
    assert facts_line(team='Mortágua ', position='CB') == '**Mortágua** · CB'
    line = facts_line(team='A*B_C', birthplace='Costa [Rica] $1')
    assert line == '**A\\*B\\_C** · Born: Costa \\[Rica\\] \\$1'
