"""A keyed selectbox whose OPTIONS change between runs.

Streamlit 1.50: the options are part of the widget's identity, so it is a new
widget, and unless its key is set in that run it falls back to index 0 — even
when its value is still one of the new options. Player Comparison's A /
template / B selectors therefore re-set their value every run; without it a
scope change left Player B on the group's first player, Player A ('Paulinho
vs Paulinho', 2026-10-27).

Newer Streamlit (seen on 1.67, 2026-11) keeps a value that is still valid.
The rule stays: re-setting the value every run is right under both, and it is
what the second test pins. The first test records which behaviour the
installed version has, so a change of version shows up here, not in a page.
No app boot.
"""
import streamlit
from streamlit.testing.v1 import AppTest

VERSION = tuple(int(part) for part in streamlit.__version__.split('.')[:2])


def _script(reset_every_run):
    import streamlit as st
    opts = ['first', 'kept', 'other']
    if st.session_state.get('wide'):
        opts = opts + ['new']                 # options change, 'kept' stays valid
    if reset_every_run:
        _v = st.session_state.get('pick')
        st.session_state['pick'] = _v if _v in opts else opts[0]
    elif st.session_state.get('pick') not in opts:
        st.session_state['pick'] = opts[0]
    st.selectbox('Pick', opts, key='pick')


def _after_options_change(reset_every_run):
    at = AppTest.from_function(_script, args=(reset_every_run,))
    at.run()
    at.selectbox(key='pick').set_value('kept').run()
    at.session_state['wide'] = True
    at.run()
    return at.selectbox(key='pick').value


def test_without_a_reset_the_value_depends_on_the_streamlit_version():
    value = _after_options_change(reset_every_run=False)
    if VERSION <= (1, 50):
        assert value == 'first'               # lost: the reason for the rule
    else:
        assert value in ('first', 'kept')     # 1.67 keeps it; never anything else


def test_re_setting_the_value_every_run_keeps_it():
    assert _after_options_change(reset_every_run=True) == 'kept'
