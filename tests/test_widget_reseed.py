"""Streamlit 1.50: a keyed selectbox whose OPTIONS change between runs is a
new widget (options are part of its identity), and unless its key is set in
that run it falls back to index 0 — even when its value is still one of the
new options. Player Comparison's A / template / B selectors therefore re-set
their value every run; without it a scope change left Player B on the
group's first player, Player A ('Paulinho vs Paulinho', 2026-10-27).

Pins the behaviour the rule exists for; no app boot.
"""
from streamlit.testing.v1 import AppTest


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


def test_a_kept_value_is_lost_when_options_change_without_a_reset():
    assert _after_options_change(reset_every_run=False) == 'first'


def test_re_setting_the_value_every_run_keeps_it():
    assert _after_options_change(reset_every_run=True) == 'kept'
