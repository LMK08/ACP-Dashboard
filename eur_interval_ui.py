"""Projected value — the range sentence under every value and the fee
calibration panel that backs it (Player Profile → Value tab).

Mirrors scoreline_ui.py: the maths lives in models/value/eur_intervals,
this module only renders. Never quote a range without the calibration
panel reachable next to it.
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme
from league_config import all_season_id_map
from models.value.cvi import CAMP_PROJECTED_EUR_PENALTY, PROJECTED_EUR_COEF, PROJECTED_EUR_EXP
from models.value.eur_intervals import (
    ENGINE_VALUE_TEMPER, HEADLINE_LEVEL, VALUE_BANDS, format_eur_range,
    format_eur_short, load_calibration, projected_eur_interval,
    range_support_reason,
)

_LEVEL_KEYS = ('0.50', '0.80')
_INK = '#1f2a24'
_PLOTLY_CFG = {'displaylogo': False, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']}


def _pct(level_key):
    return f"{float(level_key) * 100:.0f}%"


def _factor(calib, level_key):
    lv = (calib.get('levels') or {}).get(level_key) or {}
    return lv.get('factor') if lv.get('shipped') else None


def _k(calib, level_key):
    lv = (calib.get('levels') or {}).get(level_key) or {}
    return lv.get('k') if lv.get('shipped') else None


def headline_factor(calib=None):
    """The shipped headline (50%) factor, or None when it did not ship —
    the ONE gate every piece of copy must use before quoting a factor."""
    calib = load_calibration() if calib is None else calib
    return _factor(calib or {}, (calib or {}).get('headline_level', '0.50'))


def range_sentence(point, calib=None, gk=False, w_evidence=None, mins=None):
    """The line under a projected value: 'Likely fee if sold **€85k – €270k**
    · middle half of 14 real sales', or 'No likely-fee range — too few
    minutes' when the row sits outside the calibration's support. None for
    goalkeepers (their value is a different estimator and there are no
    keeper sales on record) and when no calibration is shipped."""
    calib = load_calibration() if calib is None else calib
    if gk or not calib or point is None or pd.isna(point):
        return None
    reason = range_support_reason(point, calib, w_evidence=w_evidence, mins=mins)
    if reason:
        return f"No likely-fee range — {reason} (outside the fee calibration's support)."
    lo, hi = projected_eur_interval(point, HEADLINE_LEVEL, calib, w_evidence=w_evidence, mins=mins)
    if lo is None:
        return None
    hk = calib.get('headline_level', '0.50')
    return (f"Likely fee if sold **{format_eur_range(lo, hi)}** · {_k(calib, hk)} of "
            f"{calib.get('n_calibration', '?')} real sales fell within this factor of the model value")


def help_text(calib=None, gk=False):
    """Hover help for the Projected value metric."""
    calib = load_calibration() if calib is None else calib
    base = curve_text()
    if gk:
        return ("Goalkeeper — legacy CVI→EUR value (the outfield ACP engine does "
                "not rate keepers). No likely-fee range: the keeper value is a "
                "different estimator and there are no goalkeeper sales on record.")
    f50, f80 = _factor(calib, '0.50'), _factor(calib, '0.80')
    if not f50:
        return base + " No fee calibration shipped yet, so no range is shown."
    n = calib.get('n_calibration', 0)
    sup = calib.get('support') or {}
    txt = (f"{base} Likely fee if sold = the band within which {_k(calib, '0.50')} of the {n} "
           f"real Liga 3 / Campeonato sales (built {calib.get('built', '?')}) fell relative to "
           f"the model value: ×/÷{f50:.1f}")
    if f80:
        txt += f"; {_k(calib, '0.80')} of {n} within ×/÷{f80:.1f}"
    txt += (f". Shown only inside the calibration's own support (value ≥ "
            f"{format_eur_short(sup.get('min_value_eur', 25000))}, enough minutes and evidence "
            f"behind the projection). Details and caveats: Value tab → fee calibration.")
    return txt


def curve_text():
    """The point-value formula, from the LIVE constants (so a retune can never
    leave this sentence describing the old curve)."""
    return ("ACP engine projection → EUR: percentile of the next-season "
            "projection (abs scale) × career-NPV age multiplier, through the "
            f"fee-calibrated CVI→EUR curve ({PROJECTED_EUR_COEF} × CVI^{PROJECTED_EUR_EXP} × "
            f"position multiplier, ×{CAMP_PROJECTED_EUR_PENALTY} for Campeonato seasons, "
            f"×{ENGINE_VALUE_TEMPER} temper, uncapped). No reliability ramp: the projection "
            "is already evidence-weighted.")


def pdf_range_text(point, calib=None, w_evidence=None, mins=None):
    """'85k-270k' for the one-pager 'Likely fee (EUR)' tile, or None. Short
    on purpose: the portrait A4 tile cell is ~24 mm wide at 11pt bold."""
    calib = load_calibration() if calib is None else calib
    lo, hi = projected_eur_interval(point, HEADLINE_LEVEL, calib, w_evidence=w_evidence, mins=mins)
    if lo is None:
        return None
    return f"{format_eur_short(lo).replace('€', '')}-{format_eur_short(hi).replace('€', '')}"


def _fee_scatter(calib):
    rows = pd.DataFrame(calib.get('calibration') or [])
    if rows.empty:
        return None
    labels = all_season_id_map()
    q50 = ((calib['levels'].get('0.50') or {}).get('q'))
    q80 = ((calib['levels'].get('0.80') or {}).get('q'))
    lo_v = float(rows['value_eur'].min()) / 2
    hi_v = float(rows['value_eur'].max()) * 2
    xs = np.geomspace(lo_v, hi_v, 50)
    fig = go.Figure()
    for q, name, alpha in ((q80, '80% band', 0.10), (q50, '50% band', 0.18)):
        if q is None:
            continue
        fig.add_trace(go.Scatter(
            x=np.concatenate([xs, xs[::-1]]),
            y=np.concatenate([xs * math.exp(q), (xs * math.exp(-q))[::-1]]),
            fill='toself', fillcolor=f'rgba(26,71,42,{alpha})',
            line=dict(width=0), hoverinfo='skip', name=name))
    fig.add_trace(go.Scatter(x=xs, y=xs, mode='lines', name='fee = value',
                             line=dict(color=theme.FOCUS, width=1.2, dash='dot'),
                             hoverinfo='skip'))
    colours = {'L3': theme.FOCUS, 'Camp': theme.AWAY_RED}
    for league, grp in rows.groupby('league'):
        fig.add_trace(go.Scatter(
            x=grp['value_eur'], y=grp['fee_eur'], mode='markers',
            name={'L3': 'Liga 3 sale', 'Camp': 'Campeonato sale'}.get(league, league),
            marker=dict(size=11, color=colours.get(league, '#888'), opacity=0.85,
                        line=dict(color='white', width=1)),
            text=[f"<b>{r.player_name}</b> · {labels.get(int(r.season_id), r.season_id)} · {r.role or ''}"
                  f"<br>model value {format_eur_short(r.value_eur)} → fee {format_eur_short(r.fee_eur)}"
                  f" (×{r.ratio:.2f})" for r in grp.itertuples()],
            hovertemplate='%{text}<extra></extra>'))
    fig.update_layout(
        template='plotly_white', paper_bgcolor=theme.FIGURE_BG, plot_bgcolor=theme.FIGURE_BG,
        height=480, margin=dict(l=60, r=20, t=50, b=50),
        title=dict(text='Real permanent sales vs the model value (log scales)',
                   font=dict(color=_INK, size=15), x=0.02),
        xaxis=dict(type='log', title='Model value at the pre-transfer season (EUR)',
                   gridcolor='rgba(0,0,0,0.08)'),
        yaxis=dict(type='log', title='Realised fee (EUR)', gridcolor='rgba(0,0,0,0.08)'),
        legend=dict(orientation='h', y=-0.18), font=dict(color=_INK))
    return fig


def render_eur_calibration_section(calib=None, key='eurcal'):
    """The calibration panel: what the range means, how well it held, and
    the sales behind it."""
    calib = load_calibration() if calib is None else calib
    if not calib:
        st.caption("No fee calibration has been shipped yet — projected values are "
                   "shown without a range.")
        return
    lv = calib.get('levels') or {}
    n = calib.get('n_calibration', 0)
    bias = calib.get('bias') or {}
    ex = calib.get('excluded') or {}
    em = calib.get('engine_meta') or {}
    f50, f80 = _factor(calib, '0.50'), _factor(calib, '0.80')
    st.markdown(
        f"**How the range is built.** Each real permanent sale in the club's fee "
        f"records is paired with the player's engine value for the season before the "
        f"sale; the range is a split-conformal prediction interval on "
        f"log(fee ÷ value): the k-th smallest |log error| brackets the value. "
        f"Calibration set: **{n} sales** "
        f"({bias.get('L3', {}).get('n', 0)} Liga 3, {bias.get('Camp', {}).get('n', 0)} Campeonato); "
        f"excluded {ex.get('synthetic', 0)} hand-estimated anchors, {ex.get('offer', 0)} unaccepted offer(s), "
        f"{ex.get('no_engine_row', 0)} sale(s) without an engine projection. "
        f"Built {calib.get('built', '?')} against engine {em.get('rating_version', '?')} "
        f"(data through {em.get('data_through', '?')}). Older sales are valued with today's "
        f"engine and pool, with the value curve's age effect back-dated to the sale"
        + (f"; {calib.get('n_mid_season')} of the sales happened in January, so their season "
           f"row also contains matches played after the sale." if calib.get('n_mid_season') else "."))
    sup = calib.get('support') or {}
    vr = sup.get('value_range_eur') or [None, None]
    st.caption(f"Ranges are shown only above the calibration's floors — value ≥ "
               f"{format_eur_short(sup.get('min_value_eur', 25000))}, minutes ≥ "
               f"{sup.get('min_mins_played') or 0:.0f}, evidence weight ≥ "
               f"{sup.get('min_w_evidence') or 0:.2f} (the sold players' minima); below them the "
               f"value is shown alone."
               + (f" The sold players' model values ran {format_eur_short(vr[0])} – "
                  f"{format_eur_short(vr[1])}, so a range above that is the same width "
                  f"extrapolated." if vr[0] else ""))
    c = st.columns(5)
    c[0].metric("Real sales", f"{n}")
    c[1].metric("Likely range (50%)", f"×/÷{f50:.2f}" if f50 else "not shipped",
                help=(f"{_k(calib, '0.50')} of the {n} calibration sales fell within this factor of "
                      f"the model value — the band's edge is the {_k(calib, '0.50')}-th sale."
                      if f50 else "Ships once its edge is not one of the two largest residuals."))
    c[2].metric("Wide range (80%)", f"×/÷{f80:.2f}" if f80 else "not shipped",
                help=(f"{_k(calib, '0.80')} of the {n} sales fell within this factor."
                      if f80 else "Not shipped: at this sample size the 80% edge would be one of the "
                                  "two largest residuals (needs k ≤ n−2). The 90% level needs 29 sales."))
    _inside_help = ("By construction: each band's edge IS the k-th sale, so k of the n "
                    "sales sit at or inside it whatever the residuals are (the same ≤ rule "
                    "ticks the table below). A property of the method, not a test — the "
                    "prospective check below is.")
    for col, k in zip(c[3:], _LEVEL_KEYS):
        shipped = lv.get(k, {}).get('shipped')
        col.metric(f"Inside {_pct(k)} band",
                   f"{lv[k].get('k', 0)} of {n}" if shipped else "—",
                   help=_inside_help)
    cd = calib.get('centre_drift') or {}
    if cd.get('flag'):
        st.warning(cd.get('note', 'The point value and the band no longer share a centre.'))
    pro = calib.get('prospective') or {}
    if pro.get('n'):
        parts = [f"{pro.get(f'hits_{k}', 0)} of {pro.get(f'n_{k}', 0)} inside the {_pct(k)} band"
                 for k in _LEVEL_KEYS if pro.get(f'n_{k}')]
        st.info("**Since first pinned** (sales added after a calibration was written, scored "
                "against the band that was live at the time, with the model value recomputed by "
                "today's engine — the only truly out-of-sample check): " + "; ".join(parts) + ".")
    else:
        st.caption("Prospective check: no sale has been added since this calibration was "
                   "first written. Every fee added from now on is scored against the band "
                   "that was live when it was added.")
    fig = _fee_scatter(calib)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, theme=None, config=_PLOTLY_CFG,
                        key=f'{key}_scatter')
    med = ", ".join(f"{k} {v['median_ratio']:.2f} (n={v['n']})"
                    for k, v in bias.items() if v.get('median_ratio') is not None)
    st.caption(f"Median fee ÷ value: {med}. A median near 1.0 means the point is centred; "
               f"the range width is the same for every player.")
    rows = pd.DataFrame(calib.get('calibration') or [])
    if not rows.empty:
        labels = all_season_id_map()
        t = pd.DataFrame({
            'Player': rows['player_name'],
            'Season': rows['season_id'].map(lambda s: labels.get(int(s), s)),
            'League': rows['league'].map({'L3': 'Liga 3', 'Camp': 'Campeonato'}).fillna(rows['league']),
            'Role': rows['role'],
            'Model value': rows['value_eur'].map(format_eur_short),
            'Fee': rows['fee_eur'].map(format_eur_short),
            'Fee ÷ value': rows['ratio'].map(lambda r: f"×{r:.2f}"),
            'Inside 50%': rows['score'].map(lambda s: '✓' if s <= (lv.get('0.50', {}).get('q') or 0) else '—'),
        }).sort_values('Fee ÷ value')
        st.dataframe(t, hide_index=True, use_container_width=True, key=f'{key}_sales')
    strata = pd.DataFrame(calib.get('strata') or [])
    if not strata.empty and not rows.empty:
        # in-sample counts inside the POOLED band per slice (the same ≤ q rule
        # as the sales table) — shown so thin slices are visible, never as a test
        q50 = (lv.get('0.50') or {}).get('q'); q80 = (lv.get('0.80') or {}).get('q')
        rows['_band'] = rows['value_eur'].map(
            lambda v: next((lab for lo_, hi_, lab in VALUE_BANDS if lo_ <= v < hi_), VALUE_BANDS[-1][2]))
        col_of = {'league': 'league', 'role_group': 'role_group', 'value_band': '_band'}

        def _inside(kind, name, q):
            if q is None:
                return '—'
            g = rows[rows[col_of[kind]].astype(str) == str(name)]
            return f"{int((g['score'] <= q).sum())} of {len(g)}"
        s = pd.DataFrame({
            'Slice': strata['kind'].map({'league': 'League', 'role_group': 'Role group',
                                          'value_band': 'Value band'}).fillna(strata['kind']),
            'Group': strata['name'].map({'L3': 'Liga 3', 'Camp': 'Campeonato'}).fillna(strata['name']),
            'n': strata['n'],
            'Median fee ÷ value': strata['median_ratio'].map(lambda r: f"×{r:.2f}"),
            'In 50% band': [_inside(k, nm, q50) for k, nm in zip(strata['kind'], strata['name'])],
            'In 80% band': [_inside(k, nm, q80) for k, nm in zip(strata['kind'], strata['name'])],
        })
        st.dataframe(s, hide_index=True, use_container_width=True, key=f'{key}_strata')
        st.caption(f"Counts inside the one pooled band per slice (in-sample, not a test). One width "
                   f"for everyone: a slice earns its own width only from "
                   f"{calib.get('strata', [{}])[0].get('eligible_at_n', 15)} sales, and none is there yet. "
                   "Campeonato sales sit lower than the model (few of them). Goalkeepers get no "
                   "range at all: their value is a different estimator and there are no keeper sales.")
    exr = calib.get('excluded_rows') or []
    st.caption("Caveats: fees exist only for players who were sold, so this is 'if he sells, "
               "the likely fee', not a market-value confidence interval (about a third of moves "
               "at this level are free transfers — Transfermarkt transfer history, ~7,500 records, "
               "see models/eur_v2/realization.py). The curve's centre constants were tuned on the "
               "same sales, so the band is mildly optimistic; the prospective check is the honest "
               "number. Only a player's latest season carries a projection, so a real sale of a "
               "player who stayed in the data cannot always be paired, and a player who arrived "
               "from outside the data has only a post-transfer row, which is not used"
               + (f" (not paired: {', '.join(r['player_name'] for r in exr)})" if exr else "")
               + ". Campeonato sales sit below the model on average, so for Campeonato players the "
               "upper half of the range is the weaker half.")
