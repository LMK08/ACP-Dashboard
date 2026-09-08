"""Vectorised membership tests on Wyscout's ``type.secondary`` tag lists.

Every event carries ``type.secondary``: a list (ndarray after parquet) of
tags such as ``'recovery'`` or ``'progressive_pass'``. The dashboard used to
test membership with ``Series.apply(lambda x: tag in x)`` — one Python call
per event per tag, which cost the two team pages ~9 s of every cold render
(3.3M lambda calls for the season report alone, 2026-09 profile).

:class:`TagIndex` explodes the column ONCE (C-level) and answers each tag
with a numpy equality + scatter, ~20x faster on a 1.1M-row season and
byte-identical to the lambda for list / tuple / set / ndarray cells. Cells
that are None / NaN / empty carry no tags. (A bare string cell equal to the
tag would count as carrying it — the lambdas said False — no such cells
exist in Wyscout exports.)
"""
import numpy as np
import pandas as pd

__all__ = ['TagIndex', 'has_tag', 'tag_flags']


class TagIndex:
    """Membership lookups for many tags over one ``type.secondary`` column."""

    def __init__(self, secondary):
        if secondary is None:
            secondary = pd.Series(dtype='object')
        self.index = secondary.index
        self._n = len(secondary)
        # Positional copy: explode() needs no index uniqueness this way, and
        # the answer is re-labelled with the caller's index at the end.
        vals = pd.Series(np.asarray(secondary, dtype=object),
                         index=pd.RangeIndex(self._n), dtype='object')
        ex = vals.explode()
        ex = ex[ex.notna()]
        self._pos = ex.index.to_numpy()
        self._tags = ex.to_numpy(dtype=object)

    def has(self, tag):
        """Boolean Series (caller's index): does the event carry *tag*?"""
        mask = np.zeros(self._n, dtype=bool)
        if self._tags.size:
            mask[self._pos[self._tags == tag]] = True
        return pd.Series(mask, index=self.index)

    def flags(self, tags):
        return {t: self.has(t) for t in tags}


def has_tag(secondary, tag):
    """One-off membership test; build a :class:`TagIndex` for several tags."""
    return TagIndex(secondary).has(tag)


def tag_flags(secondary, tags):
    return TagIndex(secondary).flags(tags)
