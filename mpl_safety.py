# mpl_safety.py
"""One process-wide lock serialising every matplotlib operation.

Streamlit runs each browser session's script in its own thread. Matplotlib's
pyplot figure manager, the Agg renderer and freetype's shared font cache are
all global, non-thread-safe C state, so two sessions rendering charts at the
same moment can fault inside native code rather than raise. That hazard is
independent of any one chart's code and predates the current app.

Both the BUILD of a figure and its render/savefig/close must happen inside the
same critical section. Holding the lock for the build alone is not enough: a
figure that exists but has not been rasterised yet can still be torn down by
another thread's plt.close('all'), and its rasterisation would then race.

Cost is near zero. The GIL already prevents these paths from running in
parallel, so charts effectively serialise today; this makes that explicit and
safe rather than incidental.

Usage:
    from mpl_safety import MPL_LOCK, mpl_locked

    with MPL_LOCK:            # build + render + close together
        fig = build_chart(...)
        st.pyplot(fig)
        plt.close(fig)

    @mpl_locked               # self-contained: builds and closes internally
    def render_card_png(...): ...
"""

import functools
import threading

import matplotlib

# Headless Linux already defaults to Agg, but say so explicitly: an implicit
# GUI backend would add its own event loop and thread-affinity rules.
matplotlib.use('Agg', force=True)

# Re-entrant so a locked builder may call another locked helper, and so a
# `with MPL_LOCK:` call site may invoke an @mpl_locked function inside it.
MPL_LOCK = threading.RLock()


def mpl_locked(fn):
    """Run `fn` while holding MPL_LOCK.

    Only correct for functions that build AND finish with their figures
    internally (savefig/close before returning). If a function returns a live
    Figure for someone else to render, the caller must hold MPL_LOCK across
    both the call and the render instead.
    """

    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        with MPL_LOCK:
            return fn(*args, **kwargs)

    return _wrapped
