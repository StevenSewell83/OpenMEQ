"""In-process ObsPyck launcher (M7.3 / OpenMEQ-c3l.3).

Run as ``python -m openmeq.review._launch_obspyck <obspyck argv>`` -- this is
the launch mechanism the OpenMEQ-c3l.1 spike found is *required*
(``docs/obspyck_notes.md`` §6a): the two ObsPyck 0.8.2 bugfix monkeypatches
(``openmeq.review._obspyck_compat``) only affect code running in the *same*
Python process that later calls ``obspyck.obspyck.main()`` -- patching
``obspyck.util.gk2lonlat``/``obspyck.obspyck.gk2lonlat`` from a parent
process has no effect on a child ``subprocess.Popen(["obspyck", ...])``,
since that spawns a *separate* interpreter that imports ``obspyck`` fresh,
unpatched. :mod:`openmeq.review.batch` (M7.3's batch loop) therefore never
launches the pip-installed ``obspyck`` console script; it always launches
this module instead, via ``[sys.executable, "-m",
"openmeq.review._launch_obspyck", ...]``.

Why the OpenMEQ config comes from an environment variable, not argv
---------------------------------------------------------------------
This launcher needs ``cfg.nlloc.grid.origin`` (to build the ``gk2lonlat``
replacement, see ``_obspyck_compat.apply_obspyck_compat_patches``), which
means it needs to load OpenMEQ's own YAML config. But ObsPyck's own
command-line parser already claims ``-c``/``--config`` for *its* ini file
(see the module-level ``COMMANDLINE_OPTIONS`` in ``obspyck.util``), and this
launcher's whole ``sys.argv[1:]`` is handed to ``obspyck.obspyck.main()``
completely unchanged (so any future ObsPyck flag keeps working without this
module needing to know about it). Reusing ``-c`` for two different files
would be ambiguous, so the OpenMEQ config path travels via the
``OPENMEQ_CONFIG`` environment variable instead, which
:mod:`openmeq.review.batch` sets on every subprocess it launches.

Reapplying the compat patches after every ``ObsPyck(...)`` construction
-------------------------------------------------------------------------
``apply_obspyck_compat_patches`` must run *after* every ``ObsPyck(...)``
construction, not just once at import (``setup_external_programs()``,
called fresh from ``ObsPyck.__init__``, reassigns
``PROGRAMS['nlloc']['Call']`` every time -- see
``openmeq.review._obspyck_compat``'s own docstring). Since this launcher
hands off to ``obspyck.obspyck.main()`` (which constructs the single
``ObsPyck(...)`` instance internally, then never returns -- it ends via
``os._exit(qApp.exec_())``), there is no call site *between* construction
and ``qApp.exec_()`` for this module to hook into directly. Instead,
:func:`_wrap_obspyck_construction` wraps ``ObsPyck.__init__`` itself (once,
idempotently) so the patches are reapplied immediately after construction,
regardless of who constructs it or how.

Test-only autosave hook
------------------------
Real manual review needs a human to click "save Event locally"; the
OpenMEQ-c3l.3 requires_gui end-to-end test (``tests/test_review_batch.py``)
instead needs to *programmatically* trigger a save from outside the GUI
event loop, against a real, launched ObsPyck window, the same way the
OpenMEQ-c3l.1 spike's own exit-code driver did (``docs/obspyck_notes.md``
§9): a ``QTimer.singleShot`` callback that nudges one pick's time (so the
save is a genuine, observable edit, mirroring the spike's own "make one
pick change" step), calls ``setXMLEventID()``/``save_event_locally()``
(the exact methods the real "save Event locally" button calls), and then
quits the application the same way a real window close would. This is
gated behind the ``OPENMEQ_REVIEW_TEST_AUTOSAVE`` environment variable
(unset/anything other than ``"1"`` in real, non-test use) so it can never
fire in a real reviewer's session.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from openmeq.config import load_config
from openmeq.review._obspyck_compat import apply_obspyck_compat_patches

if TYPE_CHECKING:
    from openmeq.config import Config

#: Env var carrying the OpenMEQ run config path (see module docstring's
#: "Why the OpenMEQ config comes from an environment variable" section).
OPENMEQ_CONFIG_ENV = "OPENMEQ_CONFIG"

#: Test-only autosave hook env var (see module docstring). ``"1"`` enables
#: it; any other value (including unset) leaves real ObsPyck behaviour
#: completely untouched.
AUTOSAVE_ENV = "OPENMEQ_REVIEW_TEST_AUTOSAVE"

#: Milliseconds to wait after window construction before firing the
#: test-only autosave hook -- long enough for the window's own constructor
#: (drawing axes, loading picks, ...) to finish first.
_AUTOSAVE_DELAY_MS = 500


def _schedule_test_autosave(window) -> None:
    """Schedule the test-only autosave hook (see module docstring).

    Parameters
    ----------
    window
        The just-constructed ``obspyck.obspyck.ObsPyck`` instance.
    """
    from PyQt5 import QtCore, QtWidgets

    def _do_autosave() -> None:
        try:
            if window.catalog and window.catalog[0].picks:
                pick = window.catalog[0].picks[0]
                pick.time = pick.time + 0.1
            window.setXMLEventID()
            window.save_event_locally()
        finally:
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.quit()

    QtCore.QTimer.singleShot(_AUTOSAVE_DELAY_MS, _do_autosave)


def _wrap_obspyck_construction(cfg: Config) -> None:
    """Reapply the compat patches after every ``ObsPyck(...)`` construction.

    Also fires the test-only autosave hook (:func:`_schedule_test_autosave`)
    when :data:`AUTOSAVE_ENV` is set to ``"1"``. Idempotent (a marker
    attribute on the wrapped ``__init__``), matching
    ``openmeq.review._obspyck_compat``'s own patches' idempotency
    convention, so calling this more than once in the same process (it
    isn't, in practice -- :func:`main` calls it exactly once -- but a test
    driving this module in-process might construct more than one launcher)
    never double-wraps.
    """
    import obspyck.obspyck as op

    original_init = op.ObsPyck.__init__
    if getattr(original_init, "_openmeq_patched", False):
        return

    def _patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        apply_obspyck_compat_patches(cfg)
        if os.environ.get(AUTOSAVE_ENV) == "1":
            _schedule_test_autosave(self)

    _patched_init._openmeq_patched = True
    op.ObsPyck.__init__ = _patched_init


def main(argv: list[str] | None = None) -> int:
    """Entry point: apply the compat patches, then hand off to ObsPyck.

    Parameters
    ----------
    argv : list of str or None, optional
        ObsPyck's own command-line arguments (``-c``/``-t``/``-d``/
        ``--event``/mseed/StationXML). Defaults to ``sys.argv[1:]`` (the
        real ``python -m openmeq.review._launch_obspyck ...`` invocation);
        an explicit list is for in-process testing only.

    Returns
    -------
    int
        Only returned if ``obspyck.obspyck.main()`` itself returns instead
        of calling ``os._exit`` (e.g. a test that stubs it out); real use
        never reaches this return, since ``main()`` calls ``os._exit``
        after the Qt event loop exits.

    Notes
    -----
    This function never touches ``QT_QPA_PLATFORM`` itself -- that decision
    belongs entirely to :mod:`openmeq.review.batch` (:func:`_resolve_launch_wrapper`),
    which knows whether a real ``$DISPLAY`` is attached and passes the
    right environment down to this process; forcing it here would hide a
    real reviewer's own display.
    """
    config_path = os.environ.get(OPENMEQ_CONFIG_ENV)
    if not config_path:
        print(f"{OPENMEQ_CONFIG_ENV} environment variable must be set", file=sys.stderr)
        return 1

    cfg = load_config(config_path)

    # Import triggers `from .util import (..., gk2lonlat, ...)` inside
    # obspyck.py, binding its own module-level copy of the name -- this is
    # exactly why `patch_gk2lonlat` (called by `apply_obspyck_compat_patches`
    # below) must patch both `obspyck.util.gk2lonlat` and
    # `obspyck.obspyck.gk2lonlat` (docs/obspyck_notes.md §6a).
    import obspyck.obspyck as op

    apply_obspyck_compat_patches(cfg)
    _wrap_obspyck_construction(cfg)

    resolved_argv = list(argv) if argv is not None else sys.argv[1:]
    sys.argv = [sys.argv[0] if sys.argv else "openmeq-obspyck-launcher", *resolved_argv]

    op.main()  # never returns in real use: ends via os._exit(qApp.exec_())
    return 0


if __name__ == "__main__":
    sys.exit(main())
