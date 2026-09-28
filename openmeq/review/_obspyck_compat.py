"""Monkeypatches for two ObsPyck 0.8.2 bugs found in the OpenMEQ-c3l.1 spike.

Reference implementation: ``scripts/obspyck_spike_smoke.py``'s
``apply_bugfix_patches`` (hard-coded to that spike's own throwaway reference
point). This module is the production, tested, project-agnostic version --
see ``docs/obspyck_notes.md`` §6 for the full empirical writeup of both bugs.

1. **``gk2lonlat``** (``obspyck.util.gk2lonlat``, also bound as
   ``obspyck.obspyck.gk2lonlat`` via a ``from .util import`` at import time)
   hard-codes ``pyproj.Transformer.from_crs(31468, 4326)`` (Gauss-Krueger
   zone 4 / DHDN, Bavaria's regional projection), regardless of the actual
   ``TRANS`` statement used to build this project's NLLoc grids. It is
   called with the raw ``x``/``y`` km offsets NLLoc prints on its
   ``HYPOCENTER`` line and used to set ``Origin.longitude``/``.latitude``
   directly -- with the wrong inverse projection this silently corrupts the
   saved event's real coordinates. :func:`make_gk2lonlat` builds the exact
   algebraic inverse of *this project's own* ``TRANS SIMPLE`` statement
   (``openmeq.nlloc_setup.write_control_common``, ``cfg.nlloc.grid.origin``)
   instead of a fixed EPSG code, and :func:`patch_gk2lonlat` installs it on
   both modules (patching only ``obspyck.util.gk2lonlat`` leaves
   ``obspyck.obspyck``'s own copy of the name -- bound at import time by its
   ``from .util import (..., gk2lonlat, ...)`` -- untouched).

2. **NLLoc phase-line parser** (``ObsPyck.loadNLLocOutput``) parses each
   NLLoc ``PHASE`` output line by fixed whitespace-split column index. The
   NonLinLoc build this project's own ``install.sh`` produces (v7.1.05)
   inserts an extra, literal, standalone ``>`` token into every ``PHASE``
   line (between the ``PriorWt`` and ``TTpred`` columns) that 0.8.2's parser
   does not expect, shifting every later field by one position and crashing
   with ``ValueError: could not convert string to float: '>'`` on every
   "do NLLoc" click -- even though NLLoc itself succeeds and writes a
   correct location. :func:`patch_nlloc_call` wraps
   ``obspyck.util.PROGRAMS["nlloc"]["Call"]`` to strip stray ``>`` tokens
   from the renamed summary file immediately after NLLoc runs and before
   ObsPyck re-reads it, with no change to ObsPyck's own parsing code needed.

Both patches must be (re)applied **after** each ``ObsPyck(...)``
construction: ``setup_external_programs()`` (called fresh from
``ObsPyck.__init__``) reassigns ``PROGRAMS['nlloc']['Call']`` every time, so
an NLLoc-call wrap installed before construction is silently discarded.
:func:`apply_obspyck_compat_patches` is safe to call as often as needed
(idempotent -- re-wrapping an already-wrapped ``Call`` is a no-op, detected
via a marker attribute) and is the one entry point the ObsPyck launcher
(M7.3) should use, once right after import and once again immediately after
every ``ObsPyck(...)`` construction.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openmeq.config import Config

#: NonLinLoc's own mean Earth radius (``external/NonLinLoc/src/include/geo.h``,
#: ``AVG_ERAD``, the GMT "Sphere" radius) and the derived km-per-degree
#: constant NLLoc's ``TRANS SIMPLE`` forward/inverse both use
#: (``GridLib.c``'s ``c111 = DEG2KM = PI * AVG_ERAD / 180.0``). Confirmed
#: directly against the installed NonLinLoc source (not the rounded
#: ``111.111`` figure ``docs/nlloc_interface.md``'s prose paraphrase uses),
#: so :func:`make_gk2lonlat`'s inverse is the *exact* algebraic inverse of
#: NLLoc's own forward transform, not just a close approximation.
_AVG_ERAD_KM = 6371.0087714
_DEG2KM = math.pi * _AVG_ERAD_KM / 180.0


def make_gk2lonlat(lat0: float, lon0: float) -> Callable:
    """Build a ``gk2lonlat``-signature-compatible replacement for a project
    using ``TRANS SIMPLE <lat0> <lon0> 0.0`` (no rotation -- the only
    ``TRANS`` variant/``rotAngle`` this project's control files ever write,
    ``openmeq.nlloc_setup.write_control_common``).

    Implements ``NonLinLoc/src/GridLib.c``'s ``rect2latlon`` for
    ``MAP_TRANS_SIMPLE`` with ``rotAngle=0`` (``map_cosang=1``,
    ``map_sinang=0``, so the rotation terms drop out)::

        lat = lat0 + y / DEG2KM
        lon = lon0 + x / (DEG2KM * cos(radians(lat)))   # note: this lat, not lat0

    the exact algebraic inverse of that same file's ``latlon2rect`` (the
    forward transform NLLoc itself runs when writing ``TRANS SIMPLE``
    control-file-relative coordinates), confirmed by round-tripping against
    a real NLLoc run's own ``HYPOCENTER``/``GEOGRAPHIC`` output lines
    (``tests/test_review_obspyck_compat.py``).

    Parameters
    ----------
    lat0, lon0 : float
        ``cfg.nlloc.grid.origin`` -- the same ``TRANS SIMPLE latOrig
        lonOrig`` this project's own NLLoc control files use.

    Returns
    -------
    Callable
        A function with ``obspyck.util.gk2lonlat``'s exact signature,
        ``(x, y, m_to_km=True) -> (lon, lat)``, accepting scalars or numpy
        arrays for ``x``/``y`` (``ObsPyck.util.readNLLocScatter`` calls it
        with arrays). ``m_to_km`` follows the original's own (confusingly
        named) convention: ``True`` (the default, and what both real
        ObsPyck call sites always pass) means ``x``/``y`` are already in
        km, matching NLLoc's own ``HYPOCENTER`` line; ``False`` means
        ``x``/``y`` are in metres.
    """

    def _gk2lonlat(x, y, m_to_km: bool = True):
        x_km = np.asarray(x, dtype=float) if m_to_km else np.asarray(x, dtype=float) / 1000.0
        y_km = np.asarray(y, dtype=float) if m_to_km else np.asarray(y, dtype=float) / 1000.0
        lat = lat0 + y_km / _DEG2KM
        lon = lon0 + x_km / (_DEG2KM * np.cos(np.radians(lat)))
        lon = np.where(lon < -180.0, lon + 360.0, np.where(lon > 180.0, lon - 360.0, lon))
        if np.ndim(x) == 0 and np.ndim(y) == 0:
            return float(lon), float(lat)
        return lon, lat

    return _gk2lonlat


def patch_gk2lonlat(lat0: float, lon0: float) -> Callable:
    """Install :func:`make_gk2lonlat`'s replacement on **both**
    ``obspyck.util.gk2lonlat`` and ``obspyck.obspyck.gk2lonlat``.

    ``obspyck.obspyck`` does ``from .util import (..., gk2lonlat, ...)`` at
    import time, which copies the name into ``obspyck.py``'s own module
    namespace -- reassigning ``obspyck.util.gk2lonlat`` alone does not
    affect the call inside ``ObsPyck.loadNLLocOutput`` (confirmed by the
    OpenMEQ-c3l.1 spike testing both separately, ``docs/obspyck_notes.md``
    §6a). Both names must be reassigned to the same replacement function.

    Parameters
    ----------
    lat0, lon0 : float
        ``cfg.nlloc.grid.origin``.

    Returns
    -------
    Callable
        The installed replacement function (for tests / introspection).
    """
    import obspyck.obspyck as op
    import obspyck.util as ou

    replacement = make_gk2lonlat(lat0, lon0)
    ou.gk2lonlat = replacement
    op.gk2lonlat = replacement
    return replacement


def _strip_stray_gt_tokens(text: str) -> str:
    """Remove stray standalone ``>`` tokens from NLLoc ``PHASE`` data lines.

    NonLinLoc 7.1.05 inserts a literal ``>`` token between the ``PriorWt``
    and ``TTpred`` columns of every ``PHASE`` line in its summary output,
    which ``ObsPyck.loadNLLocOutput``'s fixed-column-index parser (written
    against an older NonLinLoc that never emitted this token) does not
    expect, raising ``ValueError: could not convert string to float: '>'``.
    Operates only between a ``PHASE ID`` header line and the matching
    ``END_PHASE`` line, leaving every other line (including the
    ``HYPOCENTER``/``GEOGRAPHIC`` lines :func:`make_gk2lonlat`'s caller
    reads) untouched.
    """
    lines = text.splitlines(keepends=True)
    in_phase = False
    fixed = []
    for line in lines:
        if line.startswith("PHASE ID"):
            in_phase = True
            fixed.append(line)
        elif line.startswith("END_PHASE"):
            in_phase = False
            fixed.append(line)
        elif in_phase and line.strip():
            newline = "\n" if line.endswith("\n") else ""
            parts = [p for p in line.split() if p != ">"]
            fixed.append(" ".join(parts) + newline)
        else:
            fixed.append(line)
    return "".join(fixed)


def patch_nlloc_call() -> None:
    """Wrap ``obspyck.util.PROGRAMS["nlloc"]["Call"]`` to strip stray ``>``
    tokens (:func:`_strip_stray_gt_tokens`) from the renamed NLLoc summary
    file immediately after NLLoc runs and before ``ObsPyck.loadNLLocOutput``
    re-reads it.

    ``setup_external_programs()`` reassigns ``PROGRAMS['nlloc']['Call']``
    fresh on **every** ``ObsPyck()`` construction (``obspyck/util.py``), so
    this must be called again after each one -- see
    :func:`apply_obspyck_compat_patches`. Idempotent: if ``Call`` is already
    one of this function's wrappers (checked via a marker attribute), it is
    left alone rather than double-wrapped.

    ``obspyck.util.PROGRAMS['nlloc']`` itself only gains a ``'Call'`` key
    once ``setup_external_programs()`` has actually run at least once (its
    module-level initial value has only a ``'filenames'`` key -- confirmed
    directly against the installed ``obspyck.util`` source); this is a no-op
    when called before that has happened (e.g. the launcher's "once after
    import" call, before any ``ObsPyck(...)`` has been constructed in this
    process yet) rather than raising ``KeyError`` -- the launcher's
    post-construction call always finds and wraps the real ``Call`` once it
    exists (confirmed empirically: the OpenMEQ-c3l.3 batch loop's
    requires_gui end-to-end test crashed with exactly this ``KeyError``
    until this guard was added).
    """
    import obspyck.util as ou

    prog_dict = ou.PROGRAMS["nlloc"]
    current_call = prog_dict.get("Call")
    if current_call is None:
        return
    if getattr(current_call, "_openmeq_patched", False):
        return

    original_call = current_call

    def _wrapped_call(prog_dict, controlfilename):
        result = original_call(prog_dict, controlfilename)
        summary = prog_dict["files"]["summary"]
        try:
            with open(summary) as f:
                text = f.read()
        except OSError:
            return result
        with open(summary, "w") as f:
            f.write(_strip_stray_gt_tokens(text))
        return result

    _wrapped_call._openmeq_patched = True
    prog_dict["Call"] = _wrapped_call


def apply_obspyck_compat_patches(cfg: Config) -> None:
    """Apply both ObsPyck 0.8.2 bugfix monkeypatches (:func:`patch_gk2lonlat`,
    :func:`patch_nlloc_call`) using this project's own NLLoc reference point.

    Call once after importing ``obspyck``, and **again immediately after
    every** ``ObsPyck(...)`` **construction** (``patch_nlloc_call``'s
    docstring explains why; ``patch_gk2lonlat`` only needs to run once per
    process since ``obspyck.util``/``obspyck.obspyck`` module objects don't
    get recreated per instance, but re-running it is harmless).

    Parameters
    ----------
    cfg : openmeq.config.Config
        Only ``cfg.nlloc.grid.origin`` (``[lat, lon]``) is used.
    """
    lat0, lon0 = cfg.nlloc.grid.origin
    patch_gk2lonlat(lat0, lon0)
    patch_nlloc_call()
