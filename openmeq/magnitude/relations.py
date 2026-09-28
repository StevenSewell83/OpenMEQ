"""Local magnitude (ML) attenuation relations (M8.2 / OpenMEQ-625.2).

This module implements the pure-formula layer that turns a Wood-Anderson
station amplitude and a hypocentral distance into a station ML:

- Hutton & Boore (1987), in the "/100" form (ARCHITECTURE.md's magnitude
  section; see the module-level docstring below for the exact formula and
  its equivalence to the original Richter-anchored form).
- A locally-calibrated variant of the same 3-coefficient formula
  (``cfg.magnitude.ml.local.coefficients``).
- A locally-calibrated distance table of -logA0(R) values, linearly
  interpolated and clamped outside the table's range
  (``cfg.magnitude.ml.local.file``; format documented in
  docs/ARCHITECTURE.md's "Local ML calibration file format" section).
- An optional per-station correction, added after either relation
  (``cfg.magnitude.ml.station_corrections``).

Upstream (M8.1): amplitudes are measured in **metres** (a `StationAmplitude`-
shaped value); this module is the single named place that converts to mm
for the log10 formula (see :func:`station_ml`).
Downstream (M8.3): calls :func:`station_ml` once per station amplitude, then
aggregates station MLs to an event ML (median/mean) and writes QuakeML.

Caching note
------------
The local calibration table and the resolved station-correction dict are
cached at module level, keyed by the resolved file path (or ``id(cfg)`` for
an inline mapping), so repeated calls in the same process don't re-read the
file/parse the mapping. Per ``openmeq.parallel``'s worker model (spawn, one
process per worker), **each pmap worker process gets its own copy of this
module and therefore its own cache** -- there is no cross-process sharing,
matching M8.1's response-cache pattern.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from openmeq.config import Config, ConfigError

# ---------------------------------------------------------------------------
# module-level caches (per-process; see module docstring)
# ---------------------------------------------------------------------------

#: resolved file path (str) -> loaded table (np.ndarray, columns [distance_km, neg_log_a0])
_TABLE_CACHE: dict[str, np.ndarray] = {}

#: cache key (resolved file path, or id(cfg) for an inline mapping) -> {"NET.STA": correction}
_CORRECTIONS_CACHE: dict[object, dict[str, float]] = {}

#: Count of interp_neg_log_a0 calls whose distance fell outside the table's
#: range and were therefore clamped rather than interpolated, so M8.3's
#: diagnostics can report how many stations were extrapolated. Module-level
#: (per-process; see module docstring), reset with :func:`reset_clamp_counter`.
_CLAMP_COUNT = 0


def reset_clamp_counter() -> None:
    """Reset the module-level out-of-range clamp counter to 0."""
    global _CLAMP_COUNT
    _CLAMP_COUNT = 0


def get_clamp_counter() -> int:
    """Return the number of :func:`interp_neg_log_a0` calls clamped so far
    (since the last :func:`reset_clamp_counter`, or process start)."""
    return _CLAMP_COUNT


def clear_caches() -> None:
    """Clear the module-level table/corrections caches (tests only; a long
    running process never needs this since a resolved file path is a stable
    cache key for the lifetime of a config)."""
    _TABLE_CACHE.clear()
    _CORRECTIONS_CACHE.clear()


# ---------------------------------------------------------------------------
# local calibration table
# ---------------------------------------------------------------------------


def load_neg_log_a0_table(path: str | Path) -> np.ndarray:
    """Load and validate a local -logA0(R) calibration table.

    File format (docs/ARCHITECTURE.md's local ML calibration section): a CSV
    with header ``distance_km,neg_log_a0``, distance strictly increasing, at
    least 2 rows, no missing/NaN values, e.g.::

        distance_km,neg_log_a0
        0,1.4
        10,1.6
        50,2.2
        100,3.0
        200,3.4

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.

    Returns
    -------
    np.ndarray
        Shape ``(n, 2)``, columns ``[distance_km, neg_log_a0]``, sorted by
        distance (already ascending, but sorted defensively).

    Raises
    ------
    ConfigError
        If the file is missing, has fewer than 2 rows, contains a NaN, or
        distances are not strictly increasing. The error names the file path
        and, for a bad row, the offending row number (1-based, counting the
        header as row 1, matching what a spreadsheet/text editor shows).
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Local ML calibration table not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - pandas' own message is enough
        raise ConfigError(f"Could not parse local ML calibration table {path}: {exc}") from exc

    expected_cols = ["distance_km", "neg_log_a0"]
    if list(df.columns) != expected_cols:
        raise ConfigError(f"{path}: expected header {expected_cols}, got {list(df.columns)}")

    if len(df) < 2:
        raise ConfigError(
            f"{path}: local ML calibration table needs at least 2 rows, got {len(df)}"
        )

    for i, row in df.iterrows():
        row_num = i + 2  # +1 for 0-based index, +1 for the header row
        if pd.isna(row["distance_km"]) or pd.isna(row["neg_log_a0"]):
            raise ConfigError(f"{path}: row {row_num} has a missing/NaN value")

    distances = df["distance_km"].to_numpy(dtype=float)
    if np.any(np.diff(distances) <= 0):
        bad_idx = int(np.argmax(np.diff(distances) <= 0)) + 1
        row_num = bad_idx + 2
        raise ConfigError(
            f"{path}: row {row_num} has distance_km={distances[bad_idx]} which does not strictly "
            f"increase over the previous row's distance_km={distances[bad_idx - 1]}"
        )

    return df[expected_cols].to_numpy(dtype=float)


def interp_neg_log_a0(distance_km: float, table: np.ndarray) -> float:
    """Interpolate -logA0(R) from a loaded calibration table.

    Uses ``numpy.interp`` (piecewise-linear interpolation). Note that
    ``numpy.interp``'s default behaviour for an ``x`` outside the table's
    range is to **clamp to the first/last table value rather than
    extrapolate linearly** -- this looks like a bug at a glance but is the
    explicitly required behaviour here (see docs/ARCHITECTURE.md and this
    task's design notes), so it is relied on rather than reimplemented.

    Every out-of-range call increments the module-level clamp counter (see
    :func:`get_clamp_counter`), so M8.3's diagnostics can report how many
    stations were extrapolated.

    Parameters
    ----------
    distance_km : float
        Hypocentral distance, km.
    table : np.ndarray
        Shape ``(n, 2)``, columns ``[distance_km, neg_log_a0]``, as returned
        by :func:`load_neg_log_a0_table`.

    Returns
    -------
    float
        The interpolated (or clamped) -logA0 value.
    """
    global _CLAMP_COUNT
    distances = table[:, 0]
    values = table[:, 1]
    if distance_km < distances[0] or distance_km > distances[-1]:
        _CLAMP_COUNT += 1
    return float(np.interp(distance_km, distances, values))


def _get_local_table(file_path: Path) -> np.ndarray:
    """Load (and cache, per-process, keyed by resolved path) the local
    calibration table at ``file_path``."""
    key = str(Path(file_path).resolve())
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = load_neg_log_a0_table(file_path)
    return _TABLE_CACHE[key]


# ---------------------------------------------------------------------------
# station corrections
# ---------------------------------------------------------------------------


def load_station_corrections(cfg: Config) -> dict[str, float]:
    """Resolve ``cfg.magnitude.ml.station_corrections`` to a
    ``{"NET.STA": correction}`` dict.

    - A string/Path value is treated as a CSV path with header
      ``net,sta,correction``, and loaded (result cached per-process, keyed by
      the resolved path).
    - A mapping is used directly (keys are already ``"NET.STA"``, per
      docs/ARCHITECTURE.md §6 "Station keys: NET.STA everywhere").
    - Empty/``None`` returns ``{}``.

    Parameters
    ----------
    cfg : Config
        The full top-level config (this function reads
        ``cfg.magnitude.ml.station_corrections``).

    Returns
    -------
    dict[str, float]
        ``{"NET.STA": correction}``. A station absent from the dict defaults
        to a 0.0 correction (see :func:`station_ml`).
    """
    source = cfg.magnitude.ml.station_corrections

    if source is None:
        return {}

    if isinstance(source, dict):
        if not source:
            return {}
        return dict(source)

    path = Path(source)
    key = str(path.resolve())
    if key in _CORRECTIONS_CACHE:
        return _CORRECTIONS_CACHE[key]

    if not path.is_file():
        raise ConfigError(f"Station corrections CSV not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - pandas' own message is enough
        raise ConfigError(f"Could not parse station corrections CSV {path}: {exc}") from exc

    expected_cols = ["net", "sta", "correction"]
    if list(df.columns) != expected_cols:
        raise ConfigError(f"{path}: expected header {expected_cols}, got {list(df.columns)}")

    corrections = {
        f"{row['net']}.{row['sta']}": float(row["correction"]) for _, row in df.iterrows()
    }
    _CORRECTIONS_CACHE[key] = corrections
    return corrections


# ---------------------------------------------------------------------------
# station ML
# ---------------------------------------------------------------------------


def station_ml(amplitude_m: float, distance_km: float, cfg: Config, net: str, sta: str) -> float:
    """Compute a station ML from a Wood-Anderson amplitude and distance.

    ``ML = log10(A_mm) + a*log10(R/100) + b*(R-100) + c`` (Hutton & Boore
    1987 form, or the equivalent locally-calibrated 3-coefficient variant),
    or ``ML = log10(A_mm) + neg_log_a0(R)`` (local distance-table variant),
    plus an optional per-station correction.

    Parameters
    ----------
    amplitude_m : float
        Zero-to-peak Wood-Anderson amplitude, in **metres** (M8.1's
        convention -- upstream `StationAmplitude` objects carry amplitude in
        metres). This is the single named point where it is converted to mm
        for the log10 formula.
    distance_km : float
        Hypocentral distance, km.
    cfg : Config
        The full top-level config (per ARCHITECTURE.md §2's convention that
        step code holds one ``cfg`` with a section per attribute); this
        function reads ``cfg.magnitude.ml.*``.
    net, sta : str
        Network and station codes, used to look up
        ``cfg.magnitude.ml.station_corrections`` as ``f"{net}.{sta}"``
        (ARCHITECTURE.md §6's ``NET.STA`` convention -- not ``NET.STA.LOC``).

    Returns
    -------
    float
        The station ML, including any station correction.

    Raises
    ------
    ConfigError
        If ``cfg.magnitude.ml.relation`` is not ``"hutton_boore"`` or
        ``"local"``, or if it is ``"local"`` but neither coefficients nor a
        file are configured.
    """
    ml_cfg = cfg.magnitude.ml

    # Single named unit-conversion point (M8.1 note): amplitude arrives in
    # metres, the log10(A) formula wants millimetres.
    a_mm = amplitude_m * 1000.0

    relation = ml_cfg.relation
    if relation == "hutton_boore":
        hb = ml_cfg.hutton_boore
        ml = (
            math.log10(a_mm)
            + hb.a * math.log10(distance_km / 100.0)
            + hb.b * (distance_km - 100.0)
            + hb.c
        )
    elif relation == "local":
        coeffs = ml_cfg.local.coefficients
        if coeffs.a is not None:
            ml = (
                math.log10(a_mm)
                + coeffs.a * math.log10(distance_km / 100.0)
                + coeffs.b * (distance_km - 100.0)
                + coeffs.c
            )
        elif ml_cfg.local.file is not None:
            table = _get_local_table(ml_cfg.local.file)
            ml = math.log10(a_mm) + interp_neg_log_a0(distance_km, table)
        else:
            raise ConfigError(
                "magnitude.ml.relation is 'local' but neither "
                "magnitude.ml.local.coefficients nor magnitude.ml.local.file is set"
            )
    else:
        raise ConfigError(
            f"Unknown magnitude.ml.relation {relation!r}; expected 'hutton_boore' or 'local'"
        )

    corrections = load_station_corrections(cfg)
    ml += corrections.get(f"{net}.{sta}", 0.0)

    return ml
