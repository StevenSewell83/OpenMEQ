"""MiniSEED data index (``openmeq.index``, M1.1, ARCHITECTURE.md §2/§4.1).

This module is the foundation of the whole pipeline (PLAN.md §6 Step 1):
every later step finds waveform files through a :class:`DataIndex` instead
of re-scanning directories. It builds and queries ``inventory/index.json``,
which maps each ``NET.STA.LOC.CHA`` seed id to the list of on-disk miniSEED
segments (file, start, end, sampling_rate) that hold it.

Design
------
* **Header-only everywhere in indexing.** :func:`_scan_file` calls
  ``obspy.read(path, headonly=True)``, which parses miniSEED headers
  without decompressing STEIM-encoded sample data -- the only way this
  scales to years of continuous data across dozens of channels.
* **Selective, native reads for queries.** :meth:`DataIndex.read` passes
  ``starttime``/``endtime`` straight to ``obspy.read()`` per file, which
  skips miniSEED records outside the window at the libmseed level. It never
  reads a whole file and calls ``.trim()`` afterwards.
* **Index scope vs. query scope.** :func:`build_index` scans **every** file
  matching ``cfg.data.file_pattern`` under ``cfg.data.waveform_dirs``,
  regardless of ``cfg.data.start``/``cfg.data.end``. Those bounds are a
  *query*-time concern, applied by callers through
  ``DataIndex.read(t1, t2)``/``DataIndex.files(t1, t2)``. This means
  changing the configured date range in a later run never requires
  re-scanning the archive -- only the index build (an I/O-bound directory
  walk plus header parse) is expensive; date filtering is a cheap in-memory
  comparison.
* **Gaps preserved, not filled.** :meth:`DataIndex.read` merges with
  ``stream.merge(method=0)`` (obspy's default), which produces masked
  arrays across any gap rather than interpolating across it. Downstream
  consumers that can't tolerate masked samples (filters, STA/LTA, ...) must
  handle this explicitly (``np.ma.compressed``, or ``Stream.split()``
  first).
* **Multi-segment / multi-channel files.** A single miniSEED file can hold
  more than one trace -- either several channels, or (as in the synthetic
  fixture's deliberate gap) two time segments of the *same* channel split
  around a real gap. :func:`_scan_file` emits one tuple per trace returned
  by ``obspy.read(headonly=True)``, so the same file path can legitimately
  appear more than once under the same seed id, or under several seed ids.

Performance
-----------
On a 28-core machine, the header scan is embarrassingly parallel (each file
is independent I/O + header parse), so :func:`build_index` runs it through
``openmeq.parallel.pmap`` and scales close to linearly with worker count up
to the point I/O saturates. ``pmap``'s default ``chunksize=1`` is fine for a
first cut; if scanning many small files ever becomes overhead-bound (lots of
tiny per-item IPC round trips relative to the header-parse cost), raising
``chunksize`` amortizes that overhead.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from obspy import Stream, UTCDateTime, read

from openmeq.log import get_logger
from openmeq.parallel import pmap

if TYPE_CHECKING:
    from openmeq.config import Config

logger = get_logger("index")

#: One on-disk segment: (absolute path, start ISO string, end ISO string,
#: sampling rate). Matches ARCHITECTURE.md §4.1's JSON array shape exactly,
#: so it round-trips through ``json.dump``/``json.load`` with no
#: translation.
Segment = tuple[str, str, str, float]

_INDEX_VERSION = 1


def _iso_z(t: UTCDateTime) -> str:
    """``UTCDateTime`` -> ISO-8601 string with a ``Z`` suffix.

    ``UTCDateTime.isoformat()`` does not append ``Z`` on its own (it just
    omits the offset); ARCHITECTURE.md §1 requires the explicit ``Z``.
    """
    return t.isoformat() + "Z"


def _scan_file(path: str) -> list[Segment]:
    """Header-only scan of one miniSEED file, returning one segment per trace.

    Top-level and picklable (required by ``openmeq.parallel.pmap`` when
    ``n_workers > 1``). Never raises: a file that can't be read as miniSEED
    (corrupted, truncated, or simply not miniSEED despite matching
    ``cfg.data.file_pattern``) is logged as a warning and skipped, returning
    an empty list, so one bad file never aborts an entire index build.

    Parameters
    ----------
    path : str
        Path to a candidate miniSEED file. A plain ``str`` (not ``Path``),
        since items sent through ``pmap`` to worker processes must be
        picklable and simple.

    Returns
    -------
    list of tuple
        ``(seed_id, start_iso, end_iso, sampling_rate)`` per trace found in
        the file's headers.
    """
    try:
        stream = read(path, headonly=True)
    except Exception as exc:  # noqa: BLE001 - any unreadable file is skippable
        logger.warning("Skipping unreadable file %s: %s", path, exc)
        return []

    segments: list[Segment] = []
    for trace in stream:
        seed_id = trace.get_id()
        segments.append(
            (
                seed_id,
                _iso_z(trace.stats.starttime),
                _iso_z(trace.stats.endtime),
                float(trace.stats.sampling_rate),
            )
        )
    return segments


@dataclass
class DataIndex:
    """In-memory index of every miniSEED segment under a set of root directories.

    Mirrors ``inventory/index.json`` (ARCHITECTURE.md §4.1) field for field.

    Attributes
    ----------
    version : int
        Index format version (currently always 1).
    created : str
        ISO-8601 ``...Z`` timestamp of when the index was built.
    roots : list of str
        Absolute paths of the scanned root directories
        (``cfg.data.waveform_dirs``, resolved).
    channels : dict
        ``{"NET.STA.LOC.CHA": [(abs_path, start_iso, end_iso,
        sampling_rate), ...]}``, each list sorted by start time.
    """

    version: int
    created: str
    roots: list[str]
    channels: dict[str, list[Segment]] = field(default_factory=dict)

    # -- I/O ---------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str) -> DataIndex:
        """Read an index previously written by :meth:`write` (or :func:`build_index`)."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        channels = {
            seed_id: [tuple(segment) for segment in segments]
            for seed_id, segments in data["channels"].items()
        }
        return cls(
            version=data["version"],
            created=data["created"],
            roots=list(data["roots"]),
            channels=channels,
        )

    def write(self, path: Path | str) -> None:
        """Write this index to ``path`` as JSON (ARCHITECTURE.md §4.1 shape)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "created": self.created,
            "roots": self.roots,
            "channels": {
                seed_id: [list(segment) for segment in segments]
                for seed_id, segments in self.channels.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # -- queries -------------------------------------------------------------

    def seed_ids(self, pattern: str = "*") -> list[str]:
        """``NET.STA.LOC.CHA`` keys matching an fnmatch-style ``pattern``.

        Parameters
        ----------
        pattern : str
            An ``fnmatch``-style glob, e.g. ``"XX.STA01.*.HH?"``. Defaults
            to ``"*"`` (every seed id).

        Returns
        -------
        list of str
        """
        return [seed_id for seed_id in self.channels if fnmatch.fnmatch(seed_id, pattern)]

    def files(self, seed_id: str, t1: UTCDateTime, t2: UTCDateTime) -> list[Path]:
        """Files for ``seed_id`` whose ``[start, end]`` overlaps ``[t1, t2]``.

        Parameters
        ----------
        seed_id : str
            Exact ``NET.STA.LOC.CHA`` key (not a pattern -- see
            :meth:`seed_ids` to resolve wildcards first).
        t1, t2 : obspy.UTCDateTime
            Query window.

        Returns
        -------
        list of pathlib.Path
            Matching files, in start-time order, each listed once even if
            it holds multiple overlapping segments for this seed id (e.g.
            split around a gap -- ordered by that file's earliest
            overlapping segment). Empty if the seed id is unknown or
            nothing overlaps.
        """
        segments = self.channels.get(seed_id, [])
        earliest_start: dict[Path, UTCDateTime] = {}
        for abs_path, start_iso, end_iso, _sr in segments:
            start = UTCDateTime(start_iso)
            end = UTCDateTime(end_iso)
            if start <= t2 and end >= t1:
                path = Path(abs_path)
                if path not in earliest_start or start < earliest_start[path]:
                    earliest_start[path] = start
        return [path for path, _start in sorted(earliest_start.items(), key=lambda item: item[1])]

    def read(
        self,
        seed_ids: str | list[str],
        t1: UTCDateTime,
        t2: UTCDateTime,
        pad: float = 0.0,
    ) -> Stream:
        """Read the merged waveform data for one or more seed ids over a window.

        Resolves ``seed_ids`` (supporting the same fnmatch wildcards as
        :meth:`seed_ids`) to concrete seed ids, finds the overlapping files
        for each via :meth:`files`, and reads each file with
        ``obspy.read(path, starttime=t1-pad, endtime=t2+pad)`` -- native
        MiniSEED selective decoding at the libmseed level, never a full read
        followed by ``.trim()``.

        Parameters
        ----------
        seed_ids : str or list of str
            One seed id, one fnmatch pattern, or a list of either.
        t1, t2 : obspy.UTCDateTime
            Query window (before padding).
        pad : float, optional
            Seconds to widen the window symmetrically on each side
            (``t1 - pad``, ``t2 + pad``), e.g. for day-boundary overlap
            reads. Default 0.0.

        Returns
        -------
        obspy.Stream
            The merged stream (``stream.merge(method=0)``: gaps become
            masked arrays, never filled/interpolated). Empty if no seed id
            matches or no file overlaps the window.
        """
        if isinstance(seed_ids, str):
            patterns = [seed_ids]
        else:
            patterns = list(seed_ids)

        resolved: list[str] = []
        seen = set()
        for pattern in patterns:
            for seed_id in self.seed_ids(pattern):
                if seed_id not in seen:
                    seen.add(seed_id)
                    resolved.append(seed_id)

        read_start = t1 - pad
        read_end = t2 + pad

        stream = Stream()
        seen_paths: set[Path] = set()
        for seed_id in resolved:
            for path in self.files(seed_id, read_start, read_end):
                if path in seen_paths:
                    # A file can hold several seed ids or be reached twice
                    # through overlapping patterns; read it only once.
                    continue
                seen_paths.add(path)
                stream += read(str(path), starttime=read_start, endtime=read_end)

        if len(stream) > 0:
            stream.merge(method=0)
        return stream


def build_index(cfg: Config) -> DataIndex:
    """Scan ``cfg.data.waveform_dirs`` and build+write the miniSEED index.

    Recursively globs each of ``cfg.data.waveform_dirs`` (independently, so
    duplicate files reachable through more than one root are only scanned
    per root they actually reside under) for ``cfg.data.file_pattern``,
    header-scans every match in parallel via ``openmeq.parallel.pmap``, and
    assembles the resulting segments into a :class:`DataIndex`.

    Scans **all** matching files regardless of ``cfg.data.start``/``.end``
    -- see the module docstring's "Index scope vs. query scope" note. This
    is a deliberate design decision: changing the configured date range
    never requires re-scanning the archive.

    The result is written to ``RunPaths(cfg).index_json`` before being
    returned.

    Parameters
    ----------
    cfg : openmeq.config.Config

    Returns
    -------
    DataIndex
    """
    # Imported here (not at module scope) to avoid a paths<->index import
    # cycle risk and to keep this module's import graph minimal for callers
    # that only need DataIndex.load()/.read() and never build an index.
    from openmeq.paths import RunPaths

    roots = [Path(root).resolve() for root in cfg.data.waveform_dirs]

    files: list[str] = []
    for root in roots:
        files.extend(str(p) for p in sorted(root.rglob(cfg.data.file_pattern)))

    logger.info(
        "Scanning %d candidate files under %d root(s) matching %r",
        len(files),
        len(roots),
        cfg.data.file_pattern,
    )

    # Each worker's result is a list of (seed_id, start_iso, end_iso,
    # sampling_rate) tuples for that one file (see _scan_file); pmap is
    # ordered, so zipping back against `files` recovers which path each
    # result came from without threading the path through pickling twice.
    scan_results = pmap(_scan_file, files, cfg.parallel, desc="index")

    channels: dict[str, list[Segment]] = {}
    for path, segments in zip(files, scan_results, strict=True):
        for seed_id, start_iso, end_iso, sampling_rate in segments:
            channels.setdefault(seed_id, []).append((path, start_iso, end_iso, sampling_rate))
    for segments in channels.values():
        segments.sort(key=lambda segment: UTCDateTime(segment[1]))

    index = DataIndex(
        version=_INDEX_VERSION,
        created=_iso_z(UTCDateTime.now()),
        roots=[str(root) for root in roots],
        channels=channels,
    )

    paths = RunPaths(cfg)
    index.write(paths.index_json)
    return index
