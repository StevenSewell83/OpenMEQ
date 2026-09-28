"""Step manifests for skip-unchanged reruns (``StepManifest``, ``fingerprint``).

Every step writes ``<step>/manifest.json`` (ARCHITECTURE.md §3), a simple,
stable file format::

    {"step": "qc", "entries": {"<key>": "<fingerprint>"}}

``key`` is whatever unit of work the calling step considers its own (a
``seed_id`` for the index step, an ``event_id`` for chunking, ...) --
:class:`StepManifest` itself is generic and never interprets ``key``; it
just persists a ``{key: fingerprint}`` mapping.

Concurrency assumption
-----------------------
PLAN.md §4's worker design has parallel workers return results to the main
process, which then decides what to (re)do and calls ``mark_done`` -- workers
never touch shared mutable state themselves. Accordingly, :class:`StepManifest`
here is a simple read-modify-write over one JSON file, used from the single
main process; no inter-process locking is implemented, since PLAN.md's
design never has two processes writing the same manifest concurrently. If a
concrete need for that shows up in a later milestone, add locking there --
don't build it speculatively here.

``mark_done`` persists to disk immediately (write to a temp file, then
``Path.replace`` -- an atomic rename on POSIX), rather than requiring an
explicit ``.flush()``, so a killed process never corrupts the manifest and a
resumed run always sees every previously completed key.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from openmeq.paths import RunPaths


def _canonical(obj: Any) -> str:
    """Render ``obj`` as a canonical string for hashing.

    * A pydantic model (e.g. a config section) uses ``model_dump_json()``,
      which is stable across calls for equal field values.
    * A ``Path`` is hashed as ``(path, size, mtime)`` -- i.e. its content
      identity on disk, not its string form -- so touching a dependent input
      file changes the fingerprint even if the path itself doesn't. A
      missing file still fingerprints (as ``(path, None, None)``) rather
      than raising, so fingerprinting a not-yet-created *output* path (to
      detect it needs (re)creating) works too.
    * Everything else falls back to a JSON encoding (``dict``/``list``/
      ``tuple`` recurse; scalars are encoded directly) with dict keys
      sorted, or to ``repr()`` for anything JSON can't encode.
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump_json()
    if isinstance(obj, Path):
        try:
            stat = obj.stat()
            size, mtime = stat.st_size, stat.st_mtime
        except OSError:
            size, mtime = None, None
        return json.dumps([str(obj), size, mtime])
    if isinstance(obj, dict):
        return json.dumps({str(k): _canonical(v) for k, v in obj.items()}, sort_keys=True)
    if isinstance(obj, list | tuple):
        return "[" + ",".join(_canonical(item) for item in obj) + "]"
    try:
        return json.dumps(obj, sort_keys=True)
    except TypeError:
        return repr(obj)


def fingerprint(*objs: Any) -> str:
    """A stable hash of ``objs``, for "skip unchanged" comparisons.

    Deterministic across calls for equal inputs (same process or a fresh
    one), and sensitive to any change in any ``obj`` -- e.g. a config
    section's field values, or an input file's size/mtime.

    Parameters
    ----------
    *objs
        Any mix of pydantic models (e.g. a ``cfg.<section>``), ``Path``s
        (hashed via their on-disk size/mtime), and JSON-encodable values.

    Returns
    -------
    str
        A hex-encoded SHA-256 digest.
    """
    digest = hashlib.sha256()
    for obj in objs:
        digest.update(_canonical(obj).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


class StepManifest:
    """Reads/writes ``paths.manifest_json(step)`` for "skip unchanged" reruns.

    Parameters
    ----------
    paths : openmeq.paths.RunPaths
    step : str
        Forwarded to ``paths.manifest_json(step)``; also stored as the
        ``"step"`` field in the JSON file (informational only).
    """

    def __init__(self, paths: RunPaths, step: str) -> None:
        self._path = Path(paths.manifest_json(step))
        self._step = step
        self._entries: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError):
            # A partial/corrupt manifest (e.g. from a killed process) is
            # treated as empty -- every key is simply redone this run.
            return
        self._entries = dict(data.get("entries", {}))

    def is_done(self, key: str, fingerprint: str) -> bool:
        """Whether ``key`` was last marked done with this exact ``fingerprint``."""
        return self._entries.get(key) == fingerprint

    def mark_done(self, key: str, fingerprint: str) -> None:
        """Record ``key`` as done with ``fingerprint``, and persist immediately."""
        self._entries[key] = fingerprint
        self._flush()

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"step": self._step, "entries": self._entries}
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp_path.replace(self._path)
