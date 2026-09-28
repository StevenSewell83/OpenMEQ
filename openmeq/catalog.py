"""QuakeML helpers shared across pipeline steps.

This module is the single place every step-to-step handoff (QuakeML files,
per ``docs/ARCHITECTURE.md`` §5) goes through: event id conventions, the
``openmeq`` ``extra`` namespace round-trip helpers, the never-delete-always-flag
rejection helpers, and centralised catalogue I/O.

Implemented in M0.7.
"""

from __future__ import annotations

import os
from pathlib import Path

from obspy import Catalog, UTCDateTime, read_events
from obspy.core.event import Comment, Event
from obspy.core.event.base import ResourceIdentifier

#: The `extra` namespace used for all openmeq-specific QuakeML extensions
#: (ARCHITECTURE.md §5).
EXTRA_NAMESPACE = "http://openmeq.org/xmlns/1.0"
EXTRA_PREFIX = "openmeq"

#: Prefix used for the QuakeML `Event.resource_id` (ARCHITECTURE.md §4.4).
_EVENT_RESOURCE_PREFIX = "smi:openmeq/event/"


def iso_z(t: UTCDateTime) -> str:
    """Format a time as ISO-8601 UTC with microseconds and a ``Z`` suffix.

    Always ``YYYY-MM-DDTHH:MM:SS.ffffffZ`` (ARCHITECTURE.md §1). This is the
    only writer of that format: ``UTCDateTime.isoformat()`` drops the
    fractional part when it is exactly zero, which gives a mixed-format
    column that ``pandas.to_datetime`` then fails to parse.
    """
    return t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def event_id_from_time(t: UTCDateTime) -> str:
    """Build an event id from a trigger time.

    The id is ``ev`` followed by the detection trigger time as
    ``YYYYMMDDTHHMMSS_mmm`` (UTC, milliseconds), e.g. ``ev20250101T123456_789``
    (ARCHITECTURE.md §4.4). This is the only implementation of that format;
    everything else that needs an event id derived from a time calls this
    function.

    Parameters
    ----------
    t : obspy.UTCDateTime
        The detection trigger time (UTC).

    Returns
    -------
    str
        The event id, e.g. ``"ev20250101T123456_789"``.
    """
    t = UTCDateTime(t)
    millis = t.microsecond // 1000
    return f"ev{t.strftime('%Y%m%dT%H%M%S')}_{millis:03d}"


def event_id(event: Event) -> str:
    """Extract the ``ev...``-style event id from an `Event`'s resource id.

    Parameters
    ----------
    event : obspy.core.event.Event
        An event whose ``resource_id`` is ``smi:openmeq/event/<event_id>``.

    Returns
    -------
    str
        The bare event id (e.g. ``"ev20250101T123456_789"``).

    Raises
    ------
    ValueError
        If the event's resource id does not follow the expected
        ``smi:openmeq/event/<event_id>`` shape.
    """
    rid = str(event.resource_id)
    if not rid.startswith(_EVENT_RESOURCE_PREFIX):
        raise ValueError(
            f"Event resource_id {rid!r} does not start with {_EVENT_RESOURCE_PREFIX!r}"
        )
    return rid[len(_EVENT_RESOURCE_PREFIX) :]


def make_event(event_id: str) -> Event:
    """Create a new `Event` with the standard openmeq resource id.

    Parameters
    ----------
    event_id : str
        The bare event id (e.g. ``"ev20250101T123456_789"``).

    Returns
    -------
    obspy.core.event.Event
        A new, otherwise-empty `Event` whose ``resource_id`` is
        ``smi:openmeq/event/<event_id>``.
    """
    return Event(resource_id=ResourceIdentifier(_EVENT_RESOURCE_PREFIX + event_id))


def accepted(catalog: Catalog) -> Catalog:
    """Return only the events that have not been rejected.

    Per ARCHITECTURE.md §5, "rejection" never deletes anything; it flags the
    preferred origin's ``evaluation_status`` as ``"rejected"``. This function
    filters those out.

    Parameters
    ----------
    catalog : obspy.Catalog
        The full catalogue (including any rejected events).

    Returns
    -------
    obspy.Catalog
        A new `Catalog` (same `Catalog`-level metadata, e.g. ``resource_id``,
        ``description``) containing only the events whose preferred origin's
        ``evaluation_status`` is not ``"rejected"``. An event with no
        preferred origin at all is treated as not rejected.
    """
    kept = [ev for ev in catalog if not _is_rejected(ev)]
    return Catalog(events=kept, resource_id=catalog.resource_id, description=catalog.description)


def _is_rejected(event: Event) -> bool:
    origin = event.preferred_origin()
    if origin is None:
        return False
    return origin.evaluation_status == "rejected"


def reject_pick(pick, arrival, reason: str, step: str) -> None:
    """Flag a pick (and its arrival on the current origin) as rejected.

    Never deletes anything (ARCHITECTURE.md §5): sets
    ``pick.evaluation_status = "rejected"``, sets ``arrival.time_weight = 0``
    on the arrival passed in (the caller passes the `Arrival` belonging to
    whichever `Origin` is "current" for the calling step), and appends a
    `Comment` of the form ``"rejected:<step>:<reason>"`` to the pick.

    Parameters
    ----------
    pick : obspy.core.event.Pick
        The pick to reject.
    arrival : obspy.core.event.Arrival or None
        The arrival, on the current origin, referring to this pick. ``None``
        for a rejection that happens *before* any relocation has produced an
        arrival for this pick yet (M6.3 / OpenMEQ-xdf.3's pass-B S-prediction
        screening: a pick can be screened out ahead of ever being handed to
        NLLoc, so there is no ``Origin``/``Arrival`` for it to carry a
        ``time_weight`` on) -- in that case only ``pick.evaluation_status``
        and the ``Comment`` are set.
    reason : str
        Short machine-parseable reason (no colons), e.g. ``"residual>3sigma"``.
    step : str
        The pipeline step doing the rejecting, e.g. ``"locate_passA"``.
    """
    pick.evaluation_status = "rejected"
    if arrival is not None:
        arrival.time_weight = 0
    pick.comments.append(Comment(text=f"rejected:{step}:{reason}"))


def reject_event(event: Event, reason: str, step: str) -> None:
    """Flag an event's preferred origin (and the event) as rejected.

    Never deletes anything (ARCHITECTURE.md §5): sets the event's preferred
    origin's ``evaluation_status = "rejected"`` and appends a `Comment` of the
    form ``"rejected:<step>:<reason>"`` to the event.

    Parameters
    ----------
    event : obspy.core.event.Event
        The event to reject.
    reason : str
        Short machine-parseable reason (no colons).
    step : str
        The pipeline step doing the rejecting, e.g. ``"review"``.

    Raises
    ------
    ValueError
        If the event has no preferred origin.
    """
    origin = event.preferred_origin()
    if origin is None:
        raise ValueError(f"Event {event.resource_id} has no preferred origin to reject")
    origin.evaluation_status = "rejected"
    event.comments.append(Comment(text=f"rejected:{step}:{reason}"))


def set_extra(obj, key: str, value) -> None:
    """Set a value in an ObsPy object's ``extra`` dict under the openmeq namespace.

    Hides ObsPy's required wrapper shape (``{"value": ..., "namespace": ...,
    "type": ...}``, ARCHITECTURE.md §5) from callers: pass a plain Python
    value (``str``/``float``/``int``/``bool``) and it is stored so it survives
    a QuakeML write/read round-trip via `write_catalog`/`read_catalog`.

    Parameters
    ----------
    obj
        Any ObsPy event-tree object supporting ``.extra`` (`Pick`, `Origin`,
        `Event`, ...).
    key : str
        The extra element/attribute name (e.g. ``"probability"``).
    value : str, float, int, or bool
        The value to store. Serialised to its ``str()`` form in the
        underlying QuakeML; `get_extra` coerces it back (see there for the
        coercion rule).
    """
    if not hasattr(obj, "extra") or obj.extra is None:
        obj.extra = {}
    obj.extra[key] = {
        "value": str(value),
        "namespace": EXTRA_NAMESPACE,
        "type": "attribute",
    }


def get_extra(obj, key: str, default=None):
    """Read a value previously set with `set_extra`.

    ObsPy's QuakeML serialiser round-trips every ``extra`` value as a string
    (confirmed by writing/re-reading a real QuakeML file), regardless of what
    Python type was originally stored. To hide that from callers, this
    function coerces the string back to a plain Python value using the
    following rule, applied in order:

    1. ``"true"``/``"false"`` (case-insensitive) -> `bool`.
    2. Parseable as `int` (no ``.``, ``e``/``E``) -> `int`.
    3. Parseable as `float` -> `float`.
    4. Otherwise returned as the original `str`.

    This is a heuristic (e.g. a deliberately-stored string ``"true"`` also
    comes back as a `bool`); callers needing an unambiguous type should be
    aware of this and coerce explicitly downstream if it matters.

    Parameters
    ----------
    obj
        Any ObsPy event-tree object supporting ``.extra``.
    key : str
        The extra element/attribute name.
    default
        Returned if ``obj`` has no extra dict, or no entry under ``key``.

    Returns
    -------
    str, float, int, or bool
        The coerced value, or ``default``.
    """
    extra = getattr(obj, "extra", None)
    if not extra or key not in extra:
        return default
    raw = extra[key]["value"]
    return _coerce(raw)


def _coerce(raw):
    if isinstance(raw, bool):
        return raw
    if not isinstance(raw, str):
        return raw
    low = raw.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def read_catalog(path) -> Catalog:
    """Read a QuakeML catalogue.

    Thin wrapper around `obspy.read_events`, always passing
    ``format="QUAKEML"`` so every step's I/O is identical.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the QuakeML file.

    Returns
    -------
    obspy.Catalog
    """
    return read_events(str(path), format="QUAKEML")


def write_catalog(cat: Catalog, path) -> None:
    """Write a QuakeML catalogue.

    Thin wrapper around `Catalog.write`, always passing ``format="QUAKEML"``
    and the openmeq `extra` namespace mapping (so it is written with the
    ``openmeq`` prefix, ARCHITECTURE.md §5) and ensuring the parent directory
    exists, so every step's I/O is identical.

    Parameters
    ----------
    cat : obspy.Catalog
        The catalogue to write.
    path : str or pathlib.Path
        Destination path; created if it doesn't exist, along with its parent
        directory.
    """
    path = Path(path)
    if path.parent and not path.parent.exists():
        os.makedirs(path.parent, exist_ok=True)
    cat.write(str(path), format="QUAKEML", nsmap={EXTRA_PREFIX: EXTRA_NAMESPACE})
