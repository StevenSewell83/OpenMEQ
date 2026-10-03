"""Instrument group and scalar sensitivity per channel, for the noise plots (OpenMEQ-h8j).

Noise levels are only comparable between channels with the same sensor (and,
in counts, the same digitiser gain), so :mod:`openmeq.qc.noise_plots` draws
one set of figures per instrument group, each on its own colour/axis range.

The group comes from the StationXML sensor of the channel's epoch in force
during the run (``Sensor.model``, else ``description``, else ``type``),
choosing the latest-starting epoch when several overlap (epochs with no end
date are common). ``cfg.qc.instrument_groups`` overrides it, for metadata
known to be wrong (e.g. RK-NM's RT05 is a GS-11D although its StationXML
says LE-3Dlite).

The sensitivity is the epoch's scalar ``InstrumentSensitivity`` (absolute
value: a sign is a polarity convention). ``sens_flag`` records how it was
found: ``ok`` (this channel), ``assumed_component`` (another channel of the
same station and component, e.g. HQZ metadata for EHZ data), or ``missing``
(none: the channel stays in counts).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd
from obspy import Inventory, UTCDateTime

INSTRUMENT_COLUMNS = [
    "seed_id",
    "network",
    "station",
    "location",
    "channel",
    "group",
    "group_source",
    "sensitivity",
    "sensitivity_units",
    "sens_flag",
]

UNKNOWN_GROUP = "unknown"


def _sensor_name(channel) -> str | None:
    sensor = channel.sensor
    if sensor is None:
        return None
    for value in (sensor.model, sensor.description, sensor.type):
        if value and str(value).strip():
            return str(value).strip()
    return None


def _epoch_at(channels: list, when: UTCDateTime):
    """The channel epoch in force at ``when`` (latest start wins), else the latest one."""
    if not channels:
        return None

    def start(ch):
        return ch.start_date or UTCDateTime(0)

    active = [
        ch for ch in channels if start(ch) <= when and (ch.end_date is None or ch.end_date >= when)
    ]
    return max(active or channels, key=start)


def _sensitivity(channel) -> tuple[float | None, str | None]:
    resp = channel.response
    if resp is None or resp.instrument_sensitivity is None:
        return None, None
    value = resp.instrument_sensitivity.value
    if not value:
        return None, None
    return abs(float(value)), resp.instrument_sensitivity.input_units


def _override_groups(overrides: Mapping[str, Iterable[str]]) -> dict[str, str]:
    """``{group: [STA | NET.STA | seed_id]}`` -> ``{key: group}``."""
    out = {}
    for group, members in overrides.items():
        for member in members:
            out[str(member)] = str(group)
    return out


def instrument_table(
    inv: Inventory,
    seed_ids: Iterable[str],
    when: UTCDateTime,
    overrides: Mapping[str, Iterable[str]] | None = None,
) -> pd.DataFrame:
    """One row per seed id: instrument group and scalar sensitivity.

    Parameters
    ----------
    inv : obspy.Inventory
        Merged station metadata (``openmeq.stations.load_inventory``).
    seed_ids : iterable of str
        ``NET.STA.LOC.CHA`` ids with data.
    when : obspy.UTCDateTime
        Time whose metadata epoch applies (e.g. the middle of the run).
    overrides : mapping, optional
        ``cfg.qc.instrument_groups``.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`INSTRUMENT_COLUMNS`. ``sensitivity`` is NaN when
        ``sens_flag == "missing"``.
    """
    override_of = _override_groups(overrides or {})
    rows = []
    for seed_id in seed_ids:
        net, sta, loc, cha = seed_id.split(".")
        exact = inv.select(network=net, station=sta, location=loc, channel=cha)
        chans = [ch for n in exact for s in n for ch in s]
        flag = "ok"
        if not chans:
            same_comp = inv.select(network=net, station=sta, channel=f"??{cha[-1]}")
            chans = [ch for n in same_comp for s in n for ch in s]
            flag = "assumed_component"
        epoch = _epoch_at(chans, when)

        sens, units = _sensitivity(epoch) if epoch is not None else (None, None)
        if sens is None:
            flag = "missing"
        group = _sensor_name(epoch) if epoch is not None else None
        source = "stationxml" if group else "unknown"
        for key in (seed_id, f"{net}.{sta}", sta):
            if key in override_of:
                group, source = override_of[key], "config"
                break
        rows.append(
            {
                "seed_id": seed_id,
                "network": net,
                "station": sta,
                "location": loc,
                "channel": cha,
                "group": group or UNKNOWN_GROUP,
                "group_source": source,
                "sensitivity": sens if sens is not None else float("nan"),
                "sensitivity_units": units or "",
                "sens_flag": flag,
            }
        )
    return pd.DataFrame(rows, columns=INSTRUMENT_COLUMNS)
