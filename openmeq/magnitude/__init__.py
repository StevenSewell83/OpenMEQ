"""Local magnitude (ML) calculation.

Implemented in M8. ``openmeq.magnitude.amplitude`` (M8.1) implements the
per-event Wood-Anderson amplitude measurement:
:func:`~openmeq.magnitude.amplitude.measure_event_amplitudes`.
``openmeq.magnitude.relations`` (M8.2) implements the attenuation-relation
layer: :func:`~openmeq.magnitude.relations.station_ml` and its supporting
loaders. Event-level ML aggregation, catalogue selection and the
``openmeq magnitude`` CLI command are M8.3.
"""
