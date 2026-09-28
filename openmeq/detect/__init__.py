"""Network coincidence detection.

Implemented in M2. ``openmeq.detect.worker`` (M2.1, ``OpenMEQ-ujx.1``) has
the per-window/per-day detection core (:func:`detect_window`,
:func:`build_windows`); merging across windows, de-duplication, event ids
and the ``openmeq detect`` CLI command follow in M2.2/M2.3.
"""
