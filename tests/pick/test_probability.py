"""Tests for ``openmeq.pick.probability`` (OpenMEQ-9kn)."""

from __future__ import annotations

import numpy as np
import pytest

from openmeq.pick.probability import peak_half_width_s

_SR = 100.0


def _gaussian(n: int, centre: int, sigma_s: float, peak: float = 0.9) -> np.ndarray:
    t = (np.arange(n) - centre) / _SR
    return peak * np.exp(-0.5 * (t / sigma_s) ** 2)


@pytest.mark.parametrize("sigma_s", [0.03, 0.1, 0.3])
def test_half_width_of_gaussian(sigma_s):
    """Half the FWHM of a Gaussian is sigma * sqrt(2 ln 2)."""
    curve = _gaussian(2000, 1000, sigma_s)
    expected = sigma_s * np.sqrt(2 * np.log(2))
    assert peak_half_width_s(curve, 1000, _SR) == pytest.approx(expected, rel=0.02)


def test_frac_and_off_by_one_index():
    curve = _gaussian(2000, 1000, 0.1)
    # A pick index one sample off the maximum still measures the same peak.
    assert peak_half_width_s(curve, 1001, _SR) == pytest.approx(peak_half_width_s(curve, 1000, _SR))
    # A lower level gives a wider half-width: sigma * sqrt(-2 ln frac).
    expected = 0.1 * np.sqrt(-2 * np.log(0.2))
    assert peak_half_width_s(curve, 1000, _SR, frac=0.2) == pytest.approx(expected, rel=0.02)


def test_neighbouring_peak_is_not_merged():
    """A second, separated peak (e.g. a spurious P near S) doesn't widen the first."""
    curve = _gaussian(2000, 1000, 0.05) + _gaussian(2000, 1100, 0.05)
    expected = 0.05 * np.sqrt(2 * np.log(2))
    assert peak_half_width_s(curve, 1000, _SR) == pytest.approx(expected, rel=0.05)


def test_edges_and_invalid():
    curve = _gaussian(100, 0, 0.1)
    # Left side runs into the array edge: width is measured from sample 0.
    expected = 0.1 * np.sqrt(2 * np.log(2)) / 2
    assert peak_half_width_s(curve, 0, _SR) == pytest.approx(expected, rel=0.05)
    assert peak_half_width_s(curve, 100, _SR) is None
    assert peak_half_width_s(np.zeros(10), 5, _SR) is None
