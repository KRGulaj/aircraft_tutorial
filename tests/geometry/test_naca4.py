# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Tests for the analytic NACA four-digit sections against Abbott & von Doenhoff."""

from __future__ import annotations

import numpy as np
import pytest

from aircraft_tutorial.geometry.airfoil import cosine_stations
from aircraft_tutorial.geometry.naca4 import mean_line, naca4, parse_naca4, thickness


def test_thickness_at_thirty_percent_matches_hand_calculation() -> None:
    """Eq. (6.2), t = 0.12, x = 0.3:
    0.6·(0.16261883 − 0.0378 − 0.031644 + 0.0076761 − 0.00082215) = 0.0600173."""
    assert thickness(np.array([0.3]), 0.12)[0] == pytest.approx(0.0600173, abs=1e-7)


def test_thickness_trailing_edge_is_open() -> None:
    """Eq. (6.2) at x = 1: 0.6·(0.2969 − 0.126 − 0.3516 + 0.2843 − 0.1015) = 0.00126."""
    assert thickness(np.array([1.0]), 0.12)[0] == pytest.approx(0.00126, abs=1e-12)


def test_mean_line_2412_matches_hand_values() -> None:
    """Eq. (6.4), m = 0.02, p = 0.4: y_c(0.13) = 0.125·(0.104 − 0.0169);
    y_c(0.4) = 0.02 with zero slope; y_c(0.7) = 0.02/0.36·(0.2 + 0.56 − 0.49) = 0.015."""
    y_c, dyc = mean_line(np.array([0.13, 0.4, 0.7]), 0.02, 0.4)

    np.testing.assert_allclose(y_c, [0.125 * (0.104 - 0.0169), 0.02, 0.015], atol=1e-15)
    assert dyc[1] == pytest.approx(0.0, abs=1e-15)


# Abbott App. III, p.410, NACA 2412, stations in % c on the surface (shifted x).
# The table is rounded to 0.01 % c and differs from eqs. (6.1)-(6.4) by up to 0.018 % c
# (x = 60 %, lower surface), so the tolerance is 0.02 % c.
ABBOTT_2412 = [
    (10.0, 5.63, -3.75), (20.0, 7.26, -4.23), (30.0, 7.88, -4.12), (40.0, 7.80, -3.80),
    (60.0, 6.36, -2.76), (80.0, 3.75, -1.50), (95.0, 1.14, -0.48),
]


@pytest.mark.parametrize(("x_pct", "upper_pct", "lower_pct"), ABBOTT_2412)
def test_naca2412_matches_abbott_ordinates(x_pct: float, upper_pct: float,
                                           lower_pct: float) -> None:
    """Generated surfaces at the tabulated stations equal Abbott's ordinates."""
    xu, yu, xl, yl = naca4("2412", cosine_stations(801)).surfaces()

    assert np.interp(x_pct / 100.0, xu, yu) * 100.0 == pytest.approx(upper_pct, abs=0.02)
    assert np.interp(x_pct / 100.0, xl, yl) * 100.0 == pytest.approx(lower_pct, abs=0.02)


@pytest.mark.parametrize("bad", ["241", "24120", "2a12", "2012", "0412", "2400"])
def test_parse_naca4_invalid_designation_raises(bad: str) -> None:
    """Wrong length, non-digits, camber without position (and reverse), zero thickness."""
    with pytest.raises(ValueError):
        parse_naca4(bad)
