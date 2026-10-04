# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Tests for the NACA 66-series sections against Abbott & von Doenhoff tables."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aircraft_tutorial.geometry.airfoil import cosine_stations, max_thickness
from aircraft_tutorial.geometry.naca6 import (
    mean_line_a10,
    naca66,
    parse_naca66,
    thickness,
    thickness_table,
)


def test_thickness_reproduces_every_table_point() -> None:
    """The spline passes through all tabulated 66-010 ordinates (App. I, p.375)."""
    xt, yt, _ = thickness_table(10)

    np.testing.assert_allclose(thickness(xt, 10), yt, atol=1e-14)


def test_thickness_table_holds_scan_corrections() -> None:
    """x = 2.5 % → 1.516 and x = 15 % → 3.530 (scan), not the OCR values 1.616 and 3.630."""
    xt, yt, r_le = thickness_table(10)

    assert yt[np.isclose(xt, 0.025)][0] == pytest.approx(0.01516, abs=1e-12)
    assert yt[np.isclose(xt, 0.15)][0] == pytest.approx(0.03530, abs=1e-12)
    assert r_le == pytest.approx(0.00662, abs=1e-12)


def test_thickness_near_leading_edge_follows_le_radius() -> None:
    """For x → 0, y_t → √(2·r_LE·x); at x = 1e-6 the ratio is 1 within 1 %."""
    x = np.array([1e-6])

    assert thickness(x, 10)[0] / math.sqrt(2.0 * 0.00662 * 1e-6) == pytest.approx(1.0, abs=0.01)


# Abbott App. II, p.405, a = 1.0 mean line for c_li = 1.0: (x %, y_c %, dy_c/dx).
# The tabulated ordinates lie up to 0.0021 % c below the closed form (x = 50 %: closed form
# 5.5159, table 5.515), so the ordinate tolerance is 0.0025 % c for c_li = 1. Slopes are
# tabulated to 0.00005.
ABBOTT_A10 = [(0.5, 0.250, 0.42120), (2.5, 0.930, 0.29155), (10.0, 2.585, 0.17485),
              (20.0, 3.980, 0.11030), (50.0, 5.515, 0.0), (75.0, 4.475, -0.08745),
              (95.0, 1.580, -0.23430)]


@pytest.mark.parametrize(("x_pct", "yc_pct", "slope"), ABBOTT_A10)
def test_mean_line_a10_matches_abbott_scaled_to_cli_04(x_pct: float, yc_pct: float,
                                                       slope: float) -> None:
    """Ordinate and slope scale linearly with c_li."""
    y_c, dyc = mean_line_a10(np.array([x_pct / 100.0]), 0.4)

    assert y_c[0] * 100.0 == pytest.approx(0.4 * yc_pct, abs=0.4 * 0.0025)
    assert dyc[0] == pytest.approx(0.4 * slope, abs=0.4 * 0.00006)


def test_mean_line_a10_midchord_matches_closed_form() -> None:
    """y_c(0.5) = (c_li / 4π)·ln 2 = 0.0220636 for c_li = 0.4."""
    y_c, _ = mean_line_a10(np.array([0.5]), 0.4)

    assert y_c[0] == pytest.approx(0.4 / (4.0 * math.pi) * math.log(2.0), rel=1e-12)


def test_mean_line_a10_ends_are_zero_and_slope_clamped() -> None:
    """y_c = 0 at both ends; slope at x = 0 equals the slope at x = 0.005 (Abbott p.113)."""
    y_c, dyc = mean_line_a10(np.array([0.0, 0.005, 1.0]), 0.4)

    assert y_c[0] == 0.0
    assert y_c[2] == 0.0
    assert dyc[0] == dyc[1]


def test_naca66_410_thickness_is_ten_percent() -> None:
    """66-410 is 10 % thick; vertical thickness of the cambered section within 0.0001."""
    t, _ = max_thickness(naca66("66-410", cosine_stations(401)))

    assert t == pytest.approx(0.100, abs=1e-4)


@pytest.mark.parametrize("bad", ["66-41", "65-410", "66-412", "66410"])
def test_parse_naca66_invalid_or_untabulated_raises(bad: str) -> None:
    """Wrong format, other series, and a thickness without a table are rejected."""
    with pytest.raises(ValueError):
        parse_naca66(bad)
