# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Tests for the thickness/camber split of a contour difference."""

from __future__ import annotations

import pytest

from aircraft_tutorial.geometry.airfoil import Airfoil, cosine_stations
from aircraft_tutorial.geometry.compare import compare
from aircraft_tutorial.geometry.naca4 import naca4


def test_compare_shifted_contour_is_pure_camber_offset() -> None:
    """A vertical shift of −0.0013 changes the camber by +0.0013 everywhere, thickness by 0."""
    ref = naca4("2412", cosine_stations(61))
    shifted = Airfoil("shifted", ref.x, ref.y - 0.0013)

    diff = compare(ref, shifted)

    assert diff.thickness_max_abs == pytest.approx(0.0, abs=1e-15)
    assert diff.camber_min == pytest.approx(0.0013, abs=1e-15)
    assert diff.camber_max == pytest.approx(0.0013, abs=1e-15)


def test_compare_thicker_section_is_pure_thickness_difference() -> None:
    """NACA 2412 against 2415: same mean line, so Δc = 0 and Δt < 0 (reference is thinner)."""
    x = cosine_stations(61)

    diff = compare(naca4("2412", x), naca4("2415", x))

    assert diff.camber_rms == pytest.approx(0.0, abs=2e-4)
    assert diff.thickness_max_abs == pytest.approx(0.03, abs=2e-3)
