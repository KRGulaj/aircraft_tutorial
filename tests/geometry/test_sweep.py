# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-09
"""Tests for the sweep-normal section and the streamwise angle of attack."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aircraft_tutorial.geometry.airfoil import (
    Airfoil,
    AirfoilError,
    cosine_stations,
    max_thickness,
)
from aircraft_tutorial.geometry.naca4 import naca4
from aircraft_tutorial.geometry.sweep import streamwise_angle_deg, sweep_normal_section

COS_37_5: float = 0.7933533403
"""cos 37.5°, from tables (10 significant digits)."""


@pytest.fixture
def section() -> Airfoil:
    """NACA 2412 on 41 cosine stations per surface, leading edge at (0, 0), chord 1."""
    return naca4("2412", cosine_stations(41))


def test_sweep_normal_section_37_5_deg_divides_y_by_cos_keeps_x(section: Airfoil) -> None:
    """Same thickness on the chord c·cos Λ: z/c_n = (z/c) / cos 37.5°, x/c unchanged."""
    normal = sweep_normal_section(section, 37.5)

    np.testing.assert_allclose(normal.x, section.x, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(normal.y, section.y / COS_37_5, rtol=1e-9, atol=0.0)
    assert normal.name == "NACA 2412 normal to 37.5 deg sweep"


def test_sweep_normal_section_37_5_deg_thickness_ratio_matches_hand_value(
        section: Airfoil) -> None:
    """t/c_n = 0.12 / cos 37.5° = 0.15126 (Anderson Fig. 11.14); the 4-digit vertical t/c is
    0.12 to within 0.5 % for 2 % camber, and x of the maximum stays at 0.30 c."""
    t_n, x_n = max_thickness(sweep_normal_section(section, 37.5))

    assert t_n == pytest.approx(0.12 / COS_37_5, rel=5e-3)
    assert x_n == pytest.approx(0.30, abs=0.01)


def test_sweep_normal_section_unnormalised_input_is_normalised_first(section: Airfoil) -> None:
    """A contour shifted by (0.31, −0.17) and scaled by 2.9 gives the same normal section."""
    moved = Airfoil(name=section.name, x=0.31 + 2.9 * section.x, y=-0.17 + 2.9 * section.y)

    normal = sweep_normal_section(moved, 37.5)

    np.testing.assert_allclose(normal.x, section.x, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(normal.y, section.y / COS_37_5, rtol=0.0, atol=1e-11)


def test_sweep_normal_section_zero_sweep_keeps_coordinates(section: Airfoil) -> None:
    """Degenerate case: Λ = 0 returns the streamwise section."""
    normal = sweep_normal_section(section, 0.0)

    np.testing.assert_array_equal(normal.x, section.x)
    np.testing.assert_array_equal(normal.y, section.y)


@pytest.mark.parametrize("sweep_deg", [-3.5, 90.0, math.nan], ids=["negative", "ninety", "nan"])
def test_sweep_normal_section_invalid_sweep_raises(section: Airfoil, sweep_deg: float) -> None:
    """A sweep outside [0, 90) deg is rejected."""
    with pytest.raises(AirfoilError, match="sweep"):
        sweep_normal_section(section, sweep_deg)


def test_streamwise_angle_37_5_deg_matches_velocity_components() -> None:
    """Independent check by vectors: V = (cos α_s, 0, sin α_s), x streamwise, y spanwise.
    The in-plane normal to a sweep line swept back by Λ is n = (cos Λ, −sin Λ, 0), so
    α_n = atan2(V_z, V·n). Converting α_n back must return α_s = −2.9°."""
    alpha_s = math.radians(-2.9)
    sweep = math.radians(37.5)
    v_normal = math.cos(alpha_s) * math.cos(sweep)
    alpha_n_deg = math.degrees(math.atan2(math.sin(alpha_s), v_normal))

    result = streamwise_angle_deg(alpha_n_deg, 37.5)

    assert alpha_n_deg == pytest.approx(-3.6535, abs=1e-3)
    assert result == pytest.approx(-2.9, abs=1e-12)


def test_streamwise_angle_zero_sweep_is_identity() -> None:
    """Degenerate case: Λ = 0 leaves the angle unchanged."""
    assert streamwise_angle_deg(-3.26, 0.0) == pytest.approx(-3.26, abs=1e-12)


@pytest.mark.parametrize(("alpha_deg", "sweep_deg"), [(90.0, 37.5), (math.nan, 37.5),
                                                      (-2.1, 90.0)],
                         ids=["alpha_ninety", "alpha_nan", "sweep_ninety"])
def test_streamwise_angle_out_of_range_raises(alpha_deg: float, sweep_deg: float) -> None:
    """An angle of attack outside (−90, 90) deg or a sweep outside [0, 90) deg is rejected."""
    with pytest.raises(AirfoilError):
        streamwise_angle_deg(alpha_deg, sweep_deg)
