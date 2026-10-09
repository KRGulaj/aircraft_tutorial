# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Tests for the cruise design point and the sweep-normal section condition."""

from __future__ import annotations

import math

import pytest

from aircraft_tutorial.common.atmosphere import isa
from aircraft_tutorial.common.units import G0
from aircraft_tutorial.wing.cruise import critical_pressure_coefficient, cruise_point

B747 = cruise_point(0.855, 10_668.0, 380_000.0, 554.0, 9.855, 37.5)


def test_cruise_point_lift_equals_weight() -> None:
    """q·S·CL_des returns m·g0 exactly, and V = M·a."""
    assert B747.dynamic_pressure_pa * 554.0 * B747.cl_design == pytest.approx(
        380_000.0 * G0, rel=1e-12)
    assert B747.velocity_m_s == pytest.approx(0.855 * isa(10_668.0).speed_of_sound_m_s)


def test_cruise_point_b747_matches_hand_calculation() -> None:
    """ρ = 0.37960, V = 253.54 m/s, q = 12 201 Pa → CL_des = 0.5513; μ(218.81 K) = 1.4334e-5
    Pa·s; ρV/μ = 6.714e6 1/m; Re_MAC = 6.62e7 (hand calculation)."""
    assert B747.dynamic_pressure_pa == pytest.approx(12_201.0, rel=1e-4)
    assert B747.cl_design == pytest.approx(0.5513, abs=1e-4)
    assert B747.dynamic_viscosity_pa_s == pytest.approx(1.4334e-5, rel=1e-4)
    assert B747.reynolds_per_m == pytest.approx(6.714e6, rel=1e-3)
    assert B747.reynolds_mac == pytest.approx(6.62e7, rel=2e-3)


def test_flow_label_states_mach_and_reynolds() -> None:
    """Plot titles get M and Re_MAC."""
    assert B747.flow_label == "M = 0.855, Re_MAC = 6.62e+07, h = 10668 m"


def test_reynolds_of_mac_matches_field() -> None:
    """reynolds(MAC) is the stored Re_MAC; reynolds_normal(MAC) is the stored Re_n."""
    assert B747.reynolds(9.855) == pytest.approx(B747.reynolds_mac, rel=1e-12)
    assert B747.reynolds_normal(9.855) == pytest.approx(B747.reynolds_normal_mac, rel=1e-12)


def test_sweep_normal_condition_follows_simple_sweep_theory() -> None:
    """M_n = M·cos Λ = 0.6783, Re_n = Re·cos²Λ = 4.17e7, cl_n = CL / cos²Λ = 0.876."""
    cos2 = math.cos(math.radians(37.5)) ** 2

    assert B747.mach_normal == pytest.approx(0.855 * math.cos(math.radians(37.5)), rel=1e-12)
    assert B747.reynolds_normal_mac == pytest.approx(B747.reynolds_mac * cos2, rel=1e-12)
    assert B747.reynolds_normal_mac == pytest.approx(4.17e7, rel=2e-3)
    assert B747.cl_design_normal == pytest.approx(0.876, abs=1e-3)


def test_zero_sweep_leaves_condition_unchanged() -> None:
    """Λ = 0: the normal values equal the streamwise ones."""
    point = cruise_point(0.7, 10_000.0, 60_000.0, 120.0, 4.0, 0.0)

    assert (point.mach_normal, point.reynolds_normal_mac, point.cl_design_normal) == (
        point.mach, point.reynolds_mac, point.cl_design)


def test_critical_pressure_coefficient_known_points() -> None:
    """Cp* = 0 at M = 1; Cp*(0.6783) = −0.873 (hand calculation); rises with Mach."""
    assert critical_pressure_coefficient(1.0) == pytest.approx(0.0, abs=1e-15)
    assert B747.cp_crit_normal == pytest.approx(-0.873, abs=1e-3)
    assert critical_pressure_coefficient(0.6) < critical_pressure_coefficient(0.7)


@pytest.mark.parametrize("mach", [0.0, 1.1])
def test_critical_pressure_coefficient_outside_range_raises(mach: float) -> None:
    """Mach outside (0, 1] is rejected."""
    with pytest.raises(ValueError, match="mach"):
        critical_pressure_coefficient(mach)
