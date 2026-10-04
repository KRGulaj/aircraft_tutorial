# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Tests for the ISA model and Sutherland's law."""

from __future__ import annotations

import math

import pytest

from aircraft_tutorial.common.atmosphere import (
    GAMMA,
    P0,
    R_AIR,
    T0,
    AtmosphereError,
    isa,
    sutherland_viscosity,
)


def test_isa_sea_level_returns_reference_values() -> None:
    """At h = 0: T0, p0, ρ = p0 / (R·T0) = 1.2250 kg/m³, a = 340.29 m/s."""
    air = isa(0.0)

    assert (air.temperature_k, air.pressure_pa) == (T0, P0)
    assert air.density_kg_m3 == pytest.approx(1.2250, abs=5e-5)
    assert air.speed_of_sound_m_s == pytest.approx(340.29, abs=5e-3)


def test_isa_35000_ft_matches_icao_table() -> None:
    """ICAO table at 10 668 m (35 000 ft): T = 218.81 K, p = 23 842 Pa, ρ = 0.37956 kg/m³."""
    air = isa(10_668.0)

    assert air.temperature_k == pytest.approx(218.808, abs=1e-3)
    assert air.pressure_pa == pytest.approx(23_842.0, rel=2e-4)
    assert air.density_kg_m3 == pytest.approx(0.37956, rel=2e-4)


def test_isa_is_continuous_at_the_tropopause() -> None:
    """Both branches give the same state at 11 km."""
    below, above = isa(11_000.0 - 1e-6), isa(11_000.0 + 1e-6)

    assert above.temperature_k == pytest.approx(below.temperature_k, rel=1e-9)
    assert above.pressure_pa == pytest.approx(below.pressure_pa, rel=1e-9)


def test_isa_state_satisfies_gas_law_and_sound_speed() -> None:
    """p = ρ·R·T and a = √(γ·R·T) in the stratosphere branch."""
    air = isa(15_000.0)

    assert air.pressure_pa == pytest.approx(air.density_kg_m3 * R_AIR * air.temperature_k)
    assert air.speed_of_sound_m_s == pytest.approx(math.sqrt(GAMMA * R_AIR * air.temperature_k))


@pytest.mark.parametrize("altitude_m", [-1.0, 20_001.0])
def test_isa_outside_range_raises(altitude_m: float) -> None:
    """Altitudes outside [0, 20 km] are rejected."""
    with pytest.raises(AtmosphereError):
        isa(altitude_m)


def test_sutherland_viscosity_at_reference_temperature_returns_mu0() -> None:
    """At 273.15 K the law returns its reference value 1.716e-5 Pa·s."""
    assert sutherland_viscosity(273.15) == pytest.approx(1.716e-5, rel=1e-12)
