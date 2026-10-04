# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Tests for the SI / imperial conversions against exact definitions."""

from __future__ import annotations

import pytest

from aircraft_tutorial.units import kg_to_lb, kmh_to_ms, lb_to_kg, m2_to_ft2


def test_kg_to_lb_mtow_matches_exact_definition() -> None:
    """447 700 kg / 0.45359237 kg/lb = 987 009.548 lb (hand calculation)."""
    result = kg_to_lb(447_700.0)

    assert result == pytest.approx(987_009.548, rel=1e-8)


def test_lb_to_kg_round_trip_preserves_value() -> None:
    """kg -> lb -> kg returns the input value."""
    mass_kg = 21_500.37

    result = lb_to_kg(kg_to_lb(mass_kg))

    assert result == pytest.approx(mass_kg, rel=1e-14)


def test_m2_to_ft2_reference_area_matches_exact_definition() -> None:
    """554 m^2 / 0.09290304 m^2/ft^2 = 5963.206 ft^2 (hand calculation)."""
    result = m2_to_ft2(554.0)

    assert result == pytest.approx(5963.206, rel=1e-6)


def test_kmh_to_ms_takeoff_speed_matches_hand_calculation() -> None:
    """270 km/h / 3.6 = 75.0 m/s."""
    result = kmh_to_ms(270.0)

    assert result == pytest.approx(75.0, rel=1e-14)
