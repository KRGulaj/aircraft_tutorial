# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Tests for the trapezoidal planform."""

from __future__ import annotations

import math

import pytest

from aircraft_tutorial.wing.planform import PlanformError, TrapezoidalPlanform

B747 = TrapezoidalPlanform(span_m=68.40, area_m2=554.0, root_chord_m=14.63,
                           sweep_quarter_chord_deg=37.5)


def test_taper_of_b747_trapezoid_matches_hand_calculation() -> None:
    """λ = 2·554 / (68.40·14.63) − 1 = 0.107234 (hand calculation)."""
    assert B747.taper == pytest.approx(0.107234, abs=1e-6)
    assert B747.tip_chord_m == pytest.approx(0.107234 * 14.63, abs=1e-5)
    assert B747.aspect_ratio == pytest.approx(68.40**2 / 554.0, rel=1e-12)


def test_area_from_chords_returns_input_area() -> None:
    """(b/2)·(c_r + c_t) recovers S."""
    assert B747.semi_span_m * (B747.root_chord_m + B747.tip_chord_m) == pytest.approx(554.0)


def test_mac_equals_chord_integral() -> None:
    """MAC = (2/S)·∫c² dy and y_MAC = (2/S)·∫c·y dy, by the midpoint rule."""
    n = 20_000
    dy = B747.semi_span_m / n
    ys = [(i + 0.5) * dy for i in range(n)]
    mac = 2.0 / B747.area_m2 * sum(B747.chord_at(y) ** 2 for y in ys) * dy
    y_mac = 2.0 / B747.area_m2 * sum(B747.chord_at(y) * y for y in ys) * dy

    assert B747.mac_m == pytest.approx(mac, rel=1e-7)
    assert B747.y_mac_m == pytest.approx(y_mac, rel=1e-7)


def test_sweep_at_quarter_chord_returns_input_and_le_is_larger() -> None:
    """The c/4 line has the input sweep; a tapered wing has more sweep at the leading edge."""
    assert B747.sweep_deg(0.25) == pytest.approx(37.5, abs=1e-12)
    assert B747.sweep_deg(0.0) > 37.5 > B747.sweep_deg(1.0)


def test_x_le_mac_lies_on_the_leading_edge() -> None:
    """x_LE of the MAC is y_MAC·tan Λ_LE."""
    expected = B747.y_mac_m * math.tan(math.radians(B747.sweep_deg(0.0)))

    assert B747.x_le_mac_m == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("area_m2", [1100.0, 400.0], ids=["inverse_taper", "negative_taper"])
def test_area_that_does_not_fit_a_trapezoid_raises(area_m2: float) -> None:
    """S above b·c_r = 1000.7 m² (λ > 1) or below b·c_r / 2 (λ < 0) is rejected."""
    with pytest.raises(PlanformError, match="taper"):
        TrapezoidalPlanform(span_m=68.40, area_m2=area_m2, root_chord_m=14.63,
                            sweep_quarter_chord_deg=37.5)
