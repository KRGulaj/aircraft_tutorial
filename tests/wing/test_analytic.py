# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Tests for the analytical (DATCOM) lift curve and twist sizing."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aircraft_tutorial.wing.analytic import (
    AnalyticError,
    datcom_lift_slope,
    size_twist,
    twist_effectiveness,
)


def test_datcom_slope_b747_matches_hand_calculation() -> None:
    """A = 8.445, M = 0.855 (β = 0.5186), Λ_0.5c = 33.90°, η = 0.95:
    √(4 + 4.610²·(1 + 0.4516/0.2689)) = 7.806 → CL_α = 2π·8.445 / 9.806 = 5.411 /rad."""
    slope = datcom_lift_slope(68.4**2 / 554.0, 0.855, 33.8986, 0.95)

    assert slope == pytest.approx(5.411, abs=2e-3)


def test_datcom_slope_two_dimensional_limit() -> None:
    """Unswept, incompressible, η = 1, A → ∞: the slope tends to 2π, thin-airfoil theory."""
    assert datcom_lift_slope(1e6, 0.0, 0.0, 1.0) == pytest.approx(2.0 * math.pi, rel=1e-5)


def test_datcom_slope_low_aspect_ratio_limit() -> None:
    """A → 0: CL_α → πA/2, slender-wing theory."""
    assert datcom_lift_slope(1e-4, 0.0, 0.0, 1.0) == pytest.approx(math.pi * 1e-4 / 2.0,
                                                                  rel=1e-6)


@pytest.mark.parametrize(("mach", "eta"), [(1.0, 0.95), (0.5, 0.0)], ids=["sonic", "zero_eta"])
def test_datcom_slope_invalid_input_raises(mach: float, eta: float) -> None:
    """Sonic Mach and non-positive η are rejected."""
    with pytest.raises(AnalyticError):
        datcom_lift_slope(8.0, mach, 30.0, eta)


def test_twist_effectiveness_limits() -> None:
    """Rectangular wing (λ = 1): 1/2, the mean of a linear ramp. Pointed wing (λ = 0): 1/3."""
    assert twist_effectiveness(1.0) == pytest.approx(0.5)
    assert twist_effectiveness(0.0) == pytest.approx(1.0 / 3.0)


def test_size_twist_returns_design_cl_at_zero_body_angle() -> None:
    """The sized model gives CL_des at α = 0 and zero lift at α_0L."""
    wing = size_twist(0.5513, 5.082, taper=0.107234, root_incidence_deg=4.0,
                      section_alpha_0l_deg=-2.2, max_twist_deg=6.0)

    assert float(wing.cl(0.0)) == pytest.approx(0.5513, rel=1e-12)
    assert float(wing.cl(wing.alpha_zero_lift_deg)) == pytest.approx(0.0, abs=1e-15)
    np.testing.assert_allclose(np.diff(wing.cl(np.array([0.0, 1.0]))),
                               [math.radians(1.0) * 5.082])


def test_size_twist_beyond_limit_raises() -> None:
    """A design CL far above what ±6° can give raises with the required twist."""
    with pytest.raises(AnalyticError, match="beyond"):
        size_twist(1.0, 5.0, taper=0.1, root_incidence_deg=4.0, section_alpha_0l_deg=-2.0,
                   max_twist_deg=6.0)
