# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Tests for the secant twist sizing, on model CL(ε) functions."""

from __future__ import annotations

import pytest

from aircraft_tutorial.wing.twist import TwistError, size_twist


def test_linear_cl_converges_in_one_secant_step() -> None:
    """CL = 0.586 + 0.009·ε: two starting runs, one secant step, ε = (0.551 − 0.586) / 0.009."""
    result = size_twist(lambda t: 0.586 + 0.009 * t, 0.551, max_twist_deg=6.0,
                        cl_tolerance=1e-9, max_iter=6)

    assert result.twist_deg == pytest.approx(-0.035 / 0.009, rel=1e-12)
    assert result.cl == pytest.approx(0.551, abs=1e-12)
    assert len(result.history) == 3


def test_nonlinear_cl_converges_within_tolerance() -> None:
    """A mildly nonlinear CL(ε) needs more steps but meets the tolerance."""
    result = size_twist(lambda t: 0.5 + 0.01 * t + 0.0005 * t**2, 0.48, max_twist_deg=6.0,
                        cl_tolerance=1e-8, max_iter=10)

    assert 0.5 + 0.01 * result.twist_deg + 0.0005 * result.twist_deg**2 == pytest.approx(
        0.48, abs=1e-8)


def test_zero_twist_already_on_target_returns_after_one_run() -> None:
    """If ε = 0 already gives CL_des, no further run is made."""
    result = size_twist(lambda t: 0.55 + 0.01 * t, 0.55, max_twist_deg=6.0, cl_tolerance=1e-6,
                        max_iter=6)

    assert (result.twist_deg, len(result.history)) == (0.0, 1)


def test_twist_beyond_limit_raises_with_required_value() -> None:
    """CL_des out of reach within ±6 deg raises and reports the twist it would need."""
    with pytest.raises(TwistError, match="-10.000 deg"):
        size_twist(lambda t: 0.6 + 0.01 * t, 0.5, max_twist_deg=6.0, cl_tolerance=1e-6,
                   max_iter=6)


def test_cl_independent_of_twist_raises() -> None:
    """A model where twist has no effect cannot be sized."""
    with pytest.raises(TwistError, match="does not change"):
        size_twist(lambda t: 0.6, 0.5, max_twist_deg=6.0, cl_tolerance=1e-6, max_iter=6)
