# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Tests for the closed-form least-squares line fit."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aircraft_tutorial.common.fitting import FitError, fit_line


def test_fit_line_three_points_matches_hand_calculation() -> None:
    """Hand calculation for x = (1.3, 2.3, 4.3), y = (1.3, 2.1, 4.4).

    x̄ = 2.63333, ȳ = 2.6, Sxx = 42/9, Sxy = 4.9, Syy = 5.18.
    b = 4.9 / (42/9) = 1.05, a = 2.6 − 1.05·2.63333 = −0.165.
    Residuals (0.1, −0.15, 0.05), SSE = 0.035, R² = 1 − 0.035/5.18 = 0.993243.
    se(b) = √(0.035 / 1 / (42/9)) = √0.0075 = 0.0866025.
    """
    x = np.array([1.3, 2.3, 4.3])
    y = np.array([1.3, 2.1, 4.4])

    fit = fit_line(x, y)

    assert fit.slope == pytest.approx(1.05, rel=1e-12)
    assert fit.intercept == pytest.approx(-0.165, rel=1e-12)
    assert fit.r_squared == pytest.approx(1.0 - 0.035 / 5.18, rel=1e-12)
    assert fit.se_slope == pytest.approx(math.sqrt(0.0075), rel=1e-12)
    assert fit.max_abs_residual == pytest.approx(0.15, rel=1e-12)
    assert (fit.n_points, fit.x_min, fit.x_max) == (3, 1.3, 4.3)


def test_fit_line_exact_line_recovers_coefficients_and_root() -> None:
    """Seven points on y = −2.9 + 0.37·x; root at x = 2.9 / 0.37 = 7.837838."""
    x = np.array([-4.1, -1.7, 0.3, 2.9, 5.3, 8.6, 11.9])
    y = -2.9 + 0.37 * x

    fit = fit_line(x, y)

    assert fit.slope == pytest.approx(0.37, rel=1e-13)
    assert fit.intercept == pytest.approx(-2.9, rel=1e-13)
    assert fit.x_intercept == pytest.approx(2.9 / 0.37, rel=1e-13)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-13)
    assert fit.se_slope == pytest.approx(0.0, abs=1e-13)


def test_fit_line_swapped_xy_changes_result() -> None:
    """Swapping x and y must not give the same slope (guards an argument mix-up)."""
    x = np.array([1.3, 2.3, 4.3])
    y = np.array([1.3, 2.1, 4.4])

    assert fit_line(x, y).slope != pytest.approx(fit_line(y, x).slope)


@pytest.mark.parametrize(
    ("x", "y"),
    [
        (np.array([0.31, 1.7]), np.array([0.2, 0.9])),
        (np.array([2.9, 2.9, 2.9]), np.array([0.2, 0.9, 1.3])),
        (np.array([0.31, np.nan, 1.7]), np.array([0.2, 0.9, 1.3])),
        (np.array([0.31, 0.9, 1.7]), np.array([0.2, 0.9])),
    ],
    ids=["two_points", "equal_x", "nan_value", "shape_mismatch"],
)
def test_fit_line_degenerate_input_raises(
    x: np.ndarray[tuple[int], np.dtype[np.float64]],
    y: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> None:
    """Undefined fits raise FitError."""
    with pytest.raises(FitError):
        fit_line(x, y)


def test_x_intercept_zero_slope_raises() -> None:
    """A horizontal line has no root."""
    fit = fit_line(np.array([0.3, 1.7, 2.9]), np.array([0.41, 0.41, 0.41]))

    with pytest.raises(FitError):
        _ = fit.x_intercept
