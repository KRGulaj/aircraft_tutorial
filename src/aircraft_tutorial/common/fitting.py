# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Closed-form straight-line least-squares fit.

The fit is the textbook normal-equation solution of simple linear regression, written out so
that every reported number traces back to an equation:

    model   y = a + b·x
    Sxx = Σ(xᵢ − x̄)²,  Sxy = Σ(xᵢ − x̄)(yᵢ − ȳ),  Syy = Σ(yᵢ − ȳ)²
    b = Sxy / Sxx,  a = ȳ − b·x̄
    SSE = Σ(yᵢ − a − b·xᵢ)²,  s² = SSE / (n − 2),  se(b) = √(s² / Sxx),  R² = 1 − SSE / Syy

XFoil output is deterministic, so se(b) and R² measure linearity, not experimental scatter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

MIN_POINTS = 3


class FitError(ValueError):
    """Raised when the data cannot give a defined straight-line fit."""


@dataclass(frozen=True)
class LineFit:
    """Result of a straight-line least-squares fit y = intercept + slope·x.

    Attributes:
        slope: b [unit of y per unit of x].
        intercept: a, the value of y at x = 0 [unit of y].
        r_squared: Coefficient of determination [-]. NaN if all y are equal.
        se_slope: Standard error of the slope [unit of y per unit of x].
        max_abs_residual: Largest |yᵢ − a − b·xᵢ| [unit of y].
        n_points: Number of points in the fit.
        x_min: Smallest x in the fit [unit of x].
        x_max: Largest x in the fit [unit of x].
    """

    slope: float
    intercept: float
    r_squared: float
    se_slope: float
    max_abs_residual: float
    n_points: int
    x_min: float
    x_max: float

    @property
    def x_intercept(self) -> float:
        """Value of x where the fitted line crosses y = 0 [unit of x].

        Raises:
            FitError: If the slope is zero.
        """
        if self.slope == 0.0:
            raise FitError("slope is zero; the line never crosses y = 0")
        return -self.intercept / self.slope


def fit_line(x: NDArray[np.float64], y: NDArray[np.float64]) -> LineFit:
    """Fit y = a + b·x by ordinary least squares.

    Args:
        x: Independent values, 1-D.
        y: Dependent values, 1-D, same length as x.

    Returns:
        The fitted line with its goodness-of-fit measures.

    Raises:
        FitError: If the shapes differ, a value is not finite, there are fewer than three
            points, or all x are equal.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape:
        raise FitError(f"x and y must be 1-D with equal shape, got {x.shape} and {y.shape}")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise FitError("x and y must be finite")
    n = x.size
    if not (n >= MIN_POINTS):
        raise FitError(f"need at least {MIN_POINTS} points, got {n}")

    dx = x - x.mean()
    dy = y - y.mean()
    sxx = float(dx @ dx)
    if sxx == 0.0:
        raise FitError(f"all x values are equal ({x[0]}); slope is undefined")
    sxy = float(dx @ dy)
    syy = float(dy @ dy)

    slope = sxy / sxx
    intercept = float(y.mean()) - slope * float(x.mean())
    residuals = y - (intercept + slope * x)
    sse = float(residuals @ residuals)

    return LineFit(
        slope=slope,
        intercept=intercept,
        r_squared=1.0 - sse / syy if syy > 0.0 else math.nan,
        se_slope=math.sqrt(sse / (n - 2) / sxx),
        max_abs_residual=float(np.max(np.abs(residuals))),
        n_points=n,
        x_min=float(x.min()),
        x_max=float(x.max()),
    )
