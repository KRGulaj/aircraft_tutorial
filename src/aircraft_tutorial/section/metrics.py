# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Section characteristics extracted from a polar. One function per quantity.

Every function uses converged points only. The definitions:

- Lift-curve slope a₀ and zero-lift angle α_0L: least-squares line cl = a + a₀·α over the linear
  window [α_min, α_max]; α_0L = −a / a₀.
- cl_max, α_stall: first sustained drop of cl(α) for α ≥ 0 (see `stall`).
- cd_min: smallest cd; cl at cd_min.
- Low-drag range: the contiguous cl range around cd_min where cd ≤ (1 + k)·cd_min, edges
  interpolated linearly in cl. Only for sections with a laminar drag bucket (6-series); for a
  4-digit section the definition only measures the width of a smooth drag polar.
- Fit spread: a₀ and α_0L over the main window and the extra windows; max − min is the
  uncertainty from the choice of window.
- cm_c/4: mean quarter-chord moment over the linear window.
- Aerodynamic centre: the moment about x is cm_x = cm_c/4 + cl·(x − 0.25) (nose-up positive,
  x aft). At the aerodynamic centre dcm_x/dcl = 0, so x_ac = 0.25 − dcm_c/4/dcl, with the slope
  from a least-squares line cm_c/4(cl) over the linear window.
- (cl/cd)_max: largest cl/cd for cl > 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from aircraft_tutorial.common.fitting import LineFit, fit_line
from aircraft_tutorial.config.cases import MetricSettings
from aircraft_tutorial.section.polar import Polar

STALL_MIN_DROP: Final[float] = 0.02
"""Drop of cl below the running peak that counts as stall onset [-]."""
STALL_SUSTAIN_POINTS: Final[int] = 3
"""Points that must stay below peak − drop, so a single noisy point is not taken as stall."""


class MetricError(ValueError):
    """Raised when a characteristic cannot be extracted from the polar."""


@dataclass(frozen=True)
class Stall:
    """Maximum lift.

    Attributes:
        cl_max: Maximum lift coefficient before stall [-].
        alpha_deg: Angle of attack at cl_max [deg].
        detected: False if cl never dropped by STALL_MIN_DROP inside the swept range; cl_max is
            then only the largest swept value, a lower bound.
    """

    cl_max: float
    alpha_deg: float
    detected: bool


@dataclass(frozen=True)
class LowDragRange:
    """Low-drag range (drag bucket) edges.

    Attributes:
        cl_lower: Lower edge [-].
        cl_upper: Upper edge [-].
        factor: k in cd ≤ (1 + k)·cd_min [-].
    """

    cl_lower: float
    cl_upper: float
    factor: float


@dataclass(frozen=True)
class FitSpread:
    """Range of a₀ and α_0L over several linear windows.

    Attributes:
        a0_min_per_deg: Smallest lift-curve slope [1/deg].
        a0_max_per_deg: Largest lift-curve slope [1/deg].
        alpha_0l_min_deg: Smallest zero-lift angle [deg].
        alpha_0l_max_deg: Largest zero-lift angle [deg].
    """

    a0_min_per_deg: float
    a0_max_per_deg: float
    alpha_0l_min_deg: float
    alpha_0l_max_deg: float


@dataclass(frozen=True)
class SectionMetrics:
    """All characteristics of one polar.

    Attributes:
        lift_fit: cl(α) fit over the linear window (slope per deg, R², residuals).
        a0_per_deg: Lift-curve slope [1/deg].
        a0_per_rad: Lift-curve slope [1/rad].
        alpha_0l_deg: Zero-lift angle of attack [deg].
        spread: a₀ and α_0L over the main and extra windows.
        stall: Maximum lift.
        cd_min: Minimum drag coefficient [-].
        cl_at_cd_min: Lift coefficient at cd_min [-].
        low_drag: Low-drag range; None for a section without a laminar drag bucket.
        cm_c4: Mean quarter-chord moment over the linear window [-].
        x_ac: Aerodynamic centre [chord fraction].
        ld_max: Maximum lift-to-drag ratio [-].
        cl_at_ld_max: Lift coefficient at (cl/cd)_max [-].
        alpha_at_ld_max_deg: Angle of attack at (cl/cd)_max [deg].
    """

    lift_fit: LineFit
    a0_per_deg: float
    a0_per_rad: float
    alpha_0l_deg: float
    spread: FitSpread
    stall: Stall
    cd_min: float
    cl_at_cd_min: float
    low_drag: LowDragRange | None
    cm_c4: float
    x_ac: float
    ld_max: float
    cl_at_ld_max: float
    alpha_at_ld_max_deg: float


def lift_curve_fit(polar: Polar, alpha_min_deg: float, alpha_max_deg: float) -> LineFit:
    """Least-squares line cl(α) over the linear window.

    Args:
        polar: Section polar.
        alpha_min_deg: Lower bound of the window [deg].
        alpha_max_deg: Upper bound of the window [deg].

    Returns:
        The fit; slope in 1/deg, x-intercept is α_0L in deg.

    Raises:
        MetricError: If fewer than three converged points lie in the window.
    """
    c = _window(polar, alpha_min_deg, alpha_max_deg)
    return fit_line(c.alpha_deg, c.cl)


def fit_spread(polar: Polar, windows: tuple[tuple[float, float], ...]) -> FitSpread:
    """a₀ and α_0L range over several linear windows.

    Args:
        polar: Section polar.
        windows: (min, max) windows [deg], at least one.

    Returns:
        The smallest and largest a₀ and α_0L.

    Raises:
        MetricError: If no window is given or a window holds fewer than three points.
    """
    if not windows:
        raise MetricError("fit_spread: at least one window is required")
    fits = [lift_curve_fit(polar, lo, hi) for lo, hi in windows]
    slopes = [f.slope for f in fits]
    roots = [f.x_intercept for f in fits]
    return FitSpread(min(slopes), max(slopes), min(roots), max(roots))


def stall(polar: Polar, min_drop: float = STALL_MIN_DROP,
          sustain_points: int = STALL_SUSTAIN_POINTS) -> Stall:
    """cl_max at the first sustained drop of cl(α), sweeping up from α = 0.

    XFoil can converge well past stall onto a non-physical branch, so a plain max() over all
    converged points is not cl_max. Stall onset is the first point where cl is at least
    `min_drop` below the running peak and the next `sustain_points` points stay that far
    below; cl_max is the running peak at that point.

    Args:
        polar: Section polar.
        min_drop: Drop below the running peak that counts [-].
        sustain_points: Points that must stay below the threshold.

    Returns:
        The maximum lift and whether a stall was detected.

    Raises:
        MetricError: If no converged point exists at α ≥ 0.
    """
    c = polar.converged_only()
    m = c.alpha_deg >= 0.0
    alpha, cl = c.alpha_deg[m], c.cl[m]
    if alpha.size == 0:
        raise MetricError(f"{polar.info.airfoil}: no converged point at alpha >= 0")
    peak, peak_alpha = float(cl[0]), float(alpha[0])
    for i in range(1, alpha.size):
        if cl[i] > peak:
            peak, peak_alpha = float(cl[i]), float(alpha[i])
        elif peak - cl[i] >= min_drop and bool(
                np.all(cl[i:i + sustain_points] <= peak - min_drop)):
            return Stall(cl_max=peak, alpha_deg=peak_alpha, detected=True)
    return Stall(cl_max=peak, alpha_deg=peak_alpha, detected=False)


def minimum_drag(polar: Polar) -> tuple[float, float]:
    """Smallest cd over the converged points.

    Args:
        polar: Section polar.

    Returns:
        (cd_min [-], cl at cd_min [-]).

    Raises:
        MetricError: If no point converged.
    """
    c = _converged(polar)
    i = int(np.argmin(c.cd))
    return float(c.cd[i]), float(c.cl[i])


def low_drag_range(polar: Polar, factor: float) -> LowDragRange:
    """Contiguous cl range around cd_min where cd ≤ (1 + factor)·cd_min.

    Points are walked in α order from the cd_min point outward. Each edge is linearly
    interpolated in cl between the last point inside and the first point outside.

    Args:
        polar: Section polar.
        factor: k [-], > 0.

    Returns:
        The range edges.

    Raises:
        MetricError: If an edge is not reached inside the converged range.
    """
    c = _converged(polar)
    i0 = int(np.argmin(c.cd))
    limit = (1.0 + factor) * float(c.cd[i0])
    edges: list[float] = []
    for step in (-1, 1):
        i = i0
        while 0 <= i + step < c.cd.size and c.cd[i + step] <= limit:
            i += step
        j = i + step
        if not (0 <= j < c.cd.size):
            raise MetricError(f"{polar.info.airfoil}: low-drag range edge not reached "
                              f"(cd <= {limit:.5f} up to the end of the converged polar)")
        frac = (limit - c.cd[i]) / (c.cd[j] - c.cd[i])
        edges.append(float(c.cl[i] + frac * (c.cl[j] - c.cl[i])))
    return LowDragRange(cl_lower=min(edges), cl_upper=max(edges), factor=factor)


def quarter_chord_moment(polar: Polar, alpha_min_deg: float, alpha_max_deg: float) -> float:
    """Mean cm_c/4 over the linear window.

    Args:
        polar: Section polar.
        alpha_min_deg: Lower bound of the window [deg].
        alpha_max_deg: Upper bound of the window [deg].

    Returns:
        Mean cm_c/4 [-].
    """
    return float(np.mean(_window(polar, alpha_min_deg, alpha_max_deg).cm))


def aerodynamic_centre(polar: Polar, alpha_min_deg: float, alpha_max_deg: float) -> float:
    """x_ac = 0.25 − dcm_c/4/dcl over the linear window.

    Args:
        polar: Section polar.
        alpha_min_deg: Lower bound of the window [deg].
        alpha_max_deg: Upper bound of the window [deg].

    Returns:
        Aerodynamic centre [chord fraction].
    """
    c = _window(polar, alpha_min_deg, alpha_max_deg)
    return 0.25 - fit_line(c.cl, c.cm).slope


def max_lift_to_drag(polar: Polar) -> tuple[float, float, float]:
    """Largest cl/cd for cl > 0.

    Args:
        polar: Section polar.

    Returns:
        ((cl/cd)_max [-], cl [-], α [deg]) at the maximum.

    Raises:
        MetricError: If no converged point has cl > 0.
    """
    c = _converged(polar)
    m = c.cl > 0.0
    if not bool(np.any(m)):
        raise MetricError(f"{polar.info.airfoil}: no converged point with cl > 0")
    ratio = np.where(m, c.cl / c.cd, -np.inf)
    i = int(np.argmax(ratio))
    return float(ratio[i]), float(c.cl[i]), float(c.alpha_deg[i])


def compute_metrics(polar: Polar, settings: MetricSettings, drag_bucket: bool) -> SectionMetrics:
    """All characteristics of a polar.

    Args:
        polar: Section polar.
        settings: Linear windows and low-drag factor.
        drag_bucket: True if the section has a laminar drag bucket (low-drag range reported).

    Returns:
        The characteristics.
    """
    lo, hi = settings.linear_alpha_min_deg, settings.linear_alpha_max_deg
    fit = lift_curve_fit(polar, lo, hi)
    cd_min, cl_cd_min = minimum_drag(polar)
    ld, cl_ld, alpha_ld = max_lift_to_drag(polar)
    return SectionMetrics(
        lift_fit=fit,
        a0_per_deg=fit.slope,
        a0_per_rad=fit.slope * 180.0 / math.pi,
        alpha_0l_deg=fit.x_intercept,
        spread=fit_spread(polar, ((lo, hi), *settings.spread_windows_deg)),
        stall=stall(polar),
        cd_min=cd_min,
        cl_at_cd_min=cl_cd_min,
        low_drag=low_drag_range(polar, settings.low_drag_factor) if drag_bucket else None,
        cm_c4=quarter_chord_moment(polar, lo, hi),
        x_ac=aerodynamic_centre(polar, lo, hi),
        ld_max=ld,
        cl_at_ld_max=cl_ld,
        alpha_at_ld_max_deg=alpha_ld,
    )


def _converged(polar: Polar) -> Polar:
    """Converged points; raises if there are none."""
    c = polar.converged_only()
    if c.alpha_deg.size == 0:
        raise MetricError(f"{polar.info.airfoil}: no converged point")
    return c


def _window(polar: Polar, alpha_min_deg: float, alpha_max_deg: float) -> Polar:
    """Converged points with α in [min, max]; at least three required."""
    c = _converged(polar)
    m = (c.alpha_deg >= alpha_min_deg) & (c.alpha_deg <= alpha_max_deg)
    if not (int(np.count_nonzero(m)) >= 3):
        raise MetricError(f"{polar.info.airfoil}: fewer than 3 converged points in "
                          f"[{alpha_min_deg}, {alpha_max_deg}] deg")
    return Polar(c.info, c.alpha_deg[m], c.cl[m], c.cd[m], c.cm[m], c.cp_min[m],
                 c.converged[m], c.diverged[m], c.rms_bl[m])
