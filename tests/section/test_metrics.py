# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Tests for the section characteristics, on synthetic polars with exact answers.

Synthetic section (α in deg):
    cl = 0.1083·(α + 2.13)                 → a₀ = 0.1083 /deg, α_0L = −2.13°
    cd = 0.0061 + 0.0093·(cl − 0.37)²      → cd_min = 0.0061 at cl = 0.37
    cm = −0.053 + 0.012·cl                 → x_ac = 0.25 − 0.012 = 0.238
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from aircraft_tutorial.config.cases import MetricSettings
from aircraft_tutorial.section.metrics import (
    MetricError,
    aerodynamic_centre,
    compute_metrics,
    fit_spread,
    lift_curve_fit,
    low_drag_range,
    max_lift_to_drag,
    metrics_table,
    minimum_drag,
    quarter_chord_moment,
    stall,
)
from aircraft_tutorial.section.polar import Polar, RunInfo

INFO = RunInfo(airfoil="synthetic", reynolds=8.9e6, mach=0.15, ncrit=10.0, panel_nodes=160)
LO, HI = -4.05, 6.05  # window edges between grid points: grid −4.0 … 6.0 inside, mean α = 1.0


def _polar(alpha: NDArray[np.float64], cl: NDArray[np.float64], cd: NDArray[np.float64],
           cm: NDArray[np.float64], converged: NDArray[np.bool_] | None = None) -> Polar:
    """Build a polar; non-converged points get NaN values."""
    conv = np.ones(alpha.size, dtype=np.bool_) if converged is None else converged
    nan = np.where(conv, 0.0, np.nan)
    return Polar(INFO, alpha, cl + nan, cd + nan, cm + nan, np.full(alpha.size, -0.7) + nan,
                 conv, ~conv, np.full(alpha.size, 3.1e-5))


@pytest.fixture
def section() -> Polar:
    """Synthetic section on α = −6.3 … 13.7 in steps of 0.1 (201 points)."""
    alpha = np.round(-6.3 + 0.1 * np.arange(201), 10)
    cl = 0.1083 * (alpha + 2.13)
    return _polar(alpha, cl, 0.0061 + 0.0093 * (cl - 0.37) ** 2, -0.053 + 0.012 * cl)


def test_lift_curve_fit_recovers_slope_and_zero_lift_angle(section: Polar) -> None:
    """a₀ = 0.1083 /deg, α_0L = −2.13°, 101 points in the window."""
    fit = lift_curve_fit(section, LO, HI)

    assert fit.slope == pytest.approx(0.1083, rel=1e-12)
    assert fit.x_intercept == pytest.approx(-2.13, rel=1e-12)
    assert fit.n_points == 101


def test_minimum_drag_at_parabola_vertex(section: Polar) -> None:
    """cd_min = 0.0061 at cl = 0.37 (the grid holds cl within 0.006 of the vertex)."""
    cd_min, cl_at = minimum_drag(section)

    assert cd_min == pytest.approx(0.0061, abs=1e-6)
    assert cl_at == pytest.approx(0.37, abs=0.006)


def test_low_drag_range_matches_closed_form(section: Polar) -> None:
    """(cl − 0.37)² ≤ 0.13·0.0061/0.0093 (relative to the grid cd_min) → 0.37 ∓ 0.29203."""
    cd_min, _ = minimum_drag(section)
    half = math.sqrt(0.13 * cd_min / 0.0093 + (cd_min - 0.0061) / 0.0093)

    r = low_drag_range(section, 0.13)

    assert r.cl_lower == pytest.approx(0.37 - half, abs=5e-4)
    assert r.cl_upper == pytest.approx(0.37 + half, abs=5e-4)


def test_quarter_chord_moment_is_window_mean(section: Polar) -> None:
    """Mean cm = −0.053 + 0.012·cl(α = 1.0) = −0.053 + 0.012·0.1083·3.13."""
    assert quarter_chord_moment(section, LO, HI) == pytest.approx(
        -0.053 + 0.012 * 0.1083 * 3.13, abs=1e-12)


def test_aerodynamic_centre_from_moment_slope(section: Polar) -> None:
    """x_ac = 0.25 − dcm/dcl = 0.25 − 0.012 = 0.238."""
    assert aerodynamic_centre(section, LO, HI) == pytest.approx(0.238, abs=1e-12)


def test_max_lift_to_drag_matches_closed_form(section: Polar) -> None:
    """d(cl/cd)/dcl = 0 → cl² = (0.0061 + 0.0093·0.37²)/0.0093 = 0.792814, cl = 0.890401,
    cl/cd = 0.890401 / (0.0061 + 0.0093·0.520401²) = 103.311."""
    ld, cl_at, _ = max_lift_to_drag(section)

    assert ld == pytest.approx(103.311, rel=1e-4)
    assert cl_at == pytest.approx(0.890401, abs=0.006)


def test_compute_metrics_converts_slope_to_per_radian(section: Polar) -> None:
    """a₀ per rad = 0.1083·180/π = 6.20513; no stall inside a linear polar."""
    m = compute_metrics(section, MetricSettings(LO, HI, 0.13, ((-2.05, 4.05),)), True)

    assert m.a0_per_rad == pytest.approx(0.1083 * 180.0 / math.pi, rel=1e-12)
    assert m.stall.detected is False
    assert m.low_drag is not None


def test_compute_metrics_without_drag_bucket_has_no_low_drag_range(section: Polar) -> None:
    """A 4-digit-type section reports no low-drag range."""
    m = compute_metrics(section, MetricSettings(LO, HI, 0.13, ((-2.05, 4.05),)), False)

    assert m.low_drag is None


def test_fit_spread_bent_lift_curve_gives_window_range() -> None:
    """cl = 0.11·α for α ≤ 2, slope 0.09 above: window [−3, 2] gives a₀ = 0.11 and α_0L = 0
    exactly; window [−3, 6] gives a smaller slope and a shifted root, so both spreads are open."""
    a = np.round(-3.0 + 0.5 * np.arange(19), 10)
    cl = np.where(a <= 2.0, 0.11 * a, 0.22 + 0.09 * (a - 2.0))
    p = _polar(a, cl, np.full(a.size, 0.007), np.full(a.size, -0.05))

    s = fit_spread(p, ((-3.05, 2.05), (-3.05, 6.05)))

    assert s.a0_max_per_deg == pytest.approx(0.11, rel=1e-12)
    assert s.a0_min_per_deg < 0.11
    assert min(abs(s.alpha_0l_min_deg), abs(s.alpha_0l_max_deg)) == pytest.approx(0.0, abs=1e-12)
    assert s.alpha_0l_max_deg - s.alpha_0l_min_deg > 0.01


def _stall_polar(alphas: list[float], cls: list[float]) -> Polar:
    """Polar with only cl of interest; cd and cm filler."""
    a = np.array(alphas)
    return _polar(a, np.array(cls), np.full(a.size, 0.007), np.full(a.size, -0.05))


def test_stall_clean_peak_detected() -> None:
    """Peak 1.07 at 9.7°, then a sustained drop."""
    p = _stall_polar([0.0, 2.3, 4.6, 7.1, 9.7, 11.9, 13.3, 15.1],
                     [0.23, 0.47, 0.71, 0.93, 1.07, 0.91, 0.77, 0.63])

    s = stall(p)

    assert (s.cl_max, s.alpha_deg, s.detected) == (1.07, 9.7, True)


def test_stall_ignores_spurious_post_stall_plateau() -> None:
    """Real peak 1.80 at 17.25°; XFoil-like converged plateau near 0.75 afterwards."""
    p = _stall_polar([0.0, 5.0, 10.0, 15.0, 17.25, 20.0, 25.0, 30.0, 35.0],
                     [0.31, 0.89, 1.41, 1.75, 1.80, 1.21, 0.85, 0.75, 0.752])

    s = stall(p)

    assert (s.cl_max, s.alpha_deg) == (1.80, 17.25)


def test_stall_single_noisy_dip_is_not_stall() -> None:
    """A one-point dip of 0.03 near the peak, then cl rises again to 1.31."""
    p = _stall_polar([0.0, 3.0, 6.0, 7.0, 8.0, 10.0, 12.0, 13.0, 14.0, 15.0],
                     [0.21, 0.53, 0.86, 0.93, 0.90, 1.13, 1.31, 1.12, 1.01, 0.93])

    s = stall(p)

    assert (s.cl_max, s.alpha_deg) == (1.31, 12.0)


def test_stall_ignores_negative_stall_branch() -> None:
    """Negative stall below −10° (cl rises again toward −0.41) must not count as stall."""
    p = _stall_polar([-13.7, -11.9, -10.3, -6.1, 0.0, 4.3, 8.9, 12.1, 13.9, 15.3, 16.7],
                     [-0.41, -0.62, -0.83, -0.43, 0.23, 0.69, 1.13, 1.37, 1.11, 0.97, 0.88])

    s = stall(p)

    assert (s.cl_max, s.alpha_deg, s.detected) == (1.37, 12.1, True)


def test_stall_not_reached_reports_lower_bound() -> None:
    """Monotonic cl: largest value returned with detected = False."""
    p = _stall_polar([0.0, 1.7, 3.3, 4.9], [0.21, 0.39, 0.57, 0.73])

    s = stall(p)

    assert (s.cl_max, s.detected) == (0.73, False)


def test_low_drag_range_edge_outside_polar_raises() -> None:
    """cd falls to the end of the polar, so the upper edge is never reached."""
    a = np.array([-2.1, -0.7, 0.6, 1.9, 3.3])
    p = _polar(a, 0.11 * a, np.array([0.0091, 0.0077, 0.0068, 0.0063, 0.0061]),
               np.full(5, -0.05))

    with pytest.raises(MetricError, match="edge not reached"):
        low_drag_range(p, 0.1)


def test_window_with_too_few_points_raises(section: Polar) -> None:
    """A window holding two grid points is rejected."""
    with pytest.raises(MetricError, match="fewer than 3"):
        lift_curve_fit(section, 2.05, 2.25)


def test_metrics_skip_non_converged_points() -> None:
    """A failed point inside the window is excluded from the fit (NaN would poison it)."""
    a = np.array([-2.3, -1.1, 0.4, 1.7, 2.9])
    conv = np.array([True, True, False, True, True])
    p = _polar(a, 0.1083 * (a + 2.13), np.full(5, 0.007), np.full(5, -0.05), conv)

    fit = lift_curve_fit(p, -3.0, 3.0)

    assert fit.n_points == 4
    assert fit.slope == pytest.approx(0.1083, rel=1e-12)


def test_metrics_table_low_drag_rows_only_with_drag_bucket(section: Polar) -> None:
    """The table holds low_drag_lower/upper only when a low-drag range exists."""
    settings = MetricSettings(LO, HI, 0.13, ((-2.05, 4.05),))

    with_bucket = [q for q, _, _ in metrics_table(compute_metrics(section, settings, True))]
    without = [q for q, _, _ in metrics_table(compute_metrics(section, settings, False))]

    assert {"low_drag_lower", "low_drag_upper"} <= set(with_bucket)
    assert "low_drag_lower" not in without
    assert len(with_bucket) == len(without) + 2
