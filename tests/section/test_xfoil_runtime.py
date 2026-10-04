# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Tests for the XFoil runtime. These run the real XFoil binding."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from aircraft_tutorial.config.cases import FlowConditions, XfoilSettings
from aircraft_tutorial.geometry.airfoil import cosine_stations
from aircraft_tutorial.geometry.naca4 import naca4
from aircraft_tutorial.section.xfoil_runtime import (
    XfoilSetupError,
    alpha_legs,
    ensure_xfoil_ready,
    run_polar,
)

CONDITIONS = FlowConditions(reynolds=3.1e6, mach=0.15, ncrit=10.0)
SETTINGS = XfoilSettings(panel_nodes=160, max_iter=200, alpha_low_deg=-3.7,
                         alpha_high_deg=3.7, alpha_step_deg=3.7, stop_after_failures=5)


def test_ensure_xfoil_ready_missing_mingw_dir_raises(tmp_path: Path) -> None:
    """A missing MinGW runtime directory fails loudly with the path in the message."""
    missing = tmp_path / "no_mingw"

    with pytest.raises(XfoilSetupError, match="no_mingw"):
        ensure_xfoil_ready(missing)


def test_alpha_legs_are_exact_multiples_of_step() -> None:
    """Range [−2.5, 1.3] with step 0.4: up 0 … 1.2 (4 points), down 0 … −2.4 (7 points)."""
    s = dataclasses.replace(SETTINGS, alpha_low_deg=-2.5, alpha_high_deg=1.3,
                            alpha_step_deg=0.4)

    up, down = alpha_legs(s)

    np.testing.assert_array_equal(up, 0.4 * np.arange(4))
    np.testing.assert_array_equal(down, -0.4 * np.arange(7))


def test_run_polar_symmetric_section_gives_antisymmetric_lift() -> None:
    """NACA 0012: cl(0) = 0 and cl(+3.7°) = −cl(−3.7°); cd and cm symmetric/antisymmetric."""
    polar = run_polar(naca4("0012", cosine_stations(81)), CONDITIONS, SETTINGS)

    assert polar.alpha_deg.tolist() == [-3.7, 0.0, 3.7]
    assert bool(np.all(polar.converged))
    assert polar.cl[1] == pytest.approx(0.0, abs=1e-4)
    assert polar.cl[2] == pytest.approx(-polar.cl[0], rel=1e-3)
    assert polar.cd[2] == pytest.approx(polar.cd[0], rel=1e-3)
    assert polar.cl[2] > 0.3


def test_run_polar_repeated_run_is_bitwise_identical() -> None:
    """Two runs of the same case give the same bytes (fresh session per leg)."""
    section = naca4("2412", cosine_stations(81))

    a = run_polar(section, CONDITIONS, SETTINGS)
    b = run_polar(section, CONDITIONS, SETTINGS)

    assert a.cl.tobytes() == b.cl.tobytes()
    assert a.cd.tobytes() == b.cd.tobytes()


def test_run_polar_stops_each_leg_after_failure_budget() -> None:
    """With one Newton iteration nothing converges: each leg holds exactly 3 points, and the
    merged polar 3 + 3 − 1 = 5 (the shared 0° point appears once)."""
    s = dataclasses.replace(SETTINGS, max_iter=1, alpha_low_deg=-10.0, alpha_high_deg=10.0,
                            alpha_step_deg=0.5, stop_after_failures=3)

    polar = run_polar(naca4("2412", cosine_stations(81)), CONDITIONS, s)

    assert polar.alpha_deg.tolist() == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert not bool(np.any(polar.converged))
    assert bool(np.all(np.isnan(polar.cl)))


def test_run_polar_records_run_info() -> None:
    """The polar carries the airfoil name and the exact condition and node count."""
    s = dataclasses.replace(SETTINGS, panel_nodes=137)

    polar = run_polar(naca4("2412", cosine_stations(81)), CONDITIONS, s)

    assert (polar.info.airfoil, polar.info.reynolds, polar.info.panel_nodes) == (
        "NACA 2412", 3.1e6, 137)
