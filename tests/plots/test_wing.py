# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-10
"""Tests for the report figures of the 3D wing analysis: labels and written files."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from aircraft_tutorial.plots.wing import (
    DragPolar,
    LiftCurve,
    SpanLoading,
    flight_conditions,
    plot_drag_build_up,
    plot_drag_polars,
    plot_lift_curves,
    plot_span_loading,
    sci_tex,
    section_conditions,
    section_label,
)

ALPHA = np.array([-2.0, 0.0, 2.0])
CL = np.array([0.36, 0.55, 0.74])


@pytest.mark.parametrize(("value", "expected"),
                         [(6.617e7, r"6.62\times10^{7}"), (4.17e7, r"4.17\times10^{7}"),
                          (2.5e-3, r"2.50\times10^{-3}")],
                         ids=["re_mac", "re_section", "negative_exponent"])
def test_sci_tex_formats_mantissa_and_exponent(value: float, expected: str) -> None:
    """Three significant digits and a power of ten, for mathtext."""
    assert sci_tex(value) == expected


def test_condition_lines_state_mach_and_reynolds() -> None:
    """Brief note 11: every aerodynamic plot states its Re and M."""
    wing = flight_conditions(0.855, 6.617e7, 10_668.0)
    section = section_conditions(4.17e7, 0.1)

    assert "= 0.855," in wing and r"6.62\times10^{7}$," in wing and "10 668 m" in wing
    assert r"4.17\times10^{7}" in section and "M = 0.10" in section


@pytest.mark.parametrize(("stem", "expected"),
                         [("NACA2412", "NACA 2412"), ("NACA66-410_gen", "NACA 66-410"),
                          ("lockheed_c5a", "lockheed_c5a")],
                         ids=["four_digit", "six_series_generated", "other_name"])
def test_section_label_spaces_naca_and_drops_gen(stem: str, expected: str) -> None:
    """File stems become readable section names."""
    assert section_label(stem) == expected


def test_plots_write_one_png_each(tmp_path: Path) -> None:
    """Each figure is written as one PNG file and nothing else."""
    curve = LiftCurve("WING-1", ALPHA, CL, 0.0, 0.55)
    polar = DragPolar("WING-1", CL, np.array([0.006, 0.012, 0.020]), "span loading, Munk",
                      np.array([0.006, 0.006, 0.007]), np.array([0.003, 0.003, 0.0035]),
                      np.array([0.008, 0.011, 0.016]), 1.0, 0.499, 0.55, 0.029, 0.026)

    reference = LiftCurve("WING-1, eps = 0", ALPHA, CL + 0.1, 0.0, 0.55, has_trim=False)
    limited = LiftCurve("WING-1", ALPHA, CL, 0.0, 0.55, limit=(2.0, 0.74))
    eta = np.array([0.1, 0.5, 0.9])
    loading = SpanLoading("WING-1", eta, np.array([0.6, 0.9, 1.2]), 0.0,
                          np.array([1.2, 1.8, 2.2]), 4.0, 2.13)
    no_outside = SpanLoading("WING-2", eta, np.array([0.7, 0.9, 1.0]), 0.0, None, math.nan, 2.11)

    plot_lift_curves([curve, curve], 0.55, "Lift", ["line 1"], tmp_path / "lift.png")
    plot_lift_curves([limited, reference], 0.55, "Lift", ["line 1"], tmp_path / "limit.png")
    plot_drag_build_up(polar, 0.55, "Build-up", ["line 1"], tmp_path / "build.png")
    plot_drag_polars([polar], 0.55, "Polars", ["line 1", "line 2"], tmp_path / "polars.png")
    plot_span_loading([loading, no_outside], "Span", ["line 1"], tmp_path / "span.png")

    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == ["build.png", "lift.png", "limit.png", "polars.png", "span.png"]
    assert all(p.stat().st_size > 0 for p in tmp_path.iterdir())
