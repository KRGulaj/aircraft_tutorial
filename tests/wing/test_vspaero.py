# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Tests for the OpenVSP wing model and a short VSPAERO run. Skipped without OpenVSP."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openvsp")

from aircraft_tutorial.config.wing import VlmSettings  # noqa: E402
from aircraft_tutorial.geometry.airfoil import cosine_stations  # noqa: E402
from aircraft_tutorial.geometry.naca4 import naca4  # noqa: E402
from aircraft_tutorial.wing import vsp_model, vspaero  # noqa: E402
from aircraft_tutorial.wing.planform import TrapezoidalPlanform  # noqa: E402

PLANFORM = TrapezoidalPlanform(span_m=68.40, area_m2=554.0, root_chord_m=14.63,
                               sweep_quarter_chord_deg=37.5)
COARSE = VlmSettings(segments=4, segment_tess=2, chord_tess=13, wake_iter=1, n_cpu=2,
                     alpha_start_deg=-2.0, alpha_end_deg=2.0, alpha_npts=3)
REFERENCE = vspaero.Reference(area_m2=554.0, span_m=68.40, chord_m=PLANFORM.mac_m,
                              x_m=PLANFORM.x_le_mac_m + 0.25 * PLANFORM.mac_m, z_m=0.0)
FLOW = vspaero.FlowInput(mach=0.3, velocity_m_s=100.0, density_kg_m3=1.0, reynolds=5e7)
AIRFOIL = naca4("0012", cosine_stations(41))


def _cl_at_zero(tmp_path: Path, incidence_deg: float, twist_deg: float) -> float:
    vsp_model.new_wing_model("W", PLANFORM, AIRFOIL, dihedral_deg=0.0,
                             root_incidence_deg=incidence_deg, twist_deg=twist_deg, vlm=COARSE)
    polar, _ = vspaero.run_sweep(tmp_path / "w.vsp3", REFERENCE, FLOW, COARSE,
                                 alpha_start_deg=0.0, alpha_end_deg=0.0, alpha_npts=1)
    return float(polar.cl[0])


def test_model_has_input_area_and_span() -> None:
    """OpenVSP reproduces S and b of the planform."""
    wid = vsp_model.new_wing_model("W", PLANFORM, AIRFOIL, dihedral_deg=0.0,
                                   root_incidence_deg=4.0, twist_deg=-3.0, vlm=COARSE)

    area, span = vsp_model.total_area_and_span(wid)

    assert area == pytest.approx(554.0, rel=1e-4)
    assert span == pytest.approx(68.40, rel=1e-9)


def test_model_twist_is_linear_at_segment_boundaries() -> None:
    """At every segment boundary the section angle is i_r + ε·η; between them it stays close."""
    wid = vsp_model.new_wing_model("W", PLANFORM, AIRFOIL, dihedral_deg=0.0,
                                   root_incidence_deg=4.0, twist_deg=-6.0, vlm=COARSE)

    y, angle = vsp_model.section_angles(wid, 201)
    linear = 4.0 - 6.0 * y / PLANFORM.semi_span_m
    boundaries = vsp_model.segment_stations(COARSE.segments) * PLANFORM.semi_span_m

    np.testing.assert_allclose(np.interp(boundaries, y, angle),
                               4.0 - 6.0 * boundaries / PLANFORM.semi_span_m, atol=0.02)
    assert float(np.max(np.abs(angle - linear))) < 0.3


def test_symmetric_section_lift_follows_incidence_and_washout(tmp_path: Path) -> None:
    """Symmetric section: no lift without incidence; incidence raises CL; wash-out lowers it."""
    untwisted_flat = _cl_at_zero(tmp_path, 0.0, 0.0)
    with_incidence = _cl_at_zero(tmp_path, 4.0, 0.0)
    washed_out = _cl_at_zero(tmp_path, 4.0, -4.0)

    assert abs(untwisted_flat) < 1e-4
    assert with_incidence > washed_out > 0.0


def test_sweep_returns_every_angle_and_right_half_strips(tmp_path: Path) -> None:
    """Three angles in ascending order; strips only on y > 0, inside the semi-span."""
    vsp_model.new_wing_model("W", PLANFORM, AIRFOIL, dihedral_deg=0.0, root_incidence_deg=0.0,
                             twist_deg=0.0, vlm=COARSE)

    polar, strips = vspaero.run_sweep(tmp_path / "w.vsp3", REFERENCE, FLOW, COARSE,
                                      alpha_start_deg=-2.0, alpha_end_deg=2.0, alpha_npts=3)

    np.testing.assert_allclose(polar.alpha_deg, [-2.0, 0.0, 2.0])
    assert bool(np.all(np.diff(polar.cl) > 0.0))
    assert bool(np.all((strips.y_m > 0.0) & (strips.y_m < PLANFORM.semi_span_m)))
    assert set(np.unique(strips.alpha_deg).tolist()) == {-2.0, 0.0, 2.0}
