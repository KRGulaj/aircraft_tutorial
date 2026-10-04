# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Tests for the strip profile drag."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aircraft_tutorial.contracts.section_polar import SectionPolarData
from aircraft_tutorial.wing.profile_drag import (
    BELOW_POLAR,
    IN_RANGE,
    STALLED,
    DragLookup,
    ProfileDragError,
    reynolds_factor,
    section_cd,
    strip_profile_drag,
    sweep_drag_factor,
    turbulent_skin_friction,
    usable_polar,
)
from aircraft_tutorial.wing.vlm_results import StripLoads

NAN = float("nan")


def _polar(alpha: list[float], cl: list[float], cd: list[float],
           converged: list[bool] | None = None) -> SectionPolarData:
    n = len(alpha)
    return SectionPolarData(
        airfoil="test", reynolds=4e7, mach=0.68, ncrit=9.0,
        alpha_deg=np.array(alpha), cl=np.array(cl), cd=np.array(cd), cm=np.zeros(n),
        cp_min=np.zeros(n),
        converged=np.array(converged if converged is not None else [True] * n))


LOOKUP = DragLookup("test", np.array([0.0, 1.0, 2.0]), np.array([0.2, 0.4, 0.6]),
                    np.array([0.006, 0.007, 0.009]))


def test_usable_polar_stops_at_first_cl_drop_both_ways() -> None:
    """Negative-stall and post-stall branches, and non-converged points, are excluded."""
    polar = _polar(
        alpha=[-6.0, -5.0, -4.0, 0.0, 4.0, 8.0, 9.0, 10.0, 11.0],
        cl=[-0.3, -0.4, -0.2, 0.25, 0.7, 1.1, 1.05, 1.2, NAN],
        cd=[0.02, 0.012, 0.008, 0.006, 0.007, 0.012, 0.03, 0.05, NAN],
        converged=[True, True, True, True, True, True, True, True, False])

    lookup = usable_polar(polar)

    np.testing.assert_array_equal(lookup.alpha_deg, [-5.0, -4.0, 0.0, 4.0, 8.0])
    assert bool(np.all(np.diff(lookup.cl) > 0.0))


def test_usable_polar_without_rising_range_raises() -> None:
    """A polar whose cl falls everywhere around 0 deg has no usable range."""
    with pytest.raises(ProfileDragError, match="rising"):
        usable_polar(_polar([-1.0, 0.0, 1.0], [0.3, 0.2, 0.1], [0.01, 0.01, 0.01]))


def test_section_cd_interpolates_and_flags_out_of_range() -> None:
    """Inside: linear interpolation. Above the top: STALLED, below: BELOW_POLAR, both NaN."""
    cd, status = section_cd(np.array([0.3, 0.5, 0.7, 0.1]), LOOKUP)

    np.testing.assert_allclose(cd[:2], [0.0065, 0.008])
    assert np.isnan(cd[2]) and np.isnan(cd[3])
    np.testing.assert_array_equal(status, [IN_RANGE, IN_RANGE, STALLED, BELOW_POLAR])


def test_sweep_drag_factor_modes() -> None:
    """friction = 1, cos3 = cos³Λ, unknown mode raises."""
    assert sweep_drag_factor("friction", 37.5) == 1.0
    assert sweep_drag_factor("cos3", 37.5) == pytest.approx(math.cos(math.radians(37.5)) ** 3)
    with pytest.raises(ProfileDragError):
        sweep_drag_factor("cos2", 37.5)


def test_strip_profile_drag_uniform_strips_returns_section_cd() -> None:
    """No sweep, all strips at c_ref with the same cl: CD_profile = cd(cl) · (S/2) / (S/2)."""
    strips = StripLoads(alpha_deg=np.zeros(4), y_m=np.arange(4.0), chord_m=np.full(4, 2.0),
                        area_m2=np.full(4, 2.5), cl=np.full(4, 0.4))

    out, cd_profile = strip_profile_drag(strips, LOOKUP, sweep_deg=0.0, chord_ref_m=2.0,
                                         reynolds_ref=4e7, area_m2=20.0, mode="friction")

    assert cd_profile == pytest.approx(0.007, rel=1e-12)
    np.testing.assert_array_equal(out.status, np.full(4, IN_RANGE))


def test_turbulent_skin_friction_matches_hand_calculation() -> None:
    """Re = 1e7: 0.455 / 7^2.58 = 0.455 / 151.47 = 0.0030037 (hand calculation)."""
    assert float(turbulent_skin_friction(1e7)) == pytest.approx(0.455 / 7.0**2.58, rel=1e-12)
    assert float(turbulent_skin_friction(1e7)) == pytest.approx(0.0030037, abs=1e-7)


def test_reynolds_factor_is_one_at_reference_and_follows_local_exponent() -> None:
    """c = c_ref gives 1; a small chord change follows the local exponent −2.58 / ln Re."""
    re_ref = 4.1646e7
    factor = reynolds_factor(np.array([9.855, 9.855 * 1.001]), 9.855, re_ref)

    assert float(factor[0]) == pytest.approx(1.0, abs=1e-15)
    assert math.log(float(factor[1])) / math.log(1.001) == pytest.approx(
        -2.58 / math.log(re_ref), rel=1e-3)


def test_reynolds_factor_shorter_chord_has_more_drag() -> None:
    """Tip chord (1.569 m, Re = 6.63e6): (7.6196 / 6.8216)^2.58 = 1.330; root (14.63 m): 0.944."""
    factor = reynolds_factor(np.array([1.569, 14.63]), 9.855, 4.1646e7)

    assert float(factor[0]) == pytest.approx(1.330, abs=1e-3)
    assert float(factor[1]) == pytest.approx(0.944, abs=1e-3)


def test_strip_profile_drag_applies_sweep_and_reynolds_scaling() -> None:
    """cl_n = cl / cos²Λ is looked up; a chord twice c_ref scales cd by c_f(2·Re)/c_f(Re)."""
    sweep = 30.0
    cl_n = 0.4
    strips = StripLoads(alpha_deg=np.zeros(1), y_m=np.ones(1), chord_m=np.array([4.0]),
                        area_m2=np.array([10.0]),
                        cl=np.array([cl_n * math.cos(math.radians(sweep)) ** 2]))

    out, cd_profile = strip_profile_drag(strips, LOOKUP, sweep_deg=sweep, chord_ref_m=2.0,
                                         reynolds_ref=4e7, area_m2=20.0, mode="friction")

    assert float(out.cl_n[0]) == pytest.approx(cl_n)
    expected = 0.007 * (math.log10(4e7) / math.log10(8e7)) ** 2.58
    assert cd_profile == pytest.approx(expected, rel=1e-12)


def test_strip_profile_drag_with_a_stalled_strip_is_nan() -> None:
    """One strip beyond the polar makes the whole CD_profile NaN."""
    strips = StripLoads(alpha_deg=np.zeros(2), y_m=np.arange(2.0), chord_m=np.full(2, 2.0),
                        area_m2=np.full(2, 5.0), cl=np.array([0.4, 0.9]))

    out, cd_profile = strip_profile_drag(strips, LOOKUP, sweep_deg=0.0, chord_ref_m=2.0,
                                         reynolds_ref=4e7, area_m2=20.0, mode="friction")

    assert math.isnan(cd_profile)
    assert int(out.status[1]) == STALLED
