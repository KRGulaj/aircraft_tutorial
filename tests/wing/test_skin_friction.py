# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Tests for the lecture's flat-plate skin-friction coefficient."""

from __future__ import annotations

import math

import pytest

from aircraft_tutorial.wing.skin_friction import (
    FrictionModel,
    SkinFrictionError,
    laminar_cf,
    turbulent_cf,
)


def _model(roughness_m: float = 0.634e-5, regime: str = "transonic", laminar: float = 0.0,
           mach: float = 0.855) -> FrictionModel:
    return FrictionModel(mach=mach, roughness_m=roughness_m, cutoff_regime=regime,
                         laminar_fraction=laminar)


def test_laminar_cf_matches_blasius() -> None:
    """Re = 1e6: 1.328 / 1000 = 0.001328."""
    assert float(laminar_cf(1e6)) == pytest.approx(0.001328, rel=1e-12)


def test_turbulent_cf_matches_hand_calculation() -> None:
    """Re = 1e7, M = 0: 0.455 / 7^2.58 = 0.0030037; M = 0.855 divides by 1.1053^0.65 = 1.0672."""
    assert float(turbulent_cf(1e7, 0.0)) == pytest.approx(0.0030037, abs=1e-7)
    assert float(turbulent_cf(1e7, 0.855)) == pytest.approx(
        0.455 / 7.0**2.58 / (1.0 + 0.144 * 0.855**2) ** 0.65, rel=1e-12)
    assert float(turbulent_cf(1e7, 0.855)) == pytest.approx(0.0028146, abs=1e-7)


def test_cutoff_reynolds_matches_lecture_formulas() -> None:
    """l/k = 1e6: subsonic 38.21·1e6^1.053, transonic 44.62·1e6^1.053·M^1.16."""
    subsonic = _model(roughness_m=1e-5, regime="subsonic")
    transonic = _model(roughness_m=1e-5, regime="transonic")

    assert float(subsonic.cutoff_reynolds(10.0)) == pytest.approx(38.21 * 1e6**1.053, rel=1e-12)
    assert float(transonic.cutoff_reynolds(10.0)) == pytest.approx(
        44.62 * 1e6**1.053 * 0.855**1.16, rel=1e-12)


def test_b747_mac_smooth_paint_is_below_cutoff() -> None:
    """MAC 9.855 m, smooth paint: l/k = 1.5544e6, cutoff 44.62·(l/k)^1.053·0.855^1.16 = 1.231e8
    > Re = 6.6e7, so the actual Re is used."""
    model = _model()

    assert float(model.cutoff_reynolds(9.855)) == pytest.approx(1.231e8, rel=1e-3)
    assert float(model.effective_reynolds(6.6e7, 9.855)) == 6.6e7


def test_rough_surface_caps_reynolds_at_cutoff() -> None:
    """A rough surface gives a cutoff below the actual Re; C_f uses the cutoff."""
    model = _model(roughness_m=1e-3)
    cutoff = float(model.cutoff_reynolds(1.0))

    assert cutoff < 6.6e7
    assert float(model.cf(6.6e7, 1.0)) == pytest.approx(float(turbulent_cf(cutoff, 0.855)),
                                                        rel=1e-12)


def test_cf_weights_laminar_and_turbulent() -> None:
    """10 % laminar: C_f = 0.1·C_f,lam + 0.9·C_f,turb (lecture example)."""
    model = _model(laminar=0.1)

    expected = 0.1 * 1.328 / math.sqrt(1e7) + 0.9 * float(turbulent_cf(1e7, 0.855))
    assert float(model.cf(1e7, 5.0)) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(("roughness", "regime", "laminar"),
                         [(0.0, "transonic", 0.0), (1e-5, "hypersonic", 0.0),
                          (1e-5, "subsonic", 1.5)],
                         ids=["zero_roughness", "unknown_regime", "laminar_above_one"])
def test_invalid_model_raises(roughness: float, regime: str, laminar: float) -> None:
    """Non-positive roughness, an unknown regime or a laminar fraction above 1 are rejected."""
    with pytest.raises(SkinFrictionError):
        _model(roughness_m=roughness, regime=regime, laminar=laminar)
