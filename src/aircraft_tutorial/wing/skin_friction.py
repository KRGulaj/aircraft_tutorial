# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Flat-plate skin-friction coefficient of the AE2111-II lecture (ADSEE-2, "Flat plate skin
friction coefficient"; the same relations as Raymer, Aircraft Design: A Conceptual Approach,
component build-up method).

    laminar:     C_f = 1.328 / √Re
    turbulent:   C_f = 0.455 / ((log₁₀ Re)^2.58·(1 + 0.144·M²)^0.65)

    Re = min(ρ·V·l / μ, Re_cutoff)
    subsonic:                Re_cutoff = 38.21·(l / k)^1.053
    transonic, supersonic:   Re_cutoff = 44.62·(l / k)^1.053·M^1.16

k is the surface roughness height; below the cutoff the surface is hydraulically smooth. The total
C_f is the average of the two weighted by the laminar fraction of the chord:

    C_f = x_lam·C_f,lam + (1 − x_lam)·C_f,turb
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

CUTOFF_REGIMES: Final[tuple[str, ...]] = ("subsonic", "transonic")


class SkinFrictionError(ValueError):
    """Raised for an invalid friction-model setting."""


@dataclass(frozen=True)
class FrictionModel:
    """Settings of the flat-plate skin-friction coefficient.

    Attributes:
        mach: Mach number of the flow the coefficient is evaluated in [-].
        roughness_m: Surface roughness height k [m].
        cutoff_regime: "subsonic" or "transonic" cutoff-Reynolds formula.
        laminar_fraction: Laminar part of the chord x_lam [-].
    """

    mach: float
    roughness_m: float
    cutoff_regime: str
    laminar_fraction: float

    def __post_init__(self) -> None:
        """Check the settings."""
        if self.cutoff_regime not in CUTOFF_REGIMES:
            raise SkinFrictionError(f"cutoff_regime must be one of {CUTOFF_REGIMES}, "
                                    f"got {self.cutoff_regime!r}")
        if not (0.0 <= self.mach < 5.0 and self.roughness_m > 0.0
                and 0.0 <= self.laminar_fraction <= 1.0):
            raise SkinFrictionError(f"invalid friction model {self}")

    def cutoff_reynolds(self, length_m: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """Cutoff Reynolds number of the roughness for a run length l [-]."""
        ratio = np.asarray(length_m, dtype=np.float64) / self.roughness_m
        if self.cutoff_regime == "subsonic":
            return 38.21 * ratio**1.053
        return np.asarray(44.62 * ratio**1.053 * self.mach**1.16, dtype=np.float64)

    def effective_reynolds(self, reynolds: NDArray[np.float64] | float,
                           length_m: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """Re = min(actual Re, cutoff Re) [-]."""
        return np.minimum(np.asarray(reynolds, dtype=np.float64), self.cutoff_reynolds(length_m))

    def cf(self, reynolds: NDArray[np.float64] | float,
           length_m: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """Skin-friction coefficient, laminar and turbulent parts weighted [-].

        Args:
            reynolds: Actual Reynolds number ρ·V·l / μ [-].
            length_m: Run length l the Reynolds number is based on [m].
        """
        re = self.effective_reynolds(reynolds, length_m)
        x = self.laminar_fraction
        return x * laminar_cf(re) + (1.0 - x) * turbulent_cf(re, self.mach)


def laminar_cf(reynolds: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """Laminar flat-plate C_f = 1.328 / √Re (Blasius) [-]."""
    return 1.328 / np.sqrt(np.asarray(reynolds, dtype=np.float64))


def turbulent_cf(reynolds: NDArray[np.float64] | float, mach: float) -> NDArray[np.float64]:
    """Turbulent flat-plate C_f = 0.455 / ((log₁₀ Re)^2.58·(1 + 0.144·M²)^0.65) [-]."""
    re = np.asarray(reynolds, dtype=np.float64)
    return np.asarray(0.455 / (np.log10(re) ** 2.58 * (1.0 + 0.144 * mach**2) ** 0.65),
                      dtype=np.float64)
