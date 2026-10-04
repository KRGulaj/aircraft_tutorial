# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Analytical lift curve of the wing and the twist sizing done with it.

Used as a comparison with VSPAERO. Lift-curve slope, semi-empirical DATCOM form of the
AE2111-II lecture (ADSEE-2, "Evaluation of the wing lift curve slope", slide 20):

    CL_α = 2π·A / (2 + √(4 + (A·β/η)²·[1 + tan²Λ_0.5c / β²])) [1/rad],   β = √(1 − M²)

with η = 0.95 the airfoil efficiency factor.

Lift curve. The wing is taken at its mean geometric angle: the body angle α plus the root
incidence i_r plus the chord-weighted mean of the linear twist ε·η_s,

    ε_mean = ε·∫c·η_s dy / ∫c dy = ε·(1 + 2λ) / (3·(1 + λ)) = k_ε·ε,

measured from the section zero-lift angle α_0l:

    CL = CL_α·(α + i_r + k_ε·ε − α_0l)

so the wing zero-lift angle is α_0L = α_0l − i_r − k_ε·ε (body axis). Weighting the twist by the
chord is the strip-theory estimate of how much a twisted station adds to the lift; for this
wing VSPAERO gives ∂CL/∂ε / CL_α = 0.36 against k_ε = 0.366.

Twist sizing: CL = CL_des at α = 0 gives ε = (CL_des / CL_α − i_r + α_0l) / k_ε.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class AnalyticError(ValueError):
    """Raised for inputs outside the validity of the method or a twist beyond its limit."""


@dataclass(frozen=True)
class AnalyticWing:
    """Analytical lift model of one wing.

    Attributes:
        cl_alpha_per_rad: Wing lift-curve slope [1/rad].
        twist_effectiveness: k_ε, chord-weighted mean of a unit linear twist [-].
        root_incidence_deg: Root incidence [deg].
        section_alpha_0l_deg: Section zero-lift angle α_0l [deg].
        twist_deg: Tip twist relative to the root [deg].
    """

    cl_alpha_per_rad: float
    twist_effectiveness: float
    root_incidence_deg: float
    section_alpha_0l_deg: float
    twist_deg: float

    @property
    def alpha_zero_lift_deg(self) -> float:
        """Wing zero-lift body angle α_0L = α_0l − i_r − k_ε·ε [deg]."""
        return (self.section_alpha_0l_deg - self.root_incidence_deg
                - self.twist_effectiveness * self.twist_deg)

    def cl(self, alpha_deg: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """Wing lift coefficient at body angles of attack [-].

        Args:
            alpha_deg: Body angle of attack [deg].
        """
        alpha = np.asarray(alpha_deg, dtype=np.float64)
        return self.cl_alpha_per_rad * np.radians(alpha - self.alpha_zero_lift_deg)


def datcom_lift_slope(aspect_ratio: float, mach: float, sweep_half_chord_deg: float,
                      eta: float) -> float:
    """Wing lift-curve slope, DATCOM form of the lecture.

    Args:
        aspect_ratio: A [-].
        mach: Free-stream Mach number, below 1 [-].
        sweep_half_chord_deg: Half-chord sweep Λ_0.5c [deg].
        eta: Airfoil efficiency factor η [-].

    Returns:
        CL_α [1/rad].

    Raises:
        AnalyticError: If the Mach number is not subsonic or η is not positive.
    """
    if not (0.0 <= mach < 1.0):
        raise AnalyticError(f"mach must be in [0, 1), got {mach}")
    if not (eta > 0.0):
        raise AnalyticError(f"eta must be > 0, got {eta}")
    beta = math.sqrt(1.0 - mach**2)
    tan_sweep = math.tan(math.radians(sweep_half_chord_deg))
    root = math.sqrt(4.0 + (aspect_ratio * beta / eta) ** 2 * (1.0 + tan_sweep**2 / beta**2))
    return 2.0 * math.pi * aspect_ratio / (2.0 + root)


def twist_effectiveness(taper: float) -> float:
    """k_ε = (1 + 2λ) / (3·(1 + λ)), chord-weighted mean of a linear twist that is 1 at the tip.

    Args:
        taper: Taper ratio λ [-].
    """
    return (1.0 + 2.0 * taper) / (3.0 * (1.0 + taper))


def size_twist(cl_design: float, cl_alpha_per_rad: float, *, taper: float,
               root_incidence_deg: float, section_alpha_0l_deg: float,
               max_twist_deg: float) -> AnalyticWing:
    """Twist that gives CL_des at zero body angle, and the resulting lift model.

    Args:
        cl_design: Design lift coefficient [-].
        cl_alpha_per_rad: Wing lift-curve slope [1/rad].
        taper: Taper ratio λ [-].
        root_incidence_deg: Root incidence [deg].
        section_alpha_0l_deg: Section zero-lift angle [deg].
        max_twist_deg: Largest allowed |twist| [deg].

    Returns:
        The sized lift model.

    Raises:
        AnalyticError: If the required twist exceeds the limit.
    """
    k = twist_effectiveness(taper)
    twist = (math.degrees(cl_design / cl_alpha_per_rad) - root_incidence_deg
             + section_alpha_0l_deg) / k
    if abs(twist) > max_twist_deg:
        raise AnalyticError(f"CL_des = {cl_design:.4f} needs a twist of {twist:.3f} deg, beyond "
                            f"the allowed ±{max_twist_deg} deg")
    return AnalyticWing(cl_alpha_per_rad, k, root_incidence_deg, section_alpha_0l_deg, twist)
