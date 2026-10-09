# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-09
"""Simple sweep theory: the section normal to the sweep line of a wing with streamwise sections.

The 3D model places the airfoil of the coordinate file in the streamwise plane. A cut normal to
the sweep line Λ crosses the same surface, so the thickness and camber heights z are unchanged,
but the chord shrinks to c_n = c·cos Λ (Anderson, Fundamentals of Aerodynamics, 6th ed.,
section 11.7, Design Box, Figure 11.14). In chord fractions:

    x_n / c_n = x / c,   z_n / c_n = (z / c) / cos Λ

so t/c and the camber of the normal section are those of the streamwise section divided by
cos Λ.

For the free stream V at angle α_s in the streamwise plane, the component normal to the sweep
line in the wing plane is V·cos α_s·cos Λ and the vertical component is V·sin α_s, so the angle
of attack of the normal section is

    tan α_n = tan α_s / cos Λ.
"""

from __future__ import annotations

import math

from aircraft_tutorial.geometry.airfoil import Airfoil, AirfoilError, normalized


def _cos_sweep(sweep_deg: float) -> float:
    """cos Λ for a sweep in [0, 90) deg; raises AirfoilError otherwise."""
    if not (0.0 <= sweep_deg < 90.0):
        raise AirfoilError(f"sweep must be in [0, 90) deg, got {sweep_deg}")
    return math.cos(math.radians(sweep_deg))


def sweep_normal_section(airfoil: Airfoil, sweep_deg: float) -> Airfoil:
    """Section normal to the sweep line of a streamwise airfoil.

    Args:
        airfoil: Streamwise section.
        sweep_deg: Sweep Λ of the line the cut is normal to [deg], in [0, 90).

    Returns:
        The normal section, chord normalised to 1 [chord fraction], named
        "<name> normal to <Λ> deg sweep".

    Raises:
        AirfoilError: If the sweep is outside [0, 90) deg.
    """
    cos_sweep = _cos_sweep(sweep_deg)
    unit = normalized(airfoil)
    return Airfoil(name=f"{airfoil.name} normal to {sweep_deg:g} deg sweep", x=unit.x,
                   y=unit.y / cos_sweep)


def streamwise_angle_deg(alpha_normal_deg: float, sweep_deg: float) -> float:
    """Streamwise angle of attack from the normal-section one, tan α_s = tan α_n·cos Λ.

    Args:
        alpha_normal_deg: Angle of attack of the normal section α_n [deg], in (−90, 90).
        sweep_deg: Sweep Λ of the line the section is normal to [deg], in [0, 90).

    Returns:
        α_s [deg].

    Raises:
        AirfoilError: If an angle is outside its range.
    """
    if not (-90.0 < alpha_normal_deg < 90.0):
        raise AirfoilError(f"alpha_normal_deg must be in (-90, 90), got {alpha_normal_deg}")
    tan_alpha = math.tan(math.radians(alpha_normal_deg))
    return math.degrees(math.atan(tan_alpha * _cos_sweep(sweep_deg)))
