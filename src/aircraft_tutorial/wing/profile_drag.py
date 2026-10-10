# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Profile drag of the wing from its VSPAERO strip loads and an XFoil section polar.

The vortex-lattice solution has no viscous drag. For every spanwise strip j at one angle of
attack:

1. the streamwise local cl_j is taken to the sweep-normal section, cl_n = cl_j / cos²Λ (the
   reduction the section polar was run for, Λ = quarter-chord sweep);
2. cd_n(cl_n) is interpolated linearly in the usable part of the polar;
3. cd_n is taken to the streamwise frame (`sweep_drag_factor`) and scaled from the condition
   of the polar to that of the strip in flight, cd = cd_n·k_Λ·k_Re,j (`reynolds_factor`);
4. the strips are integrated over both halves: CD_profile = Σ cd_j·dA_j / (S/2).

Skin-friction scaling. Cruise profile drag is mostly skin friction, so cd is scaled like the
flat-plate skin-friction coefficient of the lecture (wing/skin_friction.py: laminar/turbulent
C_f weighted by the laminar fraction, turbulent C_f = 0.455 / ((log₁₀ Re)^2.58·(1 + 0.144·M²)^0.65),
Re capped at the roughness cutoff):

    k_Re,j = C_f(strip in flight) / C_f(polar)

- Denominator, the polar: XFoil models a smooth wall at the polar's Reynolds and Mach numbers,
  so C_f has no roughness cutoff and uses M of the polar.
- Numerator, the strip: the flight Mach number and the paint roughness, in the frame of the
  sweep model. With k_Λ = 1 ("friction") the friction acts along the streamline: run length
  l = c, Re = Re_n / cos²Λ, M = M∞. With k_Λ = cos³Λ ("cos3") the flow is the sweep-normal one:
  l = c·cos Λ, Re = Re_n, M = M∞·cos Λ. In both, Re_n = Re_ref·c_j / c_ref at a fixed flight
  condition.

The ratio therefore also carries the compressibility reduction of C_f, which XFoil at low Mach
number does not contain. The form factor (pressure drag) is taken as independent of Re and M.

A strip whose cl_n lies outside the usable polar gets cd = NaN, flagged STALLED above the top
and BELOW_POLAR below the bottom. The CD_profile of that angle is then NaN: no clamped or
extrapolated value reaches the drag polar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Final

import numpy as np
from numpy.typing import NDArray

from aircraft_tutorial.config.wing import SWEEP_DRAG_MODES
from aircraft_tutorial.contracts.section_polar import SectionPolarData
from aircraft_tutorial.wing.skin_friction import FrictionModel
from aircraft_tutorial.wing.vlm_results import StripLoads

IN_RANGE: Final[int] = 0
STALLED: Final[int] = 1
"""cl_n above the top of the usable polar."""
BELOW_POLAR: Final[int] = -1
"""cl_n below the bottom of the usable polar."""


class ProfileDragError(ValueError):
    """Raised when the polar has no usable range or a setting is invalid."""


@dataclass(frozen=True)
class DragLookup:
    """Usable part of a section polar for the cd(cl) lookup, cl strictly ascending.

    Attributes:
        airfoil: Section name.
        alpha_deg: Angle of attack [deg].
        cl: Lift coefficient [-].
        cd: Drag coefficient [-].
    """

    airfoil: str
    alpha_deg: NDArray[np.float64]
    cl: NDArray[np.float64]
    cd: NDArray[np.float64]


@dataclass(frozen=True)
class StripDrag:
    """Profile drag of the strips of one angle of attack.

    Attributes:
        cl_n: Sweep-normal section lift coefficient [-].
        cd_n: Section drag coefficient from the polar [-].
        cd: Streamwise strip drag coefficient on the strip chord [-].
        status: IN_RANGE, STALLED or BELOW_POLAR.
    """

    cl_n: NDArray[np.float64]
    cd_n: NDArray[np.float64]
    cd: NDArray[np.float64]
    status: NDArray[np.int_]


def usable_polar(polar: SectionPolarData) -> DragLookup:
    """Converged points on the attached-flow branch, where cl(α) rises monotonically.

    Starting at the converged point closest to α = 0, the range is extended in both directions
    while cl keeps rising with α. It ends at the first drop of cl, so XFoil's post-stall
    branches, where cl(α) and therefore cd(cl) are not single-valued, are never used.

    Args:
        polar: Section polar.

    Returns:
        The lookup table.

    Raises:
        ProfileDragError: If fewer than two points remain.
    """
    m = polar.converged
    alpha, cl, cd = polar.alpha_deg[m], polar.cl[m], polar.cd[m]
    if alpha.size < 2:
        raise ProfileDragError(f"{polar.airfoil}: fewer than 2 converged points")
    lo = hi = int(np.argmin(np.abs(alpha)))
    while lo > 0 and cl[lo - 1] < cl[lo]:
        lo -= 1
    while hi < alpha.size - 1 and cl[hi + 1] > cl[hi]:
        hi += 1
    if hi - lo < 1:
        raise ProfileDragError(f"{polar.airfoil}: no rising cl(alpha) range around 0 deg")
    s = slice(lo, hi + 1)
    return DragLookup(polar.airfoil, alpha[s].copy(), cl[s].copy(), cd[s].copy())


def normal_cl(cl: NDArray[np.float64], sweep_deg: float) -> NDArray[np.float64]:
    """Streamwise section cl → sweep-normal cl_n = cl / cos²Λ."""
    return np.asarray(cl, dtype=np.float64) / math.cos(math.radians(sweep_deg)) ** 2


def section_cd(cl_n: NDArray[np.float64], lookup: DragLookup
               ) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """cd_n at each cl_n by linear interpolation, NaN outside the usable polar.

    Returns:
        (cd_n [-], status).
    """
    cl_n = np.asarray(cl_n, dtype=np.float64)
    cd_n = np.interp(cl_n, lookup.cl, lookup.cd)
    status = np.full(cl_n.shape, IN_RANGE, dtype=np.int_)
    status[cl_n > lookup.cl[-1]] = STALLED
    status[cl_n < lookup.cl[0]] = BELOW_POLAR
    cd_n[status != IN_RANGE] = np.nan
    return cd_n, status


def sweep_drag_factor(mode: str, sweep_deg: float) -> float:
    """Factor k_Λ from the sweep-normal section cd_n to the streamwise cd.

    "friction": k_Λ = 1. Cruise profile drag is mostly skin friction, which acts along the
        local streamline and is not reduced by sweep. Conservative; the baseline.
    "cos3": k_Λ = cos³Λ. Simple sweep theory, strictly valid for pressure drag only.
        Optimistic; reported as a sensitivity.

    Raises:
        ProfileDragError: For an unknown mode.
    """
    if mode == "friction":
        return 1.0
    if mode == "cos3":
        return math.cos(math.radians(sweep_deg)) ** 3
    raise ProfileDragError(f"unknown sweep drag mode {mode!r}, expected one of {SWEEP_DRAG_MODES}")


def reynolds_factor(chord_m: NDArray[np.float64], chord_ref_m: float, reynolds_ref: float,
                    sweep_deg: float, mode: str, flight: FrictionModel,
                    polar_mach: float) -> NDArray[np.float64]:
    """Scaling of cd from the polar to the strip in flight, k_Re = C_f(strip) / C_f(polar).

    Numerator: `flight` (paint roughness) in the frame of the sweep mode, with
    Re_n = Re_ref·c_j / c_ref:

    - "friction": along the streamline, l = c, Re = Re_n / cos²Λ, M = M∞;
    - "cos3": normal to the sweep line, l = c·cos Λ, Re = Re_n, M = M∞·cos Λ.

    Denominator: smooth wall (XFoil) at Re_ref and the polar's Mach number.

    Args:
        chord_m: Streamwise strip chords [m].
        chord_ref_m: Streamwise chord whose sweep-normal Reynolds number is the polar's [m].
        reynolds_ref: Reynolds number of the polar [-].
        sweep_deg: Quarter-chord sweep [deg].
        mode: Sweep drag mode, see `sweep_drag_factor`.
        flight: Skin-friction model of the wing in flight: free-stream Mach number, roughness,
            cutoff regime, laminar fraction.
        polar_mach: Mach number of the section polar [-].

    Returns:
        The factor [-].

    Raises:
        ProfileDragError: For an unknown mode.
    """
    chord = np.asarray(chord_m, dtype=np.float64)
    cos_sweep = math.cos(math.radians(sweep_deg))
    reynolds_normal = reynolds_ref * chord / chord_ref_m
    if mode == "friction":
        cf_strip = flight.cf(reynolds_normal / cos_sweep**2, chord)
    elif mode == "cos3":
        normal = replace(flight, mach=flight.mach * cos_sweep)
        cf_strip = normal.cf(reynolds_normal, chord * cos_sweep)
    else:
        raise ProfileDragError(f"unknown sweep drag mode {mode!r}, "
                               f"expected one of {SWEEP_DRAG_MODES}")
    cf_polar = replace(flight, mach=polar_mach).smooth_cf(reynolds_ref)
    return cf_strip / cf_polar


def strip_profile_drag(strips: StripLoads, lookup: DragLookup, *, sweep_deg: float,
                       chord_ref_m: float, reynolds_ref: float, polar_mach: float,
                       area_m2: float, mode: str,
                       flight: FrictionModel) -> tuple[StripDrag, float]:
    """Profile drag of one angle of attack from its strips.

    Args:
        strips: Strips of one angle of attack, right half-wing.
        lookup: Usable section polar.
        sweep_deg: Quarter-chord sweep [deg].
        chord_ref_m: Streamwise chord whose sweep-normal Reynolds number is that of the polar [m].
        reynolds_ref: Reynolds number of the polar [-].
        polar_mach: Mach number of the polar [-].
        area_m2: Wing reference area, both halves [m^2].
        mode: Sweep drag mode, see `sweep_drag_factor`.
        flight: Skin-friction model of the wing in flight, see `reynolds_factor`.

    Returns:
        The strip values and CD_profile [-] (NaN if any strip is outside the polar).
    """
    cl_n = normal_cl(strips.cl, sweep_deg)
    cd_n, status = section_cd(cl_n, lookup)
    cd = (cd_n * sweep_drag_factor(mode, sweep_deg)
          * reynolds_factor(strips.chord_m, chord_ref_m, reynolds_ref, sweep_deg, mode, flight,
                            polar_mach))
    cd_profile = float(np.sum(cd * strips.area_m2) / (area_m2 / 2.0))
    return StripDrag(cl_n=cl_n, cd_n=cd_n, cd=cd, status=status), cd_profile
