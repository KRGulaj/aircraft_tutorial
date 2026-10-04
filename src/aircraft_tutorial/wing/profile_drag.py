# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Profile drag of the wing from its VSPAERO strip loads and an XFoil section polar.

The vortex-lattice solution has no viscous drag. For every spanwise strip j at one angle of
attack:

1. the streamwise local cl_j is taken to the sweep-normal section, cl_n = cl_j / cos²Λ (the
   reduction the section polar was run for, Λ = quarter-chord sweep);
2. cd_n(cl_n) is interpolated linearly in the usable part of the polar;
3. cd_n is taken to the streamwise frame (`sweep_drag_factor`) and scaled from the polar's
   reference chord to the strip chord, cd = cd_n·k_Λ·(c_j / c_ref)^n, n = −0.2 (turbulent flat
   plate, c_f ∝ Re^−0.2; at fixed flight condition Re ∝ c);
4. the strips are integrated over both halves: CD_profile = Σ cd_j·dA_j / (S/2).

A strip whose cl_n lies outside the usable polar gets cd = NaN, flagged STALLED above the top
and BELOW_POLAR below the bottom. The CD_profile of that angle is then NaN: no clamped or
extrapolated value reaches the drag polar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from aircraft_tutorial.config.wing import SWEEP_DRAG_MODES
from aircraft_tutorial.contracts.section_polar import SectionPolarData
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


def strip_profile_drag(strips: StripLoads, lookup: DragLookup, *, sweep_deg: float,
                       chord_ref_m: float, area_m2: float, mode: str,
                       reynolds_exponent: float) -> tuple[StripDrag, float]:
    """Profile drag of one angle of attack from its strips.

    Args:
        strips: Strips of one angle of attack, right half-wing.
        lookup: Usable section polar.
        sweep_deg: Quarter-chord sweep [deg].
        chord_ref_m: Streamwise chord whose sweep-normal Reynolds number is that of the polar [m].
        area_m2: Wing reference area, both halves [m^2].
        mode: Sweep drag mode, see `sweep_drag_factor`.
        reynolds_exponent: n in (c / c_ref)^n [-].

    Returns:
        The strip values and CD_profile [-] (NaN if any strip is outside the polar).
    """
    cl_n = normal_cl(strips.cl, sweep_deg)
    cd_n, status = section_cd(cl_n, lookup)
    cd = (cd_n * sweep_drag_factor(mode, sweep_deg)
          * (strips.chord_m / chord_ref_m) ** reynolds_exponent)
    cd_profile = float(np.sum(cd * strips.area_m2) / (area_m2 / 2.0))
    return StripDrag(cl_n=cl_n, cd_n=cd_n, cd=cd, status=status), cd_profile
