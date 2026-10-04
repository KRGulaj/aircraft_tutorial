# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Compare two contours of the same section, split into thickness and camber differences.

At each station x the thickness is t = y_U − y_L and the camber ordinate is c = (y_U + y_L) / 2.
Splitting the difference this way separates a shape error (thickness) from a shifted or
rotated chord line (camber).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from aircraft_tutorial.geometry.airfoil import Airfoil

_N_STATIONS: Final[int] = 500
EDGE_MARGIN: Final[float] = 0.005
"""Stations closer than this to either edge are skipped: interpolation across the leading-edge
curvature is not a meaningful measure of shape agreement."""


@dataclass(frozen=True)
class ContourDifference:
    """Difference reference minus other, over x in [EDGE_MARGIN, 1 − EDGE_MARGIN].

    Attributes:
        thickness_max_abs: max |Δt| [chord fraction].
        thickness_rms: RMS of Δt [chord fraction].
        camber_min: min Δc [chord fraction].
        camber_max: max Δc [chord fraction].
        camber_rms: RMS of Δc [chord fraction].
    """

    thickness_max_abs: float
    thickness_rms: float
    camber_min: float
    camber_max: float
    camber_rms: float


def compare(reference: Airfoil, other: Airfoil) -> ContourDifference:
    """Thickness and camber difference between two contours with unit chord.

    Args:
        reference: Contour taken as correct.
        other: Contour to check.

    Returns:
        The differences reference − other.
    """
    xs = np.linspace(EDGE_MARGIN, 1.0 - EDGE_MARGIN, _N_STATIONS)
    t_ref, c_ref = _thickness_camber(reference, xs)
    t_oth, c_oth = _thickness_camber(other, xs)
    dt = t_ref - t_oth
    dc = c_ref - c_oth
    return ContourDifference(
        thickness_max_abs=float(np.max(np.abs(dt))),
        thickness_rms=float(np.sqrt(np.mean(dt**2))),
        camber_min=float(dc.min()),
        camber_max=float(dc.max()),
        camber_rms=float(np.sqrt(np.mean(dc**2))),
    )


def _thickness_camber(airfoil: Airfoil, xs: NDArray[np.float64]
                      ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Thickness and camber ordinate at the stations."""
    xu, yu, xl, yl = airfoil.surfaces()
    upper = np.interp(xs, xu, yu)
    lower = np.interp(xs, xl, yl)
    return upper - lower, 0.5 * (upper + lower)
