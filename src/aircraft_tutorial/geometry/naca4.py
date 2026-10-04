# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""NACA four-digit sections from the analytic definition (Abbott & von Doenhoff, §6.4).

Thickness, eq. (6.2), with t the maximum thickness as a chord fraction:

    ±y_t = (t / 0.20)·(0.29690·√x − 0.12600·x − 0.35160·x² + 0.28430·x³ − 0.10150·x⁴)

The coefficient −0.10150 leaves a trailing-edge half-thickness of 0.0021·t / 0.20, as in the
tested models. Mean line, eq. (6.4), m the maximum camber and p its chordwise position:

    y_c = m / p²·(2p·x − x²)                       for x ≤ p
    y_c = m / (1 − p)²·[(1 − 2p) + 2p·x − x²]      for x > p
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aircraft_tutorial.geometry.airfoil import Airfoil
from aircraft_tutorial.geometry.construction import combine


@dataclass(frozen=True)
class Naca4:
    """Parameters of a NACA four-digit section.

    Attributes:
        max_camber: m [chord fraction].
        camber_position: p [chord fraction]. Zero for a symmetric section.
        thickness: t [chord fraction].
    """

    max_camber: float
    camber_position: float
    thickness: float


def parse_naca4(designation: str) -> Naca4:
    """Read a four-digit designation such as "2412".

    Args:
        designation: Four digits: camber [% c], camber position [tenths of c], thickness [% c].

    Returns:
        The section parameters.

    Raises:
        ValueError: If the designation is not four digits, or the camber and its position are
            not both zero or both non-zero.
    """
    if not (len(designation) == 4 and designation.isdigit()):
        raise ValueError(f"NACA four-digit designation must be 4 digits, got {designation!r}")
    m = int(designation[0]) / 100.0
    p = int(designation[1]) / 10.0
    t = int(designation[2:]) / 100.0
    if (m == 0.0) != (p == 0.0):
        raise ValueError(f"NACA {designation}: camber and camber position must both be zero "
                         "or both non-zero")
    if not (t > 0.0):
        raise ValueError(f"NACA {designation}: thickness must be > 0")
    return Naca4(max_camber=m, camber_position=p, thickness=t)


def thickness(x: NDArray[np.float64], t: float) -> NDArray[np.float64]:
    """Half-thickness y_t, Abbott eq. (6.2).

    Args:
        x: Chord stations [chord fraction].
        t: Maximum thickness [chord fraction].

    Returns:
        Half-thickness [chord fraction].
    """
    return (t / 0.20) * (0.29690 * np.sqrt(x) - 0.12600 * x - 0.35160 * x**2
                         + 0.28430 * x**3 - 0.10150 * x**4)


def mean_line(x: NDArray[np.float64], m: float, p: float
              ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mean-line ordinate and slope, Abbott eq. (6.4).

    Args:
        x: Chord stations [chord fraction].
        m: Maximum camber [chord fraction].
        p: Position of the maximum camber [chord fraction]; ignored if m is zero.

    Returns:
        (y_c [chord fraction], dy_c/dx [-]).
    """
    if m == 0.0:
        return np.zeros_like(x), np.zeros_like(x)
    fwd = x <= p
    y_c = np.where(fwd, m / p**2 * (2.0 * p * x - x**2),
                   m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2))
    dyc = np.where(fwd, 2.0 * m / p**2 * (p - x), 2.0 * m / (1.0 - p) ** 2 * (p - x))
    return y_c, dyc


def naca4(designation: str, x: NDArray[np.float64]) -> Airfoil:
    """Generate a NACA four-digit section at the given chord stations.

    Args:
        designation: Four digits, for example "2412".
        x: Chord stations from 0 to 1, ascending [chord fraction].

    Returns:
        The closed contour in Selig order, named "NACA <designation>".
    """
    params = parse_naca4(designation)
    y_c, dyc = mean_line(x, params.max_camber, params.camber_position)
    return combine(f"NACA {designation}", x, thickness(x, params.thickness), y_c, dyc)
