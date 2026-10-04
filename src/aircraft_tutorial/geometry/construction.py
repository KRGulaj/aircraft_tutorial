# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Combination of a mean line and a thickness distribution (Abbott & von Doenhoff, §6.3).

The thickness is laid off perpendicular to the mean line, Abbott eq. (6.1):

    x_U = x − y_t·sin θ,   y_U = y_c + y_t·cos θ
    x_L = x + y_t·sin θ,   y_L = y_c − y_t·cos θ,   θ = arctan(dy_c/dx)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from aircraft_tutorial.geometry.airfoil import Airfoil, from_surfaces


def combine(name: str, x: NDArray[np.float64], y_t: NDArray[np.float64],
            y_c: NDArray[np.float64], dyc_dx: NDArray[np.float64]) -> Airfoil:
    """Build the section contour from a mean line and a thickness distribution.

    Args:
        name: Section name.
        x: Chord stations from 0 to 1, ascending [chord fraction].
        y_t: Half-thickness at the stations [chord fraction].
        y_c: Mean-line ordinate at the stations [chord fraction].
        dyc_dx: Mean-line slope at the stations [-].

    Returns:
        The closed contour in Selig order.
    """
    theta = np.arctan(dyc_dx)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    return from_surfaces(
        name,
        x - y_t * sin_t, y_c + y_t * cos_t,
        x + y_t * sin_t, y_c - y_t * cos_t,
    )
