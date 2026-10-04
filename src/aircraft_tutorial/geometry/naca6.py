# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""NACA 66-series sections with the a = 1.0 mean line (Abbott & von Doenhoff, §6.8, §6.3).

There is no closed-form 6-series thickness distribution: it comes from a conformal mapping and is
published as a table. The thickness here is the tabulated basic form (App. I), interpolated by a
cubic spline in s = √x. Near the leading edge y_t ≈ √(2·r_LE·x) = √(2·r_LE)·s, so the spline is
clamped to dy_t/ds = √(2·r_LE) at s = 0. This uses the tabulated leading-edge radius exactly.

The a = 1.0 (uniform load) mean line is analytic, App. II p.405:

    y_c    = −(c_li / 4π)·[x·ln x + (1 − x)·ln(1 − x)]
    dy_c/dx = (c_li / 4π)·ln((1 − x) / x)

The slope is infinite at both ends. Abbott (p.113) takes the slope at x = 0.005 at the leading
edge; the same clamp is used at x = 0.995 at the trailing edge, where y_t is below 0.0005 c and
the effect on the contour is negligible.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from importlib import resources
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from aircraft_tutorial.geometry.airfoil import Airfoil
from aircraft_tutorial.geometry.construction import combine

SLOPE_CLAMP_X: Final[float] = 0.005

_DESIGNATION = re.compile(r"^66-(\d)(\d{2})$")
_THICKNESS_TABLES: Final[dict[int, tuple[str, float]]] = {
    10: ("naca66_010.csv", 0.00662),
}
"""Thickness [% c] → (table file, leading-edge radius [chord fraction])."""


@dataclass(frozen=True)
class Naca66:
    """Parameters of a NACA 66-series section with the a = 1.0 mean line.

    Attributes:
        design_cl: Design lift coefficient c_li [-].
        thickness_percent: Maximum thickness [% c].
    """

    design_cl: float
    thickness_percent: int


def parse_naca66(designation: str) -> Naca66:
    """Read a designation such as "66-410" (design c_l 0.4, 10 % thick).

    Args:
        designation: "66-" followed by the design c_l in tenths and the thickness in % c.

    Returns:
        The section parameters.

    Raises:
        ValueError: If the designation does not match, or no thickness table exists for it.
    """
    match = _DESIGNATION.match(designation)
    if match is None:
        raise ValueError(f"expected a designation like '66-410', got {designation!r}")
    t = int(match.group(2))
    if t not in _THICKNESS_TABLES:
        raise ValueError(f"NACA {designation}: no thickness table for {t} % "
                         f"(available: {sorted(_THICKNESS_TABLES)})")
    return Naca66(design_cl=int(match.group(1)) / 10.0, thickness_percent=t)


def thickness_table(thickness_percent: int) -> tuple[NDArray[np.float64], NDArray[np.float64],
                                                     float]:
    """Load a tabulated basic thickness form.

    Args:
        thickness_percent: Maximum thickness [% c].

    Returns:
        (x [chord fraction], y_t [chord fraction], leading-edge radius [chord fraction]).

    Raises:
        ValueError: If no table exists for the thickness.
    """
    if thickness_percent not in _THICKNESS_TABLES:
        raise ValueError(f"no thickness table for {thickness_percent} %")
    file, r_le = _THICKNESS_TABLES[thickness_percent]
    text = resources.files("aircraft_tutorial.geometry").joinpath("data", file).read_text(
        encoding="utf-8")
    data = _read_csv(text)
    return data[:, 0] / 100.0, data[:, 1] / 100.0, r_le


def thickness(x: NDArray[np.float64], thickness_percent: int) -> NDArray[np.float64]:
    """Half-thickness y_t from the tabulated form, spline in s = √x clamped by the LE radius.

    Args:
        x: Chord stations in [0, 1] [chord fraction].
        thickness_percent: Maximum thickness [% c].

    Returns:
        Half-thickness [chord fraction].
    """
    xt, yt, r_le = thickness_table(thickness_percent)
    spline = CubicSpline(np.sqrt(xt), yt, bc_type=((1, math.sqrt(2.0 * r_le)), "not-a-knot"))
    return np.asarray(spline(np.sqrt(x)), dtype=np.float64)


def mean_line_a10(x: NDArray[np.float64], design_cl: float
                  ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """a = 1.0 mean line ordinate and slope (App. II, p.405).

    Args:
        x: Chord stations in [0, 1] [chord fraction].
        design_cl: Design lift coefficient c_li [-].

    Returns:
        (y_c [chord fraction], dy_c/dx [-]); the slope is clamped at x = 0.005 and 0.995.
    """
    k = design_cl / (4.0 * math.pi)
    y_c = -k * (_x_ln_x(x) + _x_ln_x(1.0 - x))
    xs = np.clip(x, SLOPE_CLAMP_X, 1.0 - SLOPE_CLAMP_X)
    return y_c, k * np.log((1.0 - xs) / xs)


def naca66(designation: str, x: NDArray[np.float64]) -> Airfoil:
    """Generate a NACA 66-series section (a = 1.0 mean line) at the given chord stations.

    Args:
        designation: For example "66-410".
        x: Chord stations from 0 to 1, ascending [chord fraction].

    Returns:
        The closed contour in Selig order, named "NACA <designation>".
    """
    params = parse_naca66(designation)
    y_c, dyc = mean_line_a10(x, params.design_cl)
    return combine(f"NACA {designation}", x, thickness(x, params.thickness_percent), y_c, dyc)


def _x_ln_x(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """x·ln x with the limit value 0 at x = 0."""
    safe = np.where(x > 0.0, x, 1.0)
    return np.where(x > 0.0, x * np.log(safe), 0.0)


def _read_csv(text: str) -> NDArray[np.float64]:
    """Parse a two-column CSV with '#' comments and one header row."""
    rows = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    return np.array([[float(v) for v in row.split(",")] for row in rows[1:]], dtype=np.float64)
