# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Airfoil coordinates: the `Airfoil` container, .dat file I/O and shape properties.

Coordinates are in chord fractions, in Selig order: trailing edge → upper surface → leading
edge → lower surface → trailing edge. This is the single closed contour that XFoil's panel
method needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

MIN_POINTS: Final[int] = 10
_THICKNESS_STATIONS: Final[int] = 2001


class AirfoilError(ValueError):
    """Raised for invalid coordinates or an unreadable coordinate file."""


@dataclass(frozen=True)
class Airfoil:
    """Closed airfoil contour in Selig order.

    Attributes:
        name: Section name, for example "NACA 2412".
        x: Chordwise coordinates [chord fraction], read-only.
        y: Normal coordinates [chord fraction], read-only.
    """

    name: str
    x: NDArray[np.float64]
    y: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate the contour and freeze the arrays."""
        x = np.array(self.x, dtype=np.float64)
        y = np.array(self.y, dtype=np.float64)
        if x.ndim != 1 or x.shape != y.shape:
            raise AirfoilError(f"{self.name}: x and y must be 1-D with equal shape, "
                               f"got {x.shape} and {y.shape}")
        if not (x.size >= MIN_POINTS):
            raise AirfoilError(f"{self.name}: need at least {MIN_POINTS} points, got {x.size}")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise AirfoilError(f"{self.name}: coordinates must be finite")
        i_le = int(np.argmin(x))
        if not (0 < i_le < x.size - 1):
            raise AirfoilError(f"{self.name}: leading edge (min x) must be an interior point, "
                               f"found at index {i_le} of {x.size}")
        if not (x[0] > 0.9 and x[-1] > 0.9):
            raise AirfoilError(f"{self.name}: contour must start and end at the trailing edge, "
                               f"got x = {x[0]} and {x[-1]}")
        if not (np.mean(y[:i_le]) > np.mean(y[i_le + 1 :])):
            raise AirfoilError(f"{self.name}: first branch must be the upper surface (Selig order)")
        x.setflags(write=False)
        y.setflags(write=False)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

    @property
    def leading_edge_index(self) -> int:
        """Index of the leading-edge point (minimum x)."""
        return int(np.argmin(self.x))

    def surfaces(self) -> tuple[NDArray[np.float64], NDArray[np.float64],
                                NDArray[np.float64], NDArray[np.float64]]:
        """Split the contour into its two surfaces, each ordered leading edge → trailing edge.

        Returns:
            (x_upper, y_upper, x_lower, y_lower). Both surfaces include the leading-edge point.
        """
        i = self.leading_edge_index
        return self.x[: i + 1][::-1], self.y[: i + 1][::-1], self.x[i:], self.y[i:]


def from_surfaces(name: str, x_upper: NDArray[np.float64], y_upper: NDArray[np.float64],
                  x_lower: NDArray[np.float64], y_lower: NDArray[np.float64]) -> Airfoil:
    """Assemble an Airfoil from two surfaces that both run leading edge → trailing edge.

    The two surfaces must share their first point (the leading edge); it is kept once.

    Args:
        name: Section name.
        x_upper: Upper-surface x, leading edge first [chord fraction].
        y_upper: Upper-surface y [chord fraction].
        x_lower: Lower-surface x, leading edge first [chord fraction].
        y_lower: Lower-surface y [chord fraction].

    Returns:
        The closed contour in Selig order.

    Raises:
        AirfoilError: If the surfaces do not share the leading-edge point.
    """
    if not (x_upper[0] == x_lower[0] and y_upper[0] == y_lower[0]):
        raise AirfoilError(f"{name}: surfaces must share the leading-edge point, got "
                           f"({x_upper[0]}, {y_upper[0]}) and ({x_lower[0]}, {y_lower[0]})")
    x = np.concatenate([x_upper[::-1], x_lower[1:]])
    y = np.concatenate([y_upper[::-1], y_lower[1:]])
    return Airfoil(name=name, x=x, y=y)


def read_dat(path: Path) -> Airfoil:
    """Read a coordinate file in Selig or Lednicer format.

    - Selig: one closed contour, trailing edge → upper → leading edge → lower → trailing edge.
    - Lednicer: two blank-line-separated blocks, each one surface leading edge → trailing edge.
      An optional first line with the point counts is skipped. The blocks are merged into one
      Selig contour.

    The first line is the name if it is not a coordinate pair. Exact consecutive duplicate points
    are removed, because a zero-length panel breaks XFoil's spline.

    Args:
        path: Path of the .dat file.

    Returns:
        The airfoil, named after the header line or the file stem.

    Raises:
        AirfoilError: If the file cannot be read or does not hold a valid contour.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AirfoilError(f"{path}: cannot read: {exc}") from exc
    if not lines:
        raise AirfoilError(f"{path}: file is empty")

    name = path.stem
    if not _is_pair(lines[0]):
        name = lines[0].strip() or path.stem
        lines = lines[1:]

    blocks = _blocks(lines, path)
    if len(blocks) == 3 and len(blocks[0]) == 1:
        blocks = blocks[1:]  # Lednicer point-count line
    if len(blocks) == 1:
        pts = blocks[0]
    elif len(blocks) == 2:
        upper, lower = blocks
        pts = upper[::-1] + lower[1:]
    else:
        raise AirfoilError(f"{path}: found {len(blocks)} coordinate blocks, expected 1 (Selig) "
                           "or 2 (Lednicer)")

    xy = np.array(pts, dtype=np.float64)
    keep = np.ones(len(xy), dtype=bool)
    keep[1:] = np.any(np.diff(xy, axis=0) != 0.0, axis=1)
    xy = xy[keep]
    try:
        return Airfoil(name=name, x=xy[:, 0], y=xy[:, 1])
    except AirfoilError as exc:
        raise AirfoilError(f"{path}: {exc}") from exc


def write_dat(airfoil: Airfoil, path: Path, title: str) -> None:
    """Write a Selig .dat file that XFoil can load directly.

    The .dat format has no comment syntax, so the source of the coordinates goes into the
    single title line.

    Args:
        airfoil: Contour to write.
        path: Output path.
        title: First line of the file (name and source).

    Raises:
        AirfoilError: If the title holds a line break.
    """
    if "\n" in title or "\r" in title:
        raise AirfoilError(f"title must be one line, got {title!r}")
    rows = "\n".join(f"{x: .8f} {y: .8f}" for x, y in zip(airfoil.x, airfoil.y, strict=True))
    path.write_text(f"{title}\n{rows}\n", encoding="utf-8")


def normalized(airfoil: Airfoil) -> Airfoil:
    """Translate and scale so that the leading edge is at x = 0 and the chord is 1.

    The leading edge is the minimum-x point. The trailing edge is the midpoint of the first and
    last contour points. The contour is not rotated; a chord line that is not horizontal is
    reported by the caller, not corrected here.

    Args:
        airfoil: Contour to normalise.

    Returns:
        The normalised contour.
    """
    i = airfoil.leading_edge_index
    x_le, y_le = airfoil.x[i], airfoil.y[i]
    x_te = 0.5 * (airfoil.x[0] + airfoil.x[-1])
    chord = x_te - x_le
    return Airfoil(name=airfoil.name, x=(airfoil.x - x_le) / chord, y=(airfoil.y - y_le) / chord)


def max_thickness(airfoil: Airfoil) -> tuple[float, float]:
    """Maximum thickness measured normal to the chord line, and its position.

    Upper minus lower surface at equal x (the definition of the tabulated NACA thickness for
    symmetric sections; for cambered sections it differs from the perpendicular thickness by a
    second-order term in the camber slope).

    Args:
        airfoil: Contour, chord normalised to 1.

    Returns:
        (t/c [-], x/c of the maximum [-]).
    """
    xu, yu, xl, yl = airfoil.surfaces()
    lo = max(xu.min(), xl.min())
    hi = min(xu.max(), xl.max())
    xs = np.linspace(lo, hi, _THICKNESS_STATIONS)
    t = np.interp(xs, xu, yu) - np.interp(xs, xl, yl)
    k = int(np.argmax(t))
    return float(t[k]), float(xs[k])


def cosine_stations(n_per_surface: int) -> NDArray[np.float64]:
    """Cosine-spaced chord stations from 0 to 1, dense at both edges.

    x_i = (1 − cos(π·i / (n − 1))) / 2, i = 0 … n − 1.

    Args:
        n_per_surface: Number of stations, leading and trailing edge included.

    Returns:
        Stations [chord fraction].

    Raises:
        AirfoilError: If fewer than three stations are requested.
    """
    if not (n_per_surface >= 3):
        raise AirfoilError(f"need at least 3 stations per surface, got {n_per_surface}")
    return 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, n_per_surface)))


def _is_pair(line: str) -> bool:
    """True if the line holds exactly two numbers."""
    parts = line.split()
    if len(parts) != 2:
        return False
    try:
        float(parts[0])
        float(parts[1])
    except ValueError:
        return False
    return True


def _blocks(lines: list[str], path: Path) -> list[list[tuple[float, float]]]:
    """Split coordinate lines into blank-line-separated blocks of (x, y) pairs."""
    blocks: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            if current:
                blocks.append(current)
                current = []
            continue
        if not _is_pair(line):
            raise AirfoilError(f"{path}: line {number}: expected 'x y', got {line!r}")
        a, b = line.split()
        current.append((float(a), float(b)))
    if current:
        blocks.append(current)
    return blocks
