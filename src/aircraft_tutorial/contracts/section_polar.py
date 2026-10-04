# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Section polar hand-off from the 2D stage to the 3D stage: the polar CSV file written by
`aircraft_tutorial.section.polar.write_csv`, read without importing the section stage.

File layout: '#' header lines `# key = value` for airfoil, reynolds, mach, ncrit and
panel_nodes, then a CSV table with the columns
alpha_deg, cl, cd, cm, cp_min, converged, diverged, rms_bl. Values of non-converged points are
NaN. tests/contracts/test_section_polar.py writes a file with the section stage's writer and
reads it here, so the two ends cannot drift apart.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

COLUMNS: Final[tuple[str, ...]] = ("alpha_deg", "cl", "cd", "cm", "cp_min", "converged",
                                   "diverged", "rms_bl")
_FLOAT_COLUMNS: Final[tuple[str, ...]] = ("alpha_deg", "cl", "cd", "cm", "cp_min")


class SectionPolarError(ValueError):
    """Raised for an unreadable or inconsistent polar file."""


@dataclass(frozen=True)
class SectionPolarData:
    """A section polar as read from file, ascending in angle of attack.

    Attributes:
        airfoil: Section name.
        reynolds: Chord Reynolds number of the run [-].
        mach: Mach number of the run [-].
        ncrit: e^N transition criterion of the run [-].
        alpha_deg: Angle of attack [deg].
        cl: Lift coefficient [-].
        cd: Drag coefficient [-].
        cm: Quarter-chord pitching-moment coefficient [-].
        cp_min: Minimum surface pressure coefficient [-].
        converged: True if the viscous solution converged.
    """

    airfoil: str
    reynolds: float
    mach: float
    ncrit: float
    alpha_deg: NDArray[np.float64]
    cl: NDArray[np.float64]
    cd: NDArray[np.float64]
    cm: NDArray[np.float64]
    cp_min: NDArray[np.float64]
    converged: NDArray[np.bool_]


def read_section_polar(path: Path) -> SectionPolarData:
    """Read a section polar CSV file.

    Args:
        path: Polar CSV file.

    Returns:
        The polar.

    Raises:
        SectionPolarError: If the file is missing, the header is incomplete, a column is missing
            or malformed, or the angles are not strictly ascending.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SectionPolarError(f"{path}: cannot read: {exc}") from exc
    meta: dict[str, str] = {}
    body: list[str] = []
    for line in lines:
        if line.startswith("#"):
            key, sep, value = line[1:].partition("=")
            if not sep:
                raise SectionPolarError(f"{path}: malformed header line {line!r}")
            meta[key.strip()] = value.strip()
        elif line.strip():
            body.append(line)
    try:
        airfoil = meta["airfoil"]
        reynolds, mach, ncrit = (float(meta[k]) for k in ("reynolds", "mach", "ncrit"))
    except (KeyError, ValueError) as exc:
        raise SectionPolarError(f"{path}: incomplete run header: {exc}") from exc

    rows = list(csv.DictReader(body))
    if not rows or tuple(rows[0].keys()) != COLUMNS:
        raise SectionPolarError(f"{path}: expected columns {COLUMNS}")
    try:
        f = {c: np.array([float(r[c]) for r in rows], dtype=np.float64) for c in _FLOAT_COLUMNS}
    except ValueError as exc:
        raise SectionPolarError(f"{path}: non-numeric value: {exc}") from exc
    converged = np.array([r["converged"] == "True" for r in rows], dtype=np.bool_)
    if not bool(np.all(np.diff(f["alpha_deg"]) > 0.0)):
        raise SectionPolarError(f"{path}: alpha_deg must be strictly ascending")
    return SectionPolarData(airfoil, reynolds, mach, ncrit, f["alpha_deg"], f["cl"], f["cd"],
                            f["cm"], f["cp_min"], converged)
