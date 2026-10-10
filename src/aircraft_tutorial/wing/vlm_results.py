# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Results of a VSPAERO sweep and their CSV files.

Pure Python, so the drag post-processing reads what scripts/wing_twist.py wrote without the
OpenVSP module.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class VlmResultsError(ValueError):
    """Raised for an unreadable or malformed results file."""


@dataclass(frozen=True)
class VlmPolar:
    """Whole-wing coefficients, one entry per angle of attack, ascending.

    Attributes:
        alpha_deg: Body angle of attack [deg].
        cl: Lift coefficient [-].
        cdi: Induced drag coefficient, far-field [-].
        cmy: Pitching-moment coefficient about the reference point [-].
    """

    alpha_deg: NDArray[np.float64]
    cl: NDArray[np.float64]
    cdi: NDArray[np.float64]
    cmy: NDArray[np.float64]


@dataclass(frozen=True)
class StripLoads:
    """Spanwise strip loads of the right half-wing, one row per (angle, strip), sorted by angle
    and then by y.

    Attributes:
        alpha_deg: Body angle of attack [deg].
        y_m: Strip centre station [m].
        chord_m: Strip chord [m].
        area_m2: Strip area [m^2].
        cl: Local lift coefficient, streamwise, on the local chord [-].
    """

    alpha_deg: NDArray[np.float64]
    y_m: NDArray[np.float64]
    chord_m: NDArray[np.float64]
    area_m2: NDArray[np.float64]
    cl: NDArray[np.float64]

    def at_alpha(self, alpha_deg: float) -> StripLoads:
        """The strips of one angle of attack.

        Raises:
            VlmResultsError: If no strip belongs to that angle.
        """
        m = np.isclose(self.alpha_deg, alpha_deg)
        if not bool(np.any(m)):
            raise VlmResultsError(f"no strips at alpha = {alpha_deg} deg")
        return StripLoads(self.alpha_deg[m], self.y_m[m], self.chord_m[m], self.area_m2[m],
                          self.cl[m])


def write_polar_csv(polar: VlmPolar, path: Path) -> None:
    """Write the whole-wing polar as CSV, one column per field."""
    _write(polar, path)


def write_strips_csv(strips: StripLoads, path: Path) -> None:
    """Write the strip loads as CSV, one column per field."""
    _write(strips, path)


def read_polar_csv(path: Path) -> VlmPolar:
    """Read a polar written by `write_polar_csv`.

    Raises:
        VlmResultsError: If the file is missing or the columns do not match.
    """
    c = _read(path, tuple(f.name for f in fields(VlmPolar)))
    return VlmPolar(c["alpha_deg"], c["cl"], c["cdi"], c["cmy"])


def read_strips_csv(path: Path) -> StripLoads:
    """Read strip loads written by `write_strips_csv`.

    Raises:
        VlmResultsError: If the file is missing or the columns do not match.
    """
    c = _read(path, tuple(f.name for f in fields(StripLoads)))
    return StripLoads(c["alpha_deg"], c["y_m"], c["chord_m"], c["area_m2"], c["cl"])


def write_twist_history(history: tuple[tuple[float, float], ...], path: Path) -> None:
    """Write the evaluated (twist, CL at zero body angle) pairs of a twist sizing."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("evaluation", "twist_deg", "cl_alpha0"))
        for i, (twist, cl) in enumerate(history):
            writer.writerow((i, repr(twist), repr(cl)))


def read_twist_history(path: Path) -> tuple[tuple[float, float], ...]:
    """Read the (twist [deg], CL at zero body angle) pairs written by `write_twist_history`;
    the last pair is the sized twist.

    Raises:
        VlmResultsError: If the file is missing or the columns do not match.
    """
    c = _read(path, ("evaluation", "twist_deg", "cl_alpha0"))
    return tuple(zip(c["twist_deg"].tolist(), c["cl_alpha0"].tolist(), strict=True))


def _write(record: VlmPolar | StripLoads, path: Path) -> None:
    """Write the array fields of a record as CSV columns, full precision."""
    names = [f.name for f in fields(record)]
    columns = [getattr(record, n) for n in names]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(names)
        for row in zip(*columns, strict=True):
            writer.writerow([repr(float(v)) for v in row])


def _read(path: Path, names: tuple[str, ...]) -> dict[str, NDArray[np.float64]]:
    """Read CSV float columns with exactly the given header."""
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
    except OSError as exc:
        raise VlmResultsError(f"{path}: cannot read: {exc}") from exc
    if not rows or tuple(rows[0]) != names:
        raise VlmResultsError(f"{path}: expected columns {names}")
    try:
        data = np.array([[float(v) for v in r] for r in rows[1:]], dtype=np.float64)
    except ValueError as exc:
        raise VlmResultsError(f"{path}: non-numeric value: {exc}") from exc
    data = data.reshape(-1, len(names))
    return {n: data[:, i].copy() for i, n in enumerate(names)}
