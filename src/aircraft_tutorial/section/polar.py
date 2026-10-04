# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Section polar: the per-angle results of one XFoil run and their CSV representation.

A polar keeps every swept point, converged or not, so that the convergence record is part of the
evidence. Values of a non-converged point are NaN.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

_FLOAT_COLUMNS: Final[tuple[str, ...]] = ("alpha_deg", "cl", "cd", "cm", "cp_min", "rms_bl")
_BOOL_COLUMNS: Final[tuple[str, ...]] = ("converged", "diverged")
_COLUMNS: Final[tuple[str, ...]] = ("alpha_deg", "cl", "cd", "cm", "cp_min", "converged",
                                    "diverged", "rms_bl")


class PolarError(ValueError):
    """Raised for an inconsistent polar or an unreadable polar file."""


@dataclass(frozen=True)
class RunInfo:
    """Condition and numerics of the run that produced a polar.

    Attributes:
        airfoil: Section name.
        reynolds: Chord Reynolds number [-].
        mach: Free-stream Mach number [-].
        ncrit: e^N transition criterion [-].
        panel_nodes: Panel nodes after repanelling [-].
    """

    airfoil: str
    reynolds: float
    mach: float
    ncrit: float
    panel_nodes: int


@dataclass(frozen=True)
class Polar:
    """Results of one angle-of-attack sweep, sorted by ascending angle of attack.

    Attributes:
        info: Run condition and numerics.
        alpha_deg: Angle of attack [deg].
        cl: Lift coefficient [-].
        cd: Drag coefficient [-].
        cm: Quarter-chord pitching-moment coefficient [-].
        cp_min: Minimum pressure coefficient on the surface [-].
        converged: True if the viscous solution converged.
        diverged: True if the boundary-layer Newton solve stopped on a NaN.
        rms_bl: Final RMS residual of the boundary-layer Newton system [-].
    """

    info: RunInfo
    alpha_deg: NDArray[np.float64]
    cl: NDArray[np.float64]
    cd: NDArray[np.float64]
    cm: NDArray[np.float64]
    cp_min: NDArray[np.float64]
    converged: NDArray[np.bool_]
    diverged: NDArray[np.bool_]
    rms_bl: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Check equal lengths and ascending angles; freeze the arrays."""
        n = len(self.alpha_deg)
        for name in _COLUMNS:
            dtype = np.bool_ if name in _BOOL_COLUMNS else np.float64
            arr: NDArray[np.generic] = np.array(getattr(self, name), dtype=dtype)
            if arr.shape != (n,):
                raise PolarError(f"{self.info.airfoil}: column {name} has shape {arr.shape}, "
                                 f"expected ({n},)")
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        if not bool(np.all(np.diff(self.alpha_deg) > 0.0)):
            raise PolarError(f"{self.info.airfoil}: alpha_deg must be strictly ascending")

    def converged_only(self) -> Polar:
        """Return the polar restricted to its converged points."""
        m = self.converged
        return Polar(self.info, self.alpha_deg[m], self.cl[m], self.cd[m], self.cm[m],
                     self.cp_min[m], self.converged[m], self.diverged[m], self.rms_bl[m])


def write_csv(polar: Polar, path: Path) -> None:
    """Write a polar as CSV with the run information in '#' header lines.

    Args:
        polar: Polar to write.
        path: Output path.
    """
    lines = [f"# {f.name} = {getattr(polar.info, f.name)}" for f in fields(RunInfo)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(lines) + "\n")
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(_COLUMNS)
        for i in range(len(polar.alpha_deg)):
            writer.writerow([repr(float(getattr(polar, c)[i])) if c in _FLOAT_COLUMNS
                             else str(bool(getattr(polar, c)[i])) for c in _COLUMNS])


def read_csv(path: Path) -> Polar:
    """Read a polar written by `write_csv`.

    Args:
        path: Polar CSV file.

    Returns:
        The polar.

    Raises:
        PolarError: If the header or a column is missing or malformed.
    """
    try:
        text = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PolarError(f"{path}: cannot read: {exc}") from exc
    meta: dict[str, str] = {}
    body: list[str] = []
    for line in text:
        if line.startswith("#"):
            key, sep, value = line[1:].partition("=")
            if not sep:
                raise PolarError(f"{path}: malformed header line {line!r}")
            meta[key.strip()] = value.strip()
        elif line.strip():
            body.append(line)
    try:
        info = RunInfo(airfoil=meta["airfoil"], reynolds=float(meta["reynolds"]),
                       mach=float(meta["mach"]), ncrit=float(meta["ncrit"]),
                       panel_nodes=int(meta["panel_nodes"]))
    except (KeyError, ValueError) as exc:
        raise PolarError(f"{path}: incomplete run header: {exc}") from exc

    rows = list(csv.DictReader(body))
    if not rows or tuple(rows[0].keys()) != _COLUMNS:
        raise PolarError(f"{path}: expected columns {_COLUMNS}")
    try:
        f = {c: np.array([float(r[c]) for r in rows], dtype=np.float64) for c in _FLOAT_COLUMNS}
    except ValueError as exc:
        raise PolarError(f"{path}: non-numeric value: {exc}") from exc
    b = {c: np.array([r[c] == "True" for r in rows], dtype=np.bool_) for c in _BOOL_COLUMNS}
    return Polar(info, f["alpha_deg"], f["cl"], f["cd"], f["cm"], f["cp_min"],
                 b["converged"], b["diverged"], f["rms_bl"])
