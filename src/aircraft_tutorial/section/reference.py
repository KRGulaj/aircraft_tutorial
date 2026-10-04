# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Experimental reference values read from NACA TR 824, and the camber-increment estimate.

A reference file holds only values that are clearly readable on the report page, each with its
reading uncertainty. A value that cannot be read clearly is absent, not guessed.

Camber-increment estimate (Abbott & von Doenhoff, §6.8, §7.6): for 6-series sections with the
a = 1.0 mean line, camber and thickness effects are nearly independent. A section that has no
data (target) is estimated from a section with the same thickness (base) plus the change measured
between two sections of equal thickness that differ only in design lift coefficient (low → high):

    copied from base             a₀, cd_min, x_ac
    shifted by Δc_li             cl at cd_min, low-drag range edges
    base + (high − low)          α_0L, cl_max, α_stall, cm_c/4

Uncertainties: copied and shifted values keep the base uncertainty; for base + (high − low) the
three reading uncertainties add in quadrature.

Thin-airfoil check for the a = 1.0 mean line: Δα_0L = −Δc_li / (2π) [rad] and Δcm_c/4 = −Δc_li / 4.
Measured values are about 0.74 of the theoretical α_0L (Abbott p.129) and about three-quarters of
the theoretical cm_c/4 (Abbott §7.6, p.180).
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Final

ABBOTT_ALPHA0_FACTOR_A10: Final[float] = 0.74
"""Measured / thin-airfoil α_0L for 6-series sections with a = 1.0 (Abbott p.129)."""
ABBOTT_CM_FACTOR_A10: Final[float] = 0.75
"""Measured / thin-airfoil cm_c/4 for 6-series sections with a = 1.0 (Abbott §7.6, p.180)."""

_COPIED: Final[tuple[str, ...]] = ("a0_per_deg", "cd_min", "x_ac")
_SHIFTED: Final[tuple[str, ...]] = ("cl_at_cd_min", "low_drag_lower", "low_drag_upper")
_INCREMENTED: Final[tuple[str, ...]] = ("alpha_0l_deg", "cl_max", "alpha_stall_deg", "cm_c4")


class ReferenceDataError(ValueError):
    """Raised for an invalid reference file or an impossible estimate."""


@dataclass(frozen=True)
class Reading:
    """One value read from a report page.

    Attributes:
        value: Read value [unit of the quantity].
        uncertainty: Reading uncertainty, half-width [unit of the quantity].
        note: How the value was read.
    """

    value: float
    uncertainty: float
    note: str


@dataclass(frozen=True)
class Readings:
    """The quantities that can be compared with XFoil. None = not clearly readable.

    Attributes:
        a0_per_deg: Lift-curve slope [1/deg].
        alpha_0l_deg: Zero-lift angle [deg].
        cl_max: Maximum lift coefficient [-].
        alpha_stall_deg: Angle of attack at cl_max [deg].
        cd_min: Minimum drag coefficient [-].
        cl_at_cd_min: Lift coefficient at cd_min [-].
        low_drag_lower: Lower edge of the low-drag range [-].
        low_drag_upper: Upper edge of the low-drag range [-].
        cm_c4: Quarter-chord moment coefficient [-].
        x_ac: Aerodynamic centre [chord fraction].
    """

    a0_per_deg: Reading | None = None
    alpha_0l_deg: Reading | None = None
    cl_max: Reading | None = None
    alpha_stall_deg: Reading | None = None
    cd_min: Reading | None = None
    cl_at_cd_min: Reading | None = None
    low_drag_lower: Reading | None = None
    low_drag_upper: Reading | None = None
    cm_c4: Reading | None = None
    x_ac: Reading | None = None


@dataclass(frozen=True)
class ReferenceData:
    """Reference values of one airfoil.

    Attributes:
        airfoil: Section name.
        source: Report and page, or the estimate description.
        reynolds: Reynolds number of the read curve [-].
        design_cl: Design lift coefficient c_li [-]; NaN if not a 6-series section.
        checked: True after the group has checked every value against the page.
        estimated: True if the values are an estimate, not a measurement.
        readings: The values.
    """

    airfoil: str
    source: str
    reynolds: float
    design_cl: float
    checked: bool
    estimated: bool
    readings: Readings


def load_reference(path: Path) -> ReferenceData:
    """Read a reference TOML file.

    Layout::

        airfoil = "NACA 2412"
        source = "NACA TR 824, p.136"
        reynolds = 8.9e6
        design_cl = nan          # only 6-series: the design lift coefficient
        checked = false

        [values.cl_max]
        value = 1.68
        uncertainty = 0.03
        note = "highest diamond symbol"

    Args:
        path: Reference file.

    Returns:
        The reference data (estimated = False).

    Raises:
        ReferenceDataError: If a key is missing, unknown or invalid.
    """
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ReferenceDataError(f"{path}: cannot read: {exc}") from exc
    try:
        airfoil, source = str(raw["airfoil"]), str(raw["source"])
        reynolds, design_cl = float(raw["reynolds"]), float(raw["design_cl"])
        checked = raw["checked"]
        values = raw["values"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ReferenceDataError(f"{path}: missing or invalid header key: {exc}") from exc
    if not isinstance(checked, bool):
        raise ReferenceDataError(f"{path}: checked must be true or false, got {checked!r}")
    if not (reynolds > 0.0):
        raise ReferenceDataError(f"{path}: reynolds must be > 0, got {reynolds}")
    if not isinstance(values, dict):
        raise ReferenceDataError(f"{path}: [values] must be a table")

    known = {f.name for f in fields(Readings)}
    unknown = set(values) - known
    if unknown:
        raise ReferenceDataError(f"{path}: unknown quantities {sorted(unknown)}")
    readings: dict[str, Reading] = {}
    for name, entry in values.items():
        readings[name] = _reading(path, name, entry)
    return ReferenceData(airfoil, source, reynolds, design_cl, checked, False,
                         Readings(**readings))


def estimate_from_camber_increment(target: str, base: ReferenceData, low: ReferenceData,
                                   high: ReferenceData, target_design_cl: float
                                   ) -> ReferenceData:
    """Estimate a section without data from a base section and a camber increment.

    Args:
        target: Name of the estimated section.
        base: Section with the target thickness and a different design c_l.
        low: Calibration section with the lower design c_l.
        high: Calibration section with the higher design c_l, same thickness as `low`.
        target_design_cl: Design c_l of the target [-].

    Returns:
        The estimate (estimated = True). A quantity missing in a needed input is None.

    Raises:
        ReferenceDataError: If the calibration increment in c_li differs from the target increment.
    """
    d_cli = target_design_cl - base.design_cl
    d_cal = high.design_cl - low.design_cl
    if not (math.isclose(d_cli, d_cal, abs_tol=1e-9)):
        raise ReferenceDataError(f"camber step of the calibration pair ({d_cal}) differs from the "
                             f"step base → target ({d_cli})")
    tag = f"estimated: {base.airfoil} + ({high.airfoil} − {low.airfoil})"
    out: dict[str, Reading | None] = {}
    for name in _COPIED:
        b = getattr(base.readings, name)
        out[name] = None if b is None else replace(b, note=f"{tag}; copied from base")
    for name in _SHIFTED:
        b = getattr(base.readings, name)
        out[name] = None if b is None else Reading(
            b.value + d_cli, b.uncertainty, f"{tag}; base shifted by Δc_li = {d_cli:+.2f}")
    for name in _INCREMENTED:
        b, lo, hi = (getattr(r.readings, name) for r in (base, low, high))
        out[name] = None if (b is None or lo is None or hi is None) else Reading(
            b.value + (hi.value - lo.value),
            math.sqrt(b.uncertainty**2 + lo.uncertainty**2 + hi.uncertainty**2),
            f"{tag}; base + measured camber increment {hi.value - lo.value:+.4g}")
    return ReferenceData(
        airfoil=target,
        source=f"{base.source}; increment {low.source} → {high.source}",
        reynolds=base.reynolds,
        design_cl=target_design_cl,
        checked=base.checked and low.checked and high.checked,
        estimated=True,
        readings=Readings(**out),
    )


def thin_airfoil_a10_increment(delta_cli: float) -> tuple[float, float]:
    """Thin-airfoil change of α_0L and cm_c/4 for the a = 1.0 mean line.

    Args:
        delta_cli: Change of design lift coefficient [-].

    Returns:
        (Δα_0L [deg], Δcm_c/4 [-]).
    """
    return math.degrees(-delta_cli / (2.0 * math.pi)), -delta_cli / 4.0


def _reading(path: Path, name: str, entry: object) -> Reading:
    """Validate one [values.<name>] table."""
    if not isinstance(entry, dict):
        raise ReferenceDataError(f"{path}: values.{name} must be a table")
    try:
        value, unc, note = float(entry["value"]), float(entry["uncertainty"]), str(entry["note"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReferenceDataError(f"{path}: values.{name} needs value, uncertainty, note: {exc}") \
            from exc
    if not (math.isfinite(value) and unc > 0.0):
        raise ReferenceDataError(f"{path}: values.{name}: value must be finite and uncertainty > 0, "
                             f"got {value}, {unc}")
    return Reading(value, unc, note)
