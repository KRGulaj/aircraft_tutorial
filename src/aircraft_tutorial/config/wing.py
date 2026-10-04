# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Load the wing-analysis run definition (task 1c) from TOML into frozen dataclasses.

The TOML file holds the aircraft data, the trapezoidal planform, the cruise condition, the
twist-sizing settings, the VLM settings, the drag settings and the wings to analyse. Every value is validated on
load, and a bad value raises `CaseError` naming the key and the value. Relative file paths
resolve against the directory of the TOML file.

Expected layout::

    [aircraft]
    design_mass_kg = 380000.0

    [planform]
    span_m = 68.40
    area_m2 = 554.0
    root_chord_m = 14.63
    sweep_quarter_chord_deg = 37.5
    dihedral_deg = 0.0

    [cruise]
    mach = 0.855
    altitude_m = 10668.0

    [twist]
    root_incidence_deg = 4.0
    max_twist_deg = 6.0
    cl_tolerance = 1.0e-4
    max_iter = 6

    [vlm]
    segments = 12
    segment_tess = 3
    chord_tess = 33
    wake_iter = 5
    n_cpu = 4
    alpha_start_deg = -4.0
    alpha_end_deg = 10.0
    alpha_npts = 15

    [drag]
    korn_sweep_chord_fraction = 0.5
    sweep_drag_mode = "friction"
    sensitivity_sweep_drag_mode = "cos3"

    [[wing]]
    name = "WING-1"
    airfoil = "../airfoils/NACA2412.dat"
    polar = "../results/section/cruise/NACA2412.csv"
    kappa_a = 0.87

The section polar file is not required to exist when the file is loaded: the twist sizing does
not need it, and the drag step checks it when it reads it.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from aircraft_tutorial.common.atmosphere import H_MAX
from aircraft_tutorial.config.cases import CaseError
from aircraft_tutorial.config.fields import integer, number, require, string, table

SWEEP_DRAG_MODES: Final[tuple[str, ...]] = ("friction", "cos3")
"""Ways to take a sweep-normal section cd to the streamwise one (wing.profile_drag)."""

_WING_NAME: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]+$")
"""Wing names become result directory names."""
_GRID_TOLERANCE: Final[float] = 1e-9


@dataclass(frozen=True)
class AircraftData:
    """Aircraft-level values.

    Attributes:
        design_mass_kg: Design (cruise) mass [kg].
    """

    design_mass_kg: float


@dataclass(frozen=True)
class PlanformInput:
    """Defining values of the trapezoidal planform.

    Attributes:
        span_m: Span, tip to tip [m].
        area_m2: Reference area, both halves [m^2].
        root_chord_m: Root chord [m].
        sweep_quarter_chord_deg: Quarter-chord sweep [deg].
        dihedral_deg: Dihedral [deg].
    """

    span_m: float
    area_m2: float
    root_chord_m: float
    sweep_quarter_chord_deg: float
    dihedral_deg: float


@dataclass(frozen=True)
class CruiseInput:
    """Cruise flight condition.

    Attributes:
        mach: Free-stream Mach number [-].
        altitude_m: Geopotential altitude [m].
    """

    mach: float
    altitude_m: float


@dataclass(frozen=True)
class TwistSettings:
    """Twist-sizing settings.

    Attributes:
        root_incidence_deg: Root section incidence relative to the body axis [deg].
        max_twist_deg: Largest allowed |tip twist relative to the root| [deg].
        cl_tolerance: Convergence limit on |CL − CL_des| [-].
        max_iter: Secant steps after the two starting runs [-].
    """

    root_incidence_deg: float
    max_twist_deg: float
    cl_tolerance: float
    max_iter: int


@dataclass(frozen=True)
class VlmSettings:
    """VSPAERO vortex-lattice settings.

    Attributes:
        segments: Spanwise segments of the half-wing; the twist is linear across their
            boundaries [-].
        segment_tess: Spanwise sections per segment, ends included (OpenVSP SectTess_U);
            the half-wing has segments·(segment_tess − 1) strips [-].
        chord_tess: Chordwise points (OpenVSP Tess_W) [-].
        wake_iter: Wake relaxation iterations [-].
        n_cpu: Solver threads [-].
        alpha_start_deg: First body angle of attack of the sweep [deg].
        alpha_end_deg: Last body angle of attack of the sweep [deg].
        alpha_npts: Number of equally spaced angles [-].
    """

    segments: int
    segment_tess: int
    chord_tess: int
    wake_iter: int
    n_cpu: int
    alpha_start_deg: float
    alpha_end_deg: float
    alpha_npts: int


@dataclass(frozen=True)
class DragSettings:
    """Drag build-up settings.

    Attributes:
        korn_sweep_chord_fraction: Chord line whose sweep enters the Korn equation [-].
        sweep_drag_mode: Baseline sweep drag mode, "friction" or "cos3".
        sensitivity_sweep_drag_mode: Mode reported as the sensitivity.
    """

    korn_sweep_chord_fraction: float
    sweep_drag_mode: str
    sensitivity_sweep_drag_mode: str


@dataclass(frozen=True)
class WingEntry:
    """One wing variant.

    Attributes:
        name: Wing name, for example "WING-1"; also the result directory name.
        airfoil: Absolute path of the section coordinate file, used root to tip.
        polar: Absolute path of the section polar CSV at the cruise section condition.
        kappa_a: Korn technology factor κ_A of the section [-].
    """

    name: str
    airfoil: Path
    polar: Path
    kappa_a: float


@dataclass(frozen=True)
class WingCase:
    """Complete run definition of the wing analysis."""

    aircraft: AircraftData
    planform: PlanformInput
    cruise: CruiseInput
    twist: TwistSettings
    vlm: VlmSettings
    drag: DragSettings
    wings: tuple[WingEntry, ...]


def load_wing_case(path: Path) -> WingCase:
    """Read and validate a wing-analysis TOML file.

    Args:
        path: Path of the TOML file.

    Returns:
        The validated, immutable run definition.

    Raises:
        CaseError: If the file cannot be parsed, a key is missing, or a value is invalid.
    """
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise CaseError(f"{path}: cannot read run definition: {exc}") from exc

    return WingCase(
        aircraft=_aircraft(table(raw, "aircraft")),
        planform=_planform(table(raw, "planform")),
        cruise=_cruise(table(raw, "cruise")),
        twist=_twist(table(raw, "twist")),
        vlm=_vlm(table(raw, "vlm")),
        drag=_drag(table(raw, "drag")),
        wings=_wings(raw, path.resolve().parent),
    )


def _aircraft(t: dict[str, object]) -> AircraftData:
    """Validate the [aircraft] table."""
    mass = number(t, "aircraft.design_mass_kg")
    require(mass > 0.0, "aircraft.design_mass_kg", mass, "must be > 0")
    return AircraftData(design_mass_kg=mass)


def _planform(t: dict[str, object]) -> PlanformInput:
    """Validate the [planform] table. The taper ratio is checked by the planform itself."""
    span = number(t, "planform.span_m")
    area = number(t, "planform.area_m2")
    root = number(t, "planform.root_chord_m")
    sweep = number(t, "planform.sweep_quarter_chord_deg")
    dihedral = number(t, "planform.dihedral_deg")
    require(span > 0.0, "planform.span_m", span, "must be > 0")
    require(area > 0.0, "planform.area_m2", area, "must be > 0")
    require(root > 0.0, "planform.root_chord_m", root, "must be > 0")
    require(0.0 <= sweep < 90.0, "planform.sweep_quarter_chord_deg", sweep, "must be in [0, 90)")
    require(-45.0 < dihedral < 45.0, "planform.dihedral_deg", dihedral, "must be in (-45, 45)")
    return PlanformInput(span, area, root, sweep, dihedral)


def _cruise(t: dict[str, object]) -> CruiseInput:
    """Validate the [cruise] table."""
    mach = number(t, "cruise.mach")
    altitude = number(t, "cruise.altitude_m")
    require(0.0 < mach < 1.0, "cruise.mach", mach, "must be in (0, 1)")
    require(0.0 <= altitude <= H_MAX, "cruise.altitude_m", altitude, f"must be in [0, {H_MAX}]")
    return CruiseInput(mach=mach, altitude_m=altitude)


def _twist(t: dict[str, object]) -> TwistSettings:
    """Validate the [twist] table."""
    incidence = number(t, "twist.root_incidence_deg")
    max_twist = number(t, "twist.max_twist_deg")
    tol = number(t, "twist.cl_tolerance")
    max_iter = integer(t, "twist.max_iter")
    require(-15.0 < incidence < 15.0, "twist.root_incidence_deg", incidence, "must be in (-15, 15)")
    require(0.0 < max_twist < 20.0, "twist.max_twist_deg", max_twist, "must be in (0, 20)")
    require(tol > 0.0, "twist.cl_tolerance", tol, "must be > 0")
    require(max_iter > 0, "twist.max_iter", max_iter, "must be > 0")
    return TwistSettings(incidence, max_twist, tol, max_iter)


def _vlm(t: dict[str, object]) -> VlmSettings:
    """Validate the [vlm] table. The angle grid must contain 0 deg, the trim angle."""
    segments = integer(t, "vlm.segments")
    segment_tess = integer(t, "vlm.segment_tess")
    chord_tess = integer(t, "vlm.chord_tess")
    wake = integer(t, "vlm.wake_iter")
    cpus = integer(t, "vlm.n_cpu")
    start = number(t, "vlm.alpha_start_deg")
    end = number(t, "vlm.alpha_end_deg")
    npts = integer(t, "vlm.alpha_npts")
    require(1 <= segments <= 50, "vlm.segments", segments, "must be in [1, 50]")
    require(segment_tess >= 2, "vlm.segment_tess", segment_tess, "must be >= 2")
    require(chord_tess >= 3, "vlm.chord_tess", chord_tess, "must be >= 3")
    require(wake > 0, "vlm.wake_iter", wake, "must be > 0")
    require(cpus > 0, "vlm.n_cpu", cpus, "must be > 0")
    require(start < 0.0 < end, "vlm.alpha_start_deg/alpha_end_deg", (start, end),
            "the trim point is at 0 deg, so start < 0 < end is required")
    require(npts >= 3, "vlm.alpha_npts", npts, "must be >= 3")
    k = -start / ((end - start) / (npts - 1))
    require(abs(k - round(k)) < _GRID_TOLERANCE, "vlm.alpha_start_deg/alpha_end_deg/alpha_npts",
            (start, end, npts), "0 deg must be a grid point")
    return VlmSettings(segments, segment_tess, chord_tess, wake, cpus, start, end, npts)


def _drag(t: dict[str, object]) -> DragSettings:
    """Validate the [drag] table."""
    fraction = number(t, "drag.korn_sweep_chord_fraction")
    mode = string(t, "drag.sweep_drag_mode")
    sensitivity = string(t, "drag.sensitivity_sweep_drag_mode")
    require(0.0 <= fraction <= 1.0, "drag.korn_sweep_chord_fraction", fraction,
            "must be in [0, 1]")
    for key, value in (("drag.sweep_drag_mode", mode),
                       ("drag.sensitivity_sweep_drag_mode", sensitivity)):
        require(value in SWEEP_DRAG_MODES, key, value, f"must be one of {SWEEP_DRAG_MODES}")
    return DragSettings(fraction, mode, sensitivity)


def _wings(raw: dict[str, object], base: Path) -> tuple[WingEntry, ...]:
    """Validate the [[wing]] array; airfoil files must exist, polar files are checked on use."""
    entries = raw.get("wing")
    if not isinstance(entries, list) or not entries:
        raise CaseError("[[wing]]: at least one wing entry is required")
    result: list[WingEntry] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CaseError(f"wing[{i}]: must be a table, got {entry!r}")
        name = string(entry, f"wing[{i}].name")
        require(_WING_NAME.match(name) is not None, f"wing[{i}].name", name,
                "only letters, digits, '-' and '_' (it becomes a directory name)")
        resolved = (base / string(entry, f"wing[{i}].airfoil")).resolve()
        if not resolved.is_file():
            raise CaseError(f"wing[{i}].airfoil: {resolved} does not exist")
        polar = (base / string(entry, f"wing[{i}].polar")).resolve()
        kappa_a = number(entry, f"wing[{i}].kappa_a")
        require(0.7 <= kappa_a <= 1.0, f"wing[{i}].kappa_a", kappa_a, "must be in [0.7, 1.0]")
        result.append(WingEntry(name=name, airfoil=resolved, polar=polar, kappa_a=kappa_a))
    names = [w.name for w in result]
    if len(set(names)) != len(names):
        raise CaseError(f"[[wing]]: names must be unique, got {names}")
    return tuple(result)
