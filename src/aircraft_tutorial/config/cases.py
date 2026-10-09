# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Load a section-analysis run definition from TOML into frozen dataclasses.

The TOML file is the single place where a run is defined: airfoils, flow condition, XFoil
settings, metric settings and the sensitivity cases. Every value is validated on load, and a bad
value raises `CaseError` naming the key and the value. Relative file paths resolve against the
directory of the TOML file.

Expected layout::

    [conditions]
    reynolds = 8.9e6
    mach = 0.15
    ncrit = 10.0

    [xfoil]
    panel_nodes = 160
    max_iter = 200
    alpha_low_deg = -8.0
    alpha_high_deg = 22.0
    alpha_step_deg = 0.25
    stop_after_failures = 8

    [metrics]
    linear_alpha_min_deg = -4.0
    linear_alpha_max_deg = 6.0
    low_drag_factor = 0.10
    spread_windows_deg = [[-2.0, 4.0], [-6.0, 8.0]]

    [sensitivity]
    panel_nodes = [100, 160, 240, 320]
    ncrit = [9.0, 10.0, 11.0]

    [[airfoil]]
    name = "NACA 2412"
    file = "../airfoils/NACA2412.dat"
    drag_bucket = false

    [[extra_run]]                  # optional: polars only, at another condition
    label = "3d_input"
    reynolds = 4.17e7
    mach = 0.1
    ncrit = 10.0
    sweep_deg = 37.5               # section cut normal to this sweep line; 0 = file as given
    output = "../results/3d_input"
    report_low_drag = false
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

MIN_PANEL_NODES: Final[int] = 40
MAX_PANEL_NODES: Final[int] = 600
"""Upper bound below the XFoil array size IQX = 640 of the vendored build (i_xfoil.f90)."""


class CaseError(ValueError):
    """Raised when a run definition is missing a key or holds an invalid value."""


@dataclass(frozen=True)
class FlowConditions:
    """Flow condition of the section analysis.

    Attributes:
        reynolds: Chord Reynolds number [-].
        mach: Free-stream Mach number [-].
        ncrit: e^N transition criterion [-].
    """

    reynolds: float
    mach: float
    ncrit: float


@dataclass(frozen=True)
class XfoilSettings:
    """Numerical settings of one XFoil polar.

    Attributes:
        panel_nodes: Number of panel nodes after repanelling [-].
        max_iter: Viscous Newton iteration limit per angle of attack [-].
        alpha_low_deg: Lowest angle of attack of the sweep [deg].
        alpha_high_deg: Highest angle of attack of the sweep [deg].
        alpha_step_deg: Angle-of-attack step [deg].
        stop_after_failures: Consecutive non-converged points that end a sweep leg [-].
    """

    panel_nodes: int
    max_iter: int
    alpha_low_deg: float
    alpha_high_deg: float
    alpha_step_deg: float
    stop_after_failures: int


@dataclass(frozen=True)
class MetricSettings:
    """Settings for the values extracted from a polar.

    Attributes:
        linear_alpha_min_deg: Lower bound of the linear lift-curve window [deg].
        linear_alpha_max_deg: Upper bound of the linear lift-curve window [deg].
        low_drag_factor: k in the low-drag range definition cd <= (1 + k)·cd_min [-].
        spread_windows_deg: Extra (min, max) windows [deg]; the spread of a₀ and α_0L over the
            main and extra windows is the fit uncertainty.
    """

    linear_alpha_min_deg: float
    linear_alpha_max_deg: float
    low_drag_factor: float
    spread_windows_deg: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class SensitivitySettings:
    """Cases of the numerical sensitivity check.

    Attributes:
        panel_nodes: Panel node counts to compare [-].
        ncrit: Transition criteria to compare [-].
    """

    panel_nodes: tuple[int, ...]
    ncrit: tuple[float, ...]


@dataclass(frozen=True)
class AirfoilEntry:
    """One airfoil of the run.

    Attributes:
        name: Display name, for example "NACA 2412".
        path: Absolute path of the coordinate file.
        drag_bucket: True if the section has a laminar low-drag range (6-series); the low-drag
            range is only reported for such sections.
    """

    name: str
    path: Path
    drag_bucket: bool


@dataclass(frozen=True)
class ExtraRun:
    """Polars of every airfoil at another flow condition, written to their own folder.

    Attributes:
        label: Short name of the run.
        conditions: Flow condition.
        sweep_deg: Sweep of the line the analysed section is cut normal to [deg]. The airfoil
            of the coordinate file is taken as the streamwise section; 0 analyses it as given
            (geometry.sweep).
        output: Absolute output folder.
        report_low_drag: False if no laminar low-drag range exists at this condition, even for
            a section that has one at the production condition.
    """

    label: str
    conditions: FlowConditions
    sweep_deg: float
    output: Path
    report_low_drag: bool


@dataclass(frozen=True)
class SectionCase:
    """Complete run definition of the section analysis."""

    conditions: FlowConditions
    xfoil: XfoilSettings
    metrics: MetricSettings
    sensitivity: SensitivitySettings
    airfoils: tuple[AirfoilEntry, ...]
    extra_runs: tuple[ExtraRun, ...]


def load_section_case(path: Path) -> SectionCase:
    """Read and validate a section-analysis TOML file.

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

    base = path.resolve().parent
    return SectionCase(
        conditions=_conditions(_table(raw, "conditions")),
        xfoil=_xfoil(_table(raw, "xfoil")),
        metrics=_metrics(_table(raw, "metrics")),
        sensitivity=_sensitivity(_table(raw, "sensitivity")),
        airfoils=_airfoils(raw, base),
        extra_runs=_extra_runs(raw, base),
    )


def _conditions(t: dict[str, object], prefix: str = "conditions") -> FlowConditions:
    """Validate a flow condition (reynolds, mach, ncrit)."""
    reynolds = _float(t, f"{prefix}.reynolds")
    mach = _float(t, f"{prefix}.mach")
    ncrit = _float(t, f"{prefix}.ncrit")
    _require(reynolds > 0.0, f"{prefix}.reynolds", reynolds, "must be > 0")
    _require(0.0 <= mach < 1.0, f"{prefix}.mach", mach, "must be in [0, 1)")
    _require(ncrit > 0.0, f"{prefix}.ncrit", ncrit, "must be > 0")
    return FlowConditions(reynolds=reynolds, mach=mach, ncrit=ncrit)


def _extra_runs(raw: dict[str, object], base: Path) -> tuple[ExtraRun, ...]:
    """Validate the optional [[extra_run]] array."""
    entries = raw.get("extra_run", [])
    if not isinstance(entries, list):
        raise CaseError(f"[[extra_run]]: must be an array of tables, got {entries!r}")
    runs: list[ExtraRun] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CaseError(f"extra_run[{i}]: must be a table, got {entry!r}")
        label, output = entry.get("label"), entry.get("output")
        if not isinstance(label, str) or not label.strip():
            raise CaseError(f"extra_run[{i}].label: must be a non-empty string, got {label!r}")
        if not isinstance(output, str) or not output.strip():
            raise CaseError(f"extra_run[{i}].output: must be a non-empty string, got {output!r}")
        low_drag = entry.get("report_low_drag")
        if not isinstance(low_drag, bool):
            raise CaseError(f"extra_run[{i}].report_low_drag: must be true or false, "
                            f"got {low_drag!r}")
        sweep = _float(entry, f"extra_run[{i}].sweep_deg")
        _require(0.0 <= sweep < 90.0, f"extra_run[{i}].sweep_deg", sweep, "must be in [0, 90)")
        runs.append(ExtraRun(label, _conditions(entry, f"extra_run[{i}]"), sweep,
                             (base / output).resolve(), low_drag))
    labels = [r.label for r in runs]
    if len(set(labels)) != len(labels):
        raise CaseError(f"[[extra_run]]: labels must be unique, got {labels}")
    return tuple(runs)


def _xfoil(t: dict[str, object]) -> XfoilSettings:
    """Validate the [xfoil] table."""
    nodes = _int(t, "xfoil.panel_nodes")
    max_iter = _int(t, "xfoil.max_iter")
    low = _float(t, "xfoil.alpha_low_deg")
    high = _float(t, "xfoil.alpha_high_deg")
    step = _float(t, "xfoil.alpha_step_deg")
    stop = _int(t, "xfoil.stop_after_failures")
    _require_nodes(nodes, "xfoil.panel_nodes")
    _require(max_iter > 0, "xfoil.max_iter", max_iter, "must be > 0")
    _require(low < 0.0 < high, "xfoil.alpha_low_deg/alpha_high_deg", (low, high),
             "the sweep starts at 0 deg, so low < 0 < high is required")
    _require(0.0 < step <= (high - low) / 2.0, "xfoil.alpha_step_deg", step,
             "must be > 0 and at most half the sweep range")
    _require(stop > 0, "xfoil.stop_after_failures", stop, "must be > 0")
    return XfoilSettings(nodes, max_iter, low, high, step, stop)


def _metrics(t: dict[str, object]) -> MetricSettings:
    """Validate the [metrics] table."""
    lo = _float(t, "metrics.linear_alpha_min_deg")
    hi = _float(t, "metrics.linear_alpha_max_deg")
    k = _float(t, "metrics.low_drag_factor")
    _require(lo < hi, "metrics.linear_alpha_min_deg/max_deg", (lo, hi), "min must be < max")
    _require(k > 0.0, "metrics.low_drag_factor", k, "must be > 0")
    windows: list[tuple[float, float]] = []
    for w in _list(t, "metrics.spread_windows_deg"):
        if not (isinstance(w, list) and len(w) == 2):
            raise CaseError(f"metrics.spread_windows_deg: each window must be [min, max], got {w!r}")
        w_lo = _float_item(w[0], "metrics.spread_windows_deg")
        w_hi = _float_item(w[1], "metrics.spread_windows_deg")
        _require(w_lo < w_hi, "metrics.spread_windows_deg", w, "min must be < max")
        windows.append((w_lo, w_hi))
    return MetricSettings(lo, hi, k, tuple(windows))


def _sensitivity(t: dict[str, object]) -> SensitivitySettings:
    """Validate the [sensitivity] table."""
    nodes = tuple(_int_item(v, "sensitivity.panel_nodes") for v in _list(t, "sensitivity.panel_nodes"))
    ncrit = tuple(_float_item(v, "sensitivity.ncrit") for v in _list(t, "sensitivity.ncrit"))
    for n in nodes:
        _require_nodes(n, "sensitivity.panel_nodes")
    for value in ncrit:
        _require(value > 0.0, "sensitivity.ncrit", value, "must be > 0")
    return SensitivitySettings(nodes, ncrit)


def _airfoils(raw: dict[str, object], base: Path) -> tuple[AirfoilEntry, ...]:
    """Validate the [[airfoil]] array; files must exist."""
    entries = raw.get("airfoil")
    if not isinstance(entries, list) or not entries:
        raise CaseError("[[airfoil]]: at least one airfoil entry is required")
    result: list[AirfoilEntry] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CaseError(f"airfoil[{i}]: must be a table, got {entry!r}")
        name = entry.get("name")
        file = entry.get("file")
        if not isinstance(name, str) or not name.strip():
            raise CaseError(f"airfoil[{i}].name: must be a non-empty string, got {name!r}")
        if not isinstance(file, str):
            raise CaseError(f"airfoil[{i}].file: must be a string, got {file!r}")
        resolved = (base / file).resolve()
        if not resolved.is_file():
            raise CaseError(f"airfoil[{i}].file: {resolved} does not exist")
        bucket = entry.get("drag_bucket")
        if not isinstance(bucket, bool):
            raise CaseError(f"airfoil[{i}].drag_bucket: must be true or false, got {bucket!r}")
        result.append(AirfoilEntry(name=name, path=resolved, drag_bucket=bucket))
    names = [a.name for a in result]
    if len(set(names)) != len(names):
        raise CaseError(f"[[airfoil]]: names must be unique, got {names}")
    return tuple(result)


def _table(raw: dict[str, object], key: str) -> dict[str, object]:
    """Return a required sub-table."""
    value = raw.get(key)
    if not isinstance(value, dict):
        raise CaseError(f"[{key}]: required table is missing")
    return value


def _list(t: dict[str, object], dotted: str) -> list[object]:
    """Return a required non-empty list."""
    value = t.get(dotted.rsplit(".", 1)[1])
    if not isinstance(value, list) or not value:
        raise CaseError(f"{dotted}: must be a non-empty list, got {value!r}")
    return value


def _float(t: dict[str, object], dotted: str) -> float:
    """Return a required finite number."""
    key = dotted.rsplit(".", 1)[1]
    if key not in t:
        raise CaseError(f"{dotted}: required key is missing")
    return _float_item(t[key], dotted)


def _float_item(value: object, dotted: str) -> float:
    """Convert one value to a finite float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CaseError(f"{dotted}: must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise CaseError(f"{dotted}: must be finite, got {result}")
    return result


def _int(t: dict[str, object], dotted: str) -> int:
    """Return a required integer."""
    key = dotted.rsplit(".", 1)[1]
    if key not in t:
        raise CaseError(f"{dotted}: required key is missing")
    return _int_item(t[key], dotted)


def _int_item(value: object, dotted: str) -> int:
    """Check that one value is an integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise CaseError(f"{dotted}: must be an integer, got {value!r}")
    return value


def _require_nodes(nodes: int, dotted: str) -> None:
    """Check a panel node count against the XFoil array bounds."""
    _require(MIN_PANEL_NODES <= nodes <= MAX_PANEL_NODES, dotted, nodes,
             f"must be in [{MIN_PANEL_NODES}, {MAX_PANEL_NODES}]")


def _require(condition: bool, dotted: str, value: object, reason: str) -> None:
    """Raise CaseError if the positive-form condition is false."""
    if not condition:
        raise CaseError(f"{dotted} = {value!r}: {reason}")
