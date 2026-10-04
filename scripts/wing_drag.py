# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Task 1c: drag polars of the sized WING-1 and WING-2, with the trim point.

    CD = CDi (VSPAERO, Trefftz plane) + CD_profile (XFoil polar, strip by strip) + CD_wave (Korn)

Reads, for each wing of inputs/wing.toml:

- results/wing/<name>/polar.csv and span_loads.csv, written by scripts/wing_twist.py;
- the section polar CSV named by `polar` in the [[wing]] entry, at the sweep-normal cruise
  condition (M_n, Re_n of the MAC).

The reference chord of the strip Reynolds scaling is the streamwise chord whose sweep-normal
Reynolds number equals the polar's, c_ref = Re_polar / (Re_n per metre); it is the MAC when the
polar was run at Re_n of the MAC.

Writes drag_polar.csv, strip_drag.csv and drag_polar.png to results/wing/<name>/, and
drag_polars.png and drag_summary.md to results/wing/.

Needs no OpenVSP. Run from the repo root, after scripts.wing_twist: python -m scripts.wing_drag
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from aircraft_tutorial.config.wing import WingCase, WingEntry, load_wing_case
from aircraft_tutorial.contracts.section_polar import SectionPolarData, read_section_polar
from aircraft_tutorial.geometry.airfoil import max_thickness, normalized, read_dat
from aircraft_tutorial.plots.wing import DragPolar, plot_drag_build_up, plot_drag_polars
from aircraft_tutorial.wing import profile_drag as pd
from aircraft_tutorial.wing.cruise import CruisePoint, cruise_point
from aircraft_tutorial.wing.planform import TrapezoidalPlanform
from aircraft_tutorial.wing.vlm_results import read_polar_csv, read_strips_csv
from aircraft_tutorial.wing.wave_drag import korn_mdd, wave_drag

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
CASE_FILE: Final[Path] = ROOT / "inputs" / "wing.toml"
OUT_DIR: Final[Path] = ROOT / "results" / "wing"
MACH_TOLERANCE: Final[float] = 0.005
"""A polar Mach number further than this from M_n is reported as a mismatch [-]."""

_POLAR_COLUMNS: Final[tuple[str, ...]] = (
    "alpha_deg", "cl", "cdi", "cd_profile", "cd_profile_sens", "m_dd", "cd_wave", "cd",
    "cd_sens", "l_d", "n_stalled", "n_below_polar")
_STRIP_COLUMNS: Final[tuple[str, ...]] = (
    "alpha_deg", "y_m", "chord_m", "cl", "cl_n", "cd_n", "cd", "status")


@dataclass(frozen=True)
class WingDrag:
    """Drag polar of one wing and its trim point.

    Attributes:
        entry: The wing of the run definition.
        section: Section polar used.
        lookup: Usable part of the section polar.
        chord_ref_m: Reference chord of the strip Reynolds scaling [m].
        thickness_ratio: Section t/c [-].
        columns: drag_polar.csv columns, one entry per angle of attack.
        trim: Values at CL = CL_des, interpolated on the angles with a defined CD.
    """

    entry: WingEntry
    section: SectionPolarData
    lookup: pd.DragLookup
    chord_ref_m: float
    thickness_ratio: float
    columns: dict[str, NDArray[np.float64]]
    trim: dict[str, float]


def main() -> None:
    """Build and report the drag polar of every wing of the run definition."""
    case = load_wing_case(CASE_FILE)
    planform = TrapezoidalPlanform(case.planform.span_m, case.planform.area_m2,
                                   case.planform.root_chord_m,
                                   case.planform.sweep_quarter_chord_deg)
    cruise = cruise_point(case.cruise.mach, case.cruise.altitude_m, case.aircraft.design_mass_kg,
                          planform.area_m2, planform.mac_m, planform.sweep_quarter_chord_deg)

    results = [_analyse(case, entry, planform, cruise) for entry in case.wings]

    plot_drag_polars([_plot_data(r) for r in results], cruise.cl_design,
                     f"Drag polars of the sized wings (M = {cruise.mach:.3f})",
                     OUT_DIR / "drag_polars.png")
    (OUT_DIR / "drag_summary.md").write_text(_summary(case, planform, cruise, results),
                                             encoding="utf-8")
    print(f"Results in {OUT_DIR}")


def _analyse(case: WingCase, entry: WingEntry, planform: TrapezoidalPlanform,
             cruise: CruisePoint) -> WingDrag:
    """Drag build-up of one wing over its VSPAERO sweep."""
    out_dir = OUT_DIR / entry.name
    vlm = read_polar_csv(out_dir / "polar.csv")
    strips = read_strips_csv(out_dir / "span_loads.csv")
    section = read_section_polar(entry.polar)
    lookup = pd.usable_polar(section)
    chord_ref = section.reynolds / cruise.reynolds_normal(1.0)
    thickness = max_thickness(normalized(read_dat(entry.airfoil)))[0]
    sweep_korn = planform.sweep_deg(case.drag.korn_sweep_chord_fraction)
    sweep = planform.sweep_quarter_chord_deg
    if abs(section.mach - cruise.mach_normal) > MACH_TOLERANCE:
        print(f"WARNING {entry.name}: polar Mach {section.mach} differs from M_n = "
              f"{cruise.mach_normal:.4f}")
    print(f"{entry.name}: polar {entry.polar.name} ({section.airfoil}, Re = {section.reynolds:.4g}, "
          f"M = {section.mach}), c_ref = {chord_ref:.3f} m, usable cl_n "
          f"[{lookup.cl[0]:.3f}, {lookup.cl[-1]:.3f}], t/c = {thickness:.4f}")

    rows: dict[str, list[float]] = {c: [] for c in _POLAR_COLUMNS}
    strip_rows: list[tuple[float, ...]] = []
    for alpha, cl, cdi in zip(vlm.alpha_deg, vlm.cl, vlm.cdi, strict=True):
        s = strips.at_alpha(float(alpha))
        out, cd_prof = pd.strip_profile_drag(
            s, lookup, sweep_deg=sweep, chord_ref_m=chord_ref, area_m2=planform.area_m2,
            mode=case.drag.sweep_drag_mode, reynolds_exponent=case.drag.reynolds_exponent)
        _, cd_prof_sens = pd.strip_profile_drag(
            s, lookup, sweep_deg=sweep, chord_ref_m=chord_ref, area_m2=planform.area_m2,
            mode=case.drag.sensitivity_sweep_drag_mode,
            reynolds_exponent=case.drag.reynolds_exponent)
        m_dd = korn_mdd(entry.kappa_a, thickness, float(cl), sweep_korn)
        cd_wave = wave_drag(cruise.mach, m_dd)
        cd = float(cdi) + cd_prof + cd_wave
        for key, value in (("alpha_deg", alpha), ("cl", cl), ("cdi", cdi),
                           ("cd_profile", cd_prof), ("cd_profile_sens", cd_prof_sens),
                           ("m_dd", m_dd), ("cd_wave", cd_wave), ("cd", cd),
                           ("cd_sens", float(cdi) + cd_prof_sens + cd_wave),
                           ("l_d", float(cl) / cd),
                           ("n_stalled", int(np.sum(out.status == pd.STALLED))),
                           ("n_below_polar", int(np.sum(out.status == pd.BELOW_POLAR)))):
            rows[key].append(float(value))
        strip_rows.extend(zip(s.alpha_deg, s.y_m, s.chord_m, s.cl, out.cl_n, out.cd_n, out.cd,
                              out.status.astype(np.float64), strict=True))

    columns = {k: np.array(v, dtype=np.float64) for k, v in rows.items()}
    _write_csv(out_dir / "drag_polar.csv", _POLAR_COLUMNS,
               list(zip(*(columns[c] for c in _POLAR_COLUMNS), strict=True)))
    _write_csv(out_dir / "strip_drag.csv", _STRIP_COLUMNS, strip_rows)

    result = WingDrag(entry, section, lookup, chord_ref, thickness, columns,
                      _trim(columns, cruise.cl_design))
    plot_drag_build_up(_plot_data(result), cruise.cl_design,
                       f"{entry.name} ({entry.airfoil.stem}) drag polar, M = {cruise.mach:.3f}",
                       out_dir / "drag_polar.png")
    return result


def _trim(columns: dict[str, NDArray[np.float64]], cl_design: float) -> dict[str, float]:
    """Interpolate every column at CL = CL_des over the angles with a defined CD.

    NaN everywhere if CL_des lies outside that range.
    """
    ok = np.isfinite(columns["cd"])
    cl = columns["cl"][ok]
    if cl.size < 2 or not (cl.min() <= cl_design <= cl.max()):
        return {k: math.nan for k in columns}
    return {k: float(np.interp(cl_design, cl, v[ok])) for k, v in columns.items()}


def _plot_data(r: WingDrag) -> DragPolar:
    """Finite part of a wing's drag polar, for the figures."""
    ok = np.isfinite(r.columns["cd"])
    return DragPolar(label=r.entry.name, cl=r.columns["cl"][ok], cdi=r.columns["cdi"][ok],
                     cd_profile=r.columns["cd_profile"][ok], cd_wave=r.columns["cd_wave"][ok],
                     trim_cl=r.trim["cl"], trim_cd=r.trim["cd"])


def _write_csv(path: Path, header: tuple[str, ...], rows: list[tuple[float, ...]]) -> None:
    """Write float rows as CSV, full precision."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([repr(float(v)) for v in row])


def _summary(case: WingCase, planform: TrapezoidalPlanform, cruise: CruisePoint,
             results: list[WingDrag]) -> str:
    """Markdown tables of the inputs and the trim-point drag build-up."""
    sens = case.drag.sensitivity_sweep_drag_mode
    lines = [
        "# Task 1c: drag polars (VSPAERO + XFoil strips + Korn)",
        "",
        "Generated by `python -m scripts.wing_drag` from `inputs/wing.toml` and the results of "
        "`scripts.wing_twist`.",
        "",
        "CD = CDi (VSPAERO, Trefftz plane) + CD_profile (XFoil polar, strip by strip) + "
        "CD_wave (swept Korn + ADSEE drag rise). Isolated wing, no other components.",
        "",
        f"- Strips: cl_n = cl / cos²Λ_c/4 (Λ = {planform.sweep_quarter_chord_deg} deg), "
        f"cd = cd_n·k_Λ·(c / c_ref)^{case.drag.reynolds_exponent}, baseline k_Λ: "
        f"'{case.drag.sweep_drag_mode}', sensitivity: '{sens}'.",
        f"- Korn: sweep of the {case.drag.korn_sweep_chord_fraction:g}c line = "
        f"{planform.sweep_deg(case.drag.korn_sweep_chord_fraction):.2f} deg, wing CL.",
        f"- Cruise section condition: M_n = {cruise.mach_normal:.4f}, Re_n(MAC) = "
        f"{cruise.reynolds_normal_mac:.4e}, Cp* = {cruise.cp_crit_normal:.4f}.",
        "",
        "## Section polars used",
        "",
        "| Wing | File | Section | Re | M | Ncrit | c_ref [m] | usable cl_n | t/c | κ_A "
        "| points with cp_min < Cp* |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        s = r.section
        supersonic = int(np.sum(s.converged & (s.cp_min < cruise.cp_crit_normal)))
        lines.append(
            f"| {r.entry.name} | {r.entry.polar.name} | {s.airfoil} | {s.reynolds:.4e} "
            f"| {s.mach} | {s.ncrit} | {r.chord_ref_m:.3f} "
            f"| {r.lookup.cl[0]:.3f} … {r.lookup.cl[-1]:.3f} | {r.thickness_ratio:.4f} "
            f"| {r.entry.kappa_a} | {supersonic} of {int(np.sum(s.converged))} |")
    lines += [
        "",
        f"## Trim point (CL = CL_des = {cruise.cl_design:.4f})",
        "",
        f"| Wing | α_trim [deg] | CDi | CD_profile | CD_wave | M_dd | CD | L/D "
        f"| CD ({sens}) | L/D ({sens}) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        t = r.trim
        lines.append(
            f"| {r.entry.name} | {t['alpha_deg']:+.3f} | {t['cdi']:.5f} | {t['cd_profile']:.5f} "
            f"| {t['cd_wave']:.5f} | {t['m_dd']:.4f} | {t['cd']:.5f} "
            f"| {cruise.cl_design / t['cd']:.2f} | {t['cd_sens']:.5f} "
            f"| {cruise.cl_design / t['cd_sens']:.2f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
