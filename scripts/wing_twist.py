# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Task 1c: size the linear twist of WING-1 and WING-2 in VSPAERO and draw their lift curves.

For each wing of inputs/wing.toml:

1. Find the twist that gives CL = CL_des at zero body angle with the fixed root incidence
   (secant method, one VSPAERO point per evaluation).
2. Run the full angle-of-attack sweep of the sized wing.
3. Write polar.csv (CL, CDi, CMy), span_loads.csv, twist_sizing.csv and lift_curve.png to
   results/wing/<name>/; the VSPAERO run files go to results/wing/<name>/vspaero_run/.

Then write results/wing/twist_summary.md and results/wing/lift_curves.png for both wings.

Needs the OpenVSP Python API. Run from the repo root: python -m scripts.wing_twist
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from aircraft_tutorial.common.fitting import fit_line
from aircraft_tutorial.config.wing import WingCase, WingEntry, load_wing_case
from aircraft_tutorial.geometry.airfoil import read_dat
from aircraft_tutorial.plots.wing import LiftCurve, plot_lift_curves
from aircraft_tutorial.wing import vlm_results, vsp_model, vspaero
from aircraft_tutorial.wing.cruise import CruisePoint, cruise_point
from aircraft_tutorial.wing.planform import TrapezoidalPlanform
from aircraft_tutorial.wing.twist import TwistSizing, size_twist

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
CASE_FILE: Final[Path] = ROOT / "inputs" / "wing.toml"
OUT_DIR: Final[Path] = ROOT / "results" / "wing"
RUN_SUBDIR: Final[str] = "vspaero_run"
TWIST_SAMPLES: Final[int] = 401
"""Surface samples for the check of the twist distribution against the linear one."""


@dataclass(frozen=True)
class WingResult:
    """Everything reported for one wing.

    Attributes:
        entry: The wing of the run definition.
        sizing: Twist sizing result.
        polar: Full sweep of the sized wing.
        cl_alpha_per_rad: Lift-curve slope, least squares over the sweep [1/rad].
        cl_alpha_r_squared: R² of that fit [-].
        alpha_zero_lift_deg: Body angle of zero lift from the fit [deg].
        trim_alpha_deg: Body angle where CL = CL_des on the sweep [deg].
        trim_cdi: Induced drag coefficient at the trim point [-].
        trim_cmy: Pitching-moment coefficient at the trim point, about c/4 of the MAC [-].
        span_efficiency: e = CL_des² / (π·AR·CDi) at the trim point [-].
        twist_deviation_deg: Largest |section angle − linear twist| on the OpenVSP surface [deg].
    """

    entry: WingEntry
    sizing: TwistSizing
    polar: vlm_results.VlmPolar
    cl_alpha_per_rad: float
    cl_alpha_r_squared: float
    alpha_zero_lift_deg: float
    trim_alpha_deg: float
    trim_cdi: float
    trim_cmy: float
    span_efficiency: float
    twist_deviation_deg: float


def _wing_result(entry: WingEntry, sizing: TwistSizing, polar: vlm_results.VlmPolar,
                 cl_design: float, aspect_ratio: float, twist_deviation_deg: float) -> WingResult:
    """Lift-curve fit and trim-point values of one sized wing.

    The VLM lift curve is linear, so the fit runs over the whole sweep; R² shows that it is.
    """
    fit = fit_line(np.radians(polar.alpha_deg), polar.cl)
    trim_cdi = float(np.interp(cl_design, polar.cl, polar.cdi))
    return WingResult(
        entry=entry, sizing=sizing, polar=polar,
        cl_alpha_per_rad=fit.slope, cl_alpha_r_squared=fit.r_squared,
        alpha_zero_lift_deg=math.degrees(fit.x_intercept),
        trim_alpha_deg=float(np.interp(cl_design, polar.cl, polar.alpha_deg)),
        trim_cdi=trim_cdi,
        trim_cmy=float(np.interp(cl_design, polar.cl, polar.cmy)),
        span_efficiency=cl_design**2 / (math.pi * aspect_ratio * trim_cdi),
        twist_deviation_deg=twist_deviation_deg,
    )


def main() -> None:
    """Size, sweep and report every wing of the run definition."""
    case = load_wing_case(CASE_FILE)
    planform = TrapezoidalPlanform(case.planform.span_m, case.planform.area_m2,
                                   case.planform.root_chord_m,
                                   case.planform.sweep_quarter_chord_deg)
    cruise = cruise_point(case.cruise.mach, case.cruise.altitude_m, case.aircraft.design_mass_kg,
                          planform.area_m2, planform.mac_m, planform.sweep_quarter_chord_deg)
    reference = vspaero.Reference(area_m2=planform.area_m2, span_m=planform.span_m,
                                  chord_m=planform.mac_m,
                                  x_m=planform.x_le_mac_m + 0.25 * planform.mac_m, z_m=0.0)
    flow = vspaero.FlowInput(mach=cruise.mach, velocity_m_s=cruise.velocity_m_s,
                             density_kg_m3=cruise.density_kg_m3, reynolds=cruise.reynolds_mac)
    print(f"taper = {planform.taper:.4f}, AR = {planform.aspect_ratio:.3f}, "
          f"MAC = {planform.mac_m:.3f} m, CL_des = {cruise.cl_design:.4f}")
    print(f"XFoil cruise polar condition: M_n = {cruise.mach_normal:.4f}, "
          f"Re_n(MAC) = {cruise.reynolds_normal_mac:.4e}, Cp* = {cruise.cp_crit_normal:.4f}")

    results = [_analyse(case, entry, planform, reference, flow, cruise) for entry in case.wings]

    plot_lift_curves([_curve(r, cruise.cl_design) for r in results], cruise.cl_design,
                     "Lift curves of the sized wings (VSPAERO, M = "
                     f"{cruise.mach:.3f})", OUT_DIR / "lift_curves.png")
    (OUT_DIR / "twist_summary.md").write_text(_summary(case, planform, cruise, results),
                                              encoding="utf-8")
    print(f"Results in {OUT_DIR}")


def _analyse(case: WingCase, entry: WingEntry, planform: TrapezoidalPlanform,
             reference: vspaero.Reference, flow: vspaero.FlowInput,
             cruise: CruisePoint) -> WingResult:
    """Size the twist of one wing, then run and write its full sweep."""
    out_dir = OUT_DIR / entry.name
    run_dir = out_dir / RUN_SUBDIR
    run_dir.mkdir(parents=True, exist_ok=True)
    airfoil = read_dat(entry.airfoil)

    def build(twist_deg: float) -> str:
        wid = vsp_model.new_wing_model(
            entry.name, planform, airfoil, dihedral_deg=case.planform.dihedral_deg,
            root_incidence_deg=case.twist.root_incidence_deg, twist_deg=twist_deg, vlm=case.vlm)
        area, span = vsp_model.total_area_and_span(wid)
        if not (math.isclose(area, planform.area_m2, rel_tol=1e-4)
                and math.isclose(span, planform.span_m, rel_tol=1e-6)):
            raise RuntimeError(f"OpenVSP wing has S = {area}, b = {span}; expected "
                               f"{planform.area_m2}, {planform.span_m}")
        return wid

    def cl_at_twist(twist_deg: float) -> float:
        build(twist_deg)
        polar, _ = vspaero.run_sweep(run_dir / "wing.vsp3", reference, flow, case.vlm,
                                     alpha_start_deg=0.0, alpha_end_deg=0.0, alpha_npts=1)
        print(f"  {entry.name}: twist {twist_deg:+.4f} deg -> CL(alpha=0) = {polar.cl[0]:.5f}")
        return float(polar.cl[0])

    print(f"{entry.name} ({entry.airfoil.stem}): sizing the twist")
    sizing = size_twist(cl_at_twist, cruise.cl_design, max_twist_deg=case.twist.max_twist_deg,
                        cl_tolerance=case.twist.cl_tolerance, max_iter=case.twist.max_iter)

    wid = build(sizing.twist_deg)
    y, angle = vsp_model.section_angles(wid, TWIST_SAMPLES)
    linear = case.twist.root_incidence_deg + sizing.twist_deg * y / planform.semi_span_m
    twist_deviation = float(np.max(np.abs(angle - linear)))
    polar, strips = vspaero.run_sweep(run_dir / "wing.vsp3", reference, flow, case.vlm,
                                      alpha_start_deg=case.vlm.alpha_start_deg,
                                      alpha_end_deg=case.vlm.alpha_end_deg,
                                      alpha_npts=case.vlm.alpha_npts)
    vlm_results.write_polar_csv(polar, out_dir / "polar.csv")
    vlm_results.write_strips_csv(strips, out_dir / "span_loads.csv")
    _write_history(sizing, out_dir / "twist_sizing.csv")

    result = _wing_result(entry, sizing, polar, cruise.cl_design, planform.aspect_ratio,
                          twist_deviation)
    plot_lift_curves([_curve(result, cruise.cl_design)], cruise.cl_design,
                     f"{entry.name} ({entry.airfoil.stem}), twist {sizing.twist_deg:+.2f}°",
                     out_dir / "lift_curve.png")
    return result


def _curve(result: WingResult, cl_design: float) -> LiftCurve:
    """Lift curve of one wing with its trim point."""
    return LiftCurve(label=result.entry.name, alpha_deg=result.polar.alpha_deg,
                     cl=result.polar.cl, trim_alpha_deg=result.trim_alpha_deg, trim_cl=cl_design)


def _write_history(sizing: TwistSizing, path: Path) -> None:
    """Write the evaluated (twist, CL) pairs of the sizing."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("evaluation", "twist_deg", "cl_alpha0"))
        for i, (twist, cl) in enumerate(sizing.history):
            writer.writerow((i, repr(twist), repr(cl)))


def _summary(case: WingCase, planform: TrapezoidalPlanform, cruise: CruisePoint,
             results: list[WingResult]) -> str:
    """Markdown table of the planform, the cruise point and the sized wings."""
    lines = [
        "# Task 1c: twist sizing (VSPAERO)",
        "",
        "Generated by `python -m scripts.wing_twist` from `inputs/wing.toml`.",
        "",
        "## Planform and cruise point",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| taper ratio λ | {planform.taper:.4f} |",
        f"| tip chord | {planform.tip_chord_m:.3f} m |",
        f"| aspect ratio | {planform.aspect_ratio:.3f} |",
        f"| MAC (y_MAC) | {planform.mac_m:.3f} m ({planform.y_mac_m:.3f} m) |",
        f"| LE sweep | {planform.sweep_deg(0.0):.2f} deg |",
        f"| T, ρ, μ | {cruise.temperature_k:.2f} K, {cruise.density_kg_m3:.4f} kg/m³, "
        f"{cruise.dynamic_viscosity_pa_s:.4e} Pa·s |",
        f"| V, q | {cruise.velocity_m_s:.2f} m/s, {cruise.dynamic_pressure_pa:.0f} Pa |",
        f"| CL_des = m·g0 / (q·S) | {cruise.cl_design:.4f} |",
        f"| Re_MAC | {cruise.reynolds_mac:.3e} |",
        f"| root incidence | {case.twist.root_incidence_deg:+.1f} deg |",
        "",
        "## Section condition for the XFoil cruise polars",
        "",
        f"Simple sweep theory with Λ_c/4 = {cruise.sweep_deg:.1f} deg: M_n = M·cos Λ, "
        "Re_n = ρ·V·c/μ·cos²Λ, cl_n = cl/cos²Λ. The polar is run at the MAC; strips are "
        "scaled to their own chord.",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| ρ·V/μ | {cruise.reynolds_per_m:.4e} 1/m |",
        f"| M_n | {cruise.mach_normal:.4f} |",
        f"| **Re_n of the MAC ({planform.mac_m:.3f} m)** | **{cruise.reynolds_normal_mac:.4e}** |",
        f"| Re_n root ({planform.root_chord_m:.3f} m) / tip ({planform.tip_chord_m:.3f} m) "
        f"| {cruise.reynolds_normal(planform.root_chord_m):.3e} / "
        f"{cruise.reynolds_normal(planform.tip_chord_m):.3e} |",
        f"| cl_n of CL_des | {cruise.cl_design_normal:.4f} |",
        f"| Cp* at M_n (points with cp_min below it are supersonic) | {cruise.cp_crit_normal:.4f} |",
        "",
        "## Sized wings",
        "",
        "Twist: tip relative to root, negative = wash-out. Trim point: CL = CL_des on the sweep.",
        "",
        "| Wing | Section | Twist [deg] | CL(α=0) | VSPAERO runs | CL_α [1/rad] (R²) "
        "| α_0L [deg] | α_trim [deg] | CDi trim | e | CMy trim | max |Δε| vs linear [deg] |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.entry.name} | {r.entry.airfoil.name} | {r.sizing.twist_deg:+.3f} "
            f"| {r.sizing.cl:.5f} | {len(r.sizing.history)} "
            f"| {r.cl_alpha_per_rad:.3f} ({r.cl_alpha_r_squared:.5f}) "
            f"| {r.alpha_zero_lift_deg:+.3f} | {r.trim_alpha_deg:+.4f} | {r.trim_cdi:.5f} "
            f"| {r.span_efficiency:.3f} | {r.trim_cmy:+.4f} | {r.twist_deviation_deg:.3f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
