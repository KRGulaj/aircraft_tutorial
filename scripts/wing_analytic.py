# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Task 1c, comparison with VSPAERO: DATCOM lift curve and twist sizing of the lecture.

For each wing of inputs/wing.toml:

1. CL_α from the DATCOM formula of the lecture (η from inputs/wing.toml, M and Λ_0.5c of the
   cruise condition and the trapezoid);
2. the twist that gives CL_des at zero body angle, with the chord-weighted mean twist and the
   streamwise section α_0l of the 2D stage (`alpha_0l_streamwise_deg` in
   results/3d_input/<section>_metrics.csv; wing/analytic.py). The 2D stage analyses the section
   normal to the sweep line; its `alpha_0l_deg` is the normal-plane angle, and the method here
   uses streamwise angles;
3. the lift curve over the VSPAERO angle range, drawn with the VSPAERO lift curve of
   scripts/wing_twist.py and both trim points.

Writes analytic_lift.csv and lift_curve_comparison.png to results/wing/<name>/, and
analytic_summary.md and lift_curves_all.png (every wing, both methods) to results/wing/.

Needs no OpenVSP. Run from the repo root, after scripts.wing_twist: python -m scripts.wing_analytic
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Final

from aircraft_tutorial.common.fitting import fit_line
from aircraft_tutorial.config.wing import WingCase, load_wing_case
from aircraft_tutorial.contracts.section_metrics import read_section_metrics
from aircraft_tutorial.plots.wing import LiftCurve, plot_lift_curves
from aircraft_tutorial.wing.analytic import (
    AnalyticWing,
    datcom_lift_slope,
    size_twist,
    twist_effectiveness,
)
from aircraft_tutorial.wing.cruise import CruisePoint, cruise_point
from aircraft_tutorial.wing.planform import TrapezoidalPlanform
from aircraft_tutorial.wing.vlm_results import VlmPolar, read_polar_csv, read_twist_history

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
CASE_FILE: Final[Path] = ROOT / "inputs" / "wing.toml"
OUT_DIR: Final[Path] = ROOT / "results" / "wing"


def main() -> None:
    """Size every wing analytically and compare it with VSPAERO."""
    case = load_wing_case(CASE_FILE)
    planform = TrapezoidalPlanform(case.planform.span_m, case.planform.area_m2,
                                   case.planform.root_chord_m,
                                   case.planform.sweep_quarter_chord_deg)
    cruise = cruise_point(case.cruise.mach, case.cruise.altitude_m, case.aircraft.design_mass_kg,
                          planform.area_m2, planform.mac_m, planform.sweep_quarter_chord_deg)
    rows: list[str] = []
    all_curves: list[LiftCurve] = []
    for entry in case.wings:
        out_dir = OUT_DIR / entry.name
        vlm = read_polar_csv(out_dir / "polar.csv")
        history = read_twist_history(out_dir / "twist_sizing.csv")
        alpha_0l = read_section_metrics(entry.metrics).value("alpha_0l_streamwise_deg")
        slope = datcom_lift_slope(planform.aspect_ratio, cruise.mach, planform.sweep_deg(0.5),
                                  case.analytic.eta)
        wing = size_twist(cruise.cl_design, slope, taper=planform.taper,
                          root_incidence_deg=case.twist.root_incidence_deg,
                          section_alpha_0l_deg=alpha_0l,
                          max_twist_deg=case.twist.max_twist_deg)
        print(f"{entry.name}: CL_alpha = {slope:.4f} /rad, twist = {wing.twist_deg:+.3f} deg "
              f"(VSPAERO {history[-1][0]:+.3f} deg)")
        _write_lift(vlm, wing, out_dir / "analytic_lift.csv")
        curves = [LiftCurve(f"{entry.name} VSPAERO, ε = {history[-1][0]:+.2f}°", vlm.alpha_deg,
                            vlm.cl, 0.0, cruise.cl_design),
                  LiftCurve(f"{entry.name} DATCOM, ε = {wing.twist_deg:+.2f}°", vlm.alpha_deg,
                            wing.cl(vlm.alpha_deg), 0.0, cruise.cl_design)]
        all_curves += curves
        plot_lift_curves(curves, cruise.cl_design,
                         f"{entry.name} ({entry.airfoil.stem}): VSPAERO and DATCOM\n"
                         f"{cruise.flow_label}",
                         out_dir / "lift_curve_comparison.png")
        rows.append(_row(entry.name, alpha_0l, wing, vlm, history))
    plot_lift_curves(all_curves, cruise.cl_design,
                     f"Sized wings, VSPAERO and DATCOM\n{cruise.flow_label}",
                     OUT_DIR / "lift_curves_all.png")
    (OUT_DIR / "analytic_summary.md").write_text(_summary(case, planform, cruise, rows),
                                                 encoding="utf-8")
    print(f"Results in {OUT_DIR}")


def _write_lift(vlm: VlmPolar, wing: AnalyticWing, path: Path) -> None:
    """Analytical and VSPAERO CL over the VSPAERO angles."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("alpha_deg", "cl_datcom", "cl_vspaero"))
        for alpha, cl_a, cl_v in zip(vlm.alpha_deg, wing.cl(vlm.alpha_deg), vlm.cl, strict=True):
            writer.writerow((repr(float(alpha)), repr(float(cl_a)), repr(float(cl_v))))


def _row(name: str, alpha_0l_deg: float, wing: AnalyticWing, vlm: VlmPolar,
         history: tuple[tuple[float, float], ...]) -> str:
    """One comparison row: inputs, DATCOM values, VSPAERO values."""
    fit = fit_line(vlm.alpha_deg, vlm.cl)
    vlm_slope = math.degrees(fit.slope)
    (t0, c0), (t1, c1) = history[0], history[1]
    vlm_k = (c0 - c1) / (t0 - t1) / fit.slope
    twist_vlm = history[-1][0]
    return (f"| {name} | {alpha_0l_deg:+.2f} "
            f"| {wing.cl_alpha_per_rad:.3f} | {vlm_slope:.3f} "
            f"| {100.0 * (wing.cl_alpha_per_rad / vlm_slope - 1.0):+.1f}% "
            f"| {wing.twist_effectiveness:.3f} | {vlm_k:.3f} "
            f"| {wing.alpha_zero_lift_deg:+.2f} | {fit.x_intercept:+.2f} "
            f"| **{wing.twist_deg:+.2f}** | **{twist_vlm:+.2f}** |")


def _summary(case: WingCase, planform: TrapezoidalPlanform, cruise: CruisePoint,
             rows: list[str]) -> str:
    """Markdown comparison of the analytical method with VSPAERO."""
    beta = math.sqrt(1.0 - cruise.mach**2)
    return "\n".join([
        "# Task 1c: analytical lift curve and twist (DATCOM) against VSPAERO",
        "",
        "Generated by `python -m scripts.wing_analytic` from `inputs/wing.toml` and the results "
        "of `scripts.wing_twist`.",
        "",
        "- CL_α = 2π·A / (2 + √(4 + (A·β/η)²·[1 + tan²Λ_0.5c / β²])) (ADSEE-2 lecture, "
        "slide 20), "
        f"A = {planform.aspect_ratio:.3f}, M = {cruise.mach}, β = {beta:.4f}, "
        f"Λ_0.5c = {planform.sweep_deg(0.5):.2f} deg, η = {case.analytic.eta}.",
        "- CL = CL_α·(α + i_r + k_ε·ε − α_0l), k_ε = (1 + 2λ) / (3(1 + λ)) = "
        f"{twist_effectiveness(planform.taper):.4f} (chord-weighted mean of the linear twist), "
        f"i_r = {case.twist.root_incidence_deg:+.1f} deg.",
        f"- Twist for CL = CL_des = {cruise.cl_design:.4f} at α = 0: "
        "ε = (CL_des / CL_α − i_r + α_0l) / k_ε.",
        "- VSPAERO k_ε = (∂CL/∂ε) / CL_α from the first two twist-sizing runs (ε = 0 and −ε_max).",
        "",
        "| Wing | α_0l streamwise, XFoil [deg] | CL_α DATCOM [1/rad] | CL_α VSPAERO [1/rad] | Δ "
        "| k_ε DATCOM | k_ε VSPAERO | α_0L DATCOM [deg] | α_0L VSPAERO [deg] "
        "| ε DATCOM [deg] | ε VSPAERO [deg] |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        *rows,
        "",
    ])


if __name__ == "__main__":
    main()
