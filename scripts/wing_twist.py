# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Task 1c: size the linear twist of WING-1 and WING-2 in VSPAERO and draw their lift curves.

For each wing of inputs/wing.toml:

1. Find the twist that gives CL = CL_des at zero body angle with the fixed root incidence
   (secant method, one VSPAERO point per evaluation).
2. Run the full angle-of-attack sweep of the untwisted wing, the reference that shows the effect
   of the twist; its zero-lift angle plus the root incidence is the section zero-lift angle that
   the vortex lattice uses.
3. Check the grid convergence: the sized wing at zero body angle on the mesh levels of
   [mesh_study].
4. Run the full angle-of-attack sweep of the sized wing.
5. Write polar.csv (CL, CDi, CMy), span_loads.csv, polar_untwisted.csv, mesh_study.csv,
   twist_sizing.csv and lift_curve.png to results/wing/<name>/. The VSPAERO run files go to
   results/wing/<name>/vspaero_run/; wing.vsp3 there is the sized wing, the untwisted and
   mesh-study runs use its subdirectories.

Then write results/wing/twist_summary.md and results/wing/lift_curves.png for both wings.

Needs the OpenVSP Python API. Run from the repo root: python -m scripts.wing_twist
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np

from aircraft_tutorial.common.fitting import fit_line
from aircraft_tutorial.config.wing import VlmSettings, WingCase, WingEntry, load_wing_case
from aircraft_tutorial.geometry.airfoil import read_dat
from aircraft_tutorial.plots.wing import (
    LiftCurve,
    flight_conditions,
    plot_lift_curves,
    section_label,
)
from aircraft_tutorial.wing import vlm_results, vsp_model, vspaero
from aircraft_tutorial.wing.cruise import CruisePoint, cruise_point
from aircraft_tutorial.wing.planform import TrapezoidalPlanform
from aircraft_tutorial.wing.span_efficiency import loading_span_efficiency
from aircraft_tutorial.wing.twist import TwistSizing, size_twist

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
CASE_FILE: Final[Path] = ROOT / "inputs" / "wing.toml"
OUT_DIR: Final[Path] = ROOT / "results" / "wing"
RUN_SUBDIR: Final[str] = "vspaero_run"
UNTWISTED_SUBDIR: Final[str] = "untwisted"
MESH_SUBDIR: Final[str] = "mesh_study"
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
        untwisted: Full sweep of the same wing without twist.
        untwisted_alpha_zero_lift_deg: Body angle of zero lift of the untwisted wing [deg].
        mesh: Grid-convergence check of the sized wing at zero body angle.
        span_efficiency_loading: e from the strip loading at zero body angle, the check of
            the far-field value (wing.span_efficiency) [-].
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
    untwisted: vlm_results.VlmPolar
    untwisted_alpha_zero_lift_deg: float
    mesh: vlm_results.MeshStudy
    span_efficiency_loading: float


def _wing_result(entry: WingEntry, sizing: TwistSizing, polar: vlm_results.VlmPolar,
                 untwisted: vlm_results.VlmPolar, mesh: vlm_results.MeshStudy,
                 cl_design: float, aspect_ratio: float, twist_deviation_deg: float,
                 span_efficiency_loading: float) -> WingResult:
    """Lift-curve fits and trim-point values of one sized wing.

    The VLM lift curve is linear, so the fit runs over the whole sweep; R² shows that it is.
    """
    fit = fit_line(np.radians(polar.alpha_deg), polar.cl)
    fit_untwisted = fit_line(np.radians(untwisted.alpha_deg), untwisted.cl)
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
        untwisted=untwisted,
        untwisted_alpha_zero_lift_deg=math.degrees(fit_untwisted.x_intercept),
        mesh=mesh,
        span_efficiency_loading=span_efficiency_loading,
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

    plot_lift_curves([c for r in results for c in _curves(r, cruise.cl_design)],
                     cruise.cl_design, "Lift curves of the sized clean wings (VSPAERO)",
                     _conditions(case, cruise) + [
                         r"Both wings are sized to $C_{L,des}$ at $\alpha$ = 0: the twisted "
                         "curves coincide"],
                     OUT_DIR / "lift_curves.png")
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

    def build(twist_deg: float, vlm: VlmSettings) -> str:
        wid = vsp_model.new_wing_model(
            entry.name, planform, airfoil, dihedral_deg=case.planform.dihedral_deg,
            root_incidence_deg=case.twist.root_incidence_deg, twist_deg=twist_deg, vlm=vlm)
        area, span = vsp_model.total_area_and_span(wid)
        if not (math.isclose(area, planform.area_m2, rel_tol=1e-4)
                and math.isclose(span, planform.span_m, rel_tol=1e-6)):
            raise vspaero.VspaeroError(f"OpenVSP wing has S = {area}, b = {span}; expected "
                                       f"{planform.area_m2}, {planform.span_m}")
        return wid

    def trim_point(twist_deg: float, vlm: VlmSettings, vsp3: Path
                   ) -> tuple[vlm_results.VlmPolar, vlm_results.StripLoads]:
        build(twist_deg, vlm)
        return vspaero.run_sweep(vsp3, reference, flow, vlm, alpha_start_deg=0.0,
                                 alpha_end_deg=0.0, alpha_npts=1)

    def cl_at_twist(twist_deg: float) -> float:
        cl = float(trim_point(twist_deg, case.vlm, run_dir / "wing.vsp3")[0].cl[0])
        print(f"  {entry.name}: twist {twist_deg:+.4f} deg -> CL(alpha=0) = {cl:.5f}")
        return cl

    def full_sweep(vsp3: Path) -> tuple[vlm_results.VlmPolar, vlm_results.StripLoads]:
        return vspaero.run_sweep(vsp3, reference, flow, case.vlm,
                                 alpha_start_deg=case.vlm.alpha_start_deg,
                                 alpha_end_deg=case.vlm.alpha_end_deg,
                                 alpha_npts=case.vlm.alpha_npts)

    print(f"{entry.name} ({entry.airfoil.stem}): sizing the twist")
    sizing = size_twist(cl_at_twist, cruise.cl_design, max_twist_deg=case.twist.max_twist_deg,
                        cl_tolerance=case.twist.cl_tolerance, max_iter=case.twist.max_iter)

    print(f"  {entry.name}: sweep of the untwisted wing")
    untwisted_dir = run_dir / UNTWISTED_SUBDIR
    untwisted_dir.mkdir(exist_ok=True)
    build(0.0, case.vlm)
    untwisted, _ = full_sweep(untwisted_dir / "wing.vsp3")

    mesh = _mesh_study(case, planform, sizing.twist_deg, run_dir / MESH_SUBDIR, trim_point)

    # The sized wing goes last, so vspaero_run/wing.vsp3 is the model of the reported sweep.
    wid = build(sizing.twist_deg, case.vlm)
    y, angle = vsp_model.section_angles(wid, TWIST_SAMPLES)
    linear = case.twist.root_incidence_deg + sizing.twist_deg * y / planform.semi_span_m
    twist_deviation = float(np.max(np.abs(angle - linear)))
    polar, strips = full_sweep(run_dir / "wing.vsp3")
    vlm_results.write_polar_csv(polar, out_dir / "polar.csv")
    vlm_results.write_strips_csv(strips, out_dir / "span_loads.csv")
    vlm_results.write_polar_csv(untwisted, out_dir / "polar_untwisted.csv")
    vlm_results.write_mesh_study_csv(mesh, out_dir / "mesh_study.csv")
    vlm_results.write_twist_history(sizing.history, out_dir / "twist_sizing.csv")

    e_loading = _loading_efficiency(strips.at_alpha(0.0), planform,
                                    case.drag.loading_fourier_terms)
    result = _wing_result(entry, sizing, polar, untwisted, mesh, cruise.cl_design,
                          planform.aspect_ratio, twist_deviation, e_loading)
    plot_lift_curves(_curves(result, cruise.cl_design), cruise.cl_design,
                     f"{entry.name} ({section_label(entry.airfoil.stem)}): lift curve of the "
                     "clean wing (VSPAERO)",
                     _conditions(case, cruise), out_dir / "lift_curve.png")
    return result


def _mesh_study(case: WingCase, planform: TrapezoidalPlanform, twist_deg: float,
                study_dir: Path,
                trim_point: Callable[[float, VlmSettings, Path],
                                     tuple[vlm_results.VlmPolar, vlm_results.StripLoads]]
                ) -> vlm_results.MeshStudy:
    """CL, far-field CDi and loading-based CDi of the sized wing at zero body angle on every
    mesh level of [mesh_study]."""
    levels = case.mesh_study
    rows: list[tuple[float, ...]] = []
    for segment_tess, chord_tess in zip(levels.segment_tess, levels.chord_tess, strict=True):
        vlm = replace(case.vlm, segment_tess=segment_tess, chord_tess=chord_tess)
        level_dir = study_dir / f"s{segment_tess}_c{chord_tess}"
        level_dir.mkdir(parents=True, exist_ok=True)
        polar, strips = trim_point(twist_deg, vlm, level_dir / "wing.vsp3")
        cl, cdi = float(polar.cl[0]), float(polar.cdi[0])
        e_load = _loading_efficiency(strips, planform, case.drag.loading_fourier_terms)
        rows.append((segment_tess, chord_tess, strips.y_m.size, cl, cdi, e_load,
                     cl**2 / (math.pi * planform.aspect_ratio * e_load)))
        print(f"  mesh {segment_tess}/{chord_tess}: CL = {cl:.5f}, CDi = {cdi:.6f}, "
              f"e from loading = {e_load:.4f}")
    cols = [np.array(c, dtype=np.float64) for c in zip(*rows, strict=True)]
    return vlm_results.MeshStudy(*cols)


def _loading_efficiency(strips: vlm_results.StripLoads, planform: TrapezoidalPlanform,
                        terms: int) -> float:
    """Span efficiency from the strip loading c·cl of one angle (wing.span_efficiency)."""
    return loading_span_efficiency(strips.y_m / planform.semi_span_m,
                                   strips.cl * strips.chord_m, terms)


def _conditions(case: WingCase, cruise: CruisePoint) -> list[str]:
    """Condition-box lines of the VSPAERO lift-curve figures."""
    return ["VSPAERO (vortex lattice), isolated wing",
            flight_conditions(cruise.mach, cruise.reynolds_mac, cruise.altitude_m),
            rf"Root incidence $i_r$ = {case.twist.root_incidence_deg:+.1f}°, linear twist"]


def _curves(result: WingResult, cl_design: float) -> list[LiftCurve]:
    """Lift curve of one sized wing with its trim point, and of the same wing untwisted."""
    name = f"{result.entry.name} ({section_label(result.entry.airfoil.stem)})"
    return [LiftCurve(label=rf"{name}, $\varepsilon$ = {result.sizing.twist_deg:+.2f}°",
                      alpha_deg=result.polar.alpha_deg, cl=result.polar.cl,
                      trim_alpha_deg=result.trim_alpha_deg, trim_cl=cl_design),
            LiftCurve(label=rf"{name}, $\varepsilon$ = 0 (untwisted)",
                      alpha_deg=result.untwisted.alpha_deg, cl=result.untwisted.cl,
                      trim_alpha_deg=math.nan, trim_cl=math.nan, has_trim=False)]


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
        "e far-field = CL_des² / (π·A·CDi); e loading from a sine-series fit of the strip "
        "loading c·cl at α = 0 (Munk: planar-wake CDi depends only on the span loading).",
        "",
        "| Wing | Section | Twist [deg] | CL(α=0) | VSPAERO runs | CL_α [1/rad] (R²) "
        "| α_0L [deg] | α_trim [deg] | CDi trim | e far-field | e loading | CMy trim "
        "| max |Δε| vs linear [deg] |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.entry.name} | {r.entry.airfoil.name} | {r.sizing.twist_deg:+.3f} "
            f"| {r.sizing.cl:.5f} | {len(r.sizing.history)} "
            f"| {r.cl_alpha_per_rad:.3f} ({r.cl_alpha_r_squared:.5f}) "
            f"| {r.alpha_zero_lift_deg:+.3f} | {r.trim_alpha_deg:+.4f} | {r.trim_cdi:.5f} "
            f"| {r.span_efficiency:.3f} | {r.span_efficiency_loading:.3f} | {r.trim_cmy:+.4f} "
            f"| {r.twist_deviation_deg:.3f} |")
    lines += [
        "",
        "## Untwisted wing",
        "",
        "Same wing with ε = 0. Its zero-lift body angle plus the root incidence is the section "
        "zero-lift angle that the vortex lattice uses (camber surface, inviscid).",
        "",
        "| Wing | CL(α=0) | α_0L untwisted, body [deg] | α_0l effective = α_0L + i_r [deg] |",
        "|---|---|---|---|",
    ]
    for r in results:
        cl_0 = float(np.interp(0.0, r.untwisted.alpha_deg, r.untwisted.cl))
        lines.append(f"| {r.entry.name} | {cl_0:.5f} | {r.untwisted_alpha_zero_lift_deg:+.3f} "
                     f"| {r.untwisted_alpha_zero_lift_deg + case.twist.root_incidence_deg:+.3f} |")
    lines += [
        "",
        "## Grid convergence",
        "",
        "Sized wing at α = 0. Δ is relative to the finest level. CDi far-field is the VSPAERO "
        "value; CDi loading = CL² / (π·A·e loading).",
        "",
        "| Wing | segment_tess | chord_tess | strips | CL | ΔCL | CDi far-field | ΔCDi "
        "| e loading | CDi loading | ΔCDi loading |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        m = r.mesh
        for i in range(m.cl.size):
            lines.append(
                f"| {r.entry.name} | {m.segment_tess[i]:.0f} | {m.chord_tess[i]:.0f} "
                f"| {m.strips[i]:.0f} | {m.cl[i]:.5f} | {100 * (m.cl[i] / m.cl[-1] - 1):+.2f}% "
                f"| {m.cdi[i]:.6f} | {100 * (m.cdi[i] / m.cdi[-1] - 1):+.2f}% "
                f"| {m.e_loading[i]:.4f} | {m.cdi_loading[i]:.6f} "
                f"| {100 * (m.cdi_loading[i] / m.cdi_loading[-1] - 1):+.2f}% |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
