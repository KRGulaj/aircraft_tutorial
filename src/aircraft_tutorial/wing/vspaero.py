# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""VSPAERO vortex-lattice angle-of-attack sweep of the model in memory, and its results.

The wing is analysed as a thin (camber) surface with y-symmetry. Kept from VSPAERO: CL, the
pitching moment CMy about the reference point, the wake (Trefftz-plane) induced drag CDiw and the
spanwise strip loads. VSPAERO's own CDo, a flat-plate skin-friction estimate, is not used; the
profile drag comes from the XFoil section polars.

Compressibility enters through VSPAERO's Prandtl-Glauert correction at the run Mach number.

Needs the OpenVSP Python API (imported at module level).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from aircraft_tutorial.config.wing import VlmSettings
from aircraft_tutorial.wing.openvsp_api import vsp
from aircraft_tutorial.wing.vlm_results import StripLoads, VlmPolar

_STRIP_COLUMNS: Final[tuple[str, ...]] = ("alpha_deg", "y_m", "chord_m", "area_m2", "cl")


class VspaeroError(RuntimeError):
    """Raised when VSPAERO cannot be run or returns no results."""


@dataclass(frozen=True)
class Reference:
    """Reference quantities of the coefficients.

    Attributes:
        area_m2: Reference area [m^2].
        span_m: Reference span [m].
        chord_m: Reference chord (MAC) [m].
        x_m: Moment reference x [m].
        z_m: Moment reference z [m].
    """

    area_m2: float
    span_m: float
    chord_m: float
    x_m: float
    z_m: float


@dataclass(frozen=True)
class FlowInput:
    """Flow condition of a run.

    Attributes:
        mach: Free-stream Mach number [-].
        velocity_m_s: True airspeed [m/s].
        density_kg_m3: Density [kg/m^3].
        reynolds: Reynolds number based on the reference chord [-].
    """

    mach: float
    velocity_m_s: float
    density_kg_m3: float
    reynolds: float


def alpha_grid(vlm: VlmSettings) -> NDArray[np.float64]:
    """The equally spaced body angles of the sweep [deg]."""
    return np.linspace(vlm.alpha_start_deg, vlm.alpha_end_deg, vlm.alpha_npts)


def run_sweep(vsp3_path: Path, reference: Reference, flow: FlowInput, vlm: VlmSettings, *,
              alpha_start_deg: float, alpha_end_deg: float, alpha_npts: int
              ) -> tuple[VlmPolar, StripLoads]:
    """Save the model in memory and run a VSPAERO angle-of-attack sweep on it.

    VSPAERO writes its input and output files next to the .vsp3 file.

    Args:
        vsp3_path: Where to save the model.
        reference: Reference area, span, chord and moment point.
        flow: Mach, velocity, density and Reynolds number.
        vlm: Solver settings (wake iterations, threads).
        alpha_start_deg: First body angle [deg].
        alpha_end_deg: Last body angle [deg].
        alpha_npts: Number of angles.

    Returns:
        The whole-wing polar and the strip loads.

    Raises:
        VspaeroError: If vspaero.exe is missing or the run returns no results.
    """
    vsp.WriteVSPFile(str(vsp3_path))
    vsp_dir = str(Path(vsp.__file__).parent)
    vsp.SetVSPAEROPath(vsp_dir)
    if not vsp.CheckForVSPAERO(vsp_dir):
        raise VspaeroError(f"vspaero.exe not found in {vsp_dir}")

    geometry = "VSPAEROComputeGeometry"
    vsp.SetAnalysisInputDefaults(geometry)
    vsp.SetIntAnalysisInput(geometry, "GeomSet", [vsp.SET_NONE])
    vsp.SetIntAnalysisInput(geometry, "ThinGeomSet", [vsp.SET_ALL])
    vsp.SetIntAnalysisInput(geometry, "Symmetry", [1])
    vsp.ExecAnalysis(geometry)

    a = "VSPAEROSweep"
    vsp.SetAnalysisInputDefaults(a)
    vsp.SetIntAnalysisInput(a, "GeomSet", [vsp.SET_NONE])
    vsp.SetIntAnalysisInput(a, "ThinGeomSet", [vsp.SET_ALL])
    vsp.SetIntAnalysisInput(a, "Symmetry", [1])
    vsp.SetIntAnalysisInput(a, "RefFlag", [vsp.MANUAL_REF])
    vsp.SetDoubleAnalysisInput(a, "Sref", [reference.area_m2])
    vsp.SetDoubleAnalysisInput(a, "bref", [reference.span_m])
    vsp.SetDoubleAnalysisInput(a, "cref", [reference.chord_m])
    vsp.SetDoubleAnalysisInput(a, "Xcg", [reference.x_m])
    vsp.SetDoubleAnalysisInput(a, "Ycg", [0.0])
    vsp.SetDoubleAnalysisInput(a, "Zcg", [reference.z_m])
    vsp.SetDoubleAnalysisInput(a, "AlphaStart", [alpha_start_deg])
    vsp.SetDoubleAnalysisInput(a, "AlphaEnd", [alpha_end_deg])
    vsp.SetIntAnalysisInput(a, "AlphaNpts", [alpha_npts])
    vsp.SetDoubleAnalysisInput(a, "BetaStart", [0.0])
    vsp.SetDoubleAnalysisInput(a, "BetaEnd", [0.0])
    vsp.SetIntAnalysisInput(a, "BetaNpts", [1])
    vsp.SetDoubleAnalysisInput(a, "MachStart", [flow.mach])
    vsp.SetDoubleAnalysisInput(a, "MachEnd", [flow.mach])
    vsp.SetIntAnalysisInput(a, "MachNpts", [1])
    vsp.SetDoubleAnalysisInput(a, "ReCref", [flow.reynolds])
    vsp.SetDoubleAnalysisInput(a, "ReCrefEnd", [flow.reynolds])
    vsp.SetIntAnalysisInput(a, "ReCrefNpts", [1])
    vsp.SetDoubleAnalysisInput(a, "Vinf", [flow.velocity_m_s])
    vsp.SetDoubleAnalysisInput(a, "Rho", [flow.density_kg_m3])
    vsp.SetIntAnalysisInput(a, "WakeNumIter", [vlm.wake_iter])
    vsp.SetIntAnalysisInput(a, "NCPU", [vlm.n_cpu])
    vsp.ExecAnalysis(a)

    errors = vsp.ErrorMgrSingleton.getInstance()
    messages: list[str] = []
    while errors.GetNumTotalErrors() > 0:
        messages.append(str(errors.PopLastError().GetErrorString()))
    polar = _read_polar()
    if len(polar.alpha_deg) != alpha_npts:
        raise VspaeroError(f"VSPAERO returned {len(polar.alpha_deg)} of {alpha_npts} angles; "
                           f"files in {vsp3_path.parent}; messages: {messages}")
    return polar, _read_strips()


def _results(rid: str, name: str) -> NDArray[np.float64]:
    """One result vector as a float array."""
    return np.array(list(vsp.GetDoubleResults(rid, name)), dtype=np.float64)


def _read_polar() -> VlmPolar:
    """Whole-wing coefficients of the latest sweep."""
    rid: str = vsp.FindLatestResultsID("VSPAERO_Polar")
    if not rid:
        raise VspaeroError("VSPAERO produced no VSPAERO_Polar results")
    alpha = _results(rid, "Alpha")
    order = np.argsort(alpha)
    return VlmPolar(alpha_deg=alpha[order], cl=_results(rid, "CLtot")[order],
                    cdi=_results(rid, "CDiw")[order], cmy=_results(rid, "CMytot")[order])


def _read_strips() -> StripLoads:
    """Strip loads of the right half-wing (y > 0) of every angle of the latest sweep."""
    cols: dict[str, list[NDArray[np.float64]]] = {c: [] for c in _STRIP_COLUMNS}
    for i in range(vsp.GetNumResults("VSPAERO_Load")):
        rid: str = vsp.FindResultsID("VSPAERO_Load", i)
        y = _results(rid, "Yavg")
        right = y > 0.0
        cols["alpha_deg"].append(np.full(int(right.sum()), _results(rid, "FC_AoA_")[0]))
        cols["y_m"].append(y[right])
        cols["chord_m"].append(_results(rid, "Chord")[right])
        cols["area_m2"].append(_results(rid, "dArea")[right])
        cols["cl"].append(_results(rid, "cl")[right])
    if not cols["y_m"]:
        raise VspaeroError("VSPAERO produced no VSPAERO_Load results")
    flat = {c: np.concatenate(v) for c, v in cols.items()}
    order = np.lexsort((flat["y_m"], flat["alpha_deg"]))
    return StripLoads(**{c: v[order] for c, v in flat.items()})
