# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""OpenVSP model of the trapezoidal wing with a root incidence and a linear twist.

- The root incidence is a rotation of the whole wing about the y axis through the root leading
  edge, so the VSPAERO angle of attack is the body angle of attack.
- The twist varies linearly with span, ε(η) = ε_tip·η. OpenVSP lofts a segment linearly between
  the coordinates of its two end sections, so inside one segment the section angle is a
  chord-weighted mean of the end angles, not a linear function of η. With the strong taper of
  this wing (λ ≈ 0.11) a single segment twisted −6° at the tip is only −0.58° at mid-span. The
  half-wing is therefore split into segments whose end sections carry the exact linear twist;
  the boundaries are half-cosine spaced, η_i = sin(π·i / 2n), so the segments are short towards
  the tip where the chord ratio across a segment is largest.
- Every segment keeps the quarter-chord sweep and the chords of the one trapezoid, and every
  section is twisted about its quarter chord (OpenVSP Twist is absolute, about Twist_Location).

Needs the OpenVSP Python API (imported at module level).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from aircraft_tutorial.config.wing import VlmSettings
from aircraft_tutorial.wing.openvsp_api import vsp
from aircraft_tutorial.geometry.airfoil import Airfoil, normalized
from aircraft_tutorial.wing.planform import TrapezoidalPlanform

TWIST_LOCATION = 0.25
"""Chord fraction of the twist axis [-]."""


def segment_stations(segments: int) -> NDArray[np.float64]:
    """Half-cosine spaced segment boundaries η_i = sin(π·i / 2n), i = 0 … n [-]."""
    return np.sin(0.5 * math.pi * np.arange(segments + 1) / segments)


def new_wing_model(name: str, planform: TrapezoidalPlanform, airfoil: Airfoil, *,
                   dihedral_deg: float, root_incidence_deg: float, twist_deg: float,
                   vlm: VlmSettings) -> str:
    """Clear the OpenVSP model and add the wing.

    Args:
        name: Geom name.
        planform: Trapezoidal planform.
        airfoil: Section used at every station.
        dihedral_deg: Dihedral [deg].
        root_incidence_deg: Root incidence relative to the body axis [deg].
        twist_deg: Tip twist relative to the root, negative = wash-out [deg].
        vlm: Segment count and mesh settings.

    Returns:
        The OpenVSP geom ID of the wing.
    """
    vsp.VSPRenew()
    vsp.DeleteAllResults()
    wid: str = vsp.AddGeom("WING")
    vsp.SetGeomName(wid, name)
    vsp.SetParmVal(wid, "Y_Rel_Rotation", "XForm", root_incidence_deg)
    vsp.SetParmVal(wid, "Tess_W", "Shape", vlm.chord_tess)
    for _ in range(vlm.segments - 1):
        vsp.InsertXSec(wid, 1, vsp.XS_FOUR_SERIES)
    vsp.Update()

    eta = segment_stations(vlm.segments)
    for i in range(1, vlm.segments + 1):
        grp = f"XSec_{i}"
        vsp.SetDriverGroup(wid, i, vsp.SPAN_WSECT_DRIVER, vsp.ROOTC_WSECT_DRIVER,
                           vsp.TIPC_WSECT_DRIVER)
        vsp.SetParmVal(wid, "Span", grp, (eta[i] - eta[i - 1]) * planform.semi_span_m)
        vsp.SetParmVal(wid, "Root_Chord", grp, planform.chord_at(eta[i - 1] * planform.semi_span_m))
        vsp.SetParmVal(wid, "Tip_Chord", grp, planform.chord_at(eta[i] * planform.semi_span_m))
        vsp.SetParmVal(wid, "Sweep", grp, planform.sweep_quarter_chord_deg)
        vsp.SetParmVal(wid, "Sweep_Location", grp, 0.25)
        vsp.SetParmVal(wid, "Dihedral", grp, dihedral_deg)
        vsp.SetParmVal(wid, "Twist", grp, twist_deg * eta[i])
        vsp.SetParmVal(wid, "Twist_Location", grp, TWIST_LOCATION)
        vsp.SetParmVal(wid, "SectTess_U", grp, vlm.segment_tess)
        vsp.Update()

    xu, yu, xl, yl = normalized(airfoil).surfaces()
    upper = [vsp.vec3d(float(x), float(y), 0.0) for x, y in zip(xu, yu, strict=True)]
    lower = [vsp.vec3d(float(x), float(y), 0.0) for x, y in zip(xl, yl, strict=True)]
    surf = vsp.GetXSecSurf(wid, 0)
    for index in range(vlm.segments + 1):
        vsp.ChangeXSecShape(surf, index, vsp.XS_FILE_AIRFOIL)
        vsp.SetAirfoilPnts(vsp.GetXSec(surf, index), upper, lower)
    vsp.Update()
    return wid


def total_area_and_span(wid: str) -> tuple[float, float]:
    """Area and span of the wing as OpenVSP computes them, for a check against the input.

    Returns:
        (area, both halves [m^2], span [m]).
    """
    return (float(vsp.GetParmVal(wid, "TotalArea", "WingGeom")),
            float(vsp.GetParmVal(wid, "TotalSpan", "WingGeom")))


def section_angles(wid: str, samples: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Geometric section angle along the span, measured on the OpenVSP surface.

    The angle is that of the leading-edge-to-trailing-edge line in the x-z plane, nose up
    positive; it includes the root incidence.

    Args:
        wid: Geom ID of the wing.
        samples: Number of samples along the surface parameter.

    Returns:
        (y [m], section angle [deg]), ordered root to tip.
    """
    ys: list[float] = []
    angles: list[float] = []
    for u in np.linspace(0.0, 1.0, samples):
        te = vsp.CompPnt01(wid, 0, float(u), 0.0)
        le = vsp.CompPnt01(wid, 0, float(u), 0.5)
        ys.append(float(le.y()))
        angles.append(math.degrees(math.atan2(-(te.z() - le.z()), te.x() - le.x())))
    y = np.array(ys, dtype=np.float64)
    keep = np.concatenate([[True], np.diff(y) > 1e-9])  # the end caps repeat a station
    return y[keep], np.array(angles, dtype=np.float64)[keep]
