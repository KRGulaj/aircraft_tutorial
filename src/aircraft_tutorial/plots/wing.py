# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Figures of the 3D wing analysis: the lift curve and the drag polar, each with the trim point."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.typing import ColorType  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402


@dataclass(frozen=True)
class LiftCurve:
    """One lift curve to draw.

    Attributes:
        label: Legend label.
        alpha_deg: Body angle of attack [deg].
        cl: Lift coefficient [-].
        trim_alpha_deg: Body angle of the trim point [deg].
        trim_cl: Lift coefficient of the trim point [-].
    """

    label: str
    alpha_deg: NDArray[np.float64]
    cl: NDArray[np.float64]
    trim_alpha_deg: float
    trim_cl: float


def plot_lift_curves(curves: list[LiftCurve], cl_design: float, title: str, path: Path) -> None:
    """Draw CL against body angle of attack, each curve with its trim point.

    Args:
        curves: Lift curves to draw.
        cl_design: Design lift coefficient, drawn as a horizontal line [-].
        title: Figure title.
        path: Output image path.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for curve in curves:
        (line,) = ax.plot(curve.alpha_deg, curve.cl, marker="o", markersize=3, label=curve.label)
        ax.plot(curve.trim_alpha_deg, curve.trim_cl, marker="*", markersize=14,
                color=line.get_color(), markeredgecolor="black", linestyle="none",
                label=f"{curve.label} trim: α = {curve.trim_alpha_deg:.2f}°, "
                      f"CL = {curve.trim_cl:.3f}")
    ax.axhline(cl_design, color="grey", linestyle="--", linewidth=1,
               label=f"CL_des = {cl_design:.3f}")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.axvline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel("body angle of attack α [deg]")
    ax.set_ylabel("CL [-]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


@dataclass(frozen=True)
class DragPolar:
    """One drag polar to draw, with its build-up.

    Attributes:
        label: Legend label.
        cl: Lift coefficient [-].
        cdi: Induced drag coefficient [-].
        cd_profile: Profile drag coefficient [-]; NaN where a strip is outside the polar.
        cd_wave: Wave drag coefficient [-].
        trim_cl: Lift coefficient of the trim point [-].
        trim_cd: Total drag coefficient of the trim point [-].
    """

    label: str
    cl: NDArray[np.float64]
    cdi: NDArray[np.float64]
    cd_profile: NDArray[np.float64]
    cd_wave: NDArray[np.float64]
    trim_cl: float
    trim_cd: float

    @property
    def cd(self) -> NDArray[np.float64]:
        """Total drag coefficient [-]."""
        return self.cdi + self.cd_profile + self.cd_wave


def plot_drag_build_up(polar: DragPolar, cl_design: float, title: str, path: Path) -> None:
    """Draw CL against the cumulative drag terms CDi, + CD_profile, + CD_wave, with the trim
    point on the total.

    Args:
        polar: Drag polar.
        cl_design: Design lift coefficient [-].
        title: Figure title.
        path: Output image path.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(polar.cdi, polar.cl, linestyle=":", marker=".", label="CDi (VSPAERO)")
    ax.plot(polar.cdi + polar.cd_profile, polar.cl, linestyle="--", marker=".",
            label="CDi + CD_profile (XFoil strips)")
    (line,) = ax.plot(polar.cd, polar.cl, marker="o", markersize=3,
                      label="CD = CDi + CD_profile + CD_wave")
    _trim(ax, polar, line.get_color())
    _finish(ax, cl_design, title, path, fig)


def plot_drag_polars(polars: list[DragPolar], cl_design: float, title: str, path: Path) -> None:
    """Draw the total drag polars of several wings, each with its trim point.

    Args:
        polars: Drag polars.
        cl_design: Design lift coefficient [-].
        title: Figure title.
        path: Output image path.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for polar in polars:
        (line,) = ax.plot(polar.cd, polar.cl, marker="o", markersize=3, label=polar.label)
        _trim(ax, polar, line.get_color())
    _finish(ax, cl_design, title, path, fig)


def _trim(ax: Axes, polar: DragPolar, color: ColorType) -> None:
    """Mark the trim point of a polar."""
    ax.plot(polar.trim_cd, polar.trim_cl, marker="*", markersize=14, color=color,
            markeredgecolor="black", linestyle="none",
            label=f"{polar.label} trim: CD = {polar.trim_cd:.5f}, CL = {polar.trim_cl:.3f}, "
                  f"L/D = {polar.trim_cl / polar.trim_cd:.1f}")


def _finish(ax: Axes, cl_design: float, title: str, path: Path, fig: Figure) -> None:
    """Design-CL line, labels, grid, legend; save and close."""
    ax.axhline(cl_design, color="grey", linestyle="--", linewidth=1,
               label=f"CL_des = {cl_design:.3f}")
    ax.set_xlim(left=0.0)
    ax.set_xlabel("CD [-]")
    ax.set_ylabel("CL [-]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
