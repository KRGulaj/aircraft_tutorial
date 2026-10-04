# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Figures of the 2D section analysis. One function per figure; each writes one PNG.

Only converged points are plotted. The drag axis ends at cd = 0.024, the range of the NACA TR 824
plots, so the XFoil figures and the report screenshots read the same way.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend first)
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from aircraft_tutorial.geometry.airfoil import Airfoil  # noqa: E402
from aircraft_tutorial.plots import style  # noqa: E402
from aircraft_tutorial.section.metrics import SectionMetrics  # noqa: E402
from aircraft_tutorial.section.polar import Polar, RunInfo  # noqa: E402

CD_AXIS_MAX: Final[float] = 0.024
BOX_FONT: Final[float] = 6.0
LEGEND_FONT: Final[float] = 6.0
_BOX: Final[dict[str, object]] = {"boxstyle": "round,pad=0.3", "facecolor": "white",
                                  "edgecolor": style.GRID, "linewidth": 0.5}


def condition_label(info: RunInfo) -> str:
    """'Re = 8.9×10⁶, M = 0.15, N_crit = 10, 240 panel nodes' in mathtext."""
    exp = int(math.floor(math.log10(info.reynolds)))
    mant = info.reynolds / 10**exp
    return (rf"Re = ${mant:.3g}\times10^{{{exp}}}$, M = {info.mach:g}, "
            rf"$N_{{crit}}$ = {info.ncrit:g}, {info.panel_nodes} panel nodes")


def plot_polar_panels(polar: Polar, m: SectionMetrics, colour: str,
                      window: tuple[float, float], path: Path) -> None:
    """Three panels for one airfoil: cl–α, cl–cd (drag polar), cm–α, with the key values.

    Args:
        polar: Polar of the airfoil.
        m: Its characteristics.
        colour: Series colour of the airfoil.
        window: Linear lift-curve window (min, max) [deg].
        path: Output PNG.
    """
    style.apply()
    c = polar.converged_only()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(style.FULL_WIDTH_IN, 2.6))

    ax1.plot(c.alpha_deg, c.cl, color=colour, marker="o", markersize=2.0)
    a_fit = np.array([m.alpha_0l_deg - 1.0, window[1] + 3.0])
    ax1.plot(a_fit, m.lift_fit.intercept + m.a0_per_deg * a_fit, color=style.FIT_LINE,
             linestyle="--", linewidth=0.8)
    _ring(ax1, m.stall.alpha_deg, m.stall.cl_max, colour)
    ax1.text(0.96, 0.04, "\n".join([
        rf"$a_0$ = {_n(m.a0_per_deg, '.4f')} /deg",
        rf"({_n(m.a0_per_rad, '.2f')} /rad)",
        rf"$\alpha_{{0L}}$ = {_n(m.alpha_0l_deg, '.2f')}°",
        rf"$c_{{l,max}}$ = {_n(m.stall.cl_max, '.3f')}",
        rf"at $\alpha$ = {_n(m.stall.alpha_deg, 'g')}°"]),
        transform=ax1.transAxes, ha="right", va="bottom", bbox=_BOX, fontsize=BOX_FONT)
    _labels(ax1, r"$\alpha$ [deg]", r"$c_l$ [–]")

    ax2.plot(c.cd, c.cl, color=colour, marker="o", markersize=2.0)
    _ring(ax2, m.cd_min, m.cl_at_cd_min, colour)
    if m.low_drag is not None:
        ax2.axhspan(m.low_drag.cl_lower, m.low_drag.cl_upper, color=style.GRID, alpha=0.6,
                    linewidth=0)
    lines = [rf"$c_{{d,min}}$ = {_n(m.cd_min, '.5f')}",
             rf"at $c_l$ = {_n(m.cl_at_cd_min, '.2f')}"]
    if m.low_drag is not None:
        lines.append("low-drag range:")
        lines.append(rf"$c_l$ = {_n(m.low_drag.cl_lower, '.2f')}…"
                     rf"{_n(m.low_drag.cl_upper, '.2f')}")
    ax2.text(0.96, 0.04, "\n".join(lines), transform=ax2.transAxes, ha="right", va="bottom",
             bbox=_BOX, fontsize=BOX_FONT)
    ax2.set_xlim(0.0, CD_AXIS_MAX)
    _labels(ax2, r"$c_d$ [–]", r"$c_l$ [–]")

    ax3.plot(c.alpha_deg, c.cm, color=colour, marker="o", markersize=2.0)
    ax3.hlines(m.cm_c4, window[0], window[1], color=style.FIT_LINE, linestyle="--",
               linewidth=0.8)
    ax3.text(0.04, 0.96, "\n".join([
        rf"$c_{{m,c/4}}$ = {_n(m.cm_c4, '.4f')}",
        rf"(mean, {_n(window[0], 'g')}…{_n(window[1], 'g')}°)",
        rf"$x_{{ac}}/c$ = {_n(m.x_ac, '.3f')}"]),
        transform=ax3.transAxes, va="top", bbox=_BOX, fontsize=BOX_FONT)
    _labels(ax3, r"$\alpha$ [deg]", r"$c_{m,c/4}$ [–]")

    fig.suptitle(f"{polar.info.airfoil}, XFoil: {condition_label(polar.info)}", y=1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_overlay(polars: dict[str, Polar], colours: dict[str, str], path: Path) -> None:
    """Three panels with every airfoil overlaid: cl–α, cl–cd, cm–α.

    Args:
        polars: Airfoil name → polar (same condition).
        colours: Airfoil name → series colour.
        path: Output PNG.
    """
    style.apply()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(style.FULL_WIDTH_IN, 2.6))
    for name, p in polars.items():
        c = p.converged_only()
        ax1.plot(c.alpha_deg, c.cl, color=colours[name], label=name)
        ax2.plot(c.cd, c.cl, color=colours[name])
        ax3.plot(c.alpha_deg, c.cm, color=colours[name])
    ax2.set_xlim(0.0, CD_AXIS_MAX)
    _labels(ax1, r"$\alpha$ [deg]", r"$c_l$ [–]")
    _labels(ax2, r"$c_d$ [–]", r"$c_l$ [–]")
    _labels(ax3, r"$\alpha$ [deg]", r"$c_{m,c/4}$ [–]")
    ax1.legend(loc="upper left", frameon=False, fontsize=LEGEND_FONT)
    info = next(iter(polars.values())).info
    fig.suptitle(f"XFoil: {condition_label(info)}", y=1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_drag_polars(polars: dict[str, Polar], colours: dict[str, str], path: Path) -> None:
    """One panel, cl–cd of every airfoil, with the full converged range.

    Args:
        polars: Airfoil name → polar (same condition).
        colours: Airfoil name → series colour.
        path: Output PNG.
    """
    style.apply()
    fig, ax = plt.subplots(figsize=(style.FULL_WIDTH_IN / 2.0, 2.8))
    for name, p in polars.items():
        c = p.converged_only()
        ax.plot(c.cd, c.cl, color=colours[name], marker="o", markersize=2.0, label=name)
    ax.set_xlim(0.0, CD_AXIS_MAX)
    _labels(ax, r"$c_d$ [–]", r"$c_l$ [–]")
    ax.legend(loc="lower right", frameon=False, fontsize=LEGEND_FONT)
    info = next(iter(polars.values())).info
    ax.set_title(f"XFoil drag polars\n{condition_label(info)}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_sections(sections: dict[str, Airfoil], colours: dict[str, str], path: Path) -> None:
    """Section shapes at true aspect ratio, one row per airfoil.

    Args:
        sections: Airfoil name → contour (unit chord).
        colours: Airfoil name → series colour.
        path: Output PNG.
    """
    style.apply()
    fig, axes = plt.subplots(len(sections), 1, figsize=(style.FULL_WIDTH_IN, 1.6 * len(sections)),
                             squeeze=False)
    for ax, (name, a) in zip(axes[:, 0], sections.items(), strict=True):
        ax.plot(a.x, a.y, color=colours[name], linewidth=1.0)
        ax.set_aspect("equal")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.08, 0.12)
        ax.set_title(name, loc="left", fontsize=BOX_FONT + 1.0)
        ax.set_ylabel(r"$y/c$")
    axes[-1, 0].set_xlabel(r"$x/c$")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _n(value: float, fmt: str) -> str:
    """Format a number with a typographic minus sign (U+2212)."""
    return format(value, fmt).replace("-", "−")


def _labels(ax: Axes, xlabel: str, ylabel: str) -> None:
    """Axis labels with units."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _ring(ax: Axes, x: float, y: float, colour: str) -> None:
    """Hollow marker that highlights one point."""
    ax.plot([x], [y], marker="o", markersize=6.5, markerfacecolor="white",
            markeredgecolor=colour, markeredgewidth=1.2, linestyle="none", zorder=5)
