# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
# matplotlib annotates the **kwargs of Axes.plot, axhline, set_title, legend, savefig etc. as
# untyped, so strict pyright reports every call as partially unknown. The check is off for
# member types in this file only; all arguments passed here are typed.
# pyright: reportUnknownMemberType=false
"""Report figures of the 3D wing analysis: lift curves and drag polars with the trim point.

The figures follow the reporting notes of the AE2111-II homework brief:

- every axis has a quantity and a unit (note 11);
- every figure states the Mach and Reynolds numbers of the analysis in a box on the plot
  (note 11);
- the trim point carries its values on the plot, not only in a long legend (notes 6, 12);
- the figure is sized for an A4 page, 16 cm wide, with print-size fonts, and is saved as a
  300 dpi PNG (note 12).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.offsetbox import AnchoredText  # noqa: E402
from matplotlib.ticker import AutoMinorLocator  # noqa: E402
from matplotlib.typing import ColorType, RcKeyType  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

CM_PER_INCH: Final[float] = 2.54
FIGURE_WIDTH_CM: Final[float] = 16.0
"""Text width of an A4 page with 2.5 cm margins, minus a little."""
FIGURE_HEIGHT_CM: Final[float] = 12.0
PNG_DPI: Final[int] = 300

_STYLE: Final[dict[RcKeyType, float | str | bool]] = {
    "font.size": 10.0,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10.0,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 8.5,
    "lines.linewidth": 1.4,
    "axes.grid": True,
    "axes.grid.which": "both",
    "grid.alpha": 0.25,
    "mathtext.default": "regular",
}
_LINESTYLES: Final[tuple[str, ...]] = ("-", "--", ":", "-.")
_MARKERS: Final[tuple[str, ...]] = ("o", "s", "^", "D")
_TRIM_MARKER_SIZE: Final[float] = 15.0
_BOUND_TRIM_SIZE: Final[float] = 7.0
"""Open circle of the trim point on the bounding-k_Λ drag polar."""
_ALPHA_LABEL: Final[str] = r"Body angle of attack $\alpha$ [deg]"
_CL_LABEL: Final[str] = r"Lift coefficient $C_L$ [-]"
_CD_LABEL: Final[str] = r"Drag coefficient $C_D$ [-]"


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


@dataclass(frozen=True)
class DragPolar:
    """One drag polar to draw, with its build-up and the two bounds of the sweep factor k_Λ.

    Attributes:
        label: Legend label.
        cl: Lift coefficient [-].
        cdi: Induced drag coefficient [-].
        cd_profile: Profile drag coefficient with the baseline sweep factor [-].
        cd_profile_bound: Profile drag coefficient with the bounding sweep factor [-].
        cd_wave: Wave drag coefficient [-].
        k_sweep: Baseline sweep factor k_Λ of the profile drag [-].
        k_sweep_bound: Bounding sweep factor k_Λ of the profile drag [-].
        trim_cl: Lift coefficient of the trim point [-].
        trim_cd: Total drag coefficient of the trim point, baseline k_Λ [-].
        trim_cd_bound: Total drag coefficient of the trim point, bounding k_Λ [-].
    """

    label: str
    cl: NDArray[np.float64]
    cdi: NDArray[np.float64]
    cd_profile: NDArray[np.float64]
    cd_profile_bound: NDArray[np.float64]
    cd_wave: NDArray[np.float64]
    k_sweep: float
    k_sweep_bound: float
    trim_cl: float
    trim_cd: float
    trim_cd_bound: float

    @property
    def cd(self) -> NDArray[np.float64]:
        """Total drag coefficient, baseline k_Λ [-]."""
        return self.cdi + self.cd_profile + self.cd_wave

    @property
    def cd_bound(self) -> NDArray[np.float64]:
        """Total drag coefficient, bounding k_Λ [-]."""
        return self.cdi + self.cd_profile_bound + self.cd_wave


def sci_tex(value: float, digits: int = 3) -> str:
    """A number in scientific notation for mathtext, for example 6.62\\times10^{7}.

    Args:
        value: Number to format, not zero.
        digits: Significant digits.

    Returns:
        The mathtext string, without the enclosing dollar signs.
    """
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / 10.0**exponent
    return rf"{mantissa:.{digits - 1}f}\times10^{{{exponent}}}"


def flight_conditions(mach: float, reynolds_mac: float, altitude_m: float) -> str:
    """Box line with the flight condition of the wing analysis.

    Args:
        mach: Free-stream Mach number [-].
        reynolds_mac: Reynolds number based on the MAC [-].
        altitude_m: Altitude [m].
    """
    altitude = f"{altitude_m:,.0f}".replace(",", " ")  # 10 668, SI thousands spacing
    return (rf"Wing: $M_\infty$ = {mach:.3f}, $Re_{{MAC}}$ = ${sci_tex(reynolds_mac)}$, "
            f"h = {altitude} m")


def section_conditions(reynolds: float, mach: float) -> str:
    """Box line with the condition of the XFoil section polars.

    Args:
        reynolds: Reynolds number of the polars [-].
        mach: Mach number of the polars [-].
    """
    return rf"Sections (XFoil, sweep-normal): Re = ${sci_tex(reynolds)}$, M = {mach:.2f}"


def section_label(airfoil_stem: str) -> str:
    """Readable section name from a coordinate-file stem, for example "NACA2412" → "NACA 2412".

    Args:
        airfoil_stem: File name without extension.
    """
    return re.sub(r"^NACA(?=\d)", "NACA ", airfoil_stem.removesuffix("_gen"))


def plot_lift_curves(curves: list[LiftCurve], cl_design: float, title: str,
                     conditions: list[str], path: Path) -> None:
    """Draw C_L against body angle of attack with the trim points.

    Coinciding trim points are annotated once.

    Args:
        curves: Lift curves to draw.
        cl_design: Design lift coefficient, drawn as a horizontal line [-].
        title: Figure title.
        conditions: Lines of the condition box (method, Mach and Reynolds numbers).
        path: Output PNG path.
    """
    with plt.rc_context(_STYLE):
        fig, ax = _new_figure()
        for i, curve in enumerate(curves):
            # Wings sized to the same CL_des can coincide: distinct line styles and open
            # markers keep every curve visible.
            (line,) = ax.plot(curve.alpha_deg, curve.cl,
                              linestyle=_LINESTYLES[i % len(_LINESTYLES)],
                              marker=_MARKERS[i % len(_MARKERS)], markersize=4.5,
                              markerfacecolor="none", label=curve.label)
            _trim_star(ax, curve.trim_alpha_deg, curve.trim_cl, line.get_color())
        for alpha, cl in _unique_points([(c.trim_alpha_deg, c.trim_cl) for c in curves]):
            _annotate(ax, alpha, cl, rf"Trim: $\alpha$ = {alpha:.2f}°, $C_L$ = {cl:.3f}",
                      offset=(18.0, -34.0))
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.axvline(0.0, color="black", linewidth=0.7)
        ax.set_xlabel(_ALPHA_LABEL)
        ax.set_ylabel(_CL_LABEL)
        ax.add_artist(AnchoredText("\n".join(conditions), loc="upper left", frameon=True,
                                   prop={"size": 8.5}))
        _finish(fig, ax, title, cl_design, path)


def plot_drag_build_up(polar: DragPolar, cl_design: float, title: str,
                       conditions: list[str], path: Path) -> None:
    """Draw C_L against the cumulative drag terms C_Di, + C_D,p, + C_D,w, with the trim point
    on the total.

    The terms with profile drag are drawn for both sweep factors k_Λ: a solid line for the
    baseline, a dashed line for the bound and a shaded band between them.

    Args:
        polar: Drag polar.
        cl_design: Design lift coefficient [-].
        title: Figure title.
        conditions: Lines of the condition box.
        path: Output PNG path.
    """
    k, k_bound = _k_sweep_text(polar.k_sweep), _k_sweep_text(polar.k_sweep_bound)
    with plt.rc_context(_STYLE):
        fig, ax = _new_figure()
        ax.plot(polar.cdi, polar.cl, linestyle=":", marker="^", markersize=4,
                markerfacecolor="none", label=r"$C_{D_i}$ (VSPAERO, far-field)")
        profile = r"$C_{D_i} + C_{D,p}$ (XFoil strips)"
        (line,) = ax.plot(polar.cdi + polar.cd_profile, polar.cl, marker="s", markersize=4,
                          markerfacecolor="none", label=f"{profile}, {k}")
        _bound_band(ax, polar.cdi + polar.cd_profile, polar.cdi + polar.cd_profile_bound,
                    polar.cl, line.get_color(), f"{profile}, {k_bound}")
        total = r"$C_D = C_{D_i} + C_{D,p} + C_{D,w}$ (Korn)"
        (line,) = ax.plot(polar.cd, polar.cl, marker="o", markersize=4,
                          label=f"{total}, {k}")
        _bound_band(ax, polar.cd, polar.cd_bound, polar.cl, line.get_color(),
                    f"{total}, {k_bound}")
        _drag_trim(ax, polar, line.get_color())
        _annotate(ax, polar.trim_cd, polar.trim_cl,
                  f"{polar.label} trim, $C_L$ = {polar.trim_cl:.3f}:\n"
                  f"{k}: $C_D$ = {polar.trim_cd:.4f}, L/D = {polar.trim_cl / polar.trim_cd:.1f}\n"
                  f"{k_bound}: $C_D$ = {polar.trim_cd_bound:.4f}, "
                  f"L/D = {polar.trim_cl / polar.trim_cd_bound:.1f}",
                  offset=(16.0, -52.0))
        _bound_trim_legend(ax, polar.k_sweep_bound)
        _drag_axes(ax, conditions)
        _finish(fig, ax, title, cl_design, path)


def plot_drag_polars(polars: list[DragPolar], cl_design: float, title: str,
                     conditions: list[str], path: Path) -> None:
    """Draw the total drag polars of several wings, each with its trim point.

    Each wing has a solid line for the baseline sweep factor k_Λ, a dashed line for the bound
    and a shaded band between them. The trim annotations are short; the polar with the lowest
    trim drag is annotated to the upper left, the others to the lower right, so the boxes do
    not overlap.

    Args:
        polars: Drag polars, all with the same pair of sweep factors.
        cl_design: Design lift coefficient [-].
        title: Figure title.
        conditions: Lines of the condition box.
        path: Output PNG path.
    """
    lowest = min(range(len(polars)), key=lambda i: polars[i].trim_cd)
    with plt.rc_context(_STYLE):
        fig, ax = _new_figure()
        for i, polar in enumerate(polars):
            (line,) = ax.plot(polar.cd, polar.cl, marker=_MARKERS[i % len(_MARKERS)],
                              markersize=4.5,
                              label=f"{polar.label}, {_k_sweep_text(polar.k_sweep)}")
            _bound_band(ax, polar.cd, polar.cd_bound, polar.cl, line.get_color(),
                        f"{polar.label}, {_k_sweep_text(polar.k_sweep_bound)}")
            _drag_trim(ax, polar, line.get_color())
            _annotate(ax, polar.trim_cd, polar.trim_cl,
                      f"{polar.label} trim\n"
                      f"$C_D$ = {polar.trim_cd:.4f} / {polar.trim_cd_bound:.4f}\n"
                      f"L/D = {polar.trim_cl / polar.trim_cd:.1f} / "
                      f"{polar.trim_cl / polar.trim_cd_bound:.1f}",
                      offset=(-66.0, 48.0) if i == lowest else (16.0, -52.0))
        _bound_trim_legend(ax, polars[0].k_sweep_bound)
        trim_values = (f"Trim values: {_k_sweep_text(polars[0].k_sweep)} / "
                       f"{_k_sweep_text(polars[0].k_sweep_bound)}")
        _drag_axes(ax, [*conditions, trim_values])
        _finish(fig, ax, title, cl_design, path)


def _new_figure() -> tuple[Figure, Axes]:
    """A 16 cm wide figure with constrained layout, so an outside legend gets its own room."""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_CM / CM_PER_INCH,
                                    FIGURE_HEIGHT_CM / CM_PER_INCH), layout="constrained")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    return fig, ax


def _drag_trim(ax: Axes, polar: DragPolar, color: ColorType) -> None:
    """Trim star on the baseline drag polar and an open circle on the bound."""
    _trim_star(ax, polar.trim_cd, polar.trim_cl, color)
    ax.plot(polar.trim_cd_bound, polar.trim_cl, marker="o", markersize=_BOUND_TRIM_SIZE,
            color=color, markerfacecolor="white", linestyle="none", zorder=5)


def _bound_trim_legend(ax: Axes, k_sweep_bound: float) -> None:
    """Legend entry of the open trim circle on the bound (an empty line, nothing drawn)."""
    ax.plot([], [], marker="o", markersize=_BOUND_TRIM_SIZE, markerfacecolor="white",
            markeredgecolor="black", linestyle="none",
            label=f"Trim point, {_k_sweep_text(k_sweep_bound)}")


def _bound_band(ax: Axes, cd: NDArray[np.float64], cd_bound: NDArray[np.float64],
                cl: NDArray[np.float64], color: ColorType, label: str) -> None:
    """Dashed line of the bounding k_Λ and a shaded band to the baseline, in one colour."""
    ax.plot(cd_bound, cl, linestyle="--", linewidth=1.0, color=color, label=label)
    ax.fill_betweenx(cl, cd_bound, cd, color=color, alpha=0.15, linewidth=0.0)


def _k_sweep_text(k_sweep: float) -> str:
    """Legend text of a sweep factor, for example "$k_\\Lambda$ = 0.499"."""
    return rf"$k_\Lambda$ = {k_sweep:.3g}"


def _trim_star(ax: Axes, x: float, y: float, color: ColorType) -> None:
    """Trim-point star in the colour of its curve, drawn above the curves."""
    ax.plot(x, y, marker="*", markersize=_TRIM_MARKER_SIZE, color=color,
            markeredgecolor="black", linestyle="none", zorder=5)


def _drag_axes(ax: Axes, conditions: list[str]) -> None:
    """Axis labels, C_D axis from 0 and the condition box at the free lower-right corner."""
    ax.set_xlim(left=0.0)
    ax.set_xlabel(_CD_LABEL)
    ax.set_ylabel(_CL_LABEL)
    ax.add_artist(AnchoredText("\n".join(conditions), loc="lower right", frameon=True,
                               prop={"size": 8.5}))


def _annotate(ax: Axes, x: float, y: float, text: str, offset: tuple[float, float]) -> None:
    """Text with an arrow to a point, offset in points."""
    ax.annotate(text, xy=(x, y), xytext=offset, textcoords="offset points", fontsize=8.5,
                arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 0.8},
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.6"})


def _unique_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Points with duplicates (to 1e-6) removed, first occurrence kept."""
    unique: list[tuple[float, float]] = []
    for x, y in points:
        if not any(abs(x - u) < 1e-6 and abs(y - v) < 1e-6 for u, v in unique):
            unique.append((x, y))
    return unique


def _finish(fig: Figure, ax: Axes, title: str, cl_design: float, path: Path) -> None:
    """Design-C_L line in the legend, title, legend below the axes; save the PNG, close."""
    ax.axhline(cl_design, color="0.4", linestyle="--", linewidth=1.0)
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color="0.4", linestyle="--", linewidth=1.0),
                Line2D([], [], marker="*", markersize=_TRIM_MARKER_SIZE,
                       markerfacecolor="white", markeredgecolor="black", linestyle="none")]
    labels += [rf"$C_{{L,des}}$ = {cl_design:.3f} (cruise, W = $W_{{des}}$)", "Trim point"]
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
    ax.set_title(title)
    fig.savefig(path, dpi=PNG_DPI)
    plt.close(fig)
