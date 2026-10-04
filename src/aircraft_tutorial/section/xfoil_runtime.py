# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""XFoil runtime: process setup, one configured session per run, and the polar sweep.

The XFoil Python binding is the patched xfoil-python build from
system_design/externals/xfoil-python. Its `XFoil.a(alpha)` returns
(cl, cd, cm, cp_min, diverged, rms_bl): `diverged` is True only if the boundary-layer Newton
solve stopped on a NaN, and `rms_bl` is the final RMS residual of that system. A point that ran
out of iterations is therefore distinguishable from one that blew up.

Every XFoil call in this package goes through this module, so the process setup below always
runs before the binding is imported.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import math
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from aircraft_tutorial.config.cases import FlowConditions, XfoilSettings
from aircraft_tutorial.geometry.airfoil import Airfoil
from aircraft_tutorial.section.polar import Polar, RunInfo

logger = logging.getLogger(__name__)

MINGW_BIN: Final[Path] = Path(r"C:\msys64\mingw64\bin")
"""MinGW runtime (libgfortran, libgcc_s_seh, libwinpthread, libquadmath) that libxfoil.dll
links against."""

_state: dict[str, bool] = {"ready": False}


class XfoilSetupError(RuntimeError):
    """Raised when the XFoil binding cannot be prepared for use."""


def ensure_xfoil_ready(mingw_bin: Path = MINGW_BIN) -> None:
    """Prepare the process for the XFoil binding. Safe to call more than once.

    1. Register the MinGW runtime directory with the Windows DLL loader. Since Python 3.8,
       ctypes does not search PATH for the dependencies of a loaded DLL, so the directory must be
       added with `os.add_dll_directory` before `import xfoil`.
    2. Declare `kernel32.FreeLibrary(HMODULE)` with a pointer argument. The binding's
       `XFoil.__del__` calls it without argtypes, ctypes then passes the 64-bit handle as a
       32-bit int, the call fails, and every instance leaks its DLL handle and its temporary DLL
       copy.

    Args:
        mingw_bin: Directory of the MinGW runtime DLLs.

    Raises:
        XfoilSetupError: If the directory does not exist.
    """
    if not mingw_bin.is_dir():
        raise XfoilSetupError(f"MinGW runtime directory not found: {mingw_bin}. libxfoil.dll "
                              "needs its gfortran/gcc runtime DLLs from there.")
    if _state["ready"]:
        return
    os.add_dll_directory(str(mingw_bin))
    ctypes.windll.kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]  # type: ignore[attr-defined,unused-ignore]
    ctypes.windll.kernel32.FreeLibrary.restype = ctypes.c_int  # type: ignore[attr-defined,unused-ignore]
    _state["ready"] = True


@contextlib.contextmanager
def xfoil_session(airfoil: Airfoil, conditions: FlowConditions,
                  settings: XfoilSettings) -> Iterator[Any]:
    """Yield a fresh XFoil instance configured for one run.

    A new instance per run keeps boundary-layer state from one airfoil or condition out of the
    next. The geometry is repanelled with XFoil's `PANE` distribution to `settings.panel_nodes`
    nodes, so the result does not depend on the point spacing of the coordinate file.

    Args:
        airfoil: Section contour, unit chord, Selig order.
        conditions: Reynolds number, Mach number, Ncrit.
        settings: Panel nodes and iteration limit.

    Yields:
        The configured `xfoil.XFoil` instance.
    """
    ensure_xfoil_ready()
    # Imported here, not at module level: the import must follow ensure_xfoil_ready().
    from xfoil import XFoil
    from xfoil.model import Airfoil as XfoilAirfoil

    xf = XFoil()
    try:
        xf.print = False
        xf.airfoil = XfoilAirfoil(np.array(airfoil.x), np.array(airfoil.y))
        xf.repanel(n_nodes=settings.panel_nodes)
        xf.Re = conditions.reynolds
        xf.M = conditions.mach
        xf.n_crit = conditions.ncrit
        xf.max_iter = settings.max_iter
        yield xf
    finally:
        del xf


def alpha_legs(settings: XfoilSettings) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Angles of the upward and downward sweep legs, both starting at 0 deg.

    Angles are integer multiples of the step, so no rounding drift accumulates.

    Args:
        settings: Sweep range and step.

    Returns:
        (upward leg 0 → alpha_high, downward leg 0 → alpha_low) [deg].
    """
    step = settings.alpha_step_deg
    n_up = math.floor(settings.alpha_high_deg / step + 1e-9)
    n_down = math.floor(-settings.alpha_low_deg / step + 1e-9)
    return step * np.arange(n_up + 1, dtype=np.float64), -step * np.arange(n_down + 1,
                                                                           dtype=np.float64)


def run_leg(xf: Any, alphas: NDArray[np.float64], stop_after_failures: int
            ) -> list[tuple[float, float, float, float, float, bool, bool, float]]:
    """Run one leg, one angle at a time, in the given order.

    Each converged point warm-starts the next. After a failed point the boundary layer is reset,
    so one bad solution does not seed the next. The leg ends after `stop_after_failures`
    consecutive failures: past that point the sweep is beyond breakdown.

    Args:
        xf: Configured XFoil instance.
        alphas: Angles in sweep order [deg].
        stop_after_failures: Consecutive failures that end the leg.

    Returns:
        Rows (alpha, cl, cd, cm, cp_min, converged, diverged, rms_bl).
    """
    rows: list[tuple[float, float, float, float, float, bool, bool, float]] = []
    failures = 0
    for alpha in alphas:
        cl, cd, cm, cp_min, diverged, rms_bl = xf.a(float(alpha))
        converged = bool(np.isfinite(cl))
        rows.append((float(alpha), float(cl), float(cd), float(cm), float(cp_min), converged,
                     bool(diverged), float(rms_bl)))
        logger.debug("alpha %.2f: converged=%s cl=%.4f rms_bl=%.3g", alpha, converged, cl, rms_bl)
        if converged:
            failures = 0
            continue
        failures += 1
        xf.reset_bls()
        if failures >= stop_after_failures:
            logger.info("leg stopped at alpha %.2f after %d consecutive failures", alpha, failures)
            break
    return rows


def run_polar(airfoil: Airfoil, conditions: FlowConditions, settings: XfoilSettings) -> Polar:
    """Full polar as two warm-started legs from 0 deg, upward and downward, merged.

    A cold start at a large angle often diverges for several points and can use up the failure
    budget before a convergent point is reached. Starting both legs at 0 deg avoids that, so the
    failure that ends a leg is the real breakdown near stall. Each leg uses its own session, so a
    post-stall excursion on one leg cannot affect the other.

    Args:
        airfoil: Section contour, unit chord, Selig order.
        conditions: Reynolds number, Mach number, Ncrit.
        settings: Sweep and numerical settings.

    Returns:
        The merged polar, ascending in angle of attack.
    """
    up, down = alpha_legs(settings)
    logger.info("%s: Re=%.4g M=%.3f Ncrit=%.1f nodes=%d", airfoil.name, conditions.reynolds,
                conditions.mach, conditions.ncrit, settings.panel_nodes)
    with xfoil_session(airfoil, conditions, settings) as xf:
        rows_up = run_leg(xf, up, settings.stop_after_failures)
    with xfoil_session(airfoil, conditions, settings) as xf:
        rows_down = run_leg(xf, down, settings.stop_after_failures)[1:]  # 0 deg is in rows_up

    rows = sorted(rows_down + rows_up, key=lambda r: r[0])
    cols = list(zip(*rows, strict=True))
    info = RunInfo(airfoil=airfoil.name, reynolds=conditions.reynolds, mach=conditions.mach,
                   ncrit=conditions.ncrit, panel_nodes=settings.panel_nodes)
    return Polar(
        info=info,
        alpha_deg=np.array(cols[0], dtype=np.float64),
        cl=np.array(cols[1], dtype=np.float64),
        cd=np.array(cols[2], dtype=np.float64),
        cm=np.array(cols[3], dtype=np.float64),
        cp_min=np.array(cols[4], dtype=np.float64),
        converged=np.array(cols[5], dtype=np.bool_),
        diverged=np.array(cols[6], dtype=np.bool_),
        rms_bl=np.array(cols[7], dtype=np.float64),
    )
