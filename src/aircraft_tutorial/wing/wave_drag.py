# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Wave drag of the wing: drag-divergence Mach from the Korn equation with simple sweep theory,
and the ADSEE empirical drag rise. XFoil and the vortex lattice model no shocks, so this term is
added on top.

Korn equation, swept (Λ = sweep of the chosen chord line, t/c streamwise, CL of the wing):

    M_dd = κ_A / cos Λ − (t/c) / cos²Λ − CL / (10·cos³Λ)

κ_A ≈ 0.87 for conventional sections, ≈ 0.95 for supercritical ones. ADSEE drag rise, 0.002 at
M = M_dd:

    M <  M_dd:  CD_wave = 0.002·[1 + 2.5·(M_dd − M) / 0.05]^−1
    M ≥  M_dd:  CD_wave = 0.002·[1 + (M − M_dd) / 0.05]^2.5
"""

from __future__ import annotations

import math


def korn_mdd(kappa_a: float, thickness_ratio: float, cl: float, sweep_deg: float) -> float:
    """Drag-divergence Mach number of a swept wing.

    Args:
        kappa_a: Technology factor κ_A [-].
        thickness_ratio: Streamwise t/c [-].
        cl: Wing lift coefficient [-].
        sweep_deg: Sweep of the chord line used [deg].

    Returns:
        M_dd [-]. Reduces to κ_A − t/c − CL/10 for zero sweep.
    """
    c = math.cos(math.radians(sweep_deg))
    return kappa_a / c - thickness_ratio / c**2 - cl / (10.0 * c**3)


def wave_drag(mach: float, m_dd: float) -> float:
    """ADSEE wave drag coefficient.

    Args:
        mach: Free-stream Mach number [-].
        m_dd: Drag-divergence Mach number [-].

    Returns:
        CD_wave [-]; 0.002 at M = M_dd.
    """
    if mach < m_dd:
        return 0.002 / (1.0 + 2.5 * (m_dd - mach) / 0.05)
    return float(0.002 * (1.0 + (mach - m_dd) / 0.05) ** 2.5)
