# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-11
"""Span efficiency of a wing from its spanwise lift distribution (Munk, lifting-line form).

For a planar wake, the induced drag depends only on the spanwise circulation distribution, not on
sweep or chordwise detail (Munk's stagger theorem). With y = (b/2)·cos θ, a symmetric loading is
an odd sine series,

    Γ(θ) ∝ c·cl = Σ A_n·sin(nθ),   n = 1, 3, 5, …

and

    e = 1 / (1 + δ),   δ = Σ_{n ≥ 3} n·(A_n / A_1)²,   CDi = CL² / (π·A·e).

The coefficients come from a least-squares fit of the strip values c·cl at their stations. This
gives an induced drag that is independent of how the solver evaluates the far field, so it
checks the VSPAERO far-field CDi.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SpanEfficiencyError(ValueError):
    """Raised when the loading cannot be fitted."""


def loading_span_efficiency(eta: NDArray[np.float64], cl_c: NDArray[np.float64],
                            terms: int) -> float:
    """Span efficiency e of a symmetric loading from a fit of odd sine terms.

    Args:
        eta: Strip stations 2y/b of one half-wing, in [0, 1] [-].
        cl_c: Strip loading c·cl at those stations, any consistent unit.
        terms: Number of odd sine terms n = 1, 3, …, 2·terms − 1.

    Returns:
        e [-].

    Raises:
        SpanEfficiencyError: If there are fewer strips than terms, a station is outside [0, 1],
            or the fitted A_1 is not positive.
    """
    eta = np.asarray(eta, dtype=np.float64)
    load = np.asarray(cl_c, dtype=np.float64)
    if terms < 1 or eta.size < terms:
        raise SpanEfficiencyError(f"{eta.size} strips cannot fit {terms} sine terms")
    if bool(np.any((eta < 0.0) | (eta > 1.0))):
        raise SpanEfficiencyError(f"stations must be in [0, 1], got {eta.min()} … {eta.max()}")
    n = np.arange(1, 2 * terms, 2, dtype=np.float64)
    basis = np.sin(np.outer(np.arccos(eta), n))
    coef, *_ = np.linalg.lstsq(basis, load, rcond=None)
    if coef[0] <= 0.0:
        raise SpanEfficiencyError(f"fitted A_1 = {coef[0]} is not positive")
    delta = float(np.sum(n[1:] * (coef[1:] / coef[0]) ** 2))
    return 1.0 / (1.0 + delta)
