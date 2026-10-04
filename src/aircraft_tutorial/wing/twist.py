# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Size the linear twist so that the wing gives the design lift coefficient at zero body angle.

The twist ε is the tip section angle relative to the root section (negative = wash-out). With
the root incidence fixed, the lift coefficient at zero body angle, CL(ε), is found by the
secant method:

    ε_{k+1} = ε_k + (CL_des − CL(ε_k))·(ε_k − ε_{k−1}) / (CL(ε_k) − CL(ε_{k−1}))

starting from ε = 0 and ε = −ε_max. In a vortex-lattice model CL is linear in ε, so the first
secant step lands on the answer; the run at that ε is the check that it does.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


class TwistError(RuntimeError):
    """Raised when no twist within the allowed range gives the design lift coefficient."""


@dataclass(frozen=True)
class TwistSizing:
    """Result of the twist sizing.

    Attributes:
        twist_deg: Tip twist relative to the root [deg].
        cl: Lift coefficient at zero body angle with that twist [-].
        history: Every evaluated (twist [deg], CL [-]) pair, in evaluation order.
    """

    twist_deg: float
    cl: float
    history: tuple[tuple[float, float], ...]


def size_twist(cl_at_twist: Callable[[float], float], cl_target: float, *,
               max_twist_deg: float, cl_tolerance: float, max_iter: int) -> TwistSizing:
    """Find the twist that gives the target lift coefficient at zero body angle.

    Args:
        cl_at_twist: Lift coefficient at zero body angle as a function of twist [deg] → [-].
        cl_target: Design lift coefficient [-].
        max_twist_deg: Largest allowed |twist| [deg].
        cl_tolerance: Convergence limit on |CL − CL_target| [-].
        max_iter: Secant steps after the two starting evaluations.

    Returns:
        The converged twist with the evaluation history.

    Raises:
        TwistError: If CL does not depend on the twist, the required twist exceeds the limit, or
            the iteration does not converge.
    """
    history: list[tuple[float, float]] = []

    def evaluate(twist: float) -> float:
        cl = float(cl_at_twist(twist))
        history.append((twist, cl))
        return cl

    t0, t1 = 0.0, -max_twist_deg
    c0 = evaluate(t0)
    if abs(c0 - cl_target) <= cl_tolerance:
        return TwistSizing(t0, c0, tuple(history))
    c1 = evaluate(t1)
    for _ in range(max_iter):
        if abs(c1 - cl_target) <= cl_tolerance:
            return TwistSizing(t1, c1, tuple(history))
        if c1 == c0:
            raise TwistError(f"CL = {c1} does not change with twist; check the model")
        t2 = t1 + (cl_target - c1) * (t1 - t0) / (c1 - c0)
        if abs(t2) > max_twist_deg:
            raise TwistError(f"CL_des = {cl_target:.4f} needs a twist of {t2:.3f} deg, beyond "
                             f"the allowed ±{max_twist_deg} deg (history {history})")
        t0, c0 = t1, c1
        t1, c1 = t2, evaluate(t2)
    if abs(c1 - cl_target) <= cl_tolerance:
        return TwistSizing(t1, c1, tuple(history))
    raise TwistError(f"no convergence to |ΔCL| <= {cl_tolerance} in {max_iter} secant steps "
                     f"(history {history})")
