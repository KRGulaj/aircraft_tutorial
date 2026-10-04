# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Straight-tapered (trapezoidal) wing planform defined by span, area, root chord and
quarter-chord sweep, and the quantities that follow from them.

    λ = 2S / (b·c_r) − 1                       taper ratio, from S = (b/2)·c_r·(1 + λ)
    c_t = λ·c_r,   AR = b² / S
    MAC = (2/3)·c_r·(1 + λ + λ²) / (1 + λ),   y_MAC = (b/6)·(1 + 2λ) / (1 + λ)
    tan Λ_n = tan Λ_c/4 − (n − 1/4)·(c_r − c_t) / (b/2)     sweep of the n-chord line

The root leading edge is the origin; x points aft, y to the right wing tip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class PlanformError(ValueError):
    """Raised when the defining values do not give a valid trapezoid."""


@dataclass(frozen=True)
class TrapezoidalPlanform:
    """Trapezoidal planform.

    Attributes:
        span_m: Span, tip to tip [m].
        area_m2: Reference area, both halves [m^2].
        root_chord_m: Root chord [m].
        sweep_quarter_chord_deg: Quarter-chord sweep [deg].
    """

    span_m: float
    area_m2: float
    root_chord_m: float
    sweep_quarter_chord_deg: float

    def __post_init__(self) -> None:
        """Check that the values give a taper ratio in (0, 1]."""
        if not (self.span_m > 0.0 and self.area_m2 > 0.0 and self.root_chord_m > 0.0):
            raise PlanformError(f"span, area and root chord must be > 0, got {self.span_m}, "
                                f"{self.area_m2}, {self.root_chord_m}")
        if not (0.0 < self.taper <= 1.0):
            raise PlanformError(f"taper ratio 2S/(b·c_r) − 1 = {self.taper:.4f} is outside "
                                "(0, 1]: the area does not fit a trapezoid with this span and "
                                "root chord")

    @property
    def semi_span_m(self) -> float:
        """Half span [m]."""
        return self.span_m / 2.0

    @property
    def taper(self) -> float:
        """Taper ratio λ = c_t / c_r [-]."""
        return 2.0 * self.area_m2 / (self.span_m * self.root_chord_m) - 1.0

    @property
    def tip_chord_m(self) -> float:
        """Tip chord [m]."""
        return self.taper * self.root_chord_m

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio b² / S [-]."""
        return self.span_m**2 / self.area_m2

    @property
    def mac_m(self) -> float:
        """Mean aerodynamic chord [m]."""
        lam = self.taper
        return 2.0 / 3.0 * self.root_chord_m * (1.0 + lam + lam**2) / (1.0 + lam)

    @property
    def y_mac_m(self) -> float:
        """Spanwise station of the MAC [m]."""
        lam = self.taper
        return self.span_m / 6.0 * (1.0 + 2.0 * lam) / (1.0 + lam)

    @property
    def x_le_mac_m(self) -> float:
        """Leading-edge x of the MAC [m]."""
        return self.y_mac_m * math.tan(math.radians(self.sweep_deg(0.0)))

    def chord_at(self, y_m: float) -> float:
        """Local chord at a spanwise station [m].

        Args:
            y_m: Spanwise station, 0 to b/2 [m].
        """
        eta = abs(y_m) / self.semi_span_m
        return self.root_chord_m * (1.0 - eta * (1.0 - self.taper))

    def sweep_deg(self, chord_fraction: float) -> float:
        """Sweep of the line at a given chord fraction [deg].

        Args:
            chord_fraction: 0 for the leading edge, 0.25 for the quarter chord, 1 for the
                trailing edge [-].
        """
        tan_qc = math.tan(math.radians(self.sweep_quarter_chord_deg))
        slope = (self.root_chord_m - self.tip_chord_m) / self.semi_span_m
        return math.degrees(math.atan(tan_qc - (chord_fraction - 0.25) * slope))
