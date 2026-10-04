# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Unit conversions between SI and the imperial units of semi-empirical methods.

All factors are exact by definition (International Yard and Pound Agreement, 1959;
standard gravity, CGPM 1901).
"""

from __future__ import annotations

from typing import Final

KG_PER_LB: Final[float] = 0.45359237
M_PER_FT: Final[float] = 0.3048
M2_PER_FT2: Final[float] = M_PER_FT**2
G0: Final[float] = 9.80665
"""Standard gravity [m/s^2]."""
M_PER_S_PER_KMH: Final[float] = 1.0 / 3.6


def kg_to_lb(mass_kg: float) -> float:
    """Convert mass from kilogram to pound.

    Args:
        mass_kg: Mass [kg].

    Returns:
        Mass [lb].
    """
    return mass_kg / KG_PER_LB


def lb_to_kg(mass_lb: float) -> float:
    """Convert mass from pound to kilogram.

    Args:
        mass_lb: Mass [lb].

    Returns:
        Mass [kg].
    """
    return mass_lb * KG_PER_LB


def m2_to_ft2(area_m2: float) -> float:
    """Convert area from square metre to square foot.

    Args:
        area_m2: Area [m^2].

    Returns:
        Area [ft^2].
    """
    return area_m2 / M2_PER_FT2


def kmh_to_ms(speed_kmh: float) -> float:
    """Convert speed from kilometre per hour to metre per second.

    Args:
        speed_kmh: Speed [km/h].

    Returns:
        Speed [m/s].
    """
    return speed_kmh * M_PER_S_PER_KMH
