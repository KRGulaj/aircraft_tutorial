# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""ICAO standard atmosphere (ISA) up to 20 km, and the viscosity of air.

    troposphere (h ≤ 11 km):   T = T0 − L·h,   p = p0·(T / T0)^(g0 / (L·R))
    lower stratosphere:        T = 216.65 K,   p = p11·exp(−g0·(h − 11 km) / (R·T))
    ρ = p / (R·T),   a = √(γ·R·T)

h is the geopotential altitude. Sutherland's law: μ = μ0·(T / T0)^1.5·(T0 + S) / (T + S).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from aircraft_tutorial.common.units import G0

T0: Final[float] = 288.15
"""Sea-level temperature [K]."""
P0: Final[float] = 101_325.0
"""Sea-level pressure [Pa]."""
LAPSE_RATE: Final[float] = 0.0065
"""Tropospheric temperature lapse rate [K/m]."""
R_AIR: Final[float] = 287.05287
"""Specific gas constant of dry air [J/(kg·K)]."""
GAMMA: Final[float] = 1.4
"""Ratio of specific heats of air [-]."""
H_TROPOPAUSE: Final[float] = 11_000.0
"""Tropopause altitude [m]."""
H_MAX: Final[float] = 20_000.0
"""Upper end of the lower stratosphere, the validity limit of this module [m]."""

_SUTHERLAND_MU0: Final[float] = 1.716e-5
"""Reference viscosity at 273.15 K [Pa·s]."""
_SUTHERLAND_T0: Final[float] = 273.15
_SUTHERLAND_S: Final[float] = 110.4


class AtmosphereError(ValueError):
    """Raised for an altitude outside the modelled range."""


@dataclass(frozen=True)
class IsaState:
    """Air properties at one altitude.

    Attributes:
        altitude_m: Geopotential altitude [m].
        temperature_k: Static temperature [K].
        pressure_pa: Static pressure [Pa].
        density_kg_m3: Density [kg/m^3].
        speed_of_sound_m_s: Speed of sound [m/s].
    """

    altitude_m: float
    temperature_k: float
    pressure_pa: float
    density_kg_m3: float
    speed_of_sound_m_s: float


def isa(altitude_m: float) -> IsaState:
    """Air properties of the standard atmosphere.

    Args:
        altitude_m: Geopotential altitude [m], 0 to 20 000.

    Returns:
        Temperature, pressure, density and speed of sound.

    Raises:
        AtmosphereError: If the altitude is outside [0, 20 000] m.
    """
    if not (0.0 <= altitude_m <= H_MAX):
        raise AtmosphereError(f"altitude {altitude_m} m is outside [0, {H_MAX}] m")
    exponent = G0 / (LAPSE_RATE * R_AIR)
    if altitude_m <= H_TROPOPAUSE:
        t = T0 - LAPSE_RATE * altitude_m
        p = P0 * (t / T0) ** exponent
    else:
        t = T0 - LAPSE_RATE * H_TROPOPAUSE
        p11 = P0 * (t / T0) ** exponent
        p = p11 * math.exp(-G0 * (altitude_m - H_TROPOPAUSE) / (R_AIR * t))
    return IsaState(
        altitude_m=altitude_m,
        temperature_k=t,
        pressure_pa=p,
        density_kg_m3=p / (R_AIR * t),
        speed_of_sound_m_s=math.sqrt(GAMMA * R_AIR * t),
    )


def sutherland_viscosity(temperature_k: float) -> float:
    """Dynamic viscosity of air from Sutherland's law.

    Args:
        temperature_k: Static temperature [K].

    Returns:
        Dynamic viscosity [Pa·s].
    """
    return float(_SUTHERLAND_MU0 * (temperature_k / _SUTHERLAND_T0) ** 1.5
                 * (_SUTHERLAND_T0 + _SUTHERLAND_S) / (temperature_k + _SUTHERLAND_S))
