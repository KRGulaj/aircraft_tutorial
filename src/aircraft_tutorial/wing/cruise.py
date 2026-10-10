# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Cruise design point: the flight condition, the lift coefficient the wing must give, and the
sweep-normal section condition at which the XFoil cruise polars are run.

    V = M·a,   q = ½·ρ·V²
    CL_des = m·g0 / (q·S)            steady level flight, lift = weight, no tail load
    μ from Sutherland's law,   Re(c) = ρ·V·c / μ

Simple sweep theory with the quarter-chord sweep Λ: the section sees the velocity and the chord
normal to the sweep line, V_n = V·cos Λ and c_n = c·cos Λ, so

    M_n = M·cos Λ,   Re_n(c) = ρ·V_n·c_n / μ = Re(c)·cos²Λ,   cl_n = cl / cos²Λ

The cruise polar is run at M_n and Re_n of the MAC; the strip profile drag scales each strip
from the MAC to its own chord. The critical pressure coefficient at M_n,

    Cp* = 2 / (γ·M_n²) · {[(2 + (γ − 1)·M_n²) / (γ + 1)]^(γ / (γ − 1)) − 1},

marks the polar points with sonic flow on the section (cp_min < Cp*), where XFoil's
Kármán-Tsien correction no longer holds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from aircraft_tutorial.common.atmosphere import GAMMA, isa, sutherland_viscosity
from aircraft_tutorial.common.units import G0


@dataclass(frozen=True)
class CruisePoint:
    """Cruise flight condition, design lift coefficient and sweep-normal section condition.

    Attributes:
        mach: Free-stream Mach number [-].
        altitude_m: Geopotential altitude [m].
        temperature_k: Static temperature [K].
        density_kg_m3: Density [kg/m^3].
        speed_of_sound_m_s: Speed of sound [m/s].
        dynamic_viscosity_pa_s: Dynamic viscosity [Pa·s].
        velocity_m_s: True airspeed [m/s].
        dynamic_pressure_pa: Dynamic pressure [Pa].
        cl_design: Lift coefficient for lift = weight [-].
        mac_m: Mean aerodynamic chord, the reference chord of the section polars [m].
        reynolds_mac: Reynolds number of the MAC, streamwise [-].
        sweep_deg: Quarter-chord sweep of the sweep-theory reduction [deg].
        mach_normal: Sweep-normal Mach number M·cos Λ [-].
        reynolds_normal_mac: Sweep-normal Reynolds number of the MAC, Re_MAC·cos²Λ [-].
        cl_design_normal: Design lift coefficient normal to the sweep, CL_des / cos²Λ [-].
        cp_crit_normal: Critical pressure coefficient at M_n [-].
    """

    mach: float
    altitude_m: float
    temperature_k: float
    density_kg_m3: float
    speed_of_sound_m_s: float
    dynamic_viscosity_pa_s: float
    velocity_m_s: float
    dynamic_pressure_pa: float
    cl_design: float
    mac_m: float
    reynolds_mac: float
    sweep_deg: float
    mach_normal: float
    reynolds_normal_mac: float
    cl_design_normal: float
    cp_crit_normal: float

    @property
    def reynolds_per_m(self) -> float:
        """Unit Reynolds number ρ·V / μ [1/m]."""
        return self.density_kg_m3 * self.velocity_m_s / self.dynamic_viscosity_pa_s

    def reynolds(self, chord_m: float) -> float:
        """Streamwise Reynolds number of a chord [-].

        Args:
            chord_m: Streamwise chord [m].
        """
        return self.reynolds_per_m * chord_m

    def reynolds_normal(self, chord_m: float) -> float:
        """Sweep-normal Reynolds number of a streamwise chord, Re(c)·cos²Λ [-].

        Args:
            chord_m: Streamwise chord [m].
        """
        return self.reynolds(chord_m) * math.cos(math.radians(self.sweep_deg)) ** 2


def critical_pressure_coefficient(mach: float) -> float:
    """Pressure coefficient at which the local flow reaches M = 1 (isentropic).

    Args:
        mach: Free-stream Mach number, in (0, 1] [-].

    Returns:
        Cp* [-]; 0 at M = 1, increasingly negative as M falls.

    Raises:
        ValueError: If the Mach number is outside (0, 1].
    """
    if not (0.0 < mach <= 1.0):
        raise ValueError(f"mach must be in (0, 1], got {mach}")
    g = GAMMA
    ratio = (2.0 + (g - 1.0) * mach**2) / (g + 1.0)
    return float(2.0 / (g * mach**2) * (ratio ** (g / (g - 1.0)) - 1.0))


def cruise_point(mach: float, altitude_m: float, mass_kg: float, area_m2: float,
                 mac_m: float, sweep_deg: float) -> CruisePoint:
    """Flight condition, design lift coefficient and section condition in steady level cruise.

    Args:
        mach: Free-stream Mach number [-].
        altitude_m: Geopotential altitude [m].
        mass_kg: Aircraft mass in cruise [kg].
        area_m2: Wing reference area [m^2].
        mac_m: Mean aerodynamic chord [m].
        sweep_deg: Quarter-chord sweep [deg].

    Returns:
        The cruise point.
    """
    air = isa(altitude_m)
    mu = sutherland_viscosity(air.temperature_k)
    velocity = mach * air.speed_of_sound_m_s
    q = 0.5 * air.density_kg_m3 * velocity**2
    cl_design = mass_kg * G0 / (q * area_m2)
    reynolds_mac = air.density_kg_m3 * velocity * mac_m / mu
    cos_sweep = math.cos(math.radians(sweep_deg))
    return CruisePoint(
        mach=mach,
        altitude_m=altitude_m,
        temperature_k=air.temperature_k,
        density_kg_m3=air.density_kg_m3,
        speed_of_sound_m_s=air.speed_of_sound_m_s,
        dynamic_viscosity_pa_s=mu,
        velocity_m_s=velocity,
        dynamic_pressure_pa=q,
        cl_design=cl_design,
        mac_m=mac_m,
        reynolds_mac=reynolds_mac,
        sweep_deg=sweep_deg,
        mach_normal=mach * cos_sweep,
        reynolds_normal_mac=reynolds_mac * cos_sweep**2,
        cl_design_normal=cl_design / cos_sweep**2,
        cp_crit_normal=critical_pressure_coefficient(mach * cos_sweep),
    )
