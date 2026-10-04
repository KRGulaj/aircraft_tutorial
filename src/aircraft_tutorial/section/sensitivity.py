# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Numerical sensitivity of the section results: panel nodes and Ncrit, one at a time.

Each case changes one setting from the production run and keeps the rest. The panel count is
accepted if the result at the production count differs from the finest count by less than a set
relative tolerance for every checked quantity.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Final

from aircraft_tutorial.config.cases import FlowConditions, SectionCase, XfoilSettings

PANEL_TOLERANCE: Final[float] = 0.01
"""Accepted relative change, production nodes → finest nodes."""


@dataclass(frozen=True)
class SensitivityCase:
    """One run of the sensitivity check.

    Attributes:
        varied: "panel_nodes" or "ncrit".
        conditions: Flow condition of the run.
        xfoil: XFoil settings of the run.
    """

    varied: str
    conditions: FlowConditions
    xfoil: XfoilSettings


def sensitivity_cases(case: SectionCase) -> list[SensitivityCase]:
    """All one-at-a-time variations of the production run.

    Args:
        case: The run definition.

    Returns:
        Panel-node cases first, then Ncrit cases.
    """
    out = [SensitivityCase("panel_nodes", case.conditions,
                           dataclasses.replace(case.xfoil, panel_nodes=n))
           for n in case.sensitivity.panel_nodes]
    out += [SensitivityCase("ncrit", dataclasses.replace(case.conditions, ncrit=nc), case.xfoil)
            for nc in case.sensitivity.ncrit]
    return out


def relative_change(value: float, reference: float) -> float:
    """|value − reference| / |reference|.

    Args:
        value: Value at the production setting.
        reference: Value at the finest setting, non-zero.

    Returns:
        Relative change [-].

    Raises:
        ValueError: If the reference is zero or not finite.
    """
    if not (math.isfinite(reference) and reference != 0.0):
        raise ValueError(f"reference must be finite and non-zero, got {reference}")
    return abs(value - reference) / abs(reference)
