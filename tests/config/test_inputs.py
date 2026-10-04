# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""The committed run definition loads and holds the agreed task 1a condition."""

from __future__ import annotations

from pathlib import Path

from aircraft_tutorial.config.cases import load_section_case

INPUT = Path(__file__).resolve().parents[2] / "inputs" / "section.toml"


def test_section_toml_holds_agreed_condition() -> None:
    """Re 8.9e6, M 0.15, Ncrit 10, 240 nodes, airfoils 2412 and 66-410."""
    case = load_section_case(INPUT)

    assert (case.conditions.reynolds, case.conditions.mach, case.conditions.ncrit) == (
        8.9e6, 0.15, 10.0)
    assert case.xfoil.panel_nodes == 240
    assert [a.name for a in case.airfoils] == ["NACA 2412", "NACA 66-410"]
