# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Tests for the sensitivity case list and the relative-change check."""

from __future__ import annotations

from pathlib import Path

import pytest

from aircraft_tutorial.config.cases import load_section_case
from aircraft_tutorial.section.sensitivity import relative_change, sensitivity_cases

INPUT = Path(__file__).resolve().parents[2] / "inputs" / "section.toml"


def test_sensitivity_cases_vary_one_setting_at_a_time() -> None:
    """5 node cases at Ncrit 10, then 3 Ncrit cases at the production 240 nodes."""
    case = load_section_case(INPUT)

    cases = sensitivity_cases(case)

    assert [(c.varied, c.xfoil.panel_nodes, c.conditions.ncrit) for c in cases] == [
        ("panel_nodes", 100, 10.0), ("panel_nodes", 160, 10.0), ("panel_nodes", 240, 10.0),
        ("panel_nodes", 320, 10.0), ("panel_nodes", 400, 10.0), ("ncrit", 240, 9.0),
        ("ncrit", 240, 10.0), ("ncrit", 240, 11.0)]
    assert all(c.conditions.reynolds == 8.9e6 for c in cases)


def test_relative_change_matches_hand_value() -> None:
    """|0.1137 − 0.1129| / 0.1129 = 0.0070860."""
    assert relative_change(0.1137, 0.1129) == pytest.approx(0.0008 / 0.1129, rel=1e-12)


def test_relative_change_negative_reference_uses_magnitude() -> None:
    """|−2.13 − (−2.17)| / 2.17 = 0.018433."""
    assert relative_change(-2.13, -2.17) == pytest.approx(0.04 / 2.17, rel=1e-12)


def test_relative_change_zero_reference_raises() -> None:
    """A zero reference has no relative change."""
    with pytest.raises(ValueError):
        relative_change(0.3, 0.0)
