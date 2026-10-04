# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Tests for the section-analysis TOML loader."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aircraft_tutorial.config.cases import CaseError, load_section_case

VALID = """
[conditions]
reynolds = 8.9e6
mach = 0.15
ncrit = 10.0

[xfoil]
panel_nodes = 160
max_iter = 200
alpha_low_deg = -8.5
alpha_high_deg = 22.0
alpha_step_deg = 0.25
stop_after_failures = 7

[metrics]
linear_alpha_min_deg = -4.0
linear_alpha_max_deg = 6.0
low_drag_factor = 0.13

[sensitivity]
panel_nodes = [100, 160, 240, 320]
ncrit = [9.0, 10.0, 11.0]

[[airfoil]]
name = "NACA 2412"
file = "../airfoils/a.dat"

[[airfoil]]
name = "NACA 66-410"
file = "../airfoils/b.dat"
"""


def _write_case(tmp_path: Path, text: str) -> Path:
    """Write the TOML in tmp/inputs and two empty coordinate files in tmp/airfoils."""
    (tmp_path / "inputs").mkdir()
    (tmp_path / "airfoils").mkdir()
    (tmp_path / "airfoils" / "a.dat").write_text("x", encoding="utf-8")
    (tmp_path / "airfoils" / "b.dat").write_text("x", encoding="utf-8")
    path = tmp_path / "inputs" / "section.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_section_case_valid_file_returns_every_value(tmp_path: Path) -> None:
    """Every value of the file arrives in the dataclasses, paths resolve against the TOML dir."""
    case = load_section_case(_write_case(tmp_path, VALID))

    assert (case.conditions.reynolds, case.conditions.mach, case.conditions.ncrit) == (
        8.9e6, 0.15, 10.0)
    assert (case.xfoil.panel_nodes, case.xfoil.alpha_low_deg, case.xfoil.stop_after_failures) == (
        160, -8.5, 7)
    assert case.metrics.low_drag_factor == 0.13
    assert case.sensitivity.panel_nodes == (100, 160, 240, 320)
    assert [a.name for a in case.airfoils] == ["NACA 2412", "NACA 66-410"]
    assert case.airfoils[1].path == (tmp_path / "airfoils" / "b.dat").resolve()


@pytest.mark.parametrize(
    ("old", "new", "fragment"),
    [
        ("reynolds = 8.9e6", "", "conditions.reynolds"),
        ("reynolds = 8.9e6", "reynolds = -8.9e6", "conditions.reynolds"),
        ("mach = 0.15", "mach = nan", "conditions.mach"),
        ("mach = 0.15", "mach = 1.2", "conditions.mach"),
        ("panel_nodes = 160", "panel_nodes = 700", "xfoil.panel_nodes"),
        ("panel_nodes = 160", "panel_nodes = 160.5", "xfoil.panel_nodes"),
        ("max_iter = 200", "max_iter = true", "xfoil.max_iter"),
        ("alpha_low_deg = -8.5", "alpha_low_deg = 1.5", "alpha_low_deg"),
        ("low_drag_factor = 0.13", "low_drag_factor = 0.0", "metrics.low_drag_factor"),
        ("ncrit = [9.0, 10.0, 11.0]", "ncrit = []", "sensitivity.ncrit"),
        ("[100, 160, 240, 320]", "[100, 30]", "sensitivity.panel_nodes"),
        ('file = "../airfoils/b.dat"', 'file = "../airfoils/missing.dat"', "airfoil[1].file"),
        ('name = "NACA 66-410"', 'name = "NACA 2412"', "unique"),
    ],
    ids=[
        "missing_key", "negative_re", "nan_mach", "supersonic", "too_many_nodes",
        "float_nodes", "bool_as_int", "alpha_range_without_zero", "zero_k",
        "empty_list", "too_few_nodes", "missing_file", "duplicate_name",
    ],
)
def test_load_section_case_invalid_value_raises_naming_key(
    tmp_path: Path, old: str, new: str, fragment: str
) -> None:
    """Each invalid edit raises CaseError, and the message names the offending key."""
    assert old in VALID
    path = _write_case(tmp_path, VALID.replace(old, new, 1))

    with pytest.raises(CaseError, match=re.escape(fragment)):
        load_section_case(path)


def test_load_section_case_unparsable_file_raises(tmp_path: Path) -> None:
    """A TOML syntax error is reported as CaseError, not as a raw parser error."""
    path = _write_case(tmp_path, "[conditions\nreynolds = 1")

    with pytest.raises(CaseError, match="cannot read"):
        load_section_case(path)
