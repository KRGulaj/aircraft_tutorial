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
spread_windows_deg = [[-2.5, 3.5], [-6.5, 8.5]]

[sensitivity]
panel_nodes = [100, 160, 240, 320]
ncrit = [9.0, 10.0, 11.0]

[[airfoil]]
name = "NACA 2412"
file = "../airfoils/a.dat"
drag_bucket = false

[[airfoil]]
name = "NACA 66-410"
file = "../airfoils/b.dat"
drag_bucket = true

[[extra_run]]
label = "3d_input"
reynolds = 4.17e7
mach = 0.1
ncrit = 10.0
sweep_deg = 37.5
output = "../results/3d_input"
report_low_drag = false
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
    assert case.metrics.spread_windows_deg == ((-2.5, 3.5), (-6.5, 8.5))
    assert [a.drag_bucket for a in case.airfoils] == [False, True]
    assert case.extra_runs[0].label == "3d_input"
    assert case.extra_runs[0].conditions.reynolds == 4.17e7
    assert case.extra_runs[0].sweep_deg == 37.5
    assert case.extra_runs[0].output == (tmp_path / "results" / "3d_input").resolve()
    assert case.extra_runs[0].report_low_drag is False
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
        ("drag_bucket = true", 'drag_bucket = "yes"', "airfoil[1].drag_bucket"),
        ("[[-2.5, 3.5], [-6.5, 8.5]]", "[[3.5, -2.5]]", "metrics.spread_windows_deg"),
        ("reynolds = 4.17e7", "reynolds = 0.0", "extra_run[0].reynolds"),
        ('label = "3d_input"', "", "extra_run[0].label"),
        ("report_low_drag = false", "", "extra_run[0].report_low_drag"),
        ("sweep_deg = 37.5", "", "extra_run[0].sweep_deg"),
        ("sweep_deg = 37.5", "sweep_deg = -3.5", "extra_run[0].sweep_deg"),
        ("sweep_deg = 37.5", "sweep_deg = 90.0", "extra_run[0].sweep_deg"),
    ],
    ids=[
        "missing_key", "negative_re", "nan_mach", "supersonic", "too_many_nodes",
        "float_nodes", "bool_as_int", "alpha_range_without_zero", "zero_k",
        "empty_list", "too_few_nodes", "missing_file", "duplicate_name", "bucket_not_bool",
        "reversed_window", "extra_zero_re", "extra_no_label", "extra_no_low_drag_flag",
        "extra_no_sweep", "extra_negative_sweep", "extra_sweep_90",
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


def test_load_section_case_without_extra_run_is_empty(tmp_path: Path) -> None:
    """[[extra_run]] is optional."""
    text = VALID[: VALID.index("[[extra_run]]")]

    case = load_section_case(_write_case(tmp_path, text))

    assert case.extra_runs == ()


def test_load_section_case_unparsable_file_raises(tmp_path: Path) -> None:
    """A TOML syntax error is reported as CaseError, not as a raw parser error."""
    path = _write_case(tmp_path, "[conditions\nreynolds = 1")

    with pytest.raises(CaseError, match="cannot read"):
        load_section_case(path)
