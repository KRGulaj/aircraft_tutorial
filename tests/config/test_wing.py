# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Tests for the wing-analysis TOML loader and the committed task 1c run definition."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aircraft_tutorial.config.cases import CaseError
from aircraft_tutorial.config.wing import load_wing_case

INPUT = Path(__file__).resolve().parents[2] / "inputs" / "wing.toml"

VALID = """
[aircraft]
design_mass_kg = 380000.0

[planform]
span_m = 68.40
area_m2 = 554.0
root_chord_m = 14.63
sweep_quarter_chord_deg = 37.5
dihedral_deg = 0.0

[cruise]
mach = 0.855
altitude_m = 10668.0

[twist]
root_incidence_deg = 4.0
max_twist_deg = 6.0
cl_tolerance = 1.0e-4
max_iter = 6

[vlm]
segments = 12
segment_tess = 3
chord_tess = 33
wake_iter = 5
n_cpu = 4
alpha_start_deg = -4.0
alpha_end_deg = 10.0
alpha_npts = 15

[drag]
korn_sweep_chord_fraction = 0.5
sweep_drag_mode = "friction"
sensitivity_sweep_drag_mode = "cos3"

[[wing]]
name = "WING-1"
airfoil = "../airfoils/a.dat"
polar = "../results/a.csv"
kappa_a = 0.87

[[wing]]
name = "WING-2"
airfoil = "../airfoils/b.dat"
polar = "../results/b.csv"
kappa_a = 0.95
"""


def _write_case(tmp_path: Path, text: str) -> Path:
    """Write the TOML in tmp/inputs and two coordinate files in tmp/airfoils."""
    (tmp_path / "inputs").mkdir()
    (tmp_path / "airfoils").mkdir()
    (tmp_path / "airfoils" / "a.dat").write_text("x", encoding="utf-8")
    (tmp_path / "airfoils" / "b.dat").write_text("x", encoding="utf-8")
    path = tmp_path / "inputs" / "wing.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_wing_case_valid_file_returns_every_value(tmp_path: Path) -> None:
    """Values arrive in the dataclasses; airfoil paths resolve against the TOML directory."""
    case = load_wing_case(_write_case(tmp_path, VALID))

    assert case.aircraft.design_mass_kg == 380_000.0
    assert (case.planform.span_m, case.planform.root_chord_m) == (68.40, 14.63)
    assert (case.cruise.mach, case.twist.root_incidence_deg, case.twist.max_iter) == (
        0.855, 4.0, 6)
    assert (case.vlm.segments, case.vlm.segment_tess, case.vlm.alpha_npts) == (12, 3, 15)
    assert [w.name for w in case.wings] == ["WING-1", "WING-2"]
    assert case.wings[1].airfoil == (tmp_path / "airfoils" / "b.dat").resolve()
    assert case.wings[1].polar == (tmp_path / "results" / "b.csv").resolve()
    assert (case.wings[1].kappa_a, case.drag.sweep_drag_mode) == (0.95, "friction")


@pytest.mark.parametrize(
    ("old", "new", "fragment"),
    [
        ("design_mass_kg = 380000.0", "", "aircraft.design_mass_kg"),
        ("span_m = 68.40", "span_m = 0.0", "planform.span_m"),
        ("sweep_quarter_chord_deg = 37.5", "sweep_quarter_chord_deg = 90.0", "sweep"),
        ("mach = 0.855", "mach = 1.2", "cruise.mach"),
        ("altitude_m = 10668.0", "altitude_m = 25000.0", "cruise.altitude_m"),
        ("max_twist_deg = 6.0", "max_twist_deg = 0.0", "twist.max_twist_deg"),
        ("max_iter = 6", "max_iter = 6.5", "twist.max_iter"),
        ("segments = 12", "segments = true", "vlm.segments"),
        ("alpha_start_deg = -4.0", "alpha_start_deg = 1.0", "alpha_start_deg"),
        ("alpha_npts = 15", "alpha_npts = 14", "0 deg must be a grid point"),
        ('name = "WING-2"', 'name = "WING 2"', "wing[1].name"),
        ('name = "WING-2"', 'name = "WING-1"', "unique"),
        ('airfoil = "../airfoils/b.dat"', 'airfoil = "../airfoils/c.dat"', "wing[1].airfoil"),
        ("kappa_a = 0.95", "kappa_a = 1.5", "wing[1].kappa_a"),
        ('polar = "../results/b.csv"', "", "wing[1].polar"),
        ('sweep_drag_mode = "friction"', 'sweep_drag_mode = "cos2"', "drag.sweep_drag_mode"),
        ("korn_sweep_chord_fraction = 0.5", "korn_sweep_chord_fraction = 1.5",
         "drag.korn_sweep_chord_fraction"),
    ],
    ids=[
        "missing_key", "zero_span", "sweep_90", "supersonic", "above_isa_range", "zero_max_twist",
        "float_iter", "bool_as_int", "alpha_range_without_zero", "zero_off_grid",
        "space_in_name", "duplicate_name", "missing_airfoil", "kappa_out_of_range",
        "missing_polar_key", "unknown_mode", "korn_fraction_above_one",
    ],
)
def test_load_wing_case_invalid_value_raises_naming_key(
    tmp_path: Path, old: str, new: str, fragment: str
) -> None:
    """Each invalid edit raises CaseError, and the message names the offending key."""
    assert old in VALID
    path = _write_case(tmp_path, VALID.replace(old, new, 1))

    with pytest.raises(CaseError, match=re.escape(fragment)):
        load_wing_case(path)


def test_wing_toml_holds_brief_values() -> None:
    """The committed file holds the B747-8 brief values and the two task 1c wings."""
    case = load_wing_case(INPUT)

    assert (case.planform.span_m, case.planform.area_m2, case.planform.root_chord_m,
            case.planform.sweep_quarter_chord_deg) == (68.40, 554.0, 14.63, 37.5)
    assert (case.cruise.mach, case.cruise.altitude_m) == (0.855, 10_668.0)
    assert (case.twist.root_incidence_deg, case.twist.max_twist_deg) == (4.0, 6.0)
    assert [(w.name, w.airfoil.name) for w in case.wings] == [
        ("WING-1", "NACA2412.dat"), ("WING-2", "NACA66-410_gen.dat")]
