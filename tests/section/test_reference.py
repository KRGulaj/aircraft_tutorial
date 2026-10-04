# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Tests for the reference-value loader and the camber-increment estimate."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from aircraft_tutorial.section.reference import (
    Reading,
    ReferenceData,
    ReferenceDataError,
    Readings,
    estimate_from_camber_increment,
    load_reference,
    thin_airfoil_a10_increment,
)

INPUTS = Path(__file__).resolve().parents[2] / "inputs" / "reference"

VALID = """
airfoil = "NACA 66-210"
source = "TR 824, p.251"
reynolds = 9.0e6
design_cl = 0.2
checked = false

[values.cl_max]
value = 1.28
uncertainty = 0.03
note = "top symbol"
"""


def _ref(name: str, cli: float, **values: tuple[float, float]) -> ReferenceData:
    """Reference data with the given (value, uncertainty) pairs."""
    readings = Readings(**{k: Reading(v, u, "test") for k, (v, u) in values.items()})
    return ReferenceData(name, f"src {name}", 9.0e6, cli, True, False, readings)


def test_load_reference_reads_header_and_value(tmp_path: Path) -> None:
    """The header and the one reading arrive; absent quantities are None."""
    path = tmp_path / "r.toml"
    path.write_text(VALID, encoding="utf-8")

    ref = load_reference(path)

    assert (ref.airfoil, ref.reynolds, ref.design_cl, ref.checked, ref.estimated) == (
        "NACA 66-210", 9.0e6, 0.2, False, False)
    assert ref.readings.cl_max == Reading(1.28, 0.03, "top symbol")
    assert ref.readings.cd_min is None


@pytest.mark.parametrize(
    ("old", "new", "fragment"),
    [
        ("uncertainty = 0.03", "", "needs value, uncertainty, note"),
        ("uncertainty = 0.03", "uncertainty = 0.0", "uncertainty > 0"),
        ("[values.cl_max]", "[values.cl_maximum]", "unknown quantities"),
        ("checked = false", 'checked = "no"', "checked"),
        ("reynolds = 9.0e6", "", "header key"),
    ],
    ids=["missing_uncertainty", "zero_uncertainty", "unknown_quantity", "checked_not_bool",
         "missing_reynolds"],
)
def test_load_reference_invalid_file_raises(tmp_path: Path, old: str, new: str,
                                            fragment: str) -> None:
    """Each invalid edit is rejected with a message naming the problem."""
    path = tmp_path / "r.toml"
    path.write_text(VALID.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(ReferenceDataError, match=fragment):
        load_reference(path)


def test_estimate_applies_copy_shift_and_increment_rules() -> None:
    """Base 66-210 (c_li 0.2), pair 215 → 415: copied, shifted by +0.2, incremented.

    α_0L = −1.37 + (−2.43 − (−1.51)) = −2.29, uncertainty √(0.21² + 0.17² + 0.23²) = 0.35511.
    """
    base = _ref("66-210", 0.2, alpha_0l_deg=(-1.37, 0.21), cd_min=(0.0031, 0.0002),
                low_drag_upper=(0.31, 0.03))
    low = _ref("66-215", 0.2, alpha_0l_deg=(-1.51, 0.17))
    high = _ref("66-415", 0.4, alpha_0l_deg=(-2.43, 0.23))

    est = estimate_from_camber_increment("66-410", base, low, high, 0.4)
    r = est.readings

    assert r.alpha_0l_deg is not None and r.cd_min is not None and r.low_drag_upper is not None
    assert r.alpha_0l_deg.value == pytest.approx(-2.29, abs=1e-12)
    assert r.alpha_0l_deg.uncertainty == pytest.approx(math.sqrt(0.21**2 + 0.17**2 + 0.23**2),
                                                       rel=1e-12)
    assert (r.cd_min.value, r.cd_min.uncertainty) == (0.0031, 0.0002)
    assert r.low_drag_upper.value == pytest.approx(0.51, abs=1e-12)
    assert est.estimated is True


def test_estimate_missing_calibration_value_gives_none() -> None:
    """cl_max present in the base only: no increment exists, so the estimate has none."""
    base = _ref("66-210", 0.2, cl_max=(1.28, 0.03))
    low = _ref("66-215", 0.2)
    high = _ref("66-415", 0.4)

    est = estimate_from_camber_increment("66-410", base, low, high, 0.4)

    assert est.readings.cl_max is None


def test_estimate_mismatched_camber_step_raises() -> None:
    """Pair step 0.2 → 0.4 cannot estimate a 0.2 → 0.6 target."""
    base, low, high = _ref("b", 0.2), _ref("l", 0.2), _ref("h", 0.4)

    with pytest.raises(ReferenceDataError, match="camber step"):
        estimate_from_camber_increment("t", base, low, high, 0.6)


def test_thin_airfoil_increment_for_cli_step_02() -> None:
    """Δα_0L = −0.2/(2π) rad = −1.823781°, Δcm = −0.05."""
    d_alpha, d_cm = thin_airfoil_a10_increment(0.2)

    assert d_alpha == pytest.approx(-0.2 / (2.0 * math.pi) * 180.0 / math.pi, rel=1e-12)
    assert d_alpha == pytest.approx(-1.823781, abs=1e-6)
    assert d_cm == pytest.approx(-0.05, abs=1e-15)


@pytest.mark.parametrize("name", ["naca2412", "naca66-210", "naca66_2-215", "naca66_2-415"])
def test_committed_reference_files_load(name: str) -> None:
    """Every committed reference file is valid."""
    ref = load_reference(INPUTS / f"{name}.toml")

    assert ref.readings.x_ac is not None
