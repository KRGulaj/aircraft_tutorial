# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Tests for the section metrics hand-off."""

from __future__ import annotations

from pathlib import Path

import pytest

from aircraft_tutorial.contracts.section_metrics import SectionMetricsError, read_section_metrics

COMMITTED = Path(__file__).resolve().parents[2] / "results" / "3d_input" / "naca2412_metrics.csv"


def test_read_section_metrics_returns_values_and_units(tmp_path: Path) -> None:
    """Values and units arrive by quantity name."""
    path = tmp_path / "m.csv"
    path.write_text("quantity,value,unit\nalpha_0l_deg,-2.14031,deg\na0_per_rad,6.61073,1/rad\n",
                    encoding="utf-8")

    data = read_section_metrics(path)

    assert (data.value("alpha_0l_deg"), data.units["a0_per_rad"]) == (-2.14031, "1/rad")


def test_missing_quantity_raises_naming_it(tmp_path: Path) -> None:
    """Asking for a quantity the file does not hold names it in the error."""
    path = tmp_path / "m.csv"
    path.write_text("quantity,value,unit\ncl_max,2.1,-\n", encoding="utf-8")

    with pytest.raises(SectionMetricsError, match="alpha_0l_deg"):
        read_section_metrics(path).value("alpha_0l_deg")


def test_wrong_header_raises(tmp_path: Path) -> None:
    """A file that is not a metrics table is rejected."""
    path = tmp_path / "m.csv"
    path.write_text("alpha_deg,cl\n0.0,0.2\n", encoding="utf-8")

    with pytest.raises(SectionMetricsError, match="header"):
        read_section_metrics(path)


@pytest.mark.skipif(not COMMITTED.is_file(), reason="3D-input metrics not generated")
def test_committed_3d_input_metrics_hold_the_zero_lift_angle() -> None:
    """The 2D stage's file for NACA 2412 has a finite α_0l."""
    assert -5.0 < read_section_metrics(COMMITTED).value("alpha_0l_deg") < 0.0
