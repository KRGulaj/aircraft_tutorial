# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""The 3D stage reads exactly what the 2D stage writes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aircraft_tutorial.contracts.section_polar import SectionPolarError, read_section_polar
from aircraft_tutorial.section.polar import Polar, RunInfo, write_csv


def _polar() -> Polar:
    nan = float("nan")
    return Polar(
        info=RunInfo(airfoil="NACA 2412", reynolds=4.1646e7, mach=0.6783, ncrit=9.0,
                     panel_nodes=160),
        alpha_deg=np.array([-1.0, 0.0, 1.0, 2.0]),
        cl=np.array([0.1, 0.2, 0.31, nan]),
        cd=np.array([0.006, 0.0061, 0.0063, nan]),
        cm=np.array([-0.05, -0.051, -0.052, nan]),
        cp_min=np.array([-0.5, -0.6, -0.7, nan]),
        converged=np.array([True, True, True, False]),
        diverged=np.array([False, False, False, True]),
        rms_bl=np.array([1e-5, 1e-5, 2e-5, 3.0]),
    )


def test_read_section_polar_round_trips_section_writer(tmp_path: Path) -> None:
    """Header, values, NaNs and the converged flags survive write_csv → read_section_polar."""
    path = tmp_path / "polar.csv"
    polar = _polar()
    write_csv(polar, path)

    data = read_section_polar(path)

    assert (data.airfoil, data.reynolds, data.mach, data.ncrit) == (
        "NACA 2412", 4.1646e7, 0.6783, 9.0)
    np.testing.assert_array_equal(data.alpha_deg, polar.alpha_deg)
    np.testing.assert_array_equal(data.cl, polar.cl)
    np.testing.assert_array_equal(data.cd, polar.cd)
    np.testing.assert_array_equal(data.converged, polar.converged)


def test_read_section_polar_missing_file_raises(tmp_path: Path) -> None:
    """A missing file is a SectionPolarError naming the path."""
    with pytest.raises(SectionPolarError, match="cannot read"):
        read_section_polar(tmp_path / "missing.csv")


def test_read_section_polar_missing_header_key_raises(tmp_path: Path) -> None:
    """Without the Reynolds number the polar cannot be placed; the file is rejected."""
    path = tmp_path / "polar.csv"
    write_csv(_polar(), path)
    text = "\n".join(line for line in path.read_text(encoding="utf-8").splitlines()
                     if not line.startswith("# reynolds"))
    path.write_text(text, encoding="utf-8")

    with pytest.raises(SectionPolarError, match="reynolds"):
        read_section_polar(path)
