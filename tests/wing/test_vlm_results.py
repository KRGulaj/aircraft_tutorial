# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Tests for the VSPAERO result files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aircraft_tutorial.wing.vlm_results import (
    StripLoads,
    VlmPolar,
    VlmResultsError,
    read_polar_csv,
    read_strips_csv,
    write_polar_csv,
    write_strips_csv,
)


def test_polar_csv_round_trip_is_exact(tmp_path: Path) -> None:
    """Every value is written at full precision and read back unchanged."""
    polar = VlmPolar(np.array([-1.0, 0.0, 1.0]), np.array([0.45, 0.551337, 0.65]),
                     np.array([0.008, 0.012, 0.0165]), np.array([-0.14, -0.1438, -0.15]))
    write_polar_csv(polar, tmp_path / "p.csv")

    back = read_polar_csv(tmp_path / "p.csv")

    for name in ("alpha_deg", "cl", "cdi", "cmy"):
        np.testing.assert_array_equal(getattr(back, name), getattr(polar, name))


def test_strips_round_trip_and_select_one_angle(tmp_path: Path) -> None:
    """Strips survive the file; at_alpha returns only that angle's strips."""
    strips = StripLoads(np.array([0.0, 0.0, 1.0]), np.array([1.0, 2.0, 1.0]),
                        np.array([14.0, 12.0, 14.0]), np.array([20.0, 18.0, 20.0]),
                        np.array([0.5, 0.6, 0.55]))
    write_strips_csv(strips, tmp_path / "s.csv")

    back = read_strips_csv(tmp_path / "s.csv").at_alpha(0.0)

    np.testing.assert_array_equal(back.y_m, [1.0, 2.0])
    with pytest.raises(VlmResultsError, match="no strips"):
        strips.at_alpha(5.0)


def test_read_with_wrong_columns_raises(tmp_path: Path) -> None:
    """A strips file is not a polar file."""
    path = tmp_path / "s.csv"
    path.write_text("alpha_deg,y_m\n0.0,1.0\n", encoding="utf-8")

    with pytest.raises(VlmResultsError, match="expected columns"):
        read_polar_csv(path)
