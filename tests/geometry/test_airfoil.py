# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Tests for the Airfoil container, .dat I/O and shape properties."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aircraft_tutorial.geometry.airfoil import (
    Airfoil,
    AirfoilError,
    cosine_stations,
    max_thickness,
    normalized,
    read_dat,
    write_dat,
)
from aircraft_tutorial.geometry.naca4 import naca4


@pytest.fixture
def section() -> Airfoil:
    """NACA 2412 on 13 cosine stations per surface (25 points)."""
    return naca4("2412", cosine_stations(13))


def test_cosine_stations_seven_points_match_formula() -> None:
    """x_i = (1 − cos(π i / 6)) / 2: x_2 = 0.25, x_3 = 0.5, x_5 = 0.9330127."""
    x = cosine_stations(7)

    assert x[0] == 0.0
    assert x[-1] == pytest.approx(1.0, abs=1e-15)
    assert x[2] == pytest.approx(0.25, abs=1e-15)
    assert x[3] == pytest.approx(0.5, abs=1e-15)
    assert x[5] == pytest.approx((1.0 + np.sqrt(3.0) / 2.0) / 2.0, abs=1e-15)


def test_write_then_read_dat_returns_same_contour(tmp_path: Path, section: Airfoil) -> None:
    """Round trip through the file keeps every coordinate to the written 8 decimals."""
    path = tmp_path / "s.dat"

    write_dat(section, path, "NACA 2412 (test)")
    back = read_dat(path)

    assert back.name == "NACA 2412 (test)"
    np.testing.assert_allclose(back.x, section.x, atol=5e-9)
    np.testing.assert_allclose(back.y, section.y, atol=5e-9)


def test_read_dat_lednicer_equals_selig(tmp_path: Path, section: Airfoil) -> None:
    """A Lednicer file (count line, two blocks LE → TE) gives the same contour as Selig."""
    xu, yu, xl, yl = section.surfaces()
    rows_u = "\n".join(f"{float(a)!r} {float(b)!r}" for a, b in zip(xu, yu, strict=True))
    rows_l = "\n".join(f"{float(a)!r} {float(b)!r}" for a, b in zip(xl, yl, strict=True))
    path = tmp_path / "l.dat"
    path.write_text(f"LEDNICER\n{len(xu)}. {len(xl)}.\n\n{rows_u}\n\n{rows_l}\n",
                    encoding="utf-8")

    back = read_dat(path)

    np.testing.assert_allclose(back.x, section.x, atol=1e-15)
    np.testing.assert_allclose(back.y, section.y, atol=1e-15)


def test_read_dat_drops_consecutive_duplicate_point(tmp_path: Path, section: Airfoil) -> None:
    """A repeated leading-edge row is removed; the point count is unchanged."""
    i = section.leading_edge_index
    rows = [f"{float(a)!r} {float(b)!r}" for a, b in zip(section.x, section.y, strict=True)]
    rows.insert(i, rows[i])
    path = tmp_path / "d.dat"
    path.write_text("DUP\n" + "\n".join(rows) + "\n", encoding="utf-8")

    back = read_dat(path)

    assert len(rows) == section.x.size + 1
    assert back.x.size == section.x.size


def test_airfoil_lower_surface_first_raises(section: Airfoil) -> None:
    """Reversed point order (lower surface first) is rejected."""
    with pytest.raises(AirfoilError, match="upper surface"):
        Airfoil("reversed", section.x[::-1], section.y[::-1])


def test_airfoil_arrays_are_read_only(section: Airfoil) -> None:
    """The coordinate arrays cannot be modified in place."""
    with pytest.raises(ValueError):
        section.x[3] = 0.37


def test_normalized_scaled_shifted_contour_recovers_unit_chord(section: Airfoil) -> None:
    """Contour scaled by 2.9 and shifted by (0.31, −1.7) normalises back to the original."""
    moved = Airfoil("moved", 2.9 * section.x + 0.31, 2.9 * section.y - 1.7)

    back = normalized(moved)

    np.testing.assert_allclose(back.x, section.x, atol=1e-14)
    np.testing.assert_allclose(back.y, section.y, atol=1e-14)


def test_max_thickness_naca0012_is_twelve_percent_near_thirty_percent() -> None:
    """NACA 0012: maximum thickness at x ≈ 0.30 (Abbott §6.4); eq. (6.2) gives 0.12003."""
    t, x_t = max_thickness(naca4("0012", cosine_stations(401)))

    assert t == pytest.approx(0.12003, abs=2e-5)
    assert x_t == pytest.approx(0.30, abs=0.01)
