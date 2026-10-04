# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Tests for the Polar container and its CSV representation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aircraft_tutorial.section.polar import Polar, PolarError, RunInfo, read_csv, write_csv

INFO = RunInfo(airfoil="NACA 2412", reynolds=8.9e6, mach=0.15, ncrit=10.0, panel_nodes=160)


@pytest.fixture
def polar() -> Polar:
    """Seven points, the sixth one not converged (NaN values)."""
    nan = np.nan
    return Polar(
        info=INFO,
        alpha_deg=np.array([-2.75, -1.5, 0.0, 1.25, 3.5, 17.75, 18.0]),
        cl=np.array([-0.07, 0.07, 0.23, 0.37, 0.61, nan, 1.41]),
        cd=np.array([0.0061, 0.0058, 0.0059, 0.0063, 0.0071, nan, 0.031]),
        cm=np.array([-0.051, -0.052, -0.053, -0.053, -0.054, nan, -0.02]),
        cp_min=np.array([-0.31, -0.42, -0.61, -0.83, -1.29, nan, -9.7]),
        converged=np.array([True, True, True, True, True, False, True]),
        diverged=np.array([False, False, False, False, False, True, False]),
        rms_bl=np.array([2.1e-5, 3.3e-5, 1.7e-5, 4.4e-5, 6.1e-5, 7.3, 9.2e-5]),
    )


def test_write_then_read_csv_is_bitwise_identical(tmp_path: Path, polar: Polar) -> None:
    """Round trip keeps every value exactly (repr of floats), NaN included, and the run info."""
    path = tmp_path / "p.csv"

    write_csv(polar, path)
    back = read_csv(path)

    assert back.info == INFO
    for name in ("alpha_deg", "cl", "cd", "cm", "cp_min", "rms_bl"):
        assert getattr(back, name).tobytes() == getattr(polar, name).tobytes()
    assert np.array_equal(back.converged, polar.converged)
    assert np.array_equal(back.diverged, polar.diverged)


def test_converged_only_drops_failed_point(polar: Polar) -> None:
    """Six of seven points remain; the 17.75 deg point is gone."""
    sub = polar.converged_only()

    assert len(sub.alpha_deg) == 6
    assert 17.75 not in sub.alpha_deg
    assert bool(np.all(np.isfinite(sub.cl)))


def test_polar_unsorted_alpha_raises(polar: Polar) -> None:
    """Angles must be strictly ascending."""
    with pytest.raises(PolarError, match="ascending"):
        Polar(INFO, polar.alpha_deg[::-1], polar.cl, polar.cd, polar.cm, polar.cp_min,
              polar.converged, polar.diverged, polar.rms_bl)


def test_polar_column_length_mismatch_raises(polar: Polar) -> None:
    """A column with a different length is rejected and named."""
    with pytest.raises(PolarError, match="cd"):
        Polar(INFO, polar.alpha_deg, polar.cl, polar.cd[:-1], polar.cm, polar.cp_min,
              polar.converged, polar.diverged, polar.rms_bl)


def test_read_csv_missing_header_key_raises(tmp_path: Path, polar: Polar) -> None:
    """A file without the Reynolds header line is rejected."""
    path = tmp_path / "p.csv"
    write_csv(polar, path)
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if not ln.startswith("# reynolds")]
    path.write_text("\n".join(lines), encoding="utf-8")

    with pytest.raises(PolarError, match="reynolds"):
        read_csv(path)
