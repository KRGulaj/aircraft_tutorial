# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Tests for the Korn drag-divergence Mach number and the ADSEE wave drag."""

from __future__ import annotations

import pytest

from aircraft_tutorial.wing.wave_drag import korn_mdd, wave_drag


def test_korn_mdd_unswept_reduces_to_plain_korn() -> None:
    """Λ = 0: M_dd = κ − t/c − CL/10 = 0.87 − 0.12 − 0.05 = 0.70."""
    assert korn_mdd(0.87, 0.12, 0.5, 0.0) == pytest.approx(0.70, abs=1e-12)


def test_korn_mdd_swept_matches_hand_calculation() -> None:
    """Λ = 30°: 0.87/0.86603 − 0.12/0.75 − 0.5/(10·0.64952) = 1.00459 − 0.16 − 0.07698."""
    assert korn_mdd(0.87, 0.12, 0.5, 30.0) == pytest.approx(0.76761, abs=1e-5)


def test_wave_drag_is_0002_at_mdd_and_continuous() -> None:
    """Both branches give 0.002 at M = M_dd."""
    assert wave_drag(0.8, 0.8) == pytest.approx(0.002)
    assert wave_drag(0.8 - 1e-9, 0.8) == pytest.approx(0.002, rel=1e-6)


def test_wave_drag_branches_match_formula() -> None:
    """M_dd − M = 0.05 → 0.002/3.5; M − M_dd = 0.05 → 0.002·2^2.5."""
    assert wave_drag(0.75, 0.80) == pytest.approx(0.002 / 3.5)
    assert wave_drag(0.85, 0.80) == pytest.approx(0.002 * 2.0**2.5)
