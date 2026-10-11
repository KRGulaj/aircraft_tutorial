# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-11
"""Tests for the span efficiency from the spanwise loading (lifting-line sine series)."""

from __future__ import annotations

import numpy as np
import pytest

from aircraft_tutorial.wing.span_efficiency import SpanEfficiencyError, loading_span_efficiency

ETA = np.cos(np.linspace(0.02, np.pi / 2 - 0.02, 24))[::-1]
"""24 stations on the half-wing, cosine spaced, root to tip."""


def test_elliptic_loading_gives_one() -> None:
    """Γ ∝ √(1 − η²) = sin θ: only A_1, so δ = 0 and e = 1 (Prandtl)."""
    e = loading_span_efficiency(ETA, np.sqrt(1.0 - ETA**2), terms=8)

    assert e == pytest.approx(1.0, abs=1e-12)


def test_third_harmonic_gives_hand_value() -> None:
    """Γ = sin θ + 0.1·sin 3θ: δ = 3·0.1² = 0.03, e = 1 / 1.03 = 0.97087."""
    theta = np.arccos(ETA)
    e = loading_span_efficiency(ETA, np.sin(theta) + 0.1 * np.sin(3.0 * theta), terms=8)

    assert e == pytest.approx(1.0 / 1.03, rel=1e-10)


def test_scaling_the_loading_does_not_change_e() -> None:
    """e depends on the shape of the loading only."""
    theta = np.arccos(ETA)
    load = np.sin(theta) - 0.05 * np.sin(5.0 * theta)

    e1 = loading_span_efficiency(ETA, load, terms=6)
    e2 = loading_span_efficiency(ETA, 7.3 * load, terms=6)

    assert e1 == pytest.approx(e2, rel=1e-12)


@pytest.mark.parametrize(("eta", "load", "terms", "match"),
                         [(ETA[:3], np.ones(3), 6, "cannot fit"),
                          (np.array([0.2, 0.5, 1.2]), np.ones(3), 2, r"\[0, 1\]"),
                          (ETA, -np.sqrt(1.0 - ETA**2), 4, "not positive")],
                         ids=["too_few_strips", "station_outside", "negative_lift"])
def test_invalid_input_raises(eta: np.ndarray, load: np.ndarray, terms: int, match: str) -> None:
    """Too few strips, a station beyond the tip or a negative loading are rejected."""
    with pytest.raises(SpanEfficiencyError, match=match):
        loading_span_efficiency(eta, load, terms=terms)
