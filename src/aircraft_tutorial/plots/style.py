# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Shared figure style for the printed report.

Colour carries only the identity of an airfoil. Every airfoil keeps one hue in every figure,
assigned from a fixed categorical order (validated for colour-vision deficiency: worst adjacent
protan ΔE 24.7, normal-vision ΔE 33.6, contrast ≥ 3:1 on white). Text, values and labels use
neutral ink, never the series colour.
"""

from __future__ import annotations

from typing import Final

import matplotlib as mpl

CATEGORICAL: Final[tuple[str, ...]] = ("#2a78d6", "#eb6834")
INK: Final[str] = "#1f1f1e"
INK_MUTED: Final[str] = "#6b6a63"
GRID: Final[str] = "#dedcd5"
FIT_LINE: Final[str] = "#6b6a63"

FULL_WIDTH_IN: Final[float] = 6.7
"""Text width of an A4 page with 20 mm margins [in]."""
DPI: Final[int] = 300


class StyleError(ValueError):
    """Raised when more series are requested than the palette holds."""


def series_colours(names: list[str]) -> dict[str, str]:
    """Fixed colour per airfoil, in the given order.

    Args:
        names: Airfoil names in run-definition order.

    Returns:
        Name → hex colour.

    Raises:
        StyleError: If there are more names than palette slots.
    """
    if len(names) > len(CATEGORICAL):
        raise StyleError(f"{len(names)} series, palette has {len(CATEGORICAL)} slots")
    return dict(zip(names, CATEGORICAL, strict=False))


def apply() -> None:
    """Set the report rcParams: thin lines, recessive grid, neutral ink, serif-free text."""
    mpl.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.0,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "figure.dpi": 110,
    })
