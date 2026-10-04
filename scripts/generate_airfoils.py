# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Generate the analysis coordinate files and cross-check them against the group file.

Writes, with 51 cosine-spaced stations per surface (101 points, the resolution of the group file):

- airfoils/NACA2412.dat           analytic, Abbott eqs. (6.1)-(6.4)
- airfoils/NACA66-410_gen.dat     Abbott 66-010 table + a = 1.0 mean line, eq. (6.1)
- results/section/geometry_check.md   thickness and camber difference to airfoils/NACA66-410.DAT

Run from the repo root: python -m scripts.generate_airfoils
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from aircraft_tutorial.geometry.airfoil import (
    cosine_stations,
    max_thickness,
    normalized,
    read_dat,
    write_dat,
)
from aircraft_tutorial.geometry.compare import EDGE_MARGIN, compare
from aircraft_tutorial.geometry.naca4 import naca4
from aircraft_tutorial.geometry.naca6 import naca66

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
STATIONS_PER_SURFACE: Final[int] = 51
GROUP_FILE: Final[Path] = ROOT / "airfoils" / "NACA66-410.DAT"


def main() -> None:
    """Write both coordinate files and the cross-check table."""
    x = cosine_stations(STATIONS_PER_SURFACE)
    a2412 = naca4("2412", x)
    a66 = naca66("66-410", x)
    write_dat(a2412, ROOT / "airfoils" / "NACA2412.dat",
              "NACA 2412 (analytic, Abbott & von Doenhoff eqs. 6.1-6.4)")
    write_dat(a66, ROOT / "airfoils" / "NACA66-410_gen.dat",
              "NACA 66-410 (Abbott 66-010 table p.375 + a=1.0 mean line, eq. 6.1)")

    group = normalized(read_dat(GROUP_FILE))
    d = compare(a66, group)
    t_gen, x_gen = max_thickness(a66)
    t_grp, x_grp = max_thickness(group)

    out = ROOT / "results" / "section"
    out.mkdir(parents=True, exist_ok=True)
    table = "\n".join([
        "# Geometry cross-check: NACA 66-410",
        "",
        f"Generated contour minus group file, x/c in [{EDGE_MARGIN}, {1 - EDGE_MARGIN}].",
        "",
        "| Quantity | Value [% c] |",
        "|---|---|",
        f"| max thickness, generated (x/c = {x_gen:.3f}) | {100 * t_gen:.3f} |",
        f"| max thickness, group file (x/c = {x_grp:.3f}) | {100 * t_grp:.3f} |",
        f"| thickness difference, max abs | {100 * d.thickness_max_abs:.4f} |",
        f"| thickness difference, RMS | {100 * d.thickness_rms:.4f} |",
        f"| camber difference, min / max | {100 * d.camber_min:.4f} / {100 * d.camber_max:.4f} |",
        f"| camber difference, RMS | {100 * d.camber_rms:.4f} |",
        "",
    ])
    (out / "geometry_check.md").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
