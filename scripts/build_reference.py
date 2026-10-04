# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Collect the TR 824 readings, estimate NACA 66-410, and check the increment against theory.

Writes:

- results/section/reference/reference_values.csv   every reading and the 66-410 estimate
- results/section/reference/camber_increment.csv   measured increment vs thin-airfoil theory

Run from the repo root: python -m scripts.build_reference
"""

from __future__ import annotations

import csv
from dataclasses import fields
from pathlib import Path
from typing import Final

from aircraft_tutorial.section.reference import (
    ABBOTT_ALPHA0_FACTOR_A10,
    ABBOTT_CM_FACTOR_A10,
    ReferenceData,
    Readings,
    estimate_from_camber_increment,
    load_reference,
    thin_airfoil_a10_increment,
)

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
INPUTS: Final[Path] = ROOT / "inputs" / "reference"
OUT: Final[Path] = ROOT / "results" / "section" / "reference"


def main() -> None:
    """Load the readings, build the estimate and write both tables."""
    n2412 = load_reference(INPUTS / "naca2412.toml")
    base = load_reference(INPUTS / "naca66-210.toml")
    low = load_reference(INPUTS / "naca66_2-215.toml")
    high = load_reference(INPUTS / "naca66_2-415.toml")
    est = estimate_from_camber_increment("NACA 66-410", base, low, high, 0.4)

    OUT.mkdir(parents=True, exist_ok=True)
    _write_values([n2412, base, low, high, est], OUT / "reference_values.csv")
    rows = _increment_rows(low, high)
    _write_rows(OUT / "camber_increment.csv", rows)
    for row in rows:
        print(", ".join(f"{k}={v}" for k, v in row.items()))


def _write_values(refs: list[ReferenceData], path: Path) -> None:
    """One row per (airfoil, quantity) that has a value."""
    rows: list[dict[str, str]] = []
    for ref in refs:
        for f in fields(Readings):
            r = getattr(ref.readings, f.name)
            if r is None:
                continue
            rows.append({
                "airfoil": ref.airfoil, "quantity": f.name, "value": f"{r.value:.6g}",
                "uncertainty": f"{r.uncertainty:.3g}", "reynolds": f"{ref.reynolds:.3g}",
                "estimated": str(ref.estimated), "checked": str(ref.checked),
                "source": ref.source, "note": r.note,
            })
    _write_rows(path, rows)


def _increment_rows(low: ReferenceData, high: ReferenceData) -> list[dict[str, str]]:
    """Measured α_0L and cm_c/4 increments against thin-airfoil theory times Abbott's factor."""
    d_cli = high.design_cl - low.design_cl
    d_alpha_th, d_cm_th = thin_airfoil_a10_increment(d_cli)
    rows: list[dict[str, str]] = []
    for name, theory, factor in (("alpha_0l_deg", d_alpha_th, ABBOTT_ALPHA0_FACTOR_A10),
                                 ("cm_c4", d_cm_th, ABBOTT_CM_FACTOR_A10)):
        lo, hi = getattr(low.readings, name), getattr(high.readings, name)
        if lo is None or hi is None:
            raise ValueError(f"calibration pair lacks {name}")
        measured = hi.value - lo.value
        unc = (lo.uncertainty**2 + hi.uncertainty**2) ** 0.5
        rows.append({
            "quantity": name, "delta_cli": f"{d_cli:.2f}", "measured": f"{measured:.4g}",
            "measured_uncertainty": f"{unc:.3g}", "thin_airfoil": f"{theory:.4g}",
            "abbott_factor": f"{factor:.2f}", "thin_airfoil_times_factor": f"{theory * factor:.4g}",
        })
    return rows


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    """Write dict rows as CSV with LF line ends."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
