# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Task 1a production run: polars, characteristics, numerical sensitivity, extra runs.

Reads inputs/section.toml and writes:

- results/section/<airfoil>/polar.csv       full polar at the production condition
- results/section/<airfoil>/metrics.csv     characteristics (quantity, value, unit)
- results/section/metrics.csv               characteristics of all airfoils side by side
- results/section/sensitivity.csv           one-at-a-time panel-node and Ncrit cases
- results/section/panel_check.csv           production nodes vs finest nodes, accepted or not
- <extra_run.output>/<airfoil>_section.dat  analysed section: the coordinate file cut normal
                                            to the run's sweep line (geometry.sweep)
- <extra_run.output>/<airfoil>_polar.csv    polar of that section at the run's condition
- <extra_run.output>/<airfoil>_metrics.csv  its characteristics, plus alpha_0l_streamwise_deg

Run from the repo root: python -m scripts.run_section
"""

from __future__ import annotations

import csv
import dataclasses
import logging
import math
from pathlib import Path
from typing import Final

from aircraft_tutorial.config.cases import SectionCase, load_section_case
from aircraft_tutorial.geometry.airfoil import Airfoil, read_dat, write_dat
from aircraft_tutorial.geometry.sweep import streamwise_angle_deg, sweep_normal_section
from aircraft_tutorial.section.metrics import (
    MetricError,
    SectionMetrics,
    compute_metrics,
    metrics_table,
)
from aircraft_tutorial.section.polar import write_csv
from aircraft_tutorial.section.sensitivity import (
    PANEL_TOLERANCE,
    relative_change,
    sensitivity_cases,
)
from aircraft_tutorial.section.xfoil_runtime import run_polar

logger = logging.getLogger("scripts.run_section")

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
INPUT: Final[Path] = ROOT / "inputs" / "section.toml"
OUT: Final[Path] = ROOT / "results" / "section"
CHECKED: Final[tuple[str, ...]] = ("a0_per_deg", "cl_max", "cd_min")
"""Quantities of the panel-count acceptance check."""


def slug(name: str) -> str:
    """File-system name of an airfoil: "NACA 66-410" → "naca66-410"."""
    return name.lower().replace(" ", "")


def main() -> None:
    """Run every stage of the production run."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    case = load_section_case(INPUT)
    sections = {a.name: dataclasses.replace(read_dat(a.path), name=a.name)
                for a in case.airfoils}

    production = _production(case, sections)
    _write_rows(OUT / "metrics.csv", _side_by_side(production))
    sens_rows = _sensitivity(case, sections)
    _write_rows(OUT / "sensitivity.csv", sens_rows)
    _write_rows(OUT / "panel_check.csv", _panel_check(case, sens_rows))
    _extra_runs(case, sections)


def _production(case: SectionCase, sections: dict[str, Airfoil]
                ) -> dict[str, SectionMetrics]:
    """Polar and characteristics of every airfoil at the production condition."""
    out: dict[str, SectionMetrics] = {}
    for entry in case.airfoils:
        polar = run_polar(sections[entry.name], case.conditions, case.xfoil)
        folder = OUT / slug(entry.name)
        folder.mkdir(parents=True, exist_ok=True)
        write_csv(polar, folder / "polar.csv")
        m = compute_metrics(polar, case.metrics, entry.drag_bucket)
        _write_rows(folder / "metrics.csv", [
            {"quantity": q, "value": f"{v:.6g}", "unit": u} for q, v, u in metrics_table(m)])
        out[entry.name] = m
        logger.info("%s: %d/%d points converged", entry.name,
                    int(polar.converged.sum()), polar.converged.size)
    return out


def _side_by_side(production: dict[str, SectionMetrics]) -> list[dict[str, str]]:
    """One row per quantity, one column per airfoil; a quantity missing for one airfoil is ''."""
    tables = {name: {q: (v, u) for q, v, u in metrics_table(m)} for name, m in production.items()}
    order: list[str] = []
    for t in tables.values():
        order += [q for q in t if q not in order]
    rows: list[dict[str, str]] = []
    for q in order:
        unit = next(t[q][1] for t in tables.values() if q in t)
        row = {"quantity": q, "unit": unit}
        row.update({name: f"{t[q][0]:.6g}" if q in t else "" for name, t in tables.items()})
        rows.append(row)
    return rows


def _sensitivity(case: SectionCase, sections: dict[str, Airfoil]) -> list[dict[str, str]]:
    """Characteristics of every one-at-a-time case for every airfoil."""
    rows: list[dict[str, str]] = []
    for entry in case.airfoils:
        for sc in sensitivity_cases(case):
            polar = run_polar(sections[entry.name], sc.conditions, sc.xfoil)
            row = {"airfoil": entry.name, "varied": sc.varied,
                   "panel_nodes": str(sc.xfoil.panel_nodes), "ncrit": f"{sc.conditions.ncrit:g}",
                   "converged": f"{int(polar.converged.sum())}/{polar.converged.size}"}
            try:
                m = compute_metrics(polar, case.metrics, entry.drag_bucket)
            except MetricError as exc:
                logger.warning("%s %s: %s", entry.name, sc.varied, exc)
                row.update({q: "nan" for q in ("a0_per_deg", "alpha_0l_deg", "cl_max",
                                               "alpha_stall_deg", "cd_min", "cl_at_cd_min")})
                rows.append(row)
                continue
            row.update({"a0_per_deg": f"{m.a0_per_deg:.6g}", "alpha_0l_deg": f"{m.alpha_0l_deg:.6g}",
                        "cl_max": f"{m.stall.cl_max:.6g}", "alpha_stall_deg": f"{m.stall.alpha_deg:.6g}",
                        "cd_min": f"{m.cd_min:.6g}", "cl_at_cd_min": f"{m.cl_at_cd_min:.6g}"})
            rows.append(row)
    return rows


def _panel_check(case: SectionCase, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Production node count against the finest node count, per airfoil and quantity."""
    prod, finest = case.xfoil.panel_nodes, max(case.sensitivity.panel_nodes)
    out: list[dict[str, str]] = []
    for entry in case.airfoils:
        by_nodes = {int(r["panel_nodes"]): r for r in rows
                    if r["airfoil"] == entry.name and r["varied"] == "panel_nodes"}
        for q in CHECKED:
            v, ref = float(by_nodes[prod][q]), float(by_nodes[finest][q])
            rel = relative_change(v, ref) if math.isfinite(v) else math.nan
            out.append({"airfoil": entry.name, "quantity": q, f"nodes_{prod}": f"{v:.6g}",
                        f"nodes_{finest}": f"{ref:.6g}", "relative_change": f"{rel:.4g}",
                        "accepted": str(bool(rel < PANEL_TOLERANCE))})
            logger.info("panel check %s %s: %.4g %s", entry.name, q, rel,
                        "accepted" if rel < PANEL_TOLERANCE else "NOT ACCEPTED")
    return out


def _extra_runs(case: SectionCase, sections: dict[str, Airfoil]) -> None:
    """Polar and characteristics of the sweep-normal section of every airfoil at every
    [[extra_run]] condition.

    The zero-lift angle is also given in the streamwise plane, tan α_0l,s = tan α_0l,n·cos Λ,
    the value a wing method with streamwise sections needs.
    """
    for run in case.extra_runs:
        run.output.mkdir(parents=True, exist_ok=True)
        for entry in case.airfoils:
            section = sweep_normal_section(sections[entry.name], run.sweep_deg)
            stem = slug(entry.name)
            write_dat(section, run.output / f"{stem}_section.dat",
                      f"{section.name} (from {entry.path.name}, y/c divided by cos sweep)")
            polar = run_polar(section, run.conditions, case.xfoil)
            write_csv(polar, run.output / f"{stem}_polar.csv")
            m = compute_metrics(polar, case.metrics, entry.drag_bucket and run.report_low_drag)
            rows = metrics_table(m) + [
                ("alpha_0l_streamwise_deg", streamwise_angle_deg(m.alpha_0l_deg, run.sweep_deg),
                 "deg")]
            _write_rows(run.output / f"{stem}_metrics.csv", [
                {"quantity": q, "value": f"{v:.6g}", "unit": u} for q, v, u in rows])
            logger.info("%s %s: %d/%d points converged, stall detected: %s", run.label,
                        entry.name, int(polar.converged.sum()), polar.converged.size,
                        m.stall.detected)


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    """Write dict rows as CSV with LF line ends."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
