# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 KRGulaj
# Created: 2026-10-04
"""Figures of task 1a from the written results (no XFoil calls).

Reads results/section/<airfoil>/polar.csv and the [[extra_run]] polars, writes to
results/section/figures/:

- <airfoil>_polar.png       cl–α, drag polar, cm–α with the key values
- overlay_polar.png         both airfoils in one 3-panel figure
- sections.png              section shapes at true aspect ratio
and, for each [[extra_run]], to its own output folder:

- <airfoil>_polar.png       cl–α, drag polar, cm–α with the key values
- drag_polars.png           drag polars of every airfoil

Run from the repo root after scripts.run_section: python -m scripts.make_section_plots
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Final

from aircraft_tutorial.config.cases import load_section_case
from aircraft_tutorial.geometry.airfoil import read_dat
from aircraft_tutorial.plots import section as plots
from aircraft_tutorial.plots.style import series_colours
from aircraft_tutorial.section.metrics import compute_metrics
from aircraft_tutorial.section.polar import read_csv
from scripts.run_section import slug

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
INPUT: Final[Path] = ROOT / "inputs" / "section.toml"
RESULTS: Final[Path] = ROOT / "results" / "section"
FIGURES: Final[Path] = RESULTS / "figures"


def main() -> None:
    """Write every task 1a figure."""
    case = load_section_case(INPUT)
    names = [a.name for a in case.airfoils]
    colours = series_colours(names)
    FIGURES.mkdir(parents=True, exist_ok=True)
    window = (case.metrics.linear_alpha_min_deg, case.metrics.linear_alpha_max_deg)

    polars = {a.name: read_csv(RESULTS / slug(a.name) / "polar.csv") for a in case.airfoils}
    for entry in case.airfoils:
        m = compute_metrics(polars[entry.name], case.metrics, entry.drag_bucket)
        plots.plot_polar_panels(polars[entry.name], m, colours[entry.name], window,
                                FIGURES / f"{slug(entry.name)}_polar.png")
    plots.plot_overlay(polars, colours, FIGURES / "overlay_polar.png")
    sections = {a.name: dataclasses.replace(read_dat(a.path), name=a.name) for a in case.airfoils}
    plots.plot_sections(sections, colours, FIGURES / "sections.png")

    for run in case.extra_runs:
        extra = {a.name: read_csv(run.output / f"{slug(a.name)}_polar.csv")
                 for a in case.airfoils}
        for entry in case.airfoils:
            m = compute_metrics(extra[entry.name], case.metrics,
                                entry.drag_bucket and run.report_low_drag)
            plots.plot_polar_panels(extra[entry.name], m, colours[entry.name], window,
                                    run.output / f"{slug(entry.name)}_polar.png")
        plots.plot_drag_polars(extra, colours, run.output / "drag_polars.png")


if __name__ == "__main__":
    main()
