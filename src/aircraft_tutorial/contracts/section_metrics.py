# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Section characteristics hand-off from the 2D stage to the 3D stage: the metrics CSV in
results/3d_input/ with the columns quantity, value, unit (one row per characteristic, for
example a0_per_rad, alpha_0l_deg, cl_max).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


class SectionMetricsError(ValueError):
    """Raised for an unreadable metrics file or a missing or non-numeric quantity."""


@dataclass(frozen=True)
class SectionMetricsData:
    """Section characteristics read from file.

    Attributes:
        path: File the values were read from.
        values: Value of each quantity, in the unit of the file.
        units: Unit of each quantity.
    """

    path: Path
    values: dict[str, float]
    units: dict[str, str]

    def value(self, quantity: str) -> float:
        """One finite characteristic.

        Raises:
            SectionMetricsError: If the quantity is missing or not finite.
        """
        if quantity not in self.values:
            raise SectionMetricsError(f"{self.path}: quantity {quantity!r} is missing")
        result = self.values[quantity]
        if not math.isfinite(result):
            raise SectionMetricsError(f"{self.path}: quantity {quantity!r} = {result} is not finite")
        return result


def read_section_metrics(path: Path) -> SectionMetricsData:
    """Read a metrics CSV file.

    Args:
        path: Metrics CSV file.

    Returns:
        The characteristics.

    Raises:
        SectionMetricsError: If the file is missing, the header is not quantity,value,unit, or
            a value is not a number.
    """
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
    except OSError as exc:
        raise SectionMetricsError(f"{path}: cannot read: {exc}") from exc
    if not rows or rows[0] != ["quantity", "value", "unit"]:
        raise SectionMetricsError(f"{path}: expected the header quantity,value,unit")
    values: dict[str, float] = {}
    units: dict[str, str] = {}
    for number, row in enumerate(rows[1:], start=2):
        if len(row) != 3:
            raise SectionMetricsError(f"{path}: line {number}: expected 3 fields, got {row}")
        try:
            values[row[0]] = float(row[1])
        except ValueError as exc:
            raise SectionMetricsError(f"{path}: line {number}: {exc}") from exc
        units[row[0]] = row[2]
    return SectionMetricsData(path=path, values=values, units=units)
