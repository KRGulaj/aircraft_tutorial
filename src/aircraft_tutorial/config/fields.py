# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""Typed access to the values of a parsed TOML run definition.

Every function names the offending key in dotted form ("planform.span_m") when it raises, so
an error message points straight at the line to fix.
"""

from __future__ import annotations

import math
from typing import cast

from aircraft_tutorial.config.cases import CaseError


def table(raw: dict[str, object], key: str) -> dict[str, object]:
    """Return a required sub-table.

    Raises:
        CaseError: If the table is missing.
    """
    value = raw.get(key)
    if not isinstance(value, dict):
        raise CaseError(f"[{key}]: required table is missing")
    return cast(dict[str, object], value)  # a TOML table: its keys are str


def number(t: dict[str, object], dotted: str) -> float:
    """Return a required finite number.

    Raises:
        CaseError: If the key is missing or the value is not a finite number.
    """
    value = _get(t, dotted)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CaseError(f"{dotted}: must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise CaseError(f"{dotted}: must be finite, got {result}")
    return result


def integer(t: dict[str, object], dotted: str) -> int:
    """Return a required integer.

    Raises:
        CaseError: If the key is missing or the value is not an integer.
    """
    value = _get(t, dotted)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CaseError(f"{dotted}: must be an integer, got {value!r}")
    return value


def string(t: dict[str, object], dotted: str) -> str:
    """Return a required non-empty string.

    Raises:
        CaseError: If the key is missing or the value is not a non-empty string.
    """
    value = _get(t, dotted)
    if not isinstance(value, str) or not value.strip():
        raise CaseError(f"{dotted}: must be a non-empty string, got {value!r}")
    return value


def require(condition: bool, dotted: str, value: object, reason: str) -> None:
    """Raise CaseError if the positive-form condition is false."""
    if not condition:
        raise CaseError(f"{dotted} = {value!r}: {reason}")


def _get(t: dict[str, object], dotted: str) -> object:
    """Return the value of the last component of a dotted key."""
    key = dotted.rsplit(".", 1)[-1]
    if key not in t:
        raise CaseError(f"{dotted}: required key is missing")
    return t[key]
