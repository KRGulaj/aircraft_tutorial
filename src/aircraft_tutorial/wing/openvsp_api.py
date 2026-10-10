# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-10
"""The OpenVSP Python API, imported once for the modules that drive OpenVSP and VSPAERO.

`openvsp` is a SWIG extension without type stubs. A plain `import openvsp` leaves every call
Unknown, and strict type checkers (Pylance, pyright) then report each use. Here the module is
typed `Any` at a single boundary, so the callers stay strict everywhere else.
"""

from __future__ import annotations

import importlib
from typing import Any

vsp: Any = importlib.import_module("openvsp")
"""OpenVSP API module; untyped C extension, so `Any` is unavoidable here."""
