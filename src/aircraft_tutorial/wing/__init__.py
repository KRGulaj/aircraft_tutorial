# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-04
"""3D wing analysis: trapezoidal planform, cruise design point, OpenVSP model, VSPAERO runs and
twist sizing.

`planform`, `cruise` and `twist` are pure Python. `vsp_model` and `vspaero` import the OpenVSP
Python API (Python 3.11 build) at module level.
"""
