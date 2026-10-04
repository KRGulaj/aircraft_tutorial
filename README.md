<!--
SPDX-License-Identifier: GPL-3.0-or-later
Copyright (C) 2026 KRGulaj
Created: 2026-10-04
-->

# Aircraft Tutorial (AE2111-II)

This repo holds the calculations for the AE2111-II aircraft tutorial group report.
The subject is a wing re-design of the Boeing 747-8:

1. Airfoil selection, trapezoidal wing planform, twist sizing and divergence Mach number.
2. Class II wing weight (Raymer) and design weight convergence.
3. Trailing-edge high-lift device sizing.

## Setup

```
C:\ProgramData\miniforge3\Scripts\conda.exe env create -f environment.yml
"$USERPROFILE/.conda/envs/aircraft_tutorial/python.exe" -m pip install ../system_design/externals/xfoil-python
```

VS Code uses the env automatically (`.vscode/settings.json`).

## Checks

```
"$USERPROFILE/.conda/envs/aircraft_tutorial/python.exe" -m pytest
"$USERPROFILE/.conda/envs/aircraft_tutorial/python.exe" -m mypy
```

## License

GPL-3.0-or-later, full text in [`LICENSE`](LICENSE).
