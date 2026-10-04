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
"$USERPROFILE/.conda/envs/aircraft_tutorial/python.exe" -m pip install --no-build-isolation ../system_design/externals/xfoil-python
```

The XFoil build needs MSYS2 `mingw64in` and CMake first on `PATH`; the steps are in
`system_design/wp2/README.md`.

Install the package in editable mode, once:

```
"$USERPROFILE/.conda/envs/aircraft_tutorial/python.exe" -m pip install -e . --no-deps
```

VS Code uses the env automatically (`.vscode/settings.json`).

## Layout

```
airfoils/                 airfoil coordinate files (.dat)
inputs/                   run definitions and values read from reference reports
scripts/                  runnable steps: python -m scripts.<name>, from the repo root
src/aircraft_tutorial/
  common/                 tools for every stage (units, fitting)
  config/                 input loading and validation
  geometry/               airfoil geometry
  section/                2D section analysis (XFoil)
  contracts/              data passed between stages
  plots/                  figures
results/                  generated data and figures, one folder per stage
tests/                    pytest suite, mirrors src/
```

A stage package never imports another stage package. Data passes between stages through
`contracts/` only. A package is created when it gets its first module.

## Checks

```
"$USERPROFILE/.conda/envs/aircraft_tutorial/python.exe" -m pytest
"$USERPROFILE/.conda/envs/aircraft_tutorial/python.exe" -m mypy
```

## License

GPL-3.0-or-later, full text in [`LICENSE`](LICENSE). Airfoil coordinate files supplied
by the group from external sources keep their own terms.
