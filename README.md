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
"$USERPROFILE/.conda/envs/aircraft_tutorial/python.exe" -m pip install -e . --no-deps
```

Then build XFoil (next section). VS Code uses the env automatically (`.vscode/settings.json`).

## Installing XFoil

The Python binding is built from the vendored source in
[`externals/xfoil-python`](externals/xfoil-python): a copy of
[DARcorporation/xfoil-python](https://github.com/DARcorporation/xfoil-python), commit `0a8c2fc`,
GPL-3.0, with the local patches listed below. XFoil's core is Fortran 90, so the build needs
gfortran.

### Requirements

- MSYS2 (https://www.msys2.org), installed at `C:\msys64`. From an MSYS2 shell:
  `pacman -S mingw-w64-x86_64-gcc-fortran mingw-w64-x86_64-make`
- CMake, installed at `C:\Program Files\CMake`.

### Build (PowerShell, repo root)

```powershell
$env:Path = "C:\msys64\mingw64\bin;C:\Program Files\CMake\bin;" + $env:Path
$py = "$env:USERPROFILE\.conda\envs\aircraft_tutorial\python.exe"
& $py -m pip install wheel setuptools
& $py -m pip install --no-build-isolation .\externals\xfoil-python
```

After a change to `externals/xfoil-python`, run `pip uninstall -y xfoil` and delete
`externals/xfoil-python/build/` first, or pip keeps the old build.

### Local patches

| File | Change | Reason |
|---|---|---|
| `pyproject.toml` | Build backend `setuptools` instead of `scikit-build` | scikit-build selects the Visual Studio generator, which has no Fortran compiler |
| `setup.py` | CMake generator `MinGW Makefiles` on Windows | Selects gfortran from MSYS2 |
| `src/api.f90`, `xfoil/xfoil.py` | `alfa_` and `XFoil.a()` also return `diverged` and `rms_bl` | Separates a point that ran out of iterations from one where the boundary-layer solve stopped on a NaN |

### Runtime notes

`aircraft_tutorial.section.xfoil_runtime` does two things before the first `import xfoil`:

1. It registers `C:\msys64\mingw64\bin` with `os.add_dll_directory`. Since Python 3.8, ctypes
   does not use `PATH` to find the MinGW runtime DLLs that `libxfoil.dll` needs.
2. It declares `kernel32.FreeLibrary` with a pointer argument. Without this, `XFoil.__del__`
   fails on the 64-bit handle, and every instance leaks its DLL handle and temporary DLL file.

### Check

```
"$USERPROFILE/.conda/envs/aircraft_tutorial/python.exe" -m pytest tests/section
```

## Layout

```
airfoils/                 airfoil coordinate files (.dat)
externals/xfoil-python/   vendored XFoil binding source (GPL-3.0, local patches)
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

GPL-3.0-or-later, full text in [`LICENSE`](LICENSE). Airfoil coordinate files supplied by the
group from external sources keep their own terms. `externals/xfoil-python` is GPL-3.0 (its own
`LICENSE`).
