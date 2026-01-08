# Battery Thermal Simulation and Analysis

This repository contains research notebooks for battery thermal modeling,
simulation, and dataset generation under realistic drive cycles.

The code is primarily provided as Jupyter notebooks intended for exploratory
research and validation rather than turnkey execution.

## Overview

The notebooks implement:
- Physics based battery thermal modeling
- Heat propagation simulation under dynamic load profiles
- Signal filtering and numerical integration pipelines
- Preliminary dataset generation for machine learning tasks

The work is part of an ongoing research effort and is not a packaged library.

## Repository Structure

- `.ipynb` files: core simulation and analysis notebooks
- `drivecycles/` (not included): external drive cycle inputs
- No standalone Python package structure is assumed

## External Data Dependency

This repository depends on standard battery drive cycle data
such as LA92 and similar profiles.

These datasets are **not included** because:
- They are distributed by government or institutional sources
- Access may require academic or organizational verification
- Redistribution is not permitted

Users must obtain compatible drive cycle data independently
and place it in a local `drivecycles/` directory.

## Execution Environment

There is no `requirements.txt` by design.

The notebooks are intended to be run inside a configured Python
environment or Jupyter kernel with the required scientific libraries
already available.

Typical dependencies include:
- NumPy, pandas, SciPy, Matplotlib
- PyBaMM for battery modeling
- TensorFlow for neural components
- Numba for performance acceleration
- Pint, xarray, tqdm, uncertainties
- Optional CoolProp for thermophysical properties

Exact versions depend on the host environment.

## Notes on Reproducibility

Results depend on:
- Drive cycle source and preprocessing
- Numerical solver settings
- Python environment and library versions

This repository prioritizes research clarity over strict reproducibility.

## Intended Use

This code is intended for:
- Academic research
- Method exploration and validation
- Internal experimentation

It is **not** intended for production deployment.

## License

This repository is provided for research and educational use only.
