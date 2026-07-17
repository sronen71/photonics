# Photonics LLE examples

Small Python solvers for the normalized Lugiato--Lefever equation (LLE) on a
periodic ring. Generated figures and data are written to `results/`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Examples

Plot the uniform steady-state response curves:

```bash
python3 steady_state_solver.py
```

Integrate the time-dependent LLE from a noisy uniform state:

```bash
python3 lle_solver.py --initial-guess pattern --alpha 2 --f-real 1.8 --beta -0.02
```

Find a stationary nonuniform pattern:

```bash
python3 nonuniform_steady_solver.py --initial-guess pattern --alpha 2 --f-real 1.8 --beta -0.02
```

Find a stationary bright soliton:

```bash
python3 nonuniform_steady_solver.py --initial-guess soliton --alpha 4 --f-real 2 --beta -0.02
```

Bright-soliton seeds require anomalous dispersion (`--beta < 0`). Run
`lle_solver.py` or `nonuniform_steady_solver.py` with `--help` to see all
available parameters.
