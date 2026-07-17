# Photonics LLE examples

Small Python solvers for the normalized Lugiato--Lefever equation (LLE) on a
periodic ring. Generated figures and data are written to `results/`.

The time-dependent solver uses a pyLLE-style iterative split-step Fourier
method: an explicit pump update, spectral linear half-steps, and an exponential
Kerr step refined by fixed-point iteration. The `lle` section sets
`split_step_tolerance` and `split_step_max_iterations` for that inner solve.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Examples

Edit `config.yaml`, then run a solver without parameter flags. The `physics`
section supplies the shared `alpha`, pump, and `beta` values used by both the
`lle` and `steady` solvers; their numerical settings remain in separate
sections.

Plot the uniform response curves (`uniform` section):

```bash
python3 uniform_solver.py
```

Integrate the time-dependent LLE (`lle` section):

```bash
python3 lle_solver.py
```

Find a stationary bright soliton (`steady` section):

```bash
python3 steady_solver.py
```

An alternate common configuration file can be selected with the only supported
parameter flag:

```bash
python3 steady_solver.py --config soliton.yaml
```

The default physics and initial shape target a dissipative Kerr soliton (DKS).
In the LLE section, `initial_shape` selects `empty` or `soliton` independently
of the `direct` or `scan` `operation_mode`. The steady solver always starts
from the soliton seed before Newton--Krylov refinement.
