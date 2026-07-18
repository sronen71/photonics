# Photonics LLE examples

Small Python solvers for the normalized Lugiato--Lefever equation (LLE) on a
periodic ring. Generated figures and data are written to `results/`.

The convention used here is

```text
dA/dt = -(1 + i*alpha)A + i|A|^2 A
        - i*(beta/2)*d^2A/dtheta^2 + F.
```

It is the convention of Godey et al.: `t = kappa*t_physical/2`, where `kappa`
is the loaded-cavity linewidth; positive `alpha` is red pump detuning; and
`beta < 0` is anomalous dispersion.

The time-dependent solver uses a split-step Fourier method. It alternates exact
half-step evolution under the driven linear cavity (loss, detuning, dispersion,
and pump) with exact Kerr phase evolution.

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

## Physics validation

Run the scalar-LLE physics benchmarks with

```bash
python3 -m unittest discover -s tests -v
```

The benchmarks check the following literature predictions:

- The homogeneous response
  `|F|^2 = I [1 + (alpha - I)^2]`, including the bistability cusp at
  `alpha = sqrt(3)` and its two fold points.
- Exact small-signal cavity filling and ringdown, including the phase acquired
  by a nonzero azimuthal mode from second-order dispersion.
- The modulational-instability gain of the `alpha=1`, `beta=-0.04`, `I=1.2`
  example in Godey et al. The test seeds the unstable eigenvector at mode 8 and
  measures its exponential growth rate.
- The large-detuning bright-soliton laws `I_peak approximately 2*alpha` and
  `FWHM approximately 2*arcosh(sqrt(2))*sqrt(|beta|/(2*alpha))`, followed by an
  independent time-domain check that the stationary pulse remains stationary.

Primary references:

- C. Godey et al., [Stability analysis of the spatiotemporal
  Lugiato--Lefever model](https://doi.org/10.1103/PhysRevA.89.063814),
  Phys. Rev. A 89, 063814 (2014).
- S. Coen and M. Erkintalo, [Universal scaling laws of Kerr frequency
  combs](https://doi.org/10.1364/OL.38.001790), Opt. Lett. 38, 1790 (2013).
