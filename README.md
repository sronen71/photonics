# Photonics LLE examples

Small Python solvers for the normalized Lugiato--Lefever equation (LLE) on a
periodic ring. Generated figures and data are written to `results/`.

## Model

The normalized LLE solved here is

$$
\frac{\partial A}{\partial t}
= -(1+i\alpha)A+i|A|^2A
-i\frac{\beta}{2}\frac{\partial^2 A}{\partial\theta^2}+F,
\qquad A(\theta+2\pi,t)=A(\theta,t).
$$

Here, $A(\theta,t)$ is the complex intracavity field, $\alpha$ is detuning,
$\beta$ is dispersion, and $F$ is the real pump amplitude.

The `lle` and `steady` solvers can instead use an arbitrary real modal
dispersion $D(k)$:

$$
\frac{\partial A}{\partial t}
=-(1+i\alpha)A+i|A|^2A+iD(-i\partial_\theta)A+F.
$$

For the default quadratic model, $D(k)=\beta k^2/2$, so this is identical to
the equation above.

They can also start from the dimensional photon-amplitude equation

$$
\frac{\partial a}{\partial t_{\mathrm{phys}}}
=-\left(\frac{\kappa}{2}+i(\omega_0-\omega_p)\right)a
+ig_0|a|^2a-iD_{\rm int}(-i\partial_\theta)a
+\sqrt{\kappa_{\rm ex}}s_{\rm in},
\qquad |s_{\rm in}|^2=\frac{P_{\rm in}}{\hbar\omega_p}.
$$

Here $|a|^2$ is photon number, $\kappa$ is the loaded full-width angular
linewidth, $\kappa_{\rm ex}$ is its external-coupling contribution, $g_0$ is
the single-photon Kerr shift, and
$D_{\rm int}(k)=\omega_k-(\omega_0+2\pi\,\mathrm{FSR}\,k)$.

The internal conversion is

$$
\tau=\frac{\kappa t_{\rm phys}}{2},\qquad
A=\sqrt{\frac{2g_0}{\kappa}}a,\qquad
\alpha=\frac{2(\omega_0-\omega_p)}{\kappa},
$$

$$
F=\sqrt{\frac{8g_0\kappa_{\rm ex}P_{\rm in}}
{\kappa^3\hbar\omega_p}},qquad
D(k)=-\frac{2D_{\rm int}(k)}{\kappa}.
$$

For $D_{\rm int}(k)=D_2k^2/2$, this gives
$\beta=-2D_2/\kappa$. The FSR establishes the co-rotating frame and therefore
cancels from the normalized dynamics, but it is retained for physical
interpretation.

This is the convention of Godey et al.: $t=\kappa t_{\mathrm{physical}}/2$,
where $\kappa$ is the loaded-cavity linewidth; positive $\alpha$ is red pump
detuning; and $\beta<0$ is anomalous dispersion.

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

Edit `configs/config.yaml`, then run a solver without parameter flags. The `physics`
section supplies the shared detuning, pump, and dispersion used by both the
`lle` and `steady` solvers; their numerical settings remain in separate
sections.

### Dispersion CSV

By default, `physics.beta` selects the original quadratic dispersion. To use
an external relation, replace `beta` with a CSV path:

```yaml
physics:
  units: normalized
  alpha: 4.0
  f_real: 2.0
  dispersion_csv: dispersion.csv
```

The path is resolved relative to the YAML file. The CSV header selects one of
two representations.

For a polynomial, use `order,beta` rows:

```csv
order,beta
2,-0.02
3,0.0001
```

These coefficients define
$D(k)=\sum_n \beta_n k^n/n!$. Thus a file containing only
`2,-0.02` exactly reproduces the default `beta: -0.02` model.

For tabulated values, use `k,dispersion` rows:

```csv
k,dispersion
-256,-655.36
-255,-650.25
0,0.0
254,-645.16
255,-650.25
```

Here `dispersion` is $D(k)$ itself, not $\beta$. Values are linearly
interpolated, and the grid must cover every FFT mode used by the solver:
$-N/2$ through $N/2-1$ for even `spatial_points: N`. Grid rows may be unevenly
spaced, and comment lines beginning with `#` are ignored.

The analytic bright-soliton initial shape uses the polynomial $\beta_2$. For a
grid, it uses the local discrete curvature
$D(1)-2D(0)+D(-1)$. This coefficient must be negative when `initial_shape` is
`soliton` and for the steady solver's bright-soliton seed; the actual solve
always uses the complete CSV relation.

An odd part of $D(k)$, such as third-order dispersion, breaks reflection
symmetry and generally makes a localized state drift around the ring. The
time-dependent solver represents that motion directly. The steady solver
solves only the stationary equation in the selected co-rotating frame and
does not solve for an unknown drift velocity. A drifting state can therefore
have steady modal powers while its fixed-frame stationary residual remains
large. A linear term in $D(k)$ selects a different co-rotating frame when that
frame velocity is known.

### Physical parameters

Set `physics.units` to `SI` to derive the normalized detuning, pump, and
dispersion internally:

```yaml
physics:
  units: SI
  kappa_rad_s: 1.0e9
  kappa_external_rad_s: 5.0e8
  omega_0_rad_s: 1.2e15
  omega_pump_rad_s: 1.199998e15
  fsr_hz: 1.0e11
  g_0_rad_s: 6.0
  pump_power_w: 0.0211
  d2_rad_s: 1.0e7
```

All quantities with the `_rad_s` suffix are angular frequencies in rad/s;
`fsr_hz` is in Hz and `pump_power_w` is in watts. `kappa_rad_s` is the total
loaded linewidth, not the half-width. The complete runnable example
`configs/config_physical.yaml` maps to approximately the same
`alpha: 4`, `F: 2`, and `beta: -0.02` as the default normalized configuration:

```bash
python3 lle_solver.py --config configs/config_physical.yaml
python3 steady_solver.py --config configs/config_physical.yaml
```

For physical polynomial dispersion, replace `d2_rad_s` with
`dispersion_csv` and use dimensional $D_n$ coefficients in rad/s:

```csv
order,d
2,1.0e7
3,2500.0
```

This defines $D_{\rm int}(k)=\sum_n D_nk^n/n!$. A physical grid keeps the
same `k,dispersion` header described above, but its `dispersion` values are
dimensional $D_{\rm int}(k)$ in rad/s. Do not include the FSR term in either
CSV representation. Relative CSV paths are resolved beside the YAML file.

The `lle` times (`dt`, `final_time`, `scan_time`, and
`spectrum_average_time`) remain normalized times. For `physics.units: SI`,
each solver prints the corresponding seconds per
normalized time unit, $2/\kappa$.

Plot the uniform response curves (`uniform` section):

```bash
python3 uniform_solver.py
```

Integrate the time-dependent LLE (`lle` section):

```bash
python3 lle_solver.py
```

The time-dependent output spectrum replaces the former final-snapshot mode
plot. Each saved field in the final `spectrum_average_time` interval is
Fourier transformed around the ring, and the resulting powers are averaged in
time. The default interval is the final 5 normalized time units. SI and
normalized axes use the same conventions as the steady solver, and the data
are saved to `results/lle_output_spectrum.npz`.

After integration, the solver also evaluates the stationary LLE residual of
the final field. It reports whether the maximum residual satisfies
`lle.stationary_tolerance`; this report does not stop a valid time-dependent
solution that remains nonstationary.

Find a stationary bright soliton (`steady` section):

```bash
python3 steady_solver.py
```

The steady-state figure includes the complex field, spatial intensity, and an
output spectrum. With `physics.units: SI`, the spectrum panel shows
through-port power in dBm versus optical frequency in THz. It includes
interference between the input pump and cavity leakage at the pumped mode;
sidebands contain cavity leakage only. With `physics.units: normalized`, it
instead shows normalized spectral power in dB versus mode number.

The SI frequency labels use the nominal grid
$\omega_p+2\pi\,\mathrm{FSR}\,k$. This is exact for a stationary comb in that
co-rotating frame. For a drifting or breathing state, the saved values are
time-averaged modal powers rather than a slow-time-resolved optical spectrum;
the code does not infer repetition-rate shifts or resolve breathing sidebands.

`results/steady_solution.npz` stores the underlying spectrum. SI results use
`output_frequency_thz`, `output_power_w`, and `output_power_dbm`; normalized
results use `output_mode_number`, `output_normalized_power`, and
`output_normalized_power_db`. Exact SI powers are saved in watts before the
finite plotting floor is applied to the dBm array.

An alternate common configuration file can be selected with the only supported
parameter flag:

```bash
python3 steady_solver.py --config configs/soliton.yaml
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

The benchmarks separate exact consequences of the normalized LLE from
large-detuning asymptotic checks.

Exact benchmarks:

- The homogeneous response
  `|F|^2 = I [1 + (alpha - I)^2]`, including the bistability cusp at
  `alpha = sqrt(3)` and its two fold points.
- Exact small-signal cavity filling and ringdown, including the opposite phase
  shifts acquired by a nonzero azimuthal mode for normal and anomalous
  second-order dispersion.
- The modulational-instability gain of the `alpha=1`, `beta=-0.04`, `I=1.2`
  anomalous-dispersion example in Godey et al. The test seeds the unstable
  eigenvector at mode 8 and measures its exponential growth rate.
- The stable normal-dispersion dark cavity soliton reported by Godey et al. at
  `alpha=2.5`, `beta=+0.0125`, and `F^2=2.61`. A pulse-like perturbation must
  relax to one stationary intensity dip on the upper CW background with a
  nonzero first comb sideband.
- The third-order-dispersion stabilization reported by Parra-Rivas et al. at
  `theta=6.1`, `u0=4`, and `d3=0.15`. The test distinguishes a steady comb
  spectrum in a moving frame from a stationary field, and verifies that the
  corresponding `d3=0` soliton breathes strongly.
- The dimensional normalization and through-port input-output relation. At
  critical coupling, a Kerr-shifted resonant continuous wave must cancel the
  incident pump at the through port, while a sideband has the expected
  out-coupled photon flux and optical frequency.
- The exact finite-window average power of a freely decaying cavity mode,
  which checks the time-averaged spectrum calculation.

Asymptotic consistency check:

- Coen and Erkintalo's large-detuning bright-soliton laws
  `I_peak approximately 2*alpha` and
  `FWHM approximately 2*arcosh(sqrt(2))*sqrt(|beta|/(2*alpha))`. These are not
  exact finite-detuning LLE solutions. At `alpha=10`, `F=sqrt(10)`, and
  `beta=-0.2`, the spatially converged stationary solution has peak intensity
  `21.090354` versus the asymptotic value `20`, and background-subtracted pulse
  FWHM `0.172077` versus `0.176275`. The independent time-domain solver then
  checks that this numerical stationary solution remains stationary.

Primary references:

The exact publicly available author-manuscript PDFs used for these benchmarks
are indexed in [docs/LLE_REFERENCES.md](docs/LLE_REFERENCES.md).

- C. Godey et al., [Stability analysis of the spatiotemporal
  Lugiato--Lefever model](https://doi.org/10.1103/PhysRevA.89.063814),
  Phys. Rev. A 89, 063814 (2014).
- S. Coen and M. Erkintalo, [Universal scaling laws of Kerr frequency
  combs](https://doi.org/10.1364/OL.38.001790), Opt. Lett. 38, 1790 (2013).
- Y. K. Chembo and C. R. Menyuk, [Spatiotemporal Lugiato--Lefever formalism
  for Kerr-comb generation in whispering-gallery-mode
  resonators](https://doi.org/10.1103/PhysRevA.87.053852), Phys. Rev. A 87,
  053852 (2013).
- P. Parra-Rivas et al., [Third-order chromatic dispersion stabilizes Kerr
  frequency combs](https://doi.org/10.1364/OL.39.002971), Opt. Lett. 39,
  2971 (2014).
