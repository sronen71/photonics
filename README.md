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

The path is resolved relative to the YAML file. Headered CSV files select one
of two direct dispersion representations; SI configurations can also use the
headerless pyLLE resonance format described below.

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
time-dependent solver represents that motion directly and measures its late-
time velocity. The steady solver checks the parity of $D(k)$ automatically.
For even dispersion it solves the simpler $v=0$ stationary equation. For odd
dispersion it solves
$R[U]+v\,\partial_\theta U=0$ for both the co-moving profile $U$ and velocity
$v$, with a phase condition fixing the arbitrary pulse position. A short
time integration supplies a stable moving-state seed before Newton refinement;
its duration and step are set by `steady.relaxation_time` and
`steady.relaxation_dt`.

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

The SI configuration also accepts pyLLE's headerless resonance-file format
through the same `dispersion_csv` setting:

```csv
210,190785936704274.4
211,190885932958774.4
212,190985931710274.4
213,191085932958774.4
214,191185936704274.4
```

The first column is the absolute integer azimuthal mode order and the second
is its resonance frequency in Hz. The row nearest `omega_0_rad_s/(2*pi)` is
the pumped mode. As in pyLLE, the co-rotating-grid frequency $D_1/(2\pi)$ is
the linear coefficient of a quadratic fit to that mode and the two adjacent
modes on each side. The loader then computes
$D_{\rm int}(k)=2\pi[f_m-f_{m_0}-kD_1/(2\pi)]$ and applies the usual
$D(k)=-2D_{\rm int}(k)/\kappa$ normalization. Consequently, the file must
include those five consecutive modes, and its mode range must cover every FFT
mode in the simulation. Unlike pyLLE, this loader does not spline-extrapolate
beyond the supplied resonance data.

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

After integration, the solver evaluates the fixed-frame LLE residual and fits
translation over the spectrum-averaging window. It distinguishes a stationary
field from a rigidly translating profile and saves the fitted velocity,
linearity error, aligned-profile variation, and moving-frame residual.

Find a steady bright soliton in its automatically selected frame (`steady`
section):

```bash
python3 steady_solver.py
```

The steady-state figure includes the co-moving complex field, spatial
intensity, drift velocity, and an output spectrum. With `physics.units: SI`,
the spectrum panel shows
through-port power in dBm versus optical frequency in THz. It includes
interference between the input pump and cavity leakage at the pumped mode;
sidebands contain cavity leakage only. With `physics.units: normalized`, it
instead shows normalized spectral power in dB versus mode number.

For a stationary comb, SI frequency labels use the nominal grid
$\omega_p+2\pi\,\mathrm{FSR}\,k$. For a rigidly drifting comb, the measured
normalized velocity gives
$\delta f_{\rm rep}=\kappa v/(4\pi)$, and the labels instead use
$\omega_p+2\pi(\mathrm{FSR}+\delta f_{\rm rep})k$. Modal powers do not change
under rigid translation. Breathing and chaotic states retain the nominal axis
and a time-averaged power envelope; slow-time modulation sidebands are not
resolved.

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

## Comb plus homogeneous auxiliary mode

`two_mode.py` implements the coupled equations in Appendix A of Liu et al.
(2025). It evolves one dispersive comb field $A(\theta,t)$ and one homogeneous
auxiliary amplitude $B(t)$. In normalized form,

$$
\dot A=-(1+i\alpha_1)A+iD A
+i(|A|^2+r^2|B|^2)A+C_{12}B+F_1,
$$

$$
\dot B=-(\gamma+i\alpha_2)B
+i\left(|B|^2+r^2\sum_\mu|A_\mu|^2\right)B+C_{21}A_0+F_2.
$$

Here $A_0=\operatorname{mean}(A)$ and
$\sum_\mu|A_\mu|^2=\operatorname{mean}(|A|^2)$ under the implemented FFT
normalization. The auxiliary variable is deliberately a scalar: the paper's
homogeneous-mode assumption excludes auxiliary comb generation and therefore
removes its dispersion term. `TwoModeParameters.from_physical` maps the
dimensional linewidths, detunings, coupling, pump power, and integrated
dispersion directly into the repository convention. `solve_two_mode_lle`
provides time integration, while `solve_two_mode_steady_state` removes the
localized comb's neutral translation mode during steady refinement.

## Pseudo-arclength continuation

`continuation.py` provides adaptive, one-parameter pseudo-arclength
continuation for any square real residual. It uses two seed-based
fixed-parameter solves, a weighted secant predictor, a matrix-free bordered
Newton corrector, retry with step reduction, and step growth after easy
corrections. Unlike ordinary detuning stepping, the continued parameter is an
unknown, so a branch can turn around at a saddle-node.

Public phase-conditioned residual adapters are available for localized scalar,
two-mode, and bidirectional states. They remove the neutral translation mode
without putting model physics into the continuation engine. See
[`docs/CONTINUATION.md`](docs/CONTINUATION.md) for the normalization,
algorithm, and complete usage examples.

## Bidirectional PhCR circuit

`bidirectional_solver.py` implements the coupled forward/backward model used
by Zang et al. for a photonic-crystal resonator (PhCR) followed by a finite-band
waveguide reflector. In this repository's modal-dispersion notation, the model
is

$$
\dot E^f_\mu=-(1+i\alpha)E^f_\mu+iD(\mu)E^f_\mu
+i\mathcal F(|E^f|^2E^f)_\mu+2iP_bE^f_\mu
+\delta_{\mu0}\left(F-\frac{i\epsilon_{\rm PhC}}2E^b_0\right),
$$

$$
\dot E^b_\mu=-(1+i\alpha)E^b_\mu+iD(\mu)E^b_\mu
+i\mathcal F(|E^b|^2E^b)_\mu+2iP_fE^b_\mu
+\delta_{\mu0}\left(rF-\frac{i\epsilon_{\rm PhC}}2E^f_0\right)
-I_\Omega(\mu)\gamma rE^f_\mu,
$$

where $P_{f,b}=\sum_j|E^{f,b}_j|^2$,
$\gamma=2K/(K+1)$, and $I_\Omega$ restricts reflection to the configured
mode band. The reflector coefficient is
$r=\sqrt R\exp(i\phi)$, exactly as it appears in the displayed equation.

The implementation keeps this repository's sign conventions. Positive
$\alpha=2(\omega_0-\omega_p)/\kappa$ is red pump detuning, and
$D(\mu)=\beta_2\mu^2/2=-d_2\mu^2/2$, so a dimensional coefficient is converted
with $\beta_2=-2D_2/\kappa$. This matches the displayed evolution equation and
positive-detuning scans in the paper while avoiding the opposite detuning sign
printed in its prose definition.

Run the paper benchmark configuration with

```bash
python3 bidirectional_solver.py
```

There is a $\pi$ inconsistency between the paper's displayed equation and its
plotted phase label. The linear pump matrix in that equation is
$-(1+i\alpha)I-i\epsilon_{\rm PhC}\sigma_x/2$. At the red split resonance
$\alpha=+\epsilon_{\rm PhC}/2$, its resonant eigenvector is proportional to
$(1,-1)$. Since the external drive is proportional to $(1,r)$, that red mode
is driven constructively by $r<0$, or equation phase $\phi=\pi$, while
$\phi=0$ drives the other split mode. The same group's preceding theory states
explicitly that the red mode has $E^f_0\simeq-E^b_0$ and maximum conversion at
$r=-1$ ([Liu et al., 2024](https://doi.org/10.1103/PhysRevLett.132.023801)).
The newer paper instead calls the constructive fabricated-device setting
$\phi=0$ but retains $r=\sqrt R\exp(i\phi)$ and the same negative coupling
sign. Thus its plotted device phase and literal equation phase differ by
$\pi$.

The configuration and code use the literal equation phase, with
`reflector_phase: pi`; no fitted phase-reference parameter is present. The
default starts from noise, scans to $\alpha=6.98$, time-averages both external
ports, and refines the final state with the phase-conditioned even-dispersion
solver. The localized refinement includes a translational phase condition:
this removes the neutral freedom to place the same pulse anywhere on the ring
while retaining the physical even-dispersion velocity $v=0$. It produces a
stable, predominantly backward comb with about 0.608
pump-to-comb conversion efficiency and about 0.013 remaining pump power.
Setting the literal equation phase to zero suppresses the comb by more than a
factor of 20 in the physics benchmark. This is a reconstruction after a
derived $\pi$ relabeling of the paper's plotted phase axis, not a fit of an
additional physical parameter. The configured $R=0.97$ is the upper end of the
paper's reported measured range.

### Standalone Supplementary Figure S1 reconstruction

Run the detuning scan used to reconstruct all three panels of Supplementary
Fig. S1 with

```bash
python3 reproduce/reproduce_figure_s1.py
```

The script uses the same bidirectional LLE and circuit input-output functions
as the normal solver. By default it downloads the official
[Supplementary Figure source data](https://www.nature.com/articles/s41566-025-01624-1#Sec13),
creates a paper-style figure, overlays the published arrays with the new
simulation, and writes objective comparison metrics under `results/figure_s1/`.
Use `--paper-data PATH` for an existing
`SourceData_Supplemental_Information_S1.xlsx`, or
`--no-paper-comparison` for an offline simulation-only run.

The paper specifies $K=4.5$ for Fig. 1d/S1, in contrast with the $K=3$
Fig. 1e/S2 configuration above. It does not publish every S1 simulation input
or the detuning-scan protocol. The standalone reconstruction therefore labels
the following choices as inferred rather than reported: $F=2.0$, $R=0.90$, the
two scan rates, and both stochastic perturbations. It uses literal equation
phase $\phi=\pi$, which is the paper's constructive device phase after the
derived phase relabeling.

An exactly homogeneous initial field remains homogeneous under the published
deterministic LLE, so a noiseless reconstruction has exactly zero non-pump
power at $\alpha=4.7$. The published simulation instead contains weak
precursor sidebands. The default reconstruction applies a weak Gaussian modal
Wiener drive during only the pre-soliton scan, with normalized strength
$3\times10^{-5}$ per square root of normalized time, modal width 3, and cutoff
$\mu\in\{-19,\ldots,19\}$. The drive is explicitly an inferred symmetry-breaking
model, not a method reported by the authors, and it is disabled before the
localized branch is followed. Scalar noise in $F(t)$ is not used because the
paper's $\delta_{\mu0}F$ drive excites only the pump mode and cannot break the
azimuthal symmetry of an exactly homogeneous state.

At $\alpha=4.8$, a one-time Gaussian modal seed with RMS amplitude 0.02 and
modal width 19 accesses the subcritical soliton branch. Its normalized Fourier
amplitudes are independent of the spatial grid. This replaces the former
point-space kick, whose modal strength changed with the number of grid points.
Both perturbations and their random seeds are written to the metrics report and
can be varied from the command line.

At the default 256-point resolution, comparison with the official simulated
source arrays gives backward CE 0.6685 versus 0.6741 at $\alpha=6.98$,
remaining pump 0.0553 versus 0.0360, collapse detuning 7.5859 versus 7.5600,
an overall power-trace RMSE of 0.0470, and spectral-map correlations of 0.716
forward and 0.931 backward. The selected backward-spectrum median RMSE is
13.3 dB; most of that difference is in weak spectral wings. At $\alpha=4.7$,
the backward-spectrum RMSE falls from the noiseless 53.8 dB to 29.8 dB, and
the precursor backward-comb log-power RMSE falls from 4.80 to 1.98 decades.
The remaining mismatch is reported rather than hidden by a stronger fitted
noise floor. This is a standalone quantitative audit, not an additional unit
test.

The 256- and 512-point runs agree to numerical roundoff at fixed time step.
Halving the time step leaves formation and collapse unchanged and changes the
$\alpha=6.98$ CE by $3.1\times10^{-5}$; an individual stochastic precursor
snapshot changes by a few decibels, as expected, while its distribution is
time-step independent. After holding the $\alpha=6.98$ state for 100 normalized
time units, its CE has relative variation $6.2\times10^{-15}$. Steady-state
refinement then gives a maximum continuous-equation residual of
$8.9\times10^{-14}$.

The numerical pieces are intentionally separate:

- `bidirectional.py` contains only the coupled residual and exact split-step
  subflows.
- `bidirectional_steady.py` phase-conditions and refines the paper's
  even-$D_2$, $v=0$ state.
- `bidirectional_spectrum.py` applies the circuit scattering relations and
  reports both port spectra, conversion efficiency, pump consumption, and the
  steady power budget.
- `stochastic.py` defines grid-normalized Gaussian modal seeds and Wiener
  increments shared by the scan protocol and the bidirectional solver.
- `integration.py`, `spectral.py`, and `spectrum.time_average` are shared by
  the scalar and bidirectional solvers.

The runner saves the dynamic histories, the refined steady fields, both port
spectra, power-flow traces, and residuals in
`results/bidirectional_lle_output.npz`; its summary figure is
`results/bidirectional_lle_response.png`. The steady refiner deliberately uses
$v=0$ after verifying even dispersion. Its auxiliary shift rate only borders
the translation null mode, and the original fixed-frame residual must still
meet tolerance. A future bidirectional odd-dispersion extension would instead
solve for one physical shared drift velocity, as the scalar traveling-state
solver already does.

## Physics validation

Run all physics benchmarks with

```bash
python3 -m unittest discover -s tests -v
```

The benchmarks separate exact consequences of the normalized LLE from
large-detuning asymptotic checks.

Exact benchmarks:

- Liu's two-mode normalization, shared-bus and intracavity coupling phases,
  Parseval reduction of $\sum_\mu|A_\mu|^2$, exact linear and Kerr subflows,
  and recovery of the full Appendix A vector field in the small-step limit.
- Pseudo-arclength traversal of the exact saddle-node
  $x^2-\lambda=0$, including parameter reversal, plus direct continuation of
  the repository's scalar-LLE residual.
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
  spectrum in a moving frame from a stationary field, verifies constant drift,
  refines the profile and velocity to a moving-frame root, and verifies that
  the corresponding `d3=0` soliton breathes strongly.
- The dimensional normalization and through-port input-output relation. At
  critical coupling, a Kerr-shifted resonant continuous wave must cancel the
  incident pump at the through port, while a sideband has the expected
  out-coupled photon flux and optical frequency. A separate check verifies the
  drift-induced repetition-rate correction.
- The exact finite-window average power of a freely decaying cavity mode,
  which checks the time-averaged spectrum calculation.
- The bidirectional PhCR model of Zang et al.: dimensional-dispersion sign
  conversion, reduction to the scalar LLE when coupling is disabled, exact
  modal reflector propagation, factor-two cross-phase modulation, steady port
  energy conservation, the red split-mode phase derivation, the high-efficiency
  contrast after relabeling the plotted phase axis by $\pi$, and the common
  translation orbit of the localized steady state.
- Complex Gaussian modal increments have the prescribed Wiener variance and
  Gaussian envelope, while both continuous increments and one-time modal
  seeds retain identical normalized Fourier amplitudes under grid refinement.

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
- J. Zang et al., [Laser power consumption of soliton formation in a
  bidirectional Kerr resonator](https://doi.org/10.1038/s41566-025-01624-1),
  Nature Photonics 19, 510--517 (2025).
- H. Liu et al., [Threshold and laser conversion in nanostructured-resonator
  parametric oscillators](https://doi.org/10.1103/PhysRevLett.132.023801),
  Phys. Rev. Lett. 132, 023801 (2024).
