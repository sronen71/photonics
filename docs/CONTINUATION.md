# Two-mode LLE and pseudo-arclength continuation

## Liu et al. two-mode normalization

`two_mode.py` implements Appendix A of Yang Liu et al., *Breaking the
bandwidth-efficiency trade-off in soliton microcombs via mode coupling* (2025).
There is one periodic comb field $A(\theta,\tau)$ and one complex scalar
auxiliary amplitude $B(\tau)$. The auxiliary spatial mode is homogeneous, so
it has no $\theta$ grid and no auxiliary dispersion operator.

With

$$
\tau=\frac{\kappa_1t}{2},\qquad
A=\sqrt{\frac{2g}{\kappa_1}}a,\qquad
B=\sqrt{\frac{2g}{\kappa_1}}b,
$$

the implemented equations are

$$
\partial_\tau A=-(1+i\alpha_1)A+iD(-i\partial_\theta)A
+i\left(|A|^2+r^2|B|^2\right)A+C_{12}B+F_1,
$$

$$
\dot B=-(\gamma+i\alpha_2)B
+i\left(|B|^2+r^2\sum_\mu|A_\mu|^2\right)B+C_{21}A_0+F_2,
$$

where

$$
\alpha_j=\frac{2\omega_j}{\kappa_1},\quad
\gamma=\frac{\kappa_2}{\kappa_1},\quad
D(\mu)=-\frac{2D_{\mathrm{int}}(\mu)}{\kappa_1},
$$

$$
C_{12}=i\frac{K_c}{\kappa_1}
-\frac{\sqrt{\kappa_{\mathrm{ex},1}\kappa_{\mathrm{ex},2}}}{\kappa_1},
\qquad
C_{21}=i\frac{K_c^*}{\kappa_1}
-\frac{\sqrt{\kappa_{\mathrm{ex},1}\kappa_{\mathrm{ex},2}}}{\kappa_1},
$$

and

$$
F_j=\sqrt{\frac{8g\kappa_{\mathrm{ex},j}P_{\mathrm{in}}}
{\kappa_1^3\hbar\omega_0}}.
$$

The FFT convention is

$$
A_\mu=\frac{1}{N}\operatorname{FFT}(A)_\mu.
$$

Consequently, $A_0=\langle A\rangle_\theta$ and Parseval's identity gives
$\sum_\mu|A_\mu|^2=\langle|A|^2\rangle_\theta$. This is why the code contains
an explicit mean comb power in the $B$ equation, but no uncontracted $\mu$
index.

`TwoModeParameters.from_physical(...)` performs the normalization above.
`solve_two_mode_lle(...)` uses a Strang split integrator. The linearly coupled
pair $(A_0,B)$, including both input drives, is propagated by one exact affine
matrix exponential. Every other comb mode has its exact dispersive cavity
flow, and the self- and cross-phase subflow is also exact.

## Continuation problem contract

`continuation.py` treats a steady model as a square real residual

$$
R(u,\lambda)=0,
$$

where $u\in\mathbb R^n$ and $\lambda$ is one scalar continuation parameter.
The model owns all packing and physical constraints. This keeps the
continuation algorithm independent of whether $u$ represents one complex
field, two fields, a scalar auxiliary mode, or additional drift variables.

For a localized periodic state, translation makes the physical Jacobian
singular. Such a model must supply a square, phase-conditioned residual before
continuation. Public adapters are provided for all localized models:

- `steady_solver.scalar_phase_conditioned_residual`
- `two_mode.phase_conditioned_real_residual`
- `bidirectional_steady.bidirectional_phase_conditioned_residual`

Each adapter adds one shift-rate or velocity unknown and one translation phase
condition. With even dispersion, the converged shift rate should be zero. With
odd dispersion, it is the physical moving-frame velocity.

## Algorithm

Two ordinary fixed-parameter solves initialize the branch. At every subsequent
step, the implementation:

1. normalizes the secant through the preceding two branch points;
2. predicts one weighted arclength step along that tangent;
3. solves the physical equations together with

   $$
   \left\langle t, z-z_{\mathrm{pred}}\right\rangle_W=0,
   \qquad z=(u,\lambda);
   $$

4. orients the new secant consistently with the previous tangent;
5. grows or shrinks the step according to corrector convergence, retrying a
   failed correction down to the configured minimum step.

The corrector uses SciPy's matrix-free Newton-Krylov solver. The default state
metric is a mean-square norm rather than a sum, so refining an FFT grid does
not automatically increase the state contribution to arclength. Use
`parameter_scale` to express the parameter change that should count as one
unit, and `state_weights` when different state components require explicit
physical weighting.

## Two-mode detuning branch

Start from an analytic or time-integrated soliton seed. Pump-frequency tuning
changes both intrinsic detunings by the same amount, which
`parameters_with_pump_detuning` handles explicitly:

```python
from continuation import initialize_branch, pseudo_arclength_continuation
from drift import translation_gauge
from two_mode import (
    pack_phase_conditioned_state,
    parameters_with_pump_detuning,
    phase_conditioned_real_residual,
)

reference, phase_direction = translation_gauge(comb_seed)
state_seed = pack_phase_conditioned_state(comb_seed, auxiliary_seed, 0.0)

def residual(state, alpha_comb):
    point_parameters = parameters_with_pump_detuning(
        parameters, alpha_comb
    )
    return phase_conditioned_real_residual(
        state,
        point_parameters,
        reference,
        phase_direction,
    )

first, second = initialize_branch(
    residual,
    state_seed,
    parameters.alpha_comb,
    parameter_step=0.05,
)
branch = pseudo_arclength_continuation(
    residual,
    first,
    second,
    number_of_steps=100,
    step_size=0.05,
    minimum_step_size=1.0e-4,
    maximum_step_size=0.2,
    parameter_scale=1.0,
)
```

To continue uncoupled mode spacing instead, replace only
`alpha_auxiliary`. To continue pump power, use
`parameters_with_pump_scale`; its argument is an amplitude factor, so input
power changes by the square of that factor. A two-dimensional existence map
is obtained with nested one-parameter branches, for example one detuning
branch at each selected pump power or mode spacing.

## Existing model adapters

The same driver continues a localized scalar state:

```python
from continuation import initialize_branch, pseudo_arclength_continuation
from drift import translation_gauge
from steady_solver import (
    pack_phase_conditioned_state,
    scalar_phase_conditioned_residual,
)

reference, direction = translation_gauge(scalar_seed)
state_seed = pack_phase_conditioned_state(scalar_seed)

def residual(state, alpha):
    return scalar_phase_conditioned_residual(
        state, alpha, forcing, dispersion, reference, direction
    )
```

For the bidirectional model, form the gauge from both fields and replace the
desired dataclass parameter at each branch point:

```python
from dataclasses import replace

import numpy as np

from bidirectional_steady import (
    bidirectional_phase_conditioned_residual,
    pack_phase_conditioned_fields,
)
from drift import translation_gauge

reference, direction = translation_gauge(
    np.stack((forward_seed, backward_seed))
)
state_seed = pack_phase_conditioned_fields(forward_seed, backward_seed)

def residual(state, alpha):
    return bidirectional_phase_conditioned_residual(
        state, replace(parameters, alpha=alpha), reference, direction
    )
```

The two initialization and continuation calls are identical to the two-mode
example.

## Stability and branch acceptance

Pseudo-arclength continuation finds mathematical steady solutions, including
unstable ones. Dynamic stability is a separate calculation. Evaluate
eigenvalues of the linearized physical evolution equation at each point. Do
not interpret eigenvalues of the pseudo-arclength row, the translation phase
condition, or the auxiliary shift-rate border as physical modes. The neutral
translation eigenvalue should be identified separately when classifying a
localized branch.
