"""Comb-field plus homogeneous-auxiliary-mode LLE of Liu et al. (2025).

In the repository normalization, time is ``tau=kappa_1*t/2`` and both photon
amplitudes are multiplied by ``sqrt(2*g/kappa_1)``.  Appendix A then becomes

    dA/dtau = -(1+i*alpha_1)A + i*D A
              + i*(|A|^2+r^2|B|^2)A + C_12 B + F_1,

    dB/dtau = -(gamma+i*alpha_2)B
              + i*(|B|^2+r^2*sum_mu|A_mu|^2)B + C_21 A_0 + F_2.

``A`` is a periodic, dispersive comb field and ``B`` is one scalar amplitude.
With ``A_mu=FFT(A)_mu/N``, Parseval gives
``sum_mu|A_mu|^2=mean_theta|A|^2`` and ``A_0=mean_theta(A)``.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import expm
from scipy.optimize import NoConvergence, newton_krylov

from dispersion import DispersionRelation, as_dispersion
from drift import (
    spectral_derivative,
    translation_gauge,
    translation_phase_condition,
)
from integration import integrate_snapshots
from physics import HBAR_J_S
from spectral import mode_numbers


@dataclass(frozen=True)
class TwoModeParameters:
    """Normalized parameters of the comb and homogeneous auxiliary mode."""

    alpha_comb: float
    alpha_auxiliary: float
    forcing_comb: complex
    forcing_auxiliary: complex
    beta: object
    auxiliary_loss_ratio: float
    cavity_coupling: complex
    bus_coupling: float
    intermodal_ratio: float = 1.0

    def __post_init__(self):
        finite_real = (
            self.alpha_comb,
            self.alpha_auxiliary,
            self.auxiliary_loss_ratio,
            self.bus_coupling,
            self.intermodal_ratio,
        )
        finite_complex = (
            self.forcing_comb,
            self.forcing_auxiliary,
            self.cavity_coupling,
        )
        if not all(np.isfinite(float(value)) for value in finite_real):
            raise ValueError("two-mode real parameters must be finite")
        if not all(
            np.isfinite(complex(value).real)
            and np.isfinite(complex(value).imag)
            for value in finite_complex
        ):
            raise ValueError("two-mode complex parameters must be finite")
        if self.auxiliary_loss_ratio <= 0.0:
            raise ValueError("auxiliary_loss_ratio must be positive")
        if self.bus_coupling < 0.0:
            raise ValueError("bus_coupling must be nonnegative")
        if self.intermodal_ratio < 0.0:
            raise ValueError("intermodal_ratio must be nonnegative")

    @property
    def comb_to_auxiliary_coupling(self):
        """Return i*Kc*/kappa1-sqrt(kex1*kex2)/kappa1."""
        return 1j * np.conj(self.cavity_coupling) - self.bus_coupling

    @property
    def auxiliary_to_comb_coupling(self):
        """Return i*Kc/kappa1-sqrt(kex1*kex2)/kappa1."""
        return 1j * self.cavity_coupling - self.bus_coupling

    @property
    def cross_phase_coefficient(self):
        """Return the r^2 coefficient used in Appendix A."""
        return self.intermodal_ratio**2

    @classmethod
    def from_physical(
        cls,
        *,
        kappa_comb_rad_s,
        kappa_external_comb_rad_s,
        kappa_auxiliary_rad_s,
        kappa_external_auxiliary_rad_s,
        detuning_comb_rad_s,
        detuning_auxiliary_rad_s,
        nonlinear_shift_rad_s,
        pump_power_w,
        photon_angular_frequency_rad_s,
        integrated_dispersion,
        cavity_coupling_rad_s,
        intermodal_ratio=1.0,
    ):
        """Normalize the dimensional coefficients written in Appendix A.

        ``integrated_dispersion`` is dimensional ``D_int(mu)`` in rad/s.  A
        scalar is interpreted as ``D_2`` and a ``DispersionRelation`` may
        represent a higher-order polynomial or sampled relation.
        """
        kappa_comb = float(kappa_comb_rad_s)
        kappa_external_comb = float(kappa_external_comb_rad_s)
        kappa_auxiliary = float(kappa_auxiliary_rad_s)
        kappa_external_auxiliary = float(kappa_external_auxiliary_rad_s)
        nonlinear_shift = float(nonlinear_shift_rad_s)
        pump_power = float(pump_power_w)
        photon_frequency = float(photon_angular_frequency_rad_s)
        positive = (
            kappa_comb,
            kappa_auxiliary,
            nonlinear_shift,
            photon_frequency,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError(
                "linewidths, nonlinear shift, and photon frequency must be positive"
            )
        if (
            not np.isfinite(kappa_external_comb)
            or not 0.0 <= kappa_external_comb <= kappa_comb
            or not np.isfinite(kappa_external_auxiliary)
            or not 0.0 <= kappa_external_auxiliary <= kappa_auxiliary
        ):
            raise ValueError(
                "external linewidths must lie between zero and total linewidth"
            )
        if not np.isfinite(pump_power) or pump_power < 0.0:
            raise ValueError("pump_power_w must be finite and nonnegative")

        dimensional_dispersion = as_dispersion(integrated_dispersion)
        dispersion_scale = -2.0 / kappa_comb
        normalized_dispersion = DispersionRelation(
            kind=f"normalized-{dimensional_dispersion.kind}",
            description=(
                "normalized integrated dispersion: "
                + dimensional_dispersion.description
            ),
            evaluator=lambda modes: (
                dispersion_scale * dimensional_dispersion.values(modes)
            ),
            seed_beta=(
                None
                if dimensional_dispersion.seed_beta is None
                else dispersion_scale * dimensional_dispersion.seed_beta
            ),
            source=dimensional_dispersion.source,
        )
        forcing_scale = np.sqrt(
            8.0
            * nonlinear_shift
            * pump_power
            / (kappa_comb**3 * HBAR_J_S * photon_frequency)
        )
        return cls(
            alpha_comb=2.0 * float(detuning_comb_rad_s) / kappa_comb,
            alpha_auxiliary=(
                2.0 * float(detuning_auxiliary_rad_s) / kappa_comb
            ),
            forcing_comb=forcing_scale * np.sqrt(kappa_external_comb),
            forcing_auxiliary=(
                forcing_scale * np.sqrt(kappa_external_auxiliary)
            ),
            beta=normalized_dispersion,
            auxiliary_loss_ratio=kappa_auxiliary / kappa_comb,
            cavity_coupling=(
                complex(cavity_coupling_rad_s) / kappa_comb
            ),
            bus_coupling=(
                np.sqrt(kappa_external_comb * kappa_external_auxiliary)
                / kappa_comb
            ),
            intermodal_ratio=float(intermodal_ratio),
        )


@dataclass(frozen=True)
class TwoModeLinearStep:
    """Exact affine linear flow for all comb modes and scalar B."""

    comb_propagator: np.ndarray
    pump_propagator: np.ndarray
    pump_drive_increment: np.ndarray


def two_mode_residual(comb, auxiliary, parameters):
    """Return the continuous-equation residuals ``dA/dtau`` and ``dB/dtau``."""
    comb = np.asarray(comb, dtype=complex)
    auxiliary_array = np.asarray(auxiliary, dtype=complex)
    if comb.ndim != 1 or comb.size < 2 or auxiliary_array.ndim != 0:
        raise ValueError("comb must be a 1-D field and auxiliary must be scalar")
    auxiliary = complex(auxiliary_array)
    modes = mode_numbers(comb.size)
    modal_dispersion = as_dispersion(parameters.beta).values(modes)
    dispersed_comb = np.fft.ifft(modal_dispersion * np.fft.fft(comb))
    comb_power = float(np.mean(np.abs(comb) ** 2))
    pump_mode = complex(np.mean(comb))
    cross_phase = parameters.cross_phase_coefficient
    comb_residual = (
        -(1.0 + 1j * parameters.alpha_comb) * comb
        + 1j * dispersed_comb
        + 1j
        * (np.abs(comb) ** 2 + cross_phase * abs(auxiliary) ** 2)
        * comb
        + parameters.auxiliary_to_comb_coupling * auxiliary
        + parameters.forcing_comb
    )
    auxiliary_residual = (
        -(
            parameters.auxiliary_loss_ratio
            + 1j * parameters.alpha_auxiliary
        )
        * auxiliary
        + 1j
        * (abs(auxiliary) ** 2 + cross_phase * comb_power)
        * auxiliary
        + parameters.comb_to_auxiliary_coupling * pump_mode
        + parameters.forcing_auxiliary
    )
    return comb_residual, complex(auxiliary_residual)


def linear_step_parameters(parameters, modes, duration, modal_dispersion=None):
    """Construct the exact driven linear subflow for one duration."""
    modes = np.asarray(modes)
    duration = float(duration)
    if modes.ndim != 1 or duration < 0.0 or not np.isfinite(duration):
        raise ValueError("modes must be one-dimensional and duration nonnegative")
    zero_indices = np.flatnonzero(modes == 0)
    if zero_indices.size != 1:
        raise ValueError("modes must contain exactly one pump mode")
    if modal_dispersion is None:
        modal_dispersion = as_dispersion(parameters.beta).values(modes)
    modal_dispersion = np.asarray(modal_dispersion, dtype=float)
    if modal_dispersion.shape != modes.shape:
        raise ValueError("modal_dispersion must match modes")

    comb_generator = (
        -(1.0 + 1j * parameters.alpha_comb) + 1j * modal_dispersion
    )
    pump_index = int(zero_indices[0])
    coupled_generator = np.array(
        [
            [
                comb_generator[pump_index],
                parameters.auxiliary_to_comb_coupling,
            ],
            [
                parameters.comb_to_auxiliary_coupling,
                -(
                    parameters.auxiliary_loss_ratio
                    + 1j * parameters.alpha_auxiliary
                ),
            ],
        ],
        dtype=complex,
    )
    augmented = np.zeros((3, 3), dtype=complex)
    augmented[:2, :2] = coupled_generator
    augmented[:2, 2] = (
        parameters.forcing_comb,
        parameters.forcing_auxiliary,
    )
    affine_flow = expm(duration * augmented)
    return TwoModeLinearStep(
        comb_propagator=np.exp(duration * comb_generator),
        pump_propagator=affine_flow[:2, :2],
        pump_drive_increment=affine_flow[:2, 2],
    )


def apply_linear_step(comb, auxiliary, step):
    """Apply a precomputed exact linear subflow."""
    comb = np.asarray(comb, dtype=complex)
    auxiliary = complex(auxiliary)
    comb_modes = np.fft.fft(comb)
    propagated_modes = step.comb_propagator * comb_modes
    pump_state = step.pump_propagator @ np.array(
        [comb_modes[0] / comb.size, auxiliary], dtype=complex
    )
    pump_state += step.pump_drive_increment
    propagated_modes[0] = comb.size * pump_state[0]
    return np.fft.ifft(propagated_modes), np.asarray(pump_state[1])


def nonlinear_step(comb, auxiliary, duration, intermodal_ratio=1.0):
    """Apply the exact self- and cross-phase nonlinear subflow."""
    comb = np.asarray(comb, dtype=complex)
    auxiliary = complex(auxiliary)
    duration = float(duration)
    intermodal_ratio = float(intermodal_ratio)
    if (
        not np.isfinite(duration)
        or not np.isfinite(intermodal_ratio)
        or duration < 0.0
    ):
        raise ValueError("duration and intermodal_ratio must be finite")
    cross_phase = intermodal_ratio**2
    comb_power = float(np.mean(np.abs(comb) ** 2))
    auxiliary_power = abs(auxiliary) ** 2
    return (
        comb
        * np.exp(
            1j
            * duration
            * (np.abs(comb) ** 2 + cross_phase * auxiliary_power)
        ),
        np.asarray(
            auxiliary
            * np.exp(
                1j
                * duration
                * (auxiliary_power + cross_phase * comb_power)
            )
        ),
    )


def split_step(comb, auxiliary, step_size, linear_half_step, intermodal_ratio=1.0):
    """Advance one Strang split step of the two-mode model."""
    comb, auxiliary = apply_linear_step(comb, auxiliary, linear_half_step)
    comb, auxiliary = nonlinear_step(
        comb, auxiliary, step_size, intermodal_ratio
    )
    return apply_linear_step(comb, auxiliary, linear_half_step)


def solve_two_mode_lle(
    parameters,
    spatial_points,
    final_time,
    time_step,
    snapshots,
    *,
    initial_comb=0.0j,
    initial_auxiliary=0.0j,
    initial_noise=0.0,
    seed=0,
    parameter_schedule=None,
):
    """Integrate Liu's coupled LLE with exact linear and Kerr subflows.

    ``parameter_schedule(time)`` may return a replacement
    :class:`TwoModeParameters` at the midpoint of each integration step.
    """
    if spatial_points < 8 or final_time <= 0.0 or time_step <= 0.0:
        raise ValueError("spatial points and times must be positive")
    if initial_noise < 0.0:
        raise ValueError("initial_noise must be nonnegative")
    initial_comb_array = np.asarray(initial_comb, dtype=complex)
    if initial_comb_array.ndim == 0:
        comb = np.full(spatial_points, initial_comb_array, dtype=complex)
    elif initial_comb_array.shape == (spatial_points,):
        comb = initial_comb_array.copy()
    else:
        raise ValueError(
            "initial_comb must be scalar or contain spatial_points values"
        )
    auxiliary = np.asarray(initial_auxiliary, dtype=complex)
    if auxiliary.ndim != 0:
        raise ValueError("initial_auxiliary must be scalar")
    auxiliary = complex(auxiliary)
    rng = np.random.default_rng(seed)
    if initial_noise:
        comb += initial_noise * (
            rng.standard_normal(spatial_points)
            + 1j * rng.standard_normal(spatial_points)
        )
        auxiliary += initial_noise * (
            rng.standard_normal() + 1j * rng.standard_normal()
        )

    modes = mode_numbers(spatial_points)
    modal_dispersion = as_dispersion(parameters.beta).values(modes)
    constant_half_step = linear_step_parameters(
        parameters,
        modes,
        0.5 * time_step,
        modal_dispersion=modal_dispersion,
    )

    def advance(state, step_start, step_size):
        step_parameters = parameters
        if parameter_schedule is not None:
            step_parameters = parameter_schedule(
                step_start + 0.5 * step_size
            )
            if not isinstance(step_parameters, TwoModeParameters):
                raise TypeError(
                    "parameter_schedule must return TwoModeParameters"
                )
        if parameter_schedule is not None or step_size != time_step:
            step_dispersion = as_dispersion(
                step_parameters.beta
            ).values(modes)
            linear_half_step = linear_step_parameters(
                step_parameters,
                modes,
                0.5 * step_size,
                modal_dispersion=step_dispersion,
            )
        else:
            linear_half_step = constant_half_step
        return split_step(
            *state,
            step_size,
            linear_half_step,
            step_parameters.intermodal_ratio,
        )

    times, (comb_fields, auxiliary_fields) = integrate_snapshots(
        (comb, np.asarray(auxiliary)),
        final_time,
        time_step,
        snapshots,
        advance,
    )
    theta = 2.0 * np.pi * np.arange(spatial_points) / spatial_points
    return theta, times, comb_fields, auxiliary_fields


def pack_two_mode_state(comb, auxiliary):
    """Pack one complex field and one complex scalar into a real vector."""
    comb = np.asarray(comb, dtype=complex)
    auxiliary = complex(auxiliary)
    if comb.ndim != 1:
        raise ValueError("comb must be one-dimensional")
    return np.concatenate((
        comb.real,
        comb.imag,
        [auxiliary.real, auxiliary.imag],
    ))


def unpack_two_mode_state(vector):
    """Unpack the real representation returned by ``pack_two_mode_state``."""
    vector = np.asarray(vector, dtype=float)
    spatial_points = (vector.size - 2) // 2
    if vector.ndim != 1 or vector.size != 2 * spatial_points + 2:
        raise ValueError("two-mode state must have length 2*N+2")
    comb = (
        vector[:spatial_points]
        + 1j * vector[spatial_points : 2 * spatial_points]
    )
    auxiliary = complex(vector[-2], vector[-1])
    return comb, auxiliary


def two_mode_real_residual(vector, parameters):
    """Return the packed square residual used by steady solvers."""
    comb, auxiliary = unpack_two_mode_state(vector)
    return pack_two_mode_state(
        *two_mode_residual(comb, auxiliary, parameters)
    )


def pack_phase_conditioned_state(comb, auxiliary, shift_rate):
    """Pack a two-mode state with the scalar bordering translation."""
    return np.concatenate((
        pack_two_mode_state(comb, auxiliary),
        [float(shift_rate)],
    ))


def unpack_phase_conditioned_state(vector):
    """Unpack a two-mode state and its translation-border shift rate."""
    vector = np.asarray(vector, dtype=float)
    comb, auxiliary = unpack_two_mode_state(vector[:-1])
    return comb, auxiliary, float(vector[-1])


def phase_conditioned_real_residual(
    vector,
    parameters,
    reference_comb,
    phase_direction,
):
    """Return a square residual with the neutral comb translation removed."""
    comb, auxiliary, shift_rate = unpack_phase_conditioned_state(vector)
    comb_residual, auxiliary_residual = two_mode_residual(
        comb, auxiliary, parameters
    )
    comb_residual += shift_rate * spectral_derivative(comb)
    phase_condition = translation_phase_condition(
        comb, reference_comb, phase_direction
    )
    return np.concatenate((
        pack_two_mode_state(comb_residual, auxiliary_residual),
        [phase_condition],
    ))


def _newton_candidate(real_residual, initial_vector, tolerance, max_iterations):
    iteration_count = 0

    def count_iteration(_solution, _residual):
        nonlocal iteration_count
        iteration_count += 1

    try:
        candidate = newton_krylov(
            real_residual,
            initial_vector,
            f_tol=tolerance,
            maxiter=max_iterations,
            callback=count_iteration,
            line_search="armijo",
            verbose=False,
        )
    except NoConvergence as error:
        candidate = np.asarray(error.args[0], dtype=float)
    except ValueError as error:
        if "Jacobian inversion yielded zero vector" not in str(error):
            raise
        candidate = np.asarray(initial_vector, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    residual = np.asarray(real_residual(candidate), dtype=float)
    return candidate, float(np.max(np.abs(residual))), iteration_count


def solve_two_mode_steady_state(
    initial_comb,
    initial_auxiliary,
    parameters,
    *,
    tolerance=1.0e-9,
    max_iterations=100,
    initial_shift_rate=0.0,
):
    """Refine a two-mode equilibrium or relative equilibrium.

    A nonuniform comb is automatically bordered by a translation phase
    condition.  The returned shift rate is zero for a fixed equilibrium and
    is its normalized angular drift velocity when odd dispersion is present.
    """
    initial_comb = np.asarray(initial_comb, dtype=complex)
    if initial_comb.ndim != 1:
        raise ValueError("initial_comb must be a one-dimensional field")
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("tolerance and max_iterations must be positive")
    try:
        reference, phase_direction = translation_gauge(initial_comb)
    except ValueError:
        real_residual = lambda vector: two_mode_real_residual(
            vector, parameters
        )
        initial_vector = pack_two_mode_state(
            initial_comb, initial_auxiliary
        )
        candidate, maximum_residual, iterations = _newton_candidate(
            real_residual, initial_vector, tolerance, max_iterations
        )
        comb, auxiliary = unpack_two_mode_state(candidate)
        shift_rate = 0.0
    else:
        real_residual = lambda vector: phase_conditioned_real_residual(
            vector,
            parameters,
            reference,
            phase_direction,
        )
        initial_vector = pack_phase_conditioned_state(
            initial_comb, initial_auxiliary, initial_shift_rate
        )
        candidate, maximum_residual, iterations = _newton_candidate(
            real_residual, initial_vector, tolerance, max_iterations
        )
        comb, auxiliary, shift_rate = unpack_phase_conditioned_state(
            candidate
        )
    if maximum_residual > tolerance:
        raise RuntimeError(
            "two-mode steady solve did not converge after "
            f"{iterations} Newton--Krylov iterations; "
            f"maximum residual={maximum_residual:.3e}"
        )
    return comb, auxiliary, shift_rate, maximum_residual, iterations


def parameters_with_pump_detuning(parameters, alpha_comb):
    """Shift the pump frequency while preserving intrinsic mode spacing.

    Both normalized detunings change by the same amount because a pump-frequency
    shift is common to the comb and auxiliary resonances.
    """
    alpha_comb = float(alpha_comb)
    shift = alpha_comb - parameters.alpha_comb
    return replace(
        parameters,
        alpha_comb=alpha_comb,
        alpha_auxiliary=parameters.alpha_auxiliary + shift,
    )


def parameters_with_pump_scale(parameters, amplitude_scale):
    """Scale both bus drives by one real pump-amplitude factor.

    Optical input power scales as ``amplitude_scale**2``.  Applying the same
    factor to both drives preserves the single-input relation in Appendix A.
    """
    amplitude_scale = float(amplitude_scale)
    if not np.isfinite(amplitude_scale) or amplitude_scale < 0.0:
        raise ValueError("amplitude_scale must be finite and nonnegative")
    return replace(
        parameters,
        forcing_comb=amplitude_scale * parameters.forcing_comb,
        forcing_auxiliary=amplitude_scale * parameters.forcing_auxiliary,
    )
