"""Bidirectional photonic-crystal-resonator LLE physics and integration."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import expm

from dispersion import as_dispersion
from integration import integrate_snapshots
from spectral import mode_numbers


def loaded_linewidth_rad_s(
    center_frequency_hz, quality_factor_intrinsic, coupling_factor
):
    """Return kappa=(1+K)*omega_center/Qi in rad/s."""
    return (
        (1.0 + float(coupling_factor))
        * 2.0
        * np.pi
        * float(center_frequency_hz)
        / float(quality_factor_intrinsic)
    )


def normalized_beta_2(
    d2_rad_s,
    center_frequency_hz,
    quality_factor_intrinsic,
    coupling_factor,
):
    """Map physical D2 to this repository's beta2=-2*D2/kappa."""
    kappa = loaded_linewidth_rad_s(
        center_frequency_hz,
        quality_factor_intrinsic,
        coupling_factor,
    )
    return -2.0 * float(d2_rad_s) / kappa


@dataclass(frozen=True)
class BidirectionalParameters:
    """Normalized parameters for the coupled forward/backward LLE."""

    alpha: float
    forcing: complex
    beta: object
    epsilon_phc: float
    coupling_factor: float
    reflectivity: float
    reflector_phase: float
    reflector_half_width: int
    reflector_phase_reference: float = 0.0

    def __post_init__(self):
        finite_real = {
            "alpha": self.alpha,
            "epsilon_phc": self.epsilon_phc,
            "coupling_factor": self.coupling_factor,
            "reflectivity": self.reflectivity,
            "reflector_phase": self.reflector_phase,
            "reflector_phase_reference": self.reflector_phase_reference,
        }
        if not all(np.isfinite(float(value)) for value in finite_real.values()):
            raise ValueError("bidirectional parameters must be finite")
        if not np.isfinite(self.forcing.real) or not np.isfinite(
            self.forcing.imag
        ):
            raise ValueError("bidirectional forcing must be finite")
        if self.epsilon_phc < 0.0:
            raise ValueError("epsilon_phc must be nonnegative")
        if self.coupling_factor <= 0.0:
            raise ValueError("coupling factor must be positive")
        if not 0.0 <= self.reflectivity <= 1.0:
            raise ValueError("reflectivity must lie in [0, 1]")
        if (
            not isinstance(self.reflector_half_width, (int, np.integer))
            or self.reflector_half_width < 0
        ):
            raise ValueError("reflector_half_width must be a nonnegative integer")

    @property
    def effective_reflector_phase(self):
        """Return the reflector phase in this equation's field basis."""
        return self.reflector_phase + self.reflector_phase_reference

    @property
    def reflector_amplitude(self):
        """Return r=sqrt(R)*exp(i*phi), with R a power reflectivity."""
        return np.sqrt(self.reflectivity) * np.exp(
            1j * self.effective_reflector_phase
        )

    @property
    def bus_coupling(self):
        """Return gamma=2K/(K+1), the normalized bus-ring transfer."""
        return 2.0 * self.coupling_factor / (self.coupling_factor + 1.0)


@dataclass(frozen=True)
class LinearStep:
    """Exact affine flow of the bidirectional modal linear system."""

    diagonal_propagator: np.ndarray
    feedback_increment: np.ndarray
    pump_propagator: np.ndarray
    pump_drive_increment: np.ndarray


def reflector_mask(modes, half_width):
    """Return the finite reflection band I_Omega(mu)."""
    return np.abs(modes) <= int(half_width)


def linear_step_parameters(parameters, modes, duration, modal_dispersion=None):
    """Construct the exact driven-linear flow for one substep.

    Away from the pump, reflection is one-way forward-to-backward coupling
    with a nilpotent generator.  The driven pump mode is exponentiated as one
    augmented 3x3 affine system.
    """
    modes = np.asarray(modes)
    if modal_dispersion is None:
        modal_dispersion = as_dispersion(parameters.beta).values(modes)
    reflected = reflector_mask(modes, parameters.reflector_half_width)
    diagonal_generator = (
        -(1.0 + 1j * parameters.alpha) + 1j * modal_dispersion
    )
    diagonal_propagator = np.exp(duration * diagonal_generator)
    feedback_generator = np.where(
        reflected,
        -parameters.bus_coupling * parameters.reflector_amplitude,
        0.0j,
    )
    feedback_increment = duration * feedback_generator

    pump_index = int(np.flatnonzero(modes == 0.0)[0])
    pump_generator = np.array(
        [
            [
                diagonal_generator[pump_index],
                -0.5j * parameters.epsilon_phc,
            ],
            [
                -0.5j * parameters.epsilon_phc
                + feedback_generator[pump_index],
                diagonal_generator[pump_index],
            ],
        ],
        dtype=complex,
    )
    drive = np.array(
        [
            parameters.forcing,
            parameters.reflector_amplitude * parameters.forcing,
        ],
        dtype=complex,
    )
    augmented = np.zeros((3, 3), dtype=complex)
    augmented[:2, :2] = pump_generator
    augmented[:2, 2] = drive
    affine_flow = expm(duration * augmented)
    return LinearStep(
        diagonal_propagator=diagonal_propagator,
        feedback_increment=feedback_increment,
        pump_propagator=affine_flow[:2, :2],
        pump_drive_increment=affine_flow[:2, 2],
    )


def apply_linear_step(forward, backward, step):
    """Apply one precomputed exact affine modal step."""
    spatial_points = forward.size
    old_forward = np.fft.fft(forward)
    old_backward = np.fft.fft(backward)
    forward_modes = step.diagonal_propagator * old_forward
    backward_modes = step.diagonal_propagator * (
        old_backward + step.feedback_increment * old_forward
    )
    pump_state = step.pump_propagator @ np.array(
        [old_forward[0], old_backward[0]], dtype=complex
    )
    pump_state += spatial_points * step.pump_drive_increment
    forward_modes[0], backward_modes[0] = pump_state
    return np.fft.ifft(forward_modes), np.fft.ifft(backward_modes)


def nonlinear_step(forward, backward, duration):
    """Apply the exact self- and cross-phase Kerr flow."""
    forward_power = float(np.mean(np.abs(forward) ** 2))
    backward_power = float(np.mean(np.abs(backward) ** 2))
    return (
        forward
        * np.exp(
            1j
            * duration
            * (np.abs(forward) ** 2 + 2.0 * backward_power)
        ),
        backward
        * np.exp(
            1j
            * duration
            * (np.abs(backward) ** 2 + 2.0 * forward_power)
        ),
    )


def split_step(forward, backward, step_size, linear_half_step):
    """Advance the coupled fields by one Strang split step."""
    forward, backward = apply_linear_step(
        forward, backward, linear_half_step
    )
    forward, backward = nonlinear_step(forward, backward, step_size)
    return apply_linear_step(forward, backward, linear_half_step)


def _initial_field(value, spatial_points, name):
    array = np.asarray(value, dtype=complex)
    if array.ndim == 0:
        return np.full(spatial_points, array, dtype=complex)
    if array.shape != (spatial_points,):
        raise ValueError(f"{name} must be scalar or have spatial_points entries")
    return array.copy()


def solve_bidirectional_lle(
    parameters,
    spatial_points,
    final_time,
    time_step,
    initial_noise,
    snapshots,
    seed,
    initial_forward=0.0j,
    initial_backward=0.0j,
    alpha_schedule=None,
):
    """Integrate the bidirectional LLE with exact split subflows."""
    if spatial_points < 8 or final_time <= 0.0 or time_step <= 0.0:
        raise ValueError("spatial points and times must be positive")
    if initial_noise < 0.0:
        raise ValueError("initial noise must be nonnegative")

    theta = 2.0 * np.pi * np.arange(spatial_points) / spatial_points
    modes = mode_numbers(spatial_points)
    modal_dispersion = as_dispersion(parameters.beta).values(modes)
    forward = _initial_field(
        initial_forward, spatial_points, "initial_forward"
    )
    backward = _initial_field(
        initial_backward, spatial_points, "initial_backward"
    )
    rng = np.random.default_rng(seed)
    if initial_noise:
        forward += initial_noise * (
            rng.standard_normal(spatial_points)
            + 1j * rng.standard_normal(spatial_points)
        )
        backward += initial_noise * (
            rng.standard_normal(spatial_points)
            + 1j * rng.standard_normal(spatial_points)
        )

    constant_half_step = linear_step_parameters(
        parameters,
        modes,
        0.5 * time_step,
        modal_dispersion=modal_dispersion,
    )

    def advance(state, step_start, step_size):
        step_parameters = parameters
        if alpha_schedule is not None:
            step_parameters = replace(
                parameters,
                alpha=float(
                    alpha_schedule(step_start + 0.5 * step_size)
                ),
            )
        if alpha_schedule is not None or step_size != time_step:
            linear_half_step = linear_step_parameters(
                step_parameters,
                modes,
                0.5 * step_size,
                modal_dispersion=modal_dispersion,
            )
        else:
            linear_half_step = constant_half_step
        return split_step(*state, step_size, linear_half_step)

    times, (forward_fields, backward_fields) = integrate_snapshots(
        (forward, backward),
        final_time,
        time_step,
        snapshots,
        advance,
    )
    return theta, times, forward_fields, backward_fields


def bidirectional_residual(forward, backward, parameters):
    """Return the two continuous-equation residual fields."""
    forward = np.asarray(forward, dtype=complex)
    backward = np.asarray(backward, dtype=complex)
    if forward.shape != backward.shape or forward.ndim != 1:
        raise ValueError(
            "forward and backward fields must be equal one-dimensional arrays"
        )
    modes = mode_numbers(forward.size)
    dispersion = as_dispersion(parameters.beta).values(modes)
    dispersed_forward = np.fft.ifft(
        dispersion * np.fft.fft(forward)
    )
    dispersed_backward = np.fft.ifft(
        dispersion * np.fft.fft(backward)
    )
    reflected_forward = np.fft.ifft(
        reflector_mask(modes, parameters.reflector_half_width)
        * np.fft.fft(forward)
    )
    forward_power = np.mean(np.abs(forward) ** 2)
    backward_power = np.mean(np.abs(backward) ** 2)
    forward_residual = (
        -(1.0 + 1j * parameters.alpha) * forward
        + 1j * dispersed_forward
        + 1j * np.abs(forward) ** 2 * forward
        + 2j * backward_power * forward
        + parameters.forcing
        - 0.5j * parameters.epsilon_phc * np.mean(backward)
    )
    backward_residual = (
        -(1.0 + 1j * parameters.alpha) * backward
        + 1j * dispersed_backward
        + 1j * np.abs(backward) ** 2 * backward
        + 2j * forward_power * backward
        + parameters.reflector_amplitude * parameters.forcing
        - 0.5j * parameters.epsilon_phc * np.mean(forward)
        - parameters.bus_coupling
        * parameters.reflector_amplitude
        * reflected_forward
    )
    return forward_residual, backward_residual
