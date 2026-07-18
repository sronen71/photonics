"""Steady-state refinement for the bidirectional PhCR LLE."""

import numpy as np
from scipy.linalg import circulant
from scipy.optimize import NoConvergence, newton_krylov, root

from bidirectional import bidirectional_residual, reflector_mask
from dispersion import as_dispersion, dispersion_is_even
from drift import (
    spectral_derivative,
    translation_gauge,
    translation_phase_condition,
)
from spectral import mode_numbers


def pack_fields(forward, backward):
    """Pack two complex fields into one real Newton vector."""
    return np.concatenate(
        (forward.real, forward.imag, backward.real, backward.imag)
    )


def unpack_fields(vector):
    """Unpack one real Newton vector into two complex fields."""
    spatial_points = vector.size // 4
    if vector.size != 4 * spatial_points:
        raise ValueError("bidirectional field vector length must be divisible by four")
    forward = (
        vector[:spatial_points]
        + 1j * vector[spatial_points : 2 * spatial_points]
    )
    backward = (
        vector[2 * spatial_points : 3 * spatial_points]
        + 1j * vector[3 * spatial_points :]
    )
    return forward, backward


def _newton_candidate(real_residual, initial_vector, tolerance, max_iterations):
    """Return the best Newton--Krylov candidate and iteration count."""
    iteration_count = 0

    def count_iteration(_solution, _residual):
        nonlocal iteration_count
        iteration_count += 1

    try:
        vector = newton_krylov(
            real_residual,
            initial_vector,
            f_tol=tolerance / np.sqrt(2.0),
            maxiter=max_iterations,
            callback=count_iteration,
        )
    except NoConvergence as error:
        vector = np.asarray(error.args[0])
    return np.asarray(vector), iteration_count


def _maximum_physical_residual(forward, backward, parameters):
    residuals = bidirectional_residual(forward, backward, parameters)
    maximum = max(float(np.max(np.abs(value))) for value in residuals)
    return residuals, maximum


def _solve_uniform_state(
    initial_forward,
    initial_backward,
    parameters,
    tolerance,
    max_iterations,
):
    """Refine a translation-invariant state, which has no shift null mode."""
    def real_residual(vector):
        forward, backward = unpack_fields(vector)
        return pack_fields(*bidirectional_residual(
            forward, backward, parameters
        ))

    vector, iterations = _newton_candidate(
        real_residual,
        pack_fields(initial_forward, initial_backward),
        tolerance,
        max_iterations,
    )
    forward, backward = unpack_fields(vector)
    _, maximum_residual = _maximum_physical_residual(
        forward, backward, parameters
    )
    return forward, backward, maximum_residual, iterations


def _pack_phase_conditioned(forward, backward, shift_rate):
    """Pack two fields and the scalar used to border the shift mode."""
    return np.concatenate((pack_fields(forward, backward), [shift_rate]))


def _unpack_phase_conditioned(vector):
    """Unpack two fields and the scalar used to border the shift mode."""
    forward, backward = unpack_fields(vector[:-1])
    return forward, backward, float(vector[-1])


def _place_real_linear_block(
    jacobian,
    row_real,
    row_imag,
    column_real,
    column_imag,
    direct,
    conjugate,
):
    """Place dR=A*dE+B*conj(dE) in a real-valued Jacobian."""
    direct_plus_conjugate = direct + conjugate
    direct_minus_conjugate = direct - conjugate
    jacobian[row_real, column_real] = direct_plus_conjugate.real
    jacobian[row_real, column_imag] = -direct_minus_conjugate.imag
    jacobian[row_imag, column_real] = direct_plus_conjugate.imag
    jacobian[row_imag, column_imag] = direct_minus_conjugate.real


def _solve_localized_state(
    initial_forward,
    initial_backward,
    parameters,
    tolerance,
    max_iterations,
    gauge,
):
    """Refine a localized v=0 state with its neutral translation removed."""
    reference, phase_direction = gauge
    spatial_points = initial_forward.size
    modes = mode_numbers(spatial_points)
    linear_matrix = circulant(np.fft.ifft(
        -(1.0 + 1j * parameters.alpha)
        + 1j * as_dispersion(parameters.beta).values(modes)
    ))
    derivative_matrix = circulant(np.fft.ifft(1j * modes))
    reflection_matrix = circulant(np.fft.ifft(
        reflector_mask(
            modes, parameters.reflector_half_width
        ).astype(float)
    ))
    mean_matrix = (
        np.ones((spatial_points, spatial_points), dtype=complex)
        / spatial_points
    )
    diagonal = np.diag_indices(spatial_points)

    def real_residual(vector):
        forward, backward, shift_rate = _unpack_phase_conditioned(vector)
        residuals = bidirectional_residual(forward, backward, parameters)
        derivatives = (
            spectral_derivative(forward),
            spectral_derivative(backward),
        )
        bordered_residuals = tuple(
            residual + shift_rate * derivative
            for residual, derivative in zip(residuals, derivatives)
        )
        phase_condition = translation_phase_condition(
            np.stack((forward, backward)), reference, phase_direction
        )
        return np.concatenate((
            pack_fields(*bordered_residuals),
            [phase_condition],
        ))

    def real_jacobian(vector):
        forward, backward, shift_rate = _unpack_phase_conditioned(vector)
        forward_power = float(np.mean(np.abs(forward) ** 2))
        backward_power = float(np.mean(np.abs(backward) ** 2))

        direct_forward_forward = (
            linear_matrix + shift_rate * derivative_matrix
        ).copy()
        direct_forward_forward[diagonal] += 2j * (
            np.abs(forward) ** 2 + backward_power
        )
        conjugate_forward_forward = np.diag(1j * forward**2)

        direct_backward_backward = (
            linear_matrix + shift_rate * derivative_matrix
        ).copy()
        direct_backward_backward[diagonal] += 2j * (
            np.abs(backward) ** 2 + forward_power
        )
        conjugate_backward_backward = np.diag(1j * backward**2)

        direct_forward_backward = (
            np.outer(
                2j * forward,
                np.conj(backward) / spatial_points,
            )
            - 0.5j * parameters.epsilon_phc * mean_matrix
        )
        conjugate_forward_backward = np.outer(
            2j * forward, backward / spatial_points
        )
        direct_backward_forward = (
            np.outer(
                2j * backward,
                np.conj(forward) / spatial_points,
            )
            - 0.5j * parameters.epsilon_phc * mean_matrix
            - parameters.bus_coupling
            * parameters.reflector_amplitude
            * reflection_matrix
        )
        conjugate_backward_forward = np.outer(
            2j * backward, forward / spatial_points
        )

        size = 4 * spatial_points + 1
        jacobian = np.empty((size, size), dtype=float)
        forward_real = slice(0, spatial_points)
        forward_imag = slice(spatial_points, 2 * spatial_points)
        backward_real = slice(2 * spatial_points, 3 * spatial_points)
        backward_imag = slice(3 * spatial_points, 4 * spatial_points)
        _place_real_linear_block(
            jacobian,
            forward_real,
            forward_imag,
            forward_real,
            forward_imag,
            direct_forward_forward,
            conjugate_forward_forward,
        )
        _place_real_linear_block(
            jacobian,
            forward_real,
            forward_imag,
            backward_real,
            backward_imag,
            direct_forward_backward,
            conjugate_forward_backward,
        )
        _place_real_linear_block(
            jacobian,
            backward_real,
            backward_imag,
            forward_real,
            forward_imag,
            direct_backward_forward,
            conjugate_backward_forward,
        )
        _place_real_linear_block(
            jacobian,
            backward_real,
            backward_imag,
            backward_real,
            backward_imag,
            direct_backward_backward,
            conjugate_backward_backward,
        )

        derivatives = spectral_derivative(np.stack((forward, backward)))
        jacobian[forward_real, -1] = derivatives[0].real
        jacobian[forward_imag, -1] = derivatives[0].imag
        jacobian[backward_real, -1] = derivatives[1].real
        jacobian[backward_imag, -1] = derivatives[1].imag
        jacobian[-1, forward_real] = (
            phase_direction[0].real / reference.size
        )
        jacobian[-1, forward_imag] = (
            phase_direction[0].imag / reference.size
        )
        jacobian[-1, backward_real] = (
            phase_direction[1].real / reference.size
        )
        jacobian[-1, backward_imag] = (
            phase_direction[1].imag / reference.size
        )
        jacobian[-1, -1] = 0.0
        return jacobian

    result = root(
        real_residual,
        _pack_phase_conditioned(initial_forward, initial_backward, 0.0),
        jac=real_jacobian,
        method="hybr",
        options={
            "xtol": max(1.0e-12, tolerance * 0.1),
            "maxfev": max_iterations,
        },
    )
    forward, backward, shift_rate = _unpack_phase_conditioned(result.x)
    residuals, maximum_residual = _maximum_physical_residual(
        forward, backward, parameters
    )
    derivatives = (
        spectral_derivative(forward),
        spectral_derivative(backward),
    )
    bordered_maximum = max(
        float(np.max(np.abs(residual + shift_rate * derivative)))
        for residual, derivative in zip(residuals, derivatives)
    )
    phase_error = abs(translation_phase_condition(
        np.stack((forward, backward)), reference, phase_direction
    ))
    if max(bordered_maximum, phase_error) > tolerance:
        maximum_residual = max(
            maximum_residual, bordered_maximum, phase_error
        )
    return forward, backward, maximum_residual, int(result.nfev)


def solve_bidirectional_steady_state(
    initial_forward,
    initial_backward,
    parameters,
    tolerance=1.0e-9,
    max_iterations=1000,
):
    """Refine an even-dispersion equilibrium with physical velocity v=0.

    A localized equilibrium has a neutral common-translation mode even though
    its physical velocity is zero.  The solve is therefore bordered by an
    auxiliary shift rate and a phase condition.  Convergence is still judged
    using the original fixed-frame residual, so a genuinely drifting state is
    not accepted as steady.  Uniform fields need no phase condition.
    """
    initial_forward = np.asarray(initial_forward, dtype=complex)
    initial_backward = np.asarray(initial_backward, dtype=complex)
    if (
        initial_forward.ndim != 1
        or initial_forward.shape != initial_backward.shape
    ):
        raise ValueError("initial bidirectional fields must be equal 1-D arrays")
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("steady tolerance and iteration limit must be positive")
    modes = mode_numbers(initial_forward.size)
    if not dispersion_is_even(parameters.beta, modes):
        raise ValueError(
            "the bidirectional steady solver currently requires even dispersion"
        )

    try:
        gauge = translation_gauge(
            np.stack((initial_forward, initial_backward))
        )
    except ValueError:
        result = _solve_uniform_state(
            initial_forward,
            initial_backward,
            parameters,
            tolerance,
            max_iterations,
        )
    else:
        result = _solve_localized_state(
            initial_forward,
            initial_backward,
            parameters,
            tolerance,
            max_iterations,
            gauge,
        )
    forward, backward, maximum_residual, iteration_count = result
    if maximum_residual > tolerance:
        raise RuntimeError(
            "bidirectional steady solve did not converge after "
            f"{iteration_count} nonlinear-solver steps; "
            f"maximum residual={maximum_residual:.3e}"
        )
    return forward, backward, maximum_residual, iteration_count
