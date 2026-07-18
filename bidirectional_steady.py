"""Steady-state refinement for the bidirectional PhCR LLE."""

import numpy as np
from scipy.optimize import NoConvergence, newton_krylov

from bidirectional import bidirectional_residual
from dispersion import dispersion_is_even
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


def solve_bidirectional_steady_state(
    initial_forward,
    initial_backward,
    parameters,
    tolerance=1.0e-9,
    max_iterations=1000,
):
    """Refine an even-dispersion relative equilibrium with v=0.

    The paper model contains only D2, so its steady fields are fixed-frame
    equilibria.  An odd-dispersion extension would require the shared moving
    velocity and phase condition used by the scalar traveling-state solver.
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

    def real_residual(vector):
        forward, backward = unpack_fields(vector)
        residuals = bidirectional_residual(forward, backward, parameters)
        return pack_fields(*residuals)

    iteration_count = 0

    def count_iteration(_solution, _residual):
        nonlocal iteration_count
        iteration_count += 1

    try:
        vector = newton_krylov(
            real_residual,
            pack_fields(initial_forward, initial_backward),
            f_tol=tolerance / np.sqrt(2.0),
            maxiter=max_iterations,
            callback=count_iteration,
        )
    except NoConvergence as error:
        vector = np.asarray(error.args[0])
    forward, backward = unpack_fields(vector)
    residuals = bidirectional_residual(forward, backward, parameters)
    maximum_residual = max(
        float(np.max(np.abs(residual))) for residual in residuals
    )
    if maximum_residual > tolerance:
        raise RuntimeError(
            "bidirectional steady solve did not converge after "
            f"{iteration_count} Newton-Krylov iterations; "
            f"maximum residual={maximum_residual:.3e}"
        )
    return forward, backward, maximum_residual, iteration_count
