#!/usr/bin/env python3
"""Find a steady Lugiato--Lefever state on a ring.

For even modal dispersion the stationary equation is

    0 = -(1+i*alpha)A + i|A|^2 A
        + i*D(-i*d/dtheta)A + F,

with A(theta + 2*pi) = A(theta).

When dispersion has an odd part, the solver instead finds the relative
equilibrium A(theta, t)=U(theta-v*t) and its unknown velocity v.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import circulant
from scipy.optimize import NoConvergence, newton_krylov, root

from config_loader import (
    ConfigurationError,
    config_parser,
    load_section,
)
from dispersion import as_dispersion, dispersion_is_even, soliton_seed_beta
from drift import (
    estimate_drift,
    spectral_derivative,
    translation_gauge,
    translation_phase_condition,
)
from physics import load_solver_physics, normalized_summary
from spectral import mode_numbers
from spectrum import output_spectrum


RESULTS_DIRECTORY = Path("results")
CONFIGURATION_KEYS = {
    "spatial_points",
    "tolerance",
    "max_iterations",
    "relaxation_time",
    "relaxation_dt",
}


def uniform_states(alpha: float, forcing: complex):
    """Return uniform states ordered by increasing intensity."""
    roots = np.roots(
        [1.0, -2.0 * alpha, 1.0 + alpha**2, -abs(forcing) ** 2]
    )
    intensities = sorted(
        root.real
        for root in roots
        if abs(root.imag) < 1e-9 and root.real >= 0.0
    )
    return [
        forcing / (1.0 + 1j * (alpha - intensity))
        for intensity in intensities
    ]


def single_soliton_seed(theta, alpha, forcing, beta):
    """Construct a lower-background plus sech-pulse initial field."""
    beta = soliton_seed_beta(beta)
    if beta >= 0.0:
        raise ValueError("the bright-soliton seed requires beta < 0")
    background = uniform_states(alpha, forcing)[0]
    distance = (theta - np.pi + np.pi) % (2.0 * np.pi) - np.pi
    inverse_width = np.sqrt(2.0 * alpha / abs(beta))
    phase_cosine = np.sqrt(8.0 * alpha) / (np.pi * abs(forcing))
    if phase_cosine > 1.0:
        raise ValueError("pump is too weak for the analytic bright-soliton seed")
    pulse_phase = np.angle(forcing) + np.arccos(phase_cosine)
    pulse = (
        np.sqrt(2.0 * alpha)
        * np.exp(1j * pulse_phase)
        / np.cosh(inverse_width * distance)
    )
    return background + pulse


def stationary_residual(
    amplitude,
    alpha,
    forcing,
    beta,
    mode_number,
    modal_dispersion=None,
):
    """Return the complex residual of the stationary ring equation."""
    if modal_dispersion is None:
        modal_dispersion = as_dispersion(beta).values(mode_number)
    dispersed_field = np.fft.ifft(modal_dispersion * np.fft.fft(amplitude))
    return (
        -(1.0 + 1j * alpha) * amplitude
        + 1j * np.abs(amplitude) ** 2 * amplitude
        + 1j * dispersed_field
        + forcing
    )


def moving_frame_residual(
    amplitude,
    velocity,
    alpha,
    forcing,
    beta,
    mode_number,
    modal_dispersion=None,
):
    """Return the residual for A(theta,t)=U(theta-velocity*t)."""
    return stationary_residual(
        amplitude,
        alpha,
        forcing,
        beta,
        mode_number,
        modal_dispersion=modal_dispersion,
    ) + velocity * spectral_derivative(amplitude)


def pack(amplitude):
    """Convert a complex field into a real vector for SciPy."""
    return np.concatenate((amplitude.real, amplitude.imag))


def unpack(vector):
    """Convert a real solver vector back into a complex field."""
    spatial_points = vector.size // 2
    return vector[:spatial_points] + 1j * vector[spatial_points:]


def _solve_fixed_frame(
    initial_guess, alpha, forcing, beta, tolerance, max_iterations
):
    """Solve the v=0 equation after reflection symmetry has been verified."""
    spatial_points = initial_guess.size
    mode_number = mode_numbers(spatial_points)
    modal_dispersion = as_dispersion(beta).values(mode_number)

    def real_residual(vector):
        amplitude = unpack(vector)
        return pack(
            stationary_residual(
                amplitude,
                alpha,
                forcing,
                beta,
                mode_number,
                modal_dispersion=modal_dispersion,
            )
        )

    iteration_count = 0

    def count_iteration(_vector, _residual):
        nonlocal iteration_count
        iteration_count += 1

    try:
        solution_vector = newton_krylov(
            real_residual,
            pack(initial_guess),
            f_tol=tolerance / np.sqrt(2.0),
            maxiter=max_iterations,
            callback=count_iteration,
            verbose=False,
        )
    except NoConvergence as error:
        # SciPy includes the full field in this exception. Report only its
        # residual so a failed solve does not print the complete solution.
        best_solution = unpack(np.asarray(error.args[0]))
        best_residual = stationary_residual(
            best_solution,
            alpha,
            forcing,
            beta,
            mode_number,
            modal_dispersion=modal_dispersion,
        )
        maximum_residual = float(np.max(np.abs(best_residual)))
        raise RuntimeError(
            "stationary Newton--Krylov solve did not converge after "
            f"{iteration_count} iterations; maximum residual={maximum_residual:.3e}"
        ) from None
    solution = unpack(np.asarray(solution_vector))
    residual = stationary_residual(
        solution,
        alpha,
        forcing,
        beta,
        mode_number,
        modal_dispersion=modal_dispersion,
    )
    maximum_residual = float(np.max(np.abs(residual)))
    if maximum_residual > tolerance:
        raise RuntimeError(
            "stationary Newton--Krylov solve returned without satisfying "
            f"the residual tolerance; maximum residual={maximum_residual:.3e}"
        )
    return solution, maximum_residual, iteration_count


def _pack_moving(amplitude, velocity):
    """Pack a complex field and one real velocity into a real vector."""
    return np.concatenate((amplitude.real, amplitude.imag, [velocity]))


def _unpack_moving(vector):
    """Unpack a complex field and real velocity from a real vector."""
    spatial_points = (vector.size - 1) // 2
    amplitude = (
        vector[:spatial_points]
        + 1j * vector[spatial_points:2 * spatial_points]
    )
    return amplitude, float(vector[-1])


def pack_phase_conditioned_state(amplitude, velocity=0.0):
    """Public representation for continuation of localized scalar states."""
    return _pack_moving(np.asarray(amplitude, dtype=complex), velocity)


def unpack_phase_conditioned_state(vector):
    """Unpack a localized scalar state and its translation-border velocity."""
    return _unpack_moving(np.asarray(vector, dtype=float))


def scalar_phase_conditioned_residual(
    vector,
    alpha,
    forcing,
    beta,
    reference,
    phase_direction,
):
    """Return a square scalar-LLE residual with translation removed.

    This is the continuation adapter for localized scalar states.  The final
    unknown is the physical moving-frame velocity for odd dispersion and an
    auxiliary zero-valued shift rate for an even-dispersion equilibrium.
    """
    amplitude, velocity = unpack_phase_conditioned_state(vector)
    modes = mode_numbers(amplitude.size)
    residual = moving_frame_residual(
        amplitude,
        velocity,
        alpha,
        forcing,
        beta,
        modes,
    )
    phase_condition = translation_phase_condition(
        amplitude, reference, phase_direction
    )
    return np.concatenate((pack(residual), [phase_condition]))


def _solve_moving_frame(
    initial_guess,
    initial_velocity,
    alpha,
    forcing,
    beta,
    tolerance,
    max_iterations,
):
    """Solve for a relative equilibrium and its translation velocity."""
    spatial_points = initial_guess.size
    mode_number = mode_numbers(spatial_points)
    modal_dispersion = as_dispersion(beta).values(mode_number)
    try:
        reference, phase_direction = translation_gauge(initial_guess)
    except ValueError:
        raise ValueError(
            "a moving-frame solve requires a nonuniform initial field"
        ) from None

    spectral_linear_operator = (
        -(1.0 + 1j * alpha) + 1j * modal_dispersion
    )
    linear_matrix = circulant(np.fft.ifft(spectral_linear_operator))
    derivative_matrix = circulant(np.fft.ifft(1j * mode_number))
    diagonal = np.diag_indices(spatial_points)

    def real_residual(vector):
        amplitude, velocity = _unpack_moving(vector)
        residual = moving_frame_residual(
            amplitude,
            velocity,
            alpha,
            forcing,
            beta,
            mode_number,
            modal_dispersion=modal_dispersion,
        )
        phase_condition = translation_phase_condition(
            amplitude, reference, phase_direction
        )
        return np.concatenate((
            residual.real,
            residual.imag,
            [phase_condition],
        ))

    def real_jacobian(vector):
        amplitude, velocity = _unpack_moving(vector)
        direct = linear_matrix + velocity * derivative_matrix
        direct = direct.copy()
        direct[diagonal] += 2j * np.abs(amplitude) ** 2
        conjugate = np.diag(1j * amplitude**2)
        direct_plus = direct + conjugate
        direct_minus = direct - conjugate

        size = 2 * spatial_points + 1
        jacobian = np.empty((size, size), dtype=float)
        jacobian[:spatial_points, :spatial_points] = direct_plus.real
        jacobian[:spatial_points, spatial_points:-1] = -direct_minus.imag
        jacobian[spatial_points:-1, :spatial_points] = direct_plus.imag
        jacobian[spatial_points:-1, spatial_points:-1] = direct_minus.real

        amplitude_derivative = spectral_derivative(amplitude)
        jacobian[:spatial_points, -1] = amplitude_derivative.real
        jacobian[spatial_points:-1, -1] = amplitude_derivative.imag
        jacobian[-1, :spatial_points] = (
            phase_direction.real / spatial_points
        )
        jacobian[-1, spatial_points:-1] = (
            phase_direction.imag / spatial_points
        )
        jacobian[-1, -1] = 0.0
        return jacobian

    result = root(
        real_residual,
        _pack_moving(initial_guess, initial_velocity),
        jac=real_jacobian,
        method="hybr",
        options={
            "xtol": max(1.0e-12, tolerance * 0.1),
            "maxfev": max_iterations,
        },
    )
    solution, velocity = _unpack_moving(result.x)
    residual = moving_frame_residual(
        solution,
        velocity,
        alpha,
        forcing,
        beta,
        mode_number,
        modal_dispersion=modal_dispersion,
    )
    maximum_residual = float(np.max(np.abs(residual)))
    if maximum_residual > tolerance:
        raise RuntimeError(
            "moving-frame Newton solve did not converge after "
            f"{result.nfev} residual evaluations; "
            f"maximum residual={maximum_residual:.3e}; {result.message}"
        )
    return solution, velocity, maximum_residual, int(result.nfev)


def solve_steady_state(
    initial_guess,
    alpha,
    forcing,
    beta,
    tolerance,
    max_iterations,
    initial_velocity=0.0,
):
    """Solve in the fixed frame for even D(k), otherwise solve for v."""
    spatial_points = initial_guess.size
    mode_number = mode_numbers(spatial_points)
    if dispersion_is_even(beta, mode_number):
        solution, residual, iterations = _solve_fixed_frame(
            initial_guess,
            alpha,
            forcing,
            beta,
            tolerance,
            max_iterations,
        )
        return solution, 0.0, residual, iterations
    return _solve_moving_frame(
        initial_guess,
        initial_velocity,
        alpha,
        forcing,
        beta,
        tolerance,
        max_iterations,
    )


def save_results(
    theta,
    amplitude,
    alpha,
    forcing,
    beta,
    residual,
    iterations,
    velocity=0.0,
    physics=None,
):
    """Save the co-moving steady field, drift, and mode amplitudes."""
    dispersion_relation = as_dispersion(beta)
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    intensity = np.abs(amplitude) ** 2
    magnitude = np.abs(amplitude)
    mode_number = np.fft.fftshift(mode_numbers(theta.size))
    mode_power = (
        np.abs(np.fft.fftshift(np.fft.fft(amplitude))) ** 2 / theta.size**2
    )
    unshifted_mode_number = mode_numbers(theta.size)
    dispersion_values = np.fft.fftshift(
        dispersion_relation.values(unshifted_mode_number)
    )
    plotted_output = output_spectrum(
        amplitude, physics, drift_velocity=velocity
    )

    saved_results = dict(
        theta=theta,
        amplitude=amplitude,
        magnitude=magnitude,
        intensity=intensity,
        mode_number=mode_number,
        mode_power=mode_power,
        alpha=alpha,
        forcing=forcing,
        beta=(
            np.nan
            if dispersion_relation.seed_beta is None
            else dispersion_relation.seed_beta
        ),
        beta2=(
            np.nan
            if dispersion_relation.seed_beta is None
            else dispersion_relation.seed_beta
        ),
        dispersion_kind=dispersion_relation.kind,
        dispersion=dispersion_values,
        maximum_residual=residual,
        newton_iterations=iterations,
        drift_velocity_normalized=velocity,
    )
    saved_results.update(plotted_output["saved"])
    np.savez(RESULTS_DIRECTORY / "steady_solution.npz", **saved_results)

    figure = plt.figure(figsize=(11, 9))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.15, 1.0))
    profile_axis = figure.add_subplot(grid[0, :])
    intensity_axis = figure.add_subplot(grid[1, 0])
    output_axis = figure.add_subplot(grid[1, 1])
    profile_axis.plot(
        theta, amplitude.real, color="tab:blue", linewidth=2,
        label=r"$\mathrm{Re}(A)$",
    )
    profile_axis.plot(
        theta, amplitude.imag, color="tab:orange", linewidth=2,
        label=r"$\mathrm{Im}(A)$",
    )
    profile_axis.set_xlim(0.0, 2.0 * np.pi)
    profile_axis.set_xlabel(r"Azimuthal angle $\theta$")
    profile_axis.set_ylabel(r"$A(\theta)$")
    profile_axis.set_title("Steady complex field in its co-moving frame")
    profile_axis.grid(alpha=0.25)
    profile_axis.legend()

    intensity_axis.plot(theta, intensity, color="tab:purple", linewidth=2)
    intensity_axis.set_xlim(0.0, 2.0 * np.pi)
    intensity_axis.set_xlabel(r"Azimuthal angle $\theta$")
    intensity_axis.set_ylabel(r"$|A(\theta)|^2$")
    intensity_axis.set_title("Steady co-moving intensity")
    intensity_axis.grid(alpha=0.25)

    output_axis.scatter(
        plotted_output["axis"],
        plotted_output["power_db"],
        color="tab:green",
        s=9,
    )
    output_axis.set_xlim(
        plotted_output["axis"].min(), plotted_output["axis"].max()
    )
    output_axis.set_xlabel(plotted_output["axis_label"])
    output_axis.set_ylabel(plotted_output["power_label"])
    output_axis.set_title(plotted_output["title"])
    output_axis.grid(alpha=0.25)

    figure.suptitle(
        rf"$\alpha={alpha:g}$, $F={forcing.real:g}$, "
        + rf"$v={velocity:.6g}$, "
        + f"{dispersion_relation.description}",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(
        RESULTS_DIRECTORY / "steady_solution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    parser = config_parser(__doc__)
    command_line = parser.parse_args()
    try:
        physics = load_solver_physics(command_line.config)
        dispersion = physics.dispersion
        arguments = load_section(
            command_line.config, "steady", CONFIGURATION_KEYS
        )
        arguments.alpha = physics.alpha
        arguments.tolerance = float(arguments.tolerance)
        arguments.relaxation_time = float(arguments.relaxation_time)
        arguments.relaxation_dt = float(arguments.relaxation_dt)
        for name in ("spatial_points", "max_iterations"):
            setattr(arguments, name, int(getattr(arguments, name)))
    except (ConfigurationError, TypeError, ValueError) as error:
        parser.error(str(error))
    if (
        arguments.spatial_points < 8
        or arguments.tolerance <= 0.0
        or arguments.max_iterations < 1
        or arguments.relaxation_time <= 0.0
        or arguments.relaxation_dt <= 0.0
    ):
        parser.error(
            "steady.spatial_points must be at least 8; tolerance and "
            "all iteration and relaxation settings must be positive"
        )

    forcing = physics.forcing
    if physics.units == "SI":
        print(
            "Converted physical input to normalized LLE: "
            + normalized_summary(physics),
            flush=True,
        )
        print(
            "One normalized time unit is "
            f"{physics.physical_time_per_normalized_unit_s:.6g} s; "
            f"FSR={physics.fsr_hz:g} Hz",
            flush=True,
        )
    print("Generating stationary initial guess...", flush=True)
    theta_grid = np.linspace(
        0.0, 2.0 * np.pi, arguments.spatial_points, endpoint=False
    )
    initial_guess = single_soliton_seed(
        theta_grid, arguments.alpha, forcing, dispersion
    )

    mode_number = mode_numbers(arguments.spatial_points)
    modal_dispersion = dispersion.values(mode_number)
    initial_velocity = 0.0
    if not dispersion_is_even(dispersion, mode_number):
        # A stable translating attractor supplies a close initial profile for
        # the augmented Newton solve.  The Newton solve then removes the
        # finite-time and split-step error to the requested tolerance.
        from lle_solver import solve_lle

        print(
            "Odd dispersion detected; generating a moving-state initial guess...",
            flush=True,
        )
        snapshots = max(51, min(
            501,
            int(np.ceil(arguments.relaxation_time / arguments.relaxation_dt))
            + 1,
        ))
        _, relaxation_times, relaxation_fields = solve_lle(
            alpha=arguments.alpha,
            forcing=forcing,
            beta=dispersion,
            spatial_points=arguments.spatial_points,
            final_time=arguments.relaxation_time,
            time_step=arguments.relaxation_dt,
            initial_noise=0.0,
            snapshots=snapshots,
            seed=0,
            initial_background=initial_guess,
        )
        diagnostic_start = relaxation_times[-1] - min(
            10.0, 0.5 * arguments.relaxation_time
        )
        diagnostic_indices = np.flatnonzero(
            relaxation_times >= diagnostic_start
        )
        drift = estimate_drift(
            relaxation_times[diagnostic_indices],
            relaxation_fields[diagnostic_indices],
        )
        if not drift.is_rigid_translation:
            raise RuntimeError(
                "odd-dispersion relaxation did not approach a rigidly "
                "translating state; try a different seed or parameters"
            )
        initial_guess = relaxation_fields[-1]
        initial_velocity = drift.velocity
        print(
            f"Initial drift estimate v={initial_velocity:.6g}; "
            f"aligned shape variation={drift.shape_variation:.3e}",
            flush=True,
        )

    print("Solving steady equation in the symmetry-appropriate frame...", flush=True)
    amplitude, velocity, residual, iterations = solve_steady_state(
        initial_guess,
        arguments.alpha,
        forcing,
        dispersion,
        arguments.tolerance,
        arguments.max_iterations,
        initial_velocity=initial_velocity,
    )
    print(
        f"Converged after {iterations} residual evaluations; "
        f"v={velocity:.6g}; maximum moving-frame residual={residual:.3e}",
        flush=True,
    )

    print("Saving results...", flush=True)
    save_results(
        theta_grid,
        amplitude,
        arguments.alpha,
        forcing,
        dispersion,
        residual,
        iterations,
        velocity=velocity,
        physics=physics,
    )
    print("Results saved.", flush=True)


if __name__ == "__main__":
    main()
