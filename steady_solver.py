#!/usr/bin/env python3
"""Find a stationary Lugiato--Lefever state on a ring.

The stationary equation is

    0 = -(1+i*alpha)A + i|A|^2 A
        + i*D(-i*d/dtheta)A + F,

with A(theta + 2*pi) = A(theta).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import NoConvergence, newton_krylov

from config_loader import (
    ConfigurationError,
    config_parser,
    load_section,
)
from dispersion import as_dispersion, soliton_seed_beta
from physics import load_solver_physics, normalized_summary
from spectrum import output_spectrum


RESULTS_DIRECTORY = Path("results")
CONFIGURATION_KEYS = {
    "spatial_points",
    "tolerance",
    "max_iterations",
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


def pack(amplitude):
    """Convert a complex field into a real vector for SciPy."""
    return np.concatenate((amplitude.real, amplitude.imag))


def unpack(vector):
    """Convert a real solver vector back into a complex field."""
    spatial_points = vector.size // 2
    return vector[:spatial_points] + 1j * vector[spatial_points:]


def solve_stationary(
    initial_guess, alpha, forcing, beta, tolerance, max_iterations
):
    """Refine an initial field with a matrix-free Newton--Krylov solver."""
    spatial_points = initial_guess.size
    angular_step = 2.0 * np.pi / spatial_points
    mode_number = 2.0 * np.pi * np.fft.fftfreq(
        spatial_points, d=angular_step
    )
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


def save_results(
    theta,
    amplitude,
    alpha,
    forcing,
    beta,
    residual,
    iterations,
    physics=None,
):
    """Save the stationary field, magnitude profile, and mode amplitudes."""
    dispersion_relation = as_dispersion(beta)
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    intensity = np.abs(amplitude) ** 2
    magnitude = np.abs(amplitude)
    mode_number = np.fft.fftshift(
        np.fft.fftfreq(theta.size, d=1.0 / theta.size)
    )
    mode_power = (
        np.abs(np.fft.fftshift(np.fft.fft(amplitude))) ** 2 / theta.size**2
    )
    unshifted_mode_number = np.fft.fftfreq(theta.size, d=1.0 / theta.size)
    dispersion_values = np.fft.fftshift(
        dispersion_relation.values(unshifted_mode_number)
    )
    plotted_output = output_spectrum(amplitude, physics)

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
    profile_axis.set_title("Nonuniform stationary complex field around the ring")
    profile_axis.grid(alpha=0.25)
    profile_axis.legend()

    intensity_axis.plot(theta, intensity, color="tab:purple", linewidth=2)
    intensity_axis.set_xlim(0.0, 2.0 * np.pi)
    intensity_axis.set_xlabel(r"Azimuthal angle $\theta$")
    intensity_axis.set_ylabel(r"$|A(\theta)|^2$")
    intensity_axis.set_title("Stationary intensity")
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
        f"{dispersion_relation.description}",
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
        for name in ("spatial_points", "max_iterations"):
            setattr(arguments, name, int(getattr(arguments, name)))
    except (ConfigurationError, TypeError, ValueError) as error:
        parser.error(str(error))
    if (
        arguments.spatial_points < 8
        or arguments.tolerance <= 0.0
        or arguments.max_iterations < 1
    ):
        parser.error(
            "steady.spatial_points must be at least 8; tolerance and "
            "max_iterations must be positive"
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

    print("Solving stationary equation...", flush=True)
    amplitude, residual, iterations = solve_stationary(
        initial_guess,
        arguments.alpha,
        forcing,
        dispersion,
        arguments.tolerance,
        arguments.max_iterations,
    )
    print(
        f"Converged in {iterations} Newton--Krylov iterations; "
        f"maximum residual={residual:.3e}",
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
        physics=physics,
    )
    print("Results saved.", flush=True)


if __name__ == "__main__":
    main()
