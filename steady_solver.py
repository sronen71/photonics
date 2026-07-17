#!/usr/bin/env python3
"""Find a stationary Lugiato--Lefever state on a ring.

The stationary equation is

    0 = -(1+i*alpha)A + i|A|^2 A
        - i*(beta/2)*d^2A/dtheta^2 + F,

with A(theta + 2*pi) = A(theta).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import NoConvergence, newton_krylov

from config_loader import (
    PHYSICS_KEYS,
    ConfigurationError,
    config_parser,
    load_section,
)
from lle_solver import solve_lle


RESULTS_DIRECTORY = Path("results")
CONFIGURATION_KEYS = {
    "spatial_points",
    "initial_guess_time",
    "dt",
    "initial_noise",
    "seed",
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


def stationary_residual(amplitude, alpha, forcing, beta, mode_number):
    """Return the complex residual of the stationary ring equation."""
    second_derivative = np.fft.ifft(
        -(mode_number**2) * np.fft.fft(amplitude)
    )
    return (
        -(1.0 + 1j * alpha) * amplitude
        + 1j * np.abs(amplitude) ** 2 * amplitude
        - 0.5j * beta * second_derivative
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

    def real_residual(vector):
        amplitude = unpack(vector)
        return pack(
            stationary_residual(amplitude, alpha, forcing, beta, mode_number)
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
            best_solution, alpha, forcing, beta, mode_number
        )
        maximum_residual = float(np.max(np.abs(best_residual)))
        raise RuntimeError(
            "stationary Newton--Krylov solve did not converge after "
            f"{iteration_count} iterations; maximum residual={maximum_residual:.3e}"
        ) from None
    solution = unpack(np.asarray(solution_vector))
    residual = stationary_residual(solution, alpha, forcing, beta, mode_number)
    maximum_residual = float(np.max(np.abs(residual)))
    if maximum_residual > tolerance:
        raise RuntimeError(
            "stationary Newton--Krylov solve returned without satisfying "
            f"the residual tolerance; maximum residual={maximum_residual:.3e}"
        )
    return solution, maximum_residual, iteration_count


def save_results(theta, amplitude, alpha, forcing, beta, residual, iterations):
    """Save the stationary field, magnitude profile, and mode amplitudes."""
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    intensity = np.abs(amplitude) ** 2
    magnitude = np.abs(amplitude)
    mode_number = np.fft.fftshift(
        np.fft.fftfreq(theta.size, d=1.0 / theta.size)
    )
    mode_power = (
        np.abs(np.fft.fftshift(np.fft.fft(amplitude))) ** 2 / theta.size**2
    )

    np.savez(
        RESULTS_DIRECTORY / "steady_solution.npz",
        theta=theta,
        amplitude=amplitude,
        magnitude=magnitude,
        intensity=intensity,
        mode_number=mode_number,
        mode_power=mode_power,
        alpha=alpha,
        forcing=forcing,
        beta=beta,
        maximum_residual=residual,
        newton_iterations=iterations,
    )

    figure = plt.figure(figsize=(11, 9))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.15, 1.0))
    profile_axis = figure.add_subplot(grid[0, :])
    intensity_axis = figure.add_subplot(grid[1, 0])
    spectrum_axis = figure.add_subplot(grid[1, 1])
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

    positive = mode_power[mode_power > 0.0]
    floor = positive.max() * 1e-12 if positive.size else 1e-12
    spectrum_axis.scatter(
        mode_number, np.maximum(mode_power, floor), color="tab:red", s=9
    )
    spectrum_axis.set_yscale("log")
    spectrum_axis.set_xlim(mode_number.min(), mode_number.max())
    spectrum_axis.set_xlabel("Azimuthal mode number")
    spectrum_axis.set_ylabel("Mode power")
    spectrum_axis.set_title("Stationary mode power")
    spectrum_axis.grid(alpha=0.25)

    figure.suptitle(
        rf"$\alpha={alpha:g}$, $F={forcing.real:g}{forcing.imag:+g}i$, "
        rf"$\beta={beta:g}$",
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
        physics = load_section(command_line.config, "physics", PHYSICS_KEYS)
        arguments = load_section(
            command_line.config, "steady", CONFIGURATION_KEYS
        )
        for name in PHYSICS_KEYS:
            setattr(arguments, name, float(getattr(physics, name)))
        for name in (
            "dt",
            "initial_noise",
            "tolerance",
            "initial_guess_time",
        ):
            setattr(arguments, name, float(getattr(arguments, name)))
        for name in ("spatial_points", "seed", "max_iterations"):
            setattr(arguments, name, int(getattr(arguments, name)))
    except (ConfigurationError, TypeError, ValueError) as error:
        parser.error(str(error))
    if (
        arguments.initial_noise < 0.0
        or arguments.initial_guess_time <= 0.0
        or arguments.tolerance <= 0.0
        or arguments.max_iterations < 1
    ):
        parser.error(
            "steady initial_noise must be nonnegative; initial_guess_time, "
            "tolerance, and max_iterations must be positive"
        )

    forcing = complex(arguments.f_real, arguments.f_imag)
    print("Generating stationary initial guess...", flush=True)
    theta_grid = np.linspace(
        0.0, 2.0 * np.pi, arguments.spatial_points, endpoint=False
    )
    initial_background = single_soliton_seed(
        theta_grid, arguments.alpha, forcing, arguments.beta
    )
    theta, _, fields = solve_lle(
        alpha=arguments.alpha,
        forcing=forcing,
        beta=arguments.beta,
        spatial_points=arguments.spatial_points,
        final_time=arguments.initial_guess_time,
        time_step=arguments.dt,
        initial_noise=arguments.initial_noise,
        snapshots=2,
        seed=arguments.seed,
        initial_background=initial_background,
    )

    print("Solving stationary equation...", flush=True)
    amplitude, residual, iterations = solve_stationary(
        fields[-1],
        arguments.alpha,
        forcing,
        arguments.beta,
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
        theta,
        amplitude,
        arguments.alpha,
        forcing,
        arguments.beta,
        residual,
        iterations,
    )
    print("Results saved.", flush=True)


if __name__ == "__main__":
    main()
