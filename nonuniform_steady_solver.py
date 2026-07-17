#!/usr/bin/env python3
"""Find a nonuniform stationary Lugiato--Lefever state on a ring.

The stationary equation is

    0 = -(1+i*alpha)A + i|A|^2 A
        - i*(beta/2)*d^2A/dtheta^2 + F,

with A(theta + 2*pi) = A(theta).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import NoConvergence, newton_krylov

from lle_solver import solve_lle


RESULTS_DIRECTORY = Path("results")
INITIAL_GUESS_TIME = 50.0


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
    points = vector.size // 2
    return vector[:points] + 1j * vector[points:]


def solve_stationary(initial_guess, alpha, forcing, beta, tolerance, iterations):
    """Refine an initial field with a matrix-free Newton--Krylov solver."""
    points = initial_guess.size
    angular_step = 2.0 * np.pi / points
    mode_number = 2.0 * np.pi * np.fft.fftfreq(points, d=angular_step)

    def real_residual(vector):
        amplitude = unpack(vector)
        return pack(
            stationary_residual(amplitude, alpha, forcing, beta, mode_number)
        )

    try:
        solution_vector = newton_krylov(
            real_residual,
            pack(initial_guess),
            f_tol=tolerance,
            maxiter=iterations,
            verbose=False,
        )
    except NoConvergence as error:
        # SciPy includes the full field in this exception.  Keep the best
        # iterate for diagnostics, but never print the exception or solution.
        solution_vector = error.args[0]
    solution = unpack(np.asarray(solution_vector))
    residual = stationary_residual(solution, alpha, forcing, beta, mode_number)
    return solution, float(np.max(np.abs(residual)))


def save_results(theta, amplitude, alpha, forcing, beta, residual):
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
        RESULTS_DIRECTORY / "nonuniform_steady_state.npz",
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
        RESULTS_DIRECTORY / "nonuniform_steady_state.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--f-real", type=float, default=1.8)
    parser.add_argument("--f-imag", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--noise", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--initial-guess",
        choices=("pattern", "soliton"),
        default="pattern",
        help="Upper-branch noise for a pattern, or one localized pulse",
    )
    arguments = parser.parse_args()

    forcing = complex(arguments.f_real, arguments.f_imag)
    print("Generating nonuniform initial guess...", flush=True)
    theta_grid = np.linspace(0.0, 2.0 * np.pi, arguments.points, endpoint=False)
    if arguments.initial_guess == "soliton":
        initial_background = single_soliton_seed(
            theta_grid, arguments.alpha, forcing, arguments.beta
        )
    else:
        initial_background = uniform_states(arguments.alpha, forcing)[-1]
    theta, _, fields = solve_lle(
        alpha=arguments.alpha,
        forcing=forcing,
        beta=arguments.beta,
        points=arguments.points,
        final_time=INITIAL_GUESS_TIME,
        time_step=arguments.dt,
        noise=arguments.noise,
        snapshots=2,
        seed=arguments.seed,
        initial_background=initial_background,
    )

    print("Solving stationary equation...", flush=True)
    amplitude, residual = solve_stationary(
        fields[-1],
        arguments.alpha,
        forcing,
        arguments.beta,
        arguments.tolerance,
        arguments.iterations,
    )

    print("Saving results...", flush=True)
    save_results(
        theta,
        amplitude,
        arguments.alpha,
        forcing,
        arguments.beta,
        residual,
    )
    print("Results saved.", flush=True)


if __name__ == "__main__":
    main()
