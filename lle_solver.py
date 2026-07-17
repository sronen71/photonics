#!/usr/bin/env python3
"""Solve the Lugiato--Lefever equation on a ring.

    dA/dt = -(1 + i*alpha)A + i|A|^2 A
            - i*(beta/2)*d^2A/dtheta^2 + F

The azimuthal coordinate theta is periodic on [0, 2*pi).  The pump F,
detuning alpha, and dispersion beta are constant.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIRECTORY = Path("results")


def uniform_states(alpha: float, forcing: complex):
    """Return uniform steady states ordered by increasing intensity."""
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
    """Return a lower-background plus phase-matched sech pulse."""
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


def nonlinear_step(amplitude: np.ndarray, step: float, forcing: complex) -> np.ndarray:
    """RK4 step for the local Kerr nonlinearity and pump."""
    def derivative(field):
        return 1j * np.abs(field) ** 2 * field + forcing

    k1 = derivative(amplitude)
    k2 = derivative(amplitude + 0.5 * step * k1)
    k3 = derivative(amplitude + 0.5 * step * k2)
    k4 = derivative(amplitude + step * k3)
    return amplitude + step * (k1 + 2*k2 + 2*k3 + k4) / 6.0


def solve_lle(
    alpha: float,
    forcing: complex,
    beta: float,
    points: int,
    final_time: float,
    time_step: float,
    noise: float,
    snapshots: int,
    seed: int,
    initial_background: complex = 0.0j,
    alpha_schedule=None,
    forcing_schedule=None,
):
    """Integrate the spatial equation using Strang split stepping."""
    if points < 8 or final_time <= 0.0 or time_step <= 0.0:
        raise ValueError("points, final time, and time step must be positive")

    theta = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    angular_step = 2.0 * np.pi / points
    # Integer azimuthal mode numbers for a 2*pi-periodic ring.
    mode_number = 2.0 * np.pi * np.fft.fftfreq(points, d=angular_step)

    # Fourier representation of -(1+i*alpha)A - i*(beta/2)A_theta,theta.
    linear_operator = -(1.0 + 1j * alpha) + 0.5j * beta * mode_number**2
    half_linear_step = np.exp(0.5 * time_step * linear_operator)

    rng = np.random.default_rng(seed)
    # Seed all spatial modes with small complex noise.  The time-dependent
    # executable uses the default empty-cavity background; stationary solvers
    # may supply a different branch as their initial guess.
    amplitude = initial_background + noise * (
        rng.standard_normal(points) + 1j * rng.standard_normal(points)
    )

    number_of_steps = int(np.ceil(final_time / time_step))
    save_every = max(1, number_of_steps // max(1, snapshots - 1))
    saved_times = [0.0]
    saved_fields = [amplitude.copy()]

    for index in range(number_of_steps):
        step = min(time_step, final_time - index * time_step)
        if alpha_schedule is not None:
            step_alpha = alpha_schedule(index * time_step + 0.5 * step)
            step_operator = (
                -(1.0 + 1j * step_alpha) + 0.5j * beta * mode_number**2
            )
            propagator = np.exp(0.5 * step * step_operator)
        elif step != time_step:
            propagator = np.exp(0.5 * step * linear_operator)
        else:
            propagator = half_linear_step

        step_forcing = (
            forcing_schedule(index * time_step + 0.5 * step)
            if forcing_schedule is not None
            else forcing
        )
        amplitude = np.fft.ifft(propagator * np.fft.fft(amplitude))
        amplitude = nonlinear_step(amplitude, step, step_forcing)
        amplitude = np.fft.ifft(propagator * np.fft.fft(amplitude))

        if not np.all(np.isfinite(amplitude)):
            raise RuntimeError("solution diverged; reduce --dt")

        if (index + 1) % save_every == 0 or index == number_of_steps - 1:
            saved_times.append(min((index + 1) * time_step, final_time))
            saved_fields.append(amplitude.copy())

    return theta, np.asarray(saved_times), np.asarray(saved_fields)


def save_figure(
    theta, times, fields, alpha: float, forcing: complex, beta: float,
    alpha_start=None,
) -> None:
    """Save space-time field magnitude, final profile, and mode amplitudes."""
    magnitude = np.abs(fields)
    final_field = fields[-1]
    mode_numbers = np.fft.fftshift(np.fft.fftfreq(theta.size, d=1.0 / theta.size))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(final_field))) / theta.size

    figure = plt.figure(figsize=(10, 9))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.45, 1.0))
    evolution_axis = figure.add_subplot(grid[0, :])
    profile_axis = figure.add_subplot(grid[1, 0])
    spectrum_axis = figure.add_subplot(grid[1, 1])

    image = evolution_axis.pcolormesh(
        theta, times, magnitude, shading="auto", cmap="magma", vmin=0.0
    )
    evolution_axis.set_xlabel(r"Azimuthal angle $\theta$")
    evolution_axis.set_ylabel(r"Time $t$")
    evolution_axis.set_title(r"Spatiotemporal field magnitude $|A(\theta,t)|$")
    figure.colorbar(image, ax=evolution_axis, label=r"$|A|$")

    profile_axis.plot(
        theta, final_field.real, color="tab:blue", linewidth=1.8,
        label=r"$\mathrm{Re}(A)$",
    )
    profile_axis.plot(
        theta, final_field.imag, color="tab:orange", linewidth=1.8,
        label=r"$\mathrm{Im}(A)$",
    )
    profile_axis.set_xlabel(r"Azimuthal angle $\theta$")
    profile_axis.set_ylabel(r"$A(\theta,t_f)$")
    profile_axis.set_title("Final complex field around the ring")
    profile_axis.grid(alpha=0.25)
    profile_axis.legend()

    positive = spectrum[spectrum > 0.0]
    floor = positive.max() * 1e-12 if positive.size else 1e-12
    spectrum_axis.scatter(
        mode_numbers, np.maximum(spectrum, floor), color="tab:red", s=9
    )
    spectrum_axis.set_yscale("log")
    spectrum_axis.set_xlim(mode_numbers.min(), mode_numbers.max())
    spectrum_axis.set_xlabel("Spatial mode number")
    spectrum_axis.set_ylabel("Mode amplitude")
    spectrum_axis.set_title("Final mode amplitudes")
    spectrum_axis.grid(alpha=0.25)

    alpha_text = (
        rf"$\alpha:{alpha_start:g}\to{alpha:g}$"
        if alpha_start is not None
        else rf"$\alpha={alpha:g}$"
    )
    figure.suptitle(
        rf"Ring LLE: {alpha_text}, $F={forcing.real:g}{forcing.imag:+g}i$, "
        rf"$\beta={beta:g}$",
        fontsize=14,
    )
    figure.tight_layout()
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        RESULTS_DIRECTORY / "spatial_lle_response.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--f-real", type=float, default=1.8)
    parser.add_argument("--f-imag", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--final-time", type=float, default=50.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--noise", type=float, default=1e-3)
    parser.add_argument("--snapshots", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--initial-guess",
        choices=("empty", "pattern", "soliton", "scan", "cat", "seeded"),
        default="empty",
        help="Empty, pattern, one pulse, scan, CAT, or one-FSR parametric seeding",
    )
    parser.add_argument(
        "--scan-alpha-start",
        type=float,
        default=-2.0,
        help="Initial detuning for --initial-guess scan",
    )
    parser.add_argument(
        "--scan-time",
        type=float,
        default=30.0,
        help="Time over which detuning reaches the final --alpha",
    )
    parser.add_argument(
        "--cat-dwell-time",
        type=float,
        default=20.0,
        help="Initial Turing-roll dwell time for the CAT protocol",
    )
    parser.add_argument(
        "--cat-sweep-time",
        type=float,
        default=15.0,
        help="Coordinated detuning/pump sweep time for the CAT protocol",
    )
    parser.add_argument("--seed-power", type=float, default=0.3)
    parser.add_argument("--seed-mode", type=int, default=1)
    parser.add_argument("--seed-phase", type=float, default=0.0)
    parser.add_argument("--seed-scan-time", type=float, default=50.0)
    parser.add_argument("--seed-on-time", type=float, default=150.0)
    parser.add_argument("--seed-ramp-time", type=float, default=0.0)
    arguments = parser.parse_args()

    print("Solving spatial equation...", flush=True)
    forcing = complex(arguments.f_real, arguments.f_imag)
    theta_grid = np.linspace(0.0, 2.0 * np.pi, arguments.points, endpoint=False)
    alpha_schedule = None
    forcing_schedule = None
    alpha_start = None
    if arguments.initial_guess == "soliton":
        initial_background = single_soliton_seed(
            theta_grid, arguments.alpha, forcing, arguments.beta
        )
    elif arguments.initial_guess == "pattern":
        initial_background = uniform_states(arguments.alpha, forcing)[-1]
    elif arguments.initial_guess == "scan":
        if arguments.scan_time <= 0.0:
            parser.error("--scan-time must be positive")
        initial_background = 0.0j
        alpha_start = arguments.scan_alpha_start

        def alpha_schedule(time):
            fraction = min(max(time / arguments.scan_time, 0.0), 1.0)
            return alpha_start + fraction * (arguments.alpha - alpha_start)
    elif arguments.initial_guess == "cat":
        if arguments.cat_dwell_time < 0.0 or arguments.cat_sweep_time <= 0.0:
            parser.error("CAT dwell must be nonnegative and sweep time positive")
        alpha_start = 0.0
        initial_background = 0.0j
        sweep_start = arguments.cat_dwell_time
        sweep_end = sweep_start + arguments.cat_sweep_time
        if sweep_end >= arguments.final_time:
            parser.error("CAT dwell + sweep time must be less than --final-time")

        def cat_power(detuning):
            return 4.15 * np.exp(-3.09 * detuning) + 2.15 * np.exp(0.196 * detuning)

        power_offset = abs(forcing) ** 2 - cat_power(arguments.alpha)
        if cat_power(0.0) + power_offset <= 0.0:
            parser.error("requested endpoint gives nonpositive pump power on CAT")
        pump_phase = np.angle(forcing)

        def alpha_schedule(time):
            if time <= sweep_start:
                return 0.0
            fraction = min(max((time - sweep_start) / arguments.cat_sweep_time, 0.0), 1.0)
            return fraction * arguments.alpha

        def forcing_schedule(time):
            detuning = alpha_schedule(time)
            power = cat_power(detuning) + power_offset
            return np.sqrt(max(power, 0.0)) * np.exp(1j * pump_phase)
    elif arguments.initial_guess == "seeded":
        if (
            arguments.seed_power < 0.0
            or arguments.seed_scan_time <= 0.0
            or arguments.seed_on_time < arguments.seed_scan_time
            or arguments.seed_ramp_time < 0.0
        ):
            parser.error("invalid parametric-seed timing or power")
        if arguments.seed_on_time + arguments.seed_ramp_time >= arguments.final_time:
            parser.error("seed turn-on and ramp must finish before --final-time")
        initial_background = 0.0j
        alpha_start = arguments.scan_alpha_start
        seed_profile = (
            np.sqrt(arguments.seed_power)
            * np.exp(
                1j
                * (
                    arguments.seed_mode * theta_grid
                    + arguments.seed_phase
                )
            )
        )

        def alpha_schedule(time):
            fraction = min(max(time / arguments.seed_scan_time, 0.0), 1.0)
            return alpha_start + fraction * (arguments.alpha - alpha_start)

        def forcing_schedule(time):
            if arguments.seed_ramp_time == 0.0:
                seed_fraction = float(time >= arguments.seed_on_time)
            else:
                seed_fraction = min(
                    max(
                        (time - arguments.seed_on_time) / arguments.seed_ramp_time,
                        0.0,
                    ),
                    1.0,
                )
            return forcing + seed_fraction * seed_profile
    else:
        initial_background = 0.0j
    theta, times, fields = solve_lle(
        alpha=arguments.alpha,
        forcing=forcing,
        beta=arguments.beta,
        points=arguments.points,
        final_time=arguments.final_time,
        time_step=arguments.dt,
        noise=arguments.noise,
        snapshots=arguments.snapshots,
        seed=arguments.seed,
        initial_background=initial_background,
        alpha_schedule=alpha_schedule,
        forcing_schedule=forcing_schedule,
    )
    print("Saving figure...", flush=True)
    save_figure(
        theta,
        times,
        fields,
        arguments.alpha,
        forcing,
        arguments.beta,
        alpha_start=alpha_start,
    )
    print("Figure saved.", flush=True)


if __name__ == "__main__":
    main()
