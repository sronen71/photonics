#!/usr/bin/env python3
"""Solve the Lugiato--Lefever equation on a ring.

    dA/dt = -(1 + i*alpha)A + i|A|^2 A
            - i*(beta/2)*d^2A/dtheta^2 + F

The azimuthal coordinate theta is periodic on [0, 2*pi).  The pump F,
detuning alpha, and dispersion beta are constant.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config_loader import (
    PHYSICS_KEYS,
    ConfigurationError,
    config_parser,
    load_section,
)


RESULTS_DIRECTORY = Path("results")
CONFIGURATION_KEYS = {
    "spatial_points",
    "final_time",
    "dt",
    "split_step_tolerance",
    "split_step_max_iterations",
    "initial_noise",
    "snapshots",
    "seed",
    "initial_shape",
    "operation_mode",
    "scan_alpha_start",
    "scan_time",
}
INITIAL_SHAPES = {"empty", "soliton"}
OPERATION_MODES = {"direct", "scan"}


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


def iterative_split_step(
    amplitude,
    step,
    forcing,
    half_linear_propagator,
    tolerance,
    max_iterations,
):
    """Advance one pyLLE-style symmetric split step."""
    # Apply the constant drive explicitly, as in pyLLE's SSFM half-step.
    driven_field = amplitude + step * forcing
    first_linear_half = np.fft.ifft(
        half_linear_propagator * np.fft.fft(driven_field)
    )
    initial_nonlinearity = 1j * np.abs(driven_field) ** 2
    iterate = driven_field

    for _ in range(max_iterations):
        final_nonlinearity = 1j * np.abs(iterate) ** 2
        averaged_nonlinearity = 0.5 * (
            initial_nonlinearity + final_nonlinearity
        )
        nonlinear_field = first_linear_half * np.exp(
            step * averaged_nonlinearity
        )
        updated = np.fft.ifft(
            half_linear_propagator * np.fft.fft(nonlinear_field)
        )

        scale = max(np.linalg.norm(iterate), np.finfo(float).tiny)
        relative_change = np.linalg.norm(updated - iterate) / scale
        if relative_change < tolerance:
            return updated
        iterate = updated

    raise RuntimeError(
        "iterative split step did not converge; reduce lle.dt in the "
        "configuration"
    )


def solve_lle(
    alpha: float,
    forcing: complex,
    beta: float,
    spatial_points: int,
    final_time: float,
    time_step: float,
    initial_noise: float,
    snapshots: int,
    seed: int,
    initial_background: complex = 0.0j,
    alpha_schedule=None,
    split_step_tolerance: float = 1.0e-2,
    split_step_max_iterations: int = 10,
):
    """Integrate using an iterative symmetric split-step Fourier method."""
    if (
        spatial_points < 8
        or final_time <= 0.0
        or time_step <= 0.0
        or split_step_tolerance <= 0.0
        or split_step_max_iterations < 1
    ):
        raise ValueError(
            "spatial points, times, and split-step convergence settings "
            "must be positive"
        )

    theta = np.linspace(0.0, 2.0 * np.pi, spatial_points, endpoint=False)
    angular_step = 2.0 * np.pi / spatial_points
    # Integer azimuthal mode numbers for a 2*pi-periodic ring.
    mode_number = 2.0 * np.pi * np.fft.fftfreq(
        spatial_points, d=angular_step
    )

    # Fourier representation of -(1+i*alpha)A - i*(beta/2)A_theta,theta.
    linear_operator = -(1.0 + 1j * alpha) + 0.5j * beta * mode_number**2
    half_linear_step = np.exp(0.5 * time_step * linear_operator)

    rng = np.random.default_rng(seed)
    # Seed all spatial modes with small complex noise.  The time-dependent
    # executable uses the default empty-cavity background; stationary solvers
    # may supply a different branch as their initial guess.
    amplitude = initial_background + initial_noise * (
        rng.standard_normal(spatial_points)
        + 1j * rng.standard_normal(spatial_points)
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

        amplitude = iterative_split_step(
            amplitude,
            step,
            forcing,
            propagator,
            split_step_tolerance,
            split_step_max_iterations,
        )

        if not np.all(np.isfinite(amplitude)):
            raise RuntimeError("solution diverged; reduce lle.dt in the configuration")

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
        rf"Ring LLE: {alpha_text}, $F={forcing.real:g}$, "
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
    parser = config_parser(__doc__)
    command_line = parser.parse_args()
    try:
        physics = load_section(command_line.config, "physics", PHYSICS_KEYS)
        arguments = load_section(command_line.config, "lle", CONFIGURATION_KEYS)
        for name in PHYSICS_KEYS:
            setattr(arguments, name, float(getattr(physics, name)))
        for name in (
            "final_time",
            "dt",
            "split_step_tolerance",
            "initial_noise",
            "scan_alpha_start",
            "scan_time",
        ):
            setattr(arguments, name, float(getattr(arguments, name)))
        for name in (
            "spatial_points",
            "split_step_max_iterations",
            "snapshots",
            "seed",
        ):
            setattr(arguments, name, int(getattr(arguments, name)))
    except (ConfigurationError, TypeError, ValueError) as error:
        parser.error(str(error))
    if arguments.initial_shape not in INITIAL_SHAPES:
        parser.error(
            "lle.initial_shape must be one of: "
            + ", ".join(sorted(INITIAL_SHAPES))
        )
    if arguments.operation_mode not in OPERATION_MODES:
        parser.error(
            "lle.operation_mode must be one of: "
            + ", ".join(sorted(OPERATION_MODES))
        )
    if (
        arguments.initial_noise < 0.0
        or arguments.split_step_tolerance <= 0.0
        or arguments.split_step_max_iterations < 1
        or arguments.snapshots < 1
    ):
        parser.error(
            "lle.initial_noise must be nonnegative; split-step convergence "
            "settings and snapshots must be positive"
        )

    print("Solving spatial equation...", flush=True)
    forcing = complex(arguments.f_real)
    theta_grid = np.linspace(
        0.0, 2.0 * np.pi, arguments.spatial_points, endpoint=False
    )
    alpha_schedule = None
    alpha_start = None
    if arguments.initial_shape == "soliton":
        initial_background = single_soliton_seed(
            theta_grid, arguments.alpha, forcing, arguments.beta
        )
    else:
        initial_background = 0.0j

    if arguments.operation_mode == "scan":
        if arguments.scan_time <= 0.0:
            parser.error("lle.scan_time must be positive")
        alpha_start = arguments.scan_alpha_start

        def alpha_schedule(time):
            fraction = min(max(time / arguments.scan_time, 0.0), 1.0)
            return alpha_start + fraction * (arguments.alpha - alpha_start)
    theta, times, fields = solve_lle(
        alpha=arguments.alpha,
        forcing=forcing,
        beta=arguments.beta,
        spatial_points=arguments.spatial_points,
        final_time=arguments.final_time,
        time_step=arguments.dt,
        initial_noise=arguments.initial_noise,
        snapshots=arguments.snapshots,
        seed=arguments.seed,
        initial_background=initial_background,
        alpha_schedule=alpha_schedule,
        split_step_tolerance=arguments.split_step_tolerance,
        split_step_max_iterations=arguments.split_step_max_iterations,
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
