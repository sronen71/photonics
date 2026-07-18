#!/usr/bin/env python3
"""Solve the Lugiato--Lefever equation on a ring.

    dA/dt = -(1 + i*alpha)A + i|A|^2 A
            + i*D(-i*d/dtheta)A + F

The azimuthal coordinate theta is periodic on [0, 2*pi).  The pump F,
detuning alpha, and modal dispersion D(k) are constant.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config_loader import (
    ConfigurationError,
    config_parser,
    load_physics,
    load_section,
)
from dispersion import as_dispersion, load_dispersion, soliton_seed_beta


RESULTS_DIRECTORY = Path("results")
CONFIGURATION_KEYS = {
    "spatial_points",
    "final_time",
    "dt",
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


def linear_half_step_parameters(
    alpha, forcing, beta, mode_number, step, modal_dispersion=None
):
    """Return the exact driven-linear flow for half a time step."""
    if modal_dispersion is None:
        modal_dispersion = as_dispersion(beta).values(mode_number)
    linear_operator = -(1.0 + 1j * alpha) + 1j * modal_dispersion
    half_step = 0.5 * step
    propagator = np.exp(half_step * linear_operator)

    # The spatially uniform pump acts only on mode zero.  Its affine
    # contribution can be integrated exactly together with loss and
    # detuning: F * (exp(L_0 h/2) - 1) / L_0.
    pump_operator = linear_operator[0]
    drive_increment = (
        forcing * np.expm1(half_step * pump_operator) / pump_operator
    )
    return propagator, drive_increment


def split_step(
    amplitude,
    step,
    half_linear_propagator,
    half_drive_increment,
):
    """Advance one driven-linear/Kerr Strang split step."""
    first_linear_half = np.fft.ifft(
        half_linear_propagator * np.fft.fft(amplitude)
    ) + half_drive_increment
    nonlinear_field = first_linear_half * np.exp(
        1j * step * np.abs(first_linear_half) ** 2
    )
    return np.fft.ifft(
        half_linear_propagator * np.fft.fft(nonlinear_field)
    ) + half_drive_increment


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
):
    """Integrate using a driven-linear/Kerr split-step Fourier method."""
    if (
        spatial_points < 8
        or final_time <= 0.0
        or time_step <= 0.0
    ):
        raise ValueError(
            "spatial points and times must be positive"
        )

    theta = np.linspace(0.0, 2.0 * np.pi, spatial_points, endpoint=False)
    angular_step = 2.0 * np.pi / spatial_points
    # Integer azimuthal mode numbers for a 2*pi-periodic ring.
    mode_number = 2.0 * np.pi * np.fft.fftfreq(
        spatial_points, d=angular_step
    )
    modal_dispersion = as_dispersion(beta).values(mode_number)

    half_linear_step, half_drive_step = linear_half_step_parameters(
        alpha,
        forcing,
        beta,
        mode_number,
        time_step,
        modal_dispersion=modal_dispersion,
    )

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
            propagator, drive_increment = linear_half_step_parameters(
                step_alpha,
                forcing,
                beta,
                mode_number,
                step,
                modal_dispersion=modal_dispersion,
            )
        elif step != time_step:
            propagator, drive_increment = linear_half_step_parameters(
                alpha,
                forcing,
                beta,
                mode_number,
                step,
                modal_dispersion=modal_dispersion,
            )
        else:
            propagator = half_linear_step
            drive_increment = half_drive_step

        amplitude = split_step(
            amplitude,
            step,
            propagator,
            drive_increment,
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
    dispersion_text = as_dispersion(beta).description
    figure.suptitle(
        rf"Ring LLE: {alpha_text}, $F={forcing.real:g}$, {dispersion_text}",
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
        physics = load_physics(command_line.config)
        dispersion = load_dispersion(physics, command_line.config)
        arguments = load_section(command_line.config, "lle", CONFIGURATION_KEYS)
        arguments.alpha = float(physics.alpha)
        arguments.f_real = float(physics.f_real)
        for name in (
            "final_time",
            "dt",
            "initial_noise",
            "scan_alpha_start",
            "scan_time",
        ):
            setattr(arguments, name, float(getattr(arguments, name)))
        for name in (
            "spatial_points",
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
        or arguments.snapshots < 1
    ):
        parser.error(
            "lle.initial_noise must be nonnegative and snapshots must be "
            "positive"
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
            theta_grid, arguments.alpha, forcing, dispersion
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
        beta=dispersion,
        spatial_points=arguments.spatial_points,
        final_time=arguments.final_time,
        time_step=arguments.dt,
        initial_noise=arguments.initial_noise,
        snapshots=arguments.snapshots,
        seed=arguments.seed,
        initial_background=initial_background,
        alpha_schedule=alpha_schedule,
    )
    print("Saving figure...", flush=True)
    save_figure(
        theta,
        times,
        fields,
        arguments.alpha,
        forcing,
        dispersion,
        alpha_start=alpha_start,
    )
    print("Figure saved.", flush=True)


if __name__ == "__main__":
    main()
