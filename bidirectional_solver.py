#!/usr/bin/env python3
"""Run the bidirectional photonic-crystal-resonator LLE model.

The model follows Zang et al., Nature Photonics 19, 510--517 (2025),
while retaining this repository's detuning convention

    alpha = 2 * (omega_center - omega_pump) / kappa.
"""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bidirectional import (
    BidirectionalParameters,
    bidirectional_residual,
    loaded_linewidth_rad_s,
    normalized_beta_2,
    solve_bidirectional_lle,
)
from bidirectional_spectrum import bidirectional_output
from bidirectional_steady import solve_bidirectional_steady_state
from config_loader import (
    ConfigurationError,
    finite_float,
    finite_integer,
    load_section,
)


RESULTS_DIRECTORY = Path("results")
DEFAULT_CONFIG_PATH = (
    Path(__file__).with_name("configs") / "config_bidirectional.yaml"
)

PHYSICS_KEYS = {
    "alpha",
    "f_real",
    "center_frequency_hz",
    "quality_factor_intrinsic",
    "d2_rad_s",
    "epsilon_phc",
    "coupling_factor",
    "reflectivity",
    "reflector_phase",
    "reflector_half_width",
    "fsr_hz",
}

SOLVER_KEYS = {
    "spatial_points",
    "final_time",
    "dt",
    "initial_noise",
    "snapshots",
    "seed",
    "operation_mode",
    "scan_alpha_start",
    "scan_time",
    "spectrum_average_time",
    "refine_steady_state",
    "steady_tolerance",
    "steady_max_iterations",
}


def load_parameters(config_path):
    """Load model parameters and display-frequency metadata."""
    values = load_section(config_path, "bidirectional_physics", PHYSICS_KEYS)
    numbers = {
        name: finite_float(
            getattr(values, name), f"bidirectional_physics.{name}"
        )
        for name in PHYSICS_KEYS - {"reflector_half_width"}
    }
    half_width = finite_integer(
        values.reflector_half_width,
        "bidirectional_physics.reflector_half_width",
    )
    if numbers["f_real"] < 0.0:
        raise ConfigurationError("bidirectional_physics.f_real must be nonnegative")
    if (
        numbers["center_frequency_hz"] <= 0.0
        or numbers["quality_factor_intrinsic"] <= 0.0
        or numbers["fsr_hz"] <= 0.0
    ):
        raise ConfigurationError(
            "bidirectional_physics optical frequencies must be positive"
        )
    beta = normalized_beta_2(
        numbers["d2_rad_s"],
        numbers["center_frequency_hz"],
        numbers["quality_factor_intrinsic"],
        numbers["coupling_factor"],
    )
    parameters = BidirectionalParameters(
        alpha=numbers["alpha"],
        forcing=complex(numbers["f_real"]),
        beta=beta,
        epsilon_phc=numbers["epsilon_phc"],
        coupling_factor=numbers["coupling_factor"],
        reflectivity=numbers["reflectivity"],
        reflector_phase=numbers["reflector_phase"],
        reflector_half_width=half_width,
    )
    kappa = loaded_linewidth_rad_s(
        numbers["center_frequency_hz"],
        numbers["quality_factor_intrinsic"],
        numbers["coupling_factor"],
    )
    pump_frequency_hz = (
        numbers["center_frequency_hz"]
        - numbers["alpha"] * kappa / (4.0 * np.pi)
    )
    if pump_frequency_hz <= 0.0:
        raise ConfigurationError(
            "bidirectional_physics detuning gives a nonpositive pump frequency"
        )
    return parameters, pump_frequency_hz, numbers["fsr_hz"]


def load_solver_settings(config_path):
    """Load and validate numerical integration settings."""
    settings = load_section(config_path, "bidirectional_lle", SOLVER_KEYS)
    for name in (
        "final_time",
        "dt",
        "initial_noise",
        "scan_alpha_start",
        "scan_time",
        "spectrum_average_time",
        "steady_tolerance",
    ):
        setattr(
            settings,
            name,
            finite_float(
                getattr(settings, name), f"bidirectional_lle.{name}"
            ),
        )
    for name in (
        "spatial_points",
        "snapshots",
        "seed",
        "steady_max_iterations",
    ):
        setattr(
            settings,
            name,
            finite_integer(
                getattr(settings, name), f"bidirectional_lle.{name}"
            ),
        )
    if not isinstance(settings.refine_steady_state, bool):
        raise ConfigurationError(
            "bidirectional_lle.refine_steady_state must be true or false"
        )
    if settings.operation_mode not in {"direct", "scan"}:
        raise ConfigurationError(
            "bidirectional_lle.operation_mode must be direct or scan"
        )
    if (
        settings.spatial_points < 8
        or settings.final_time <= 0.0
        or settings.dt <= 0.0
        or settings.initial_noise < 0.0
        or settings.snapshots < 1
        or settings.spectrum_average_time <= 0.0
        or settings.spectrum_average_time > settings.final_time
        or settings.steady_tolerance <= 0.0
        or settings.steady_max_iterations < 1
        or (settings.operation_mode == "scan" and settings.scan_time <= 0.0)
    ):
        raise ConfigurationError(
            "bidirectional solver grid and time settings are invalid"
        )
    return settings


def alpha_schedule(parameters, settings):
    """Return the requested increasing-detuning schedule, if any."""
    if settings.operation_mode == "direct":
        return None

    def schedule(time):
        fraction = min(max(time / settings.scan_time, 0.0), 1.0)
        return settings.scan_alpha_start + fraction * (
            parameters.alpha - settings.scan_alpha_start
        )

    return schedule


def _analysis_indices(times, duration):
    indices = np.flatnonzero(times >= times[-1] - duration)
    if indices.size < 2:
        indices = np.arange(max(0, times.size - 2), times.size)
    return indices


def steady_result_payload(
    steady_forward,
    steady_backward,
    steady_residual,
    steady_solver_steps,
    parameters,
    pump_frequency_hz=None,
    fsr_hz=None,
):
    """Return archive entries for a Newton-refined bidirectional state."""
    steady_output = bidirectional_output(
        steady_forward,
        steady_backward,
        parameters,
        pump_frequency_hz=pump_frequency_hz,
        fsr_hz=fsr_hz,
    )
    return {
        "steady_forward": steady_forward,
        "steady_backward": steady_backward,
        "steady_maximum_residual": steady_residual,
        "steady_solver_steps": steady_solver_steps,
        "steady_forward_output_power_ratio": (
            steady_output["forward_power_ratio"]
        ),
        "steady_backward_output_power_ratio": (
            steady_output["backward_power_ratio"]
        ),
        "steady_pump_power_ratio": steady_output["pump_power_ratio"][0],
        "steady_conversion_efficiency": (
            steady_output["conversion_efficiency"][0]
        ),
        "steady_intrinsic_loss_ratio": (
            steady_output["intrinsic_loss_ratio"][0]
        ),
        "refined_steady_energy_balance": (
            steady_output["steady_energy_balance"][0]
        ),
    }


def save_results(
    theta,
    times,
    forward_fields,
    backward_fields,
    parameters,
    pump_frequency_hz,
    fsr_hz,
    average_time,
    steady_state=None,
):
    """Save histories, port powers, residuals, and a summary figure."""
    indices = _analysis_indices(times, average_time)
    output = bidirectional_output(
        forward_fields[indices],
        backward_fields[indices],
        parameters,
        times=times[indices],
        pump_frequency_hz=pump_frequency_hz,
        fsr_hz=fsr_hz,
    )
    trace = bidirectional_output(
        forward_fields,
        backward_fields,
        parameters,
        pump_frequency_hz=pump_frequency_hz,
        fsr_hz=fsr_hz,
    )
    residuals = bidirectional_residual(
        forward_fields[-1], backward_fields[-1], parameters
    )
    maximum_residual = max(
        float(np.max(np.abs(residual))) for residual in residuals
    )

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    saved = dict(
        theta=theta,
        times=times,
        forward_fields=forward_fields,
        backward_fields=backward_fields,
        output_mode_number=output["mode_number"],
        output_frequency_thz=output["optical_frequency_thz"],
        forward_output_power_ratio=output["forward_power_ratio"],
        backward_output_power_ratio=output["backward_power_ratio"],
        pump_power_ratio=trace["pump_power_ratio"],
        forward_comb_power_ratio=trace["forward_comb_ratio"],
        backward_comb_power_ratio=trace["backward_comb_ratio"],
        conversion_efficiency=trace["conversion_efficiency"],
        pump_consumption=trace["pump_consumption"],
        intrinsic_loss_ratio=trace["intrinsic_loss_ratio"],
        steady_energy_balance=trace["steady_energy_balance"],
        maximum_bidirectional_residual=maximum_residual,
        **parameters.__dict__,
    )
    if steady_state is not None:
        (
            steady_forward,
            steady_backward,
            steady_residual,
            steady_solver_steps,
        ) = steady_state
        saved.update(steady_result_payload(
            steady_forward,
            steady_backward,
            steady_residual,
            steady_solver_steps,
            parameters,
            pump_frequency_hz,
            fsr_hz,
        ))
    np.savez(
        RESULTS_DIRECTORY / "bidirectional_lle_output.npz",
        **saved,
    )
    save_figure(
        theta,
        times,
        forward_fields,
        backward_fields,
        parameters,
        output,
        trace,
    )
    return output, trace, maximum_residual


def save_figure(
    theta,
    times,
    forward_fields,
    backward_fields,
    parameters,
    output,
    trace,
):
    """Plot the intracavity fields and both external ports."""
    figure, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    image = axes[0, 0].pcolormesh(
        theta,
        times,
        np.abs(backward_fields) ** 2,
        shading="auto",
        cmap="magma",
    )
    axes[0, 0].set(
        xlabel=r"Azimuthal angle $\theta$",
        ylabel="Normalized time",
        title="Backward intracavity intensity",
    )
    figure.colorbar(image, ax=axes[0, 0], label=r"$|E^b|^2$")

    axes[0, 1].plot(times, trace["pump_power_ratio"], label="remaining pump")
    axes[0, 1].plot(times, trace["forward_comb_ratio"], label="forward comb")
    axes[0, 1].plot(times, trace["backward_comb_ratio"], label="backward comb")
    axes[0, 1].set(
        xlabel="Normalized time",
        ylabel=r"Power / $P_{\rm pump}$",
        title="Circuit power flow",
        ylim=(0.0, None),
    )
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()

    axes[1, 0].plot(theta, np.abs(forward_fields[-1]) ** 2, label="forward")
    axes[1, 0].plot(theta, np.abs(backward_fields[-1]) ** 2, label="backward")
    axes[1, 0].set(
        xlabel=r"Azimuthal angle $\theta$",
        ylabel="Intracavity intensity",
        title="Final intracavity profiles",
    )
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    floor = 1.0e-15
    for direction, color in (("forward", "tab:green"), ("backward", "tab:blue")):
        axes[1, 1].plot(
            output["optical_frequency_thz"],
            10.0
            * np.log10(np.maximum(output[f"{direction}_power_ratio"], floor)),
            color=color,
            label=direction,
        )
    axes[1, 1].set(
        xlabel="Optical frequency (THz)",
        ylabel=r"Power / $P_{\rm pump}$ (dB)",
        title="Time-averaged port spectra",
        ylim=(-100.0, 5.0),
    )
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()

    figure.suptitle(
        rf"Bidirectional PhCR LLE: $\alpha={parameters.alpha:g}$, "
        rf"$F^2={abs(parameters.forcing) ** 2:g}$, "
        rf"$K={parameters.coupling_factor:g}$, "
        rf"$R={parameters.reflectivity:g}$, "
        rf"$\phi={parameters.reflector_phase / np.pi:g}\pi$"
    )
    figure.tight_layout()
    figure.savefig(
        RESULTS_DIRECTORY / "bidirectional_lle_response.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    command_line = parser.parse_args()
    try:
        parameters, pump_frequency_hz, fsr_hz = load_parameters(
            command_line.config
        )
        settings = load_solver_settings(command_line.config)
    except (ConfigurationError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(
        "Solving bidirectional PhCR LLE with "
        f"alpha={parameters.alpha:g}, F^2={abs(parameters.forcing) ** 2:g}, "
        f"K={parameters.coupling_factor:g}, R={parameters.reflectivity:g}, "
        f"phi={parameters.reflector_phase / np.pi:g} pi...",
        flush=True,
    )
    theta, times, forward, backward = solve_bidirectional_lle(
        parameters,
        spatial_points=settings.spatial_points,
        final_time=settings.final_time,
        time_step=settings.dt,
        initial_noise=settings.initial_noise,
        snapshots=settings.snapshots,
        seed=settings.seed,
        alpha_schedule=alpha_schedule(parameters, settings),
    )
    steady_state = None
    if settings.refine_steady_state:
        print("Refining the final even-dispersion steady state...", flush=True)
        steady_state = solve_bidirectional_steady_state(
            forward[-1],
            backward[-1],
            parameters,
            tolerance=settings.steady_tolerance,
            max_iterations=settings.steady_max_iterations,
        )
    _, trace, residual = save_results(
        theta,
        times,
        forward,
        backward,
        parameters,
        pump_frequency_hz,
        fsr_hz,
        settings.spectrum_average_time,
        steady_state=steady_state,
    )
    print(
        "Final power ratios: "
        f"pump={trace['pump_power_ratio'][-1]:.4f}, "
        f"forward comb={trace['forward_comb_ratio'][-1]:.4f}, "
        f"backward comb={trace['backward_comb_ratio'][-1]:.4f}, "
        f"CE={trace['conversion_efficiency'][-1]:.4f}; "
        f"maximum residual={residual:.3e}",
        flush=True,
    )
    if steady_state is not None:
        print(
            "Steady refinement: "
            f"residual={steady_state[2]:.3e}, "
            f"solver steps={steady_state[3]}",
            flush=True,
        )
    print("Results saved.", flush=True)


if __name__ == "__main__":
    main()
