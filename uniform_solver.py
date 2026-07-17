#!/usr/bin/env python3
"""Plot uniform responses of the driven Kerr equation."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config_loader import ConfigurationError, config_parser, load_section


CRITICAL_ALPHA = np.sqrt(3.0)
RESULTS_DIRECTORY = Path("results")
INTENSITY_SAMPLES = 2000
BRANCH_SAMPLES = 4000
CONFIGURATION_KEYS = {
    "alpha_grid",
    "power_grid",
    "minimum_alpha",
    "maximum_alpha",
}


def pump_power(response_intensity, alpha):
    """Return |F|^2 for a uniform response intensity |A|^2."""
    return response_intensity * (1.0 + (alpha - response_intensity) ** 2)


def fold_points(alpha):
    """Return the two fold intensities when alpha > sqrt(3)."""
    if alpha <= CRITICAL_ALPHA:
        return np.array([])
    root = np.sqrt(alpha**2 - 3.0)
    return np.array([(2.0 * alpha - root) / 3.0, (2.0 * alpha + root) / 3.0])


def inclusive_grid(configuration, name):
    """Return a validated start-to-end grid from a YAML mapping."""
    if not isinstance(configuration, dict):
        raise ConfigurationError(f"uniform.{name} must be a mapping")
    expected = {"start", "end", "increment"}
    supplied = set(configuration)
    missing = sorted(expected - supplied)
    unknown = sorted(supplied - expected)
    if missing:
        raise ConfigurationError(
            f"uniform.{name} is missing: {', '.join(missing)}"
        )
    if unknown:
        raise ConfigurationError(
            f"uniform.{name} has unknown settings: {', '.join(unknown)}"
        )

    start = float(configuration["start"])
    end = float(configuration["end"])
    increment = float(configuration["increment"])
    if increment <= 0.0 or end < start:
        raise ConfigurationError(
            f"uniform.{name} requires increment > 0 and end >= start"
        )
    intervals = (end - start) / increment
    rounded_intervals = round(intervals)
    if not np.isclose(intervals, rounded_intervals):
        raise ConfigurationError(
            f"uniform.{name}.increment must land exactly on the end value"
        )
    return start + increment * np.arange(rounded_intervals + 1)


def main():
    parser = config_parser(__doc__)
    command_line = parser.parse_args()
    try:
        configuration = load_section(
            command_line.config, "uniform", CONFIGURATION_KEYS
        )
        alphas = inclusive_grid(configuration.alpha_grid, "alpha_grid")
        powers = inclusive_grid(configuration.power_grid, "power_grid")
        minimum_alpha = float(configuration.minimum_alpha)
        maximum_alpha = float(configuration.maximum_alpha)
    except (ConfigurationError, TypeError, ValueError) as error:
        parser.error(str(error))

    if np.any(powers <= 0.0):
        parser.error("uniform.power_grid must contain positive values")
    if minimum_alpha >= maximum_alpha:
        parser.error("uniform.minimum_alpha must be less than maximum_alpha")

    if alphas[0] <= CRITICAL_ALPHA <= alphas[-1] and not np.any(
        np.isclose(alphas, CRITICAL_ALPHA)
    ):
        alphas = np.sort(np.append(alphas, CRITICAL_ALPHA))
    folds = [fold_points(alpha) for alpha in alphas]
    if not any(values.size for values in folds):
        parser.error(r"uniform.alphas must include a value greater than sqrt(3)")
    fold_intensities = np.concatenate([values for values in folds if values.size])
    fold_powers = np.concatenate(
        [
            pump_power(values, alpha)
            for alpha, values in zip(alphas, folds)
            if values.size
        ]
    )

    maximum_intensity = 1.3 * fold_intensities.max()
    intensity = np.linspace(0.0, maximum_intensity, INTENSITY_SAMPLES)

    print("Preparing uniform response curves...", flush=True)
    figure, (power_axis, detuning_axis) = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(alphas)))

    for alpha, color in zip(alphas, colors):
        critical = np.isclose(alpha, CRITICAL_ALPHA)
        power_axis.plot(
            pump_power(intensity, alpha),
            intensity,
            color="black" if critical else color,
            linestyle="--" if critical else "-",
            linewidth=2.2 if critical else 2.0,
            alpha=0.9,
            label=(
                r"critical $\alpha=\sqrt{3}$"
                if critical else rf"$\alpha={alpha:g}$"
            ),
        )

    power_axis.set_xlim(0.0, 1.25 * fold_powers.max())
    power_axis.set_ylim(0.0, maximum_intensity)
    power_axis.set_xlabel(r"Pump intensity $|F|^2$")
    power_axis.set_ylabel(r"Response intensity $|A|^2$")
    power_axis.set_title(r"Response versus pump intensity")
    power_axis.grid(alpha=0.25)
    power_axis.legend(fontsize=9, ncol=2)

    power_colors = plt.cm.plasma(np.linspace(0.08, 0.9, len(powers)))
    for power, color in zip(powers, power_colors):
        # From |F|^2 = |A|^2[1 + (alpha - |A|^2)^2], the two
        # parametric branches are alpha = |A|^2 +/- sqrt(|F|^2/|A|^2 - 1).
        branch_intensity = np.linspace(1e-5, power, BRANCH_SAMPLES)
        offset = np.sqrt(power / branch_intensity - 1.0)
        for sign, label in ((-1.0, rf"$|F|^2={power:g}$"), (1.0, None)):
            branch_alpha = branch_intensity + sign * offset
            visible = (branch_alpha >= minimum_alpha) & (branch_alpha <= maximum_alpha)
            detuning_axis.plot(
                branch_alpha[visible],
                branch_intensity[visible],
                color=color,
                linewidth=2.0,
                alpha=0.9,
                label=label,
            )

    detuning_axis.set_xlim(minimum_alpha, maximum_alpha)
    detuning_axis.set_ylim(0.0, max(powers))
    detuning_axis.set_xlabel(r"Detuning $\alpha$")
    detuning_axis.set_ylabel(r"Response intensity $|A|^2$")
    detuning_axis.set_title(r"Response versus detuning")
    detuning_axis.grid(alpha=0.25)
    detuning_axis.legend(fontsize=9)
    figure.tight_layout()
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        RESULTS_DIRECTORY / "uniform_response.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)
    print("Figure saved.", flush=True)


if __name__ == "__main__":
    main()
