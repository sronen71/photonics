#!/usr/bin/env python3
"""Plot uniform steady states of the driven Kerr equation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CRITICAL_ALPHA = np.sqrt(3.0)
RESULTS_DIRECTORY = Path("results")


def pump_power(response_intensity, alpha):
    """Return |F|^2 for a uniform response intensity |A|^2."""
    return response_intensity * (1.0 + (alpha - response_intensity) ** 2)


def fold_points(alpha):
    """Return the two fold intensities when alpha > sqrt(3)."""
    if alpha <= CRITICAL_ALPHA:
        return np.array([])
    root = np.sqrt(alpha**2 - 3.0)
    return np.array([(2.0 * alpha - root) / 3.0, (2.0 * alpha + root) / 3.0])


def main():
    alphas = [0.0, 1.0, CRITICAL_ALPHA, 2.0, 3.0, 4.0]
    fixed_powers = [1.0, 2.0, 4.0, 8.0, 12.0]
    folds = [fold_points(alpha) for alpha in alphas]
    fold_intensities = np.concatenate([values for values in folds if values.size])
    fold_powers = np.concatenate(
        [pump_power(values, alpha) for alpha, values in zip(alphas, folds) if values.size]
    )

    maximum_intensity = 1.3 * fold_intensities.max()
    intensity = np.linspace(0.0, maximum_intensity, 2000)

    print("Preparing steady-state curves...", flush=True)
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

    minimum_alpha = -2.0
    maximum_alpha = 6.0
    power_colors = plt.cm.plasma(np.linspace(0.08, 0.9, len(fixed_powers)))
    for power, color in zip(fixed_powers, power_colors):
        # From |F|^2 = |A|^2[1 + (alpha - |A|^2)^2], the two
        # parametric branches are alpha = |A|^2 +/- sqrt(|F|^2/|A|^2 - 1).
        branch_intensity = np.linspace(1e-5, power, 4000)
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
    detuning_axis.set_ylim(0.0, max(fixed_powers))
    detuning_axis.set_xlabel(r"Detuning $\alpha$")
    detuning_axis.set_ylabel(r"Response intensity $|A|^2$")
    detuning_axis.set_title(r"Response versus detuning")
    detuning_axis.grid(alpha=0.25)
    detuning_axis.legend(fontsize=9)
    figure.tight_layout()
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        RESULTS_DIRECTORY / "steady_state_response.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)
    print("Figure saved.", flush=True)


if __name__ == "__main__":
    main()
