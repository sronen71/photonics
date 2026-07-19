"""Quantitative PhCR benchmarks from Liu (2025) and Sakaue (2026)."""

import unittest

import numpy as np
from scipy.optimize import brentq

from bidirectional import BidirectionalParameters, solve_bidirectional_lle
from bidirectional_stability import (
    pump_split_effective_parameters,
    pump_split_sideband_gain,
)
from bidirectional_steady import solve_bidirectional_uniform_state


def pump_split_parameters(alpha, forcing, splitting, beta):
    """Build the pump-mode-only coupled LLE used in both papers."""
    return BidirectionalParameters(
        alpha=alpha,
        forcing=forcing,
        beta=beta,
        epsilon_phc=splitting,
        coupling_factor=1.0,
        reflectivity=0.0,
        reflector_phase=0.0,
        reflector_half_width=0,
    )


class Sakaue2026PhysicsBenchmarks(unittest.TestCase):
    def test_figure_3_modulation_instability(self):
        """Recover the direction and mode of the gains in Figs. 3(c,d)."""
        parameters = pump_split_parameters(
            alpha=0.0,
            forcing=np.sqrt(6.0),
            splitting=4.0,
            beta=-0.02,
        )
        forward, backward, residual, _ = solve_bidirectional_uniform_state(
            1.0 + 1.0j,
            0.5 + 0.5j,
            parameters,
            tolerance=1.0e-12,
        )
        self.assertLess(residual, 1.0e-12)

        modes = np.arange(1, 51)
        forward_gain, backward_gain = pump_split_sideband_gain(
            forward, backward, parameters, modes
        )
        peak_mode = int(modes[np.argmax(forward_gain)])

        # Figure 3(c) places the unstable FW band near mu=+/-20 with a
        # maximum normalized growth rate of about 0.6.  The BW field is
        # stable throughout this finite-sideband band.
        self.assertIn(peak_mode, range(19, 22))
        self.assertGreater(float(np.max(forward_gain)), 0.55)
        self.assertLess(float(np.max(forward_gain)), 0.62)
        self.assertLess(float(np.max(backward_gain)), 0.0)

        # Excite the unstable eigenmode and measure its exponential growth
        # with the full time-dependent solver.
        mismatch = (
            2.0 * (abs(forward) ** 2 + abs(backward) ** 2)
            - parameters.alpha
            + 0.5 * parameters.beta * peak_mode**2
        )
        stability_matrix = np.array(
            [
                [
                    -1.0 + 1j * mismatch,
                    1j * forward**2,
                ],
                [
                    -1j * np.conj(forward) ** 2,
                    -1.0 - 1j * mismatch,
                ],
            ]
        )
        eigenvalues, eigenvectors = np.linalg.eig(stability_matrix)
        unstable_index = int(np.argmax(eigenvalues.real))
        predicted_growth = float(eigenvalues[unstable_index].real)
        self.assertAlmostEqual(
            predicted_growth,
            float(np.max(forward_gain)),
            places=12,
        )

        spatial_points = 128
        theta = 2.0 * np.pi * np.arange(spatial_points) / spatial_points
        eigenvector = eigenvectors[:, unstable_index]
        perturbation = 1.0e-9 * (
            eigenvector[0] * np.exp(1j * peak_mode * theta)
            + np.conj(eigenvector[1])
            * np.exp(-1j * peak_mode * theta)
        )
        _, times, forward_fields, _ = solve_bidirectional_lle(
            parameters,
            spatial_points=spatial_points,
            final_time=8.0,
            time_step=0.005,
            initial_noise=0.0,
            snapshots=81,
            seed=0,
            initial_forward=forward + perturbation,
            initial_backward=backward,
        )
        forward_modes = np.fft.fft(
            forward_fields - forward, axis=1
        ) / spatial_points
        sideband_amplitude = np.sqrt(
            abs(forward_modes[:, peak_mode]) ** 2
            + abs(forward_modes[:, -peak_mode]) ** 2
        )
        measured_growth = float(
            np.polyfit(times, np.log(sideband_amplitude), 1)[0]
        )
        self.assertAlmostEqual(
            measured_growth, predicted_growth, delta=5.0e-3
        )

        detuned_parameters = pump_split_parameters(
            alpha=4.0,
            forcing=np.sqrt(6.0),
            splitting=4.0,
            beta=-0.02,
        )
        forward, backward, _, _ = solve_bidirectional_uniform_state(
            0.8 + 0.35j,
            -0.5 - 0.97j,
            detuned_parameters,
            tolerance=1.0e-12,
        )
        forward_gain, backward_gain = pump_split_sideband_gain(
            forward, backward, detuned_parameters, modes
        )

        # Figure 3(d) is centered at the pump mode.  Pump coupling makes the
        # exact mu=0 problem different from the nonzero-sideband formula, so
        # mu=1 is the closest physical sideband represented by that formula.
        self.assertLess(float(np.max(forward_gain)), 0.0)
        self.assertEqual(int(modes[np.argmax(backward_gain)]), 1)
        self.assertGreater(float(np.max(backward_gain)), 0.15)

    def test_figure_6_soliton_existence_boundaries(self):
        """Recover the four effective-LLE boundaries quoted for Fig. 6."""
        def existence_margin(alpha, direction, initial_fields):
            parameters = pump_split_parameters(
                alpha=alpha,
                forcing=np.sqrt(6.0),
                splitting=6.75,
                beta=-0.02,
            )
            forward, backward, _, _ = solve_bidirectional_uniform_state(
                *initial_fields,
                parameters,
                tolerance=1.0e-10,
            )
            effective = pump_split_effective_parameters(
                forward, backward, parameters
            )
            if direction == "forward":
                effective_alpha = effective.forward_alpha
                effective_forcing = effective.forward_forcing
            else:
                effective_alpha = effective.backward_alpha
                effective_forcing = effective.backward_forcing
            return (
                np.pi**2 * abs(effective_forcing) ** 2 / 8.0
                - effective_alpha
            )

        boundary_specs = (
            ("forward", (3.7, 4.0), (0.5 + 0.4j, -0.4 - 0.8j)),
            ("forward", (9.4, 9.8), (0.0j, 0.0j)),
            ("backward", (2.0, 2.3), (0.3 + 0.2j, -0.2 - 0.7j)),
            ("backward", (6.3, 6.5), (0.2 - 0.6j, -0.2 + 0.3j)),
        )
        measured = []
        for direction, bracket, initial_fields in boundary_specs:
            measured.append(
                brentq(
                    lambda alpha: existence_margin(
                        alpha, direction, initial_fields
                    ),
                    *bracket,
                    xtol=1.0e-11,
                )
            )

        # Sakaue and Kuse report FW [3.86, 9.62] and BW [2.18, 6.36].
        np.testing.assert_allclose(
            measured,
            [3.86, 9.62, 2.18, 6.36],
            rtol=0.0,
            atol=0.03,
        )


class Liu2025PhysicsBenchmarks(unittest.TestCase):
    def test_normal_dispersion_instability_onset(self):
        """Recover the primary-comb threshold at alpha approximately 5.5."""
        def backward_threshold(alpha):
            parameters = pump_split_parameters(
                alpha=alpha,
                forcing=3.0,
                splitting=9.3,
                beta=0.02,
            )
            _, backward, _, _ = solve_bidirectional_uniform_state(
                0.0j,
                0.0j,
                parameters,
                tolerance=1.0e-10,
            )
            # Liu et al. Eq. (4): normal-GVD primary combs require the
            # intracavity intensity in the corresponding direction to reach
            # one in normalized units.
            return abs(backward) ** 2 - 1.0

        onset = brentq(backward_threshold, 5.4, 5.7, xtol=1.0e-11)

        # Figure 4 quotes alpha about 5.5.  This onset does not depend on the
        # unreported magnitude of D2, unlike the later nonlinear evolution.
        self.assertGreater(onset, 5.5)
        self.assertLess(onset, 5.65)


if __name__ == "__main__":
    unittest.main()
