"""Physics benchmarks for the bidirectional PhCR LLE."""

import unittest

import numpy as np
from scipy.linalg import expm

from bidirectional import (
    BidirectionalParameters,
    apply_linear_step,
    bidirectional_residual,
    linear_step_parameters,
    nonlinear_step,
    normalized_beta_2,
    solve_bidirectional_lle,
)
from bidirectional_spectrum import bidirectional_output
from bidirectional_steady import solve_bidirectional_steady_state
from lle_solver import solve_lle
from spectral import mode_numbers


class BidirectionalPhCRPhysicsBenchmarks(unittest.TestCase):
    def test_physical_d2_uses_repository_dispersion_sign(self):
        """Map the paper's normal D2 to positive beta2 in this code."""
        beta = normalized_beta_2(
            d2_rad_s=-2.0 * np.pi * 8.5e6,
            center_frequency_hz=193.1e12,
            quality_factor_intrinsic=2.7e6,
            coupling_factor=3.0,
        )
        self.assertGreater(beta, 0.0)
        self.assertAlmostEqual(beta, 0.0594251683065769, places=14)

    def test_uncoupled_forward_field_reduces_to_scalar_lle(self):
        """Recover the existing scalar solver when both couplings vanish."""
        parameters = BidirectionalParameters(
            alpha=1.2,
            forcing=1.0e-4,
            beta=-0.1,
            epsilon_phc=0.0,
            coupling_factor=2.0,
            reflectivity=0.0,
            reflector_phase=0.0,
            reflector_half_width=0,
        )
        _, times, forward, backward = solve_bidirectional_lle(
            parameters,
            spatial_points=32,
            final_time=1.0,
            time_step=0.02,
            initial_noise=0.0,
            snapshots=3,
            seed=0,
        )
        _, scalar_times, scalar = solve_lle(
            alpha=parameters.alpha,
            forcing=parameters.forcing,
            beta=parameters.beta,
            spatial_points=32,
            final_time=1.0,
            time_step=0.02,
            initial_noise=0.0,
            snapshots=3,
            seed=0,
        )
        np.testing.assert_array_equal(times, scalar_times)
        np.testing.assert_allclose(forward, scalar, rtol=0.0, atol=2.0e-19)
        np.testing.assert_array_equal(backward, 0.0)

    def test_linear_reflector_coupling_matches_matrix_exponential(self):
        """Check the exact two-direction modal linear subflow."""
        spatial_points = 32
        selected_mode = 4
        theta = 2.0 * np.pi * np.arange(spatial_points) / spatial_points
        forward_amplitude = 0.2 + 0.1j
        backward_amplitude = -0.05 + 0.07j
        forward = forward_amplitude * np.exp(1j * selected_mode * theta)
        backward = backward_amplitude * np.exp(1j * selected_mode * theta)
        parameters = BidirectionalParameters(
            alpha=0.7,
            forcing=0.0,
            beta=-0.08,
            epsilon_phc=2.0,
            coupling_factor=3.0,
            reflectivity=0.64,
            reflector_phase=0.3,
            reflector_half_width=8,
        )
        duration = 0.37
        modes = mode_numbers(spatial_points)
        step = linear_step_parameters(parameters, modes, duration)
        numerical = apply_linear_step(forward, backward, step)

        dispersion = 0.5 * parameters.beta * selected_mode**2
        diagonal = -(1.0 + 1j * parameters.alpha) + 1j * dispersion
        generator = np.array(
            [
                [diagonal, 0.0],
                [
                    -parameters.bus_coupling
                    * parameters.reflector_amplitude,
                    diagonal,
                ],
            ],
            dtype=complex,
        )
        expected_amplitudes = expm(duration * generator) @ np.array(
            [forward_amplitude, backward_amplitude]
        )
        expected = tuple(
            amplitude * np.exp(1j * selected_mode * theta)
            for amplitude in expected_amplitudes
        )
        for measured, reference in zip(numerical, expected):
            np.testing.assert_allclose(
                measured, reference, rtol=2.0e-15, atol=2.0e-15
            )

    def test_kerr_subflow_has_factor_two_xpm_and_conserves_power(self):
        """Verify the exact counterpropagating nonlinear phase rotation."""
        forward = np.array([0.4 + 0.2j, -0.1 + 0.3j, 0.2 - 0.5j])
        backward = np.array([0.15 - 0.2j, 0.4 + 0.1j, -0.3j])
        duration = 0.23
        forward_power = np.mean(np.abs(forward) ** 2)
        backward_power = np.mean(np.abs(backward) ** 2)
        measured_forward, measured_backward = nonlinear_step(
            forward, backward, duration
        )
        expected_forward = forward * np.exp(
            1j * duration * (np.abs(forward) ** 2 + 2.0 * backward_power)
        )
        expected_backward = backward * np.exp(
            1j * duration * (np.abs(backward) ** 2 + 2.0 * forward_power)
        )
        np.testing.assert_allclose(measured_forward, expected_forward)
        np.testing.assert_allclose(measured_backward, expected_backward)
        np.testing.assert_allclose(abs(measured_forward), abs(forward))
        np.testing.assert_allclose(abs(measured_backward), abs(backward))

    def test_steady_port_power_obeys_energy_conservation(self):
        """Close the input/output power budget of a nonlinear CW state."""
        parameters = BidirectionalParameters(
            alpha=0.4,
            forcing=0.3,
            beta=0.05,
            epsilon_phc=1.0,
            coupling_factor=2.0,
            reflectivity=0.6,
            reflector_phase=0.2,
            reflector_half_width=3,
        )
        _, _, forward, backward = solve_bidirectional_lle(
            parameters,
            spatial_points=16,
            final_time=15.0,
            time_step=0.02,
            initial_noise=0.0,
            snapshots=2,
            seed=0,
        )
        steady_forward, steady_backward, residual, _ = (
            solve_bidirectional_steady_state(
                forward[-1],
                backward[-1],
                parameters,
                tolerance=1.0e-10,
                max_iterations=100,
            )
        )
        self.assertLess(residual, 1.0e-10)
        output = bidirectional_output(
            steady_forward, steady_backward, parameters
        )
        self.assertAlmostEqual(
            output["steady_energy_balance"][0], 1.0, places=10
        )
        residuals = bidirectional_residual(
            steady_forward, steady_backward, parameters
        )
        self.assertLess(max(np.max(abs(value)) for value in residuals), 1.0e-10)

    def test_paper_regime_has_one_sided_high_efficiency_soliton(self):
        """Reproduce the high-efficiency phase contrast of Figs. 1e and S2."""
        beta = normalized_beta_2(
            d2_rad_s=-2.0 * np.pi * 8.5e6,
            center_frequency_hz=193.1e12,
            quality_factor_intrinsic=2.7e6,
            coupling_factor=3.0,
        )

        # The absolute coherent-scattering phase is a gauge/device reference
        # not fixed by the listed scalar parameters.  Calibrate that reference
        # once to the paper's constructive-interference phi=0 device.  Adding
        # pi then represents its destructive-interference phi=pi device.
        phase_reference = 1.1 * np.pi
        efficiencies = {}
        pump_ratios = {}
        backward_fractions = {}
        variations = {}
        for label, phase in (
            ("constructive", 0.0),
            ("destructive", np.pi),
        ):
            parameters = BidirectionalParameters(
                alpha=6.98,
                forcing=2.1,
                beta=beta,
                epsilon_phc=6.0,
                coupling_factor=3.0,
                reflectivity=0.97,
                reflector_phase=phase,
                reflector_half_width=19,
                reflector_phase_reference=phase_reference,
            )

            def scan(time):
                return 3.0 + (parameters.alpha - 3.0) * min(time / 50.0, 1.0)

            _, _, forward, backward = solve_bidirectional_lle(
                parameters,
                spatial_points=64,
                final_time=80.0,
                time_step=0.01,
                initial_noise=0.003,
                snapshots=161,
                seed=7,
                alpha_schedule=scan,
            )
            output = bidirectional_output(forward, backward, parameters)
            efficiencies[label] = float(output["conversion_efficiency"][-1])
            pump_ratios[label] = float(output["pump_power_ratio"][-1])
            backward_fractions[label] = float(
                output["backward_comb_ratio"][-1]
                / max(output["conversion_efficiency"][-1], 1.0e-30)
            )
            variations[label] = float(
                np.std(output["conversion_efficiency"][-20:])
                / max(np.mean(output["conversion_efficiency"][-20:]), 1.0e-30)
            )

        self.assertGreater(efficiencies["constructive"], 0.57)
        self.assertLess(efficiencies["constructive"], 0.67)
        self.assertLess(pump_ratios["constructive"], 0.03)
        self.assertGreater(backward_fractions["constructive"], 0.99)
        self.assertLess(variations["constructive"], 1.0e-4)
        self.assertLess(efficiencies["destructive"], 0.05)
        self.assertGreater(
            efficiencies["constructive"],
            20.0 * efficiencies["destructive"],
        )


if __name__ == "__main__":
    unittest.main()
