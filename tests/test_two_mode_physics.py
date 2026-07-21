"""Physics checks for the Liu et al. comb-plus-auxiliary-mode equations."""

import unittest

import numpy as np
from scipy.linalg import expm

from physics import HBAR_J_S
from spectral import mode_numbers
from lle_solver import solve_lle
from two_mode import (
    TwoModeParameters,
    apply_linear_step,
    linear_step_parameters,
    nonlinear_step,
    split_step,
    solve_two_mode_lle,
    two_mode_residual,
)


class LiuTwoModePhysicsBenchmarks(unittest.TestCase):
    def test_uncoupled_comb_reduces_to_existing_scalar_lle(self):
        """Recover the scalar time solver when the auxiliary mode is absent."""
        spatial_points = 32
        theta = 2.0 * np.pi * np.arange(spatial_points) / spatial_points
        initial_comb = (
            0.12 + 0.04j
            + (0.03 - 0.01j) * np.exp(3j * theta)
        )
        alpha = 0.8
        forcing = 0.35
        beta = -0.09
        parameters = TwoModeParameters(
            alpha_comb=alpha,
            alpha_auxiliary=-0.2,
            forcing_comb=forcing,
            forcing_auxiliary=0.0,
            beta=beta,
            auxiliary_loss_ratio=2.0,
            cavity_coupling=0.0,
            bus_coupling=0.0,
        )
        _, two_mode_times, comb_fields, auxiliary_fields = solve_two_mode_lle(
            parameters,
            spatial_points,
            final_time=0.6,
            time_step=0.02,
            snapshots=5,
            initial_comb=initial_comb,
        )
        _, scalar_times, scalar_fields = solve_lle(
            alpha=alpha,
            forcing=forcing,
            beta=beta,
            spatial_points=spatial_points,
            final_time=0.6,
            time_step=0.02,
            initial_noise=0.0,
            snapshots=5,
            seed=0,
            initial_background=initial_comb,
        )
        np.testing.assert_array_equal(two_mode_times, scalar_times)
        np.testing.assert_allclose(
            comb_fields, scalar_fields, rtol=2.0e-14, atol=2.0e-15
        )
        np.testing.assert_array_equal(auxiliary_fields, 0.0)

    def test_appendix_a_dimensional_normalization(self):
        """Recover every coefficient after scaling time by kappa_1/2."""
        kappa_1 = 2.0e9
        kappa_ex1 = 0.6e9
        kappa_2 = 5.0e9
        kappa_ex2 = 1.5e9
        detuning_1 = 1.2e9
        detuning_2 = -0.4e9
        g = 7.0
        pump_power = 0.025
        omega = 1.21e15
        d_2 = 3.0e7
        coupling = (0.8 + 0.3j) * 1.0e9
        parameters = TwoModeParameters.from_physical(
            kappa_comb_rad_s=kappa_1,
            kappa_external_comb_rad_s=kappa_ex1,
            kappa_auxiliary_rad_s=kappa_2,
            kappa_external_auxiliary_rad_s=kappa_ex2,
            detuning_comb_rad_s=detuning_1,
            detuning_auxiliary_rad_s=detuning_2,
            nonlinear_shift_rad_s=g,
            pump_power_w=pump_power,
            photon_angular_frequency_rad_s=omega,
            integrated_dispersion=d_2,
            cavity_coupling_rad_s=coupling,
            intermodal_ratio=0.7,
        )

        self.assertAlmostEqual(parameters.alpha_comb, 2.0 * detuning_1 / kappa_1)
        self.assertAlmostEqual(
            parameters.alpha_auxiliary, 2.0 * detuning_2 / kappa_1
        )
        self.assertAlmostEqual(parameters.auxiliary_loss_ratio, kappa_2 / kappa_1)
        self.assertAlmostEqual(
            parameters.bus_coupling,
            np.sqrt(kappa_ex1 * kappa_ex2) / kappa_1,
        )
        self.assertAlmostEqual(
            parameters.cavity_coupling.real, coupling.real / kappa_1
        )
        self.assertAlmostEqual(
            parameters.cavity_coupling.imag, coupling.imag / kappa_1
        )
        expected_bus_coupling = np.sqrt(kappa_ex1 * kappa_ex2) / kappa_1
        np.testing.assert_allclose(
            parameters.auxiliary_to_comb_coupling,
            1j * coupling / kappa_1 - expected_bus_coupling,
        )
        np.testing.assert_allclose(
            parameters.comb_to_auxiliary_coupling,
            1j * np.conj(coupling) / kappa_1 - expected_bus_coupling,
        )
        expected_forcing_1 = np.sqrt(
            8.0 * g * kappa_ex1 * pump_power
            / (kappa_1**3 * HBAR_J_S * omega)
        )
        expected_forcing_2 = np.sqrt(
            8.0 * g * kappa_ex2 * pump_power
            / (kappa_1**3 * HBAR_J_S * omega)
        )
        self.assertAlmostEqual(parameters.forcing_comb, expected_forcing_1)
        self.assertAlmostEqual(parameters.forcing_auxiliary, expected_forcing_2)
        modes = np.array([-3.0, 0.0, 4.0])
        expected_dispersion = -2.0 / kappa_1 * 0.5 * d_2 * modes**2
        np.testing.assert_allclose(
            parameters.beta.values(modes), expected_dispersion
        )

    def test_auxiliary_xpm_is_sum_of_modal_comb_powers(self):
        """Use Parseval to reproduce sum_mu |A_mu|^2 in Appendix A."""
        spatial_points = 32
        theta = 2.0 * np.pi * np.arange(spatial_points) / spatial_points
        pump = 0.3 + 0.1j
        sideband = 0.2 - 0.05j
        uniform_comb = np.full(spatial_points, pump)
        comb_with_sideband = pump + sideband * np.exp(3j * theta)
        auxiliary = -0.15 + 0.2j
        parameters = TwoModeParameters(
            alpha_comb=0.0,
            alpha_auxiliary=0.0,
            forcing_comb=0.0,
            forcing_auxiliary=0.0,
            beta=0.0,
            auxiliary_loss_ratio=1.0,
            cavity_coupling=0.0,
            bus_coupling=0.0,
            intermodal_ratio=0.6,
        )

        uniform_b = two_mode_residual(
            uniform_comb, auxiliary, parameters
        )[1]
        sideband_b = two_mode_residual(
            comb_with_sideband, auxiliary, parameters
        )[1]
        expected_change = (
            1j
            * parameters.intermodal_ratio**2
            * abs(sideband) ** 2
            * auxiliary
        )
        np.testing.assert_allclose(
            sideband_b - uniform_b,
            expected_change,
            rtol=2.0e-14,
            atol=2.0e-16,
        )
        self.assertAlmostEqual(np.mean(comb_with_sideband), pump)

    def test_linear_pump_pair_matches_the_exact_matrix_exponential(self):
        """Check both Kc and shared-bus coupling with modal FFT scaling."""
        spatial_points = 16
        theta = 2.0 * np.pi * np.arange(spatial_points) / spatial_points
        pump = 0.25 - 0.1j
        sideband = -0.07 + 0.04j
        auxiliary = 0.11 + 0.08j
        comb = pump + sideband * np.exp(2j * theta)
        parameters = TwoModeParameters(
            alpha_comb=0.7,
            alpha_auxiliary=-0.3,
            forcing_comb=0.2,
            forcing_auxiliary=0.35,
            beta=-0.08,
            auxiliary_loss_ratio=1.8,
            cavity_coupling=0.4 * np.exp(0.3j),
            bus_coupling=0.17,
            intermodal_ratio=0.9,
        )
        duration = 0.23
        modes = mode_numbers(spatial_points)
        step = linear_step_parameters(parameters, modes, duration)
        measured_comb, measured_auxiliary = apply_linear_step(
            comb, auxiliary, step
        )

        generator = np.array([
            [
                -(1.0 + 1j * parameters.alpha_comb),
                parameters.auxiliary_to_comb_coupling,
                parameters.forcing_comb,
            ],
            [
                parameters.comb_to_auxiliary_coupling,
                -(
                    parameters.auxiliary_loss_ratio
                    + 1j * parameters.alpha_auxiliary
                ),
                parameters.forcing_auxiliary,
            ],
            [0.0, 0.0, 0.0],
        ], dtype=complex)
        expected_pump_pair = expm(duration * generator) @ np.array(
            [pump, auxiliary, 1.0], dtype=complex
        )
        expected_sideband = sideband * np.exp(
            duration
            * (
                -(1.0 + 1j * parameters.alpha_comb)
                + 0.5j * parameters.beta * 2**2
            )
        )
        expected_comb = (
            expected_pump_pair[0]
            + expected_sideband * np.exp(2j * theta)
        )
        np.testing.assert_allclose(
            measured_comb, expected_comb, rtol=2.0e-15, atol=2.0e-15
        )
        np.testing.assert_allclose(
            measured_auxiliary,
            expected_pump_pair[1],
            rtol=2.0e-15,
            atol=2.0e-15,
        )

    def test_nonlinear_subflow_conserves_each_mode_population(self):
        """Reproduce the self- and cross-phase rotations exactly."""
        comb = np.array([0.2 + 0.1j, -0.1 + 0.4j, 0.3 - 0.2j])
        auxiliary = -0.15 + 0.25j
        duration = 0.31
        ratio = 0.8
        measured_comb, measured_auxiliary = nonlinear_step(
            comb, auxiliary, duration, ratio
        )
        comb_power = np.mean(np.abs(comb) ** 2)
        expected_comb = comb * np.exp(
            1j * duration * (np.abs(comb) ** 2 + ratio**2 * abs(auxiliary) ** 2)
        )
        expected_auxiliary = auxiliary * np.exp(
            1j * duration * (abs(auxiliary) ** 2 + ratio**2 * comb_power)
        )
        np.testing.assert_allclose(measured_comb, expected_comb)
        np.testing.assert_allclose(measured_auxiliary, expected_auxiliary)
        np.testing.assert_allclose(abs(measured_comb), abs(comb))
        self.assertAlmostEqual(abs(measured_auxiliary), abs(auxiliary))

    def test_split_step_generator_matches_full_two_mode_equations(self):
        """Recover the Appendix A vector field in the small-step limit."""
        spatial_points = 16
        theta = 2.0 * np.pi * np.arange(spatial_points) / spatial_points
        comb = 0.2 + 0.1j + (0.08 - 0.03j) * np.exp(2j * theta)
        auxiliary = -0.1 + 0.07j
        parameters = TwoModeParameters(
            alpha_comb=1.1,
            alpha_auxiliary=0.4,
            forcing_comb=0.6,
            forcing_auxiliary=0.9,
            beta=-0.06,
            auxiliary_loss_ratio=2.2,
            cavity_coupling=0.3 - 0.2j,
            bus_coupling=0.13,
            intermodal_ratio=0.75,
        )
        step_size = 1.0e-5
        modes = mode_numbers(spatial_points)
        half_step = linear_step_parameters(
            parameters, modes, 0.5 * step_size
        )
        next_comb, next_auxiliary = split_step(
            comb,
            auxiliary,
            step_size,
            half_step,
            parameters.intermodal_ratio,
        )
        expected_comb_rate, expected_auxiliary_rate = two_mode_residual(
            comb, auxiliary, parameters
        )
        measured_comb_rate = (next_comb - comb) / step_size
        measured_auxiliary_rate = (next_auxiliary - auxiliary) / step_size
        np.testing.assert_allclose(
            measured_comb_rate,
            expected_comb_rate,
            rtol=2.0e-5,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            measured_auxiliary_rate,
            expected_auxiliary_rate,
            rtol=2.0e-5,
            atol=2.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
