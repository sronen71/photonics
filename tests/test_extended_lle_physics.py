"""Physics benchmarks for generalized-dispersion and dimensional LLE output.

The checks here target physical predictions introduced by the generalized
dispersion, SI normalization, and time-averaged spectrum paths.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dispersion import DispersionRelation
from drift import estimate_drift
from lle_solver import single_soliton_seed, solve_lle, stationary_residual
from physics import HBAR_J_S, load_solver_physics
from spectrum import output_spectrum
from steady_solver import solve_steady_state


def polynomial_dispersion(beta_2, beta_3=0.0):
    """Return D(k)=beta_2*k^2/2+beta_3*k^3/6."""
    return DispersionRelation(
        kind="polynomial",
        description="test polynomial dispersion",
        evaluator=lambda mode_number: (
            0.5 * beta_2 * mode_number**2
            + beta_3 * mode_number**3 / 6.0
        ),
        seed_beta=beta_2,
    )


class ExtendedLLEPhysicsBenchmarks(unittest.TestCase):
    def test_third_order_dispersion_stabilizes_the_published_comb(self):
        """Reproduce the TOD stabilization in Parra-Rivas et al. Fig. 1."""
        # Parra-Rivas et al., Opt. Lett. 39, 2971 (2014), Eq. (1) and
        # Fig. 1 use
        #
        #   du/dt=-(1+i*theta)u+i|u|^2u+u0+i*d_tau^2 u+d3*d_tau^3 u
        #
        # at theta=6.1 and u0=4. Their d3=0 soliton breathes, whereas
        # d3=0.15 has a steady comb spectrum while translating at constant
        # velocity. On a 2*pi ring with tau=theta_ring/q, q=2*pi/L, the
        # code's modal dispersion is D(k)=-(q*k)^2-d3*(q*k)^3.
        alpha = 6.1
        forcing = 4.0
        fast_time_window = 32.0
        q = 2.0 * np.pi / fast_time_window
        beta_2 = -2.0 * q**2
        spatial_points = 256
        theta = (
            2.0 * np.pi * np.arange(spatial_points) / spatial_points
        )

        sideband_power_variation = {}
        fixed_frame_change = {}
        fixed_frame_residual = {}
        drift_diagnostics = {}
        final_fields = {}
        dispersions = {}
        for d_3 in (0.0, 0.15):
            beta_3 = -6.0 * d_3 * q**3
            dispersion = polynomial_dispersion(beta_2, beta_3)
            initial_field = single_soliton_seed(
                theta, alpha, forcing, dispersion
            )
            _, times, fields = solve_lle(
                alpha=alpha,
                forcing=forcing,
                beta=dispersion,
                spatial_points=spatial_points,
                final_time=50.0,
                time_step=0.005,
                initial_noise=0.0,
                snapshots=501,
                seed=0,
                initial_background=initial_field,
            )

            late_fields = fields[times >= 40.0]
            modal_power = np.abs(
                np.fft.fft(late_fields, axis=1) / spatial_points
            ) ** 2
            total_sideband_power = np.sum(modal_power[:, 1:], axis=1)
            sideband_power_variation[d_3] = (
                np.std(total_sideband_power)
                / np.mean(total_sideband_power)
            )
            fixed_frame_change[d_3] = (
                np.linalg.norm(fields[-1] - fields[-2])
                / np.linalg.norm(fields[-1])
            )
            fixed_frame_residual[d_3] = np.max(
                np.abs(
                    stationary_residual(
                        fields[-1], alpha, forcing, dispersion
                    )
                )
            )
            drift_diagnostics[d_3] = estimate_drift(
                times[times >= 40.0], late_fields
            )
            final_fields[d_3] = fields[-1]
            dispersions[d_3] = dispersion

        # The second-order-only state has a strongly breathing spectrum.
        self.assertGreater(sideband_power_variation[0.0], 0.1)

        # TOD suppresses that oscillation by more than an order of magnitude,
        # reproducing the steady spectrum in the published moving frame.
        self.assertLess(sideband_power_variation[0.15], 5.0e-3)
        self.assertLess(
            sideband_power_variation[0.15],
            sideband_power_variation[0.0] / 20.0,
        )

        # Steady modal powers do not imply a stationary field. The paper
        # explicitly reports constant-velocity drift when TOD is present.
        self.assertGreater(fixed_frame_change[0.15], 0.1)
        self.assertGreater(fixed_frame_residual[0.15], 1.0)

        # Aligning the TOD snapshots removes their evolution: the pulse center
        # follows a straight line and the profile is steady in that frame.
        tod_drift = drift_diagnostics[0.15]
        self.assertTrue(tod_drift.is_rigid_translation)
        self.assertGreater(abs(tod_drift.velocity), 0.1)
        self.assertLess(tod_drift.fit_rms, 5.0e-4)
        self.assertLess(tod_drift.shape_variation, 5.0e-3)
        self.assertFalse(drift_diagnostics[0.0].is_rigid_translation)

        # The steady solver must automatically augment the equation for odd
        # dispersion and refine the time-integrated profile together with v.
        moving_solution, velocity, residual, _ = solve_steady_state(
            final_fields[0.15],
            alpha,
            forcing,
            dispersions[0.15],
            tolerance=1.0e-9,
            max_iterations=100,
            initial_velocity=tod_drift.velocity,
        )
        self.assertLess(residual, 1.0e-9)
        self.assertLess(abs(velocity / tod_drift.velocity - 1.0), 2.0e-3)
        self.assertGreater(np.max(np.abs(moving_solution)), 1.0)

    def test_si_normalization_and_through_port_input_output_relation(self):
        """Check dimensional LLE scaling and critical-coupling extinction."""
        # The photon-amplitude normalization follows Chembo and Menyuk,
        # Phys. Rev. A 87, 053852 (2013):
        # alpha=2*(omega_0-omega_p)/kappa,
        # F^2=8*g0*kappa_ex*P/(kappa^3*hbar*omega_p), and
        # D(k)=-2*D_int(k)/kappa. The same coupled-mode convention gives
        # s_out=s_in-sqrt(kappa_ex)*a at the through port.
        kappa = 2.0e9
        kappa_external = 0.5 * kappa
        omega_pump = 1.2e15
        fsr_hz = 1.0e11
        g_0 = 5.0
        target_alpha = 4.0
        target_forcing = 2.0
        target_beta_2 = -0.02
        target_beta_3 = -0.003
        omega_0 = omega_pump + 0.5 * target_alpha * kappa
        pump_power = (
            target_forcing**2
            * kappa**3
            * HBAR_J_S
            * omega_pump
            / (8.0 * g_0 * kappa_external)
        )
        d_2 = -0.5 * kappa * target_beta_2
        d_3 = -0.5 * kappa * target_beta_3

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "dispersion.csv").write_text(
                "order,d\n"
                f"2,{d_2:.17g}\n"
                f"3,{d_3:.17g}\n",
                encoding="utf-8",
            )
            config_path = directory / "physical.yaml"
            config_path.write_text(
                "physics:\n"
                "  units: SI\n"
                f"  kappa_rad_s: {kappa:.17g}\n"
                "  kappa_external_rad_s: "
                f"{kappa_external:.17g}\n"
                f"  omega_0_rad_s: {omega_0:.17g}\n"
                f"  omega_pump_rad_s: {omega_pump:.17g}\n"
                f"  fsr_hz: {fsr_hz:.17g}\n"
                f"  g_0_rad_s: {g_0:.17g}\n"
                f"  pump_power_w: {pump_power:.17g}\n"
                "  dispersion_csv: dispersion.csv\n",
                encoding="utf-8",
            )
            physics = load_solver_physics(config_path)

        self.assertAlmostEqual(physics.alpha, target_alpha, places=14)
        self.assertAlmostEqual(
            physics.forcing.real, target_forcing, places=14
        )
        self.assertAlmostEqual(
            physics.physical_time_per_normalized_unit_s,
            2.0 / kappa,
            places=22,
        )
        modes = np.array([-7.0, -2.0, 0.0, 3.0, 8.0])
        expected_dispersion = (
            0.5 * target_beta_2 * modes**2
            + target_beta_3 * modes**3 / 6.0
        )
        np.testing.assert_allclose(
            physics.dispersion.values(modes),
            expected_dispersion,
            rtol=2.0e-15,
            atol=1.0e-15,
        )

        # At alpha=F^2 the Kerr shift places the uniform field A=F on
        # effective resonance. With kappa_ex=kappa/2, the incident pump and
        # cavity leakage cancel at the through port. Add one sideband to also
        # check its photon-flux-to-power conversion and optical frequency.
        spatial_points = 32
        sideband_mode = 3
        sideband_amplitude = 0.2 + 0.1j
        theta = (
            2.0 * np.pi * np.arange(spatial_points) / spatial_points
        )
        field = target_forcing + sideband_amplitude * np.exp(
            1j * sideband_mode * theta
        )
        saved = output_spectrum(field, physics=physics)["saved"]
        pump_index = spatial_points // 2
        sideband_index = pump_index + sideband_mode

        self.assertLess(
            saved["output_power_w"][pump_index] / pump_power, 1.0e-24
        )
        sideband_omega = (
            omega_pump + 2.0 * np.pi * fsr_hz * sideband_mode
        )
        expected_sideband_power = (
            HBAR_J_S
            * sideband_omega
            * kappa_external
            * kappa
            / (2.0 * g_0)
            * abs(sideband_amplitude) ** 2
        )
        self.assertAlmostEqual(
            saved["output_power_w"][sideband_index]
            / expected_sideband_power,
            1.0,
            places=13,
        )
        self.assertAlmostEqual(
            saved["output_frequency_thz"][sideband_index],
            sideband_omega / (2.0 * np.pi * 1.0e12),
            places=13,
        )

        # A relative equilibrium with normalized angular velocity v has
        # f_rep=FSR+kappa*v/(4*pi). Modal powers are unchanged, but the
        # physical comb-frequency axis must use this repetition rate.
        drift_velocity = 0.4
        drifting_saved = output_spectrum(
            field,
            physics=physics,
            drift_velocity=drift_velocity,
        )["saved"]
        expected_shift_hz = kappa * drift_velocity / (4.0 * np.pi)
        self.assertAlmostEqual(
            drifting_saved["repetition_rate_shift_hz"],
            expected_shift_hz,
            places=7,
        )
        self.assertAlmostEqual(
            drifting_saved["effective_repetition_rate_hz"],
            fsr_hz + expected_shift_hz,
            places=7,
        )
        expected_drifting_sideband_hz = (
            omega_pump / (2.0 * np.pi)
            + sideband_mode * (fsr_hz + expected_shift_hz)
        )
        self.assertAlmostEqual(
            drifting_saved["output_frequency_thz"][sideband_index],
            expected_drifting_sideband_hz / 1.0e12,
            places=13,
        )

    def test_time_averaged_spectrum_of_exact_cavity_ringdown(self):
        """Reproduce the analytic mean power of a freely decaying mode."""
        # In normalized LLE time, any linear unforced modal amplitude loses
        # energy as |A_k(t)|^2=|A_k(0)|^2*exp(-2*t), independently of its
        # dispersive phase. Its exact average over [0,T] follows directly.
        spatial_points = 32
        mode = 4
        initial_amplitude = 0.3
        final_time = 2.0
        times = np.linspace(0.0, final_time, 1001)
        theta = (
            2.0 * np.pi * np.arange(spatial_points) / spatial_points
        )
        fields = (
            initial_amplitude
            * np.exp(-times[:, None])
            * np.exp(1j * mode * theta[None, :])
        )
        saved = output_spectrum(fields, times=times)["saved"]
        measured_average = saved["output_normalized_power"][
            spatial_points // 2 + mode
        ]
        expected_average = (
            initial_amplitude**2
            * (1.0 - np.exp(-2.0 * final_time))
            / (2.0 * final_time)
        )
        self.assertLess(
            abs(measured_average / expected_average - 1.0), 2.0e-6
        )


if __name__ == "__main__":
    unittest.main()
