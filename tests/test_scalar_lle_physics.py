"""Physics benchmarks for the normalized scalar Lugiato--Lefever equation.

These tests compare observables with analytic results used in the foundational
scalar-LLE literature.  They intentionally test physical predictions rather
than implementation details.
"""

import unittest

import numpy as np
from scipy.signal import resample

from lle_solver import solve_lle, uniform_states
from steady_solver import single_soliton_seed, solve_stationary
from uniform_solver import CRITICAL_ALPHA, fold_points, pump_power


def periodic_fwhm(values, refinement=64):
    """Return a spectrally interpolated FWHM on a 2*pi periodic grid."""
    values = np.asarray(values)
    spatial_points = values.size
    peak_index = int(np.argmax(values))
    centered = np.roll(values, spatial_points // 2 - peak_index)
    centered = resample(centered, refinement * spatial_points)
    centered = np.roll(
        centered, centered.size // 2 - int(np.argmax(centered))
    )
    coordinate = 2.0 * np.pi * np.arange(centered.size) / centered.size
    center = centered.size // 2
    half_maximum = 0.5 * centered[center]

    left = center
    while left > 0 and centered[left] >= half_maximum:
        left -= 1
    left_crossing = coordinate[left] + (
        (half_maximum - centered[left])
        * (coordinate[left + 1] - coordinate[left])
        / (centered[left + 1] - centered[left])
    )

    right = center
    while right < centered.size - 1 and centered[right] >= half_maximum:
        right += 1
    right_crossing = coordinate[right - 1] + (
        (half_maximum - centered[right - 1])
        * (coordinate[right] - coordinate[right - 1])
        / (centered[right] - centered[right - 1])
    )
    return right_crossing - left_crossing


class ScalarLLEPhysicsBenchmarks(unittest.TestCase):
    def test_homogeneous_bistability_cusp_and_folds(self):
        """Reproduce the dispersive optical-bistability cubic and cusp."""
        # Coen and Erkintalo, Opt. Lett. 38, 1790 (2013), Eq. (2), and
        # Godey et al., Phys. Rev. A 89, 063814 (2014), Eq. (3).
        cusp_intensity = 2.0 / np.sqrt(3.0)
        cusp_power = 8.0 / (3.0 * np.sqrt(3.0))
        self.assertAlmostEqual(CRITICAL_ALPHA, np.sqrt(3.0), places=14)
        self.assertAlmostEqual(
            pump_power(cusp_intensity, CRITICAL_ALPHA),
            cusp_power,
            places=14,
        )

        alpha = 2.0
        expected_folds = np.array([1.0, 5.0 / 3.0])
        np.testing.assert_allclose(fold_points(alpha), expected_folds)

        forcing = np.sqrt(1.9)
        states = uniform_states(alpha, forcing)
        self.assertEqual(len(states), 3)
        for state in states:
            intensity = abs(state) ** 2
            self.assertAlmostEqual(
                pump_power(intensity, alpha), abs(forcing) ** 2, places=12
            )

    def test_nonlinear_uniform_dynamics_relaxes_to_cw_solution(self):
        """Check the nonlinear dynamic solver against the exact CW state."""
        alpha = 0.0
        forcing = 2.0
        expected = uniform_states(alpha, forcing)[0]
        _, _, fields = solve_lle(
            alpha=alpha,
            forcing=forcing,
            beta=-0.2,
            spatial_points=16,
            final_time=20.0,
            time_step=0.01,
            initial_noise=0.0,
            snapshots=2,
            seed=0,
        )
        final_field = fields[-1]
        relative_error = np.linalg.norm(final_field - expected) / (
            np.sqrt(final_field.size) * abs(expected)
        )
        self.assertLess(relative_error, 5.0e-4)

    def test_linear_cavity_transient_and_dispersive_ringdown(self):
        """Reproduce two exact small-signal solutions of the normalized LLE."""
        alpha = 1.3
        forcing = 1.0e-7
        final_time = 1.7
        _, _, fields = solve_lle(
            alpha=alpha,
            forcing=forcing,
            beta=-0.7,
            spatial_points=32,
            final_time=final_time,
            time_step=0.05,
            initial_noise=0.0,
            snapshots=2,
            seed=0,
        )
        exact_driven_field = forcing * (
            1.0 - np.exp(-(1.0 + 1j * alpha) * final_time)
        ) / (1.0 + 1j * alpha)
        driven_relative_error = abs(fields[-1, 0] - exact_driven_field) / abs(
            exact_driven_field
        )
        self.assertLess(driven_relative_error, 1.0e-8)

        spatial_points = 32
        mode = 3
        amplitude = 1.0e-6
        theta = 2.0 * np.pi * np.arange(spatial_points) / spatial_points
        initial_mode = amplitude * np.exp(1j * mode * theta)
        final_time = 1.1
        for beta in (-0.3, 0.3):
            with self.subTest(beta=beta):
                _, _, fields = solve_lle(
                    alpha=alpha,
                    forcing=0.0,
                    beta=beta,
                    spatial_points=spatial_points,
                    final_time=final_time,
                    time_step=0.05,
                    initial_noise=0.0,
                    snapshots=2,
                    seed=0,
                    initial_background=initial_mode,
                )
                numerical_mode = (
                    np.fft.fft(fields[-1])[mode] / spatial_points
                )
                mode_generator = (
                    -(1.0 + 1j * alpha) + 0.5j * beta * mode**2
                )
                exact_mode = amplitude * np.exp(
                    mode_generator * final_time
                )
                relative_error = abs(numerical_mode - exact_mode) / abs(
                    exact_mode
                )
                self.assertLess(relative_error, 1e-8)

    def test_modulational_instability_gain_of_mode_eight(self):
        """Reproduce the analytic MI gain for the Godey et al. example."""
        # Godey et al., Phys. Rev. A 89, 063814 (2014), Eqs. (16)--(18)
        # and Fig. 3: alpha=1, beta=-0.04, CW intensity rho=1.2.  The
        # nearest integer to the maximum-gain mode is l=8.
        alpha = 1.0
        beta = -0.04
        intensity = 1.2
        mode = 8
        forcing = np.sqrt(
            intensity * (1.0 + (alpha - intensity) ** 2)
        )
        background = forcing / (1.0 + 1j * (alpha - intensity))
        phase_mismatch = (
            2.0 * intensity - alpha + 0.5 * beta * mode**2
        )
        expected_gain = -1.0 + np.sqrt(
            intensity**2 - phase_mismatch**2
        )

        stability_matrix = np.array(
            [
                [
                    -1.0 + 1j * phase_mismatch,
                    1j * background**2,
                ],
                [
                    -1j * np.conj(background) ** 2,
                    -1.0 - 1j * phase_mismatch,
                ],
            ]
        )
        eigenvalues, eigenvectors = np.linalg.eig(stability_matrix)
        unstable_index = int(np.argmax(eigenvalues.real))
        unstable_vector = eigenvectors[:, unstable_index]
        self.assertAlmostEqual(
            eigenvalues[unstable_index].real, expected_gain, places=12
        )

        spatial_points = 64
        theta = 2.0 * np.pi * np.arange(spatial_points) / spatial_points
        perturbation_size = 1.0e-7
        perturbation = perturbation_size * (
            unstable_vector[0] * np.exp(1j * mode * theta)
            + np.conj(unstable_vector[1]) * np.exp(-1j * mode * theta)
        )
        _, times, fields = solve_lle(
            alpha=alpha,
            forcing=forcing,
            beta=beta,
            spatial_points=spatial_points,
            final_time=8.0,
            time_step=0.01,
            initial_noise=0.0,
            snapshots=161,
            seed=0,
            initial_background=background + perturbation,
        )
        spectrum = np.fft.fft(fields, axis=1) / spatial_points
        sideband_amplitude = np.sqrt(
            abs(spectrum[:, mode]) ** 2 + abs(spectrum[:, -mode]) ** 2
        )
        measured_gain = np.polyfit(times, np.log(sideband_amplitude), 1)[0]
        self.assertLess(abs(measured_gain - expected_gain), 2.0e-3)

        # Starting from broadband noise, the same parameters should evolve
        # into the eight-roll Turing pattern reported in Fig. 3.
        pattern_points = 128
        _, _, pattern_fields = solve_lle(
            alpha=alpha,
            forcing=forcing,
            beta=beta,
            spatial_points=pattern_points,
            final_time=60.0,
            time_step=0.01,
            initial_noise=1.0e-3,
            snapshots=2,
            seed=7,
            initial_background=background,
        )
        pattern = pattern_fields[-1]
        pattern_spectrum = abs(np.fft.fft(pattern) / pattern_points)
        dominant_positive_mode = (
            int(np.argmax(pattern_spectrum[1 : pattern_points // 2])) + 1
        )
        self.assertEqual(dominant_positive_mode, mode)

        pattern_intensity = abs(pattern) ** 2
        local_maximum = (pattern_intensity > np.roll(pattern_intensity, 1)) & (
            pattern_intensity > np.roll(pattern_intensity, -1)
        )
        significant_peaks = local_maximum & (
            pattern_intensity > pattern_intensity.mean()
        )
        self.assertEqual(int(np.sum(significant_peaks)), mode)

    def test_normal_dispersion_dark_soliton_of_godey_et_al(self):
        """Reproduce the stable normal-GVD dark soliton in Godey Fig. 4."""
        # Godey et al., Phys. Rev. A 89, 063814 (2014), Figs. 4 and 5:
        # alpha=2.5, beta=+0.0125, and F^2=2.61 support a stable dark
        # cavity soliton between the upper and lower homogeneous states.
        alpha = 2.5
        beta = 0.0125
        forcing = np.sqrt(2.61)
        spatial_points = 256
        theta = 2.0 * np.pi * np.arange(spatial_points) / spatial_points
        distance = (theta + np.pi) % (2.0 * np.pi) - np.pi
        lower, middle, upper = uniform_states(alpha, forcing)

        # A smooth lower-state domain within the upper state supplies the
        # pulse-like perturbation used to select the dark-soliton basin.
        notch_width = 0.4
        edge_width = 0.06
        lower_state_window = 0.5 * (
            np.tanh((distance + 0.5 * notch_width) / edge_width)
            - np.tanh((distance - 0.5 * notch_width) / edge_width)
        )
        initial_field = upper + (lower - upper) * lower_state_window

        _, _, fields = solve_lle(
            alpha=alpha,
            forcing=forcing,
            beta=beta,
            spatial_points=spatial_points,
            final_time=50.0,
            time_step=0.01,
            initial_noise=0.0,
            snapshots=6,
            seed=0,
            initial_background=initial_field,
        )
        final_field = fields[-1]
        intensity = abs(final_field) ** 2
        lower_intensity = abs(lower) ** 2
        middle_intensity = abs(middle) ** 2
        upper_intensity = abs(upper) ** 2

        # The published solution is a single localized intensity hole whose
        # background approaches the upper CW state and whose minimum lies
        # between the lower and intermediate CW intensities.
        self.assertLess(
            abs(intensity.max() / upper_intensity - 1.0), 5.0e-4
        )
        self.assertGreater(intensity.min(), lower_intensity)
        self.assertLess(intensity.min(), middle_intensity)
        self.assertGreater(np.ptp(intensity), 1.5)

        local_minimum = (intensity < np.roll(intensity, 1)) & (
            intensity < np.roll(intensity, -1)
        )
        significant_minimum = local_minimum & (
            intensity < middle_intensity
        )
        self.assertEqual(int(np.sum(significant_minimum)), 1)

        # A single dark pulse has a one-FSR comb: the first sideband must be
        # present rather than the field having relaxed to a flat CW solution.
        spectrum = abs(np.fft.fft(final_field)) / spatial_points
        self.assertGreater(spectrum[1] / spectrum[0], 0.05)

        late_time_drift = np.linalg.norm(fields[-1] - fields[-2]) / (
            np.linalg.norm(final_field)
        )
        self.assertLess(late_time_drift, 1.0e-6)

    def test_bright_soliton_asymptotics_and_stationarity(self):
        """Check large-detuning sech asymptotics and LLE stationarity."""
        # Coen and Erkintalo, Opt. Lett. 38, 1790 (2013):
        # At large detuning, |A|^2_peak approaches 2*alpha and, after
        # translating their beta=-2 convention to this code, FWHM approaches
        # 2*arcosh(sqrt(2))*sqrt(|beta|/(2*alpha)).
        # These are asymptotic scaling laws, not exact finite-alpha solutions.
        alpha = 10.0
        forcing = np.sqrt(10.0)
        beta = -0.2
        coarse_points = 128
        coarse_theta = (
            2.0 * np.pi * np.arange(coarse_points) / coarse_points
        )
        coarse_guess = single_soliton_seed(
            coarse_theta, alpha, forcing, beta
        )
        coarse_solution, coarse_residual, _ = solve_stationary(
            coarse_guess,
            alpha,
            forcing,
            beta,
            tolerance=2.0e-8,
            max_iterations=500,
        )
        self.assertLess(coarse_residual, 2.0e-8)

        # Continue the converged solution onto a finer spectral grid. Direct
        # Newton iteration from the analytic asymptotic seed becomes poorly
        # conditioned at this resolution, while continuation follows the same
        # localized stationary branch robustly.
        spatial_points = 256
        theta = 2.0 * np.pi * np.arange(spatial_points) / spatial_points
        background = uniform_states(alpha, forcing)[0]
        initial_guess = resample(coarse_solution, spatial_points)
        solution, residual, _ = solve_stationary(
            initial_guess,
            alpha,
            forcing,
            beta,
            tolerance=2.0e-8,
            max_iterations=1500,
        )
        self.assertLess(residual, 2.0e-8)

        intensity = abs(solution) ** 2
        expected_peak = 2.0 * alpha
        coarse_peak = np.max(abs(coarse_solution) ** 2)
        self.assertLess(abs(intensity.max() / coarse_peak - 1.0), 2.0e-4)
        self.assertLess(abs(intensity.max() / expected_peak - 1.0), 0.08)

        # The sech law describes the localized pulse, so measure the FWHM of
        # the field after subtracting its CW background. Spectral interpolation
        # avoids a percent-level bias from the coarse-grid half-maximum crossing.
        pulse_intensity = abs(solution - background) ** 2
        measured_width = periodic_fwhm(pulse_intensity)
        coarse_width = periodic_fwhm(abs(coarse_solution - background) ** 2)
        self.assertLess(abs(measured_width / coarse_width - 1.0), 1.0e-3)
        expected_width = (
            2.0
            * np.arccosh(np.sqrt(2.0))
            * np.sqrt(abs(beta) / (2.0 * alpha))
        )
        self.assertLess(abs(measured_width / expected_width - 1.0), 0.04)

        # A stationary solution should remain fixed under the independent
        # time-domain integrator, up to its time-discretization error.
        drifts = []
        for time_step in (0.01, 0.005):
            _, _, fields = solve_lle(
                alpha=alpha,
                forcing=forcing,
                beta=beta,
                spatial_points=spatial_points,
                final_time=2.0,
                time_step=time_step,
                initial_noise=0.0,
                snapshots=2,
                seed=0,
                initial_background=solution,
            )
            drifts.append(
                np.linalg.norm(fields[-1] - solution)
                / np.linalg.norm(solution)
            )
        self.assertLess(drifts[1], 2.0e-3)
        self.assertGreater(drifts[0] / drifts[1], 3.5)


if __name__ == "__main__":
    unittest.main()
