"""Physics checks for stochastic modal driving."""

import unittest

import numpy as np

from spectral import mode_numbers
from stochastic import GaussianModalSeed, GaussianModalWienerNoise


class GaussianModalWienerPhysicsBenchmarks(unittest.TestCase):
    def test_complex_wiener_variance_follows_time_and_modal_envelope(self):
        """Recover E[|dE_mu|^2]=sigma^2*w_mu^2*dt."""
        modes = mode_numbers(64)
        strength = 0.03
        width = 4.0
        step_size = 0.04
        process = GaussianModalWienerNoise(
            strength=strength,
            mode_width=width,
            mode_half_width=8,
        )
        increments = process.modal_increments(
            np.random.default_rng(12),
            modes,
            step_size,
            number_of_fields=20_000,
        )

        for mode in (-8, -4, -1, 1, 4, 8):
            index = int(np.flatnonzero(modes == mode)[0])
            measured = np.mean(np.abs(increments[:, index]) ** 2)
            expected = (
                strength**2
                * step_size
                * np.exp(-np.square(mode / width))
            )
            self.assertAlmostEqual(measured / expected, 1.0, delta=0.025)

        pump_index = int(np.flatnonzero(modes == 0)[0])
        outside_index = int(np.flatnonzero(modes == 9)[0])
        np.testing.assert_array_equal(increments[:, pump_index], 0.0)
        np.testing.assert_array_equal(increments[:, outside_index], 0.0)

    def test_normalized_modal_noise_is_independent_of_spectral_grid(self):
        """Represent the same resolved stochastic modes on finer grids."""
        process = GaussianModalWienerNoise(
            strength=1.0e-4,
            mode_width=3.0,
            mode_half_width=19,
        )
        coarse_modes = mode_numbers(64)
        fine_modes = mode_numbers(128)
        coarse_fields = process.field_increments(
            np.random.default_rng(7),
            coarse_modes,
            0.005,
            number_of_fields=2,
        )
        fine_fields = process.field_increments(
            np.random.default_rng(7),
            fine_modes,
            0.005,
            number_of_fields=2,
        )
        coarse_modal = np.fft.fft(coarse_fields, axis=-1) / coarse_modes.size
        fine_modal = np.fft.fft(fine_fields, axis=-1) / fine_modes.size

        for mode in range(-19, 20):
            coarse_index = int(np.flatnonzero(coarse_modes == mode)[0])
            fine_index = int(np.flatnonzero(fine_modes == mode)[0])
            np.testing.assert_allclose(
                coarse_modal[:, coarse_index],
                fine_modal[:, fine_index],
                rtol=0.0,
                atol=3.0e-20,
            )

    def test_one_time_seed_has_grid_independent_modal_rms(self):
        """Avoid changing a branch-access perturbation under refinement."""
        seed = GaussianModalSeed(
            rms_amplitude=0.02,
            mode_width=19.0,
            mode_half_width=19,
        )
        coarse_modes = mode_numbers(64)
        fine_modes = mode_numbers(128)
        coarse_fields = seed.field_samples(
            np.random.default_rng(19), coarse_modes, number_of_fields=2
        )
        fine_fields = seed.field_samples(
            np.random.default_rng(19), fine_modes, number_of_fields=2
        )
        coarse_modal = np.fft.fft(coarse_fields, axis=-1) / coarse_modes.size
        fine_modal = np.fft.fft(fine_fields, axis=-1) / fine_modes.size

        for mode in range(-19, 20):
            coarse_index = int(np.flatnonzero(coarse_modes == mode)[0])
            fine_index = int(np.flatnonzero(fine_modes == mode)[0])
            np.testing.assert_allclose(
                coarse_modal[:, coarse_index],
                fine_modal[:, fine_index],
                rtol=0.0,
                atol=2.0e-17,
            )


if __name__ == "__main__":
    unittest.main()
