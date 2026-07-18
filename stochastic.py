"""Reusable stochastic processes for modal ring-field simulations."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianModalWienerNoise:
    """Complex Gaussian Wiener increments with a modal envelope.

    ``strength`` is the RMS complex modal-amplitude increment per square root
    of normalized time at the center of the envelope.  The normalized modal
    amplitudes use ``fft(field) / field.size``.  Independent increments are
    sampled for every requested field and active mode.
    """

    strength: float
    mode_width: float
    mode_half_width: int
    include_pump: bool = False

    def __post_init__(self):
        if not np.isfinite(self.strength) or self.strength < 0.0:
            raise ValueError("noise strength must be finite and nonnegative")
        if not np.isfinite(self.mode_width) or self.mode_width <= 0.0:
            raise ValueError("noise mode width must be finite and positive")
        if (
            not isinstance(self.mode_half_width, (int, np.integer))
            or self.mode_half_width < 0
        ):
            raise ValueError("noise mode half-width must be nonnegative")

    def _active_modes_and_indices(self, modes):
        modes = np.asarray(modes)
        if modes.ndim != 1:
            raise ValueError("mode numbers must be one-dimensional")
        active_modes = np.arange(
            -self.mode_half_width,
            self.mode_half_width + 1,
            dtype=int,
        )
        if not self.include_pump:
            active_modes = active_modes[active_modes != 0]
        indices = []
        for mode in active_modes:
            matches = np.flatnonzero(modes == mode)
            if matches.size != 1:
                raise ValueError(
                    "spectral grid does not resolve every requested noise mode"
                )
            indices.append(int(matches[0]))
        return active_modes, np.asarray(indices, dtype=int)

    def modal_increments(self, rng, modes, step_size, number_of_fields=1):
        """Sample normalized modal increments for one integration step."""
        if not np.isfinite(step_size) or step_size < 0.0:
            raise ValueError("noise step size must be finite and nonnegative")
        if (
            not isinstance(number_of_fields, (int, np.integer))
            or number_of_fields <= 0
        ):
            raise ValueError("number_of_fields must be a positive integer")

        modes = np.asarray(modes)
        active_modes, active_indices = self._active_modes_and_indices(modes)
        increments = np.zeros(
            (int(number_of_fields), modes.size), dtype=complex
        )
        if not active_modes.size or self.strength == 0.0 or step_size == 0.0:
            return increments

        weights = np.exp(
            -0.5 * np.square(active_modes / self.mode_width)
        )
        scale = self.strength * np.sqrt(0.5 * step_size)
        samples = scale * weights[None, :] * (
            rng.standard_normal((number_of_fields, active_modes.size))
            + 1j
            * rng.standard_normal((number_of_fields, active_modes.size))
        )
        increments[:, active_indices] = samples
        return increments

    def field_increments(self, rng, modes, step_size, number_of_fields=1):
        """Sample physical-grid increments for one integration step."""
        modal = self.modal_increments(
            rng,
            modes,
            step_size,
            number_of_fields=number_of_fields,
        )
        return np.fft.ifft(modes.size * modal, axis=-1)


@dataclass(frozen=True)
class GaussianModalSeed:
    """One-time complex Gaussian seed in normalized modal amplitudes."""

    rms_amplitude: float
    mode_width: float
    mode_half_width: int
    include_pump: bool = False

    def __post_init__(self):
        self._unit_time_process()

    def _unit_time_process(self):
        return GaussianModalWienerNoise(
            strength=self.rms_amplitude,
            mode_width=self.mode_width,
            mode_half_width=self.mode_half_width,
            include_pump=self.include_pump,
        )

    def modal_samples(self, rng, modes, number_of_fields=1):
        """Sample normalized modal amplitudes with the configured RMS."""
        return self._unit_time_process().modal_increments(
            rng,
            modes,
            1.0,
            number_of_fields=number_of_fields,
        )

    def field_samples(self, rng, modes, number_of_fields=1):
        """Sample physical-grid fields with the configured modal RMS."""
        return self._unit_time_process().field_increments(
            rng,
            modes,
            1.0,
            number_of_fields=number_of_fields,
        )
