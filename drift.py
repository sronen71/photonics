"""Translation diagnostics for periodic Lugiato--Lefever fields."""

from dataclasses import dataclass

import numpy as np

from spectral import mode_numbers


@dataclass(frozen=True)
class DriftDiagnostics:
    """Describe whether sampled fields form a rigid relative equilibrium."""

    velocity: float
    fit_rms: float
    shape_variation: float
    total_shift: float
    tracking_harmonic: int
    modulation_fraction: float
    is_rigid_translation: bool

    def saved_results(self):
        """Return scalar values suitable for ``numpy.savez``."""
        return {
            "drift_velocity_normalized": self.velocity,
            "drift_fit_rms_rad": self.fit_rms,
            "drift_aligned_shape_variation": self.shape_variation,
            "drift_total_shift_rad": self.total_shift,
            "drift_tracking_harmonic": self.tracking_harmonic,
            "drift_modulation_fraction": self.modulation_fraction,
            "drift_is_rigid_translation": self.is_rigid_translation,
        }


def spectral_derivative(field):
    """Differentiate a complex periodic field with respect to ring angle."""
    field = np.asarray(field)
    mode_number = mode_numbers(field.shape[-1])
    return np.fft.ifft(
        1j * mode_number * np.fft.fft(field, axis=-1), axis=-1
    )


def translation_gauge(reference_fields):
    """Return a normalized tangent for fixing a periodic translation.

    ``reference_fields`` may contain one field or several fields sharing the
    same ring coordinate.  The returned direction is the tangent to their
    common translation orbit.  A phase condition formed with this direction
    removes the neutral shift mode without choosing a preferred physical
    position on the ring.
    """
    reference = np.asarray(reference_fields, dtype=complex)
    if reference.ndim < 1 or reference.shape[-1] < 2:
        raise ValueError("a translation gauge requires periodic field samples")
    tangent = spectral_derivative(reference)
    tangent_rms = float(np.sqrt(np.mean(np.abs(tangent) ** 2)))
    field_rms = float(np.sqrt(np.mean(np.abs(reference) ** 2)))
    uniform_threshold = (
        np.sqrt(np.finfo(float).eps) * max(1.0, field_rms)
    )
    if tangent_rms <= uniform_threshold:
        raise ValueError(
            "a translation gauge requires a nonuniform reference field"
        )
    return reference.copy(), tangent / tangent_rms


def translation_phase_condition(fields, reference, direction):
    """Return the real orthogonality condition that fixes translation."""
    fields = np.asarray(fields, dtype=complex)
    reference = np.asarray(reference, dtype=complex)
    direction = np.asarray(direction, dtype=complex)
    if fields.shape != reference.shape or direction.shape != reference.shape:
        raise ValueError("translation-gauge fields must have matching shapes")
    return float(
        np.real(np.vdot(direction, fields - reference)) / reference.size
    )


def translate_fields(fields, shift):
    """Evaluate periodic fields after the coordinate change theta -> theta+shift."""
    fields = np.asarray(fields)
    shift = np.asarray(shift, dtype=float)
    if fields.ndim == 1:
        fields = fields[None, :]
        shift = np.atleast_1d(shift)
        squeeze = True
    else:
        squeeze = False
    if fields.ndim != 2 or shift.shape != (fields.shape[0],):
        raise ValueError("fields and shifts must have compatible time axes")
    mode_number = mode_numbers(fields.shape[1])
    translated = np.fft.ifft(
        np.fft.fft(fields, axis=1)
        * np.exp(1j * shift[:, None] * mode_number[None, :]),
        axis=1,
    )
    return translated[0] if squeeze else translated


def estimate_drift(times, fields, maximum_tracking_harmonic=16):
    """Fit rigid translation of a localized or patterned periodic state.

    A translating intensity profile obeys

        I(theta, t) = I_0(theta - X(t)).

    Its spatial moment of order m therefore has phase m*X(t).  The strongest
    low-order nonzero moment is used to track X, which also works for equally
    spaced multi-pulse states whose first few moments vanish.
    """
    times = np.asarray(times, dtype=float)
    fields = np.asarray(fields)
    if (
        fields.ndim != 2
        or fields.shape[0] < 3
        or fields.shape[1] < 4
        or times.shape != (fields.shape[0],)
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError(
            "drift estimation needs at least three ordered periodic fields"
        )

    spatial_points = fields.shape[1]
    theta = 2.0 * np.pi * np.arange(spatial_points) / spatial_points
    intensity = np.abs(fields) ** 2
    maximum_harmonic = min(
        int(maximum_tracking_harmonic), spatial_points // 2 - 1
    )
    harmonics = np.arange(1, maximum_harmonic + 1)
    moments = np.column_stack([
        np.sum(
            intensity * np.exp(1j * harmonic * theta)[None, :], axis=1
        )
        for harmonic in harmonics
    ])
    moment_strength = np.mean(np.abs(moments), axis=0)
    selected = int(np.argmax(moment_strength))
    harmonic = int(harmonics[selected])
    total_intensity = np.mean(np.sum(intensity, axis=1))
    modulation_fraction = float(
        moment_strength[selected] / max(total_intensity, np.finfo(float).tiny)
    )

    position = np.unwrap(np.angle(moments[:, selected])) / harmonic
    velocity, intercept = np.polyfit(times, position, 1)
    fitted_position = intercept + velocity * times
    fit_rms = float(np.sqrt(np.mean((position - fitted_position) ** 2)))
    total_shift = float(position[-1] - position[0])

    aligned = translate_fields(fields, position - position[0])
    aligned_modes = np.fft.fft(aligned, axis=1)
    aligned_modes[:, 0] = 0.0
    aligned_sidebands = np.fft.ifft(aligned_modes, axis=1)
    mean_sidebands = np.mean(aligned_sidebands, axis=0)
    reference_norm = float(np.sum(np.abs(mean_sidebands) ** 2))
    if reference_norm <= np.finfo(float).tiny:
        shape_variation = float("nan")
    else:
        shape_variation = float(np.sqrt(
            np.mean(np.sum(
                np.abs(aligned_sidebands - mean_sidebands[None, :]) ** 2,
                axis=1,
            ))
            / reference_norm
        ))

    is_rigid = bool(
        modulation_fraction > 1.0e-8
        and np.isfinite(shape_variation)
        and fit_rms < 2.0e-2
        and shape_variation < 5.0e-2
    )
    return DriftDiagnostics(
        velocity=float(velocity),
        fit_rms=fit_rms,
        shape_variation=shape_variation,
        total_shift=total_shift,
        tracking_harmonic=harmonic,
        modulation_fraction=modulation_fraction,
        is_rigid_translation=is_rigid,
    )
