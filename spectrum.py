"""Shared normalized and physical output-spectrum calculations."""

import numpy as np

from physics import HBAR_J_S


def _time_average(values, times):
    """Average sampled values, accounting for unequal sample intervals."""
    if values.shape[0] == 1:
        return values[0]
    if times is None:
        return np.mean(values, axis=0)
    times = np.asarray(times, dtype=float)
    if (
        times.shape != (values.shape[0],)
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("spectrum times must be finite and strictly increasing")
    intervals = np.diff(times)
    trapezoids = 0.5 * (values[1:] + values[:-1])
    return np.sum(trapezoids * intervals[:, None], axis=0) / np.sum(intervals)


def output_spectrum(fields, physics=None, times=None, drift_velocity=0.0):
    """Return a steady or time-averaged normalized/physical output spectrum.

    ``drift_velocity`` is d(theta_center)/d(normalized_time).  For SI input it
    corrects the comb repetition rate while leaving the cold-cavity FSR saved
    separately.
    """
    fields = np.asarray(fields)
    if fields.ndim == 1:
        fields = fields[None, :]
    if fields.ndim != 2 or fields.shape[1] < 2:
        raise ValueError("spectrum fields must have shape (times, spatial_points)")
    drift_velocity = float(drift_velocity)
    if not np.isfinite(drift_velocity):
        raise ValueError("spectrum drift velocity must be finite")

    spatial_points = fields.shape[1]
    mode_number = np.fft.fftshift(
        np.fft.fftfreq(spatial_points, d=1.0 / spatial_points)
    )
    modal_amplitude = np.fft.fftshift(
        np.fft.fft(fields, axis=-1), axes=-1
    ) / spatial_points

    if physics is None or physics.units == "normalized":
        normalized_power = _time_average(
            np.abs(modal_amplitude) ** 2, times
        )
        positive = normalized_power[normalized_power > 0.0]
        floor = positive.max() * 1e-15 if positive.size else 1e-30
        power_db = 10.0 * np.log10(np.maximum(normalized_power, floor))
        return {
            "axis": mode_number,
            "power_db": power_db,
            "axis_label": "Normalized optical frequency (mode number)",
            "power_label": "Normalized spectral power (dB)",
            "title": "Normalized output spectrum",
            "saved": {
                "output_mode_number": mode_number,
                "output_normalized_power": normalized_power,
                "output_normalized_power_db": power_db,
                "output_spectrum_units": "normalized",
                "spectrum_drift_velocity_normalized": drift_velocity,
            },
        }

    intracavity_modes = np.sqrt(
        physics.kappa_rad_s / (2.0 * physics.g_0_rad_s)
    ) * modal_amplitude
    output_wave = -np.sqrt(physics.kappa_external_rad_s) * intracavity_modes
    pump_index = int(np.flatnonzero(mode_number == 0.0)[0])
    input_wave = np.sqrt(
        physics.pump_power_w / (HBAR_J_S * physics.omega_pump_rad_s)
    )
    output_wave[:, pump_index] += input_wave

    repetition_rate_shift_hz = (
        physics.kappa_rad_s * drift_velocity / (4.0 * np.pi)
    )
    effective_repetition_rate_hz = (
        physics.fsr_hz + repetition_rate_shift_hz
    )
    if effective_repetition_rate_hz <= 0.0:
        raise ValueError(
            "drift correction gives a nonpositive repetition rate"
        )
    optical_omega = (
        physics.omega_pump_rad_s
        + 2.0 * np.pi * effective_repetition_rate_hz * mode_number
    )
    if np.any(optical_omega <= 0.0):
        raise ValueError(
            "the physical frequency grid contains nonpositive frequencies; "
            "reduce spatial_points or check physics.fsr_hz and "
            "physics.omega_pump_rad_s"
        )
    instantaneous_power_w = (
        HBAR_J_S * optical_omega[None, :] * np.abs(output_wave) ** 2
    )
    output_power_w = _time_average(instantaneous_power_w, times)
    positive = output_power_w[output_power_w > 0.0]
    floor_w = max(positive.max() * 1e-15, 1e-18) if positive.size else 1e-18
    output_power_dbm = 10.0 * np.log10(
        np.maximum(output_power_w, floor_w) / 1e-3
    )
    optical_frequency_thz = optical_omega / (2.0 * np.pi * 1e12)
    return {
        "axis": optical_frequency_thz,
        "power_db": output_power_dbm,
        "axis_label": "Optical frequency (THz)",
        "power_label": "Through-port output power (dBm)",
        "title": (
            "Physical output spectrum"
            if drift_velocity == 0.0
            else "Physical output spectrum (drift-corrected frequencies)"
        ),
        "saved": {
            "output_frequency_thz": optical_frequency_thz,
            "output_power_w": output_power_w,
            "output_power_dbm": output_power_dbm,
            "output_spectrum_units": "SI",
            "kappa_rad_s": physics.kappa_rad_s,
            "kappa_external_rad_s": physics.kappa_external_rad_s,
            "omega_pump_rad_s": physics.omega_pump_rad_s,
            "fsr_hz": physics.fsr_hz,
            "spectrum_drift_velocity_normalized": drift_velocity,
            "repetition_rate_shift_hz": repetition_rate_shift_hz,
            "effective_repetition_rate_hz": effective_repetition_rate_hz,
            "g_0_rad_s": physics.g_0_rad_s,
            "pump_power_w": physics.pump_power_w,
        },
    }
