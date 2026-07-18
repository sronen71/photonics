"""Input-output observables for the bidirectional PhCR circuit."""

import numpy as np

from bidirectional import reflector_mask
from spectrum import time_average
from spectral import mode_numbers


def bidirectional_output(
    forward_fields,
    backward_fields,
    parameters,
    times=None,
    pump_frequency_hz=None,
    fsr_hz=None,
):
    """Calculate both port spectra, conversion efficiency, and pump use.

    Port fields use the paper's bus-ring scattering convention.  Within the
    reflection band, a fraction R of the forward through field feeds the
    backward input and the remaining fraction 1-R exits the forward port.
    """
    forward_fields = np.asarray(forward_fields, dtype=complex)
    backward_fields = np.asarray(backward_fields, dtype=complex)
    if forward_fields.ndim == 1:
        forward_fields = forward_fields[None, :]
    if backward_fields.ndim == 1:
        backward_fields = backward_fields[None, :]
    if forward_fields.shape != backward_fields.shape:
        raise ValueError("forward and backward histories must have equal shapes")
    if forward_fields.ndim != 2 or forward_fields.shape[1] < 2:
        raise ValueError("field histories must have shape (times, spatial_points)")
    if abs(parameters.forcing) == 0.0:
        raise ValueError("nonzero forcing is required for input-normalized output")

    spatial_points = forward_fields.shape[1]
    modes = mode_numbers(spatial_points)
    reflected = reflector_mask(modes, parameters.reflector_half_width)
    bus_scale = np.sqrt(
        (parameters.coupling_factor + 1.0)
        / (2.0 * parameters.coupling_factor)
    )
    input_power = abs(bus_scale * parameters.forcing) ** 2
    forward_modes = np.fft.fft(forward_fields, axis=1) / spatial_points
    backward_modes = np.fft.fft(backward_fields, axis=1) / spatial_points
    drive_modes = np.zeros(spatial_points, dtype=complex)
    drive_modes[0] = parameters.forcing

    forward_through = bus_scale * (
        drive_modes[None, :] - parameters.bus_coupling * forward_modes
    )
    forward_output = np.where(
        reflected,
        np.sqrt(1.0 - parameters.reflectivity),
        1.0,
    )[None, :] * forward_through
    backward_output = bus_scale * (
        parameters.reflector_amplitude
        * reflected[None, :]
        * (
            drive_modes[None, :]
            - parameters.bus_coupling * forward_modes
        )
        - parameters.bus_coupling * backward_modes
    )

    frequency_weight = np.ones(spatial_points)
    optical_frequency_hz = None
    if pump_frequency_hz is not None or fsr_hz is not None:
        if pump_frequency_hz is None or fsr_hz is None:
            raise ValueError(
                "pump_frequency_hz and fsr_hz must be supplied together"
            )
        optical_frequency_hz = (
            float(pump_frequency_hz) + float(fsr_hz) * modes
        )
        if np.any(optical_frequency_hz <= 0.0):
            raise ValueError("the optical frequency grid must be positive")
        frequency_weight = optical_frequency_hz / float(pump_frequency_hz)

    forward_power = (
        np.abs(forward_output) ** 2
        * frequency_weight[None, :]
        / input_power
    )
    backward_power = (
        np.abs(backward_output) ** 2
        * frequency_weight[None, :]
        / input_power
    )
    comb = modes != 0.0
    pump_power_ratio = forward_power[:, 0] + backward_power[:, 0]
    forward_comb_ratio = np.sum(forward_power[:, comb], axis=1)
    backward_comb_ratio = np.sum(backward_power[:, comb], axis=1)
    conversion_efficiency = forward_comb_ratio + backward_comb_ratio
    intrinsic_loss_ratio = (
        2.0
        / (parameters.coupling_factor + 1.0)
        * np.sum(
            (
                np.abs(forward_modes) ** 2
                + np.abs(backward_modes) ** 2
            )
            * frequency_weight[None, :],
            axis=1,
        )
        / input_power
    )

    result = {
        "mode_number": np.fft.fftshift(modes),
        "forward_power_ratio": np.fft.fftshift(
            time_average(forward_power, times)
        ),
        "backward_power_ratio": np.fft.fftshift(
            time_average(backward_power, times)
        ),
        "pump_power_ratio": pump_power_ratio,
        "forward_comb_ratio": forward_comb_ratio,
        "backward_comb_ratio": backward_comb_ratio,
        "conversion_efficiency": conversion_efficiency,
        "pump_consumption": 1.0 - pump_power_ratio,
        "intrinsic_loss_ratio": intrinsic_loss_ratio,
        "steady_energy_balance": (
            pump_power_ratio + conversion_efficiency + intrinsic_loss_ratio
        ),
    }
    if optical_frequency_hz is not None:
        result["optical_frequency_thz"] = (
            np.fft.fftshift(optical_frequency_hz) / 1.0e12
        )
    return result
