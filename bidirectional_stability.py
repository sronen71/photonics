"""Linear physics of homogeneous pump-split bidirectional resonators."""

from dataclasses import dataclass

import numpy as np

from dispersion import as_dispersion


@dataclass(frozen=True)
class PumpSplitEffectiveParameters:
    """Single-field parameters induced by the counter-propagating pump."""

    forward_alpha: float
    backward_alpha: float
    forward_forcing: complex
    backward_forcing: complex


def _require_pump_split_only(parameters):
    if parameters.reflectivity != 0.0:
        raise ValueError(
            "pump-split formulas require zero external reflectivity"
        )


def pump_split_sideband_gain(
    forward,
    backward,
    parameters,
    mode_number,
):
    """Return FW and BW MI gain for non-pump sidebands of a CW state.

    With no external reflector, the PhCR coupling acts only on the pump
    mode.  Perturbations at nonzero modes therefore separate into the two
    standard 2x2 Kerr stability problems.  The returned gain is in inverse
    normalized slow-time units.
    """
    _require_pump_split_only(parameters)
    forward = complex(forward)
    backward = complex(backward)
    modes = np.asarray(mode_number, dtype=float)
    if not np.all(np.isfinite(modes)) or np.any(modes == 0.0):
        raise ValueError("sideband mode numbers must be finite and nonzero")
    dispersion = as_dispersion(parameters.beta).values(modes)
    total_power = abs(forward) ** 2 + abs(backward) ** 2
    mismatch = 2.0 * total_power - parameters.alpha + dispersion

    def gain(power):
        radicand = power**2 - mismatch**2
        return -1.0 + np.real(np.sqrt(radicand.astype(complex)))

    return gain(abs(forward) ** 2), gain(abs(backward) ** 2)


def pump_split_effective_parameters(forward, backward, parameters):
    """Reduce a homogeneous pump-split state to two effective scalar LLEs."""
    _require_pump_split_only(parameters)
    forward = complex(forward)
    backward = complex(backward)
    coupling = 0.5 * parameters.epsilon_phc
    return PumpSplitEffectiveParameters(
        forward_alpha=parameters.alpha - 2.0 * abs(backward) ** 2,
        backward_alpha=parameters.alpha - 2.0 * abs(forward) ** 2,
        forward_forcing=parameters.forcing - 1j * coupling * backward,
        backward_forcing=-1j * coupling * forward,
    )
