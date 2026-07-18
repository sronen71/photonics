"""Convert dimensional cavity parameters to the normalized LLE model."""

from dataclasses import dataclass

import numpy as np

from config_loader import ConfigurationError, load_physics
from dispersion import DispersionRelation, load_dispersion


HBAR_J_S = 1.054571817e-34


@dataclass(frozen=True)
class SolverPhysics:
    """Normalized values consumed by the numerical solvers."""

    alpha: float
    forcing: complex
    dispersion: DispersionRelation
    units: str
    kappa_rad_s: float | None = None
    kappa_external_rad_s: float | None = None
    omega_0_rad_s: float | None = None
    omega_pump_rad_s: float | None = None
    fsr_hz: float | None = None
    g_0_rad_s: float | None = None
    pump_power_w: float | None = None

    @property
    def physical_time_per_normalized_unit_s(self):
        """Return the dimensional duration represented by one normalized time."""
        if self.kappa_rad_s is None:
            return None
        return 2.0 / self.kappa_rad_s


def _finite_float(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ConfigurationError(f"physics.{name} must be a number") from None
    if not np.isfinite(value):
        raise ConfigurationError(f"physics.{name} must be finite")
    return value


def load_solver_physics(config_path):
    """Load physics and return its normalized solver representation."""
    physics = load_physics(config_path)
    if physics.units == "normalized":
        alpha = _finite_float(physics.alpha, "alpha")
        forcing = _finite_float(physics.f_real, "f_real")
        if forcing < 0.0:
            raise ConfigurationError("physics.f_real must be nonnegative")
        dispersion = load_dispersion(physics, config_path)
        return SolverPhysics(
            alpha=alpha,
            forcing=complex(forcing),
            dispersion=dispersion,
            units="normalized",
        )

    kappa = _finite_float(physics.kappa_rad_s, "kappa_rad_s")
    kappa_external = _finite_float(
        physics.kappa_external_rad_s, "kappa_external_rad_s"
    )
    omega_0 = _finite_float(physics.omega_0_rad_s, "omega_0_rad_s")
    omega_pump = _finite_float(
        physics.omega_pump_rad_s, "omega_pump_rad_s"
    )
    fsr_hz = _finite_float(physics.fsr_hz, "fsr_hz")
    g_0 = _finite_float(physics.g_0_rad_s, "g_0_rad_s")
    pump_power = _finite_float(physics.pump_power_w, "pump_power_w")

    positive = {
        "kappa_rad_s": kappa,
        "omega_0_rad_s": omega_0,
        "omega_pump_rad_s": omega_pump,
        "fsr_hz": fsr_hz,
        "g_0_rad_s": g_0,
    }
    for name, value in positive.items():
        if value <= 0.0:
            raise ConfigurationError(f"physics.{name} must be positive")
    if kappa_external <= 0.0 or kappa_external > kappa:
        raise ConfigurationError(
            "physics.kappa_external_rad_s must be positive and no larger "
            "than physics.kappa_rad_s"
        )
    if pump_power < 0.0:
        raise ConfigurationError("physics.pump_power_w must be nonnegative")

    alpha = 2.0 * (omega_0 - omega_pump) / kappa
    forcing = np.sqrt(
        8.0
        * g_0
        * kappa_external
        * pump_power
        / (kappa**3 * HBAR_J_S * omega_pump)
    )
    dispersion = load_dispersion(physics, config_path)
    return SolverPhysics(
        alpha=alpha,
        forcing=complex(forcing),
        dispersion=dispersion,
        units="SI",
        kappa_rad_s=kappa,
        kappa_external_rad_s=kappa_external,
        omega_0_rad_s=omega_0,
        omega_pump_rad_s=omega_pump,
        fsr_hz=fsr_hz,
        g_0_rad_s=g_0,
        pump_power_w=pump_power,
    )


def normalized_summary(physics):
    """Return a compact summary of the normalized parameters."""
    beta_2 = physics.dispersion.seed_beta
    dispersion_text = (
        f"beta_2={beta_2:g}"
        if beta_2 is not None
        else physics.dispersion.description
    )
    return (
        f"alpha={physics.alpha:g}, F={physics.forcing.real:g}, "
        f"{dispersion_text}"
    )
