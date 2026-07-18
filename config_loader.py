"""Shared YAML configuration loading for the LLE scripts."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("configs") / "config.yaml"
NORMALIZED_PHYSICS_KEYS = {
    "units",
    "alpha",
    "f_real",
    "beta",
    "dispersion_csv",
}
PHYSICAL_PHYSICS_KEYS = {
    "units",
    "kappa_rad_s",
    "kappa_external_rad_s",
    "omega_0_rad_s",
    "omega_pump_rad_s",
    "fsr_hz",
    "g_0_rad_s",
    "pump_power_w",
    "d2_rad_s",
    "dispersion_csv",
}


class ConfigurationError(ValueError):
    """Raised when a configuration file or section is invalid."""


def config_parser(description: str) -> argparse.ArgumentParser:
    """Return a parser whose only user setting is the YAML file path."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    return parser


def load_section(path: Path, section_name: str, expected_keys) -> SimpleNamespace:
    """Load one strictly validated mapping from a YAML configuration file."""
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except OSError as error:
        raise ConfigurationError(f"cannot read configuration {path}: {error}") from None
    except yaml.YAMLError as error:
        raise ConfigurationError(f"invalid YAML in {path}: {error}") from None

    if not isinstance(document, dict):
        raise ConfigurationError(f"{path} must contain a top-level mapping")
    section = document.get(section_name)
    if not isinstance(section, dict):
        raise ConfigurationError(
            f"{path} must contain a '{section_name}' mapping"
        )

    expected = set(expected_keys)
    supplied = set(section)
    missing = sorted(expected - supplied)
    unknown = sorted(supplied - expected)
    if missing:
        raise ConfigurationError(
            f"section '{section_name}' is missing: {', '.join(missing)}"
        )
    if unknown:
        raise ConfigurationError(
            f"section '{section_name}' has unknown settings: {', '.join(unknown)}"
        )
    return SimpleNamespace(**section)


def load_physics(path: Path) -> SimpleNamespace:
    """Load either normalized or dimensional physical LLE parameters."""
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except OSError as error:
        raise ConfigurationError(f"cannot read configuration {path}: {error}") from None
    except yaml.YAMLError as error:
        raise ConfigurationError(f"invalid YAML in {path}: {error}") from None

    if not isinstance(document, dict):
        raise ConfigurationError(f"{path} must contain a top-level mapping")
    section = document.get("physics")
    if not isinstance(section, dict):
        raise ConfigurationError(f"{path} must contain a 'physics' mapping")

    units = section.get("units", "normalized")
    if units not in {"normalized", "SI"}:
        raise ConfigurationError(
            "physics.units must be either normalized or SI"
        )

    supplied = set(section)
    if units == "normalized":
        allowed = NORMALIZED_PHYSICS_KEYS
        required = {"alpha", "f_real"}
        dispersion_choices = {"beta", "dispersion_csv"}
    else:
        allowed = PHYSICAL_PHYSICS_KEYS
        required = {
            "kappa_rad_s",
            "kappa_external_rad_s",
            "omega_0_rad_s",
            "omega_pump_rad_s",
            "fsr_hz",
            "g_0_rad_s",
            "pump_power_w",
        }
        dispersion_choices = {"d2_rad_s", "dispersion_csv"}

    unknown = sorted(supplied - allowed)
    missing = sorted(required - supplied)
    if missing:
        raise ConfigurationError(
            f"section 'physics' is missing: {', '.join(missing)}"
        )
    if unknown:
        raise ConfigurationError(
            f"section 'physics' has unknown settings: {', '.join(unknown)}"
        )
    choices = supplied & dispersion_choices
    if len(choices) != 1:
        raise ConfigurationError(
            "section 'physics' must contain exactly one of "
            + " or ".join(sorted(dispersion_choices))
        )
    values = dict(section)
    values["units"] = units
    return SimpleNamespace(**values)
