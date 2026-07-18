"""Shared YAML configuration loading for the LLE scripts."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
PHYSICS_KEYS = {"alpha", "f_real", "beta", "dispersion_csv"}


class ConfigurationError(ValueError):
    """Raised when a configuration file or section is invalid."""


def config_parser(description: str) -> argparse.ArgumentParser:
    """Return a parser whose only user setting is the YAML file path."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML configuration file (default: {DEFAULT_CONFIG_PATH.name})",
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
    """Load physics with either scalar beta or an external dispersion CSV."""
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

    supplied = set(section)
    unknown = sorted(supplied - PHYSICS_KEYS)
    missing = sorted({"alpha", "f_real"} - supplied)
    if missing:
        raise ConfigurationError(
            f"section 'physics' is missing: {', '.join(missing)}"
        )
    if unknown:
        raise ConfigurationError(
            f"section 'physics' has unknown settings: {', '.join(unknown)}"
        )
    choices = supplied & {"beta", "dispersion_csv"}
    if len(choices) != 1:
        raise ConfigurationError(
            "section 'physics' must contain exactly one of beta or dispersion_csv"
        )
    return SimpleNamespace(**section)
