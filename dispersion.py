"""Shared spectral dispersion relations for the LLE solvers."""

import csv
import math
from pathlib import Path

import numpy as np

from config_loader import ConfigurationError


class DispersionRelation:
    """Represent the normalized modal shift D(k) used by both solvers."""

    def __init__(
        self,
        kind,
        description,
        evaluator,
        seed_beta,
        source=None,
    ):
        self.kind = kind
        self.description = description
        self._evaluator = evaluator
        self.seed_beta = seed_beta
        self.source = source

    def values(self, mode_number):
        """Return D(k) at the requested azimuthal mode numbers."""
        values = np.asarray(self._evaluator(np.asarray(mode_number, dtype=float)))
        if not np.all(np.isfinite(values)):
            raise ValueError("dispersion relation produced non-finite values")
        return values


def quadratic_dispersion(beta):
    """Return the original second-order dispersion relation."""
    beta = float(beta)
    if not np.isfinite(beta):
        raise ValueError("beta must be finite")
    return DispersionRelation(
        kind="quadratic",
        description=rf"$\beta={beta:g}$",
        evaluator=lambda mode_number: 0.5 * beta * mode_number**2,
        seed_beta=beta,
    )


def _csv_rows(path):
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            lines = [line for line in stream if not line.lstrip().startswith("#")]
    except (OSError, csv.Error) as error:
        raise ConfigurationError(f"cannot read dispersion CSV {path}: {error}") from None

    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        raise ConfigurationError(f"dispersion CSV {path} has no header")
    fieldnames = [name.strip().lower() for name in reader.fieldnames]
    rows = []
    for row_number, row in enumerate(reader, start=2):
        if not row or all(value is None or not value.strip() for value in row.values()):
            continue
        rows.append((row_number, {
            name.strip().lower(): value.strip()
            for name, value in row.items()
            if name is not None and value is not None
        }))
    return fieldnames, rows


def _first_csv_fields(path):
    """Return the first non-comment CSV row without assuming a header."""
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            for line in stream:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                return next(csv.reader([line]))
    except (OSError, csv.Error) as error:
        raise ConfigurationError(f"cannot read dispersion CSV {path}: {error}") from None
    raise ConfigurationError(f"dispersion CSV {path} contains no data")


def _headerless_numeric_pair(fields):
    """Return whether fields look like a pyLLE mode,frequency row."""
    if len(fields) != 2:
        return False
    try:
        values = [float(field.strip()) for field in fields]
    except ValueError:
        return False
    return all(np.isfinite(values))


def _pylle_rows(path):
    """Read a headerless pyLLE mode-order,resonance-frequency file."""
    rows = []
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row_number, fields in enumerate(csv.reader(stream), start=1):
                if not fields or all(not field.strip() for field in fields):
                    continue
                if fields[0].lstrip().startswith("#"):
                    continue
                if len(fields) != 2:
                    raise ConfigurationError(
                        f"invalid pyLLE resonance data at {path}:{row_number}"
                    )
                rows.append((row_number, fields))
    except (OSError, csv.Error) as error:
        raise ConfigurationError(f"cannot read dispersion CSV {path}: {error}") from None
    return rows


def _load_polynomial(path, rows, coefficient_name="beta", scale=1.0):
    coefficients = {}
    for row_number, row in rows:
        try:
            order = int(row["order"])
            coefficient = scale * float(row[coefficient_name])
        except (KeyError, TypeError, ValueError):
            raise ConfigurationError(
                f"invalid polynomial dispersion at {path}:{row_number}"
            ) from None
        if order < 0 or order in coefficients or not np.isfinite(coefficient):
            raise ConfigurationError(
                f"invalid or duplicate polynomial order at {path}:{row_number}"
            )
        coefficients[order] = coefficient
    if not coefficients:
        raise ConfigurationError(f"dispersion CSV {path} contains no coefficients")

    def evaluate(mode_number):
        result = np.zeros_like(mode_number, dtype=float)
        for order, coefficient in coefficients.items():
            result += coefficient * mode_number**order / math.factorial(order)
        return result

    return DispersionRelation(
        kind="polynomial",
        description=f"polynomial dispersion ({path.name})",
        evaluator=evaluate,
        seed_beta=coefficients.get(2),
        source=path,
    )


def _grid_relation(path, points, kind="grid", description=None):
    """Build an interpolated relation from validated, scaled grid points."""
    points.sort()
    if len(points) < 2 or any(
        left[0] == right[0] for left, right in zip(points, points[1:])
    ):
        raise ConfigurationError(
            f"dispersion CSV {path} must contain at least two distinct k values"
        )

    grid = np.asarray([point[0] for point in points])
    values = np.asarray([point[1] for point in points])

    def evaluate(mode_number):
        minimum = float(np.min(mode_number))
        maximum = float(np.max(mode_number))
        if minimum < grid[0] or maximum > grid[-1]:
            raise ValueError(
                "dispersion grid does not cover the solver modes "
                f"[{minimum:g}, {maximum:g}]; CSV range is "
                f"[{grid[0]:g}, {grid[-1]:g}]"
            )
        return np.interp(mode_number, grid, values)

    seed_beta = None
    if grid[0] <= -1.0 and grid[-1] >= 1.0:
        local = np.interp([-1.0, 0.0, 1.0], grid, values)
        seed_beta = float(local[2] - 2.0 * local[1] + local[0])

    return DispersionRelation(
        kind=kind,
        description=description or f"gridded dispersion ({path.name})",
        evaluator=evaluate,
        seed_beta=seed_beta,
        source=path,
    )


def _load_grid(path, rows, scale=1.0):
    points = []
    for row_number, row in rows:
        try:
            mode_number = float(row["k"])
            value = scale * float(row["dispersion"])
        except (KeyError, TypeError, ValueError):
            raise ConfigurationError(
                f"invalid gridded dispersion at {path}:{row_number}"
            ) from None
        if not np.isfinite(mode_number) or not np.isfinite(value):
            raise ConfigurationError(
                f"non-finite gridded dispersion at {path}:{row_number}"
            )
        points.append((mode_number, value))
    return _grid_relation(path, points)


def _load_pylle_resonances(path, physics):
    """Convert a pyLLE mode-order,resonance-frequency file to D(k)."""
    resonances = []
    for row_number, fields in _pylle_rows(path):
        try:
            mode_number = float(fields[0].strip())
            frequency_hz = float(fields[1].strip())
        except ValueError:
            raise ConfigurationError(
                f"invalid pyLLE resonance data at {path}:{row_number}"
            ) from None
        if (
            not np.isfinite(mode_number)
            or not mode_number.is_integer()
            or not np.isfinite(frequency_hz)
            or frequency_hz <= 0.0
        ):
            raise ConfigurationError(
                f"invalid pyLLE resonance data at {path}:{row_number}"
            )
        resonances.append((int(mode_number), frequency_hz))

    resonances.sort()
    if len(resonances) < 5 or any(
        left[0] == right[0] for left, right in zip(resonances, resonances[1:])
    ):
        raise ConfigurationError(
            f"pyLLE resonance file {path} must contain at least five "
            "distinct mode orders"
        )
    if any(
        left[1] >= right[1] for left, right in zip(resonances, resonances[1:])
    ):
        raise ConfigurationError(
            f"resonance frequencies in {path} must increase with mode order"
        )

    resonance_frequency_hz = float(physics.omega_0_rad_s) / (2.0 * np.pi)
    pump_mode, pump_frequency_hz = min(
        resonances,
        key=lambda point: abs(point[1] - resonance_frequency_hz),
    )
    if abs(pump_frequency_hz - resonance_frequency_hz) > 0.5 * float(
        physics.fsr_hz
    ):
        raise ConfigurationError(
            f"pyLLE resonance file {path} has no mode within half an FSR of "
            "physics.omega_0_rad_s"
        )

    by_mode = dict(resonances)
    local_modes = np.arange(-2, 3, dtype=float)
    try:
        local_frequencies = np.asarray(
            [by_mode[pump_mode + int(offset)] for offset in local_modes]
        )
    except KeyError:
        raise ConfigurationError(
            f"pyLLE resonance file {path} must include modes "
            f"{pump_mode - 2} through {pump_mode + 2} around the pumped mode"
        ) from None

    # Match pyLLE: D1 is the linear coefficient of a quadratic fit through
    # the pumped resonance and its two nearest modes on either side.
    local_frequency_offsets = local_frequencies - pump_frequency_hz
    d1_hz = float(np.polyfit(local_modes, local_frequency_offsets, 2)[1])
    scale = -2.0 / float(physics.kappa_rad_s)
    points = []
    for mode_number, frequency_hz in resonances:
        relative_mode = mode_number - pump_mode
        dint_rad_s = 2.0 * np.pi * (
            frequency_hz - pump_frequency_hz - d1_hz * relative_mode
        )
        points.append((relative_mode, scale * dint_rad_s))

    return _grid_relation(
        path,
        points,
        kind="pylle",
        description=f"pyLLE resonance dispersion ({path.name})",
    )


def load_dispersion(physics, config_path):
    """Load scalar or CSV dispersion selected in the physics section."""
    if hasattr(physics, "beta"):
        try:
            return quadratic_dispersion(physics.beta)
        except (TypeError, ValueError):
            raise ConfigurationError("physics.beta must be a finite number") from None

    if hasattr(physics, "d2_rad_s"):
        return quadratic_dispersion(
            -2.0 * float(physics.d2_rad_s) / float(physics.kappa_rad_s)
        )

    physical_input = getattr(physics, "units", "normalized") == "SI"
    scale = -2.0 / float(physics.kappa_rad_s) if physical_input else 1.0
    coefficient_name = "d" if physical_input else "beta"

    path = Path(physics.dispersion_csv).expanduser()
    if not path.is_absolute():
        path = Path(config_path).resolve().parent / path
    first_fields = _first_csv_fields(path)
    if _headerless_numeric_pair(first_fields):
        if not physical_input:
            raise ConfigurationError(
                "headerless pyLLE resonance files require physics.units: SI"
            )
        return _load_pylle_resonances(path, physics)
    fieldnames, rows = _csv_rows(path)
    fields = set(fieldnames)
    if fields == {"order", coefficient_name}:
        return _load_polynomial(
            path,
            rows,
            coefficient_name=coefficient_name,
            scale=scale,
        )
    if fields == {"k", "dispersion"}:
        return _load_grid(path, rows, scale=scale)
    polynomial_header = f"order,{coefficient_name}"
    raise ConfigurationError(
        f"dispersion CSV {path} must have either '{polynomial_header}' or "
        "'k,dispersion' columns, or be a headerless pyLLE resonance file"
    )


def as_dispersion(relation):
    """Accept a relation object or a scalar beta for library compatibility."""
    if isinstance(relation, DispersionRelation):
        return relation
    return quadratic_dispersion(relation)


def soliton_seed_beta(relation):
    """Return the local quadratic coefficient used by the analytic seed."""
    relation = as_dispersion(relation)
    if relation.seed_beta is None:
        raise ValueError(
            "the selected dispersion does not define a local beta_2 for the "
            "analytic soliton seed"
        )
    return relation.seed_beta


def dispersion_is_even(relation, mode_number, tolerance=1.0e-12):
    """Return whether represented positive and negative modes have equal D."""
    mode_number = np.asarray(mode_number)
    values = as_dispersion(relation).values(mode_number)
    represented = {
        int(round(mode)): value
        for mode, value in zip(mode_number, values)
    }
    differences = [
        abs(value - represented[-mode])
        for mode, value in represented.items()
        if -mode in represented
    ]
    scale = max(1.0, float(np.max(np.abs(values))))
    return max(differences, default=0.0) <= tolerance * scale
