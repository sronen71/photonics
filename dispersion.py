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
    except OSError as error:
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


def _load_polynomial(path, rows):
    coefficients = {}
    for row_number, row in rows:
        try:
            order = int(row["order"])
            coefficient = float(row["beta"])
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


def _load_grid(path, rows):
    points = []
    for row_number, row in rows:
        try:
            mode_number = float(row["k"])
            value = float(row["dispersion"])
        except (KeyError, TypeError, ValueError):
            raise ConfigurationError(
                f"invalid gridded dispersion at {path}:{row_number}"
            ) from None
        if not np.isfinite(mode_number) or not np.isfinite(value):
            raise ConfigurationError(
                f"non-finite gridded dispersion at {path}:{row_number}"
            )
        points.append((mode_number, value))
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
        kind="grid",
        description=f"gridded dispersion ({path.name})",
        evaluator=evaluate,
        seed_beta=seed_beta,
        source=path,
    )


def load_dispersion(physics, config_path):
    """Load scalar or CSV dispersion selected in the physics section."""
    if hasattr(physics, "beta"):
        try:
            return quadratic_dispersion(physics.beta)
        except (TypeError, ValueError):
            raise ConfigurationError("physics.beta must be a finite number") from None

    path = Path(physics.dispersion_csv).expanduser()
    if not path.is_absolute():
        path = Path(config_path).resolve().parent / path
    fieldnames, rows = _csv_rows(path)
    fields = set(fieldnames)
    if fields == {"order", "beta"}:
        return _load_polynomial(path, rows)
    if fields == {"k", "dispersion"}:
        return _load_grid(path, rows)
    raise ConfigurationError(
        f"dispersion CSV {path} must have either 'order,beta' or "
        "'k,dispersion' columns"
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
