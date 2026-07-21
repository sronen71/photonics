"""Model-independent pseudo-arclength continuation for steady residuals.

The continuation parameter is promoted to an unknown and the physical
equations are bordered by the pseudo-arclength condition.  Models only need
to expose a square real residual ``residual(state, parameter)``.  Complex
fields, phase conditions, and any drift or shift-rate variables stay in the
model adapter that packs them into the real state vector.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import NoConvergence, newton_krylov


class ContinuationError(RuntimeError):
    """Raised when a continuation point cannot be corrected."""


@dataclass(frozen=True)
class ContinuationPoint:
    """One converged point on a one-parameter solution branch."""

    state: np.ndarray
    parameter: float
    residual_norm: float
    corrector_iterations: int = 0
    arclength_step: float = 0.0


@dataclass(frozen=True)
class ContinuationResult:
    """A continued branch and the number of rejected predictor steps."""

    points: tuple[ContinuationPoint, ...]
    rejected_steps: int

    @property
    def states(self):
        """Return branch states stacked along the first axis."""
        return np.stack([point.state for point in self.points])

    @property
    def parameters(self):
        """Return the continued parameter values."""
        return np.asarray([point.parameter for point in self.points])


class ArclengthMetric:
    """Weighted metric used to balance state and parameter changes.

    By default the state contribution is its mean-square change, making the
    metric insensitive to the number of collocation points.  ``parameter_scale``
    is the parameter change that has unit arclength.
    """

    def __init__(self, state_size, state_weights=None, parameter_scale=1.0):
        if not isinstance(state_size, (int, np.integer)) or state_size < 1:
            raise ValueError("state_size must be a positive integer")
        if state_weights is None:
            weights = np.full(state_size, 1.0 / state_size)
        else:
            weights = np.broadcast_to(
                np.asarray(state_weights, dtype=float), (state_size,)
            ).copy()
        parameter_scale = float(parameter_scale)
        if (
            not np.all(np.isfinite(weights))
            or np.any(weights <= 0.0)
            or not np.isfinite(parameter_scale)
            or parameter_scale <= 0.0
        ):
            raise ValueError(
                "arclength weights and parameter_scale must be finite and positive"
            )
        self.state_weights = weights
        self.parameter_weight = 1.0 / parameter_scale**2

    def inner(self, left_state, left_parameter, right_state, right_parameter):
        """Return the weighted inner product of two augmented vectors."""
        return float(
            np.dot(
                self.state_weights * np.asarray(left_state, dtype=float),
                np.asarray(right_state, dtype=float),
            )
            + self.parameter_weight
            * float(left_parameter)
            * float(right_parameter)
        )

    def norm(self, state, parameter):
        """Return the weighted norm of an augmented vector."""
        squared = self.inner(state, parameter, state, parameter)
        return float(np.sqrt(max(0.0, squared)))


def _real_state(state, name="state"):
    state = np.asarray(state, dtype=float)
    if state.ndim != 1 or state.size < 1 or not np.all(np.isfinite(state)):
        raise ValueError(f"{name} must be a finite one-dimensional real vector")
    return state.copy()


def _evaluate_square_residual(residual, state, parameter):
    value = np.asarray(residual(state, float(parameter)), dtype=float)
    if value.shape != state.shape:
        raise ValueError(
            "continuation residual must have the same shape as the state"
        )
    if not np.all(np.isfinite(value)):
        raise ContinuationError("continuation residual produced non-finite values")
    return value


def _maximum_norm(value):
    return float(np.max(np.abs(value)))


def _solve_nonlinear(residual, initial_guess, tolerance, max_iterations):
    """Solve a square real system and retain a useful iteration count."""
    initial_guess = _real_state(initial_guess, "initial_guess")
    initial_residual = np.asarray(residual(initial_guess), dtype=float)
    if initial_residual.shape != initial_guess.shape:
        raise ValueError("nonlinear residual and unknown must have equal shapes")
    if not np.all(np.isfinite(initial_residual)):
        raise ContinuationError("nonlinear residual produced non-finite values")
    if _maximum_norm(initial_residual) <= tolerance:
        return initial_guess, 0

    iteration_count = 0

    def count_iteration(_solution, _residual):
        nonlocal iteration_count
        iteration_count += 1

    try:
        solution = newton_krylov(
            residual,
            initial_guess,
            f_tol=tolerance,
            maxiter=max_iterations,
            callback=count_iteration,
            line_search="armijo",
            verbose=False,
        )
    except NoConvergence as error:
        candidate = np.asarray(error.args[0], dtype=float)
        candidate_residual = np.asarray(residual(candidate), dtype=float)
        if (
            candidate.shape == initial_guess.shape
            and candidate_residual.shape == candidate.shape
            and np.all(np.isfinite(candidate_residual))
            and _maximum_norm(candidate_residual) <= tolerance
        ):
            return candidate, iteration_count
        residual_norm = (
            _maximum_norm(candidate_residual)
            if candidate_residual.shape == candidate.shape
            and np.all(np.isfinite(candidate_residual))
            else float("inf")
        )
        raise ContinuationError(
            "Newton--Krylov corrector did not converge after "
            f"{iteration_count} iterations; maximum residual={residual_norm:.3e}"
        ) from None
    except ValueError as error:
        if "Jacobian inversion yielded zero vector" not in str(error):
            raise
        raise ContinuationError(
            "Newton--Krylov corrector could not construct a search direction; "
            "reduce the continuation step or improve the initial seed"
        ) from None

    solution = np.asarray(solution, dtype=float)
    final_residual = np.asarray(residual(solution), dtype=float)
    residual_norm = _maximum_norm(final_residual)
    if residual_norm > tolerance:
        raise ContinuationError(
            "Newton--Krylov corrector returned without satisfying the residual "
            f"tolerance; maximum residual={residual_norm:.3e}"
        )
    return solution, iteration_count


def solve_fixed_parameter(
    residual,
    initial_state,
    parameter,
    *,
    tolerance=1.0e-9,
    max_iterations=50,
):
    """Solve ``residual(state, parameter)=0`` at one fixed parameter."""
    state = _real_state(initial_state, "initial_state")
    parameter = float(parameter)
    if not np.isfinite(parameter):
        raise ValueError("parameter must be finite")
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("tolerance and max_iterations must be positive")

    def fixed_residual(candidate):
        return _evaluate_square_residual(residual, candidate, parameter)

    solution, iterations = _solve_nonlinear(
        fixed_residual, state, tolerance, max_iterations
    )
    residual_norm = _maximum_norm(fixed_residual(solution))
    return ContinuationPoint(
        state=solution,
        parameter=parameter,
        residual_norm=residual_norm,
        corrector_iterations=iterations,
    )


def initialize_branch(
    residual,
    initial_state,
    initial_parameter,
    parameter_step,
    *,
    tolerance=1.0e-9,
    max_iterations=50,
):
    """Use one seed to obtain the first two fixed-parameter branch points."""
    parameter_step = float(parameter_step)
    if not np.isfinite(parameter_step) or parameter_step == 0.0:
        raise ValueError("parameter_step must be finite and nonzero")
    first = solve_fixed_parameter(
        residual,
        initial_state,
        initial_parameter,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    second = solve_fixed_parameter(
        residual,
        first.state,
        float(initial_parameter) + parameter_step,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return first, second


def _unit_secant(previous, current, metric):
    state_change = current.state - previous.state
    parameter_change = current.parameter - previous.parameter
    secant_norm = metric.norm(state_change, parameter_change)
    if not np.isfinite(secant_norm) or secant_norm <= np.finfo(float).eps:
        raise ValueError("initial continuation points must be distinct")
    return state_change / secant_norm, parameter_change / secant_norm


def pseudo_arclength_continuation(
    residual,
    first,
    second,
    *,
    number_of_steps,
    step_size=None,
    minimum_step_size=None,
    maximum_step_size=None,
    tolerance=1.0e-9,
    max_corrector_iterations=20,
    target_corrector_iterations=4,
    step_growth=1.35,
    step_shrink=0.5,
    state_weights=None,
    parameter_scale=1.0,
):
    """Continue a branch with an adaptive secant predictor and PAL corrector.

    ``first`` and ``second`` may be ``ContinuationPoint`` instances returned
    by :func:`initialize_branch`, or ``(state, parameter)`` pairs.  Their order
    selects the initial branch direction.  ``number_of_steps`` counts new
    pseudo-arclength points, so both supplied points are retained in the result.
    """
    if not isinstance(number_of_steps, (int, np.integer)) or number_of_steps < 0:
        raise ValueError("number_of_steps must be a nonnegative integer")
    if tolerance <= 0.0 or max_corrector_iterations < 1:
        raise ValueError("corrector tolerance and iteration limit must be positive")
    if target_corrector_iterations < 1:
        raise ValueError("target_corrector_iterations must be positive")
    if not 1.0 < step_growth or not 0.0 < step_shrink < 1.0:
        raise ValueError("step_growth must exceed one and step_shrink lie in (0, 1)")

    def as_point(value, name):
        if isinstance(value, ContinuationPoint):
            state = _real_state(value.state, f"{name}.state")
            parameter = float(value.parameter)
            iterations = value.corrector_iterations
            arclength_step = value.arclength_step
        else:
            try:
                state, parameter = value
            except (TypeError, ValueError):
                raise ValueError(
                    f"{name} must be a ContinuationPoint or (state, parameter)"
                ) from None
            state = _real_state(state, f"{name}.state")
            parameter = float(parameter)
            iterations = 0
            arclength_step = 0.0
        if not np.isfinite(parameter):
            raise ValueError(f"{name}.parameter must be finite")
        physical_residual = _evaluate_square_residual(
            residual, state, parameter
        )
        residual_norm = _maximum_norm(physical_residual)
        if residual_norm > tolerance:
            raise ValueError(
                f"{name} is not converged; maximum residual={residual_norm:.3e}"
            )
        return ContinuationPoint(
            state,
            parameter,
            residual_norm,
            corrector_iterations=iterations,
            arclength_step=arclength_step,
        )

    first = as_point(first, "first")
    second = as_point(second, "second")
    if first.state.shape != second.state.shape:
        raise ValueError("initial continuation states must have equal shapes")
    metric = ArclengthMetric(
        first.state.size,
        state_weights=state_weights,
        parameter_scale=parameter_scale,
    )
    tangent_state, tangent_parameter = _unit_secant(first, second, metric)
    initial_distance = metric.norm(
        second.state - first.state,
        second.parameter - first.parameter,
    )
    step_size = initial_distance if step_size is None else float(step_size)
    minimum_step_size = (
        1.0e-3 * step_size
        if minimum_step_size is None
        else float(minimum_step_size)
    )
    maximum_step_size = (
        4.0 * step_size
        if maximum_step_size is None
        else float(maximum_step_size)
    )
    if (
        not np.all(np.isfinite(
            [step_size, minimum_step_size, maximum_step_size]
        ))
        or minimum_step_size <= 0.0
        or step_size < minimum_step_size
        or maximum_step_size < step_size
    ):
        raise ValueError(
            "continuation step sizes must satisfy 0 < minimum <= step <= maximum"
        )

    points = [first, second]
    rejected_steps = 0
    current_step = step_size
    for _ in range(number_of_steps):
        current = points[-1]
        attempted_step = current_step
        while True:
            predicted_state = current.state + attempted_step * tangent_state
            predicted_parameter = (
                current.parameter + attempted_step * tangent_parameter
            )
            initial_augmented = np.concatenate(
                (predicted_state, [predicted_parameter])
            )

            def augmented_residual(augmented):
                state = augmented[:-1]
                parameter = float(augmented[-1])
                physical = _evaluate_square_residual(
                    residual, state, parameter
                )
                phase = metric.inner(
                    tangent_state,
                    tangent_parameter,
                    state - predicted_state,
                    parameter - predicted_parameter,
                )
                return np.concatenate((physical, [phase]))

            try:
                corrected, iterations = _solve_nonlinear(
                    augmented_residual,
                    initial_augmented,
                    tolerance,
                    max_corrector_iterations,
                )
            except ContinuationError:
                rejected_steps += 1
                reduced_step = attempted_step * step_shrink
                if reduced_step < minimum_step_size:
                    if attempted_step <= minimum_step_size:
                        raise ContinuationError(
                            "pseudo-arclength corrector failed at the minimum "
                            f"step size {minimum_step_size:.3e}"
                        ) from None
                    reduced_step = minimum_step_size
                attempted_step = reduced_step
                continue
            break

        corrected_state = corrected[:-1]
        corrected_parameter = float(corrected[-1])
        physical_residual = _evaluate_square_residual(
            residual, corrected_state, corrected_parameter
        )
        points.append(ContinuationPoint(
            state=corrected_state,
            parameter=corrected_parameter,
            residual_norm=_maximum_norm(physical_residual),
            corrector_iterations=iterations,
            arclength_step=attempted_step,
        ))

        new_tangent_state, new_tangent_parameter = _unit_secant(
            points[-2], points[-1], metric
        )
        orientation = metric.inner(
            tangent_state,
            tangent_parameter,
            new_tangent_state,
            new_tangent_parameter,
        )
        if orientation < 0.0:
            new_tangent_state = -new_tangent_state
            new_tangent_parameter = -new_tangent_parameter
        tangent_state = new_tangent_state
        tangent_parameter = new_tangent_parameter

        current_step = attempted_step
        if iterations <= max(1, target_corrector_iterations // 2):
            current_step = min(maximum_step_size, current_step * step_growth)
        elif iterations > target_corrector_iterations:
            current_step = max(minimum_step_size, current_step * step_shrink)

    return ContinuationResult(tuple(points), rejected_steps)
