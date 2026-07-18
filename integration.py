"""Shared fixed-step integration and snapshot collection."""

import numpy as np


def integrate_snapshots(
    initial_state,
    final_time,
    time_step,
    snapshots,
    advance,
):
    """Integrate one or more array-valued state components.

    ``advance(state, step_start, step)`` returns the state after one step.
    State is always represented as a tuple, which keeps the stepping loop
    identical for scalar and coupled-field models.
    """
    state = tuple(np.asarray(component).copy() for component in initial_state)
    number_of_steps = int(np.ceil(final_time / time_step))
    save_every = max(1, number_of_steps // max(1, snapshots - 1))
    saved_times = [0.0]
    saved_components = [[component.copy()] for component in state]

    for index in range(number_of_steps):
        step_start = index * time_step
        step = min(time_step, final_time - step_start)
        state = tuple(advance(state, step_start, step))
        if len(state) != len(saved_components):
            raise RuntimeError("integrator changed the number of state components")
        if not all(np.all(np.isfinite(component)) for component in state):
            raise RuntimeError("solution diverged; reduce the configured time step")
        if (index + 1) % save_every == 0 or index == number_of_steps - 1:
            saved_times.append(min((index + 1) * time_step, final_time))
            for saved, component in zip(saved_components, state):
                saved.append(component.copy())

    return (
        np.asarray(saved_times),
        tuple(np.asarray(saved) for saved in saved_components),
    )
