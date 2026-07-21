"""Branch-level checks for the generic pseudo-arclength solver."""

import unittest

import numpy as np

from continuation import (
    initialize_branch,
    pseudo_arclength_continuation,
)
from spectral import mode_numbers
from steady_solver import pack, stationary_residual, uniform_states, unpack
from two_mode import (
    TwoModeParameters,
    pack_two_mode_state,
    parameters_with_pump_detuning,
    two_mode_real_residual,
    unpack_two_mode_state,
)


class PseudoArclengthBenchmarks(unittest.TestCase):
    def test_continuation_passes_saddle_node_with_parameter_reversal(self):
        """Trace x^2-lambda=0 through its exact fold at lambda=0."""
        def fold_residual(state, parameter):
            return np.array([state[0] ** 2 - parameter])

        branch = pseudo_arclength_continuation(
            fold_residual,
            (np.array([-1.0]), 1.0),
            (np.array([-0.9]), 0.81),
            number_of_steps=25,
            step_size=0.08,
            minimum_step_size=1.0e-4,
            maximum_step_size=0.1,
            tolerance=1.0e-10,
        )

        states = branch.states[:, 0]
        parameters = branch.parameters
        self.assertLess(np.min(states), 0.0)
        self.assertGreater(np.max(states), 0.0)
        fold_index = int(np.argmin(parameters))
        self.assertGreater(fold_index, 2)
        self.assertLess(fold_index, parameters.size - 3)
        self.assertGreater(parameters[fold_index - 1], parameters[fold_index])
        self.assertGreater(parameters[fold_index + 1], parameters[fold_index])
        np.testing.assert_allclose(parameters, states**2, atol=1.0e-10)
        self.assertEqual(branch.rejected_steps, 0)

    def test_existing_scalar_lle_residual_is_a_continuation_problem(self):
        """Continue homogeneous scalar-LLE states using the shared engine."""
        forcing = 0.7
        beta = -0.1
        spatial_points = 8
        modes = mode_numbers(spatial_points)

        def lle_residual(vector, alpha):
            amplitude = unpack(vector)
            return pack(stationary_residual(
                amplitude,
                alpha,
                forcing,
                beta,
                modes,
            ))

        alpha_start = 0.4
        initial_amplitude = uniform_states(alpha_start, forcing)[0]
        initial_state = pack(np.full(spatial_points, initial_amplitude))
        first, second = initialize_branch(
            lle_residual,
            initial_state,
            alpha_start,
            0.08,
            tolerance=1.0e-11,
        )
        branch = pseudo_arclength_continuation(
            lle_residual,
            first,
            second,
            number_of_steps=4,
            step_size=0.08,
            maximum_step_size=0.1,
            tolerance=1.0e-10,
        )

        self.assertEqual(len(branch.points), 6)
        for point in branch.points:
            field = unpack(point.state)
            self.assertLess(np.ptp(field.real), 2.0e-11)
            self.assertLess(np.ptp(field.imag), 2.0e-11)
            self.assertLess(point.residual_norm, 1.0e-10)

    def test_two_mode_detuning_branch_uses_the_same_continuation_engine(self):
        """Continue a coupled homogeneous A,B branch through the public API."""
        parameters = TwoModeParameters(
            alpha_comb=0.4,
            alpha_auxiliary=1.0,
            forcing_comb=0.3,
            forcing_auxiliary=0.5,
            beta=-0.1,
            auxiliary_loss_ratio=2.0,
            cavity_coupling=0.2 + 0.1j,
            bus_coupling=0.15,
        )
        spatial_points = 8

        def residual(vector, alpha_comb):
            point_parameters = parameters_with_pump_detuning(
                parameters, alpha_comb
            )
            return two_mode_real_residual(vector, point_parameters)

        state_seed = pack_two_mode_state(
            np.zeros(spatial_points, dtype=complex), 0.0j
        )
        first, second = initialize_branch(
            residual,
            state_seed,
            parameters.alpha_comb,
            0.08,
            tolerance=1.0e-11,
        )
        branch = pseudo_arclength_continuation(
            residual,
            first,
            second,
            number_of_steps=4,
            step_size=0.08,
            tolerance=1.0e-10,
        )

        self.assertEqual(len(branch.points), 6)
        for point in branch.points:
            comb, auxiliary = unpack_two_mode_state(point.state)
            self.assertLess(np.ptp(comb.real), 2.0e-11)
            self.assertLess(np.ptp(comb.imag), 2.0e-11)
            self.assertTrue(np.isfinite(auxiliary.real))
            self.assertTrue(np.isfinite(auxiliary.imag))
            self.assertLess(point.residual_norm, 1.0e-10)


if __name__ == "__main__":
    unittest.main()
