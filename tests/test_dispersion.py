"""Input-format tests for generalized modal dispersion."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from config_loader import ConfigurationError
from dispersion import load_dispersion


class DispersionInputTests(unittest.TestCase):
    def _physics(self, dispersion_csv, **overrides):
        values = {
            "units": "SI",
            "kappa_rad_s": 2.0e9,
            "omega_0_rad_s": 2.0 * np.pi * 193.0e12,
            "fsr_hz": 100.0e9,
            "dispersion_csv": dispersion_csv,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_headerless_pylle_resonances_define_integrated_dispersion(self):
        pump_mode = 212
        pump_frequency_hz = 193.0e12
        d1_hz = 100.0e9
        d2_hz = 2.5e6
        d3_hz = -3000.0

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            dispersion_path = directory / "pylle.csv"
            rows = []
            for relative_mode in range(-8, 9):
                frequency_hz = (
                    pump_frequency_hz
                    + d1_hz * relative_mode
                    + 0.5 * d2_hz * relative_mode**2
                    + d3_hz * relative_mode**3 / 6.0
                )
                rows.append(
                    f"{pump_mode + relative_mode},{frequency_hz:.17g}\n"
                )
            dispersion_path.write_text("".join(rows), encoding="utf-8")

            relation = load_dispersion(
                self._physics(dispersion_path.name),
                directory / "config.yaml",
            )

        # pyLLE obtains D1 from a symmetric quadratic fit through modes -2..2.
        # The cubic term therefore contributes 17*D3/30 to the fitted slope.
        fitted_d1_hz = d1_hz + 17.0 * d3_hz / 30.0
        modes = np.arange(-8.0, 9.0)
        expected_dint_rad_s = 2.0 * np.pi * (
            (d1_hz - fitted_d1_hz) * modes
            + 0.5 * d2_hz * modes**2
            + d3_hz * modes**3 / 6.0
        )
        expected_normalized = (
            -2.0 / self._physics("unused").kappa_rad_s * expected_dint_rad_s
        )
        self.assertEqual(relation.kind, "pylle")
        np.testing.assert_allclose(
            relation.values(modes),
            expected_normalized,
            rtol=3.0e-8,
            atol=2.0e-7,
        )
        self.assertAlmostEqual(
            relation.seed_beta,
            -2.0 / self._physics("unused").kappa_rad_s * 2.0 * np.pi * d2_hz,
            places=8,
        )

    def test_pylle_resonances_require_si_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            dispersion_path = directory / "pylle.csv"
            dispersion_path.write_text(
                "10,190000000000000\n11,190100000000000\n",
                encoding="utf-8",
            )
            physics = SimpleNamespace(
                units="normalized",
                dispersion_csv=dispersion_path.name,
            )
            with self.assertRaisesRegex(ConfigurationError, "require.*SI"):
                load_dispersion(physics, directory / "config.yaml")

    def test_existing_normalized_grid_format_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            dispersion_path = directory / "grid.csv"
            dispersion_path.write_text(
                "k,dispersion\n-2,-0.04\n0,0\n2,-0.04\n",
                encoding="utf-8",
            )
            physics = SimpleNamespace(
                units="normalized",
                dispersion_csv=dispersion_path.name,
            )
            relation = load_dispersion(physics, directory / "config.yaml")

        self.assertEqual(relation.kind, "grid")
        np.testing.assert_allclose(
            relation.values(np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])),
            np.asarray([-0.04, -0.02, 0.0, -0.02, -0.04]),
        )

    def test_pylle_resonances_require_five_modes_around_pump(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            dispersion_path = directory / "pylle.csv"
            dispersion_path.write_text(
                "10,192800000000000\n"
                "11,192900000000000\n"
                "12,193000000000000\n"
                "14,193200000000000\n"
                "15,193300000000000\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "modes 10 through 14"):
                load_dispersion(
                    self._physics(dispersion_path.name),
                    directory / "config.yaml",
                )


if __name__ == "__main__":
    unittest.main()
