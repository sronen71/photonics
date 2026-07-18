"""Shared spectral-grid utilities for periodic ring fields."""

import numpy as np


def mode_numbers(spatial_points):
    """Return integer azimuthal mode numbers in NumPy FFT order."""
    return np.fft.fftfreq(int(spatial_points), d=1.0 / int(spatial_points))
