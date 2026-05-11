"""Tests for the GP field estimator and pocket detection / calibration."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from sim.room_sim import RoomSimulator
from model.gp_field import GPField
from model.uq import detect_pockets, calibration_over_episode


def test_gp_predict_shape():
    nx, ny, nz = 6, 5, 4
    gp = GPField(nx=nx, ny=ny, nz=nz)
    locs  = np.array([[1, 1, 1], [4, 3, 2], [2, 2, 0]], dtype=float)
    temps = np.array([20.0, 22.0, 18.0])
    gp.fit(locs, temps)
    mean, std = gp.predict()
    assert mean.shape == (nx, ny, nz)
    assert std.shape  == (nx, ny, nz)
    assert np.all(std >= 0.0)


def test_gp_predict_requires_fit():
    gp = GPField(nx=4, ny=4, nz=3)
    with pytest.raises(RuntimeError):
        gp.predict()


def test_pocket_detection_returns_bool_arrays_of_right_shape():
    mean = np.full((4, 3, 2), 22.0)
    std  = np.full((4, 3, 2), 0.1)
    mean[0, 0, 0] = 30.0   # hot
    mean[1, 1, 0] = 14.0   # cold
    std[2, 2, 1]  = 1.5    # uncertain
    pm = detect_pockets(mean, std, setpoint=22.0)
    assert pm.hot.shape == (4, 3, 2)
    assert pm.hot[0, 0, 0] and pm.n_hot >= 1
    assert pm.cold[1, 1, 0] and pm.n_cold >= 1
    assert pm.uncertain[2, 2, 1] and pm.n_uncertain >= 1


def test_calibration_handles_3d_arrays():
    rng = np.random.default_rng(0)
    means  = [rng.normal(22.0, 1.0, size=(4, 3, 2)) for _ in range(5)]
    stds   = [np.full((4, 3, 2), 0.5)               for _ in range(5)]
    truths = [m + rng.normal(0, 0.5, size=(4, 3, 2)) for m in means]
    cal = calibration_over_episode(means, stds, truths)
    assert cal.nominal_levels.ndim == 1
    assert cal.empirical_coverage.shape == cal.nominal_levels.shape
    assert 0.0 <= cal.ECE <= 1.0
    assert cal.n_samples == 5 * 4 * 3 * 2


def test_gp_fits_simulator_observations_end_to_end():
    """One real simulator step + GP fit: sanity check the integration."""
    sim = RoomSimulator(seed=0)
    sr = sim.reset()
    gp = GPField(nx=sim.nx, ny=sim.ny, nz=sim.nz)
    gp.fit(sr.locs, sr.temps)
    mean, std = gp.predict()
    # Mean at observed sensors should be close to sensor reading.
    for (x, y, z), t in zip(sr.locs.astype(int), sr.temps):
        assert abs(mean[x, y, z] - t) < 3.0
    # Std should be positive somewhere away from sensors.
    assert std.max() > 0.0
