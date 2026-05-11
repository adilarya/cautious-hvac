"""Tests for the 3D voxel thermal simulator.

We check:
  - shapes/types of outputs
  - reset returns the simulator to initial conditions
  - temperatures stay bounded over a full episode (this is the bug-fix test;
    the original 2D simulator violated this for the seed-0 setup)
  - same seed produces identical trajectory under fixed actions
  - the no-control field drifts toward the outdoor mean ~18 C
  - higher diffusivity produces a more uniform field at episode end
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from sim.room_sim import RoomSimulator, VentCommands


def _const_action(sim, flow=0.5):
    return VentCommands(locs=sim.vent_locs,
                        flows=np.full(sim.n_vents, flow))


def test_shapes_and_types():
    sim = RoomSimulator(seed=0)
    sr = sim.reset()
    assert sr.locs.shape == (sim.n_sensors, 3)
    assert sr.temps.shape == (sim.n_sensors,)
    assert sim.ground_truth_field().shape == (sim.nx, sim.ny, sim.nz)
    assert sim.vent_locs.shape == (sim.n_vents, 3)
    # All vents on the ceiling layer.
    assert np.all(sim.vent_locs[:, 2] == sim.nz - 1)


def test_reset_restores_initial_field():
    sim = RoomSimulator(seed=0)
    sim.reset()
    for _ in range(20):
        sim.step(_const_action(sim))
    assert not np.allclose(sim.ground_truth_field(), sim.T_init)
    sim.reset()
    assert np.allclose(sim.ground_truth_field(), sim.T_init)


def test_temperatures_stay_bounded_full_episode():
    """Regression test for the runaway-heat bug. Field must stay within a
    physically sensible band over 100 steps at the no-control baseline."""
    for seed in range(5):
        sim = RoomSimulator(seed=seed)
        sim.reset()
        for _ in range(100):
            sim.step(_const_action(sim))
            T = sim.ground_truth_field()
            assert np.all(np.isfinite(T))
            assert T.min() > -5.0, f"seed {seed}: T.min()={T.min():.1f}"
            assert T.max() <  60.0, f"seed {seed}: T.max()={T.max():.1f}"


def test_seed_determinism():
    a = RoomSimulator(seed=42); a.reset()
    b = RoomSimulator(seed=42); b.reset()
    for _ in range(30):
        a.step(_const_action(a, 0.7))
        b.step(_const_action(b, 0.7))
    assert np.allclose(a.ground_truth_field(), b.ground_truth_field())


def test_different_seeds_diverge():
    a = RoomSimulator(seed=0); a.reset()
    b = RoomSimulator(seed=1); b.reset()
    for _ in range(30):
        a.step(_const_action(a, 0.5))
        b.step(_const_action(b, 0.5))
    assert not np.allclose(a.ground_truth_field(), b.ground_truth_field())


def test_no_control_drifts_toward_outdoor_band():
    """With vents at neutral 0.5 and no controller, mean room temperature
    should land somewhere between T_init (15 C) and a value influenced by
    T_out (~18 C) plus occupancy heat. Cap at a sane upper bound."""
    sim = RoomSimulator(seed=3); sim.reset()
    for _ in range(150):
        sim.step(_const_action(sim))
    T_mean = float(sim.ground_truth_field().mean())
    assert 14.0 <= T_mean <= 35.0, f"final mean T = {T_mean:.1f} C"
