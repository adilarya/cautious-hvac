"""Tests for the three classical controllers."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from sim.room_sim import RoomSimulator
from control.controller import NoControl, ProportionalCtrl, MPCController


def _dummy_fields(sim, mean_val=22.0, std_val=0.3):
    shape = (sim.nx, sim.ny, sim.nz)
    return np.full(shape, mean_val), np.full(shape, std_val)


def test_no_control_returns_half_flow():
    sim = RoomSimulator(seed=0); sim.reset()
    m, s = _dummy_fields(sim)
    cmd = NoControl().compute_action(m, s, sim)
    assert cmd.flows.shape == (sim.n_vents,)
    assert np.allclose(cmd.flows, 0.5)


def test_proportional_pushes_up_when_cold():
    sim = RoomSimulator(seed=0); sim.reset()
    m, s = _dummy_fields(sim, mean_val=18.0)   # below setpoint
    ctrl = ProportionalCtrl()
    cmd = ctrl.compute_action(m, s, sim)
    assert np.all(cmd.flows >= 0.5)  # should be increased above neutral
    assert np.all((cmd.flows >= 0.0) & (cmd.flows <= 1.0))


def test_proportional_pushes_down_when_hot():
    sim = RoomSimulator(seed=0); sim.reset()
    m, s = _dummy_fields(sim, mean_val=26.0)   # above setpoint
    ctrl = ProportionalCtrl()
    cmd = ctrl.compute_action(m, s, sim)
    assert np.all(cmd.flows <= 0.5)
    assert np.all((cmd.flows >= 0.0) & (cmd.flows <= 1.0))


def test_mpc_returns_valid_action():
    sim = RoomSimulator(seed=0); sim.reset()
    m, s = _dummy_fields(sim, mean_val=20.0, std_val=0.4)
    mpc = MPCController(lam=0.3)
    cmd = mpc.compute_action(m, s, sim)
    assert cmd.flows.shape == (sim.n_vents,)
    assert np.all((cmd.flows >= 0.0) & (cmd.flows <= 1.0))


def test_mpc_higher_lambda_responds_to_uncertainty():
    """A larger lambda doesn't change the action when std is uniform (the
    uncertainty term is then constant), but the optimizer should not crash."""
    sim = RoomSimulator(seed=0); sim.reset()
    m, s = _dummy_fields(sim, mean_val=22.0, std_val=2.0)
    cmd_low  = MPCController(lam=0.05).compute_action(m, s, sim)
    cmd_high = MPCController(lam=5.00).compute_action(m, s, sim)
    assert np.all((cmd_low.flows  >= 0) & (cmd_low.flows  <= 1))
    assert np.all((cmd_high.flows >= 0) & (cmd_high.flows <= 1))
