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


def test_mpc_higher_lambda_runs_without_crash():
    """Sanity: any lambda value produces a valid action in [0, 1]."""
    sim = RoomSimulator(seed=0); sim.reset()
    m, s = _dummy_fields(sim, mean_val=22.0, std_val=2.0)
    for lam in (0.0, 0.05, 5.00):
        cmd = MPCController(lam=lam).compute_action(m, s, sim)
        assert np.all((cmd.flows >= 0.0) & (cmd.flows <= 1.0))


def test_mpc_lambda_affects_action_when_std_nonuniform():
    """Regression test for the no-op cautious-term bug.

    With spatially non-uniform std, the cost J(u) under the corrected
    uncertainty-weighted MPC formulation depends on lambda (the weights
    1 + lam * std_field are voxel-dependent). The optimal action should
    therefore differ between lambda = 0 and lambda > 0. The original
    formulation used `lam * mean(std_field)` which was a constant in u
    and did NOT change the argmin -- a pathology caught by the
    scripts/run_lambda_ablation.py sweep.
    """
    sim = RoomSimulator(seed=0); sim.reset()
    m, _ = _dummy_fields(sim, mean_val=20.0)
    rng = np.random.default_rng(0)
    s_nonuniform = rng.uniform(0.1, 1.5, size=(sim.nx, sim.ny, sim.nz))
    cmd0 = MPCController(lam=0.0).compute_action(m, s_nonuniform, sim)
    cmd2 = MPCController(lam=2.0).compute_action(m, s_nonuniform, sim)
    assert not np.allclose(cmd0.flows, cmd2.flows, atol=1e-4), (
        f"lambda=0 and lambda=2 produced identical actions "
        f"({cmd0.flows} vs {cmd2.flows}); cautious term is a no-op"
    )
