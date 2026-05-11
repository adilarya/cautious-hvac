"""Layer 4 -- Controllers (3D voxel grid).

Three controllers with identical interface:
    compute_action(mean_field, std_field, sim) -> VentCommands

NoControl         : all vents fixed at 50% (pure baseline).
ProportionalCtrl  : feedback proportional to local voxel-neighbourhood error.
MPCController     : SLSQP-solved cost with cautious-MPC uncertainty penalty.
"""
from __future__ import annotations
import os
import sys
from typing import Optional
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sim.room_sim import VentCommands, RoomSimulator   # noqa: E402


# ------------------------------------------------------------- helpers

def _local_mean(field: np.ndarray, x: int, y: int, z: int) -> float:
    """3x3x3 mean of `field` around (x, y, z), clipped to bounds."""
    nx, ny, nz = field.shape
    x0, x1 = max(0, x - 1), min(nx, x + 2)
    y0, y1 = max(0, y - 1), min(ny, y + 2)
    z0, z1 = max(0, z - 1), min(nz, z + 2)
    return float(field[x0:x1, y0:y1, z0:z1].mean())


# ------------------------------------------------------------- controllers

class NoControl:
    """Baseline: fixed 50% flow regardless of temperature field."""
    name = "NoControl"

    def compute_action(self, mean_field: np.ndarray, std_field: np.ndarray,
                       sim: RoomSimulator) -> VentCommands:
        return VentCommands(locs=sim.vent_locs,
                            flows=np.full(sim.n_vents, 0.5))

    def reset(self) -> None: pass


class ProportionalCtrl:
    """Proportional controller with integrator-like flow accumulation.

    Each vent looks at its local 3x3x3 neighbourhood, computes the error
    (setpoint - local mean), and steps its flow toward setpoint by k_p *
    error. Flows are clipped to [0, 1]. This is the standard cheap baseline."""
    name = "Proportional"

    def __init__(self, k_p: float = 0.12):
        self.k_p = k_p
        self._flows: Optional[np.ndarray] = None

    def compute_action(self, mean_field: np.ndarray, std_field: np.ndarray,
                       sim: RoomSimulator) -> VentCommands:
        n = sim.n_vents
        if self._flows is None:
            self._flows = np.full(n, 0.5)
        for i, (x, y, z) in enumerate(sim.vent_locs):
            local_T = _local_mean(mean_field, int(x), int(y), int(z))
            err = sim.setpoint - local_T
            self._flows[i] = np.clip(self._flows[i] + self.k_p * err, 0.0, 1.0)
        return VentCommands(locs=sim.vent_locs, flows=self._flows.copy())

    def reset(self) -> None:
        self._flows = None


class MPCController:
    """Model Predictive Control with a cautious uncertainty penalty.

    Each step we solve a single-step constrained optimisation:

        J(u) = Q * mean((T_pred - T_sp)^2)
             + lam * mean(std_field)        <- cautious term
             + R * sum(u^2)                 <- effort penalty

    subject to 0 <= u_k <= 1 and sum(u_k) <= U_total.

    T_pred is a one-step linearised forward prediction: each vent nudges
    its 3x3x3 neighbourhood toward setpoint proportional to u_k.
    """
    name = "MPC"

    def __init__(self, Q: float = 1.0, R: float = 0.05,
                 lam: float = 0.3, U_total: Optional[float] = None,
                 nudge_strength: float = 0.35):
        self.Q = Q
        self.R = R
        self.lam = lam
        self._U_total = U_total
        self._nudge = nudge_strength
        self._last_u: Optional[np.ndarray] = None

    def compute_action(self, mean_field: np.ndarray, std_field: np.ndarray,
                       sim: RoomSimulator) -> VentCommands:
        n = sim.n_vents
        if self._last_u is None:
            self._last_u = np.full(n, 0.5)
        U_total = self._U_total if self._U_total is not None else 0.8 * n
        sp = sim.setpoint
        nx, ny, nz = mean_field.shape

        # Precompute index slices for each vent's 3x3x3 neighbourhood.
        slices = []
        for (x, y, z) in sim.vent_locs:
            x, y, z = int(x), int(y), int(z)
            slices.append((
                slice(max(0, x - 1), min(nx, x + 2)),
                slice(max(0, y - 1), min(ny, y + 2)),
                slice(max(0, z - 1), min(nz, z + 2)),
            ))

        def cost(u: np.ndarray) -> float:
            T_pred = mean_field.copy()
            for i, sl in enumerate(slices):
                T_pred[sl] += u[i] * (sp - mean_field[sl]) * self._nudge
            rmse_sq = float(np.mean((T_pred - sp) ** 2))
            unc     = float(np.mean(std_field))
            effort  = float(np.sum(u ** 2))
            return self.Q * rmse_sq + self.lam * unc + self.R * effort

        bounds = [(0.0, 1.0)] * n
        cons   = [{"type": "ineq", "fun": lambda u: U_total - float(np.sum(u))}]
        res = minimize(cost, self._last_u, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 80, "ftol": 1e-5})
        u = np.clip(res.x, 0.0, 1.0)
        self._last_u = u
        return VentCommands(locs=sim.vent_locs, flows=u)

    def reset(self) -> None:
        self._last_u = None
