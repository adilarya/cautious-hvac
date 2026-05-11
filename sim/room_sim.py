"""Layer 1 -- 3D voxel thermal simulator.

Models a room as an (nx * ny * nz) voxel grid. Temperature evolves via
explicit 3D finite-difference diffusion (6-neighbour Laplacian) plus:
  - ceiling-mounted vent inflow (proportional to damper command),
  - localised occupancy heat sources distributed over a 3x3x3 neighbourhood,
  - stochastic outdoor boundary condition on all six exterior faces.

Design notes
------------
This is a 3D rewrite of the original 2D simulator. Two bugs from the
2D version are fixed here:

1. **Occupancy heat was dumped into a single voxel.** The intent (per the
   docstring of the 2D file) was to share heat across the local
   neighbourhood. We now distribute the 75 W per-person heat input
   uniformly across a 3x3x3 = 27-voxel neighbourhood.

2. **Effective diffusivity was set to the molecular value of air**
   (~1e-5 m^2/s), so heat barely moved one voxel per episode and the
   "no control" baseline produced enormous gradients indistinguishable
   from active control. We now sample alpha from a range typical of
   *turbulent* (well-stirred-room) effective diffusivity, ~3e-4 to
   1e-3 m^2/s. Explicit-FD stability constraint
       alpha * dt / h^2 < 1/(2D) = 1/6
   is satisfied for alpha_max=1e-3, dt=30 s, h=0.5 m
   (1e-3 * 30 / 0.25 = 0.12 < 0.167).

Stochastic parameters are drawn at reset() so each seed produces a
different physical realisation (different alpha, sensor noise, and
occupancy heat).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


# Air thermophysical constants (1 atm, ~20 deg C)
RHO_AIR = 1.2       # kg/m^3
CP_AIR  = 1005.0    # J/(kg*K)


@dataclass
class SensorReading:
    locs:  np.ndarray   # (S, 3) float -- voxel coordinates of sensors
    temps: np.ndarray   # (S,)   float -- noisy temperature readings (deg C)


@dataclass
class VentCommands:
    locs:  np.ndarray   # (V, 3) int   -- voxel of each vent (ceiling layer)
    flows: np.ndarray   # (V,)   float -- damper opening in [0, 1]


class RoomSimulator:
    """Stochastic 3D voxel thermal simulator.

    Parameters
    ----------
    nx, ny, nz   : voxel grid dimensions (default 10 x 8 x 5 = 400 cells
                   at 0.5 m resolution -> 5 m x 4 m x 2.5 m room).
    n_vents      : number of controllable ceiling vents.
    n_sensors    : number of temperature sensors (sparse).
    setpoint     : target comfort temperature (deg C).
    T_init       : initial uniform temperature (deg C).
    seed         : RNG seed -- controls layout and stochastic dynamics.
    dt           : simulation timestep (seconds).
    voxel_size   : edge length of one voxel (m).
    """

    def __init__(
        self,
        nx: int = 10, ny: int = 8, nz: int = 5,
        n_vents: int = 4, n_sensors: int = 8,
        setpoint: float = 22.0, T_init: float = 15.0,
        seed: int = 0, dt: float = 30.0,
        voxel_size: float = 0.5,
    ):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.n_vents = n_vents
        self.n_sensors = n_sensors
        self.setpoint = setpoint
        self.T_init = T_init
        self.dt = dt
        self.voxel_size = voxel_size
        self.seed = seed

        # Fixed layout per seed
        rng_layout = np.random.default_rng(seed)
        self.vent_locs = self._place_ceiling_vents(n_vents)
        self.sensor_locs = self._place_sensors(n_sensors, avoid=self.vent_locs)

        # Two occupancy spots near floor (interior, clamped to valid range)
        self.occupancy_locs = np.array([
            [max(1, nx // 4),     max(1, ny // 4),     1],
            [max(1, 3*nx // 4),   max(1, 3*ny // 4),   1],
        ], dtype=int)
        self.occupancy_locs = np.clip(
            self.occupancy_locs,
            [1, 1, 1],
            [nx - 2, ny - 2, nz - 2],
        )

        self.T = np.full((nx, ny, nz), T_init, dtype=float)
        self.t = 0
        self._rng = np.random.default_rng(seed + 1)

        # Stochastic physical params (resampled each reset)
        self.alpha: float = 0.0
        self.sensor_noise_std: float = 0.0
        self.occupancy_heat_K_per_step: float = 0.0
        self._resample_physics()

    # ----------------------------------------------------------------- API

    def reset(self) -> SensorReading:
        """Reset state, resample stochastic parameters, return first reading."""
        self._rng = np.random.default_rng(self.seed + 1)
        self._resample_physics()
        self.T = np.full((self.nx, self.ny, self.nz), self.T_init, dtype=float)
        self.t = 0
        return self._observe()

    def step(self, commands: VentCommands) -> SensorReading:
        """Advance one timestep and return sensor readings."""
        T = self.T.copy()

        # 3D diffusion: 6-neighbour Laplacian via finite differences.
        # np.roll provides periodic BCs; we overwrite exterior faces below.
        lap = (
            np.roll(T,  1, 0) + np.roll(T, -1, 0) +
            np.roll(T,  1, 1) + np.roll(T, -1, 1) +
            np.roll(T,  1, 2) + np.roll(T, -1, 2) -
            6.0 * T
        )
        T += self.alpha * self.dt * lap / (self.voxel_size ** 2)

        # Vent inflow: relax the ceiling cell (and the one below) toward T_supply.
        for (x, y, z), flow in zip(commands.locs.astype(int), commands.flows):
            T_supply = self.setpoint + self._rng.normal(0.0, 0.3)
            T[x, y, z] += flow * (T_supply - T[x, y, z]) * 0.5
            if z - 1 >= 0:
                T[x, y, z - 1] += flow * (T_supply - T[x, y, z - 1]) * 0.25

        # Occupancy heat sources -- BUG FIX: distribute over 27-cell neighbourhood.
        for (x, y, z) in self.occupancy_locs:
            x0, x1 = max(0, x - 1), min(self.nx, x + 2)
            y0, y1 = max(0, y - 1), min(self.ny, y + 2)
            z0, z1 = max(0, z - 1), min(self.nz, z + 2)
            T[x0:x1, y0:y1, z0:z1] += self.occupancy_heat_K_per_step

        # Stochastic outdoor boundary on all six exterior faces.
        T_out = (
            18.0
            + 4.0 * np.sin(2.0 * np.pi * self.t / (86400.0 / self.dt))
            + self._rng.normal(0.0, 0.5)
        )
        mix = 0.20
        T[ 0, :, :] = (1.0 - mix) * T[ 0, :, :] + mix * T_out
        T[-1, :, :] = (1.0 - mix) * T[-1, :, :] + mix * T_out
        T[:,  0, :] = (1.0 - mix) * T[:,  0, :] + mix * T_out
        T[:, -1, :] = (1.0 - mix) * T[:, -1, :] + mix * T_out
        T[:, :,  0] = (1.0 - mix) * T[:, :,  0] + mix * T_out
        T[:, :, -1] = (1.0 - mix) * T[:, :, -1] + mix * T_out

        self.T = T
        self.t += 1
        return self._observe()

    def ground_truth_field(self) -> np.ndarray:
        """Return the full (nx, ny, nz) temperature field."""
        return self.T.copy()

    def default_commands(self) -> VentCommands:
        """All vents at 50% -- the no-control baseline command."""
        return VentCommands(locs=self.vent_locs,
                            flows=np.full(self.n_vents, 0.5))

    # ----------------------------------------------------------- Internals

    def _resample_physics(self) -> None:
        """Resample stochastic physical parameters for a fresh episode."""
        # Effective turbulent diffusivity for a stirred room.
        # Stability check: alpha_max * dt / h^2 = 1e-3 * 30 / 0.25 = 0.12 < 1/6.
        self.alpha = self._rng.uniform(3e-4, 1e-3)
        self.sensor_noise_std = self._rng.uniform(0.1, 0.5)

        # Occupancy heat: per-person ~ N(75, 10) W, distributed across the
        # 27 voxels of a 3x3x3 neighbourhood. Per-cell K/step rate equals
        #   occ_W / (rho*cp*V_local) * dt
        # where V_local = 27 * voxel_size^3.
        occ_W = float(self._rng.normal(75.0, 10.0))
        V_local = 27.0 * self.voxel_size ** 3
        rho_cp_V_local = RHO_AIR * CP_AIR * V_local
        K_per_step = max(0.0, occ_W) / rho_cp_V_local * self.dt
        # Defensive clip in case the noise gives a wildly large person.
        self.occupancy_heat_K_per_step = float(np.clip(K_per_step, 0.0, 0.20))

    def _observe(self) -> SensorReading:
        """Return sensor readings: ground-truth voxel temp + Gaussian noise."""
        temps = np.array([
            self.T[x, y, z] + self._rng.normal(0.0, self.sensor_noise_std)
            for x, y, z in self.sensor_locs
        ], dtype=float)
        return SensorReading(locs=self.sensor_locs.astype(float), temps=temps)

    def _place_ceiling_vents(self, n: int) -> np.ndarray:
        """Place n vents on the ceiling layer (z = nz - 1) in a 2D grid."""
        z = self.nz - 1
        n_along_x = max(1, int(np.ceil(np.sqrt(n * self.nx / max(1, self.ny)))))
        n_along_y = max(1, int(np.ceil(n / n_along_x)))
        xs = np.linspace(1, self.nx - 2, n_along_x).astype(int)
        ys = np.linspace(1, self.ny - 2, n_along_y).astype(int)
        locs: List[List[int]] = []
        for x in xs:
            for y in ys:
                locs.append([int(x), int(y), int(z)])
                if len(locs) == n:
                    return np.array(locs, dtype=int)
        return np.array(locs[:n], dtype=int)

    def _place_sensors(self, n: int, avoid: np.ndarray) -> np.ndarray:
        """Stratified 3D placement: distribute sensors across z-levels."""
        avoid_set = {tuple(a) for a in avoid.tolist()}
        n_z = max(1, min(self.nz - 2, int(np.ceil(np.cbrt(n)))))
        zs = np.linspace(1, self.nz - 2, n_z).astype(int)
        per_z = int(np.ceil(n / len(zs)))
        n_along_x = max(1, int(np.ceil(np.sqrt(per_z))))
        n_along_y = max(1, int(np.ceil(per_z / n_along_x)))
        xs = np.linspace(1, self.nx - 2, n_along_x).astype(int)
        ys = np.linspace(1, self.ny - 2, n_along_y).astype(int)
        locs: List[Tuple[int, int, int]] = []
        for z in zs:
            for x in xs:
                for y in ys:
                    t = (int(x), int(y), int(z))
                    if t in avoid_set:
                        continue
                    locs.append(t)
                    if len(locs) == n:
                        return np.array(locs, dtype=int)
        return np.array(locs[:n], dtype=int)
