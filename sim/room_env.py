"""Gymnasium environment wrapping the 3D room simulator for RL training.

Observation modes:
  fast_obs=True  (default for training): obs = [sensor_temps, step_fraction]
                 No GP fit per step -> ~100x faster rollouts.
  fast_obs=False (for evaluation):       obs = flattened GP posterior
                                         [mean_field, std_field] (2*nx*ny*nz).

Action: continuous Box in [0, 1]^n_vents -- vent damper openings.
Reward: -(reward_uniformity_w * RMSE + reward_lambda * mean_effort).
"""
from __future__ import annotations
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sim.room_sim import RoomSimulator, VentCommands   # noqa: E402
from model.gp_field import GPField                       # noqa: E402


class RoomEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 nx: int = 10, ny: int = 8, nz: int = 5,
                 n_vents: int = 4, n_sensors: int = 8,
                 setpoint: float = 22.0,
                 episode_length: int = 100,
                 reward_lambda: float = 0.1,
                 reward_uniformity_w: float = 1.0,
                 seed: int = 0,
                 fast_obs: bool = True):
        super().__init__()
        self.nx, self.ny, self.nz = nx, ny, nz
        self.n_vents = n_vents
        self.n_sensors = n_sensors
        self.episode_length = episode_length
        self.reward_lambda = reward_lambda
        self.reward_uniformity_w = reward_uniformity_w
        self.setpoint = setpoint
        self._seed = seed
        self.fast_obs = fast_obs

        self.sim = RoomSimulator(nx=nx, ny=ny, nz=nz,
                                 n_vents=n_vents, n_sensors=n_sensors,
                                 setpoint=setpoint, seed=seed)
        self.gp = GPField(nx=nx, ny=ny, nz=nz)

        obs_dim = (n_sensors + 1) if fast_obs else (2 * nx * ny * nz)
        self.observation_space = spaces.Box(
            low=np.full(obs_dim, -30.0, dtype=np.float32),
            high=np.full(obs_dim,  60.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.zeros(n_vents, dtype=np.float32),
            high=np.ones(n_vents, dtype=np.float32),
            dtype=np.float32,
        )
        self._step_count = 0
        self._last_sensor = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self.sim = RoomSimulator(nx=self.nx, ny=self.ny, nz=self.nz,
                                     n_vents=self.n_vents,
                                     n_sensors=self.n_sensors,
                                     setpoint=self.setpoint, seed=seed)
        self._step_count = 0
        sr = self.sim.reset()
        self._last_sensor = sr
        return self._make_obs(sr), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), 0.0, 1.0)
        sr = self.sim.step(VentCommands(locs=self.sim.vent_locs,
                                        flows=action.astype(float)))
        self._last_sensor = sr
        obs = self._make_obs(sr)
        truth = self.sim.ground_truth_field()
        rmse = float(np.sqrt(np.mean((truth - self.setpoint) ** 2)))
        effort = float(np.mean(action))
        reward = -(self.reward_uniformity_w * rmse
                   + self.reward_lambda * effort)
        self._step_count += 1
        truncated = self._step_count >= self.episode_length
        return obs, reward, False, truncated, {"rmse": rmse, "effort": effort}

    def _make_obs(self, sensor_reading):
        if self.fast_obs:
            frac = np.array([self._step_count / self.episode_length], dtype=np.float32)
            return np.concatenate([sensor_reading.temps.astype(np.float32), frac])
        self.gp.fit(sensor_reading.locs, sensor_reading.temps)
        m, s = self.gp.predict()
        return np.concatenate([m.ravel(), s.ravel()]).astype(np.float32)
