"""Layer 5 -- Evaluation: episode metrics + multi-seed runner."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List
import numpy as np


@dataclass
class EpisodeRecord:
    """Per-step recorded metrics over one episode."""
    uniformity:   List[float] = field(default_factory=list)   # std(T_true)
    rmse:         List[float] = field(default_factory=list)   # RMSE vs setpoint
    mae:          List[float] = field(default_factory=list)   # |GP mean - truth|
    effort:       List[float] = field(default_factory=list)   # mean vent flow
    ci_coverage:  List[float] = field(default_factory=list)   # 90% CI coverage
    # Per-step recorded fields for downstream calibration/plot:
    mean_history:  List[np.ndarray] = field(default_factory=list)
    std_history:   List[np.ndarray] = field(default_factory=list)
    truth_history: List[np.ndarray] = field(default_factory=list)

    def record(self, mean_field: np.ndarray, std_field: np.ndarray,
               truth_field: np.ndarray, flows: np.ndarray, setpoint: float,
               store_fields: bool = False) -> None:
        self.uniformity.append(float(np.std(truth_field)))
        self.rmse.append(float(np.sqrt(np.mean((truth_field - setpoint) ** 2))))
        self.mae.append(float(np.mean(np.abs(mean_field - truth_field))))
        self.effort.append(float(np.mean(flows)))
        z = 1.645
        ci_lo = mean_field - z * std_field
        ci_hi = mean_field + z * std_field
        cov = float(np.mean((truth_field >= ci_lo) & (truth_field <= ci_hi)))
        self.ci_coverage.append(cov)
        if store_fields:
            self.mean_history.append(mean_field.copy())
            self.std_history.append(std_field.copy())
            self.truth_history.append(truth_field.copy())

    def summary(self) -> dict:
        return {
            "uniformity_mean":  float(np.mean(self.uniformity)),
            "rmse_mean":        float(np.mean(self.rmse)),
            "mae_mean":         float(np.mean(self.mae)),
            "effort_mean":      float(np.mean(self.effort)),
            "ci_coverage_mean": float(np.mean(self.ci_coverage)),
            "final_rmse":       float(self.rmse[-1]) if self.rmse else float("nan"),
            "final_uniformity": float(self.uniformity[-1]) if self.uniformity else float("nan"),
        }


def run_episode(sim, gp, controller, episode_length: int = 100,
                store_fields: bool = False) -> EpisodeRecord:
    """Run one episode end-to-end and return its recorded metrics."""
    record = EpisodeRecord()
    obs = sim.reset()
    if hasattr(controller, "reset"):
        controller.reset()
    for _ in range(episode_length):
        gp.fit(obs.locs, obs.temps)
        mean_field, std_field = gp.predict()
        commands = controller.compute_action(mean_field, std_field, sim)
        obs = sim.step(commands)
        record.record(mean_field, std_field, sim.ground_truth_field(),
                      commands.flows, sim.setpoint,
                      store_fields=store_fields)
    return record


def multi_seed_run(make_sim_fn: Callable, make_controller_fn: Callable,
                   gp_cls,
                   seeds: List[int], episode_length: int = 100,
                   nx: int = 10, ny: int = 8, nz: int = 5) -> dict:
    """Run multiple seeds, return mean +/- std for each metric."""
    all_summaries = []
    for seed in seeds:
        sim = make_sim_fn(seed)
        gp = gp_cls(nx=nx, ny=ny, nz=nz)
        controller = make_controller_fn(sim)
        record = run_episode(sim, gp, controller, episode_length)
        all_summaries.append(record.summary())

    keys = all_summaries[0].keys()
    return {
        k: {
            "mean": float(np.mean([s[k] for s in all_summaries])),
            "std":  float(np.std([s[k] for s in all_summaries])),
        }
        for k in keys
    }
