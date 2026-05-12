"""Stage 5 -- sensor-count sensitivity sweep.

For each sensor count S in {4, 8, 16}, evaluates the three classical
controllers (NoControl, Proportional, MPC) across 5 seeds. PPO is
omitted because Sec.~5.7 shows PPO at our training budget is
initialisation-dominated regardless of reward shape, and re-training
PPO at every S to extract sensor-count information would not be
interpretable.

Connects directly to the proposal's stated sensor-placement motivation
and to the Krause et al.~2008 sensor-placement literature cited as
Future Work.

Outputs:
  results/sensor_sweep.csv      per-(S, seed, controller) row
  results/sensor_sweep.png      3-panel summary (GP MAE, ECE, RMSE vs S)

Usage:
    python scripts/run_sensor_sweep.py
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.room_sim import RoomSimulator
from model.gp_field import GPField
from model.uq import calibration_over_episode
from control.controller import NoControl, ProportionalCtrl, MPCController
from eval.metrics import run_episode


CONTROLLERS = [
    ("NoControl",    lambda sim, args: NoControl()),
    ("Proportional", lambda sim, args: ProportionalCtrl()),
    ("MPC",          lambda sim, args: MPCController(lam=args.lambda_unc)),
]


def main():
    ap = argparse.ArgumentParser(
        description="Sensor-count sensitivity sweep across S and controller.")
    ap.add_argument("--n-sensors-list", type=str, default="4,8,16")
    ap.add_argument("--seeds",          type=str, default="0,1,2,3,4")
    ap.add_argument("--episode-len",    type=int, default=100)
    ap.add_argument("--nx", type=int, default=10)
    ap.add_argument("--ny", type=int, default=8)
    ap.add_argument("--nz", type=int, default=5)
    ap.add_argument("--n-vents",     type=int,   default=4)
    ap.add_argument("--setpoint",    type=float, default=22.0)
    ap.add_argument("--lambda-unc",  type=float, default=0.3)
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    n_sensors_list = [int(x) for x in args.n_sensors_list.split(",")]
    seeds          = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_runs = len(n_sensors_list) * len(seeds) * len(CONTROLLERS)
    print(f"Sensor-count sweep")
    print(f"  S values     : {n_sensors_list}")
    print(f"  seeds        : {seeds}")
    print(f"  controllers  : {[c for c, _ in CONTROLLERS]}")
    print(f"  total runs   : {n_runs}\n")

    rows = []
    counter = 0
    for S in n_sensors_list:
        for seed in seeds:
            for name, factory in CONTROLLERS:
                counter += 1
                sim = RoomSimulator(nx=args.nx, ny=args.ny, nz=args.nz,
                                    n_vents=args.n_vents, n_sensors=S,
                                    setpoint=args.setpoint, seed=seed)
                gp   = GPField(nx=args.nx, ny=args.ny, nz=args.nz)
                ctrl = factory(sim, args)
                rec  = run_episode(sim, gp, ctrl, args.episode_len,
                                   store_fields=True)
                s   = rec.summary()
                cal = calibration_over_episode(rec.mean_history,
                                               rec.std_history,
                                               rec.truth_history)
                rows.append({
                    "S": S, "seed": seed, "controller": name,
                    "rmse_mean":    s["rmse_mean"],
                    "effort_mean":  s["effort_mean"],
                    "mae_mean":     s["mae_mean"],
                    "uniform_mean": s["uniformity_mean"],
                    "ece":          cal.ECE,
                })
                print(f"  [{counter:>2d}/{n_runs}] S={S:>2d} seed={seed} "
                      f"{name:<13} GP_MAE={s['mae_mean']:.3f}  "
                      f"ECE={cal.ECE:.4f}  RMSE={s['rmse_mean']:.3f}")

    # -------------------------------------------------------------- CSV
    csv_path = out / "sensor_sweep.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n-> {csv_path}")

    # ----------------------------------------------------------- Figure
    metrics = [
        ("mae_mean",  "GP MAE vs truth (C)"),
        ("ece",       "Expected Calibration Error"),
        ("rmse_mean", "RMSE vs setpoint (C)"),
    ]
    xs = np.arange(len(n_sensors_list))
    colors = {"NoControl":    "tab:blue",
              "Proportional": "tab:orange",
              "MPC":          "tab:green"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for (key, label), ax in zip(metrics, axes):
        for name, _ in CONTROLLERS:
            means, stds = [], []
            for S in n_sensors_list:
                vals = [r[key] for r in rows
                        if r["S"] == S and r["controller"] == name]
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
            ax.errorbar(xs, means, yerr=stds, fmt="o-",
                        capsize=4, linewidth=1.6, markersize=7,
                        color=colors[name], label=name)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(s) for s in n_sensors_list])
        ax.set_xlabel(r"sensor count $S$")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")
    fig.suptitle(
        f"Sensor-count sensitivity  "
        f"({len(seeds)} seeds per cell, episodes of {args.episode_len} steps)",
        fontsize=12, y=1.02)
    fig.tight_layout()
    png_path = out / "sensor_sweep.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"-> {png_path}")

    # ---------------------------------------------------- Stdout table
    print("\n" + "=" * 78)
    print(f"{'S':>3}  {'controller':<14}  {'GP MAE':>14}  {'ECE':>14}  {'RMSE':>14}")
    print("-" * 78)
    for S in n_sensors_list:
        for name, _ in CONTROLLERS:
            sub = [r for r in rows if r["S"] == S and r["controller"] == name]
            mae  = [r["mae_mean"]  for r in sub]
            ece  = [r["ece"]       for r in sub]
            rmse = [r["rmse_mean"] for r in sub]
            print(f"{S:>3}  {name:<14}  "
                  f"{np.mean(mae):>6.3f}+/-{np.std(mae):.3f}  "
                  f"{np.mean(ece):>6.4f}+/-{np.std(ece):.4f}  "
                  f"{np.mean(rmse):>6.3f}+/-{np.std(rmse):.3f}")


if __name__ == "__main__":
    main()
