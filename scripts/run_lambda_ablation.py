"""Stage 1 -- MPC cautious-term ablation.

Sweeps the cautious-MPC uncertainty weight lambda over a small grid,
multiple seeds per cell. Writes per-(lambda, seed) results to a CSV and
a 4-panel summary figure (ECE, RMSE, effort, GP MAE vs lambda).

The headline run reports ECE = 0.015 for MPC at lambda = 0.3 versus
0.023 for Proportional and 0.033 for NoControl. The question this
ablation answers: is the calibration win due to the cautious term
itself, or to MPC structure (its constrained optimisation, neighbourhood
prediction) regardless of lambda?

If ECE at lambda = 0 (uncautious MPC) climbs to >= 0.023, the cautious
term is responsible. If ECE is flat across the lambda range, MPC
structure alone explains the win and the framing needs revision.

Usage:
    python scripts/run_lambda_ablation.py --lambdas 0,0.1,0.3,1,3 \\
        --seeds 0,1,2,3,4 --episode-len 100
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.room_sim import RoomSimulator
from model.gp_field import GPField
from model.uq import calibration_over_episode
from control.controller import MPCController
from eval.metrics import run_episode


# Headline-run ECE values for reference lines on the ECE panel.
# From results/calibration.png reliability legend, pooled across 5 seeds.
# Note: pooled ECE is typically lower than per-seed mean ECE because
# pooling absorbs seed-to-seed variability into a single estimator.
HEADLINE_ECE = {
    "NoControl":     0.045,
    "Proportional":  0.031,
    "PPO":           0.035,
}


def aggregate(rows, metric, lambdas):
    """Return (means, stds) of `metric` per lambda, ordered by `lambdas`."""
    means, stds = [], []
    for lam in lambdas:
        vals = [r[metric] for r in rows if r["lambda"] == lam]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
    return np.array(means), np.array(stds)


def main():
    ap = argparse.ArgumentParser(description="MPC cautious-term lambda ablation.")
    ap.add_argument("--lambdas", type=str, default="0.0,0.1,0.3,1.0,3.0",
                    help="Comma-separated cautious-term weights to sweep")
    ap.add_argument("--seeds",   type=str, default="0,1,2,3,4")
    ap.add_argument("--episode-len", type=int, default=100)
    ap.add_argument("--nx", type=int, default=10)
    ap.add_argument("--ny", type=int, default=8)
    ap.add_argument("--nz", type=int, default=5)
    ap.add_argument("--n-vents",   type=int, default=4)
    ap.add_argument("--n-sensors", type=int, default=8)
    ap.add_argument("--setpoint",  type=float, default=22.0)
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "lambda_ablation.csv"
    png_path = out / "lambda_ablation.png"

    lambdas = [float(x) for x in args.lambdas.split(",")]
    seeds   = [int(s)   for s in args.seeds.split(",")]

    print(f"MPC cautious-term ablation")
    print(f"  lambdas      : {lambdas}")
    print(f"  seeds        : {seeds}")
    print(f"  episode_len  : {args.episode_len}")
    print(f"  grid         : {args.nx}x{args.ny}x{args.nz} = {args.nx*args.ny*args.nz} voxels")
    print()

    rows = []
    n_runs = len(lambdas) * len(seeds)
    counter = 0
    for lam in lambdas:
        for seed in seeds:
            counter += 1
            print(f"  [{counter:2d}/{n_runs}] lambda={lam:>5.2f}  seed={seed}  ",
                  end="", flush=True)
            sim  = RoomSimulator(nx=args.nx, ny=args.ny, nz=args.nz,
                                 n_vents=args.n_vents, n_sensors=args.n_sensors,
                                 setpoint=args.setpoint, seed=seed)
            gp   = GPField(nx=args.nx, ny=args.ny, nz=args.nz)
            ctrl = MPCController(lam=lam)
            rec  = run_episode(sim, gp, ctrl, args.episode_len, store_fields=True)
            s    = rec.summary()
            cal  = calibration_over_episode(rec.mean_history, rec.std_history,
                                            rec.truth_history)
            rows.append({
                "lambda":          lam,
                "seed":            seed,
                "rmse_mean":       s["rmse_mean"],
                "std_t_mean":      s["uniformity_mean"],
                "mae_mean":        s["mae_mean"],
                "effort_mean":     s["effort_mean"],
                "ci_cov_mean":     s["ci_coverage_mean"],
                "ece":             cal.ECE,
                "signed_bias":     cal.signed_bias,
            })
            print(f"RMSE={s['rmse_mean']:.3f}  effort={s['effort_mean']:.3f}  "
                  f"ECE={cal.ECE:.4f}  bias={cal.signed_bias:+.4f}")

    # ---------------------------------------------------------------- CSV
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n-> {csv_path}")

    # -------------------------------------------------------------- Figure
    metrics = [
        ("ece",         "Expected Calibration Error",  True ),
        ("rmse_mean",   "Time-mean RMSE vs setpoint (C)", False),
        ("effort_mean", "Mean control effort",          False),
        ("mae_mean",    "GP MAE vs ground truth (C)",   False),
    ]
    xs = np.arange(len(lambdas))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for (key, label, highlight), ax in zip(metrics, axes.ravel()):
        m, s = aggregate(rows, key, lambdas)
        color = "tab:green" if highlight else "tab:blue"
        ax.errorbar(xs, m, yerr=s, fmt="o-", capsize=4, linewidth=1.8,
                    markersize=8, color=color, label="MPC (this sweep)")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(l) for l in lambdas])
        ax.set_xlabel(r"cautious-term weight $\lambda$")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.3)
        if key == "ece":
            for name, val in HEADLINE_ECE.items():
                ax.axhline(val, linestyle="--", linewidth=0.9,
                           color="gray", alpha=0.7)
                ax.text(xs[-1] + 0.05, val, name, fontsize=8,
                        va="center", color="gray")
            ax.legend(loc="best", fontsize=9)
    fig.suptitle(
        f"MPC cautious-term ablation across {len(seeds)} seeds  "
        f"(error bars = std across seeds)",
        y=1.00, fontsize=13)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"-> {png_path}")

    # -------------------------------------------------------------- Stdout summary
    print("\n" + "=" * 64)
    print("Per-lambda summary (mean +/- std across seeds)")
    print("=" * 64)
    print(f"{'lambda':>8}  {'ECE':>14}  {'RMSE':>14}  {'effort':>14}")
    for lam in lambdas:
        e_m, e_s = aggregate(rows, "ece",         [lam])
        r_m, r_s = aggregate(rows, "rmse_mean",   [lam])
        u_m, u_s = aggregate(rows, "effort_mean", [lam])
        print(f"{lam:>8.2f}  {e_m[0]:>6.4f}+/-{e_s[0]:.4f}  "
              f"{r_m[0]:>5.3f}+/-{r_s[0]:.3f}  "
              f"{u_m[0]:>5.3f}+/-{u_s[0]:.3f}")


if __name__ == "__main__":
    main()
