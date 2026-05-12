"""Stage 3 -- PPO reward-shaping sweep.

Trains PPO at three values of the effort-penalty weight lambda_e and
evaluates each on common eval seeds. The headline run uses lambda_e=0.1
and finds the policy converges to zero effort. This script tests whether
lowering lambda_e moves PPO into an effort-spending regime where it
actually performs control, supporting the paper's reframing of PPO from
"failure mode" to "diagnosed reward-shape sensitivity."

Independent variable: lambda_e in {0.0, 0.01, 0.1}.
Train seeds per cell: 3.       (9 PPO trainings total)
Eval seeds per trained model: 5. (45 evaluations total)
Timesteps per training: 50k.

Outputs:
  results/ppo_sweep.csv     per-(lambda_e, train_seed, eval_seed) row
  results/ppo_sweep.png     3-panel summary (effort, RMSE, ECE vs lambda_e)

Usage:
    python scripts/run_ppo_reward_sweep.py
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from sim.room_env import RoomEnv
from model.uq import calibration_over_episode
from eval.metrics import EpisodeRecord


def train_one(lambda_e: float, train_seed: int, args) -> PPO:
    """Train a single PPO model at this (lambda_e, train_seed)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = RoomEnv(nx=args.nx, ny=args.ny, nz=args.nz,
                  n_vents=args.n_vents, n_sensors=args.n_sensors,
                  setpoint=args.setpoint,
                  episode_length=args.episode_len,
                  reward_lambda=lambda_e,
                  reward_uniformity_w=1.0,
                  seed=train_seed)
    model = PPO("MlpPolicy", env, verbose=0, seed=train_seed, device=device)
    model.learn(total_timesteps=args.timesteps)
    return model


def eval_one(model: PPO, eval_seed: int, lambda_e: float, args) -> dict:
    """Evaluate a trained policy on a single eval seed."""
    env = RoomEnv(nx=args.nx, ny=args.ny, nz=args.nz,
                  n_vents=args.n_vents, n_sensors=args.n_sensors,
                  setpoint=args.setpoint,
                  episode_length=args.episode_len,
                  reward_lambda=lambda_e,
                  seed=eval_seed)
    obs, _ = env.reset(seed=eval_seed)
    rec = EpisodeRecord()
    for _ in range(args.episode_len):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        sr = env._last_sensor
        env.gp.fit(sr.locs, sr.temps)
        mean_field, std_field = env.gp.predict()
        truth = env.sim.ground_truth_field()
        rec.record(mean_field, std_field, truth, action,
                   args.setpoint, store_fields=True)
        if term or trunc:
            break
    s = rec.summary()
    cal = calibration_over_episode(rec.mean_history, rec.std_history,
                                   rec.truth_history)
    return {
        "rmse_mean":   s["rmse_mean"],
        "effort_mean": s["effort_mean"],
        "mae_mean":    s["mae_mean"],
        "uniform_mean": s["uniformity_mean"],
        "ece":         cal.ECE,
    }


def main():
    ap = argparse.ArgumentParser(
        description="PPO effort-penalty (lambda_e) sweep.")
    ap.add_argument("--lambda-es",   type=str, default="0.0,0.01,0.1")
    ap.add_argument("--train-seeds", type=str, default="0,1,2")
    ap.add_argument("--eval-seeds",  type=str, default="0,1,2,3,4")
    ap.add_argument("--timesteps",   type=int, default=50_000)
    ap.add_argument("--episode-len", type=int, default=100)
    ap.add_argument("--nx", type=int, default=10)
    ap.add_argument("--ny", type=int, default=8)
    ap.add_argument("--nz", type=int, default=5)
    ap.add_argument("--n-vents",   type=int, default=4)
    ap.add_argument("--n-sensors", type=int, default=8)
    ap.add_argument("--setpoint",  type=float, default=22.0)
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    lambda_es   = [float(x) for x in args.lambda_es.split(",")]
    train_seeds = [int(x)   for x in args.train_seeds.split(",")]
    eval_seeds  = [int(x)   for x in args.eval_seeds.split(",")]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "ppo_sweep.csv"
    png_path = out / "ppo_sweep.png"

    n_trainings = len(lambda_es) * len(train_seeds)
    print(f"PPO reward-shaping sweep")
    print(f"  lambda_e values  : {lambda_es}")
    print(f"  train seeds      : {train_seeds}  ({n_trainings} PPO trainings)")
    print(f"  eval seeds       : {eval_seeds}")
    print(f"  timesteps/train  : {args.timesteps:,}")
    print(f"  device           : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()

    rows = []
    counter = 0
    for lam_e in lambda_es:
        for train_seed in train_seeds:
            counter += 1
            print(f"[{counter}/{n_trainings}] training lambda_e={lam_e}  "
                  f"train_seed={train_seed} ...", flush=True)
            model = train_one(lam_e, train_seed, args)
            for es in eval_seeds:
                metrics = eval_one(model, es, lam_e, args)
                rows.append({
                    "lambda_e":    lam_e,
                    "train_seed":  train_seed,
                    "eval_seed":   es,
                    **metrics,
                })
                print(f"      eval_seed={es}  "
                      f"effort={metrics['effort_mean']:.3f}  "
                      f"RMSE={metrics['rmse_mean']:.3f}  "
                      f"ECE={metrics['ece']:.4f}")

    # ---------------------------------------------------------------- CSV
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n-> {csv_path}")

    # --------------------------------------------------------- aggregate
    def agg(metric):
        means, stds = [], []
        for lam_e in lambda_es:
            vals = [r[metric] for r in rows if r["lambda_e"] == lam_e]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        return np.array(means), np.array(stds)

    print("\n" + "=" * 64)
    print(f"{'lambda_e':>10}  {'effort':>14}  {'RMSE':>14}  {'ECE':>14}")
    print("-" * 64)
    for i, lam_e in enumerate(lambda_es):
        e_m, e_s = agg("effort_mean")
        r_m, r_s = agg("rmse_mean")
        c_m, c_s = agg("ece")
        print(f"{lam_e:>10.3f}  {e_m[i]:>6.3f}+/-{e_s[i]:.3f}  "
              f"{r_m[i]:>6.3f}+/-{r_s[i]:.3f}  "
              f"{c_m[i]:>6.4f}+/-{c_s[i]:.4f}")

    # ------------------------------------------------------------ figure
    metrics = [
        ("effort_mean", "Mean control effort"),
        ("rmse_mean",   "RMSE vs setpoint (C)"),
        ("ece",         "Expected Calibration Error"),
    ]
    xs = np.arange(len(lambda_es))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for (key, label), ax in zip(metrics, axes):
        m, s = agg(key)
        ax.errorbar(xs, m, yerr=s, fmt="o-", capsize=4, linewidth=1.8,
                    markersize=8, color="tab:red")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(l) for l in lambda_es])
        ax.set_xlabel(r"effort-penalty weight $\lambda_e$")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"PPO reward-shaping sweep  "
        f"({len(train_seeds)} train seeds, {len(eval_seeds)} eval seeds per cell, "
        f"{args.timesteps//1000}k timesteps)",
        fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"-> {png_path}")


if __name__ == "__main__":
    main()
