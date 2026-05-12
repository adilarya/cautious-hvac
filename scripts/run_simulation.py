"""End-to-end pipeline: train RL agent, run baselines, produce paper figures.

Usage:
    python scripts/run_simulation.py [options]

Default behaviour:
  - Run NoControl, Proportional, MPC baselines across 5 seeds.
  - Train PPO on a single fixed-seed env for --timesteps steps (GPU if available).
  - Evaluate the trained PPO across the same 5 seeds.
  - Emit a comparison bar chart, calibration reliability diagram, and three
    cross-section heatmaps of the temperature field at the end of an MPC run.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from stable_baselines3 import PPO, SAC

from sim.room_sim import RoomSimulator
from sim.room_env import RoomEnv
from model.gp_field import GPField
from model.uq import detect_pockets, calibration_over_episode
from control.controller import (NoControl, ProportionalCtrl, MPCController,
                                OracleController)
from eval.metrics import run_episode, multi_seed_run, EpisodeRecord


# ----------------------------------------------------------------------- CLI

def parse_args():
    p = argparse.ArgumentParser(description="3D HVAC spatial control - train + evaluate")
    p.add_argument("--algo",         type=str,   default="ppo", choices=["ppo", "sac"])
    p.add_argument("--timesteps",    type=int,   default=100_000)
    p.add_argument("--episode-len",  type=int,   default=100)
    p.add_argument("--nx",           type=int,   default=10)
    p.add_argument("--ny",           type=int,   default=8)
    p.add_argument("--nz",           type=int,   default=5)
    p.add_argument("--n-vents",      type=int,   default=4)
    p.add_argument("--n-sensors",    type=int,   default=8)
    p.add_argument("--setpoint",     type=float, default=22.0)
    p.add_argument("--lambda-unc",   type=float, default=0.3, help="MPC cautious-uncertainty weight")
    p.add_argument("--lambda-effort",type=float, default=0.1, help="Reward effort penalty")
    p.add_argument("--seeds",        type=str,   default="0,1,2,3,4")
    p.add_argument("--out",          type=str,   default="results")
    p.add_argument("--no-train",     action="store_true",
                   help="Skip training; load saved RL agent if present")
    p.add_argument("--skip-rl",      action="store_true",
                   help="Skip the RL baseline entirely (faster smoke runs)")
    return p.parse_args()


def make_sim(seed, args):
    return RoomSimulator(nx=args.nx, ny=args.ny, nz=args.nz,
                         n_vents=args.n_vents, n_sensors=args.n_sensors,
                         setpoint=args.setpoint, seed=seed)


# ------------------------------------------------------------------- training

def train_rl(args, out: Path):
    """Train PPO/SAC on a single fixed-seed env."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = RoomEnv(nx=args.nx, ny=args.ny, nz=args.nz,
                  n_vents=args.n_vents, n_sensors=args.n_sensors,
                  setpoint=args.setpoint, episode_length=args.episode_len,
                  reward_lambda=args.lambda_effort, reward_uniformity_w=1.0,
                  seed=0)
    AlgoCls = PPO if args.algo.lower() == "ppo" else SAC
    model = AlgoCls("MlpPolicy", env, verbose=0, seed=0, device=device)
    print(f"\nTraining {args.algo.upper()} for {args.timesteps:,} timesteps on {device}...")
    model.learn(total_timesteps=args.timesteps)
    save_path = out / f"{args.algo}_agent"
    model.save(str(save_path))
    print(f"  Saved to {save_path}.zip")
    return model


def eval_rl(model, args, seeds, store_fields: bool = False):
    """Evaluate trained RL agent across seeds; same record schema as baselines.

    If `store_fields`, save mean/std/truth histories for every seed
    so the calibration figure can pool across all seeds (rather than
    base its reliability diagram on a single episode).
    """
    summaries = []
    records: list[EpisodeRecord] = []
    for seed in seeds:
        env = RoomEnv(nx=args.nx, ny=args.ny, nz=args.nz,
                      n_vents=args.n_vents, n_sensors=args.n_sensors,
                      setpoint=args.setpoint, episode_length=args.episode_len,
                      reward_lambda=args.lambda_effort, seed=seed)
        obs, _ = env.reset(seed=seed)
        record = EpisodeRecord()
        for _ in range(args.episode_len):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            sr = env._last_sensor
            env.gp.fit(sr.locs, sr.temps)
            mean_field, std_field = env.gp.predict()
            truth = env.sim.ground_truth_field()
            record.record(mean_field, std_field, truth, action,
                          args.setpoint, store_fields=store_fields)
            if term or trunc:
                break
        summaries.append(record.summary())
        if store_fields:
            records.append(record)

    keys = summaries[0].keys()
    agg = {k: {"mean": float(np.mean([s[k] for s in summaries])),
               "std":  float(np.std([s[k] for s in summaries]))} for k in keys}
    return agg, records


# ----------------------------------------------------------------- baselines

def run_baseline(ctrl_factory, name, make_sim_fn, gp_cls, seeds, args,
                 store_fields: bool = False):
    """Run a baseline controller across seeds, optionally storing field
    histories for every seed so calibration can pool across all of them."""
    summaries = []
    records: list = []
    for seed in seeds:
        sim = make_sim_fn(seed)
        gp = gp_cls(nx=args.nx, ny=args.ny, nz=args.nz)
        controller = ctrl_factory(sim)
        record = run_episode(sim, gp, controller, args.episode_len,
                             store_fields=store_fields)
        summaries.append(record.summary())
        if store_fields:
            records.append(record)
    keys = summaries[0].keys()
    agg = {k: {"mean": float(np.mean([s[k] for s in summaries])),
               "std":  float(np.std([s[k] for s in summaries]))} for k in keys}
    print(f"  {name}: done")
    return agg, records


# ---------------------------------------------------------------- reporting

def print_results_table(results: dict):
    """Print a tabulated mean +/- std for the five core metrics."""
    metrics = ["rmse_mean", "uniformity_mean", "mae_mean",
               "effort_mean", "ci_coverage_mean"]
    labels  = ["RMSE (C)", "Std T (C)", "GP MAE (C)", "Effort", "90% CI cov"]
    bar = "=" * (16 + 18 * len(results))
    print("\n" + bar)
    print(f"{'Metric':<16}" + "".join(f"{k:>18}" for k in results.keys()))
    print("-" * (16 + 18 * len(results)))
    for metric, label in zip(metrics, labels):
        row = f"{label:<16}"
        for r in results.values():
            m = r.get(metric, {"mean": float('nan'), "std": float('nan')})
            row += f"   {m['mean']:>7.3f} +/- {m['std']:>5.3f}"
        print(row)
    print(bar)


def plot_comparison(results: dict, out: Path):
    """Bar chart: mean +/- std for each metric, grouped by condition."""
    metrics = ["rmse_mean", "uniformity_mean", "mae_mean", "effort_mean"]
    labels  = ["RMSE vs setpoint (C)", "Temperature std (C)",
               "GP MAE vs truth (C)", "Mean control effort"]
    x = np.arange(len(metrics))
    width = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, r) in enumerate(results.items()):
        means = [r[m]["mean"] for m in metrics]
        stds  = [r[m]["std"]  for m in metrics]
        ax.bar(x + i * width, means, width, yerr=stds, label=name,
               capsize=4, alpha=0.85)
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("Controller comparison (mean +/- std across seeds)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out / 'comparison.png'}")


def plot_cross_sections(record: EpisodeRecord, out: Path,
                        nx: int, ny: int, nz: int, setpoint: float, name: str):
    """Three horizontal cross-sections (floor, mid, ceiling) of the truth field
    at the final timestep, with sensor and vent locations overlaid."""
    if not record.truth_history:
        return
    field = record.truth_history[-1]
    z_levels = [0, nz // 2, nz - 1]
    z_labels = ["floor (z=0)", f"mid (z={nz // 2})", f"ceiling (z={nz - 1})"]

    vmin = min(float(field.min()), setpoint - 5)
    vmax = max(float(field.max()), setpoint + 5)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, z, lab in zip(axes, z_levels, z_labels):
        im = ax.imshow(field[:, :, z].T, origin="lower", cmap="coolwarm",
                       vmin=vmin, vmax=vmax)
        ax.set_title(f"{lab}   T at final step")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, label="T (C)", fraction=0.046, pad=0.04)
    fig.suptitle(f"{name}: temperature cross-sections (setpoint = {setpoint:.1f} C)")
    fig.tight_layout()
    # Sanitise: lower-case, replace spaces with underscores, strip () and other punctuation.
    safe = (name.lower()
                .replace(" ", "_")
                .replace("(", "").replace(")", "")
                .replace("/", "_"))
    fig.savefig(out / f"crosssection_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}/crosssection_{safe}.png")


def plot_calibration(records: dict, out: Path):
    """Reliability diagram pooled across all seeds per controller.

    `records`: dict mapping controller name to a list[EpisodeRecord] whose
    mean/std/truth histories were stored. Empirical coverage is computed
    by concatenating all (voxel, time) pairs across every stored episode.
    """
    levels = np.linspace(0.50, 0.95, 10)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1,
            label="perfect calibration (y=x)")
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i, (name, recs) in enumerate(records.items()):
        if not recs:
            continue
        means  = [m for r in recs for m in r.mean_history]
        stds   = [s for r in recs for s in r.std_history]
        truths = [t for r in recs for t in r.truth_history]
        if not truths:
            continue
        cal = calibration_over_episode(means, stds, truths, nominal_levels=levels)
        n_seeds = len(recs)
        ax.plot(cal.nominal_levels, cal.empirical_coverage, "o-",
                color=colors[i % len(colors)], linewidth=1.5,
                label=f"{name} (ECE={cal.ECE:.3f}, n={n_seeds} seeds)")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Nominal coverage level alpha")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Reliability diagram (pooled across all seeds)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out / 'calibration.png'}")


# ---------------------------------------------------------------------- main

def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Room: {args.nx}x{args.ny}x{args.nz} voxels "
          f"({args.nx*args.ny*args.nz} cells) | "
          f"{args.n_vents} ceiling vents | {args.n_sensors} sensors | "
          f"setpoint={args.setpoint} C")
    print(f"Episode length: {args.episode_len} steps | Seeds: {seeds}")

    print("\nRunning baselines...")
    no_ctrl_results, no_ctrl_recs = run_baseline(
        lambda sim: NoControl(), "NoControl",
        lambda s: make_sim(s, args), GPField, seeds, args,
        store_fields=True)
    prop_results, prop_recs = run_baseline(
        lambda sim: ProportionalCtrl(), "Proportional",
        lambda s: make_sim(s, args), GPField, seeds, args,
        store_fields=True)
    mpc_results, mpc_recs = run_baseline(
        lambda sim: MPCController(lam=args.lambda_unc), "MPC",
        lambda s: make_sim(s, args), GPField, seeds, args,
        store_fields=True)
    oracle_results, oracle_recs = run_baseline(
        lambda sim: OracleController(), "Oracle",
        lambda s: make_sim(s, args), GPField, seeds, args,
        store_fields=True)

    results = {"NoControl":    no_ctrl_results,
               "Proportional": prop_results,
               "MPC":          mpc_results,
               "Oracle":       oracle_results}
    records = {"NoControl":    no_ctrl_recs,
               "Proportional": prop_recs,
               "MPC":          mpc_recs,
               "Oracle":       oracle_recs}

    if not args.skip_rl:
        model_path = out / f"{args.algo}_agent.zip"
        if args.no_train and model_path.exists():
            AlgoCls = PPO if args.algo.lower() == "ppo" else SAC
            model = AlgoCls.load(str(model_path))
            print(f"\nLoaded saved {args.algo.upper()} from {model_path}")
        else:
            model = train_rl(args, out)

        print("\nEvaluating RL agent...")
        rl_results, rl_recs = eval_rl(model, args, seeds, store_fields=True)
        results[f"RL ({args.algo.upper()})"] = rl_results
        records[f"RL ({args.algo.upper()})"] = rl_recs
        print("  RL: done")

    print_results_table(results)
    plot_comparison(results, out)
    plot_calibration(records, out)
    # Cross-sections show the seed-0 final field per controller.
    for name, recs in records.items():
        if recs:
            plot_cross_sections(recs[0], out, args.nx, args.ny, args.nz,
                                args.setpoint, name)

    print(f"\nResults saved to {out.resolve()}")


if __name__ == "__main__":
    main()
