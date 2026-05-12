"""Stage 2 -- pairwise statistical significance on per-seed ECE.

Computes per-seed expected calibration error (ECE) for each of the four
controllers (NoControl, Proportional, MPC, PPO) across a common seed
list, then reports:

  - paired t-test on each of the 6 pairwise ECE differences
  - 95% bootstrap CI on each pairwise mean difference (1000 resamples)
  - Bonferroni correction over the 6 comparisons

Outputs:
  results/significance_per_seed.csv   per-seed ECE per controller
  results/significance_pairs.csv      per-pair statistics
  results/significance.tex            LaTeX table for paper input

PPO is loaded from results/ppo_agent.zip; if absent it is skipped.

Usage:
    python scripts/run_significance.py --seeds 0,1,2,3,4
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")

from sim.room_sim import RoomSimulator
from sim.room_env import RoomEnv
from model.gp_field import GPField
from model.uq import calibration_over_episode
from control.controller import NoControl, ProportionalCtrl, MPCController
from eval.metrics import run_episode, EpisodeRecord


CONTROLLERS = ["NoControl", "Proportional", "MPC", "PPO"]


def per_seed_ece_classical(seeds, args):
    """Per-seed ECE for the three model-based / structural controllers."""
    factories = {
        "NoControl":    lambda sim: NoControl(),
        "Proportional": lambda sim: ProportionalCtrl(),
        "MPC":          lambda sim: MPCController(lam=args.lambda_unc),
    }
    out = {name: [] for name in factories}
    for name, factory in factories.items():
        for seed in seeds:
            sim  = RoomSimulator(nx=args.nx, ny=args.ny, nz=args.nz,
                                 n_vents=args.n_vents,
                                 n_sensors=args.n_sensors,
                                 setpoint=args.setpoint, seed=seed)
            gp   = GPField(nx=args.nx, ny=args.ny, nz=args.nz)
            ctrl = factory(sim)
            rec  = run_episode(sim, gp, ctrl, args.episode_len,
                               store_fields=True)
            cal  = calibration_over_episode(rec.mean_history, rec.std_history,
                                            rec.truth_history)
            out[name].append(cal.ECE)
            print(f"  {name:<14} seed={seed}  ECE={cal.ECE:.4f}")
    return out


def per_seed_ece_ppo(seeds, args):
    """Load a saved PPO model and compute per-seed ECE under it."""
    model_path = Path(args.out) / "ppo_agent.zip"
    if not model_path.exists():
        print(f"  PPO model not found at {model_path}; skipping PPO.")
        return None
    from stable_baselines3 import PPO
    model = PPO.load(str(model_path.with_suffix("")))
    out = []
    for seed in seeds:
        env = RoomEnv(nx=args.nx, ny=args.ny, nz=args.nz,
                      n_vents=args.n_vents, n_sensors=args.n_sensors,
                      setpoint=args.setpoint,
                      episode_length=args.episode_len,
                      reward_lambda=args.lambda_effort, seed=seed)
        obs, _ = env.reset(seed=seed)
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
        cal = calibration_over_episode(rec.mean_history, rec.std_history,
                                       rec.truth_history)
        out.append(cal.ECE)
        print(f"  {'PPO':<14} seed={seed}  ECE={cal.ECE:.4f}")
    return out


def bootstrap_ci(diffs, n_resample=1000, ci=0.95, rng=None):
    """Percentile bootstrap CI on the mean of paired differences."""
    if rng is None:
        rng = np.random.default_rng(0)
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    boot_means = np.empty(n_resample, dtype=float)
    for k in range(n_resample):
        idx = rng.integers(0, n, n)
        boot_means[k] = float(np.mean(diffs[idx]))
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(boot_means, alpha)), \
           float(np.quantile(boot_means, 1.0 - alpha))


def main():
    ap = argparse.ArgumentParser(
        description="Pairwise significance tests on per-seed ECE.")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--episode-len", type=int, default=100)
    ap.add_argument("--nx", type=int, default=10)
    ap.add_argument("--ny", type=int, default=8)
    ap.add_argument("--nz", type=int, default=5)
    ap.add_argument("--n-vents",   type=int, default=4)
    ap.add_argument("--n-sensors", type=int, default=8)
    ap.add_argument("--setpoint",  type=float, default=22.0)
    ap.add_argument("--lambda-unc",    type=float, default=0.3)
    ap.add_argument("--lambda-effort", type=float, default=0.1)
    ap.add_argument("--n-bootstrap",   type=int,   default=1000)
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Computing per-seed ECE for {len(seeds)} seeds, 4 controllers...\n")
    per_seed = per_seed_ece_classical(seeds, args)
    ppo_ece = per_seed_ece_ppo(seeds, args)
    if ppo_ece is not None:
        per_seed["PPO"] = ppo_ece
    else:
        # Drop PPO from comparisons if not available.
        CONTROLLERS.remove("PPO")

    # ----------------------------------------------------------- per-seed CSV
    csv_seed = out / "significance_per_seed.csv"
    with csv_seed.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed"] + CONTROLLERS)
        for i, seed in enumerate(seeds):
            w.writerow([seed] + [f"{per_seed[c][i]:.6f}" for c in CONTROLLERS])
    print(f"\n-> {csv_seed}")

    # Quick descriptive summary
    print("\nPer-controller ECE (mean +/- std across seeds):")
    for c in CONTROLLERS:
        vals = np.array(per_seed[c])
        print(f"  {c:<14}  mean={vals.mean():.4f}  std={vals.std(ddof=1):.4f}")

    # --------------------------------------------------------------- pairwise
    pairs = [(i, j) for i in range(len(CONTROLLERS))
                    for j in range(i + 1, len(CONTROLLERS))]
    n_pairs = len(pairs)
    rng = np.random.default_rng(0)
    rows = []
    print(f"\nPairwise paired t-tests + bootstrap CIs "
          f"({n_pairs} comparisons, Bonferroni applied):\n")
    print(f"{'A vs B':<24} {'mean diff':>11}  "
          f"{'95% CI':>22}  {'p (raw)':>9} {'p (Bonf.)':>10}  sig?")
    print("-" * 86)
    for (i, j) in pairs:
        a, b = CONTROLLERS[i], CONTROLLERS[j]
        diffs = np.array(per_seed[a]) - np.array(per_seed[b])
        _, p_raw = stats.ttest_rel(per_seed[a], per_seed[b])
        p_bonf = min(1.0, float(p_raw) * n_pairs)
        lo, hi = bootstrap_ci(diffs, args.n_bootstrap, rng=rng)
        sig = "yes" if p_bonf < 0.05 else "no"
        print(f"{a:>10} vs {b:<10}  {float(np.mean(diffs)):>+10.4f}  "
              f"[{lo:>+.4f}, {hi:>+.4f}]  {float(p_raw):>9.4f} {p_bonf:>10.4f}  "
              f"{sig}")
        rows.append({
            "a": a, "b": b,
            "mean_diff": float(np.mean(diffs)),
            "ci_lo": lo, "ci_hi": hi,
            "p_raw": float(p_raw), "p_bonf": p_bonf,
            "sig_005": sig,
        })

    # ----------------------------------------------------------- pairs CSV
    csv_pairs = out / "significance_pairs.csv"
    with csv_pairs.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n-> {csv_pairs}")

    # ----------------------------------------------------------- LaTeX table
    tex_path = out / "significance.tex"
    with tex_path.open("w") as f:
        f.write("% Pairwise paired-t + 95% bootstrap CIs on per-seed ECE.\n")
        f.write("% Generated by scripts/run_significance.py -- do not edit by hand.\n")
        f.write("\\begin{tabular}{llrcrr}\n")
        f.write("\\toprule\n")
        f.write("A & B & $\\Delta$ECE & 95\\% bootstrap CI & "
                "$p$ (Bonf.) & sig.\\ at 0.05? \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            f.write(f"{r['a']} & {r['b']} & "
                    f"${r['mean_diff']:+.4f}$ & "
                    f"$[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]$ & "
                    f"${r['p_bonf']:.4f}$ & {r['sig_005']} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"-> {tex_path}")


if __name__ == "__main__":
    main()
