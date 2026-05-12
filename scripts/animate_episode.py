"""Animate a single episode as a side-by-side ground-truth-vs-GP GIF.

Top row: ground-truth temperature field at floor / mid / ceiling cross-sections.
Bottom row: GP posterior mean field at the same cross-sections.
Black dots = sensors at that z-level. Red stars = vents (only on ceiling).
Title shows step number and current RMSE-from-setpoint.

Usage examples:
    python scripts/animate_episode.py                          # default: MPC, seed 0
    python scripts/animate_episode.py --controller NoControl
    python scripts/animate_episode.py --controller MPC --seed 2 --fps 12

The GIF is reasonably small (~3-5 MB) at the default settings.
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
from matplotlib.animation import FuncAnimation, PillowWriter

from sim.room_sim import RoomSimulator
from model.gp_field import GPField
from control.controller import NoControl, ProportionalCtrl, MPCController


CONTROLLERS = {
    "NoControl":    lambda lam: NoControl(),
    "Proportional": lambda lam: ProportionalCtrl(),
    "MPC":          lambda lam: MPCController(lam=lam),
}


def main():
    ap = argparse.ArgumentParser(description="Animate one episode as a GIF.")
    ap.add_argument("--controller", type=str, default="MPC",
                    choices=list(CONTROLLERS.keys()))
    ap.add_argument("--seed",         type=int,   default=0)
    ap.add_argument("--episode-len",  type=int,   default=100)
    ap.add_argument("--lambda-unc",   type=float, default=0.3)
    ap.add_argument("--nx", type=int, default=10)
    ap.add_argument("--ny", type=int, default=8)
    ap.add_argument("--nz", type=int, default=5)
    ap.add_argument("--n-vents",   type=int, default=4)
    ap.add_argument("--n-sensors", type=int, default=8)
    ap.add_argument("--setpoint",  type=float, default=22.0)
    ap.add_argument("--fps",       type=int,   default=8,
                    help="GIF playback fps")
    ap.add_argument("--frame-stride", type=int, default=2,
                    help="Render every Nth simulation step into the GIF")
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sim = RoomSimulator(nx=args.nx, ny=args.ny, nz=args.nz,
                        n_vents=args.n_vents, n_sensors=args.n_sensors,
                        setpoint=args.setpoint, seed=args.seed)
    gp   = GPField(nx=args.nx, ny=args.ny, nz=args.nz)
    ctrl = CONTROLLERS[args.controller](args.lambda_unc)

    # ----------------------------------------------------- roll out + capture
    print(f"Rolling out {args.controller} (seed={args.seed}, {args.episode_len} steps)...")
    obs = sim.reset()
    truths    : list[np.ndarray] = []
    gp_means  : list[np.ndarray] = []
    rmses     : list[float]      = []
    for t in range(args.episode_len):
        gp.fit(obs.locs, obs.temps)
        mean_field, std_field = gp.predict()
        commands = ctrl.compute_action(mean_field, std_field, sim)
        obs = sim.step(commands)
        truths.append(sim.ground_truth_field().copy())
        gp_means.append(mean_field.copy())
        rmses.append(float(np.sqrt(
            np.mean((sim.ground_truth_field() - sim.setpoint) ** 2))))
    print("  rollout done.")

    # Decimate frames for the GIF
    stride = max(1, args.frame_stride)
    frames = list(range(0, args.episode_len, stride))

    # Common colour scale across both rows and all frames so the eye
    # can track temperature evolution.
    all_T = np.concatenate(
        [t.ravel() for t in truths] + [m.ravel() for m in gp_means])
    vmin = float(np.percentile(all_T, 1))
    vmax = float(np.percentile(all_T, 99))

    z_levels = [0, args.nz // 2, args.nz - 1]
    z_labels = [f"floor (z=0)", f"mid (z={args.nz // 2})",
                f"ceiling (z={args.nz - 1})"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    title = fig.suptitle(
        f"{args.controller} | step 0 / {args.episode_len} | "
        f"RMSE = {rmses[0]:.2f} C  (setpoint {args.setpoint:.0f} C)",
        fontsize=13)

    # Build the imshow handles and overlay sensors/vents
    im_handles: list[tuple] = []  # (im, row, z)
    for row, row_label in enumerate(["Truth", "GP posterior mean"]):
        for col, (z, zlab) in enumerate(zip(z_levels, z_labels)):
            ax = axes[row, col]
            field = truths[0] if row == 0 else gp_means[0]
            im = ax.imshow(field[:, :, z].T, origin="lower", cmap="coolwarm",
                           vmin=vmin, vmax=vmax, animated=True)
            ax.set_title(f"{row_label} - {zlab}", fontsize=10)
            ax.set_xlabel("x"); ax.set_ylabel("y")
            # Sensors at this z-level
            for sx, sy, sz in sim.sensor_locs:
                if int(sz) == z:
                    ax.plot(sx, sy, "ko", markersize=5,
                            markeredgecolor="white", markeredgewidth=0.8)
            # Vents at this z-level (only on ceiling by construction)
            for vx, vy, vz in sim.vent_locs:
                if int(vz) == z:
                    ax.plot(vx, vy, "r*", markersize=12,
                            markeredgecolor="white", markeredgewidth=0.8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            im_handles.append((im, row, z))
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    def update(frame_idx):
        t = frames[frame_idx]
        for im, row, z in im_handles:
            field = truths[t] if row == 0 else gp_means[t]
            im.set_data(field[:, :, z].T)
        title.set_text(
            f"{args.controller} | step {t+1} / {args.episode_len} | "
            f"RMSE = {rmses[t]:.2f} C  (setpoint {args.setpoint:.0f} C)")
        return [im for im, _, _ in im_handles] + [title]

    out_path = out / f"animation_{args.controller.lower()}.gif"
    print(f"Writing GIF ({len(frames)} frames at {args.fps} fps) -> {out_path} ...")
    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 / args.fps, blit=False)
    anim.save(str(out_path), writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"  done.  Open the GIF in your file browser or in the GitHub repo view.")


if __name__ == "__main__":
    main()
