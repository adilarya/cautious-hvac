# HVAC Spatial Temperature Control

A 3D voxel thermal simulator + Gaussian-process spatial inference, comparing
proportional, cautious-MPC, and PPO reinforcement-learning controllers for
ceiling-vent damper control. CSCI 5512 final project (University of
Minnesota).

## Highlight result

Across five seeds (pooled), four deployable controllers and the Oracle
upper bound cluster tightly on
temperature RMSE ($\le 0.13$ °C apart) and uniformity ($\le 0.08$ °C
apart), but spread by $2.1\times$ on **calibration of the GP posterior
credible intervals**.

| Controller | RMSE (°C) | Effort | 90% CI cov | **ECE (pooled)** |
| --- | --- | --- | --- | --- |
| NoControl     | 3.62 | 0.50 | 0.867 | 0.044 |
| Proportional  | 3.58 | 1.00 | 0.929 | 0.031 |
| **MPC (cautious)** | 3.59 | 0.80 | 0.898 | **0.021** |
| Oracle (upper bound) | 3.59 | 0.80 | **0.909** | 0.026 |
| PPO           | 3.70 | 0.00 | 0.863 | 0.035 |

**Oracle = MPC on RMSE within seed noise** — perfect knowledge of the
temperature field does not improve control quality, confirming the
problem is action-bottlenecked (supply-temperature physics + airflow
budget) rather than perception-bottlenecked.

MPC tracks `y = x` on the reliability diagram most tightly across all
nominal credible-interval levels. A $\lambda$-ablation
(`scripts/run_lambda_ablation.py`) shows the MPC advantage comes from
its constrained-optimisation structure rather than the uncertainty
penalty term per se: the airflow constraint binds at every timestep,
which pins the SLSQP solution onto the same point regardless of $\lambda$.
The PPO agent converges to a zero-effort policy that exploits the
effort-penalty term — a diagnosed reward-shaping failure mode discussed
in the paper.

## Repo layout

```
project/
├── sim/
│   ├── room_sim.py     3D voxel thermal simulator (stochastic α, sensor noise, occupancy heat)
│   └── room_env.py     Gymnasium wrapper for RL training
├── model/
│   ├── gp_field.py     GP regression on sparse 3D sensors
│   └── uq.py           Pocket detection + episode-level calibration
├── control/
│   └── controller.py   NoControl / Proportional / cautious-MPC controllers
├── eval/
│   └── metrics.py      EpisodeRecord + multi-seed runner
├── scripts/
│   └── run_simulation.py  end-to-end: baselines + PPO + plots
├── tests/              pytest suite (19 cases)
├── paper/              LaTeX writeup (main.tex, refs.bib)
├── results/            generated figures and trained model (gitignored .zip)
└── requirements.txt
```

## Install

```bash
# Linux / WSL
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch is the heavy install (~2 GB with CUDA libraries). CUDA is optional;
the project falls back to CPU automatically. CPU training of PPO on this
9-dim observation env takes ~5–10 minutes for 100k timesteps.

## Run

```bash
# Tests (16 cases, ~10 s)
pytest tests -q

# Smoke run: baselines only, single seed, short episode (~30 s)
python scripts/run_simulation.py --skip-rl --seeds 0 --episode-len 30

# Full pipeline: 4 controllers x 5 seeds + PPO 100k timesteps
python scripts/run_simulation.py --seeds 0,1,2,3,4 --timesteps 100000

# Skip PPO training, evaluate a previously-saved agent
python scripts/run_simulation.py --no-train
```

All artefacts land under `results/`:

- `comparison.png` — bar chart over 4 controllers x 4 metrics
- `calibration.png` — reliability diagram per controller
- `crosssection_<controller>.png` — floor / mid / ceiling temperature
  cross-sections at the final timestep
- `ppo_agent.zip` — saved RL agent (gitignored)

## Building the paper

The LaTeX source is in `paper/`. Easiest is Overleaf — upload `main.tex`,
`refs.bib`, and the `results/` directory. Local build:

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Project notes

The simulator's vent supply temperature is clamped at the setpoint, so
vents can pull a cell toward setpoint but not beyond it. With
$T_\mathrm{init} = 15$ °C, $T_\mathrm{out} \approx 18$ °C, and setpoint
22 °C, the room equilibrates near 18.5 °C across all controllers — RMSE
from setpoint is dominated by this physics floor, not by controller
quality. The calibration story (which controllers produce GP-consistent
fields?) is what differentiates the methods. A hot-day variant
($T_\mathrm{init} = 28$ °C, vents actively cooling) is left as future
work.

## Authors

- Adil Arya — *arya0033@umn.edu*
- Mohammed Jassim Jahubar Ali — *jahub001@umn.edu*

## License

MIT. See `LICENSE`.
