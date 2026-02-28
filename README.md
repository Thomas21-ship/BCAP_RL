# Antwerp Port RL Simulator

A reinforcement-learning project that trains a PPO agent to manage berth allocation and crane assignment in a simulated container port.

The environment models:
- Quay space constraints (40 blocks)
- Global crane constraints (7 cranes total)
- Stochastic vessel arrivals and workloads
- Docking, processing, departure, and waiting-time dynamics

The training pipeline includes:
- Environment validation (`check_env`)
- PPO training with checkpointing and periodic evaluation
- Early stopping on no model improvement
- Final deterministic evaluation
- KPI export to JSON/CSV

A visualizer script generates:
- Berth utilization Gantt chart
- Vessel waiting-delay chart

---

## 1. Project Structure

- `ship_generator.py`: stochastic vessel generation (length, workload, arrival time)
- `ship_manager.py`: `Vessel` class and vessel state fields
- `port_env.py`: Gymnasium environment (`AntwerpPortEnv`)
- `train.py`: training/evaluation entrypoint (PPO + callbacks + KPI artifacts)
- `port_visualizer.py`: post-training episode visualizations
- `training_config.json`: runtime configuration
- `tests/test_port_env.py`: invariant and behavior tests
- `metrics/`: generated KPI artifacts (`run_*.json`, `runs.csv`)
- `models/`: trained model, best model, checkpoints
- `visualisations/`: generated PNG charts

---

## 2. Environment Logic (High-Level)

Each episode is one simulated week (`672` steps, 15 minutes/step).

### Actions
Action space is `Discrete(3201)`:
- `3200` docking/reallocation actions: `10 vessel slots x 40 quay positions x 8 crane options (0..7)`
- `1` explicit no-op action

Action decoding returns:
- `vessel_slot`
- `quay_position`
- `cranes`

Behavior in `step()`:
- If selected vessel is **waiting + arrived**: try docking
- If selected vessel is **already docked**: treat `cranes` as target crane allocation (reallocation)
- If action infeasible: apply penalty (and optionally terminate if configured)

### Observations
Observation shape is `(80,)`, normalized to `[0, 1]`:
- `0..39`: binary quay occupancy (`0` free, `1` occupied)
- `40..79`: up to 10 vessels x 4 features:
  - normalized arrival
  - normalized length
  - normalized containers remaining
  - normalized status code

### Rewards
Reward includes:
- positive docking reward
- positive throughput reward
- positive vessel completion reward
- idle-crane penalty when waiting vessels exist
- waiting-time penalties (base + long-wait excess)
- invalid-action penalty

---

## 3. Requirements

Recommended:
- Python 3.10+
- pip

Install dependencies:

```bash
pip install stable-baselines3 gymnasium numpy torch matplotlib
```

If you use a virtual environment, activate it before running commands.

---

## 4. Configuration

Main config file: `training_config.json`

Key sections:
- `seed`: reproducibility seed
- `total_timesteps`: PPO training timesteps
- `model_save_path`: base output path for models
- `log_dir`: callback/eval log directory
- `eval_freq`, `checkpoint_freq`
- `ppo`: PPO hyperparameters
- `env`: runtime environment knobs (penalties, thresholds)
- `early_stopping`: stop training if eval performance does not improve
- `kpi`: KPI artifact output settings

CLI overrides in `train.py`:

```bash
python train.py --config training_config.json --seed 7 --timesteps 100000
```

---

## 5. How To Run

### A) Train the agent

```bash
python train.py
```

Quick smoke run:

```bash
python train.py --timesteps 50000 --seed 42
```

Training outputs:
- `models/ppo_antwerp_port.zip` (final)
- `models/ppo_antwerp_port_best/best_model.zip` (best eval model, if available)
- `models/ppo_antwerp_port_checkpoints/` (periodic checkpoints)

### B) Run test suite

```bash
python -m unittest -v tests.test_port_env
```

Current tests cover:
- action decode/no-op mapping
- observation shape/bounds
- crane conservation
- episode termination at max steps
- vessel generator output ranges

### C) Visualize results

```bash
python port_visualizer.py
```

Behavior:
- loads `models/ppo_antwerp_port.zip` by default
- falls back to random policy if model is missing
- saves figures (if enabled) to `visualisations/`

Generated files:
- `visualisations/berth_gantt.png`
- `visualisations/waiting_delay.png`

---

## 6. KPI Artifacts

After training/evaluation, `train.py` writes:
- per-run JSON: `metrics/run_YYYYMMDD_HHMMSS.json`
- cumulative CSV: `metrics/runs.csv`

Metrics include:
- reward, steps, departed/waiting/docked counts
- completion ratio
- mean/max docking wait
- unresolved waiting at episode end
- average processed percent
- crane utilization
- invalid-action rate
- seed and model metadata

---

## 7. Reproducibility

`train.py` seeds:
- Python `random`
- NumPy
- PyTorch (CPU/CUDA)
- environment resets (train/eval/test)
- PPO model seed

Notes:
- results are reproducible in trend, but exact bitwise determinism can still vary across hardware/backends.
- PPO rollout collection (`n_steps`) may cause actual sampled timesteps per update block to exceed very small `--timesteps` values.

---

## 8. Troubleshooting

### `Model not found` in visualizer
Train first:

```bash
python train.py
```

Or set `USE_RANDOM_AGENT = True` in `port_visualizer.py`.

### `matplotlib` window does not appear
- Ensure GUI backend support is available in your environment.
- Even without GUI, PNG output is saved when `SAVE_FIGURES = True`.

### Slow training
- Reduce `total_timesteps`
- Increase hardware resources
- Tune PPO params in `training_config.json`

### High invalid-action rate / poor throughput
- Increase training timesteps
- Adjust penalties in `training_config.json` (`env` section)
- Compare best model vs final model performance

---

## 9. Current Status

Implemented and validated:
- action encoding/decoding consistency
- bounded observation normalization
- invalid-action handling and diagnostics
- crane reallocation for docked vessels
- waiting-time reward shaping
- reproducibility controls
- externalized config
- early stopping + best-model evaluation selection
- KPI artifact export
- automated invariant tests
- working visualizer compatible with updated environment

---

## 10. Typical Workflow

1. Adjust `training_config.json`
2. Run `python train.py`
3. Review terminal summary + `metrics/runs.csv`
4. Run `python port_visualizer.py`
5. Iterate config and retrain

---

## 11. License / Usage

No explicit license file is currently present in this repository. Add one if you plan to distribute externally.
