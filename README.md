# Antwerp Port Thesis Project

This repository contains the core reinforcement learning environment and training pipeline for the Antwerp port simulation.
The codebase is intentionally minimal and modular to support stable refactoring, testing, and production training.

## Structure

- `env/port_env.py`: Gymnasium environment (single source of truth).
- `env/port_env_dynamics.py`: reset/step helpers used by `port_env.py`.
- `env/port_env_spec.py`: action + observation helpers used by `port_env.py`.
- `training/train.py`: single training entrypoint (script).
- `training/config.py`: training defaults and config loader.
- `training/seed.py`: seeding utilities.
- `training/env_setup.py`: environment config and build helpers.
- `training/model.py`: model construction and training helpers.
- `training/eval.py`: evaluation utilities.
- `training_config.json`: runtime training and environment config.
- `ship_generator.py`: stochastic vessel generation.
- `ship_manager.py`: vessel state model.
- `port_env_quick_test.py`: quick environment sanity checks.

## Setup

Baseline requirements:
- Python 3.11+ recommended.
- Core dependencies: `gymnasium`, `stable-baselines3`, `numpy`, `torch`, `matplotlib`.

## Training

Use `training/train.py` as the single training entrypoint. The script follows this flow:
1. Load `training_config.json`.
2. Validate the environment (`check_env`).
3. Train the model.
4. Save the model (important: keep the best model from each run).
5. Plot training episode rewards (in-memory, no log files).
6. Optionally evaluate the model.

Environment validation:
- Use `port_env_quick_test.py` for quick sanity checks.

## Config

`training_config.json` controls both training and environment settings.
Top-level keys:
- `seed`: global seed used for reproducibility.
- `total_timesteps`: total training steps.
- `env`: environment overrides (applied to `PortEnv` attributes).
- `dqn`: DQN hyperparameters.
- `grading`: evaluation settings (episodes, eval interval, and optional flags).
- `model_save_path`: where the trained model is saved.

Note: keys under `env` map directly to `PortEnv` attributes (for example, `quay_size` or `total_cranes_limit`).

## Environment Contract (Medium Detail)

Episode model:
- One step is 15 minutes.
- One episode is one day (`max_steps = 96`).
- Fixed physical constraints: `quay_size = 20`, `total_cranes_limit = 3`.

Action model:
- Discrete encoded action for `(vessel_slot, quay_position, cranes)`.
- A dedicated no-op action exists.
- `cranes` represents an absolute target allocation (`0..3`).
- Invalid actions are fail-fast and raise `RuntimeError`.

Observation model:
- Shape `(37,)`, bounded in `[0, 1]`.
- First 20 values: quay occupancy map.
- Next 16 values: 4 vessels x 4 normalized features:
  - arrival time
  - vessel length
  - containers remaining
  - vessel status
- Final value: normalized crane usage (`cranes_in_use / total_cranes_limit`)

Reward model:
- Positive: successful docking, throughput, vessel completion.
- Negative: idle cranes under queue pressure, waiting penalties, long-wait penalties.

Core invariants:
- `0 <= cranes_in_use <= total_cranes_limit`.
- Observation shape and bounds remain valid every step.
- Episode terminates at `max_steps` unless an explicit earlier termination condition is set.

Ordering rules:
- `reset()` must be called before the first `step()`; `step()` assumes state initialized by `reset()`.
- Observation order is part of the model contract. Changing the order or shape requires retraining.

## Compatibility Notes

- `train_model` returns `(model, cfg, episode_rewards)`; reward history is in-memory and not logged to disk.
- Any observation shape or order change invalidates previously trained models.

## Engineering Rules

- Keep core logic explainable end-to-end.
- Do not add non-core layers until core behavior is fully stable.
- Treat generated outputs (`models/`, `logs/`, `metrics/`) as disposable artifacts.

## Notes on Artifacts

- Saved models are important outputs; keep the best model from each run.
