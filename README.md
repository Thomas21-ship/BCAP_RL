# Antwerp Port Thesis Project

This repository contains the core reinforcement learning environment and training pipeline for the Antwerp port simulation.
The codebase is intentionally minimal and modular to support stable refactoring, testing, and production training.

## Structure

- `env/port_env.py`: Gymnasium environment (single source of truth).
- `training/train.py`: single training entrypoint (script).
- `training/config.py`: training defaults and config loader.
- `training/seed.py`: seeding utilities.
- `training/env_setup.py`: environment config and build helpers.
- `training/model.py`: model construction and training helpers.
- `training/eval.py`: evaluation utilities.
- `training_config.json`: runtime training and environment config.
- `metrics/runs/<run_id>/eval.jsonl`: periodic eval logs (per run).
- `metrics/runs/<run_id>/training_rewards.png`: training reward plot (per run).
- `models/<run_id>/dqn_antwerp_port.zip`: saved model (per run).
- `env/vessel.py`: vessel model and queue (arrival process + queue dynamics).
- `tests/port_env_quick_test.py`: quick environment sanity checks.

## Setup

Baseline requirements:
- Python 3.11+ recommended.
- Core dependencies: `gymnasium`, `stable-baselines3`, `numpy`, `torch`, `matplotlib`.

## Training

Use `training/train.py` as the single training entrypoint. The script follows this flow:
1. Load `training_config.json`.
2. Validate the environment (`check_env`).
3. Create a per-run `run_id` (`YYYYMMDD_HHMMSS_<hash>`).
4. Train the model.
5. Save the model to `models/<run_id>/dqn_antwerp_port.zip`.
6. Plot training episode rewards to `metrics/runs/<run_id>/training_rewards.png`.
7. Periodically evaluate during training (if enabled) and log to `metrics/runs/<run_id>/eval.jsonl`.
8. Optionally run a final evaluation after training.

Environment validation:
- Use `port_env_quick_test.py` for quick sanity checks.

## Config

`training_config.json` controls both training and environment settings.
Top-level keys:
- `seed`: global seed used for reproducibility.
- `total_timesteps`: total training steps.
- `env`: environment overrides (applied to `PortEnv` attributes).
- `dqn`: DQN hyperparameters.
- `grading`: evaluation settings (episodes and eval interval).
- `model_save_path`: default model save path (overridden to per-run path in `training/train.py`).

Note: keys under `env` map directly to `PortEnv` attributes (for example, `quay_size` or `total_cranes_limit`).
Note: `training/train.py` overwrites `grading.log_path` to `metrics/runs/<run_id>/eval.jsonl` for each run.

Evaluation details:
- During training, `grading.eval_interval_steps` triggers periodic evaluations.
- Eval seeds are `seed + 1000 .. seed + 1004`.
- `eval.jsonl` records `step`, `mean_reward`, `std_reward`, `mean_length`, `std_length`, `seeds`, `timestamp`, `run_id`, and `config_hash`.

## Environment Contract (Medium Detail)

Episode model:
- One step is 15 minutes.
- One episode is one day (`max_steps = 96`).
- Fixed physical constraints: `quay_size = 20`, `total_cranes_limit = 3`.

Action model:
- Discrete encoded action for `(vessel_slot, quay_position, cranes)`.
- A dedicated no-op action exists.
- `cranes` represents an absolute target allocation (`0..3`).
- Invalid actions are penalized and flagged via `info["invalid_action"]` and
  `info["invalid_reason"]` in the step output.

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
- Positive: successful docking (+1), throughput per step, vessel completion bonus (+5).
- Negative: invalid action penalty, idle cranes when ships are waiting, waiting penalties, long-wait penalties.

VesselQueue dynamics:
- Queue capacity equals `num_vessel_slots`.
- New vessel arrivals follow an exponential inter-arrival process (`arrival_scale`).
- Vessel length is sampled (clipped) and workload is correlated to length with noise.
- Each step advances time; arrivals move from `on_the_way` to `waiting`.
- Finished vessels depart and are replaced with newly generated vessels.

Core invariants:
- `0 <= cranes_in_use <= total_cranes_limit`.
- Observation shape and bounds remain valid every step.
- Episode terminates at `max_steps` unless an explicit earlier termination condition is set.

Ordering rules:
- `reset()` must be called before the first `step()`; `step()` assumes state initialized by `reset()`.
- Observation order is part of the model contract. Changing the order or shape requires retraining.
- If `step()` is called after termination, the environment auto-resets and sets `info["autoreset"] = True`.

## Compatibility Notes

- `train_model` returns `(model, cfg, episode_rewards)`; reward history is kept in-memory and written as a plot.
- Any observation shape or order change invalidates previously trained models.

## Notes on Artifacts

- Saved models and metrics are organized under `models/<run_id>/` and `metrics/runs/<run_id>/`.
- Keep the best model from each run; raw metrics are disposable if storage is tight.
