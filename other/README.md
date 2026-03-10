# Antwerp Port Thesis Project

This is a stripped-down core project for Antwerp port reinforcement learning.
The codebase is intentionally minimal: vessel generation, vessel state, environment logic, and optional model training.
All reporting and plotting layers have been removed.

## Minimal Structure

- `ship_generator.py`: stochastic vessel generation.
- `ship_manager.py`: vessel state model.
- `port_env.py`: Gymnasium environment (`AntwerpPortEnv`).
- `train.py`: minimal DQN training script.
- `training_config.json`: runtime training/environment config.
- `tests/test_port_env.py`: core invariant tests.

## Quick Commands

Run core tests:

```bash
python -m unittest -v tests.test_port_env
```

Train a model:

```bash
python train.py --config training_config.json
```

Quick smoke training:

```bash
python train.py --timesteps 50000 --seed 42
```

## Environment Contract (Medium Detail)

Episode model:
- One step is 15 minutes.
- One episode is one week (`max_steps = 672`).
- Fixed physical constraints: `quay_size = 40`, `total_cranes_limit = 7`.

Action model:
- Discrete encoded action for `(vessel_slot, quay_position, cranes)`.
- A dedicated no-op action exists.
- `cranes` represents an absolute target allocation (`0..7`).
- Invalid actions are penalized; optional termination on invalid action exists.

Observation model:
- Shape `(80,)`, bounded in `[0, 1]`.
- First 40 values: quay occupancy map.
- Next 40 values: 10 vessels x 4 normalized features:
  - arrival time
  - vessel length
  - containers remaining
  - vessel status

Reward model:
- Positive: successful docking, throughput, vessel completion.
- Negative: idle cranes under queue pressure, waiting penalties, long-wait penalties, invalid actions.

Core invariants:
- `0 <= cranes_in_use <= total_cranes_limit`.
- Observation shape and bounds remain valid every step.
- Episode terminates at `max_steps` unless an explicit earlier termination condition is set.

## Engineering Rules

- Keep core logic explainable end-to-end.
- Do not add non-core layers until core behavior is fully stable.
- Treat generated outputs (`models/`, `logs/`, `metrics/`) as disposable artifacts.
