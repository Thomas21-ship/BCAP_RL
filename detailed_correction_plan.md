# Detailed Correction Plan (Core RL Pipeline, Excluding Visualizer)

## 1. Create a Safe Baseline Snapshot
- Record current behavior before changes by running a short training (`TOTAL_TIMESTEPS=20_000`) and one deterministic eval episode.
- Save outputs in a `baseline_notes.md` with reward, departed vessels, waiting vessels, and run config.
- Goal: prevent regressions and keep a clear before/after comparison.

## 2. Define Explicit Contracts Between Files
- In `ship_generator.py`, `ship_manager.py`, and `port_env.py`, document exact input/output contracts.
- Contract examples: generator returns `(length, workload, arrival_time)` ranges; vessel statuses allowed; env observation layout and bounds.
- Goal: remove ambiguity and make tests straightforward.

## 3. Fix Action-Space Encoding/Decoding Mismatch (P0)
- In `port_env.py`, replace magic numbers (`3201`, `321`, `% 7`) with named constants.
- Decide one consistent design:
  - Option A: include a true dedicated `NO_OP` action index.
  - Option B: include `vessel_slot == 10` and make it reachable by action-space size math.
- Ensure crane choices match intended range (`0..7` if 8 options are required).
- Update `action_space`, `_decode_action`, comments, and step handling together.
- Add bounds checks so invalid decoded tuples fail fast in tests.
- Done when: every intended action tuple is reachable exactly once, and no unintended tuple is reachable.

## 4. Fix Observation Normalization and Bounds (P0)
- In `port_env.py`, ensure all observation values always stay inside `[0,1]`.
- Replace `vessel.id/100` occupancy encoding with a stable bounded representation (recommended: binary occupancy `0/1`).
- Clamp normalized features safely (`np.clip`) for arrival and containers remaining.
- Re-check if `arrival_time` should represent absolute future time or relative time-to-arrival; whichever you choose, keep it bounded.
- Done when: random rollouts pass with zero out-of-bound values.

## 5. Stabilize Vessel ID Semantics
- Keep `vessel_id_counter` for identity/debugging, but decouple it from observation values.
- Optionally reset counter every episode if global uniqueness is unnecessary for your experiments.
- Done when: ID growth cannot break observation constraints.

## 6. Add Invalid-Action Penalties and Clearer Transition Handling (P1)
- In `port_env.py`, explicitly penalize infeasible docking attempts (occupied berth, not arrived, no cranes available, out of quay bounds).
- Keep penalty small but nonzero to guide learning away from wasted actions.
- Done when: policy receives consistent learning signal for bad actions instead of silent no-op behavior.

## 7. Improve Controllability of Crane Allocation (P1)
- Extend step logic to allow crane reassignment for already docked vessels, or add a dedicated action mode for reallocation.
- Keep global crane cap invariant at all times.
- This addresses the current limitation where crane allocation is mostly locked at docking time.
- Done when: agent can respond to queue pressure by reallocating resources mid-episode.

## 8. Refine Reward Function Toward Operational KPIs (P1)
- Keep current rewards (dock, throughput, completion), then add direct waiting-time pressure:
  - per-step penalty for each arrived waiting vessel,
  - optional penalty for long waiting tails.
- Keep coefficients configurable and documented.
- Done when: reward aligns with throughput and waiting-time reduction, not just raw container processing.

## 9. Clean Training Script and Remove Dead Logic
- In `train.py`, remove or wire up unused variables (`total_dockings`, `docked_this_step`).
- Keep evaluation summary metrics coherent and actionable.
- Done when: no unused evaluation counters remain.

## 10. Add Reproducibility Controls (P0)
- In `train.py`, add explicit global seed handling (Python/NumPy/SB3/env reset seed).
- Print seed in run header and save it with model artifacts.
- Done when: repeated runs with same seed and config show consistent trajectories within expected PPO variance.

## 11. Externalize Config for Experiments (P1)
- Move key constants from code into a config source (CLI args or a small JSON/YAML).
- Include environment constants, reward weights, PPO hyperparameters, seed, and total timesteps.
- Done when: you can run structured experiments without editing source files.

## 12. Add Automated Tests for Invariants (P0)
- Create a test suite (e.g., `tests/test_port_env.py`) covering:
  - action decode/encode consistency,
  - observation bounds and shape,
  - crane conservation (`0 <= cranes_in_use <= limit`),
  - quay occupancy consistency with docked vessels,
  - terminal condition at `max_steps`,
  - generator output ranges.
- Done when: tests pass and catch the current known failures.

## 13. Add KPI Reporting Artifact Per Run (P1)
- Write per-episode summary to CSV/JSON from `train.py`: reward, departures, mean waiting, crane utilization, completion ratio.
- Save alongside models for comparison between runs.
- Done when: run-to-run model quality can be compared programmatically.

## 14. Introduce Model Selection and Stopping Criteria (P2)
- Add early-stop logic based on evaluation plateau or KPI thresholds.
- Keep periodic checkpoints and choose best model by stable evaluation metric, not only final timestep.
- Done when: training time is reduced and selected model quality is more reliable.

## 15. Validation Pass After Fixes
- Run short smoke training first, then medium run.
- Compare against baseline snapshot from Step 1.
- Accept changes only if:
  - invariants/tests pass,
  - no observation bound violations,
  - improved waiting/departure KPIs without major reward collapse.

## Execution Order Recommendation
1. P0 first: Steps 3, 4, 10, 12.
2. Then P1 behavior quality: Steps 6, 7, 8, 11, 13.
3. Finally optimization/ops: Steps 14, 15.
