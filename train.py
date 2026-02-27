# =============================================================================
# train.py — Training the RL Agent on the Antwerp Port Simulation
# =============================================================================
#
# WHAT THIS FILE DOES (in plain English):
#   1. Loads the port environment (the bridge simulator)
#   2. Runs a pre-departure safety check on the environment
#   3. Creates a PPO agent and trains it over many simulated weeks
#   4. Saves the trained model to disk
#   5. Runs a test episode to show how the trained agent performs
#
# HOW TO RUN THIS FILE:
#   Open a terminal in the project folder and type:
#       python train.py
#
# EXPECTED RUNTIME:
#   500,000 timesteps takes roughly 3–10 minutes on a modern laptop.
#   You can reduce TOTAL_TIMESTEPS below if you just want to test the setup.
#
# =============================================================================

# --- IMPORTS ---
# Think of imports like loading charts before a voyage.
# We pull in the tools we need before the simulation starts.

from stable_baselines3 import PPO
# PPO = Proximal Policy Optimisation. This is the AI instructor.
# It watches the agent's decisions and slowly improves its strategy.

from stable_baselines3.common.env_checker import check_env
# check_env = the pre-departure safety inspection tool.
# It verifies our environment is correctly built before we waste training time.

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
# Callbacks = automatic actions that fire during training at set intervals.
# Like waypoints on a passage plan — the system checks in and logs progress.

import os
# os = operating system tools. We use it to create folders for saving files.

from port_env import AntwerpPortEnv
# Our port simulation — the bridge simulator the agent will train on.


# =============================================================================
# CONFIGURATION — Change these values to adjust training
# =============================================================================

TOTAL_TIMESTEPS = 500_000
# Total number of individual 15-minute steps across ALL training episodes.
# One episode = one simulated week = 672 steps.
# 500,000 steps ÷ 672 steps/week ≈ 744 simulated weeks of training.
#
# Start with 100_000 to test the setup quickly.
# Use 1_000_000 or more for serious training.

MODEL_SAVE_PATH = "models/ppo_antwerp_port"
# Where to save the trained model when training is complete.
# Like signing off a completed voyage log and filing it.

LOG_DIR = "logs/"
# Where TensorBoard training logs are saved.
# TensorBoard is a visual dashboard — you can watch the agent improve over time.
# To view it: open a terminal and type: tensorboard --logdir logs/

EVAL_FREQ = 10_000
# How often (in timesteps) to automatically test the agent during training.
# Every 10,000 steps, training pauses briefly and runs a test episode.
# Think of it as periodic performance reviews during a long voyage.

CHECKPOINT_FREQ = 50_000
# How often to save a snapshot of the model during training.
# Like taking a position fix every few hours — so you have a fallback
# if something goes wrong later.


# =============================================================================
# STEP 1 — Create and inspect the environment
# =============================================================================

print("=" * 60)
print("ANTWERP PORT RL — TRAINING SCRIPT")
print("=" * 60)

print("\n[1/5] Creating the port environment...")
env = AntwerpPortEnv()
# This creates a fresh instance of our port simulator.
# Nothing has happened yet — no vessels, no cranes working.
# Think of this as powering up the bridge simulator before the cadet sits down.

print("      Environment created. ✅")

# --- PRE-DEPARTURE SAFETY CHECK ---
print("\n[2/5] Running pre-departure safety check (check_env)...")
# check_env inspects the environment for common mistakes:
#   - Are observations within the declared [0, 1] range?
#   - Does the action space match what step() expects?
#   - Does reset() return the right format?
# If it finds problems, it prints warnings. Fix those before continuing.
check_env(env, warn=True)
print("      Safety check complete. ✅")
print("      (If warnings appeared above, review them before full training.)")


# =============================================================================
# STEP 2 — Create the PPO agent
# =============================================================================

print("\n[3/5] Creating the PPO agent...")

# Create output folders if they don't exist yet
os.makedirs("models", exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
# exist_ok=True means: "create the folder if it's missing, do nothing if it's there"

model = PPO(
    policy="MlpPolicy",
    # "MlpPolicy" = Multi-Layer Perceptron policy.
    # This is a standard neural network — think of it as the agent's brain.
    # MlpPolicy is the right choice when your observations are a flat list of numbers,
    # which is exactly what our 80-number observation space is.

    env=env,
    # The environment the agent will train on — our port simulator.

    verbose=1,
    # Controls how much training information is printed to the terminal.
    # 0 = silent, 1 = progress updates, 2 = very detailed.
    # Like setting your VHF squelch — 1 is a good balance.

    learning_rate=3e-4,
    # How quickly the agent updates its strategy after each batch of experience.
    # 3e-4 = 0.0003. This is the SB3 default and a safe starting value.
    # Too high: the agent overcorrects (like a helmsman fighting the wheel).
    # Too low: training takes forever (like barely touching the helm).

    n_steps=2048,
    # How many steps the agent collects before running a learning update.
    # Think of it as the length of a watch before the officer debrief.
    # Larger = more stable but slower updates.

    batch_size=64,
    # During each learning update, the collected steps are split into mini-batches.
    # The agent learns from 64 steps at a time within each batch.

    n_epochs=10,
    # How many times the agent re-reads each batch of experience to learn from it.
    # More passes = more learning from the same data, but diminishing returns.

    gamma=0.99,
    # Discount factor — how much the agent values future rewards vs immediate ones.
    # 0.99 means the agent cares a lot about long-term outcomes.
    # (A reward in the future is worth 99% as much as the same reward now.)
    # This is important for our simulation: vessels take many steps to process,
    # so the agent needs to think ahead.

    tensorboard_log=LOG_DIR,
    # Where to write training logs for visualisation in TensorBoard.
)

print("      PPO agent created. ✅")
print(f"      Policy network: {model.policy}")


# =============================================================================
# STEP 3 — Set up training callbacks (automatic waypoint checks)
# =============================================================================

print("\n[4/5] Setting up training callbacks...")

# --- EVALUATION CALLBACK ---
# Every EVAL_FREQ steps, this automatically runs 5 test episodes
# and logs the average reward. Useful for spotting if the agent has
# plateaued or if something has gone wrong.
eval_env = AntwerpPortEnv()  # A separate environment just for evaluation
                              # We keep it separate so it doesn't interfere
                              # with the training environment's state.

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_SAVE_PATH + "_best",
    log_path=LOG_DIR,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=5,         # Run 5 full episodes (weeks) per evaluation
    deterministic=True,        # During eval, the agent always picks its best action
                               # (no random exploration). Like exam conditions.
    render=False,
)

# --- CHECKPOINT CALLBACK ---
# Every CHECKPOINT_FREQ steps, save a snapshot of the model.
# Like taking a position fix during a long passage — you have a fallback.
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=MODEL_SAVE_PATH + "_checkpoints/",
    name_prefix="ppo_port",
)

print("      Callbacks ready. ✅")


# =============================================================================
# STEP 4 — Train the agent
# =============================================================================

print("\n[5/5] Starting training...")
print(f"      Total timesteps: {TOTAL_TIMESTEPS:,}")
print(f"      Approximate episodes: {TOTAL_TIMESTEPS // 672:,} simulated weeks")
print(f"      Model will be saved to: {MODEL_SAVE_PATH}.zip")
print("-" * 60)

# This is where the actual training happens.
# The agent runs the simulation over and over, collecting experience
# and slowly improving its berth allocation strategy.
# Think of this as a cadet sitting at a bridge simulator for hundreds
# of voyages, with an instructor adjusting their technique after each one.
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[eval_callback, checkpoint_callback],
    # We pass both callbacks as a list — both fire at their respective intervals.
)

print("-" * 60)
print("\n✅ Training complete!")

# Save the final trained model
model.save(MODEL_SAVE_PATH)
# This saves the model as a .zip file at the path we specified.
# You can reload this later with: model = PPO.load("models/ppo_antwerp_port")
print(f"   Final model saved to: {MODEL_SAVE_PATH}.zip")


# =============================================================================
# STEP 5 — Post-training evaluation (watch the trained agent in action)
# =============================================================================

print("\n" + "=" * 60)
print("POST-TRAINING EVALUATION — One Full Simulated Week")
print("=" * 60)
print("Running one complete episode with the trained agent...")
print("(deterministic=True means the agent always picks its best action)\n")

# Load a fresh environment for the test run
test_env = AntwerpPortEnv()
obs, info = test_env.reset()

# Counters to track what happened during the test episode
total_reward   = 0.0
total_dockings = 0
total_steps    = 0

# Step through the full episode (up to 672 steps = one week)
while True:
    # The agent looks at the observation and chooses an action.
    # deterministic=True means no random exploration — pure best guess.
    action, _states = model.predict(obs, deterministic=True)
    # _states is used by recurrent policies (LSTM etc). PPO doesn't use it — ignore it.

    obs, reward, terminated, truncated, info = test_env.step(action)

    total_reward += reward
    total_steps  += 1

    # Detect a successful docking by watching for the +1.0 docking reward component.
    # Note: reward also includes throughput and penalties, so we check the vessels.
    docked_this_step = sum(
        1 for v in test_env.vessels if v.status == "docked" and v.cranes_assigned > 0
    )

    # Episode ends when the simulated week is over
    if terminated or truncated:
        break

# --- Print summary ---
departed_vessels = sum(1 for v in test_env.vessels if v.status == "departed")
still_waiting    = sum(1 for v in test_env.vessels if v.status == "waiting")
still_docked     = sum(1 for v in test_env.vessels if v.status == "docked")

print(f"  Steps completed      : {total_steps} / {test_env.max_steps}")
print(f"  Total reward         : {total_reward:.2f}")
print(f"  Vessels departed     : {departed_vessels} / {len(test_env.vessels)}")
print(f"  Still docked at end  : {still_docked}")
print(f"  Still waiting at end : {still_waiting}")
print()
print("Vessel-by-vessel breakdown:")
for v in test_env.vessels:
    containers_done = v.workload - v.containers_remaining
    pct = (containers_done / v.workload * 100) if v.workload > 0 else 0
    print(f"  Vessel {v.id:>2} | length={v.length:>2} blocks | "
          f"workload={v.workload:>5.0f} | status={v.status:<9} | "
          f"processed={pct:.0f}%")

print()
print("=" * 60)
print("Training and evaluation complete.")
print(f"To visualise training progress, run:")
print(f"  tensorboard --logdir {LOG_DIR}")
print("=" * 60)
