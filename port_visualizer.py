# =============================================================================
# port_visualizer.py — Berth Utilisation Gantt Chart
# =============================================================================
#
# WHAT THIS FILE DOES:
#   Loads a trained PPO model, runs one full simulated week (672 steps),
#   then draws two charts:
#
#   CHART 1 — BERTH UTILISATION GANTT (the main view)
#     X-axis : Time (steps, 0–672). Each step = 15 minutes.
#     Y-axis : Quay position (blocks 0–40). Each block = 20m.
#     Each bar: One vessel's time on the berth.
#       - Bar length  = how long it was docked (docking_step → departure_step)
#       - Bar height  = physical length of the vessel (quay blocks occupied)
#       - Bar colour  = crane count assigned (colour scale, darker = more cranes)
#       - Label       = Vessel ID
#       - Red marker  = arrival time (when the vessel actually reached port)
#       - Gap between red marker and bar start = waiting time at anchorage
#
#   CHART 2 — WAITING DELAY SUMMARY
#     One row per vessel. Shows arrival time, docking time, and the gap
#     between them as a horizontal bar — like a race chart for anchorage time.
#
# HOW TO RUN:
#   python port_visualizer.py
#
#   By default it looks for: models/ppo_antwerp_port.zip
#   You can change MODEL_PATH below.
#
# WHAT YOU NEED INSTALLED:
#   pip install matplotlib stable-baselines3
#
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import os

from stable_baselines3 import PPO
from port_env import AntwerpPortEnv


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH = "models/ppo_antwerp_port"
# Path to the saved model (without .zip — SB3 adds that automatically)
# Change this if your model is saved somewhere else.

SAVE_FIGURES = True
# True  = saves the charts as PNG files (in the current folder)
# False = only shows them on screen

OUTPUT_DIR = "visualisations/"
# Where to save the PNG files (created automatically if missing)

USE_RANDOM_AGENT = False
# True  = ignores the model and uses random actions instead
#         Useful for comparing a trained agent vs random baseline
# False = uses your trained PPO model


# =============================================================================
# STEP 1 — Run one episode and collect the voyage log
# =============================================================================

def run_episode(model=None):
    """
    Runs one complete simulated week (up to 672 steps).
    Returns the voyage_log (list of docking records) and total reward.

    If model is None, a random agent is used instead.
    This is like doing a trial run of the bridge simulator and
    writing down everything that happened in the movement book.
    """
    env = AntwerpPortEnv()
    obs, _ = env.reset()

    total_reward = 0.0

    while True:
        if model is not None:
            # Trained agent: always picks its best known action (no exploration)
            action, _ = model.predict(obs, deterministic=True)
        else:
            # Random agent: picks any action at random (baseline comparison)
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    return env.voyage_log, total_reward, env.vessels, env.max_steps


# =============================================================================
# STEP 2 — Draw the Berth Utilisation Gantt Chart
# =============================================================================

def draw_gantt(voyage_log, max_steps, title="Berth Utilisation — Trained Agent"):
    """
    Draws the main Gantt chart: time on X, quay position on Y.
    Each vessel is a coloured horizontal bar on the berth board.
    """

    if not voyage_log:
        print("⚠  Voyage log is empty — no vessels were docked this episode.")
        print("   The agent may need more training, or the model path may be wrong.")
        return None

    fig, ax = plt.subplots(figsize=(16, 8))

    # --- COLOUR SCALE FOR CRANE COUNT ---
    # Crane count ranges from 1 to 7 (our total_cranes_limit).
    # We map this to a colour gradient: light blue (1 crane) → dark blue (7 cranes).
    # Think of it as a heat map: darker = more crane activity at that berth.
    crane_min, crane_max = 1, 7
    cmap = cm.get_cmap("YlOrRd")    # Yellow (few cranes) → Red (many cranes)
    norm = mcolors.Normalize(vmin=crane_min, vmax=crane_max)

    for entry in voyage_log:
        vessel_id     = entry["vessel_id"]
        quay_pos      = entry["quay_position"]      # Y-axis: where on the quay
        length        = entry["length"]             # Y-axis: height of the bar
        docking_step  = entry["docking_step"]       # X-axis: bar starts here
        departure_step= entry["departure_step"]     # X-axis: bar ends here
        arrival_time  = entry["arrival_time"]       # Red marker position
        cranes        = entry["cranes"]             # Determines bar colour
        workload      = entry["workload"]

        bar_duration = departure_step - docking_step
        bar_colour   = cmap(norm(cranes))

        # --- DRAW THE BERTH BAR ---
        # broken_barh draws a horizontal bar.
        # First argument: [(x_start, width)] — where the bar begins and how wide
        # Second argument: (y_start, height) — where on the Y-axis
        ax.broken_barh(
            [(docking_step, bar_duration)],
            (quay_pos, length),
            facecolors=bar_colour,
            edgecolors="black",
            linewidth=0.8,
            alpha=0.85,
        )

        # --- VESSEL ID LABEL inside the bar ---
        # Only draw if the bar is wide enough to fit text
        if bar_duration > 10:
            bar_centre_x = docking_step + bar_duration / 2
            bar_centre_y = quay_pos + length / 2
            ax.text(
                bar_centre_x, bar_centre_y,
                f"V{vessel_id}\n{cranes}🏗",
                ha="center", va="center",
                fontsize=7, fontweight="bold",
                color="black",
            )

        # --- ARRIVAL TIME MARKER (red vertical tick) ---
        # This shows when the vessel physically arrived at port (at anchorage).
        # If the red tick is LEFT of the bar start, the vessel waited at anchor.
        # If it lines up with the bar start, the agent docked it immediately.
        #
        # This is the "arrival vs actual docking" waiting delay visualised.
        ax.plot(
            [arrival_time, arrival_time],
            [quay_pos, quay_pos + length],
            color="red",
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
        )

        # --- WAITING DELAY ANNOTATION ---
        # Draw a small arrow/line between arrival and docking if there was a delay
        waiting_delay = docking_step - arrival_time
        if waiting_delay > 5:
            ax.annotate(
                "",
                xy=(docking_step, quay_pos + length * 0.8),
                xytext=(arrival_time, quay_pos + length * 0.8),
                arrowprops=dict(
                    arrowstyle="->",
                    color="red",
                    lw=1.2,
                ),
            )

    # --- AXES AND LABELS ---
    ax.set_xlim(0, max_steps)
    ax.set_ylim(0, 40)      # Quay is 40 blocks

    # Convert step numbers to hours on the X-axis for readability
    # 4 steps = 1 hour. 672 steps = 168 hours = 7 days.
    step_ticks = np.arange(0, max_steps + 1, 48)   # Every 48 steps = 12 hours
    hour_labels = [f"{int(t/4)}h\n(Day {int(t/96)+1})" if t % 96 == 0
                   else f"{int(t/4)}h"
                   for t in step_ticks]
    ax.set_xticks(step_ticks)
    ax.set_xticklabels(hour_labels, fontsize=7)

    # Y-axis: quay blocks, labelled in metres
    block_ticks = np.arange(0, 41, 5)
    ax.set_yticks(block_ticks)
    ax.set_yticklabels([f"{b*20}m\n(block {b})" for b in block_ticks], fontsize=7)

    ax.set_xlabel("Simulation Time  (each step = 15 minutes)", fontsize=11)
    ax.set_ylabel("Quay Position  (each block = 20m)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    ax.grid(True, axis="y", linestyle=":", alpha=0.2)

    # --- COLOUR BAR (crane count legend) ---
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Cranes Assigned", fontsize=10)
    cbar.set_ticks(range(crane_min, crane_max + 1))

    # --- ARRIVAL MARKER LEGEND ---
    arrival_line = mpatches.Patch(color="red", alpha=0.7, label="Vessel arrival at port (anchorage)")
    ax.legend(handles=[arrival_line], loc="upper right", fontsize=9)

    plt.tight_layout()
    return fig


# =============================================================================
# STEP 3 — Draw the Waiting Delay Summary Chart
# =============================================================================

def draw_waiting_delay(voyage_log, title="Vessel Waiting Delay — Arrival vs Docking"):
    """
    One horizontal bar per vessel showing:
      - Grey bar: time spent at anchorage (arrival → docking)
      - Coloured bar: time spent at berth (docking → departure)

    Like a schedule adherence chart from a VTS log.
    """

    if not voyage_log:
        return None

    fig, ax = plt.subplots(figsize=(14, max(4, len(voyage_log) * 0.9)))

    # Sort vessels by arrival time (top = first to arrive)
    sorted_log = sorted(voyage_log, key=lambda e: e["arrival_time"])

    cmap = cm.get_cmap("YlOrRd")
    norm = mcolors.Normalize(vmin=1, vmax=7)

    for i, entry in enumerate(sorted_log):
        vessel_id      = entry["vessel_id"]
        arrival_time   = entry["arrival_time"]
        docking_step   = entry["docking_step"]
        departure_step = entry["departure_step"]
        cranes         = entry["cranes"]
        workload       = entry["workload"]

        waiting_duration = docking_step - arrival_time
        berth_duration   = departure_step - docking_step

        # --- WAITING BAR (grey) — time at anchorage ---
        ax.broken_barh(
            [(arrival_time, waiting_duration)],
            (i + 0.1, 0.8),
            facecolors="lightgrey",
            edgecolors="grey",
            linewidth=0.8,
        )

        # --- BERTH BAR (coloured) — time docked and working ---
        ax.broken_barh(
            [(docking_step, berth_duration)],
            (i + 0.1, 0.8),
            facecolors=cmap(norm(cranes)),
            edgecolors="black",
            linewidth=0.8,
            alpha=0.85,
        )

        # --- LABELS ---
        # Waiting delay label in grey bar (if it's wide enough)
        if waiting_duration > 8:
            ax.text(
                arrival_time + waiting_duration / 2, i + 0.5,
                f"wait {waiting_duration:.0f} steps",
                ha="center", va="center", fontsize=7, color="dimgrey",
            )

        # Vessel label on the left Y-axis
        ax.text(
            -5, i + 0.5,
            f"V{vessel_id}  ({entry['length']}blk, {workload:.0f}c)",
            ha="right", va="center", fontsize=8,
        )

    # --- AXES ---
    ax.set_xlim(-5, 680)
    ax.set_ylim(0, len(sorted_log))
    ax.set_yticks([])
    ax.set_xlabel("Simulation Time (steps, each = 15 min)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    # Legend
    wait_patch  = mpatches.Patch(color="lightgrey",   label="Waiting at anchorage")
    berth_patch = mpatches.Patch(color="coral",        label="Docked and working (colour = cranes)")
    ax.legend(handles=[wait_patch, berth_patch], loc="lower right", fontsize=9)

    # Colour bar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Cranes Assigned", fontsize=10)
    cbar.set_ticks(range(1, 8))

    plt.tight_layout()
    return fig


# =============================================================================
# STEP 4 — Print a text summary to the terminal
# =============================================================================

def print_summary(voyage_log, vessels, total_reward):
    """
    Prints a clean text table to the terminal summarising the episode.
    Like reading the movement book at the end of the watch.
    """
    print()
    print("=" * 75)
    print("  EPISODE SUMMARY — MOVEMENT BOOK")
    print("=" * 75)
    print(f"  Total reward this episode : {total_reward:.2f}")
    print(f"  Vessels docked            : {len(voyage_log)} / {len(vessels)}")

    departed = sum(1 for v in vessels if v.status == "departed")
    waiting  = sum(1 for v in vessels if v.status == "waiting")
    docked   = sum(1 for v in vessels if v.status == "docked")
    print(f"  Departed (fully processed): {departed}")
    print(f"  Still docked at week end  : {docked}")
    print(f"  Still waiting at week end : {waiting}")
    print()

    if voyage_log:
        print(f"  {'V.ID':>4}  {'Length':>6}  {'Work':>6}  {'Arr':>6}  "
              f"{'Dock':>6}  {'Depart':>7}  {'Wait':>5}  {'Berth':>6}  {'Cranes':>6}")
        print("  " + "-" * 65)
        for entry in sorted(voyage_log, key=lambda e: e["arrival_time"]):
            vid  = entry["vessel_id"]
            leng = entry["length"]
            work = entry["workload"]
            arr  = entry["arrival_time"]
            dock = entry["docking_step"]
            dep  = entry["departure_step"]
            wait = dock - arr
            bert = dep - dock
            crn  = entry["cranes"]
            print(f"  {vid:>4}  {leng:>5}b  {work:>6.0f}  {arr:>6.1f}  "
                  f"{dock:>6}  {dep:>7}  {wait:>5.1f}  {bert:>6}  {crn:>6}")

    print("=" * 75)
    print()


# =============================================================================
# MAIN — Run everything
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  ANTWERP PORT RL — VISUALISER")
    print("=" * 60)

    # --- Load the trained model (or fall back to random) ---
    model = None
    model_label = "Random Agent (baseline)"

    if not USE_RANDOM_AGENT:
        model_zip = MODEL_PATH + ".zip"
        if os.path.exists(model_zip):
            print(f"\nLoading trained model from: {model_zip}")
            model = PPO.load(MODEL_PATH)
            model_label = f"Trained PPO Agent ({MODEL_PATH})"
            print("Model loaded. ✅")
        else:
            print(f"\n⚠  Model not found at: {model_zip}")
            print("   Falling back to random agent.")
            print(f"   Train first with: python train.py")
            print(f"   Then re-run: python port_visualizer.py")

    print(f"\nRunning one full episode ({672} steps = 1 simulated week)...")
    print(f"Agent: {model_label}")

    # --- Run the episode and collect data ---
    voyage_log, total_reward, vessels, max_steps = run_episode(model)

    # --- Print text summary ---
    print_summary(voyage_log, vessels, total_reward)

    # --- Draw the charts ---
    print("Drawing charts...")

    fig1 = draw_gantt(
        voyage_log,
        max_steps,
        title=f"Berth Utilisation Gantt — {model_label}"
    )

    fig2 = draw_waiting_delay(
        voyage_log,
        title=f"Vessel Waiting Delay — {model_label}"
    )

    # --- Save or show ---
    if SAVE_FIGURES and (fig1 or fig2):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if fig1:
            path1 = os.path.join(OUTPUT_DIR, "berth_gantt.png")
            fig1.savefig(path1, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path1}")
        if fig2:
            path2 = os.path.join(OUTPUT_DIR, "waiting_delay.png")
            fig2.savefig(path2, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path2}")

    print("\nShowing charts... (close the window to exit)")
    plt.show()
