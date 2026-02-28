import json
import os

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from port_env import AntwerpPortEnv


MODEL_PATH = "models/ppo_antwerp_port"
SAVE_FIGURES = True
OUTPUT_DIR = "visualisations"
USE_RANDOM_AGENT = False
CONFIG_PATH = "training_config.json"


def _load_env_cfg(path=CONFIG_PATH):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("env", {})


def apply_env_config(env, env_cfg):
    for key, value in env_cfg.items():
        if hasattr(env, key):
            setattr(env, key, value)


def build_voyage_log(env):
    records = []
    for v in env.vessels:
        if not hasattr(v, "docking_step"):
            continue

        departure_step = getattr(v, "departure_step", env.current_step)
        if departure_step <= v.docking_step:
            departure_step = v.docking_step + 1

        records.append(
            {
                "vessel_id": v.id,
                "quay_position": int(getattr(v, "docking_position", 0)),
                "length": int(v.length),
                "docking_step": int(v.docking_step),
                "departure_step": int(departure_step),
                "arrival_time": float(v.arrival_time),
                "cranes": int(max(1, getattr(v, "max_cranes_assigned", v.cranes_assigned))),
                "workload": float(v.workload),
            }
        )

    return sorted(records, key=lambda e: e["arrival_time"])


def run_episode(model=None, seed=42):
    env = AntwerpPortEnv()
    apply_env_config(env, _load_env_cfg())
    obs, _ = env.reset(seed=seed)

    total_reward = 0.0
    while True:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    return build_voyage_log(env), total_reward, env.vessels, env.max_steps


def draw_gantt(voyage_log, max_steps, title="Berth Utilisation - Agent"):
    if not voyage_log:
        print("No vessels were docked in this episode.")
        return None

    fig, ax = plt.subplots(figsize=(16, 8))
    crane_min, crane_max = 1, 7
    cmap = cm.get_cmap("YlOrRd")
    norm = mcolors.Normalize(vmin=crane_min, vmax=crane_max)

    for entry in voyage_log:
        vessel_id = entry["vessel_id"]
        quay_pos = entry["quay_position"]
        length = entry["length"]
        docking_step = entry["docking_step"]
        departure_step = entry["departure_step"]
        arrival_time = entry["arrival_time"]
        cranes = entry["cranes"]

        bar_duration = max(1, departure_step - docking_step)
        bar_colour = cmap(norm(cranes))

        ax.broken_barh(
            [(docking_step, bar_duration)],
            (quay_pos, length),
            facecolors=bar_colour,
            edgecolors="black",
            linewidth=0.8,
            alpha=0.85,
        )

        if bar_duration > 10:
            ax.text(
                docking_step + bar_duration / 2,
                quay_pos + length / 2,
                f"V{vessel_id}\n{cranes}c",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="black",
            )

        ax.plot(
            [arrival_time, arrival_time],
            [quay_pos, quay_pos + length],
            color="red",
            linewidth=1.2,
            linestyle="--",
            alpha=0.7,
        )

    ax.set_xlim(0, max_steps)
    ax.set_ylim(0, 40)

    step_ticks = np.arange(0, max_steps + 1, 48)
    hour_labels = [f"{int(t/4)}h" for t in step_ticks]
    ax.set_xticks(step_ticks)
    ax.set_xticklabels(hour_labels, fontsize=8)

    block_ticks = np.arange(0, 41, 5)
    ax.set_yticks(block_ticks)
    ax.set_yticklabels([f"{b*20}m" for b in block_ticks], fontsize=8)

    ax.set_xlabel("Simulation Time (each step = 15 min)", fontsize=11)
    ax.set_ylabel("Quay Position", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Peak Cranes Assigned", fontsize=10)
    cbar.set_ticks(range(crane_min, crane_max + 1))

    arrival_line = mpatches.Patch(color="red", alpha=0.7, label="Arrival time")
    ax.legend(handles=[arrival_line], loc="upper right", fontsize=9)

    plt.tight_layout()
    return fig


def draw_waiting_delay(voyage_log, title="Vessel Waiting Delay"):
    if not voyage_log:
        return None

    fig, ax = plt.subplots(figsize=(14, max(4, len(voyage_log) * 0.9)))
    cmap = cm.get_cmap("YlOrRd")
    norm = mcolors.Normalize(vmin=1, vmax=7)

    for i, entry in enumerate(voyage_log):
        arrival_time = entry["arrival_time"]
        docking_step = entry["docking_step"]
        departure_step = entry["departure_step"]
        cranes = entry["cranes"]

        waiting_duration = max(0.0, docking_step - arrival_time)
        berth_duration = max(1.0, departure_step - docking_step)

        ax.broken_barh(
            [(arrival_time, waiting_duration)],
            (i + 0.1, 0.8),
            facecolors="lightgrey",
            edgecolors="grey",
            linewidth=0.8,
        )
        ax.broken_barh(
            [(docking_step, berth_duration)],
            (i + 0.1, 0.8),
            facecolors=cmap(norm(cranes)),
            edgecolors="black",
            linewidth=0.8,
            alpha=0.85,
        )

        ax.text(
            -6,
            i + 0.5,
            f"V{entry['vessel_id']} ({entry['length']}b, {entry['workload']:.0f}c)",
            ha="right",
            va="center",
            fontsize=8,
        )

    ax.set_xlim(-6, 680)
    ax.set_ylim(0, len(voyage_log))
    ax.set_yticks([])
    ax.set_xlabel("Simulation Time (steps)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    wait_patch = mpatches.Patch(color="lightgrey", label="Waiting")
    berth_patch = mpatches.Patch(color="coral", label="Docked (color = cranes)")
    ax.legend(handles=[wait_patch, berth_patch], loc="lower right", fontsize=9)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Peak Cranes Assigned", fontsize=10)
    cbar.set_ticks(range(1, 8))

    plt.tight_layout()
    return fig


def print_summary(voyage_log, vessels, total_reward):
    print("\n" + "=" * 75)
    print("EPISODE SUMMARY")
    print("=" * 75)
    print(f"Total reward: {total_reward:.2f}")
    print(f"Vessels docked: {len(voyage_log)} / {len(vessels)}")

    departed = sum(1 for v in vessels if v.status == "departed")
    waiting = sum(1 for v in vessels if v.status == "waiting")
    docked = sum(1 for v in vessels if v.status == "docked")
    print(f"Departed: {departed}")
    print(f"Still docked: {docked}")
    print(f"Still waiting: {waiting}")

    if voyage_log:
        print("\n V.ID  Len   Work    Arr   Dock  Depart  Wait  Berth  Cranes")
        print(" " + "-" * 66)
        for e in voyage_log:
            wait = e["docking_step"] - e["arrival_time"]
            berth = e["departure_step"] - e["docking_step"]
            print(
                f" {e['vessel_id']:>4}  {e['length']:>3}  {e['workload']:>6.0f}"
                f"  {e['arrival_time']:>6.1f}  {e['docking_step']:>4}"
                f"  {e['departure_step']:>6}  {wait:>5.1f}  {berth:>5}  {e['cranes']:>6}"
            )
    print("=" * 75 + "\n")


def main():
    print("=" * 60)
    print("ANTWERP PORT RL - VISUALISER")
    print("=" * 60)

    model = None
    model_label = "Random Agent"

    if not USE_RANDOM_AGENT:
        model_zip = MODEL_PATH + ".zip"
        if os.path.exists(model_zip):
            print(f"Loading model: {model_zip}")
            model = PPO.load(MODEL_PATH)
            model_label = f"Trained PPO ({MODEL_PATH})"
        else:
            print(f"Model not found at {model_zip}; using random agent.")

    print(f"Running one full episode. Agent: {model_label}")
    voyage_log, total_reward, vessels, max_steps = run_episode(model=model)

    print_summary(voyage_log, vessels, total_reward)

    fig1 = draw_gantt(voyage_log, max_steps, title=f"Berth Utilisation - {model_label}")
    fig2 = draw_waiting_delay(voyage_log, title=f"Waiting Delay - {model_label}")

    if SAVE_FIGURES and (fig1 or fig2):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if fig1:
            p1 = os.path.join(OUTPUT_DIR, "berth_gantt.png")
            fig1.savefig(p1, dpi=150, bbox_inches="tight")
            print(f"Saved: {p1}")
        if fig2:
            p2 = os.path.join(OUTPUT_DIR, "waiting_delay.png")
            fig2.savefig(p2, dpi=150, bbox_inches="tight")
            print(f"Saved: {p2}")

    print("Showing charts...")
    plt.show()


if __name__ == "__main__":
    main()
