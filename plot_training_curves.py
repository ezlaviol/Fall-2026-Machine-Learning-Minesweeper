import csv
import matplotlib.pyplot as plt
from pathlib import Path
from simulator import TRAINING_FOLDER

GRAPHS_FOLDER = Path("graphs")
GRAPHS_FOLDER.mkdir(parents=True, exist_ok=True)

def load_csv(path, x_col, y1_col, y2_col):
    xs, y1s, y2s = [], [], []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            xs.append(int(row[x_col]))
            y1s.append(float(row[y1_col]))
            y2s.append(float(row[y2_col]))
    return xs, y1s, y2s

dqn_ep, dqn_wr, dqn_eps = load_csv(f"{TRAINING_FOLDER}/dqn_training_log.csv", "episode", "win_rate", "epsilon")
sl_ep, sl_wr, sl_loss = load_csv(f"{TRAINING_FOLDER}/sl_training_log.csv", "episode", "win_rate", "bce_loss")

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.65), dpi=300)

# Panel (a): Hybrid DQN Win Rate & Epsilon Decay
l1 = ax1.plot(dqn_ep, dqn_wr, color="#1f77b4", linewidth=1.6, label="Rolling Win Rate (%)")
ax1.set_xlabel("Training Episodes (9×9, 10 Mines)")
ax1.set_ylabel("Rolling 500-Ep Win Rate (%)", color="#1f77b4")
ax1.set_ylim(20, 85)
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.set_title("(a) Hybrid CSP + DQN Convergence", fontsize=9.5, fontweight="bold")

ax1_twin = ax1.twinx()
l2 = ax1_twin.plot(dqn_ep, dqn_eps, color="#ff7f0e", linestyle="--", linewidth=1.3, label="Exploration (ε)")
ax1_twin.set_ylabel("Epsilon (ε)", color="#ff7f0e")
ax1_twin.set_ylim(0, 1.05)

lines = l1 + l2
ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=7.5)

# Panel (b): Hybrid SL Win Rate & BCE Loss
l3 = ax2.plot(sl_ep, sl_wr, color="#2ca02c", linewidth=1.6, label="Rolling Win Rate (%)")
ax2.set_xlabel("Training Episodes (9×9, 10 Mines)")
ax2.set_ylabel("Rolling 500-Ep Win Rate (%)", color="#2ca02c")
ax2.set_ylim(20, 85)
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.set_title("(b) Hybrid CSP + SL Convergence", fontsize=9.5, fontweight="bold")

ax2_twin = ax2.twinx()
l4 = ax2_twin.plot(sl_ep, sl_loss, color="#d62728", linestyle="--", linewidth=1.3, label="BCE Loss")
ax2_twin.set_ylabel("Masked BCE Loss", color="#d62728")

lines2 = l3 + l4
ax2.legend(lines2, [l.get_label() for l in lines2], loc="center right", fontsize=7.5)

plt.tight_layout()
plt.savefig(f"{GRAPHS_FOLDER}/training_curves.pdf", bbox_inches="tight")
plt.savefig(f"{GRAPHS_FOLDER}/training_curves.png", dpi=300, bbox_inches="tight")
print("Saved training_curves.pdf and training_curves.png")