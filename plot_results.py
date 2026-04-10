"""
snake_ai/plot_results.py
------------------------
Read logs/trials.csv and generate training charts 

Usage:
    python plot_results.py
    python plot_results.py --csv logs/session_20250410_120000.csv
"""

import argparse
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="logs/trials.csv", help="CSV log file to plot")
args = parser.parse_args()

if not os.path.exists(args.csv):
    print(f"No log file found at '{args.csv}'. Train the agent first.")
    exit(1)

# ── Read data ─────────────────────────────────────────────────────────────────
trials, scores, records, avgs = [], [], [], []

with open(args.csv) as f:
    for row in csv.DictReader(f):
        trials.append(int(row["trial"]))
        scores.append(int(row["score"]))
        records.append(int(row["record"]))
        avgs.append(float(row["avg_score"]))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle("AI Snake — Deep Q-Learning Training Results", fontsize=15, fontweight="bold")

# Top: score per game + rolling average
ax1 = axes[0]
ax1.plot(trials, scores,  color="#5B9BD5", alpha=0.5, linewidth=0.8, label="Score per game")
ax1.plot(trials, avgs,    color="#ED7D31", linewidth=2.0,             label="Average score")
ax1.set_ylabel("Score")
ax1.legend(loc="upper left")
ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax1.grid(True, alpha=0.3)

# Bottom: record (best ever score)
ax2 = axes[1]
ax2.plot(trials, records, color="#70AD47", linewidth=2.0, label="Record (best score)")
ax2.set_ylabel("Record Score")
ax2.set_xlabel("Game (Trial) Number")
ax2.legend(loc="upper left")
ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out = "logs/training_results.png"
plt.savefig(out, dpi=150)
print(f"Chart saved → {out}")
plt.show()
