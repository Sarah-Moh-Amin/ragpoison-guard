# plot_paired.py
# Task 3: bar chart of the main paired results.
# Usage: python src/plot_paired.py runs/<run_folder>

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The five metrics to show, with a short label for each bar.
BARS = [
    ("initial poison exposure", "Initial\nexposure"),
    ("context poison exposure", "Context\nexposure"),
    ("generation attack success rate", "Attack\nsuccess"),
    ("clean accuracy", "Clean\naccuracy"),
    ("attacked accuracy", "Attacked\naccuracy"),
]

BAR_COLOR = "#2a78d6"
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"

# Step 1: which run to plot.
if len(sys.argv) != 2:
    print("usage: python src/plot_paired.py runs/<run_folder>")
    sys.exit(1)
run_dir = sys.argv[1].rstrip("/")

# Step 2: read paired_summary.csv.
rows = {}
split = ""
with open(os.path.join(run_dir, "paired_summary.csv")) as f:
    for row in csv.DictReader(f):
        rows[row["metric"]] = row
        split = row["split"]

# Step 3: collect the bar values (as percent) and the "n/d" labels.
labels = []
values = []
counts = []
n = None
for metric, label in BARS:
    if metric not in rows:
        print("FAIL: metric missing from paired_summary.csv:", metric)
        sys.exit(1)
    num = int(rows[metric]["numerator"])
    den = int(rows[metric]["denominator"])
    n = den
    labels.append(label)
    values.append(100.0 * num / den)
    counts.append(str(num) + "/" + str(den))

# Step 4: draw the chart.
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
positions = range(len(values))
bars = ax.bar(positions, values, width=0.55, color=BAR_COLOR)

# Write "n/12" on top of every bar.
for bar, text in zip(bars, counts):
    x = bar.get_x() + bar.get_width() / 2
    ax.text(x, bar.get_height() + 2, text, ha="center", va="bottom",
            fontsize=10, color=TEXT_COLOR)

ax.set_xticks(list(positions))
ax.set_xticklabels(labels, fontsize=9, color=TEXT_COLOR)
ax.set_ylim(0, 112)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("Percent of pairs", fontsize=9, color=MUTED_COLOR)
ax.tick_params(axis="y", labelsize=8, colors=MUTED_COLOR)
ax.tick_params(axis="x", length=0)
ax.set_title("Poisoning pilot: clean vs attacked (" + split + ", n = " + str(n) + ")",
             fontsize=11, color=TEXT_COLOR)

# Light horizontal grid behind the bars, no box around the chart.
ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(MUTED_COLOR)

# Step 5: save the picture.
fig.tight_layout()
out_path = os.path.join(run_dir, "paired_results.png")
fig.savefig(out_path)
print("Saved", out_path)
