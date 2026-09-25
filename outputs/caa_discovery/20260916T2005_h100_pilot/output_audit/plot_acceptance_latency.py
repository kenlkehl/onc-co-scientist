"""Export the descriptive unmasked acceptance-delay comparison for sharing."""
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
rows = json.loads((HERE / "matched_acceptance_latency.json").read_text())
groups = ["agrees_with_expectation", "opposes_expectation"]
selected = [
    [r for r in rows if r["condition"].endswith("_unmasked")
     and r["group"] == group and r["both_pre_evidence"]]
    for group in groups
]
assert [len(group) for group in selected] == [4, 3]
assert all(r[arm]["iteration_delay"] is not None
           for group in selected for r in group for arm in ["control", "caa"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.edgecolor": "#ADB5BD",
    "text.color": "#202A35", "axes.labelcolor": "#202A35",
    "xtick.color": "#202A35", "ytick.color": "#4C5967",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})
fig, ax = plt.subplots(figsize=(8.2, 5.5))
fig.subplots_adjust(left=0.13, right=0.97, bottom=0.29, top=0.76)
fig.text(0.065, 0.945, "Acceptance of expected and surprising findings",
         fontsize=16, weight="bold", va="top")
fig.text(0.065, 0.89, "Synthetic lung cancer datasets • Biomedical variable names retained",
         fontsize=10.5, color="#536170", va="top")

width = 0.28
for arm, label, color, offset in [
    ("control", "Control", "#667789", -width / 2),
    ("caa", "CAA", "#007F86", width / 2),
]:
    values = [mean(r[arm]["iteration_delay"] for r in group) for group in selected]
    positions = [i + offset for i in range(2)]
    ax.bar(positions, values, width=width, color=color, label=label, zorder=3)
    for x, value in zip(positions, values):
        if value == 0:
            ax.plot([x - width / 2, x + width / 2], [0, 0],
                    color=color, linewidth=3, clip_on=False, zorder=4)
        ax.text(x, value + 0.16, f"{value:g}", ha="center", va="bottom",
                fontsize=12, weight="bold", color=color)

ax.set_xticks([0, 1], ["Consistent with prior expectation\n4 matched comparisons",
                       "Contradicts prior expectation\n3 matched comparisons"])
ax.tick_params(axis="x", length=0, pad=12, labelsize=10.5)
ax.tick_params(axis="y", length=0)
ax.set_ylabel("Mean delay to formal acceptance\n(research iterations)", labelpad=12)
ax.set_ylim(0, 8.3)
ax.set_xlim(-0.55, 1.55)
ax.yaxis.set_major_locator(MultipleLocator(2))
ax.grid(axis="y", color="#E4E8EC", linewidth=0.8, zorder=0)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.035), frameon=False,
          ncol=2, borderaxespad=0, handlelength=1.3, columnspacing=1.8)
fig.text(0.065, 0.155,
         "Delay is measured from first numerical evidence; 0 = acceptance in the same iteration.\n"
         "Preliminary, descriptive results from two paired CAA/control runs.\n"
         "Comparisons are not independent replicates; controls already accepted consistent findings immediately.",
         fontsize=9.1, color="#536170", linespacing=1.5, va="top")

for extension in ["png", "pdf", "svg"]:
    path = HERE / f"acceptance_latency_unmasked.{extension}"
    fig.savefig(path, dpi=300, facecolor="white")
    print(path)
plt.close(fig)
