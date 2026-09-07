"""Editable vector overview for the Aim 1 Approach."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "svg.fonttype": "none", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.5, 6.0))
fig.subplots_adjust(0, 0, 1, 1)
ax.set(xlim=(0, 100), ylim=(0, 100)); ax.axis("off")
ink, green, blue, orange = "#21312C", "#E2EED9", "#E4EDF4", "#FAE6D4"

def box(x, y, w, h, title, body="", color="white", fontsize=8.3):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.5,rounding_size=1.1",
                              facecolor=color,edgecolor="#7F8C86",linewidth=.8))
    ax.text(x+w/2,y+h-2.6,title,ha="center",va="top",weight="bold",color=ink,fontsize=9)
    if body: ax.text(x+w/2,y+h-6.8,body,ha="center",va="top",linespacing=1.25,
                     color=ink,fontsize=fontsize)

def arrow(start, end, rad=0):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=10,
                               connectionstyle=f"arc3,rad={rad}",color=ink,linewidth=1))

ax.text(2,98,"Aim 1a. Generate datasets with known discoveries",weight="bold",va="top",fontsize=11)
box(2,77,29,15,"Propose and review", "LLM proposes hypotheses\nSearch and compare literature\nReplace unsupported candidates", green)
box(36,77,28,15,"Specify the full DGP", "Clinical or DepMap covariates\nSix embedded discoveries\nExpected · neutral · surprising",green)
box(69,77,29,15,"Select one discovery", "Overall reversal, or\nreversal within a fixed subgroup\nwith expected overall ordering",green)
arrow((31.5,84),(35,84)); arrow((64.5,84),(68,84))
ax.text(50,71.5,r"$Y_k = \mu_k(X,A) + \sum_{j:o(j)=k}\beta_j\,g_j(X,A) + \epsilon_k$",
        ha="center",va="center",fontsize=12,color=ink)
ax.text(50,66.7,"Each term specifies an exposure, comparison, outcome, and subgroup.",ha="center",fontsize=8.3)
box(4,50,41,12,"Version A: focal expected", "3 expected · 2 neutral · 1 surprising\nFocal direction: literature-concordant",blue)
box(55,50,41,12,"Version B: focal surprising", "2 expected · 2 neutral · 2 surprising\nFocal direction: literature-discordant",orange)
arrow((29,65),(27,63)); arrow((73,65),(76,63))
ax.text(50,47.1,"Five background discoveries, covariates, and residual draws remain fixed",ha="center",fontsize=8.2,color=ink)
ax.text(50,44.3,"Verify full-DGP contrasts and calibrate paired statistical difficulty",ha="center",weight="bold",fontsize=8.7)
ax.plot([2,98],[41.7,41.7],color="#B9C2BD",linewidth=.8)
ax.text(2,39.8,"Aim 1b. Deploy fresh agents and evaluate the research process",weight="bold",va="top",fontsize=11)
ax.text(50,34.6,"Fresh session · One dataset version · Identical paired budgets",ha="center",fontsize=8.4,color=ink)
box(2,18.5,20,12,"Explore", "Register hypotheses\nRecord expectations",green,7.7)
box(27,18.5,20,12,"Analyze", "Select comparisons\nReceive estimates",green,7.7)
box(52,18.5,20,12,"Appraise", "Assess results\nRequest validation",green,7.7)
box(77,18.5,21,12,"Synthesize", "Assess new evidence\nUpdate accepted claims",green,7.5)
arrow((22.5,25),(26,25)); arrow((47.5,25),(51,25))
# Independent evidence is released only after the appraisal stage.
box(54,5.5,42,9.5,"Independent validation", "Requested or scheduled evidence",blue,7.7)
arrow((62,18),(62,15.6)); arrow((88,15.6),(88,18))
ax.text(25,13.2,"Repeat through the full budget",ha="center",fontsize=8.4,weight="bold",color=ink)
ax.text(25,9.4,"25 clinical or 10 cell-line iterations",ha="center",fontsize=8.1,color=ink)
ax.text(50,3.2,"Final confirmation: recall R · precision P · F1*  |  Coverage E · responsiveness B",ha="center",fontsize=8.1)
for suffix in ("svg", "pdf", "png"):
    fig.savefig(OUT / f"aim1_overview.{suffix}",dpi=300,facecolor="white")
plt.close(fig)
