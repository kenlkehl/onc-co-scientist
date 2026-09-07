"""Editable vector overview for the Aim 1 Approach."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "svg.fonttype": "none", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.5, 6.7))
fig.subplots_adjust(0, 0, 1, 1)
ax.set(xlim=(0, 100), ylim=(-4, 100)); ax.axis("off")
ink, green, blue, orange = "#21312C", "#E2EED9", "#E4EDF4", "#FAE6D4"

def box(x, y, w, h, title, body="", color="white", fontsize=8.3):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.5,rounding_size=1.1",
                              facecolor=color,edgecolor="#7F8C86",linewidth=.8))
    ax.text(x+w/2,y+h-2.6,title,ha="center",va="top",weight="bold",color=ink,fontsize=8.5)
    if body: ax.text(x+w/2,y+h-(9.0 if "\n" in title else 6.8),body,ha="center",va="top",linespacing=1.25,
                     color=ink,fontsize=fontsize)

def arrow(start, end, rad=0):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=10,
                               connectionstyle=f"arc3,rad={rad}",color=ink,linewidth=1))

ax.text(2,98,"Aim 1a. Generate datasets with known discoveries",weight="bold",va="top",fontsize=11)
box(2,67,29,25,"Propose and review", color=green)
box(36,67,28,25,"Specify the data\ngeneration process", color=green)


def variable_icon(kind, x, y):
    """Small vector symbols remain editable and sharp in the exported figure."""
    drawing = DrawingArea(14, 14, 0, 0)

    def line(xs, ys):
        drawing.add_artist(Line2D(xs, ys, color=ink, linewidth=1.0,
                                  solid_capstyle="round"))

    if kind == "Exposures":
        drawing.add_artist(FancyBboxPatch((1, 3.5), 12, 7,
                           boxstyle="round,pad=0,rounding_size=3.5",
                           facecolor="white", edgecolor=ink, linewidth=1))
        line([7, 7], [3.5, 10.5])
    elif kind == "Comparisons":
        for start, end in [((1, 10), (13, 10)), ((13, 4), (1, 4))]:
            drawing.add_artist(FancyArrowPatch(start, end, arrowstyle="->",
                               mutation_scale=7, color=ink, linewidth=1,
                               shrinkA=0, shrinkB=0))
    elif kind == "Outcomes":
        line([1, 1, 13], [13, 1, 1])
        line([3, 6, 9, 12], [4, 8, 6, 11])
    elif kind == "Idea":
        drawing.add_artist(Circle((7, 9), 4, facecolor="white", edgecolor=ink, linewidth=1))
        line([5, 5, 9, 9], [5, 3, 3, 5])
        line([5.5, 8.5], [1, 1])
    elif kind == "Search":
        drawing.add_artist(Circle((5.5, 8.5), 4, facecolor="white", edgecolor=ink, linewidth=1))
        line([8.5, 13], [5.5, 1])
    elif kind == "Refresh":
        drawing.add_artist(FancyArrowPatch((2, 5), (11, 10), connectionstyle="arc3,rad=-.8",
                           arrowstyle="->", mutation_scale=7, color=ink, linewidth=1))
        drawing.add_artist(FancyArrowPatch((12, 9), (3, 4), connectionstyle="arc3,rad=-.8",
                           arrowstyle="->", mutation_scale=7, color=ink, linewidth=1))
    elif kind in {"Record", "Check", "Request"}:
        drawing.add_artist(FancyBboxPatch((2, 1), 10, 12,
                           boxstyle="round,pad=0,rounding_size=1",
                           facecolor="white", edgecolor=ink, linewidth=1))
        if kind == "Check":
            line([4, 6, 10], [7, 5, 10])
        elif kind == "Request":
            line([4, 10], [7, 7]); line([7, 7], [4, 10])
        else:
            for yline in [10, 7, 4]: line([4, 10], [yline, yline])
    elif kind in {"Agree", "Oppose"}:
        # Directional agreement is not a judgment that one discovery is better.
        bottom = ((1, 4), (13, 4)) if kind == "Agree" else ((13, 4), (1, 4))
        for start, end in [((1, 10), (13, 10)), bottom]:
            drawing.add_artist(FancyArrowPatch(start, end, arrowstyle="->",
                               mutation_scale=7, color=ink, linewidth=1,
                               shrinkA=0, shrinkB=0))
    elif kind == "Discoveries":
        for cx, cy in [(3, 10), (10, 10), (3, 3), (10, 3)]:
            drawing.add_artist(Circle((cx, cy), 2, facecolor="white", edgecolor=ink, linewidth=1))
    else:
        for cx, cy in [(3, 9), (11, 9), (7, 11)]:
            drawing.add_artist(Circle((cx, cy), 1.5, facecolor="white",
                                     edgecolor=ink, linewidth=.9))
            drawing.add_artist(FancyBboxPatch((cx-2, cy-6), 4, 3.5,
                               boxstyle="round,pad=0,rounding_size=1",
                               facecolor="white", edgecolor=ink, linewidth=.9))
    ax.add_artist(AnnotationBbox(drawing, (x, y), frameon=False, pad=0))


def icon_label(kind, x, y, text, fontsize=7.5):
    variable_icon(kind, x, y)
    ax.text(x+2.1, y, text, ha="left", va="center", fontsize=fontsize,
            color=ink, linespacing=1.25)


icon_label("Idea", 5.2, 83.0, "LLM proposes hypotheses", 7.8)
icon_label("Search", 5.2, 77.8, "Search and compare\nliterature", 7.8)
icon_label("Refresh", 5.2, 72.2, "Replace unsupported\ncandidates", 7.8)


for label, x, y in [("Exposures", 38.8, 82.1), ("Comparisons", 51.9, 82.1),
                    ("Outcomes", 38.8, 77.7), ("Subgroups", 51.9, 77.7)]:
    variable_icon(label, x, y)
    ax.text(x+1.9, y, label, ha="left", va="center", fontsize=7.2, color=ink)
ax.text(50, 73.6, "Clinical or cell-line covariates", ha="center", va="center",
        fontsize=7.6, color=ink)
ax.text(50, 70.4, "Six discoveries of three types:\nexpected · neutral · surprising",
        ha="center", va="center", fontsize=7.6, color=ink, linespacing=1.25)

box(69,67,29,25,"Select one discovery\nto reverse", color=green)
icon_label("Comparisons", 71.8, 80.4,
           "Change the association from\nagreeing with the literature\nto opposing it.", 7.2)
icon_label("Subgroups", 71.8, 72.8,
           "Reverse it overall, or within a\nsubgroup while the overall\nassociation remains expected.", 7.2)
arrow((31.5,84),(35,84)); arrow((64.5,84),(68,84))
box(4,50,41,12,"Key discovery: expected", color=blue)
icon_label("Agree", 7.5, 56.0, "Key direction agrees with literature", 7.5)
icon_label("Discoveries", 7.5, 52.3, "3 expected · 2 neutral · 1 surprising discoveries", 7.0)
box(55,50,41,12,"Key discovery: surprising", color=orange)
icon_label("Oppose", 58.5, 56.0, "Key direction opposes literature", 7.5)
icon_label("Discoveries", 58.5, 52.3, "2 expected · 2 neutral · 2 surprising discoveries", 7.0)
ax.plot([83.5,83.5,24.5], [66.5,64.5,64.5], color=ink, linewidth=1)
arrow((24.5,64.5),(24.5,62.5)); arrow((75.5,64.5),(75.5,62.5))
ax.text(50,47.1,"The other five discoveries, covariates, and random outcome noise remain fixed",ha="center",fontsize=8.2,color=ink)
ax.text(50,44.3,"Calibrate statistical association strength across paired datasets",ha="center",weight="bold",fontsize=8.4)
ax.plot([2,98],[41.7,41.7],color="#B9C2BD",linewidth=.8)
ax.text(2,39.8,"Aim 1b. Deploy fresh agents and evaluate the research process",weight="bold",va="top",fontsize=11)
ax.text(50,34.6,"Fresh session · One dataset version · Identical paired budgets",ha="center",fontsize=8.4,color=ink)
for x, width, title, items in [
    (2, 20, "Explore", [("Idea", "Register hypotheses"), ("Record", "Record expectations")]),
    (27, 20, "Analyze", [("Comparisons", "Select comparisons"), ("Outcomes", "Receive estimates")]),
    (52, 20, "Appraise", [("Search", "Assess results"), ("Request", "Request validation")]),
    (77, 21, "Synthesize", [("Search", "Assess new evidence"), ("Check", "Update accepted claims")]),
]:
    box(x, 18.5, width, 12, title, color=green)
    for y, (kind, label) in zip([24.0, 20.5], items):
        icon_label(kind, x+1.7, y, label, 7.0)
arrow((22.5,25),(26,25)); arrow((47.5,25),(51,25))
# Independent evidence is released only after the appraisal stage.
box(54,5.5,42,9.5,"Independent validation", color=blue)
icon_label("Refresh", 57.0, 8.2, "Same comparison on fresh simulated data", 7.5)
arrow((62,18),(62,15.6)); arrow((88,15.6),(88,18))
ax.text(26,10.2,"Repeat through the full iteration budget",ha="center",fontsize=8.1,weight="bold",color=ink)
# One continuous return path, with a long final stem and an unobstructed arrowhead.
ax.plot([98.5, 99.4, 99.4, 4], [25, 25, 3.5, 3.5],
        color=ink, linewidth=1.1, solid_joinstyle="round")
ax.add_patch(FancyArrowPatch((4, 3.5), (4, 18), arrowstyle="-|>",
                           mutation_scale=10, color=ink, linewidth=1.1,
                           shrinkA=0, shrinkB=0))
ax.text(2, -.9, "Outcomes", ha="left", va="center", fontsize=8.2,
        weight="bold", color=ink)
for kind, x, label in [
    ("Discoveries", 13, "Discovery recall"),
    ("Check", 29, "Precision"),
    ("Outcomes", 42, "F1*"),
    ("Search", 51, "Exploration coverage"),
    ("Refresh", 75, "Evidence responsiveness"),
]:
    icon_label(kind, x, -.9, label, 7.2)
for suffix in ("svg", "pdf", "png"):
    fig.savefig(OUT / f"aim1_overview.{suffix}",dpi=300,facecolor="white")
plt.close(fig)
