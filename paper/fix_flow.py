import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
BLUE, VERM, GRAY = "#0072B2", "#D55E00", "#555555"
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "figure.dpi": 150})
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")

def banana_U(x0, x1, b=0.1):
    return x0**2/200 + 0.5*(x1 + b*x0**2 - 100*b)**2

# flowchart v2
fig, ax = plt.subplots(figsize=(7.0, 2.7))
ax.set_xlim(0, 10); ax.set_ylim(-0.1, 4)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)
def box(x, y, w, h, text, fc="#EAF2FA", ec=BLUE, fs=6.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs)
def arr(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=10, color=GRAY, lw=0.9))
ax.text(2.2, 3.8, "WARM-UP (adaptive; gradient oracle)", fontsize=7, color=GRAY)
ax.text(8.35, 3.8, "PRODUCTION (frozen; zero gradients)", fontsize=7, color=GRAY, ha="center")
box(0.15, 2.55, 1.75, 0.8, "multi-start L-BFGS\npaths + Hessians")
box(2.35, 2.55, 1.75, 0.8, "candidate charts\n(evidence-ranked)")
box(4.55, 2.55, 1.75, 0.8, "forward-KL weights\n(provenance pool)")
box(4.55, 1.25, 1.75, 0.8, "chart birth +\nresidual surrogates")
box(2.35, 1.25, 1.75, 0.8, "transport detection\nshear / scale /\npolar / hier / $t$", fs=5.9)
box(0.15, 1.25, 1.75, 0.8, "pilot transitions\n+ frontier probes")
box(0.5, 0.0, 5.7, 0.72, "select surrogates ($\\kappa$) · error-knee $h$ · defensive $t$ · FREEZE",
    fc="#FDEBDD", ec=VERM, fs=6.0)
box(7.0, 2.0, 2.75, 1.15,
    "local: chart Gibbs $k\\sim r_k(x)$\nexact rotation + cached kicks\nONE density-oracle MH test")
box(7.0, 0.5, 2.75, 0.85, "global ($p_g$): independence\ndraw from defensive $g$")
arr(1.9, 2.95, 2.35, 2.95); arr(4.1, 2.95, 4.55, 2.95)
arr(5.42, 2.55, 5.42, 2.05)          # weights -> birth (loop column)
arr(4.55, 1.65, 4.1, 1.65)           # birth -> detection
arr(2.35, 1.65, 1.9, 1.65)           # detection -> pilots
arr(1.02, 1.25, 1.02, 2.55)          # pilots -> back up (loop)
arr(3.3, 1.25, 3.3, 0.72)            # loop exits -> freeze
arr(5.9, 0.36, 7.0, 0.75)            # freeze -> global
arr(5.9, 0.5, 6.6, 2.3); arr(6.6, 2.3, 7.0, 2.45)  # freeze -> local (elbow)
ax.plot([6.7, 6.7], [-0.05, 3.6], ls=":", color=GRAY, lw=0.8)
ax.text(1.05, 2.15, "rounds", fontsize=5.6, color=GRAY, rotation=90)
fig.savefig(f"{FIG}/warmup_flow.pdf", bbox_inches="tight"); plt.close(fig)
print("flow v2", flush=True)
