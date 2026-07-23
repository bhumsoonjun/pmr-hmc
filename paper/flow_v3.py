import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PMR, NUTS = "#0072B2", "#D55E00"
INK, MUT = "#333333", "#666666"
FILL, FILL2 = "#EAF2FA", "#FDEBDD"
plt.rcParams.update({"font.size": 7, "figure.dpi": 150})
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")

fig, ax = plt.subplots(figsize=(7.0, 2.45))
ax.set_xlim(0, 10); ax.set_ylim(0.15, 3.25)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)

def box(x, y, w, h, text, fc=FILL, ec=PMR, fs=6.2, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=3)

def arrow(x1, y1, x2, y2, c=MUT):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=9, color=c, lw=0.9,
                                 shrinkA=0, shrinkB=0, zorder=4))

def line(x1, y1, x2, y2, c=MUT):
    ax.plot([x1, x2], [y1, y2], color=c, lw=0.9, zorder=4,
            solid_capstyle="round")

MID = 1.72  # main flow axis

# headers
ax.text(3.55, 3.05, "WARM-UP  (adaptive; gradient oracle)",
        fontsize=7, color=MUT, ha="center")
ax.text(8.75, 3.05, "PRODUCTION  (frozen; zero gradients)",
        fontsize=7, color=MUT, ha="center")

# stage 1: initialize
box(0.12, 1.02, 1.62, 1.4,
    "INITIALIZE\n\nmulti-start L-BFGS\nHessian charts,\nevidence-ranked\nforward-KL weights",
    fs=6.0)

# stage 2: refine container with internal cycle
ax.add_patch(FancyBboxPatch((2.14, 0.42), 3.62, 2.42,
             boxstyle="round,pad=0.05", fc="none", ec="#9BB8D4",
             lw=0.9, linestyle=(0, (2.5, 2)), zorder=1))
ax.text(3.95, 2.62, "REFINE  ·  repeat $B$ rounds", fontsize=6.3,
        color=MUT, ha="center")
bw, bh = 1.52, 0.66
Ax, Ay = 2.32, 1.70   # pilot (top-left)
Bx, By = 4.08, 1.70   # detection (top-right)
Cx, Cy = 4.08, 0.60   # birth (bottom-right)
Dx, Dy = 2.32, 0.60   # refit (bottom-left)
box(Ax, Ay, bw, bh, "pilot transitions +\nfrontier probes", fs=5.8)
box(Bx, By, bw, bh, "transport detection\nshear·scale·polar·hier·$t$", fs=5.4)
box(Cx, Cy, bw, bh, "chart birth +\nresidual surrogates", fs=5.8)
box(Dx, Dy, bw, bh, "refit weights,\nrebuild caches", fs=5.8)
arrow(Ax+bw, Ay+bh/2, Bx, By+bh/2)                 # A -> B
arrow(Bx+bw/2, By, Cx+bw/2, Cy+bh)                 # B -> C (down)
arrow(Cx, Cy+bh/2, Dx+bw, Dy+bh/2)                 # C -> D (left)
arrow(Dx+bw/2, Dy+bh, Ax+bw/2, Ay)                 # D -> A (up)

# stage 3: freeze
box(6.02, 1.02, 1.34, 1.4,
    "FREEZE\n\nsurrogates ($\\kappa$)\nerror-knee $h$\ndefensive $t$",
    fc=FILL2, ec=NUTS, fs=5.9)

# main-axis arrows between stages
arrow(1.74, MID, 2.14, MID)
arrow(5.76, MID, 6.02, MID)

# divider
ax.plot([7.5, 7.5], [0.35, 2.95], ls=":", color=MUT, lw=0.9)

# freeze -> production junction (orthogonal only)
line(7.36, MID, 7.72, MID)
line(7.72, 1.02, 7.72, 2.42)
arrow(7.72, 2.42, 7.88, 2.42)
arrow(7.72, 1.02, 7.88, 1.02)

# production boxes
box(7.88, 2.02, 2.06, 0.82,
    "LOCAL transition:\nchart Gibbs $k\\sim r_k(x)$,\nexact rotation + cached kicks,\none density-oracle MH test", fs=5.5)
box(7.88, 0.62, 2.06, 0.8,
    "GLOBAL transition (prob. $p_g$):\nindependence draw\nfrom defensive $g$", fs=5.8)

fig.savefig(f"{FIG}/warmup_flow.pdf", bbox_inches="tight")
print("done")
