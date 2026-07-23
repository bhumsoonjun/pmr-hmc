"""Multi-seed complexity figure: medians + IQR bands over 8 seeds."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PMR, NUTS, EXACT = "#0072B2", "#D55E00", "#888888"
INK, MUT = "#333333", "#666666"
plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.titlecolor": INK,
    "axes.labelsize": 7.5, "axes.labelcolor": MUT, "axes.edgecolor": "#999999",
    "axes.linewidth": 0.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "xtick.labelcolor": MUT, "ytick.labelcolor": MUT,
    "xtick.color": "#999999", "ytick.color": "#999999", "figure.dpi": 150})
LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sc = json.load(open(f"{LAB}/scaling8_results.json"))

def stats(sweep, sampler, key):
    xs = sorted({r["x"] for r in sc if r["sweep"] == sweep})
    med, lo, hi = [], [], []
    for x in xs:
        v = np.array([r[key] for r in sc
                      if r["sweep"] == sweep and r["sampler"] == sampler and r["x"] == x])
        med.append(np.median(v)); lo.append(np.percentile(v, 25)); hi.append(np.percentile(v, 75))
    return np.array(xs), np.array(med), np.array(lo), np.array(hi)

def band(ax, x, m, l, h, color, marker, label_xy=None, label=None, ls="-", alpha_l=1.0):
    ax.fill_between(x, l, h, color=color, alpha=0.15, lw=0)
    ax.plot(x, m, marker + ls, color=color, ms=4, lw=1.4, alpha=alpha_l)
    if label:
        ax.text(*label_xy, label, color=color, fontsize=6.4,
                ha="right" if label_xy[0] == x[-1] else "left")

def style(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, lw=0.4); ax.set_axisbelow(True)

fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.15))
ax = axs[0]
x, m, l, h = stats("kappa", "nuts", "units_per_ess")
ax.set_xscale("log"); ax.set_yscale("log")
band(ax, x, m, l, h, NUTS, "o"); ax.text(x[-1], m[-1]*1.8, "NUTS", color=NUTS, fontsize=7, ha="right")
xr = np.array([1e2, 1e5]); ax.plot(xr, 12*np.sqrt(xr), ":", color=EXACT, lw=0.9)
ax.text(2.4e3, 2.2e3, "$\\propto\\sqrt{\\kappa}$", color=EXACT, fontsize=7)
x, m, l, h = stats("kappa", "pmr", "units_per_ess")
band(ax, x, m, l, h, PMR, "s"); ax.text(x[0]*1.1, m[0]*2.4, "PMR (amortized)", color=PMR, fontsize=6.4)
x, m, l, h = stats("kappa", "pmr", "prod_units_per_ess")
band(ax, x, m, l, h, PMR, "s", ls="--", alpha_l=0.55)
ax.text(x[-1], m[-1]*0.42, "PMR (production)", color=PMR, fontsize=6.4, ha="right", alpha=0.8)
ax.set_ylim(0.4, 2e4)
ax.set_xlabel("condition number $\\kappa$"); ax.set_ylabel("oracle units / ESS")
ax.set_title("(a) conditioning")
ax = axs[1]
ax.set_yscale("log")
x, m, l, h = stats("dim", "nuts", "units_per_ess")
band(ax, x, m, l, h, NUTS, "o"); ax.text(x[-1], m[-1]*2.6, "NUTS", color=NUTS, fontsize=7, ha="right")
x, m, l, h = stats("dim", "pmr", "prod_units_per_ess")
band(ax, x, m, l, h, PMR, "s")
ax.text(x[-1], m[-1]*0.35, "PMR (production)", color=PMR, fontsize=6.4, ha="right")
ax.set_xlabel("ambient dimension $d$ (banana)"); ax.set_title("(b) dimension")
ax = axs[2]
x, m, l, h = stats("barrier", "nuts", "switches")
band(ax, x, m, l, h, NUTS, "o")
ax.text(x[-1]-0.25, 2.6, "NUTS ($\\to$ 0 switches)", color=NUTS, fontsize=6.4, ha="right")
x, m, l, h = stats("barrier", "pmr", "switches")
band(ax, x, m, l, h, PMR, "s")
ax.text(x[-1], m[-1]*0.42, "PMR", color=PMR, fontsize=7, ha="right")
ax.set_yscale("symlog", linthresh=1.0); ax.set_ylim(-0.3, 2500)
ax.set_xlabel("mode separation"); ax.set_ylabel("mode switches")
ax.set_title("(c) multimodality")
for a in axs: style(a)
fig.tight_layout()
fig.savefig(f"{LAB}/paper/figs/scaling.pdf")
print("done")
