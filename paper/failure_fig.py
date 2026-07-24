"""Failure-analysis evidence figure: (a) acceptance signatures by failure
class, (b) skew miss visualized, (c) hierarchical collapse visualized,
(d) residual-representation deficit. Data: verify_acc.json,
verify_draws.npz, verify_resid.json produced by verify_failures.py."""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = "/Users/bsoonjun/Documents/GitHub/dataglass/dgbe/tools/pmr-hmc-lab"
FIG = f"{LAB}/paper/figs"
PMR, NUTS, EXACT, MUT, INK = "#0072B2", "#D55E00", "#888888", "#9a9a9a", "#333333"
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUT)
    if grid:
        ax.grid(axis=grid, color="#e6e6e6", lw=0.6)
        ax.set_axisbelow(True)


acc = json.load(open(f"{LAB}/verify_acc.json"))
dr = np.load(f"{LAB}/verify_draws.npz")
res = json.load(open(f"{LAB}/verify_resid.json"))

fig, axs = plt.subplots(1, 4, figsize=(10.2, 2.55))

# (a) acceptance signatures
ax = axs[0]
groups = [("wins", ["m_mix2_10_s4", "m_corr40_r0.95"], PMR),
          ("acquisition", ["m_horse15", "m_skew_a6", "m_hgauss60"], NUTS),
          ("budget/scale", ["m_eightsch_17", "m_funnel60", "m_rosen10", "m_funnel20"], "#E69F00"),
          ("representation", ["m_shell", "m_banana30_b0.2"], "#56B4E9")]
xs, ys, cs, labels, centers = [], [], [], [], []
x = 0
for gname, names, col in groups:
    g0 = x
    for n_ in names:
        xs.append(x); ys.append(acc[n_]); cs.append(col)
        labels.append(n_.replace("m_", ""))
        x += 1
    centers.append((g0 + x - 1) / 2)
    x += 1.2
ax.bar(xs, ys, color=cs, width=0.72)
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=5.2, rotation=90)
ax.set_ylabel("local acceptance")
ax.set_ylim(0, 1.24)
for cx, (gname, names, col) in zip(centers, groups):
    ax.text(cx, 1.09, gname, color=col, fontsize=6.0, ha="center")
ax.set_title("(a) three failure signatures", pad=4)
style(ax)

# (b) skew miss: first-coordinate marginal
ax = axs[1]
b = np.linspace(-2, 8, 60)
ax.hist(dr["nuts_m_skew_a6"][:, 0], bins=b, density=True, color=EXACT,
        alpha=0.85, label="reference")
ax.hist(dr["pmr_m_skew_a6"][:, 0], bins=b, density=True, histtype="step",
        lw=1.4, color=PMR, label="PMR (acc 0.03)")
ax.legend(frameon=False, fontsize=6)
ax.set_title("(b) skew $\\alpha{=}6$: chart starved")
ax.set_xlabel("$x_1$")
style(ax)

# (c) hierarchical collapse: scale coordinate vs child
ax = axs[2]
ax.plot(dr["nuts_m_hgauss60"][:, 0], dr["nuts_m_hgauss60"][:, 2], ".",
        ms=1.6, color=EXACT, alpha=0.5, label="reference")
ax.plot(dr["pmr_m_hgauss60"][:, 0], dr["pmr_m_hgauss60"][:, 2], ".",
        ms=1.6, color=PMR, alpha=0.6, label="PMR (acc 0.03)")
ax.legend(frameon=False, fontsize=6, markerscale=4)
ax.set_title("(c) hgauss60: scale coupling missed")
ax.set_xlabel("log-scale coord."); ax.set_ylabel("child coord.")
style(ax, grid=None)

# (d) residual deficit
ax = axs[3]
names = ["banana10\n$b{=}0.05$ (win)", "banana30\n$b{=}0.2$ (fail)"]
vals = [res["m_banana10_b0.05"]["sd_R"], res["m_banana30_b0.2"]["sd_R"]]
ax.bar([0, 1], vals, color=[PMR, NUTS], width=0.5)
for i, v in enumerate(vals):
    ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)
ax.set_xticks([0, 1]); ax.set_xticklabels(names, fontsize=6.4)
ax.set_ylabel("sd of residual $R$ on reference draws [nats]")
ax.set_title("(d) representation deficit")
style(ax)

fig.tight_layout()
fig.savefig(f"{FIG}/failures.pdf")
print("failures.pdf written")
