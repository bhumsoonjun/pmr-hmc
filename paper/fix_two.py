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

# residual landscape, masked to the typical set
import targets as T
from pmr_hmc import run_pmr, GaussComp
tgt = T.banana(20)
ch, info, smp = run_pmr(tgt, seed=3, n_samples=2500, transport=True, t_defense=True,
                        return_sampler=True)
xs = ch[500::4]
g0 = GaussComp(xs.mean(0), np.cov(xs.T) + 1e-6*np.eye(20))
gx, gy = np.meshgrid(np.linspace(-28, 28, 110), np.linspace(-18, 13, 110))
U2 = banana_U(gx, gy)
mask = U2 < 6.0  # ~typical set of the (x0,x1) marginal
Rg = np.full_like(gx, np.nan); Ra = np.full_like(gx, np.nan)
for i in range(gx.shape[0]):
    for j in range(gx.shape[1]):
        if not mask[i, j]:
            continue
        x = np.zeros(20); x[0], x[1] = gx[i, j], gy[i, j]
        Rg[i, j] = U2[i, j] + g0.logpdf(x)
        Ra[i, j] = U2[i, j] + smp.mix.logq(x)
Rg -= np.nanmedian(Rg); Ra -= np.nanmedian(Ra)
sp_g, sp_a = np.nanmax(Rg)-np.nanmin(Rg), np.nanmax(Ra)-np.nanmin(Ra)
lv = np.linspace(min(np.nanmin(Rg), np.nanmin(Ra)), max(np.nanmax(Rg), np.nanmax(Ra)), 14)
fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.5), sharey=True)
for ax, R, ttl in [(axs[0], Rg, f"single-Gaussian reference\nrange {sp_g:.0f} nats on the ridge"),
                   (axs[1], Ra, f"learned transport atlas\nrange {sp_a:.0f} nats on the ridge")]:
    ax.contour(gx, gy, np.exp(-U2), levels=4, colors="#CCCCCC", linewidths=0.5)
    m = ax.contourf(gx, gy, R, levels=lv, cmap="Oranges")
    ax.set_title(ttl, fontsize=8); ax.set_xlabel("$x_0$")
axs[0].set_ylabel("$x_1$")
fig.colorbar(m, ax=axs, shrink=0.85, label="$R-\\mathrm{med}\\,R$ [nats]")
fig.savefig(f"{FIG}/residual_surface.pdf", bbox_inches="tight"); plt.close(fig)
print("residual v2", np.round([sp_g, sp_a], 1), flush=True)

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
ax.text(2.9, 3.8, "WARM-UP (adaptive; gradient oracle)", fontsize=7.5, color=GRAY)
ax.text(8.1, 3.8, "PRODUCTION (frozen; zero gradients)", fontsize=7.5, color=GRAY, ha="center")
box(0.15, 2.55, 1.75, 0.8, "multi-start L-BFGS\npaths + Hessians")
box(2.35, 2.55, 1.75, 0.8, "candidate charts\n(evidence-ranked)")
box(4.55, 2.55, 1.75, 0.8, "forward-KL weights\n(provenance pool)")
box(4.55, 1.25, 1.75, 0.8, "chart birth +\nresidual surrogates")
box(2.35, 1.25, 1.75, 0.8, "transport detection\nshear / scale /\npolar / hier / $t$", fs=5.9)
box(0.15, 1.25, 1.75, 0.8, "pilot transitions\n+ frontier probes")
box(0.7, 0.0, 5.2, 0.72, "select surrogates ($\\kappa$)  ·  error-knee $h$  ·  defensive $t$  ·  \\bf{FREEZE}",
    fc="#FDEBDD", ec=VERM, fs=6.6)
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
