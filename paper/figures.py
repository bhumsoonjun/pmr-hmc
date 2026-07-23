"""All manuscript figures, unified style. Series colors fixed: PMR blue,
NUTS vermillion, exact/theory gray. Direct labels over legends; spines
left+bottom only; grids recessive; text in ink colors, never series colors
(series color only on marks and their direct labels)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PMR, NUTS, EXACT = "#0072B2", "#D55E00", "#888888"
INK, MUT, LINE = "#333333", "#666666", "#BBBBBB"
plt.rcParams.update({
    "font.size": 7.5, "text.color": INK,
    "axes.titlesize": 8, "axes.titlecolor": INK, "axes.labelsize": 7.5,
    "axes.labelcolor": MUT, "axes.edgecolor": "#999999", "axes.linewidth": 0.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "xtick.color": "#999999", "ytick.color": "#999999",
    "xtick.labelcolor": MUT, "ytick.labelcolor": MUT,
    "axes.grid": False, "legend.frameon": False, "figure.dpi": 150,
})
LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(LAB, "paper", "figs")

def style(ax, grid="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, alpha=0.18, lw=0.4)
        ax.set_axisbelow(True)

def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

def banana_U(x0, x1, b=0.1):
    return x0**2/200 + 0.5*(x1 + b*x0**2 - 100*b)**2

import targets as T
from pmr_hmc import run_pmr, GaussComp
from baselines import run_nuts

print("fitting banana/ring (shared across figures)...", flush=True)
tgt_b = T.banana(20)
ch_b, info_b, smp_b = run_pmr(tgt_b, seed=3, n_samples=4000, transport=True,
                              t_defense=True, return_sampler=True)
tgt_r = T.ring(20)
ch_r, _, _ = run_pmr(tgt_r, seed=7, n_samples=4000, transport=True, t_defense=True,
                     return_sampler=True)[:3] if False else (None, None, None)
ch_r, info_r, smp_r = run_pmr(T.ring(20), seed=7, n_samples=4000, transport=True,
                              t_defense=True, return_sampler=True)

# ===================== HERO =====================
fig, axs = plt.subplots(1, 5, figsize=(7.1, 1.8))
xx, yy = np.meshgrid(np.linspace(-30, 30, 160), np.linspace(-24, 15, 160))
Ub = banana_U(xx, yy)
ax = axs[0]
ax.contourf(xx, yy, np.exp(-Ub), levels=7, cmap="Greys")
ax.set_title("target $\\pi$\ncurved, unknown", fontsize=6.8)
ax = axs[1]
Q = np.zeros_like(xx)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        x = np.zeros(20); x[0], x[1] = xx[i, j], yy[i, j]
        Q[i, j] = smp_b.mix.logq(x)
ax.contour(xx, yy, np.exp(Q - Q.max()), levels=6, colors=PMR, linewidths=0.8)
ax.set_title("learn transport atlas\n$q=\\sum_k w_k q_k$", fontsize=6.8)
ax = axs[2]
sub = ch_b[500::5][:700]
g0 = GaussComp(ch_b[500:].mean(0), np.cov(ch_b[500:].T) + 1e-6*np.eye(20))
Ra = np.array([tgt_b.U(x) + smp_b.mix.logq(x) for x in sub])
sc = ax.scatter(sub[:, 0], sub[:, 1], c=Ra - np.median(Ra), s=2.5, cmap="Oranges",
                vmin=-8, vmax=3)
ax.set_title("residual $R=U+\\log q$\nnear-constant on $\\pi$", fontsize=6.8)
ax = axs[3]
tt = np.linspace(0, 0.5*np.pi, 40)
z0, p0 = np.array([1.6, -0.6]), np.array([0.4, 1.2])
zs = np.outer(np.cos(tt), z0) + np.outer(np.sin(tt), p0)
circ = np.linspace(0, 2*np.pi, 100)
for rr_ in (0.9, 1.5, 2.1):
    ax.plot(rr_*np.cos(circ), rr_*np.sin(circ), color=LINE, lw=0.5)
ax.plot(zs[:, 0], zs[:, 1], color=PMR, lw=1.7, solid_capstyle="round")
ax.plot(*zs[0], "o", color=MUT, ms=3.5); ax.plot(*zs[-1], "o", color=NUTS, ms=3.5)
ax.set_aspect("equal")
ax.set_title("latent dynamics: exact\nrotation, zero gradients", fontsize=6.8)
ax = axs[4]
ax.contour(xx, yy, np.exp(-Ub), levels=[0.05, 0.4], colors=LINE, linewidths=0.6)
rngd = np.random.default_rng(3)
x0d = rngd.normal(0, 10, 260); x1d = 10 - 0.1*x0d**2 + rngd.normal(0, 1, 260)
ax.plot(x0d, x1d, ".", ms=1.6, color=PMR, alpha=0.75)
ax.set_title("one density-oracle MH\ntest $\\Rightarrow$ exact draws", fontsize=6.8)
for a in axs: bare(a)
for a in (axs[0], axs[1], axs[2], axs[4]): a.set_xlim(-30, 30); a.set_ylim(-24, 15)
for i in range(4):
    fig.patches.append(FancyArrowPatch((0.187+0.192*i, 0.46), (0.213+0.192*i, 0.46),
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=11, color=MUT))
fig.suptitle("PMR-HMC: flatten geometry once, then sample gradient-free",
             fontsize=9, y=1.06, color=INK)
fig.tight_layout()
fig.savefig(f"{FIG}/hero.pdf", bbox_inches="tight"); plt.close(fig)
print("hero", flush=True)

# ===================== MECHANICS =====================
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.25))
ax = axs[0]
ax.contour(xx, yy, np.exp(-Ub), levels=[0.02, 0.1, 0.4, 0.8], colors=EXACT, linewidths=0.6)
ax.contour(xx, yy, np.exp(Q - Q.max()), levels=5, colors=PMR, linewidths=0.8)
ax.plot(ch_b[500::12, 0], ch_b[500::12, 1], ".", ms=1.5, color=NUTS, alpha=0.55)
ax.set_xlim(-32, 32); ax.set_ylim(-30, 16)
ax.set_title("(a) banana: $\\pi$ (gray), $q$ (blue), draws")
ax.set_xlabel("$x_0$"); ax.set_ylabel("$x_1$")
ax = axs[1]
xs_all = ch_b[500::4]
Rg = np.array([tgt_b.U(x) + g0.logpdf(x) for x in xs_all[:450]])
Ra2 = np.array([tgt_b.U(x) + smp_b.mix.logq(x) for x in xs_all[:450]])
bins = np.linspace(-18, 4, 45)
ax.hist(Rg - np.median(Rg), bins=bins, color=LINE, label="single Gaussian ref.")
ax.hist(Ra2 - np.median(Ra2), bins=bins, color=PMR, alpha=0.88, label="transport atlas")
ax.hist(Rg - np.median(Rg), bins=bins, histtype="step", color=MUT, lw=0.9)
ax.legend(fontsize=6, loc="upper left")
ax.set_title("(b) residual $R$ over draws (centered)")
ax.set_xlabel("$R-\\mathrm{med}\\,R$ [nats]"); ax.set_ylabel("count")
ax = axs[2]
th = np.linspace(0, 2*np.pi, 200)
ax.plot(10*np.cos(th), 10*np.sin(th), color=EXACT, lw=0.8)
ax.plot(ch_r[500::6, 0], ch_r[500::6, 1], ".", ms=1.5, color=PMR, alpha=0.5)
ax.set_aspect("equal")
ax.set_title("(c) ring: draws vs true ridge")
ax.set_xlabel("$x_0$"); ax.set_ylabel("$x_1$")
for a in axs: style(a)
fig.tight_layout()
fig.savefig(f"{FIG}/mechanics.pdf"); plt.close(fig)
print("mechanics", flush=True)

# ===================== RESIDUAL SURFACE =====================
sub2 = xs_all[:900]
Rg2 = np.array([tgt_b.U(x) + g0.logpdf(x) for x in sub2]); Rg2 -= np.median(Rg2)
Ra3 = np.array([tgt_b.U(x) + smp_b.mix.logq(x) for x in sub2]); Ra3 -= np.median(Ra3)
vmin, vmax = min(Rg2.min(), Ra3.min()), max(Rg2.max(), Ra3.max())
fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.4), sharey=True)
for ax, R, ttl in [
    (axs[0], Rg2, f"single-Gaussian reference:\nresidual spans {Rg2.max()-Rg2.min():.0f} nats"),
    (axs[1], Ra3, f"learned transport atlas:\nresidual spans {Ra3.max()-Ra3.min():.0f} nats")]:
    m = ax.scatter(sub2[:, 0], sub2[:, 1], c=R, s=4, cmap="Oranges", vmin=vmin, vmax=vmax)
    ax.set_title(ttl); ax.set_xlabel("$x_0$"); style(ax, grid=None)
axs[0].set_ylabel("$x_1$")
cb = fig.colorbar(m, ax=axs, shrink=0.85)
cb.set_label("$R-\\mathrm{med}\\,R$ [nats]", color=MUT, fontsize=7)
cb.ax.tick_params(labelsize=6.5, labelcolor=MUT, color="#999999")
cb.outline.set_visible(False)
fig.savefig(f"{FIG}/residual_surface.pdf", bbox_inches="tight"); plt.close(fig)
print("residual", flush=True)

# ===================== SCALING =====================
sc_data = json.load(open(f"{LAB}/scaling_results.json"))
def sel(sweep, sampler, key):
    return ([r["x"] for r in sc_data if r["sweep"] == sweep and r["sampler"] == sampler],
            [r[key] for r in sc_data if r["sweep"] == sweep and r["sampler"] == sampler])
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.15))
ax = axs[0]
xs_, ys_ = sel("kappa", "nuts", "units_per_ess")
ax.loglog(xs_, ys_, "o-", color=NUTS, ms=4, lw=1.4)
ax.text(xs_[-1], ys_[-1]*1.7, "NUTS", color=NUTS, fontsize=7, ha="right")
xr = np.array([1e2, 1e5]); ax.loglog(xr, 15*np.sqrt(xr), ":", color=EXACT, lw=0.9)
ax.text(2.4e3, 2.6e3, "$\\propto\\sqrt{\\kappa}$", color=EXACT, fontsize=7)
xs_, ys_ = sel("kappa", "pmr", "units_per_ess")
ax.loglog(xs_, ys_, "s-", color=PMR, ms=4, lw=1.4)
ax.text(xs_[0]*1.1, ys_[0]*2.6, "PMR (amortized)", color=PMR, fontsize=6.4)
xs_, ys_ = sel("kappa", "pmr", "prod_units_per_ess")
ax.loglog(xs_, ys_, "s--", color=PMR, ms=4, lw=1.2, alpha=0.55)
ax.text(xs_[-1], ys_[-1]*0.4, "PMR (production)", color=PMR, fontsize=6.4, ha="right", alpha=0.8)
ax.set_ylim(0.4, 2e4)
ax.set_xlabel("condition number $\\kappa$"); ax.set_ylabel("oracle units / ESS")
ax.set_title("(a) conditioning")
ax = axs[1]
xs_, ys_ = sel("dim", "nuts", "units_per_ess")
ax.semilogy(xs_, ys_, "o-", color=NUTS, ms=4, lw=1.4)
ax.text(xs_[-1], ys_[-1]*2.0, "NUTS", color=NUTS, fontsize=7, ha="right")
xs_, ys_ = sel("dim", "pmr", "prod_units_per_ess")
ax.semilogy(xs_, ys_, "s-", color=PMR, ms=4, lw=1.4)
ax.text(xs_[-1], ys_[-1]*0.38, "PMR (production)", color=PMR, fontsize=6.4, ha="right")
ax.set_xlabel("ambient dimension $d$ (banana)")
ax.set_title("(b) dimension")
ax = axs[2]
xs_, ys_ = sel("barrier", "nuts", "switches")
ax.plot(xs_, ys_, "o-", color=NUTS, ms=4, lw=1.4)
ax.text(xs_[-1]-0.25, 1.8, "NUTS (0 switches)", color=NUTS, fontsize=6.4, ha="right")
xs_, ys_ = sel("barrier", "pmr", "switches")
ax.plot(xs_, ys_, "s-", color=PMR, ms=4, lw=1.4)
ax.text(xs_[-1], ys_[-1]*0.42, "PMR", color=PMR, fontsize=7, ha="right")
ax.set_yscale("symlog", linthresh=1.0); ax.set_ylim(-0.3, 2500)
ax.set_xlabel("mode separation"); ax.set_ylabel("mode switches")
ax.set_title("(c) multimodality")
for a in axs: style(a)
fig.tight_layout()
fig.savefig(f"{FIG}/scaling.pdf"); plt.close(fig)
print("scaling", flush=True)

# ===================== CONVERGENCE =====================
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.15))
tgt = T.mixture2(10, 6.0)
chp, infop = run_pmr(tgt, seed=5, n_samples=6000, transport=True, t_defense=True)
tgt2 = T.mixture2(10, 6.0)
chn, infon = run_nuts(tgt2, seed=5, num_warmup=800, num_samples=4000)
ax = axs[0]
ln1, = ax.plot(chn[::2, 0], color=NUTS, lw=0.5)
ln2, = ax.plot(chp[:4000:2, 0], color=PMR, lw=0.5, alpha=0.62)
ax.set_ylim(-11, 15)
leg = ax.legend([ln1, ln2], ["NUTS (one mode)", "PMR (switching)"], fontsize=6,
                loc="upper right", frameon=True, framealpha=1.0, edgecolor="#CCCCCC")
leg.set_zorder(10)
ax.set_xlabel("iteration"); ax.set_ylabel("$x_0$")
ax.set_title("(a) bimodal trace")
ax = axs[1]
tgt = T.banana(20)
chp2, ip = run_pmr(tgt, seed=5, n_samples=8000, transport=True, t_defense=True)
tgt2 = T.banana(20)
chn2, inn = run_nuts(tgt2, seed=5, num_warmup=800, num_samples=3000)
warm_u = ip["warm_U"] + 2.5*ip["warm_grad"]
units_p = warm_u + np.arange(1, len(chp2)+1)
errp = np.abs(np.cumsum(chp2[:, 0])/np.arange(1, len(chp2)+1)) / 10.0
nsteps = inn["prod_grad"]/3000.0
units_n = 2.5*inn["warm_grad"] + np.arange(1, len(chn2)+1)*2.5*nsteps
errn = np.abs(np.cumsum(chn2[:, 0])/np.arange(1, len(chn2)+1)) / 10.0
ax.loglog(units_p, errp, color=PMR, lw=1.1)
ax.loglog(units_n, errn, color=NUTS, lw=1.1)
ax.text(units_n[-1], errn[-1]*1.6, "NUTS", color=NUTS, fontsize=7, ha="right")
ax.text(0.14, 0.06, "PMR (incl. warm-up)", color=PMR, fontsize=6.2, transform=ax.transAxes)
ax.set_xlabel("cumulative oracle units"); ax.set_ylabel("$|\\hat{E}x_0|/\\sigma$")
ax.set_title("(b) banana convergence")
ax = axs[2]
tgt = T.funnel(10)
chf, _ = run_pmr(tgt, seed=11, n_samples=6000, transport=True, t_defense=True)
rng = np.random.default_rng(0)
v = rng.normal(0, 3, 1400); x1f = rng.normal(0, np.exp(v/2))
ax.plot(x1f, v, ".", ms=1.5, color=LINE, alpha=0.7)
ax.plot(chf[500::4, 1], chf[500::4, 0], ".", ms=1.5, color=PMR, alpha=0.45)
ax.set_xlim(-40, 40)
ax.text(0.96, 0.92, "exact", color=MUT, fontsize=6.5, transform=ax.transAxes, ha="right")
ax.text(0.96, 0.08, "PMR", color=PMR, fontsize=6.5, transform=ax.transAxes, ha="right")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$v$")
ax.set_title("(c) funnel draws vs exact")
for a in axs: style(a)
fig.tight_layout()
fig.savefig(f"{FIG}/convergence.pdf"); plt.close(fig)
print("convergence", flush=True)

# ===================== PDB =====================
# fresh 37-posterior data: pmr from the pdb2r rerun shards, baselines from
# the cached pdb2 battery (the same cells the paper's tables use)
import glob as _glob
pmr_by, nuts_by = {}, {}
for f_ in _glob.glob(f"{LAB}/pdb2r_shard*.json"):
    for r in json.load(open(f_)):
        if "ess_ku" in r:
            pmr_by.setdefault(r["target"], []).append(r)
for r in json.load(open(f"{LAB}/pdb2_results.json")):
    if r.get("sampler") == "nuts" and "ess_ku" in r:
        nuts_by.setdefault(r["target"], []).append(r)
names, ratios, pe, ne = [], [], [], []
for n_ in sorted(pmr_by):
    rs, ns = pmr_by[n_], nuts_by.get(n_, [])
    if not ns: continue
    pm = np.median([x["ess_ku"] for x in rs]); nm = np.median([x["ess_ku"] for x in ns])
    names.append(n_.replace("pdb_", "")); ratios.append(pm/max(nm, 1e-9))
    pe.append(np.median([x["gold_mean"] for x in rs]))
    ne.append(np.median([x["gold_mean"] for x in ns]))
acc_fail = {"kilpis", "earn_h"}
order = np.argsort(ratios)
fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.0), gridspec_kw={"width_ratios": [1.65, 1]})
ax = axs[0]
yy_ = np.arange(len(names))
cols = [NUTS if (names[i] in acc_fail or ratios[i] < 1) else PMR for i in order]
ax.barh(yy_, [ratios[i] for i in order], color=cols, height=0.58)
for r_, i in enumerate(order):
    if names[i] in acc_fail and ratios[i] >= 1:
        ax.text(ratios[i]*1.2, r_, "accuracy fail", fontsize=5.2, va="center", color=NUTS)
ax.set_yticks(yy_); ax.set_yticklabels([names[i] for i in order], fontsize=5.6)
ax.set_xscale("log"); ax.axvline(1.0, color=MUT, lw=0.7)
ax.set_xlabel("PMR / NUTS efficiency ratio (oracle units per ESS)")
ax.set_title("(a) posteriordb, gold-judged (medians, 8 seeds;\nvermillion = loss or accuracy failure)", fontsize=7.4)
style(ax, grid="x")
ax = axs[1]
ax.plot([4e-3, 2], [4e-3, 2], ":", color=EXACT, lw=0.8)
ax.loglog(ne, pe, "o", ms=4.2, color=PMR, mec="white", mew=0.5)
ax.set_xlim(4e-3, 2); ax.set_ylim(4e-3, 2)
ax.set_xlabel("NUTS gold error [sd]"); ax.set_ylabel("PMR gold error [sd]")
ax.set_title("(b) gold error: below diagonal\n= PMR more accurate", fontsize=7.4)
ax.set_aspect("equal")
style(ax, grid=None)
fig.tight_layout()
fig.savefig(f"{FIG}/pdb.pdf"); plt.close(fig)
print("pdb", flush=True)

# ===================== CATALOGUE =====================
rows_n = 5
fig, axs = plt.subplots(rows_n, 2, figsize=(4.5, 8.6))
rng = np.random.default_rng(0)
x = rng.multivariate_normal([0, 0], [[4, 1.9], [1.9, 1]], 550)
axs[0, 0].plot(x[:, 0], x[:, 1], ".", ms=1.3, color=EXACT, alpha=0.45)
L = np.linalg.cholesky([[4, 1.9], [1.9, 1]])
z = np.linalg.solve(L, x.T).T
axs[0, 1].plot(z[:, 0], z[:, 1], ".", ms=1.3, color=PMR, alpha=0.45)
x0 = rng.normal(0, 10, 550); x1 = 10 - 0.1*x0**2 + rng.normal(0, 1, 550)
axs[1, 0].plot(x0, x1, ".", ms=1.3, color=EXACT, alpha=0.45)
axs[1, 1].plot(x0/10, x1 - (10 - 0.1*x0**2), ".", ms=1.3, color=PMR, alpha=0.45)
v = rng.normal(0, 3, 550); xf = rng.normal(0, np.exp(v/2))
axs[2, 0].plot(xf, v, ".", ms=1.3, color=EXACT, alpha=0.45); axs[2, 0].set_xlim(-40, 40)
axs[2, 1].plot(xf*np.exp(-v/2), v/3, ".", ms=1.3, color=PMR, alpha=0.45)
thr = rng.uniform(0, 2*np.pi, 550); rr = 10 + 0.5*rng.normal(size=550)
axs[3, 0].plot(rr*np.cos(thr), rr*np.sin(thr), ".", ms=1.3, color=EXACT, alpha=0.45)
axs[3, 0].set_aspect("equal")
axs[3, 1].plot(((thr+np.pi) % (2*np.pi) - np.pi)/4, (rr-10)/0.5, ".", ms=1.3, color=PMR, alpha=0.45)
xt = rng.standard_t(3, 550)
lam = rng.gamma(2.0, 1.0/(1.5+0.5*xt**2))
axs[4, 0].hist(xt, bins=55, color=EXACT, alpha=0.65, density=True)
axs[4, 1].hist(xt*np.sqrt(lam), bins=38, color=PMR, alpha=0.85, density=True)
labels = [
    ("Affine (Laplace)", "absorb scales + correlations · any smooth mode"),
    ("Quadratic shear", "straighten curved ridges · banana, Rosenbrock"),
    ("Conditional scale", "linearize funnels · Neal's funnel, hierarchies"),
    ("Polar (winding)", "unwrap closed ridges · ring; angle circulates"),
    ("Student-$t$ ($\\lambda$-aux.)", "dominate heavy tails · $t_\\nu$ posteriors"),
]
for i, (ttl, sub) in enumerate(labels):
    for j in range(2): bare(axs[i, j])
    axs[i, 0].set_ylabel(ttl, fontsize=7.2, rotation=0, ha="right", va="center",
                         labelpad=46, color=INK)
    axs[i, 0].set_title("geometry $x$" if i == 0 else "", fontsize=6.5, color=MUT)
    axs[i, 1].set_title("latent $z=T_k^{-1}(x)$" if i == 0 else "", fontsize=6.5, color=PMR)
    axs[i, 0].set_xlabel(sub, fontsize=5.6, color=MUT, labelpad=3)
fig.tight_layout(h_pad=1.4)
fig.savefig(f"{FIG}/catalogue.pdf", bbox_inches="tight"); plt.close(fig)
print("catalogue", flush=True)

# ===================== WINDING =====================
fig, axs = plt.subplots(1, 4, figsize=(7.1, 1.95))
circ = np.linspace(0, 2*np.pi, 200)
ax = axs[0]
ax.plot(10*np.cos(circ), 10*np.sin(circ), color=EXACT, lw=4.5, alpha=0.35,
        solid_capstyle="round")
ax.set_aspect("equal"); ax.set_title("ring target", fontsize=7)
# (2) latent band: wrapped-Gaussian shading + copies
ax = axs[1]
zt = np.linspace(-3.2, 3.2, 300)
zr = np.linspace(-2.2, 2.2, 120)
ZT, ZR = np.meshgrid(zt, zr)
dens = np.exp(-0.5*ZR**2) * np.exp(-0.5*(ZT*0)**2)  # radial Gaussian, uniform angle
ax.contourf(ZT, ZR, dens, levels=6, cmap="Greys", alpha=0.85)
c_wrap = 2*np.pi/4
for k in (-1, 0, 1):
    ax.axvline(k*c_wrap, color=MUT, lw=0.7, ls="--")
ax.text(0, 2.55, "wrapped copies every $2\\pi/c$", fontsize=5.8, color=MUT, ha="center")
ax.set_ylim(-2.3, 3.0)
ax.set_title("latent band $(z_\\theta,z_r)$", fontsize=7)
# (3) harmonic motion crossing wrap lines
ax = axs[2]
ax.contourf(ZT, ZR, dens, levels=6, cmap="Greys", alpha=0.6)
for k in (-1, 0, 1):
    ax.axvline(k*c_wrap, color=MUT, lw=0.6, ls="--", alpha=0.6)
tt3 = np.linspace(0, 3.4, 90)
zth = -2.6 + 1.7*tt3
zrr = 0.9*np.cos(2.1*tt3)
ax.plot(zth, zrr, color=PMR, lw=1.6, solid_capstyle="round")
ax.plot(zth[0], zrr[0], "o", color=MUT, ms=3.5)
ax.plot(zth[-1], zrr[-1], "o", color=NUTS, ms=3.5)
ax.set_ylim(-2.3, 3.0)
ax.set_title("harmonic motion crosses\nwrap lines freely", fontsize=7)
# (4) mapped back
ax = axs[3]
ax.plot(10*np.cos(circ), 10*np.sin(circ), color=EXACT, lw=3.5, alpha=0.3,
        solid_capstyle="round")
thpath = np.linspace(-0.4, 3.6, 90)
rpath = 10 + 0.9*np.cos(2.1*(thpath+0.4)/1.7*2.1)
rpath = 10 + 0.9*np.cos(2.47*thpath + 1.0)
ax.plot(rpath*np.cos(thpath), rpath*np.sin(thpath), color=PMR, lw=1.6)
ax.plot(rpath[0]*np.cos(thpath[0]), rpath[0]*np.sin(thpath[0]), "o", color=MUT, ms=3.5)
ax.plot(rpath[-1]*np.cos(thpath[-1]), rpath[-1]*np.sin(thpath[-1]), "o", color=NUTS, ms=3.5)
ax.set_aspect("equal")
ax.set_title("mapped back: trajectory\ncirculates the ring", fontsize=7)
for a in axs: bare(a)
for i in range(3):
    fig.patches.append(FancyArrowPatch((0.235+0.243*i, 0.48), (0.262+0.243*i, 0.48),
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=11, color=MUT))
fig.tight_layout()
fig.savefig(f"{FIG}/winding.pdf", bbox_inches="tight"); plt.close(fig)
print("winding", flush=True)

# ===================== WARM-UP FLOW =====================
fig, ax = plt.subplots(figsize=(7.0, 2.7))
ax.set_xlim(0, 10); ax.set_ylim(-0.15, 4.05)
bare(ax)
def box(x, y, w, h, text, fc="#EAF2FA", ec=PMR, fs=6.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs, color=INK)
def arr(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=10, color=MUT, lw=0.9))
ax.text(2.2, 3.82, "WARM-UP (adaptive; gradient oracle)", fontsize=7, color=MUT)
ax.text(8.35, 3.82, "PRODUCTION (frozen; zero gradients)", fontsize=7, color=MUT, ha="center")
box(0.15, 2.55, 1.75, 0.8, "multi-start L-BFGS\npaths + Hessians")
box(2.35, 2.55, 1.75, 0.8, "candidate charts\n(evidence-ranked)")
box(4.55, 2.55, 1.75, 0.8, "forward-KL weights\n(provenance pool)")
box(4.55, 1.25, 1.75, 0.8, "chart birth +\nresidual surrogates")
box(2.35, 1.25, 1.75, 0.8, "transport detection\nshear / scale /\npolar / hier / $t$", fs=5.8)
box(0.15, 1.25, 1.75, 0.8, "pilot transitions\n+ frontier probes")
box(0.5, 0.0, 5.7, 0.72,
    "select surrogates ($\\kappa$) · error-knee $h$ · defensive $t$ · FREEZE",
    fc="#FDEBDD", ec=NUTS, fs=6.0)
box(7.0, 2.0, 2.75, 1.15,
    "local: chart Gibbs $k\\sim r_k(x)$\nexact rotation + cached kicks\nONE density-oracle MH test")
box(7.0, 0.5, 2.75, 0.85, "global ($p_g$): independence\ndraw from defensive $g$")
arr(1.9, 2.95, 2.35, 2.95); arr(4.1, 2.95, 4.55, 2.95)
arr(5.42, 2.55, 5.42, 2.05)
arr(4.55, 1.65, 4.1, 1.65)
arr(2.35, 1.65, 1.9, 1.65)
arr(1.02, 1.25, 1.02, 2.55)
arr(3.3, 1.25, 3.3, 0.72)
arr(6.2, 0.36, 7.0, 0.75)
arr(6.2, 0.55, 6.65, 2.3); arr(6.65, 2.3, 7.0, 2.45)
ax.plot([6.72, 6.72], [-0.1, 3.65], ls=":", color=MUT, lw=0.8)
ax.text(0.88, 2.1, "rounds", fontsize=5.5, color=MUT, rotation=90)
fig.savefig(f"{FIG}/warmup_flow.pdf", bbox_inches="tight"); plt.close(fig)
print("flow", flush=True)
print("ALL DONE", flush=True)
