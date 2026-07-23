"""Paper figures. Okabe-Ito palette, fixed order: PMR blue, NUTS vermillion,
ChEES green, MCLMC orange. Thin marks, direct labels, single axes."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = dict(pmr="#0072B2", nuts="#D55E00", chees="#009E73", mclmc="#E69F00",
         gray="#666666", light="#BBBBBB")
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.6,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
                     "legend.frameon": False, "figure.dpi": 150})
LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(LAB, "paper", "figs")

import targets as T
from pmr_hmc import run_pmr
from baselines import run_nuts

# ---------------- fig 1: mechanics (atlas absorbs geometry) ----------------
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.3))
# (a) banana: pi contours + q contours + draws
tgt = T.banana(2) if False else T.banana(20)
ch, info, smp = run_pmr(tgt, seed=3, n_samples=3000, transport=True, t_defense=True,
                        return_sampler=True)
xx, yy = np.meshgrid(np.linspace(-32, 32, 120), np.linspace(-22, 14, 120))
U2 = xx**2/200 + 0.5*(yy + 0.1*xx**2 - 10)**2
ax = axs[0]
ax.contour(xx, yy, np.exp(-U2), levels=[0.02, 0.1, 0.4, 0.8], colors=C["gray"], linewidths=0.7)
Q = np.zeros_like(xx)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        x = np.zeros(20); x[0], x[1] = xx[i, j], yy[i, j]
        Q[i, j] = smp.mix.logq(x)
ax.contour(xx, yy, np.exp(Q - Q.max()), levels=5, colors=C["pmr"], linewidths=0.9)
ax.plot(ch[500::12, 0], ch[500::12, 1], ".", ms=1.2, color=C["nuts"], alpha=0.5)
ax.set_ylim(-30, 16); ax.set_xlim(-32, 32)
ax.set_title("(a) banana: $\\pi$ (gray), learned $q$ (blue), draws", fontsize=7)
ax.set_xlabel("$x_0$"); ax.set_ylabel("$x_1$")
# (b) residual flattening histogram
xs = ch[500::5]
Rs_atlas = np.array([tgt.U(x) + smp.mix.logq(x) for x in xs[:400]])
from pmr_hmc import GaussComp
g0 = GaussComp(xs.mean(0), np.cov(xs.T) + 1e-6*np.eye(20))
Rs_g = np.array([tgt.U(x) + g0.logpdf(x) for x in xs[:400]])
ax = axs[1]
ax.hist(Rs_g - np.median(Rs_g), bins=40, color=C["light"], label="single Gaussian ref.")
ax.hist(Rs_atlas - np.median(Rs_atlas), bins=40, color=C["pmr"], alpha=0.85, label="transport atlas")
ax.set_title("(b) residual $R=U+\\log q$ (centered)", fontsize=7)
ax.set_xlabel("$R-\\mathrm{med}(R)$ [nats]"); ax.legend(fontsize=6)
# (c) ring draws sanity
tgt = T.ring(20)
ch2, info2, smp2 = run_pmr(tgt, seed=7, n_samples=4000, transport=True, t_defense=True,
                           return_sampler=True)
ax = axs[2]
th = np.linspace(0, 2*np.pi, 200)
ax.plot(10*np.cos(th), 10*np.sin(th), color=C["gray"], lw=0.8)
ax.plot(ch2[500::6, 0], ch2[500::6, 1], ".", ms=1.2, color=C["pmr"], alpha=0.5)
ax.set_aspect("equal")
ax.set_title("(c) ring: draws vs true ridge", fontsize=7)
ax.set_xlabel("$x_0$"); ax.set_ylabel("$x_1$")
fig.tight_layout(); fig.savefig(f"{FIG}/mechanics.pdf"); plt.close(fig)
print("fig1 done", flush=True)

# ---------------- fig 2: scaling separations ----------------
sc = json.load(open(f"{LAB}/scaling_results.json"))
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.1))
def sel(sweep, sampler, key):
    return ([r["x"] for r in sc if r["sweep"] == sweep and r["sampler"] == sampler],
            [r[key] for r in sc if r["sweep"] == sweep and r["sampler"] == sampler])
ax = axs[0]
x, y = sel("kappa", "nuts", "units_per_ess"); ax.loglog(x, y, "o-", color=C["nuts"])
ax.text(x[-1], y[-1]*1.4, "NUTS", color=C["nuts"], fontsize=7, ha="right")
x, y = sel("kappa", "pmr", "units_per_ess"); ax.loglog(x, y, "s-", color=C["pmr"])
ax.text(x[0], y[0]*2.6, "PMR (amortized)", color=C["pmr"], fontsize=6.5)
x, y = sel("kappa", "pmr", "prod_units_per_ess"); ax.loglog(x, y, "s--", color=C["pmr"], alpha=0.6)
ax.text(x[-1], y[-1]*0.42, "PMR (production)", color=C["pmr"], fontsize=6.5, ha="right")
ax.set_ylim(0.4, None)
xs_ = np.array([1e2, 1e5]); ax.loglog(xs_, 15*np.sqrt(xs_), ":", color=C["gray"], lw=0.8)
ax.text(2e3, 2.2e3, "$\\propto\\sqrt{\\kappa}$", color=C["gray"], fontsize=7)
ax.set_xlabel("condition number $\\kappa$"); ax.set_ylabel("oracle units / ESS")
ax.set_title("(a) conditioning", fontsize=8)
ax = axs[1]
x, y = sel("dim", "nuts", "units_per_ess"); ax.semilogy(x, y, "o-", color=C["nuts"])
ax.text(x[-1], y[-1]*1.5, "NUTS", color=C["nuts"], fontsize=7, ha="right")
x, y = sel("dim", "pmr", "prod_units_per_ess"); ax.semilogy(x, y, "s-", color=C["pmr"])
ax.text(x[-1], y[-1]*0.4, "PMR (prod.)", color=C["pmr"], fontsize=7, ha="right")
ax.set_xlabel("ambient dimension $d$ (banana)"); ax.set_title("(b) dimension", fontsize=8)
ax = axs[2]
x, y = sel("barrier", "nuts", "switches"); ax.plot(x, y, "o-", color=C["nuts"])
ax.text(x[-1] - 0.3, 1.6, "NUTS (0 switches)", color=C["nuts"], fontsize=6.5, ha="right")
x, y = sel("barrier", "pmr", "switches"); ax.plot(x, y, "s-", color=C["pmr"])
ax.text(x[-1], y[-1]*0.45, "PMR", color=C["pmr"], fontsize=7, ha="right")
ax.set_yscale("symlog", linthresh=1.0); ax.set_ylim(-0.3, 2000)
ax.set_xlabel("mode separation"); ax.set_ylabel("mode switches")
ax.set_title("(c) multimodality", fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/scaling.pdf"); plt.close(fig)
print("fig2 done", flush=True)

# ---------------- fig 3: convergence progression + trace sanity ----------------
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.1))
tgt = T.mixture2(10, 6.0)
chp, infop = run_pmr(tgt, seed=5, n_samples=6000, transport=True, t_defense=True)
tgt2 = T.mixture2(10, 6.0)
chn, infon = run_nuts(tgt2, seed=5, num_warmup=800, num_samples=4000)
ax = axs[0]
ln1, = ax.plot(chn[::2, 0], color=C["nuts"], lw=0.5)
ln2, = ax.plot(chp[:4000:2, 0], color=C["pmr"], lw=0.5, alpha=0.65)
ax.set_ylim(-11, 14.5)
leg = ax.legend([ln1, ln2], ["NUTS (one mode)", "PMR (switching)"], fontsize=6,
          loc="upper right", framealpha=1.0, edgecolor="#999999")
leg.set_zorder(10)
ax.set_xlabel("iteration"); ax.set_ylabel("$x_0$"); ax.set_title("(a) bimodal trace", fontsize=8)
# (b) running mean error vs oracle units (banana)
tgt = T.banana(20)
chp2, ip = run_pmr(tgt, seed=5, n_samples=8000, transport=True, t_defense=True)
tgt2 = T.banana(20)
chn2, inn = run_nuts(tgt2, seed=5, num_warmup=800, num_samples=3000)
ax = axs[1]
warm_u = ip["warm_U"] + 2.5*ip["warm_grad"]
units_p = warm_u + np.arange(1, len(chp2)+1)
errp = np.abs(np.cumsum(chp2[:, 0])/np.arange(1, len(chp2)+1)) / 10.0
nsteps = inn["prod_grad"]/3000.0
units_n = 2.5*inn["warm_grad"] + np.arange(1, len(chn2)+1)*2.5*nsteps
errn = np.abs(np.cumsum(chn2[:, 0])/np.arange(1, len(chn2)+1)) / 10.0
ax.loglog(units_p, errp, color=C["pmr"]); ax.loglog(units_n, errn, color=C["nuts"])
ax.text(0.04, 0.06, "PMR (incl. warm-up)", color=C["pmr"], fontsize=6,
        transform=ax.transAxes)
ax.text(units_n[-1], errn[-1]*1.4, "NUTS", color=C["nuts"], fontsize=7, ha="right")
ax.set_xlabel("cumulative oracle units"); ax.set_ylabel("$|\\hat{E}x_0|/\\sigma$")
ax.set_title("(b) banana convergence", fontsize=8)
# (c) funnel scatter sanity
tgt = T.funnel(10)
chf, _ = run_pmr(tgt, seed=11, n_samples=6000, transport=True, t_defense=True)
ax = axs[2]
rng = np.random.default_rng(0)
v = rng.normal(0, 3, 1500); x1 = rng.normal(0, np.exp(v/2))
ax.plot(x1, v, ".", ms=1.2, color=C["light"], alpha=0.7)
ax.plot(chf[500::4, 1], chf[500::4, 0], ".", ms=1.2, color=C["pmr"], alpha=0.5)
ax.set_xlim(-40, 40)
ax.text(38, 7.5, "exact", color=C["gray"], fontsize=7, ha="right")
ax.text(38, -8.5, "PMR", color=C["pmr"], fontsize=7, ha="right")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$v$"); ax.set_title("(c) funnel draws vs exact", fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/convergence.pdf"); plt.close(fig)
print("fig3 done", flush=True)

# ---------------- fig 4: posteriordb summary ----------------
rows = json.load(open(f"{LAB}/pdb_results.json"))
by = {}
for r in rows:
    if "error" in r or "pmr_ess_ku" not in r: continue
    by.setdefault(r["target"], []).append(r)
names, ratios, pe, ne = [], [], [], []
for n, rs in sorted(by.items()):
    pm = np.median([x["pmr_ess_ku"] for x in rs]); nm = np.median([x["nuts_ess_ku"] for x in rs])
    names.append(n.replace("pdb_", "")); ratios.append(pm/max(nm, 1e-9))
    pe.append(np.median([x["pmr_gold_mean"] for x in rs]))
    ne.append(np.median([x["nuts_gold_mean"] for x in rs]))
acc_fail = {"kilpis", "earn_h"}  # accuracy-gated failures per Sec. 8.3
order = np.argsort(ratios)
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.8), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axs[0]
yy_ = np.arange(len(names))
cols = [C["nuts"] if (names[i] in acc_fail or ratios[i] < 1) else C["pmr"] for i in order]
ax.barh(yy_, [ratios[i] for i in order], color=cols, height=0.62)
for r_, i in enumerate(order):
    if names[i] in acc_fail and ratios[i] >= 1:
        ax.text(ratios[i]*1.15, r_, "accuracy fail", fontsize=5, va="center", color=C["nuts"])
ax.set_yticks(yy_); ax.set_yticklabels([names[i] for i in order], fontsize=6)
ax.set_xscale("log"); ax.axvline(1.0, color=C["gray"], lw=0.8)
ax.set_xlabel("PMR / NUTS efficiency ratio (oracle units per ESS)")
ax.set_title("(a) posteriordb, gold-judged (medians, 8 seeds;\nvermillion = loss or accuracy failure)", fontsize=7)
ax = axs[1]
ax.plot([1e-3, 1], [1e-3, 1], ":", color=C["gray"], lw=0.8)
ax.loglog(ne, pe, "o", ms=4, color=C["pmr"], mec="white", mew=0.5)
ax.set_xlabel("NUTS gold error [sd]"); ax.set_ylabel("PMR gold error [sd]")
ax.set_xlim(5e-3, 2); ax.set_ylim(5e-3, 2)
ax.set_title("(b) gold error: below diagonal\n= PMR more accurate", fontsize=7)
fig.tight_layout(); fig.savefig(f"{FIG}/pdb.pdf"); plt.close(fig)
print("fig4 done", flush=True)
