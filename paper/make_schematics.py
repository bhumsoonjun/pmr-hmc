"""Explanatory schematics: hero pipeline, chart catalogue, winding unwrap,
residual landscape, warm-up flowchart. All geometry computed from the real
transforms. Palette: PMR blue #0072B2, accent vermillion #D55E00, gray."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, VERM, GRAY, LGRAY = "#0072B2", "#D55E00", "#555555", "#BBBBBB"
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "axes.grid": False,
                     "figure.dpi": 150})
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")

def banana_U(x0, x1, b=0.1):
    return x0**2/200 + 0.5*(x1 + b*x0**2 - 100*b)**2

def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

# ================= 1. HERO PIPELINE =================
fig, axs = plt.subplots(1, 5, figsize=(7.1, 1.85))
xx, yy = np.meshgrid(np.linspace(-30, 30, 160), np.linspace(-20, 14, 160))
Ub = banana_U(xx, yy)
# (1) target
ax = axs[0]; ax.contourf(xx, yy, np.exp(-Ub), levels=8, cmap="Greys")
ax.set_title("target $\\pi$\n(curved, unknown)", fontsize=7); clean(ax)
# (2) atlas: shear-chart contours + 2 affine ellipses
ax = axs[1]
qsh = np.exp(-(xx**2/200)/1.0 - 0.5*(yy + 0.1*xx**2 - 10)**2)
ax.contour(xx, yy, qsh, levels=5, colors=BLUE, linewidths=0.9)
th = np.linspace(0, 2*np.pi, 60)
for cx, sx in [(-14, 5), (14, 5)]:
    ax.plot(cx+sx*np.cos(th), (10-0.1*cx**2)+2.2*np.sin(th), color=VERM, lw=0.9)
ax.set_title("learn transport atlas\n$q=\\sum_k w_k q_k$", fontsize=7); clean(ax)
ax.set_xlim(-30, 30); ax.set_ylim(-20, 14)
# (3) residual flattens
ax = axs[2]
r_before = Ub - (xx**2/200 + 0.5*(yy-0)**2/60)      # vs a single blob
r_after = np.zeros_like(Ub) + 0.2*np.exp(-((xx/25)**2+(yy/12)**2))
ax.contourf(xx, yy, np.clip(r_before, 0, 30), levels=8, cmap="Oranges")
ax.contour(xx, yy, r_after, levels=3, colors=BLUE, linewidths=0.8)
ax.set_title("residual $R=U+\\log q$\nflattens (blue $\\approx$ const)", fontsize=7); clean(ax)
# (4) latent harmonic dynamics
ax = axs[3]
tt = np.linspace(0, 0.5*np.pi, 40)
z0, p0 = np.array([1.6, -0.6]), np.array([0.4, 1.2])
zs = np.outer(np.cos(tt), z0) + np.outer(np.sin(tt), p0)
circ = np.linspace(0, 2*np.pi, 100)
ax.plot(1.8*np.cos(circ), 1.8*np.sin(circ), color=LGRAY, lw=0.7)
ax.plot(zs[:, 0], zs[:, 1], color=BLUE, lw=1.6)
ax.plot(*zs[0], "o", color=GRAY, ms=4); ax.plot(*zs[-1], "o", color=VERM, ms=4)
ax.set_title("latent dynamics:\nexact rotation, no gradients", fontsize=7)
ax.set_aspect("equal"); clean(ax)
# (5) MH endpoint on target
ax = axs[4]
ax.contour(xx, yy, np.exp(-Ub), levels=5, colors=LGRAY, linewidths=0.6)
rngd = np.random.default_rng(3)
x0d = rngd.normal(0, 10, 250); x1d = 10 - 0.1*x0d**2 + rngd.normal(0, 1, 250)
ax.plot(x0d, x1d, ".", ms=1.4, color=BLUE, alpha=0.7)
ax.set_title("one true-density MH test\n$\\Rightarrow$ exact draws", fontsize=7); clean(ax)
for i in range(4):
    fig.patches.append(FancyArrowPatch((0.185+0.192*i, 0.5), (0.215+0.192*i, 0.5),
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=12, color=GRAY))
fig.suptitle("PMR-HMC: flatten geometry once, then sample gradient-free", fontsize=9, y=1.04)
fig.tight_layout(); fig.savefig(f"{FIG}/hero.pdf", bbox_inches="tight"); plt.close(fig)
print("hero", flush=True)

# ================= 2. CHART CATALOGUE =================
rows = 5
fig, axs = plt.subplots(rows, 2, figsize=(4.6, 9.2))
rng = np.random.default_rng(0)
# affine
x = rng.multivariate_normal([0, 0], [[4, 1.9], [1.9, 1]], 600)
axs[0, 0].plot(x[:, 0], x[:, 1], ".", ms=1.2, color=GRAY, alpha=0.5)
L = np.linalg.cholesky([[4, 1.9], [1.9, 1]])
z = np.linalg.solve(L, x.T).T
axs[0, 1].plot(z[:, 0], z[:, 1], ".", ms=1.2, color=BLUE, alpha=0.5)
# shear
x0 = rng.normal(0, 10, 600); x1 = 10 - 0.1*x0**2 + rng.normal(0, 1, 600)
axs[1, 0].plot(x0, x1, ".", ms=1.2, color=GRAY, alpha=0.5)
axs[1, 1].plot(x0/10, x1 - (10 - 0.1*x0**2), ".", ms=1.2, color=BLUE, alpha=0.5)
# conditional scale (funnel)
v = rng.normal(0, 3, 600); xf = rng.normal(0, np.exp(v/2))
axs[2, 0].plot(xf, v, ".", ms=1.2, color=GRAY, alpha=0.5); axs[2, 0].set_xlim(-40, 40)
axs[2, 1].plot(xf*np.exp(-v/2), v/3, ".", ms=1.2, color=BLUE, alpha=0.5)
# polar (ring)
thr = rng.uniform(0, 2*np.pi, 600); rr = 10 + 0.5*rng.normal(size=600)
axs[3, 0].plot(rr*np.cos(thr), rr*np.sin(thr), ".", ms=1.2, color=GRAY, alpha=0.5)
axs[3, 0].set_aspect("equal")
axs[3, 1].plot(((thr+np.pi) % (2*np.pi) - np.pi)/4, (rr-10)/0.5, ".", ms=1.2, color=BLUE, alpha=0.5)
# student-t
xt = rng.standard_t(3, 600)*1.0
lam = rng.gamma(1.5+0.5, 1.0/(1.5+0.5*xt**2))
axs[4, 0].hist(xt, bins=60, color=GRAY, alpha=0.7, density=True)
axs[4, 1].hist(xt*np.sqrt(lam), bins=40, color=BLUE, alpha=0.8, density=True)
labels = [
    ("Affine (Laplace)", "purpose: scales+correlations · example: any smooth mode"),
    ("Quadratic shear", "purpose: straighten curved ridges · example: banana, Rosenbrock"),
    ("Conditional scale", "purpose: linearize funnels · example: Neal's funnel, hierarchies"),
    ("Polar (winding)", "purpose: unwrap closed ridges · example: ring; angle circulates"),
    ("Student-$t$ ($\\lambda$-aux.)", "purpose: dominate heavy tails · example: $t_\\nu$ posteriors"),
]
for i, (ttl, sub) in enumerate(labels):
    for j in range(2): clean(axs[i, j])
    axs[i, 0].set_ylabel(ttl, fontsize=7.5, rotation=0, ha="right", va="center", labelpad=44)
    axs[i, 0].set_title("geometry $x$", fontsize=6.5, color=GRAY)
    axs[i, 1].set_title("latent $z=T_k^{-1}(x)$", fontsize=6.5, color=BLUE)
    axs[i, 0].set_xlabel(sub, fontsize=5.8, color=GRAY, labelpad=2)
fig.tight_layout(); fig.savefig(f"{FIG}/catalogue.pdf", bbox_inches="tight"); plt.close(fig)
print("catalogue", flush=True)

# ================= 3. WINDING UNWRAP SEQUENCE =================
fig, axs = plt.subplots(1, 4, figsize=(7.1, 1.9))
ax = axs[0]
ax.plot(10*np.cos(circ), 10*np.sin(circ), color=GRAY, lw=4, alpha=0.35)
ax.set_title("ring target", fontsize=7); ax.set_aspect("equal"); clean(ax)
ax = axs[1]
for k in (-1, 0, 1):
    ax.axvspan(k*2*np.pi/4-0.02, k*2*np.pi/4+0.02, color=LGRAY)
zt = np.linspace(-2.6, 2.6, 100)
ax.fill_between(zt, -1, 1, color=GRAY, alpha=0.25)
ax.set_title("latent $(z_\\theta,z_r)$: wrapped\ncopies every $2\\pi/c$", fontsize=7); clean(ax)
ax = axs[2]
tt2 = np.linspace(0, 2.4, 60)
ztr = 0.5*np.cos(tt2) + 1.9*np.sin(tt2)
ax.fill_between(zt, -1, 1, color=GRAY, alpha=0.18)
ax.plot(0.9*tt2-1.4, 0.6*np.cos(2.2*tt2), color=BLUE, lw=1.6)
ax.set_title("harmonic motion in $z_\\theta$:\nfree travel along the band", fontsize=7); clean(ax)
ax = axs[3]
thpath = np.linspace(-0.4, 3.4, 80)
rpath = 10 + 0.45*np.cos(5*thpath)
ax.plot(10*np.cos(circ), 10*np.sin(circ), color=LGRAY, lw=3, alpha=0.4)
ax.plot(rpath*np.cos(thpath), rpath*np.sin(thpath), color=BLUE, lw=1.6)
ax.plot(rpath[0]*np.cos(thpath[0]), rpath[0]*np.sin(thpath[0]), "o", color=GRAY, ms=4)
ax.plot(rpath[-1]*np.cos(thpath[-1]), rpath[-1]*np.sin(thpath[-1]), "o", color=VERM, ms=4)
ax.set_title("mapped back: trajectory\ncirculates the ring", fontsize=7)
ax.set_aspect("equal"); clean(ax)
for i in range(3):
    fig.patches.append(FancyArrowPatch((0.235+0.24*i, 0.5), (0.265+0.24*i, 0.5),
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=12, color=GRAY))
fig.tight_layout(); fig.savefig(f"{FIG}/winding.pdf", bbox_inches="tight"); plt.close(fig)
print("winding", flush=True)

# ================= 4. RESIDUAL LANDSCAPE (real fit) =================
import targets as T
from pmr_hmc import run_pmr, GaussComp
tgt = T.banana(20)
ch, info, smp = run_pmr(tgt, seed=3, n_samples=2500, transport=True, t_defense=True,
                        return_sampler=True)
xs = ch[500::4]
g0 = GaussComp(xs.mean(0), np.cov(xs.T) + 1e-6*np.eye(20))
gx, gy = np.meshgrid(np.linspace(-28, 28, 90), np.linspace(-18, 13, 90))
Rg = np.zeros_like(gx); Ra = np.zeros_like(gx)
for i in range(gx.shape[0]):
    for j in range(gx.shape[1]):
        x = np.zeros(20); x[0], x[1] = gx[i, j], gy[i, j]
        u = banana_U(gx[i, j], gy[i, j])
        Rg[i, j] = u + g0.logpdf(x)
        Ra[i, j] = u + smp.mix.logq(x)
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
lv = np.linspace(-40, 5, 12)
m0 = axs[0].contourf(gx, gy, np.clip(Rg - np.median(Rg), -40, 5), levels=lv, cmap="Oranges_r")
axs[0].set_title("single-Gaussian reference:\n$R$ is a mountain range (40+ nats)", fontsize=8)
m1 = axs[1].contourf(gx, gy, np.clip(Ra - np.median(Ra), -40, 5), levels=lv, cmap="Oranges_r")
axs[1].set_title("learned transport atlas:\n$R$ nearly flat where $\\pi$ lives", fontsize=8)
for a in axs: a.set_xlabel("$x_0$")
axs[0].set_ylabel("$x_1$")
fig.colorbar(m1, ax=axs, shrink=0.85, label="$R-\\mathrm{med}\\,R$ [nats]")
fig.savefig(f"{FIG}/residual_surface.pdf", bbox_inches="tight"); plt.close(fig)
print("residual", flush=True)

# ================= 5. WARM-UP FLOWCHART =================
fig, ax = plt.subplots(figsize=(6.9, 2.6)); clean(ax)
ax.set_xlim(0, 10); ax.set_ylim(0, 4)
def box(x, y, w, h, text, fc="#EAF2FA", ec=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=6.6)
def arr(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=10, color=GRAY, lw=0.9))
ax.text(2.6, 3.75, "WARM-UP (adaptive, gradient oracle)", fontsize=7.5, color=GRAY)
ax.text(8.15, 3.75, "PRODUCTION (frozen, zero gradients)", fontsize=7.5, color=GRAY)
box(0.1, 2.6, 1.7, 0.75, "multi-start L-BFGS\npaths + Hessians")
box(2.2, 2.6, 1.7, 0.75, "candidate charts\n(evidence-ranked)")
box(4.3, 2.6, 1.7, 0.75, "forward-KL weights\n(provenance pool)")
box(0.1, 1.2, 1.7, 0.75, "pilot transitions\n+ frontier probes")
box(2.2, 1.2, 1.7, 0.75, "transport detection\nshear/scale/polar/hier/$t$")
box(4.3, 1.2, 1.7, 0.75, "chart birth +\nresidual surrogates")
box(2.2, 0.05, 3.8, 0.7, "select surrogates ($\\kappa$) · error-knee $h$ · defensive $t$ · FREEZE",
    fc="#FDEBDD", ec=VERM)
box(7.0, 1.9, 2.6, 1.1, "chart Gibbs $k\\sim r_k(x)$\n$\\rightarrow$ exact rotation + cached kicks\n$\\rightarrow$ ONE density MH test", fc="#EAF2FA")
box(7.0, 0.45, 2.6, 0.8, "global branch ($p_g$):\nindependence draw from $g$", fc="#EAF2FA")
arr(1.8, 2.97, 2.2, 2.97); arr(3.9, 2.97, 4.3, 2.97)
arr(5.15, 2.6, 5.15, 2.35); arr(4.3, 1.57, 3.9, 1.57); arr(2.2, 1.57, 1.8, 1.57)
arr(0.95, 1.2, 0.95, 0.75); arr(1.0, 0.55, 2.2, 0.45)
arr(6.0, 0.4, 7.0, 0.85); arr(6.35, 2.4, 7.0, 2.4)
ax.plot([6.6, 6.6], [0.0, 3.6], ls=":", color=GRAY, lw=0.8)
fig.savefig(f"{FIG}/warmup_flow.pdf", bbox_inches="tight"); plt.close(fig)
print("flow done", flush=True)
