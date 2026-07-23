import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "figure.dpi": 150})
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
import targets as T
from pmr_hmc import run_pmr, GaussComp
tgt = T.banana(20)
ch, info, smp = run_pmr(tgt, seed=3, n_samples=4000, transport=True, t_defense=True,
                        return_sampler=True)
xs = ch[500::3]
g0 = GaussComp(xs.mean(0), np.cov(xs.T) + 1e-6*np.eye(20))
sub = xs[:900]
Rg = np.array([tgt.U(x) + g0.logpdf(x) for x in sub])
Ra = np.array([tgt.U(x) + smp.mix.logq(x) for x in sub])
Rg -= np.median(Rg); Ra -= np.median(Ra)
vmin = min(Rg.min(), Ra.min()); vmax = max(Rg.max(), Ra.max())
fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.45), sharey=True)
for ax, R, ttl in [
    (axs[0], Rg, f"single-Gaussian reference: residual spans\n{Rg.max()-Rg.min():.0f} nats across the posterior"),
    (axs[1], Ra, f"learned transport atlas: residual spans\n{Ra.max()-Ra.min():.0f} nats (near-constant)")]:
    m = ax.scatter(sub[:, 0], sub[:, 1], c=R, s=4, cmap="Oranges", vmin=vmin, vmax=vmax)
    ax.set_title(ttl, fontsize=8); ax.set_xlabel("$x_0$")
axs[0].set_ylabel("$x_1$")
fig.colorbar(m, ax=axs, shrink=0.85, label="$R-\\mathrm{med}\\,R$ [nats]")
fig.savefig(f"{FIG}/residual_surface.pdf", bbox_inches="tight"); plt.close(fig)
print("spans:", round(float(Rg.max()-Rg.min()), 1), round(float(Ra.max()-Ra.min()), 1))
