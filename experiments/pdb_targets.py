"""posteriordb ports: exact Stan-model log-densities (unconstrained + Jacobians)
with constrained-space transforms for gold-reference comparison."""
import json
import zipfile

import jax
import jax.numpy as jnp
import numpy as np

from targets import CountedTarget

PDB = ("/private/tmp/claude-501/-Users-bsoonjun-Documents-GitHub-dataglass-dgbe/"
       "3a8106f8-958c-44f3-af9d-ffb9fbe8022d/scratchpad/posteriordb/posterior_database")


def _load(path):
    z = zipfile.ZipFile(path)
    return json.loads(z.read(z.namelist()[0]))


def data_for(name):
    return _load(f"{PDB}/data/data/{name}.json.zip")


def reference(posterior):
    chains = _load(f"{PDB}/reference_posteriors/draws/draws/{posterior}.json.zip")
    keys = list(chains[0].keys())
    return {k: np.concatenate([np.asarray(c[k]) for c in chains]) for k in keys}


def _cauchy_lpdf(x, s):
    return -jnp.log(1.0 + (x / s) ** 2)


def es_nc():
    d = data_for("eight_schools")
    y, s = jnp.asarray(d["y"], dtype=jnp.float64), jnp.asarray(d["sigma"], dtype=jnp.float64)

    def logp(x):
        tt, mu, lt = x[:8], x[8], x[9]
        tau = jnp.exp(lt)
        th = tt * tau + mu
        return (-0.5 * jnp.sum(tt**2) - 0.5 * jnp.sum(((y - th) / s) ** 2)
                - 0.5 * (mu / 5.0) ** 2 + _cauchy_lpdf(tau, 5.0) + lt)

    def constrained(X):
        tau = np.exp(X[:, 9])
        out = {f"theta[{j+1}]": X[:, j] * tau + X[:, 8] for j in range(8)}
        out["mu"] = X[:, 8]
        out["tau"] = tau
        return out

    t = CountedTarget("pdb_es_nc", 10, logp, init_scale=1.5, defensive=True)
    t.constrained, t.ref_name = constrained, "eight_schools-eight_schools_noncentered"
    return t


def _linreg(name, dataset, ref, cols, yname, prior=None, logy=False, cauchy_sigma=None):
    d = data_for(dataset)
    yv = np.asarray(d[yname], dtype=float)
    if logy:
        yv = np.log(yv)
    Xcols = [np.ones(len(yv))] + [np.asarray(c, dtype=float) for c in cols(d)]
    X = jnp.asarray(np.column_stack(Xcols))
    yj = jnp.asarray(yv)
    p = X.shape[1]

    def logp(x):
        b, ls = x[:p], x[p]
        sig = jnp.exp(ls)
        n = len(yv)
        ll = -0.5 * jnp.sum(((yj - X @ b) / sig) ** 2) - n * ls
        pr = ls  # Jacobian
        if cauchy_sigma:
            pr = pr + _cauchy_lpdf(sig, cauchy_sigma)
        if prior:
            pr = pr - 0.5 * jnp.sum(b**2) / prior**2 - 0.5 * (sig / prior) ** 2
        return ll + pr

    def constrained(Xd):
        out = {f"beta[{j+1}]": Xd[:, j] for j in range(p)}
        out["sigma"] = np.exp(Xd[:, p])
        return out

    t = CountedTarget(name, p + 1, logp, init_scale=1.0)
    t.constrained, t.ref_name = constrained, ref
    return t


def logearn_height():
    return _linreg("pdb_logearn", "earnings", "earnings-logearn_height",
                   lambda d: [d["height"]], "earn", logy=True)


def kid_momhsiq():
    return _linreg("pdb_kidiq", "kidiq", "kidiq-kidscore_momhsiq",
                   lambda d: [d["mom_hs"], d["mom_iq"]], "kid_score", cauchy_sigma=2.5)


def mesquite_logvol():
    def cols(d):
        lv = np.log(np.asarray(d["diam1"]) * np.asarray(d["diam2"])
                    * np.asarray(d["canopy_height"]))
        return [lv]
    return _linreg("pdb_mesquite", "mesquite", "mesquite-logmesquite_logvolume",
                   cols, "weight", logy=True)


def nes(year):
    def cols(d):
        age = np.asarray(d["age_discrete"])
        return [d["real_ideo"], d["race_adj"], (age == 2).astype(float),
                (age == 3).astype(float), (age == 4).astype(float),
                d["educ1"], d["gender"], d["income"]]
    return _linreg(f"pdb_nes{year}", f"nes{year}", f"nes{year}-nes", cols, "partyid7")


def blr(which):
    d = data_for(which)
    X = jnp.asarray(np.asarray(d["X"], dtype=float))
    yj = jnp.asarray(np.asarray(d["y"], dtype=float))
    p = X.shape[1]
    n = X.shape[0]

    def logp(x):
        b, ls = x[:p], x[p]
        sig = jnp.exp(ls)
        ll = -0.5 * jnp.sum(((yj - X @ b) / sig) ** 2) - n * ls
        return ll - 0.5 * jnp.sum(b**2) / 100.0 - 0.5 * sig**2 / 100.0 + ls

    def constrained(Xd):
        out = {f"beta[{j+1}]": Xd[:, j] for j in range(p)}
        out["sigma"] = np.exp(Xd[:, p])
        return out

    t = CountedTarget(f"pdb_{which}", p + 1, logp, init_scale=1.0)
    t.constrained, t.ref_name = constrained, f"{which}-blr"
    return t


def gauss_mix():
    d = data_for("low_dim_gauss_mix")
    yj = jnp.asarray(np.asarray(d["y"], dtype=float))

    def logp(x):
        m1, u2, ls1, ls2, tl = x[0], x[1], x[2], x[3], x[4]
        m2 = m1 + jnp.exp(u2)
        s1, s2 = jnp.exp(ls1), jnp.exp(ls2)
        th = jax.nn.sigmoid(tl)
        a = jnp.log(th) - 0.5 * ((yj - m1) / s1) ** 2 - jnp.log(s1)
        b = jnp.log1p(-th) - 0.5 * ((yj - m2) / s2) ** 2 - jnp.log(s2)
        ll = jnp.sum(jnp.logaddexp(a, b)) - len(d["y"]) * 0.5 * jnp.log(2 * jnp.pi)
        pr = (-0.5 * (s1**2 + s2**2) / 4.0 - 0.5 * (m1**2 + m2**2) / 4.0
              + 4.0 * jnp.log(th) + 4.0 * jnp.log1p(-th))  # beta(5,5)
        jac = u2 + ls1 + ls2 + jnp.log(th) + jnp.log1p(-th)
        return ll + pr + jac

    def constrained(X):
        m2 = X[:, 0] + np.exp(X[:, 1])
        return {"mu[1]": X[:, 0], "mu[2]": m2, "sigma[1]": np.exp(X[:, 2]),
                "sigma[2]": np.exp(X[:, 3]),
                "theta": 1 / (1 + np.exp(-X[:, 4]))}

    t = CountedTarget("pdb_gauss_mix", 5, logp, init_scale=1.0, defensive=True)
    t.constrained, t.ref_name = constrained, "low_dim_gauss_mix-low_dim_gauss_mix"
    return t


def pdb_suite():
    return [es_nc, logearn_height, kid_momhsiq, mesquite_logvol,
            lambda: nes(1992), lambda: nes(1996), lambda: nes(2000),
            lambda: blr("sblri"), lambda: blr("sblrc"), gauss_mix]


def earn_height():
    return _linreg("pdb_earn_h", "earnings", "earnings-earn_height",
                   lambda d: [d["height"]], "earn")


def logearn_height_male():
    return _linreg("pdb_logearn_hm", "earnings", "earnings-logearn_height_male",
                   lambda d: [d["height"], d["male"]], "earn", logy=True)


def logearn_interaction():
    def cols(d):
        h, m = np.asarray(d["height"], dtype=float), np.asarray(d["male"], dtype=float)
        return [h, m, h * m]
    return _linreg("pdb_logearn_ix", "earnings", "earnings-logearn_interaction",
                   cols, "earn", logy=True)


def kid_momhs():
    return _linreg("pdb_kid_hs", "kidiq", "kidiq-kidscore_momhs",
                   lambda d: [d["mom_hs"]], "kid_score", cauchy_sigma=2.5)


def kid_momiq():
    return _linreg("pdb_kid_iq", "kidiq", "kidiq-kidscore_momiq",
                   lambda d: [d["mom_iq"]], "kid_score", cauchy_sigma=2.5)


def kid_interaction():
    def cols(d):
        hs, iq = np.asarray(d["mom_hs"], dtype=float), np.asarray(d["mom_iq"], dtype=float)
        return [hs, iq, hs * iq]
    return _linreg("pdb_kid_ix", "kidiq", "kidiq-kidscore_interaction",
                   cols, "kid_score", cauchy_sigma=2.5)


def logmesquite():
    def cols(d):
        return [np.log(d[k]) for k in ("diam1", "diam2", "canopy_height",
                                       "total_height", "density")] + [np.asarray(d["group"], dtype=float)]
    return _linreg("pdb_mesq_full", "mesquite", "mesquite-logmesquite", cols, "weight", logy=True)


def logmesquite_logvas():
    def cols(d):
        d1, d2, ch = (np.asarray(d[k], dtype=float) for k in ("diam1", "diam2", "canopy_height"))
        return [np.log(d1 * d2 * ch), np.log(d1 * d2), np.log(d1 / d2),
                np.log(np.asarray(d["total_height"], dtype=float)),
                np.log(np.asarray(d["density"], dtype=float)),
                np.asarray(d["group"], dtype=float)]
    return _linreg("pdb_mesq_vas", "mesquite", "mesquite-logmesquite_logvas",
                   cols, "weight", logy=True)


def kilpisjarvi():
    d = data_for("kilpisjarvi_mod")
    xj = jnp.asarray(np.asarray(d["x"], dtype=float))
    yj = jnp.asarray(np.asarray(d["y"], dtype=float))
    pma, psa, pmb, psb = (float(d[k]) for k in ("pmualpha", "psalpha", "pmubeta", "psbeta"))
    n = len(d["y"])

    def logp(x):
        a, b, ls = x[0], x[1], x[2]
        sig = jnp.exp(ls)
        ll = -0.5 * jnp.sum(((yj - a - b * xj) / sig) ** 2) - n * ls
        return (ll - 0.5 * ((a - pma) / psa) ** 2 - 0.5 * ((b - pmb) / psb) ** 2 + ls)

    def constrained(X):
        return {"alpha": X[:, 0], "beta": X[:, 1], "sigma": np.exp(X[:, 2])}

    t = CountedTarget("pdb_kilpis", 3, logp, init_scale=1.0)
    t.constrained, t.ref_name = constrained, "kilpisjarvi_mod-kilpisjarvi"
    return t


def gp_regr():
    d = data_for("gp_pois_regr")
    xv = np.asarray(d["x"], dtype=float)
    yj = jnp.asarray(np.asarray(d["y"], dtype=float))
    D2 = jnp.asarray((xv[:, None] - xv[None, :]) ** 2)
    n = len(xv)

    def logp(x):
        rho, al, sig = jnp.exp(x[0]), jnp.exp(x[1]), jnp.exp(x[2])
        K = al**2 * jnp.exp(-0.5 * D2 / rho**2) + sig * jnp.eye(n)
        L = jnp.linalg.cholesky(K)
        a = jax.scipy.linalg.solve_triangular(L, yj, lower=True)
        ll = -0.5 * a @ a - jnp.sum(jnp.log(jnp.diag(L)))
        pr = (25.0 - 1.0) * jnp.log(rho) - 4.0 * rho - 0.5 * (al / 2.0) ** 2 - 0.5 * sig**2
        return ll + pr + x[0] + x[1] + x[2]

    def constrained(X):
        return {"rho": np.exp(X[:, 0]), "alpha": np.exp(X[:, 1]), "sigma": np.exp(X[:, 2])}

    t = CountedTarget("pdb_gp_regr", 3, logp, init_scale=0.7, defensive=True)
    t.constrained, t.ref_name = constrained, "gp_pois_regr-gp_regr"
    return t


def diamonds():
    d = data_for("diamonds")
    X = np.asarray(d["X"], dtype=float)
    Xc = X[:, 1:] - X[:, 1:].mean(axis=0)
    Xj = jnp.asarray(Xc)
    yj = jnp.asarray(np.asarray(d["Y"], dtype=float))
    kc = Xc.shape[1]
    n = Xc.shape[0]

    def _st_lpdf(v, df, mu, s):
        return -0.5 * (df + 1) * jnp.log1p(((v - mu) / s) ** 2 / df)

    def logp(x):
        b, b0, ls = x[:kc], x[kc], x[kc + 1]
        sig = jnp.exp(ls)
        ll = -0.5 * jnp.sum(((yj - b0 - Xj @ b) / sig) ** 2) - n * ls
        return (ll - 0.5 * jnp.sum(b**2) + _st_lpdf(b0, 3.0, 8.0, 10.0)
                + _st_lpdf(sig, 3.0, 0.0, 10.0) + ls)

    def constrained(Xd):
        out = {f"b[{j+1}]": Xd[:, j] for j in range(kc)}
        out["Intercept"] = Xd[:, kc]
        out["sigma"] = np.exp(Xd[:, kc + 1])
        return out

    t = CountedTarget("pdb_diamonds", kc + 2, logp, init_scale=0.5)
    t.constrained, t.ref_name = constrained, "diamonds-diamonds"
    return t


def pdb_suite_full():
    return pdb_suite() + [
        earn_height, logearn_height_male, logearn_interaction,
        kid_momhs, kid_momiq, kid_interaction,
        logmesquite, logmesquite_logvas, kilpisjarvi, gp_regr, diamonds,
        lambda: nes(1972), lambda: nes(1976), lambda: nes(1980),
        lambda: nes(1984), lambda: nes(1988),
    ]


def log10earn_height():
    d = data_for("earnings")
    yv = np.log10(np.asarray(d["earn"], dtype=float))
    X = jnp.asarray(np.column_stack([np.ones(len(yv)), np.asarray(d["height"], dtype=float)]))
    yj = jnp.asarray(yv)

    def logp(x):
        b, ls = x[:2], x[2]
        return (-0.5 * jnp.sum(((yj - X @ b) / jnp.exp(ls)) ** 2)
                - len(yv) * ls + ls)

    def constrained(Xd):
        return {"beta[1]": Xd[:, 0], "beta[2]": Xd[:, 1], "sigma": np.exp(Xd[:, 2])}

    t = CountedTarget("pdb_log10earn", 3, logp, init_scale=1.0)
    t.constrained, t.ref_name = constrained, "earnings-log10earn_height"
    return t


def logearn_logheight_male():
    def cols(d):
        return [np.log(np.asarray(d["height"], dtype=float)),
                np.asarray(d["male"], dtype=float)]
    return _linreg("pdb_logearn_lhm", "earnings", "earnings-logearn_logheight_male",
                   cols, "earn", logy=True)


def kid_mom_work():
    def cols(d):
        w = np.asarray(d["mom_work"])
        return [(w == 2).astype(float), (w == 3).astype(float), (w == 4).astype(float)]
    return _linreg("pdb_kid_work", "kidiq_with_mom_work", "kidiq_with_mom_work-kidscore_mom_work",
                   cols, "kid_score", cauchy_sigma=2.5)


def _kid_ix(name, ref, center):
    def cols(d):
        hs = np.asarray(d["mom_hs"], dtype=float)
        iq = np.asarray(d["mom_iq"], dtype=float)
        if center == "c":
            hs, iq = hs - hs.mean(), iq - iq.mean()
        elif center == "c2":
            hs, iq = hs - 0.5, iq - 100.0
        else:
            hs = (hs - hs.mean()) / (2 * hs.std(ddof=1))
            iq = (iq - iq.mean()) / (2 * iq.std(ddof=1))
        return [hs, iq, hs * iq]
    return _linreg(name, "kidiq_with_mom_work", ref, cols, "kid_score", cauchy_sigma=2.5)


def kid_ix_c():
    return _kid_ix("pdb_kid_ixc", "kidiq_with_mom_work-kidscore_interaction_c", "c")


def kid_ix_c2():
    return _kid_ix("pdb_kid_ixc2", "kidiq_with_mom_work-kidscore_interaction_c2", "c2")


def kid_ix_z():
    return _kid_ix("pdb_kid_ixz", "kidiq_with_mom_work-kidscore_interaction_z", "z")


def mesq_logva():
    def cols(d):
        d1, d2, ch = (np.asarray(d[k], dtype=float) for k in ("diam1", "diam2", "canopy_height"))
        return [np.log(d1 * d2 * ch), np.log(d1 * d2),
                np.asarray(d["group"], dtype=float)]
    return _linreg("pdb_mesq_va", "mesquite", "mesquite-logmesquite_logva",
                   cols, "weight", logy=True)


def mesq_logvash():
    def cols(d):
        d1, d2, ch = (np.asarray(d[k], dtype=float) for k in ("diam1", "diam2", "canopy_height"))
        return [np.log(d1 * d2 * ch), np.log(d1 * d2), np.log(d1 / d2),
                np.log(np.asarray(d["total_height"], dtype=float)),
                np.asarray(d["group"], dtype=float)]
    return _linreg("pdb_mesq_vash", "mesquite", "mesquite-logmesquite_logvash",
                   cols, "weight", logy=True)


def ark():
    d = data_for("arK")
    yv = np.asarray(d["y"], dtype=float)
    K, T = int(d["K"]), int(d["T"])
    # design: y[t] ~ alpha + sum_k beta_k y[t-k], t = K..T-1
    rows = np.column_stack([yv[K - 1 - k: T - 1 - k] for k in range(K)])
    Xj = jnp.asarray(rows)
    yj = jnp.asarray(yv[K:])
    n = T - K

    def logp(x):
        a, b, ls = x[0], x[1:1 + K], x[K + 1]
        sig = jnp.exp(ls)
        ll = -0.5 * jnp.sum(((yj - a - Xj @ b) / sig) ** 2) - n * ls
        pr = (-0.5 * (a / 10) ** 2 - 0.5 * jnp.sum((b / 10) ** 2)
              + _cauchy_lpdf(sig, 2.5) + ls)
        return ll + pr

    def constrained(Xd):
        out = {"alpha": Xd[:, 0]}
        for j in range(K):
            out[f"beta[{j+1}]"] = Xd[:, 1 + j]
        out["sigma"] = np.exp(Xd[:, K + 1])
        return out

    t = CountedTarget("pdb_arK", K + 2, logp, init_scale=0.7)
    t.constrained, t.ref_name = constrained, "arK-arK"
    return t


def arma11():
    d = data_for("arma")
    yv = jnp.asarray(np.asarray(d["y"], dtype=float))
    T = int(d["T"])

    def logp(x):
        mu, phi, th, ls = x[0], x[1], x[2], x[3]
        sig = jnp.exp(ls)

        def step(err_prev, t):
            nu = mu + phi * yv[t - 1] + th * err_prev
            err = yv[t] - nu
            return err, err

        err1 = yv[0] - (mu + phi * mu)
        _, errs = jax.lax.scan(step, err1, jnp.arange(1, T))
        allerr = jnp.concatenate([jnp.array([err1]), errs])
        ll = -0.5 * jnp.sum((allerr / sig) ** 2) - T * ls
        pr = (-0.5 * (mu / 10) ** 2 - 0.5 * (phi / 2) ** 2 - 0.5 * (th / 2) ** 2
              + _cauchy_lpdf(sig, 2.5) + ls)
        return ll + pr

    def constrained(Xd):
        return {"mu": Xd[:, 0], "phi": Xd[:, 1], "theta": Xd[:, 2],
                "sigma": np.exp(Xd[:, 3])}

    t = CountedTarget("pdb_arma11", 4, logp, init_scale=0.7)
    t.constrained, t.ref_name = constrained, "arma-arma11"
    return t


def garch11():
    d = data_for("garch")
    yv = jnp.asarray(np.asarray(d["y"], dtype=float))
    T = int(d["T"])
    s1 = float(d["sigma1"])

    def logp(x):
        mu, la0, u1, u2 = x[0], x[1], x[2], x[3]
        a0 = jnp.exp(la0)
        a1 = jax.nn.sigmoid(u1)
        b1 = (1.0 - a1) * jax.nn.sigmoid(u2)

        def step(sprev, t):
            s = jnp.sqrt(a0 + a1 * (yv[t - 1] - mu) ** 2 + b1 * sprev**2)
            return s, s

        _, ss = jax.lax.scan(step, jnp.asarray(s1), jnp.arange(1, T))
        sall = jnp.concatenate([jnp.array([s1]), ss])
        ll = -0.5 * jnp.sum(((yv - mu) / sall) ** 2) - jnp.sum(jnp.log(sall))
        jac = (la0 + jnp.log(a1) + jnp.log1p(-a1)
               + jnp.log1p(-a1) + jnp.log(jax.nn.sigmoid(u2))
               + jnp.log(jax.nn.sigmoid(-u2)))
        return ll + jac

    def constrained(Xd):
        a1 = 1 / (1 + np.exp(-Xd[:, 2]))
        return {"mu": Xd[:, 0], "alpha0": np.exp(Xd[:, 1]), "alpha1": a1,
                "beta1": (1 - a1) / (1 + np.exp(-Xd[:, 3]))}

    t = CountedTarget("pdb_garch11", 4, logp, init_scale=0.5, defensive=True)
    t.constrained, t.ref_name = constrained, "garch-garch11"
    return t


def pdb_suite_v3():
    return pdb_suite_full() + [
        log10earn_height, logearn_logheight_male, kid_mom_work,
        kid_ix_c, kid_ix_c2, kid_ix_z, mesq_logva, mesq_logvash,
        ark, arma11, garch11,
    ]
