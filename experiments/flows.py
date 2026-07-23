"""Shared learned-transport baselines: RealNVP coupling flow (JAX) trained by
reverse KL, and samplers built on it — TESS (transport elliptical slice,
zero-gradient production), flow-IMH (independence MH from the flow),
flowMC-style (MALA local + flow global), NeuTra (NUTS on the transported
potential). Oracle accounting: flow training charges steps*batch gradient
calls to warmup (autodiff bypasses CountedTarget, so counts are added
manually and reported)."""
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)


def make_flow(d, n_layers=6, hidden=64, key=None):
    keys = jax.random.split(key, n_layers * 4)
    params = []
    for i in range(n_layers):
        k1, k2, k3, k4 = keys[4*i:4*i+4]
        nin = d // 2 if i % 2 == 0 else d - d // 2
        nout = d - nin
        params.append(dict(
            w1=0.05 * jax.random.normal(k1, (nin, hidden)), b1=jnp.zeros(hidden),
            w2=0.05 * jax.random.normal(k2, (hidden, hidden)), b2=jnp.zeros(hidden),
            ws=0.01 * jax.random.normal(k3, (hidden, nout)), bs=jnp.zeros(nout),
            wt=0.01 * jax.random.normal(k4, (hidden, nout)), bt=jnp.zeros(nout)))
    return params


def _split(z, i, d):
    n = d // 2 if i % 2 == 0 else d - d // 2
    return z[..., :n], z[..., n:]


def _join(a, b):
    return jnp.concatenate([a, b], axis=-1)


def flow_forward(params, z):
    d = z.shape[-1]
    logdet = jnp.zeros(z.shape[:-1])
    for i, p in enumerate(params):
        za, zb = _split(z, i, d)
        h = jnp.tanh(za @ p["w1"] + p["b1"])
        h = jnp.tanh(h @ p["w2"] + p["b2"])
        s = 1.5 * jnp.tanh(h @ p["ws"] + p["bs"])
        t = h @ p["wt"] + p["bt"]
        zb = zb * jnp.exp(s) + t
        logdet = logdet + jnp.sum(s, axis=-1)
        z = _join(zb, za) if i % 2 == 0 else _join(za, zb)[..., ::-1][..., ::-1]
        z = _join(zb, za)  # swap halves each layer
    return z, logdet


def flow_inverse(params, x):
    d = x.shape[-1]
    logdet = jnp.zeros(x.shape[:-1])
    for i in reversed(range(len(params))):
        p = params[i]
        n_b = d - (d // 2 if i % 2 == 0 else d - d // 2)
        zb, za = x[..., :n_b], x[..., n_b:]
        h = jnp.tanh(za @ p["w1"] + p["b1"])
        h = jnp.tanh(h @ p["w2"] + p["b2"])
        s = 1.5 * jnp.tanh(h @ p["ws"] + p["bs"])
        t = h @ p["wt"] + p["bt"]
        zb = (zb - t) * jnp.exp(-s)
        logdet = logdet - jnp.sum(s, axis=-1)
        x = _join(za, zb)
    return x, logdet


def train_flow(tgt, key, steps=800, batch=32, lr=2e-3, scale0=None):
    d = tgt.d
    mu0 = jnp.zeros(d)
    sc = jnp.ones(d) * (scale0 if scale0 else tgt.init_scale)
    params = make_flow(d, key=key)
    Ub = jax.vmap(tgt.U_jax)

    def loss(params, zk):
        x, ld = flow_forward(params, zk)
        x = mu0 + sc * x
        return jnp.mean(Ub(x) - ld)  # reverse KL up to const (base entropy fixed)

    opt = optax.adam(lr)
    st = opt.init(params)
    lg = jax.jit(jax.value_and_grad(loss))
    for i in range(steps):
        key, sk = jax.random.split(key)
        zk = jax.random.normal(sk, (batch, d))
        l, g = lg(params, zk)
        up, st = opt.update(g, st)
        params = optax.apply_updates(params, up)
    # manual oracle accounting
    tgt.counts["warmup"]["grad"] += steps * batch
    tgt.counts["warmup"]["U"] += steps * batch

    def T(z):
        x, ld = flow_forward(params, z)
        return mu0 + sc * x, ld + jnp.sum(jnp.log(sc))

    def Tinv(x):
        z, ld = flow_inverse(params, (x - mu0) / sc)
        return z, ld - jnp.sum(jnp.log(sc))

    return params, T, Tinv


def run_tess(tgt, seed, n=6000, train_steps=800):
    """Transport elliptical slice: zero-gradient production; per step evaluates
    the latent log-lik L(z)=-U(T(z))+logdet+||z||^2/2 at each slice proposal."""
    key = jax.random.PRNGKey(seed)
    key, tk = jax.random.split(key)
    t0 = time.perf_counter()
    _, T, _ = train_flow(tgt, tk, steps=train_steps)
    Tj = jax.jit(lambda z: T(z))
    warm = time.perf_counter() - t0

    def L(z):
        x, ld = Tj(z)
        u = tgt.U(np.asarray(x))
        return -u + float(ld) + 0.5 * float(z @ z), np.asarray(x)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(tgt.d)
    Lz, x = L(jnp.asarray(z))
    chain = np.empty((n, tgt.d))
    t1 = time.perf_counter()
    for it in range(n):
        nu = rng.standard_normal(tgt.d)
        logy = Lz + np.log(rng.random())
        th = rng.uniform(0, 2 * np.pi)
        lo, hi = th - 2 * np.pi, th
        while True:
            zp = z * np.cos(th) + nu * np.sin(th)
            Lp, xp = L(jnp.asarray(zp))
            if Lp > logy:
                z, Lz, x = zp, Lp, xp
                break
            if th < 0: lo = th
            else: hi = th
            th = rng.uniform(lo, hi)
        chain[it] = x
    return chain, dict(warm_time=warm, prod_time=time.perf_counter() - t1)


def run_flow_imh(tgt, seed, n=6000, train_steps=800):
    key = jax.random.PRNGKey(seed)
    key, tk = jax.random.split(key)
    t0 = time.perf_counter()
    _, T, _ = train_flow(tgt, tk, steps=train_steps)
    Tj = jax.jit(lambda z: T(z))
    warm = time.perf_counter() - t0
    rng = np.random.default_rng(seed)

    def draw():
        z = rng.standard_normal(tgt.d)
        x, ld = Tj(jnp.asarray(z))
        lq = -0.5 * z @ z - float(ld)
        return np.asarray(x), float(lq)

    x, lqx = draw()
    Ux = tgt.U(x)
    chain = np.empty((n, tgt.d))
    acc = 0
    max_lw = -np.inf
    t1 = time.perf_counter()
    for it in range(n):
        y, lqy = draw()
        Uy = tgt.U(y)
        lw = (-Uy - lqy) - (-Ux - lqx)
        max_lw = max(max_lw, -Uy - lqy)
        if np.isfinite(Uy) and np.log(rng.random()) < min(0.0, lw):
            x, Ux, lqx = y, Uy, lqy
            acc += 1
        chain[it] = x
    return chain, dict(warm_time=warm, prod_time=time.perf_counter() - t1,
                       acc=acc / n)


def run_flowmc(tgt, seed, n=6000, train_steps=800, p_g=0.2, mala_h=None):
    """flowMC-style: MALA local (gradient oracle in production) + flow-IMH
    global moves using the exact flow density via the inverse map."""
    key = jax.random.PRNGKey(seed)
    key, tk = jax.random.split(key)
    t0 = time.perf_counter()
    _, T, Tinv = train_flow(tgt, tk, steps=train_steps)
    Tj = jax.jit(lambda z: T(z))
    Tij = jax.jit(lambda x: Tinv(x))
    warm = time.perf_counter() - t0
    rng = np.random.default_rng(seed)
    h = mala_h or 0.25 * tgt.init_scale / np.sqrt(tgt.d)

    def lq_of(x):
        z, ldi = Tij(jnp.asarray(x))
        return float(-0.5 * np.sum(np.asarray(z) ** 2) + float(ldi))

    x = np.zeros(tgt.d)
    Ux, gx = tgt.value_and_grad(x)
    lqx = lq_of(x)
    chain = np.empty((n, tgt.d))
    accl = accg = nl = ng = 0
    t1 = time.perf_counter()
    for it in range(n):
        if rng.random() < p_g:
            ng += 1
            z = rng.standard_normal(tgt.d)
            y, ld = Tj(jnp.asarray(z))
            y = np.asarray(y)
            lqy = float(-0.5 * z @ z - float(ld))
            Uy = tgt.U(y)
            if np.isfinite(Uy):
                la = (-Uy - lqy) - (-Ux - lqx)
                if np.log(rng.random()) < min(0.0, la):
                    Uy2, gy = tgt.value_and_grad(y)
                    x, Ux, gx, lqx = y, Uy, gy, lqy
                    accg += 1
        else:
            nl += 1
            prop = x - 0.5 * h * h * gx + h * rng.standard_normal(tgt.d)
            Up, gp = tgt.value_and_grad(prop)
            if np.isfinite(Up):
                lf = -np.sum((prop - x + 0.5 * h * h * gx) ** 2) / (2 * h * h)
                lb = -np.sum((x - prop + 0.5 * h * h * gp) ** 2) / (2 * h * h)
                la = (-Up) - (-Ux) + lb - lf
                if np.log(rng.random()) < min(0.0, la):
                    x, Ux, gx = prop, Up, gp
                    lqx = lq_of(x)
                    accl += 1
        chain[it] = x
    return chain, dict(warm_time=warm, prod_time=time.perf_counter() - t1,
                       acc_local=accl / max(nl, 1), acc_global=accg / max(ng, 1))


def run_neutra(tgt, seed, n_samples=2500, train_steps=800):
    """NeuTra-style: NUTS on the flow-transported potential (gradients in prod)."""
    key = jax.random.PRNGKey(seed)
    key, tk = jax.random.split(key)
    t0 = time.perf_counter()
    params, T, _ = train_flow(tgt, tk, steps=train_steps)
    warm = time.perf_counter() - t0

    def U_lat(z):
        x, ld = T(z)
        return tgt.U_jax(x) - ld

    from numpyro.infer import MCMC, NUTS
    kern = NUTS(potential_fn=U_lat)
    m = MCMC(kern, num_warmup=600, num_samples=n_samples, progress_bar=False)
    rng = np.random.default_rng(seed)
    init = jnp.asarray(rng.standard_normal(tgt.d) * 0.5)
    t1 = time.perf_counter()
    m.warmup(jax.random.PRNGKey(seed), init_params=init,
             extra_fields=("num_steps",), collect_warmup=True)
    ws = int(np.sum(np.asarray(m.get_extra_fields()["num_steps"])))
    m.run(jax.random.PRNGKey(seed + 1), extra_fields=("num_steps",))
    ps = int(np.sum(np.asarray(m.get_extra_fields()["num_steps"])))
    zs = np.asarray(m.get_samples())
    xs = np.asarray(jax.vmap(lambda z: T(z)[0])(jnp.asarray(zs)))
    prod = time.perf_counter() - t1
    tgt.counts["production"]["grad"] += ws + ps
    tgt.counts["production"]["U"] += n_samples
    return xs, dict(warm_time=warm, prod_time=prod)
