"""NUTS baseline via numpyro, with true-gradient accounting from num_steps."""

from __future__ import annotations

import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS


def run_nuts(target, seed=0, num_warmup=1000, num_samples=4000):
    potential = target.U_jax  # NUTS wants the potential (-logp)
    kernel = NUTS(potential_fn=potential)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)
    rng = np.random.default_rng(seed)
    init = jnp.asarray(rng.standard_normal(target.d) * target.init_scale)
    key = jax.random.PRNGKey(seed)

    t0 = time.perf_counter()
    mcmc.warmup(key, init_params=init, extra_fields=("num_steps",), collect_warmup=True)
    warm_steps = int(np.sum(np.asarray(mcmc.get_extra_fields()["num_steps"])))
    warm_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(seed + 1), extra_fields=("num_steps", "accept_prob"))
    prod_time = time.perf_counter() - t1

    extra = mcmc.get_extra_fields()
    prod_steps = int(np.sum(np.asarray(extra["num_steps"])))
    accept = float(np.mean(np.asarray(extra["accept_prob"])))
    chain = np.asarray(mcmc.get_samples())

    info = dict(
        warm_time=warm_time,
        prod_time=prod_time,
        warm_U=0,
        warm_grad=warm_steps,  # ~1 true gradient per leapfrog step
        prod_U=num_samples,  # endpoint energies (generous to NUTS: often fused)
        prod_grad=prod_steps,
        acc_local=accept,
        acc_global=float("nan"),
        h=float("nan"),
        K=0,
        mean_L=prod_steps / max(1, num_samples),
        n_anchors=0,
    )
    return chain, info
