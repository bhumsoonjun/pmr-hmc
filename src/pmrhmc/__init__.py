"""pmrhmc — learned transport atlases for exact, gradient-free production MCMC.

Warm-up (Python/numpy) learns a mixture-of-charts reference for the target;
production sampling runs in a portable C kernel with ONE density evaluation
per transition and zero gradients. Exactness holds by construction (true
endpoint Metropolis-Hastings on the augmented target).

    import pmrhmc

    s = pmrhmc.PMRHMC(logpdf, d=20)          # logpdf: unnormalized log-density
    s.fit(seed=0)                             # learn the transport atlas
    draws = s.sample(100_000, chains=4, cores=4, seed=1)

Everything is float64; densities are handled in log space end to end.
"""
from __future__ import annotations

import numpy as np

from .target import Target
from .warmup import run_pmr

__version__ = "0.1.0"
__all__ = ["PMRHMC", "Target", "run_pmr", "__version__"]


class PMRHMC:
    """High-level API: fit a transport atlas, then draw with the C kernel."""

    def __init__(self, logpdf, d, grad_logpdf=None, logpdf_batch=None,
                 init_scale=3.0, name="target", native_spec=None):
        self.target = Target(logpdf, d, grad_logpdf=grad_logpdf,
                             logpdf_batch=logpdf_batch, init_scale=init_scale,
                             name=name)
        self._logpdf = logpdf
        self.native_spec = native_spec  # optional (tid, par, X, y) fast path
        self.sampler = None
        self.fit_info = None

    def fit(self, seed=0, transport=True, t_defense=True, **warmup_kw):
        """Learn the atlas (multi-start L-BFGS charts, forward-KL weights,
        residual-guided birth, transport detection, step-size calibration).
        This is the only phase that may use gradients."""
        chain, info, sampler = run_pmr(
            self.target, seed=seed, transport=transport, t_defense=t_defense,
            n_samples=1, return_sampler=True, **warmup_kw)
        self.sampler = sampler
        self.fit_info = info
        return self

    def sample(self, n, seed=1, chains=1, cores=1, burn=None):
        """Draw from the frozen atlas with the C production kernel.

        chains/cores: independent chains over the shared atlas; native-family
        targets are threaded in C (pthreads), callback targets in Python
        threads (the C side releases the GIL between callbacks).
        Returns (chains, n_kept, d) — or (n_kept, d) when chains == 1."""
        if self.sampler is None:
            raise RuntimeError("call fit() before sample()")
        from . import native
        pk = native.pack_sampler(self.sampler)
        burn = int(0.05 * n) if burn is None else burn
        if chains == 1:
            if self.native_spec is not None:
                ch, st = native.run_native(self.native_spec, self.sampler, n, seed)
            else:
                ch, st = native.run_cb(self._logpdf, self.sampler, n, seed, pk=pk)
            self.last_stats = st
            return ch[burn:]
        seeds = np.uint64(seed) + np.arange(chains, dtype=np.uint64)
        # overdispersed starts: one atlas draw per chain
        rng = np.random.default_rng(int(seed))
        x0s = np.stack([self.sampler.mix.sample(rng, 1)[0] for _ in range(chains)])
        if self.native_spec is not None:
            chs, st = native.run_native_multi(self.native_spec, self.sampler, n,
                                              chains, seeds=seeds, x0s=x0s,
                                              cores=cores, pk=pk)
        else:
            chs, st = native.run_multi_cb(self._logpdf, self.sampler, n, chains,
                                          seeds=seeds, x0s=x0s, cores=cores, pk=pk)
        self.last_stats = st
        return chs[:, burn:]

    def sample_py(self, n):
        """Pure-Python production sampling (no C): reference implementation,
        bit-for-bit the semantics the exactness proofs cover."""
        if self.sampler is None:
            raise RuntimeError("call fit() before sample()")
        self.target.set_phase("production")
        return self.sampler.run(n)

    @property
    def diagnostics(self):
        d = dict(self.fit_info or {})
        d.pop("mixture", None)
        return d
