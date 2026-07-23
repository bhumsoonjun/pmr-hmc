"""Target adapter: wraps a user log-density into the interface the warm-up
engine expects (U / U_batch / value_and_grad / hessian, phase-tagged oracle
counters). Pure numpy; gradients fall back to central finite differences when
the user does not supply them (each FD gradient is charged as 2d density
evaluations, so oracle accounting stays honest)."""
from __future__ import annotations

import numpy as np


class Target:
    def __init__(self, logpdf, d, grad_logpdf=None, logpdf_batch=None,
                 name="target", init_scale=3.0, defensive=False, fd_eps=1e-6):
        """logpdf: x (d,) -> float, the UNNORMALIZED log density.
        grad_logpdf: optional x -> (d,) gradient of logpdf.
        logpdf_batch: optional X (n,d) -> (n,) for faster warm-up pooling."""
        self.name = name
        self.d = int(d)
        self.init_scale = float(init_scale)
        self.defensive = defensive
        self.fd_eps = float(fd_eps)
        self.truth_mean = None
        self.truth_var = None
        self._lp = logpdf
        self._glp = grad_logpdf
        self._lpb = logpdf_batch
        self.counts = {"warmup": {"U": 0, "grad": 0},
                       "production": {"U": 0, "grad": 0}}
        self.phase = "warmup"

    def set_phase(self, phase):
        assert phase in self.counts
        self.phase = phase

    def U(self, x):
        self.counts[self.phase]["U"] += 1
        return float(-self._lp(np.asarray(x, dtype=float)))

    def U_batch(self, X):
        X = np.asarray(X, dtype=float)
        self.counts[self.phase]["U"] += len(X)
        if self._lpb is not None:
            return -np.asarray(self._lpb(X), dtype=float)
        return np.array([-float(self._lp(x)) for x in X])

    def value_and_grad(self, x):
        x = np.asarray(x, dtype=float)
        c = self.counts[self.phase]
        if self._glp is not None:
            c["U"] += 1
            c["grad"] += 1
            return float(-self._lp(x)), -np.asarray(self._glp(x), dtype=float)
        # central differences: charged as density evaluations, not gradients
        c["U"] += 1 + 2 * self.d
        u0 = float(-self._lp(x))
        g = np.empty(self.d)
        h = self.fd_eps * max(1.0, float(np.max(np.abs(x))))
        for i in range(self.d):
            e = np.zeros(self.d)
            e[i] = h
            g[i] = (-self._lp(x + e) + self._lp(x - e)) / (2 * h)
        return u0, g

    def hessian(self, x):
        x = np.asarray(x, dtype=float)
        h = 1e-4 * max(1.0, float(np.max(np.abs(x))))
        H = np.empty((self.d, self.d))
        for i in range(self.d):
            e = np.zeros(self.d)
            e[i] = h
            _, gp = self.value_and_grad(x + e)
            _, gm = self.value_and_grad(x - e)
            H[i] = (gp - gm) / (2 * h)
        return 0.5 * (H + H.T)
