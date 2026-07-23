"""C production kernel: build-on-first-use loader, frozen-atlas packing, and
the three entry points (native target, Python-callback target, parallel
multi-chain). Portable: C11 + libm + pthreads, compiled for the host on first
import with whatever of clang/gcc/cc is available."""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sys
import sysconfig
import tempfile

import numpy as np

from . import warmup as P

_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_csrc", "pmr_kernel.c")


def _build_lib():
    src = open(_CSRC, "rb").read()
    tag = hashlib.sha256(src).hexdigest()[:16]
    ext = ".dylib" if sys.platform == "darwin" else (".dll" if os.name == "nt" else ".so")
    cache = os.path.join(tempfile.gettempdir(), f"pmrhmc-{tag}")
    os.makedirs(cache, exist_ok=True)
    out = os.path.join(cache, "pmr_kernel" + ext)
    if os.path.exists(out):
        return out
    cc = os.environ.get("CC") or sysconfig.get_config_var("CC") or "cc"
    cmd = cc.split() + ["-O3", "-std=gnu11", "-fPIC", "-shared", _CSRC, "-o", out, "-lm"]
    if os.name != "nt":
        cmd.append("-lpthread")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:  # surface the compiler message
        raise RuntimeError(f"pmr_kernel build failed:\n{e.stderr}") from e
    return out


_lib = ctypes.CDLL(_build_lib())
D = np.ctypeslib.ndpointer(dtype=np.float64, flags="C")
I32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C")
I64 = np.ctypeslib.ndpointer(dtype=np.int64, flags="C")
U64 = np.ctypeslib.ndpointer(dtype=np.uint64, flags="C")
UFN = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.POINTER(ctypes.c_double),
                       ctypes.c_int, ctypes.c_void_p)

_ATLAS = [I32, I64, D, I32, I64, D, D, D,
          ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
_lib.pmr_run.restype = ctypes.c_long
_lib.pmr_run.argtypes = [ctypes.c_int, D, ctypes.c_int, D, D, ctypes.c_int, ctypes.c_int,
                         ctypes.c_int, ctypes.c_int, *_ATLAS,
                         D, ctypes.c_uint64, ctypes.c_long, D, D]
_lib.pmr_run_cb.restype = ctypes.c_long
_lib.pmr_run_cb.argtypes = [UFN, ctypes.c_void_p,
                            ctypes.c_int, ctypes.c_int, *_ATLAS,
                            D, ctypes.c_uint64, ctypes.c_long, D, D]
_lib.pmr_run_multi.restype = ctypes.c_long
_lib.pmr_run_multi.argtypes = [ctypes.c_int, D, ctypes.c_int, D, D, ctypes.c_int, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, *_ATLAS,
                               D, U64, ctypes.c_long, ctypes.c_int, ctypes.c_int, D, D]


def _pack_gausslike(c, extra=()):
    return np.concatenate([c.mu, c.L.ravel(), c.Linv.ravel(), [c.logdet_half], list(extra)])


def pack_comp(c, d):
    k = c.kind
    if k == "gauss":
        return 0, _pack_gausslike(c)
    if k == "tcomp":
        return 1, _pack_gausslike(c, [c.nu])
    if k == "shear":
        base = c.base
        pairs = getattr(c, "pairs", None)
        if pairs is None:
            pairs = [(c.dr, c.t, c.gam, c.m2)]
        blob = np.concatenate([base.mu, base.L.ravel(), base.Linv.ravel(),
                               [base.logdet_half], [len(pairs)],
                               np.asarray(pairs, dtype=float).ravel()])
        return 2, blob
    if k == "scale":
        return 3, np.concatenate([[c.j, c.vbar], c.mu, c.sig, c.alpha, c.beta])
    if k == "polar":
        return 4, np.concatenate([[c.i, c.j, c.R, c.sr, c.c, c.Kw], c.mu, c.sig])
    if k == "hier":
        return 5, np.concatenate([[c.v, c.loc, c.gam, c.vbar, len(c.C)],
                                  np.asarray(c.C, dtype=float), c.mu, c.sig, c.cs])
    if k == "tri":
        return 6, np.concatenate([np.asarray(c.order, dtype=float), c.mu, c.sig,
                                  c.eps, c.delta, np.asarray(c.drv, dtype=float), c.gam])
    raise ValueError(k)


def pack_cache(q, d):
    if isinstance(q, P.ZeroCache):
        return 0, np.zeros(1)
    if isinstance(q, P.ScalarRBFCache):
        M = len(q.C)
        return 1, np.concatenate([[q.s, M], q.V.ravel(), q.C.ravel(), q.ell, q.a])
    if isinstance(q, P.ChartCache):
        n = q.n
        return 2, np.concatenate([[q.s, n, q.k, q.rho, q.f_max], q.V.ravel(),
                                  q.Y.ravel(), q.Fs.ravel(), q.B.ravel()])
    raise ValueError(type(q))


def pack_sampler(sampler):
    mix = sampler.mix
    d, K = mix.d, mix.K
    ctypes_, coffs, cblobs = [], [], []
    qtypes, qoffs, qblobs = [], [], []
    co = qo = 0
    for c, q in zip(mix.comps, sampler.caches):
        tcode, blob = pack_comp(c, d)
        ctypes_.append(tcode); coffs.append(co); cblobs.append(blob); co += len(blob)
        qcode, qblob = pack_cache(q, d)
        qtypes.append(qcode); qoffs.append(qo); qblobs.append(qblob); qo += len(qblob)
    td = sampler.t_def
    if td is not None:
        tdef = np.concatenate([[td.nu, sampler.t_eps], td.mu, np.diag(td.L)])
    else:
        tdef = np.concatenate([[-1.0, 0.0], np.zeros(d), np.ones(d)])
    return dict(
        d=d, K=K,
        ctype=np.asarray(ctypes_, dtype=np.int32),
        coff=np.asarray(coffs, dtype=np.int64),
        cblob=np.ascontiguousarray(np.concatenate(cblobs)),
        qtype=np.asarray(qtypes, dtype=np.int32),
        qoff=np.asarray(qoffs, dtype=np.int64),
        qblob=np.ascontiguousarray(np.concatenate(qblobs)),
        log_ws=np.ascontiguousarray(mix.log_ws),
        tdef=np.ascontiguousarray(tdef),
        h=sampler.h, p_global=sampler.p_global,
        T0=sampler.T_range[0], T1=sampler.T_range[1], L_cap=sampler.L_cap,
    )


def _atlas_args(pk):
    return (pk["ctype"], pk["coff"], pk["cblob"], pk["qtype"], pk["qoff"],
            pk["qblob"], pk["log_ws"], pk["tdef"],
            pk["h"], pk["p_global"], pk["T0"], pk["T1"], pk["L_cap"])


def _stats_dict(stats):
    return dict(n_local=int(stats[0]), acc_local=float(stats[1]),
                n_global=int(stats[2]), acc_global=float(stats[3]),
                n_U=int(stats[4]))


def run_cb(logpdf, sampler, n_samples, seed, x0=None, pk=None):
    """Sample with a Python log-density callback: ONE callback per transition."""
    pk = pk or pack_sampler(sampler)
    d = pk["d"]

    @UFN
    def ufn(xp, dd, _ctx):
        x = np.ctypeslib.as_array(xp, shape=(dd,))
        return -float(logpdf(x))

    if x0 is None:
        x0 = sampler._state[0]
    x0 = np.ascontiguousarray(np.asarray(x0, dtype=float))
    chain = np.zeros((n_samples, d))
    stats = np.zeros(5)
    rc = _lib.pmr_run_cb(ufn, None, d, pk["K"], *_atlas_args(pk),
                         x0, seed, n_samples, chain.reshape(-1), stats)
    if rc:
        raise RuntimeError(f"pmr_run_cb failed rc={rc} (d > 4096?)")
    return chain, _stats_dict(stats)


def run_multi_cb(logpdf, sampler, n_samples, n_chains, seeds=None, x0s=None,
                 cores=1, pk=None):
    """Parallel chains with a Python callback. ctypes releases the GIL during
    the C call and re-acquires per callback, so speedup depends on how much
    time the density itself releases the GIL; native targets scale linearly."""
    import threading
    pk = pk or pack_sampler(sampler)
    d = pk["d"]
    seeds = np.arange(1, n_chains + 1, dtype=np.uint64) if seeds is None else np.asarray(seeds, dtype=np.uint64)
    if x0s is None:
        x0s = np.tile(sampler._state[0], (n_chains, 1))
    x0s = np.ascontiguousarray(np.asarray(x0s, dtype=float))
    chains = np.zeros((n_chains, n_samples, d))
    stats = np.zeros((n_chains, 5))
    sem = threading.Semaphore(max(1, int(cores)))

    def one(c):
        with sem:
            ch, st = run_cb(logpdf, sampler, n_samples, int(seeds[c]), x0=x0s[c], pk=pk)
            chains[c] = ch
            stats[c] = [st["n_local"], st["acc_local"], st["n_global"], st["acc_global"], st["n_U"]]

    ts = [threading.Thread(target=one, args=(c,)) for c in range(n_chains)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return chains, [_stats_dict(s) for s in stats]


def run_native(native_spec, sampler, n_samples, seed, x0=None, pk=None):
    """Sample a C-native target family: native_spec = (tid, par, X, y)."""
    pk = pk or pack_sampler(sampler)
    tid, par, X, y = native_spec
    par = np.ascontiguousarray(np.asarray(par, dtype=float))
    if X is None:
        Xa, tn, tp = np.zeros(1), 0, 0
    else:
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        tn, tp = (Xa.shape if Xa.ndim == 2 else (Xa.shape[0], 0))
        Xa = Xa.ravel()
    ya = np.zeros(1) if y is None else np.ascontiguousarray(np.asarray(y, dtype=float))
    if x0 is None:
        x0 = sampler._state[0]
    x0 = np.ascontiguousarray(np.asarray(x0, dtype=float))
    chain = np.zeros((n_samples, pk["d"]))
    stats = np.zeros(5)
    rc = _lib.pmr_run(tid, par, len(par), Xa, ya, tn, tp, pk["d"], pk["K"],
                      *_atlas_args(pk), x0, seed, n_samples, chain.reshape(-1), stats)
    if rc:
        raise RuntimeError(f"pmr_run failed rc={rc} (d > 4096?)")
    return chain, _stats_dict(stats)


def run_native_multi(native_spec, sampler, n_samples, n_chains, seeds=None,
                     x0s=None, cores=1, pk=None):
    """Parallel multi-chain over a shared frozen atlas, threaded in C
    (pthreads); `cores` sets the worker count."""
    pk = pk or pack_sampler(sampler)
    d = pk["d"]
    tid, par, X, y = native_spec
    par = np.ascontiguousarray(np.asarray(par, dtype=float))
    if X is None:
        Xa, tn, tp = np.zeros(1), 0, 0
    else:
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        tn, tp = (Xa.shape if Xa.ndim == 2 else (Xa.shape[0], 0))
        Xa = Xa.ravel()
    ya = np.zeros(1) if y is None else np.ascontiguousarray(np.asarray(y, dtype=float))
    seeds = np.arange(1, n_chains + 1, dtype=np.uint64) if seeds is None else np.asarray(seeds, dtype=np.uint64)
    if x0s is None:
        x0s = np.tile(sampler._state[0], (n_chains, 1))
    x0s = np.ascontiguousarray(np.asarray(x0s, dtype=float))
    chains = np.zeros((n_chains, n_samples, d))
    stats = np.zeros((n_chains, 5))
    rc = _lib.pmr_run_multi(tid, par, len(par), Xa, ya, tn, tp, d, pk["K"],
                            *_atlas_args(pk), x0s.reshape(-1), seeds, n_samples,
                            n_chains, max(1, int(cores)),
                            chains.reshape(-1), stats.reshape(-1))
    if rc:
        raise RuntimeError(f"pmr_run_multi failed rc={rc}")
    return chains, [_stats_dict(s) for s in stats]
