"""Unified 10-method result tables for both final suites.

posteriordb-37: pmr (pdb2r_shard*, 8 seeds) + nuts/dense (pdb2_results.json,
8 seeds) + 7 panel methods (pdb_panel_shard*, 4 seeds), gold-judged.
mega-30: pmr/nuts (mega_results_shard*, 4 seeds, truth/ref-judged) + panel
(mega_panel_shard*).

Cell = median ess per 1000 oracle units; accuracy gate me<=0.25 & ve<=0.5
(dagger on violators); bold = gated-best per row. Emits PROPER.md +
paper/pdb_panel.tex + paper/mega_panel.tex (landscape-ready tabulars).
"""
import glob
import json

import numpy as np

METHODS = ["pmr", "nuts", "dense", "pf_dense", "chees", "mclmc",
           "tess", "flow_imh", "flowmc", "neutra"]
LABELS = {"pmr": "PMR", "nuts": "NUTS", "dense": "dNUTS", "pf_dense": "PF-dN",
          "chees": "ChEES", "mclmc": "MCLMC", "tess": "TESS",
          "flow_imh": "f-IMH", "flowmc": "flowMC", "neutra": "NeuTra"}


def rows_of(pat):
    out = []
    for f in glob.glob(pat):
        try:
            out += json.load(open(f))
        except Exception:
            pass
    return out


def med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def collect():
    """-> {suite: {target: {method: (rate, me, ve, nseeds)}}}"""
    P = {}

    def put(suite, tgt, meth, rs, rate_k, me_k, ve_k):
        ok = [r for r in rs if rate_k in r]
        if not ok:
            if any("error" in r for r in rs):
                P.setdefault(suite, {}).setdefault(tgt, {})[meth] = (float("nan"),) * 3 + (0,)
            return
        P.setdefault(suite, {}).setdefault(tgt, {})[meth] = (
            med([r[rate_k] for r in ok]), med([r.get(me_k) for r in ok]),
            med([r.get(ve_k) for r in ok]), len(ok))

    by = {}
    for r in rows_of("pdb2r_shard*.json"):
        if "rhat_max" not in r:
            by.setdefault((r["target"], "pmr"), []).append(r)
    for r in rows_of("pdb2_results.json"):
        if "rhat_max" not in r and r.get("sampler") in ("nuts", "dense"):
            by.setdefault((r["target"], r["sampler"]), []).append(r)
    for (t, m), rs in by.items():
        put("pdb", t, m, rs, "ess_ku", "gold_mean", "gold_var")
    by = {}
    for r in rows_of("pdb_panel_shard*.json"):
        by.setdefault((r["target"], r["method"]), []).append(r)
    for (t, m), rs in by.items():
        if m != "dense":  # 8-seed cached dense preferred over 4-seed panel dense
            put("pdb", t, m, rs, "ess_ku", "me", "ve")

    w30 = {ln.strip() for ln in open("mega30.txt")}
    by = {}
    for r in rows_of("mega_results_shard*.json"):
        if r.get("target") in w30 and r.get("sampler") in ("pmr", "nuts"):
            by.setdefault((r["target"], r["sampler"]), []).append(r)
    for (t, m), rs in by.items():
        put("mega", t, m, rs, "ess_ku", "me", "ve")
    by = {}
    for r in rows_of("mega_panel_shard*.json"):
        by.setdefault((r["target"], r["method"]), []).append(r)
    for (t, m), rs in by.items():
        put("mega", t, m, rs, "ess_ku", "me", "ve")
    return P


def gate_ok(me, ve):
    return np.isfinite(me) and np.isfinite(ve) and me <= 0.25 and ve <= 0.5


def render(P, suite, title):
    tgts = sorted(P.get(suite, {}))
    md = [f"# {title}", "",
          "Cell = median min-ESS per 1000 oracle units (density=1, gradient=2.5). "
          "\\* = accuracy-gate violation (mean err > 0.25 sd or |log var err| > 0.5 vs "
          "reference); **bold** = best among gate-passing methods.", "",
          "| target | " + " | ".join(LABELS[m] for m in METHODS) + " |",
          "|" + "---|" * (len(METHODS) + 1)]
    tex = ["\\begin{tabular}{l" + "r" * len(METHODS) + "}", "\\toprule",
           "target & " + " & ".join(LABELS[m] for m in METHODS) + " \\\\", "\\midrule"]
    wins = {m: 0 for m in METHODS}
    for t in tgts:
        cells = P[suite][t]
        best, br = None, -1.0
        for m in METHODS:
            r_ = cells.get(m)
            if r_ and np.isfinite(r_[0]) and gate_ok(r_[1], r_[2]) and r_[0] > br:
                best, br = m, r_[0]
        if best:
            wins[best] += 1
        mrow, xrow = [], []
        for m in METHODS:
            r_ = cells.get(m)
            if not r_ or not np.isfinite(r_[0]):
                mrow.append("--"); xrow.append("--")
                continue
            v = f"{r_[0]:.1f}"
            flag = "" if gate_ok(r_[1], r_[2]) else "\\*"
            mrow.append(f"**{v}**{flag}" if m == best else f"{v}{flag}")
            texflag = "" if gate_ok(r_[1], r_[2]) else "$^{*}$"
            xrow.append(f"\\textbf{{{v}}}{texflag}" if m == best else f"{v}{texflag}")
        name = t.replace("_", "\\_")
        md.append(f"| {t} | " + " | ".join(mrow) + " |")
        tex.append(f"{name} & " + " & ".join(xrow) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    md += ["", "Gated-best count: " + ", ".join(
        f"{LABELS[m]} {wins[m]}" for m in METHODS if wins[m]) + f" (of {len(tgts)})", ""]
    return "\n".join(md), "\n".join(tex), wins, len(tgts)


def write_pdb_top(path="paper/pdb_top.tex"):
    """Representative posteriordb rows for the main-text table (top ratios +
    the informative bottom rows), ending with \\bottomrule because \\input
    inside a tabular breaks the alignment scanner before a trailing rule."""
    new, old, dd = {}, {}, {}
    for r in rows_of("pdb2r_shard*.json"):
        if "ess_ku" in r:
            new.setdefault(r["target"], []).append(r)
    for r in rows_of("pdb2_results.json"):
        if "ess_ku" in r:
            old.setdefault((r["target"], r["sampler"]), []).append(r["ess_ku"])
    for r in rows_of("pdb_panel_shard*.json"):
        if "d" in r:
            dd[r["target"]] = r["d"]
    tab = []
    for t, rs in new.items():
        p = med([x["ess_ku"] for x in rs])
        n = med(old.get((t, "nuts"), []))
        de = med(old.get((t, "dense"), []))
        me = med([x["gold_mean"] for x in rs])
        ve = med([x["gold_var"] for x in rs])
        tab.append((t, dd.get(t, 0), p, n, de, p / max(n, 1e-9), p / max(de, 1e-9), me, ve))
    tab.sort(key=lambda r: -r[5])
    keep = ("pdb_gauss_mix", "pdb_gp_regr", "pdb_kid_hs", "pdb_es_nc",
            "pdb_earn_h", "pdb_kilpis")
    pick = tab[:8] + [r for r in tab if r[0] in keep and r not in tab[:8]]
    fr = lambda x: f"{x:.0f}" if x >= 10 else f"{x:.1f}"
    lines = []
    for t, d, p, n, de, rn, rd, me, ve in pick:
        fail = me > 0.25 or ve > 0.5
        rs_ = "(fail)$^\\ddagger$" if fail else f"${fr(rn)}\\times$/${fr(rd)}\\times$"
        lines.append(f"{t[4:].replace('_', chr(92) + '_')} & {d} & {p:.1f} & {n:.1f} "
                     f"& {de:.1f} & {rs_} & .{round(me * 100):02d}/.{round(ve * 100):02d}\\\\")
    lines.append("\\bottomrule")
    open(path, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    P = collect()
    write_pdb_top()
    md1, tex1, w1, n1 = render(P, "pdb", "posteriordb (37 posteriors, gold-judged, 10 methods)")
    md2, tex2, w2, n2 = render(P, "mega", "mega-30 (controlled families, truth/ref-judged, 10 methods)")
    open("PROPER.md", "w").write(md1 + "\n\n" + md2 + "\n")
    open("paper/pdb_panel.tex", "w").write(tex1 + "\n")
    open("paper/mega_panel.tex", "w").write(tex2 + "\n")
    print("pdb gated-best:", {LABELS[m]: c for m, c in w1.items() if c}, f"/{n1}")
    print("mega gated-best:", {LABELS[m]: c for m, c in w2.items() if c}, f"/{n2}")
    # PMR ratio summaries vs each baseline (gate-passing rows only)
    for suite in ("pdb", "mega"):
        rats = {}
        for t, cells in P.get(suite, {}).items():
            p = cells.get("pmr")
            if not p or not gate_ok(p[1], p[2]):
                continue
            for m in METHODS[1:]:
                r_ = cells.get(m)
                if r_ and np.isfinite(r_[0]) and r_[0] > 0:
                    rats.setdefault(m, []).append(p[0] / r_[0])
        print(suite, "median PMR ratio:",
              {LABELS[m]: round(float(np.median(v)), 1) for m, v in rats.items()})
