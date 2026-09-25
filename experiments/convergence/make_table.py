#!/usr/bin/env python3
"""Convergence-rate table (Section 5.1 of the paper).

Rates are least-squares slopes of log(error) against log(h) over N = 15..25.

    python3 experiments/convergence/make_table.py [--csv PATH]

By default the table is computed from results/convergence/results.csv if it
exists, otherwise from the reference data in data/convergence/results.csv.
Writes figures/convergence/conv_table.tex and prints the rates.
"""
import argparse
import csv
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_csv():
    p = os.path.join(ROOT, "results", "convergence", "results.csv")
    return p if os.path.exists(p) else os.path.join(ROOT, "data", "convergence", "results.csv")


ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=default_csv())
ap.add_argument("--out", default=os.path.join(ROOT, "figures", "convergence"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

d = {(int(r["q"]), r["pairing"], r["flux"], int(r["N"])):
     [float(r[k]) for k in ("h", "erru", "errv", "errux", "errE")]
     for r in csv.DictReader(open(args.csv))}


def rate(q, p, f, c):
    h, e = zip(*[(d[(q, p, f, N)][0], d[(q, p, f, N)][c]) for N in range(15, 26)])
    return np.polyfit(np.log(h), np.log(e), 1)[0]


NORMS = [r"$\|e_u\|_{L^2}$", r"$\|e_v\|_{L^2}$", r"$\|\nabla_h e_u\|_{L^2}$", r"$\|e\|_{E}$"]
FL = [("U", "upwind"), ("C", "central"), ("A", "alternating")]
QS = sorted({k[0] for k in d})

a = [r"\begin{tabular}{c cccc|cccc|cccc}", r"\toprule",
     " & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{ab}.-flux}}" for ab, _ in FL) + r" \\",
     r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
     "$q_u$ & " + " & ".join(NORMS * 3) + r" \\"]
print(f"source: {args.csv}")
for lab, p in ((r"$q_v = q_u - 1$", "mixed"), (r"$q_v = q_u$", "matched")):
    a += [r"\midrule", rf"\multicolumn{{13}}{{c}}{{{lab}}} \\", r"\midrule"]
    print(f"\n{p}:  columns = (erru, errv, errgradu, errE) x (upwind, central, alternating)")
    for q in QS:
        rates = [rate(q, p, f, c) for _, f in FL for c in (1, 2, 3, 4)]
        a.append(f"{q} & " + " & ".join(f"{x:.2f}" for x in rates) + r" \\")
        print(f"  q_u={q}: " + "  ".join(f"{x:5.2f}" for x in rates))
a += [r"\bottomrule", r"\end{tabular}"]
out = os.path.join(args.out, "conv_table.tex")
open(out, "w").write("\n".join(a) + "\n")
print(f"\nsaved {out}")
