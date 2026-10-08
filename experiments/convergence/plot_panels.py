#!/usr/bin/env python3
"""Convergence figure (Section 5.1 of the paper), one file per panel.

Top row: L² error of u; bottom row: L² error of v; columns: upwind, central,
alternating flux. Solid lines: q_v = q_u - 1; dashed lines: q_v = q_u.
The six panels share one data-area size and are assembled in LaTeX.

    python3 experiments/convergence/plot_panels.py [--csv PATH]

Input as in make_table.py; writes figures/convergence/conv2d_{erru,errv}_<flux>.pdf.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plotstyle import plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_csv():
    p = os.path.join(ROOT, "results", "convergence", "results.csv")
    return p if os.path.exists(p) else os.path.join(ROOT, "data", "convergence", "results.csv")


ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=default_csv())
ap.add_argument("--out", default=os.path.join(ROOT, "figures", "convergence"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

QS = [1, 2, 3, 4, 5, 6]
NS = range(5, 26)
FLUXES = ["upwind", "central", "alternating"]
FLUXNAME = {"upwind": "U.-flux", "central": "C.-flux", "alternating": "A.-flux"}
ROWS = [("erru", r"$\|e_u\|_{L^2(\Omega)}$"), ("errv", r"$\|e_v\|_{L^2(\Omega)}$")]
COL = {"erru": 1, "errv": 2}
COLORS = plt.cm.viridis(np.linspace(0.0, 0.88, len(QS)))

# geometry in inches: identical data area in every panel
W_DATA, H_DATA = 1.6267, 1.55
ML_Y, ML_N = 0.62, 0.10            # left margin with / without y labels
MR, MT_TITLE, MT_PLAIN = 0.10, 0.22, 0.06
MB_X, MB_PLAIN = 0.42, 0.06        # bottom margin with / without x labels

rows = {}
with open(args.csv) as f:
    rdr = csv.reader(f)
    next(rdr)
    for r in rdr:
        rows[(int(r[0]), r[1], r[2], int(r[3]))] = [float(x) for x in r[4:9]]

# common y-limits per row; one extra decade below keeps the legend clear
ylim = {}
for norm, _ in ROWS:
    vals = [v[COL[norm]] for v in rows.values() if np.isfinite(v[COL[norm]]) and v[COL[norm]] > 0]
    ylim[norm] = (10 ** (np.floor(np.log10(min(vals))) - 1), 10 ** np.ceil(np.log10(max(vals))))

for ri, (norm, ylab) in enumerate(ROWS):
    top = ri == 0
    for ci, flux in enumerate(FLUXES):
        left = ci == 0
        ml = ML_Y if left else ML_N
        mt = MT_TITLE if top else MT_PLAIN
        mb = MB_PLAIN if top else MB_X
        figw, figh = ml + W_DATA + MR, mt + H_DATA + mb

        fig = plt.figure(figsize=(figw, figh))
        ax = fig.add_axes([ml / figw, mb / figh, W_DATA / figw, H_DATA / figh])
        for qi, q in enumerate(QS):
            for pairing, ls in (("mixed", "-"), ("matched", "--")):
                pts = [(rows[k][0], rows[k][COL[norm]]) for N in NS
                       if (k := (q, pairing, flux, N)) in rows]
                pts = [(hh, e) for hh, e in pts if np.isfinite(e) and e > 0]
                if not pts:
                    continue
                hs, es = zip(*sorted(pts))
                ax.loglog(hs, es, ls, color=COLORS[qi], linewidth=0.9,
                          marker="o", markersize=1.7, markerfacecolor="none",
                          markeredgewidth=0.5,
                          label=(rf"$q_u={q}$" if pairing == "mixed" else None))

        ax.set_ylim(*ylim[norm])
        ax.set_xlim(0.037, 0.215)
        ax.grid(True, which="major", alpha=0.3, linewidth=0.5)
        if left:
            ax.set_ylabel(ylab)
        else:
            ax.tick_params(labelleft=False)
        if top:
            ax.set_title(FLUXNAME[flux])
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xticks([0.05, 0.1, 0.2])
            ax.set_xticklabels([r"$0.05$", r"$0.1$", r"$0.2$"])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(r"$h$")
        ax.legend(fontsize=6.0, loc="lower right", ncol=3,
                  handlelength=0.7, handletextpad=0.30, columnspacing=0.55,
                  borderpad=0.25, labelspacing=0.20, borderaxespad=0.3,
                  frameon=True, facecolor="white", edgecolor="0.6", framealpha=1.0)

        name = f"conv2d_{norm}_{flux}"
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(args.out, f"{name}.{ext}"), dpi=220)
        plt.close(fig)
        print(f"saved {name}")
