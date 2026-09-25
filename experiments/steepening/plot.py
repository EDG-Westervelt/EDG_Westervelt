#!/usr/bin/env python3
"""Wave-steepening figure (Section 5.3 of the paper), three files:

  steep_surface.pdf    v_h over the domain at t = 0.8 (beta = 8, absorbing BC)
  steep_recv_beta.pdf  receiver signal at (0.7, 0.5): beta = 8 vs beta = 0
  steep_recv_bc.pdf    receiver signal at (0.7, 0.5): absorbing vs Dirichlet BC

    python3 experiments/steepening/plot.py [--data DIR]

By default reads results/steepening if it exists, otherwise the reference data
in data/steepening. Writes figures/steepening/.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plotstyle import plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_data():
    p = os.path.join(ROOT, "results", "steepening")
    return p if os.path.isdir(p) else os.path.join(ROOT, "data", "steepening")


ap = argparse.ArgumentParser()
ap.add_argument("--data", default=default_data())
ap.add_argument("--out", default=os.path.join(ROOT, "figures", "steepening"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

W_LEFT, W_RIGHT, H_TOT = 2.85, 3.15, 2.90
H_HALF = H_TOT / 2
TSNAP = "0.80"


def recv(case, irx=2):
    """Time and v_h at receiver irx (1: (0.45,0.5), 2: (0.7,0.5), 3: (0.85,0.5))."""
    r = np.loadtxt(os.path.join(args.data, case, "recv.csv"), delimiter=",")
    return r[:, 0], r[:, irx]


# ---- 1. v_h over the domain as a surface ----
g = np.loadtxt(os.path.join(args.data, "nl", "grid.csv"), delimiter=",")
G = np.loadtxt(os.path.join(args.data, "nl", f"snap_{TSNAP}.csv"), delimiter=",")
s = 2
X, Y = np.meshgrid(g[::s, 0], g[::s, 1])
Z = G[::s, ::s]
vmax = np.abs(Z).max()

fig = plt.figure(figsize=(W_LEFT, H_TOT))
ax = fig.add_subplot(projection="3d")
ax.set_position([0.0, 0.02, 0.90, 0.96])
ax.plot_surface(X, Y, Z, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                rcount=160, ccount=160, linewidth=0, antialiased=False, rasterized=True)
ax.set_xlabel(r"$x$", labelpad=-5)
ax.set_ylabel(r"$y$", labelpad=-5)
ax.set_zlabel(r"$v_h$", labelpad=-6)
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])
ax.set_zticks([-0.01, 0, 0.01])
ax.tick_params(axis="both", pad=-2, labelsize=6)
ax.tick_params(axis="z", pad=0, labelsize=6)
ax.view_init(elev=34, azim=-56)
ax.set_box_aspect((1, 1, 0.5), zoom=0.92)
for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
    pane.pane.set_alpha(0.0)
ax.grid(False)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(args.out, f"steep_surface.{ext}"), dpi=220)
plt.close(fig)
print("saved steep_surface")

# ---- 2 & 3. receiver signals ----
ML, MR, MT, MB_X, MB_PLAIN = 0.60, 0.06, 0.06, 0.40, 0.04
W_DATA = W_RIGHT - ML - MR
PANELS = [
    ("steep_recv_beta", False, (("nl", r"$\beta = 8$", "#1d9e75", "-"),
                                ("lin", r"$\beta = 0$", "0.35", "--"))),
    ("steep_recv_bc", True, (("nl", "absorbing", "#1d9e75", "-"),
                             ("nl_dbc", "Dirichlet", "#d85a30", "--"))),
]
for name, with_x, series in PANELS:
    mb = MB_X if with_x else MB_PLAIN
    fig = plt.figure(figsize=(W_RIGHT, H_HALF))
    ax = fig.add_axes([ML / W_RIGHT, mb / H_HALF, W_DATA / W_RIGHT, (H_HALF - MT - mb) / H_HALF])
    for case, lab, colr, ls in series:
        t, v = recv(case)
        ax.plot(t, v, ls, color=colr, lw=0.9, label=lab)
    ax.set_xlim(0.4, 1.8)
    ax.set_ylabel(r"$v_h(0.7,\,0.5)$")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    if with_x:
        ax.set_xlabel(r"$t$")
    else:
        ax.tick_params(labelbottom=False)
    ax.legend(fontsize=6.5, loc="upper left", ncol=2, handlelength=1.3,
              columnspacing=0.9, borderpad=0.28, handletextpad=0.4,
              frameon=True, facecolor="white", edgecolor="0.6", framealpha=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"{name}.{ext}"), dpi=220)
    plt.close(fig)
    print(f"saved {name}")
