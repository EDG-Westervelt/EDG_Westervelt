"""Shared matplotlib settings for the figure scripts."""
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# LaTeX text rendering is used when a LaTeX installation is available
# (as for the figures in the paper); otherwise matplotlib's mathtext.
plt.rcParams.update({
    "text.usetex": shutil.which("latex") is not None,
    "font.family": "serif",
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
})
if plt.rcParams["text.usetex"]:
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
