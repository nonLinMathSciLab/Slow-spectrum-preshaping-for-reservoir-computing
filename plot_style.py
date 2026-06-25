from __future__ import annotations

import matplotlib.pyplot as plt


def apply_publication_style() -> None:
    """Apply the paper-style Matplotlib settings used by the public figures."""

    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["font.size"] = 20
    plt.rcParams["legend.fontsize"] = 18
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.framealpha"] = 0.5
    plt.rcParams["legend.edgecolor"] = "black"
    plt.rcParams["legend.markerscale"] = 1.0
