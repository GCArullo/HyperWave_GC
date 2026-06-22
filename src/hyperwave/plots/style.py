from __future__ import annotations

import math

import matplotlib
import matplotlib.pyplot as plt

# -----------------------------
# Shared colour palette
# -----------------------------
#: injected signal (amber) and raw data (grey) — shared by all reconstruction plots
SIGNAL_COLOR = "#f2ab15"
DATA_COLOR = "#8c8c8c"

#: per-detector reconstruction palette (injected slate / per-IFO accent / noise grey)
IFO_COLORS = {
    "L1": {"injected": "#455A64", "reconstructed": "#ca0147", "noise": "lightgray"},
    "H1": {"injected": "#455A64", "reconstructed": "#0f9b8e", "noise": "lightgray"},
    "V1": {"injected": "#455A64", "reconstructed": "#f2ab15", "noise": "lightgray"},
}
_FALLBACK_IFO = {"injected": "#455A64", "reconstructed": "#ca0147", "noise": "lightgray"}


def ifo_palette(ifo):
    """Colour dict (``injected``/``reconstructed``/``noise``) for a detector."""
    return IFO_COLORS.get(str(ifo), _FALLBACK_IFO)


# -----------------------------
# Presets
# -----------------------------
_PRESETS = {
    # Good for single-column figure in paper
    "prd": dict(label=22, tick=18, legend=16, title=22, lw=1.2),
    # Slightly larger typography
    "nature": dict(label=26, tick=22, legend=18, title=26, lw=1.3),
    # For drafts / talks
    "draft": dict(label=30, tick=26, legend=22, title=30, lw=1.4),
}

def _scale_fonts(base: dict, panel_scale: float) -> dict:
    """
    panel_scale: 1.0 means full-size figure.
                 0.33 means one-third width (typical 3-column grid).
    We increase font sizes as plots get smaller in LaTeX.
    """
    # Empirical: if you shrink to 1/3 width, need ~1.35–1.6× larger fonts.
    # We use a smooth scaling that doesn't explode.
    s = 1.0 / max(panel_scale, 1e-6)
    factor = 1.0 + 0.30 * math.log(s)  # log scaling
    factor = max(1.0, min(factor, 1.8))  # clamp

    return dict(
        label=int(round(base["label"] * factor)),
        tick=int(round(base["tick"] * factor)),
        legend=int(round(base["legend"] * factor)),
        title=int(round(base["title"] * factor)),
        lw=base["lw"],
    )

def apply_style(
    preset: str = "nature",
    black_background: bool = False,
    transparent: bool = False,
    panel_scale: float = 1.0,
    font_family: str = "STIXGeneral",
):
    """
    Apply global matplotlib rcParams for HyperWave figures.

    preset: 'nature' | 'prd' | 'draft'
    black_background: True for dark-theme plots
    transparent: True to keep axes/figure facecolors transparent (nice for slides)
    panel_scale: approximate scaling relative to full-size (e.g. 0.31 for subfigure 0.31\\textwidth)
    """
    if preset not in _PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Choose from {list(_PRESETS)}")

    sizes = _scale_fonts(_PRESETS[preset], panel_scale)

    # PDF/PS font embedding: avoids weird tiny/Type3 fonts in LaTeX
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    # Typography
    plt.rcParams.update({
        "font.family": font_family,
        "mathtext.fontset": "stix",
        "text.usetex": False,

        "font.size": sizes["label"],          # baseline
        "axes.labelsize": sizes["label"],
        "axes.titlesize": sizes["title"],
        "xtick.labelsize": sizes["tick"],
        "ytick.labelsize": sizes["tick"],
        "legend.fontsize": sizes["legend"],

        "axes.linewidth": sizes["lw"],
        "lines.linewidth": sizes["lw"],
        "xtick.major.width": sizes["lw"],
        "ytick.major.width": sizes["lw"],
        "xtick.minor.width": 0.8 * sizes["lw"],
        "ytick.minor.width": 0.8 * sizes["lw"],
    })

    if black_background:
        plt.rcParams.update({
            "axes.edgecolor": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "text.color": "white",
        })
    else:
        plt.rcParams.update({
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "text.color": "#222222",
        })


    if transparent:
        plt.rcParams.update({
            "axes.facecolor": "none",
            "figure.facecolor": "none",
        })


def style_axes(
    ax: plt.Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    labelsize: int | None = None,
    ticksize: int | None = None,
    titlesize: int | None = None,
    grid: bool = False,
):
    """
    Strong overrides at the Axes level (wins vs any rcparams).
    Use this in every plot just before save/show.
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # If user didn't pass explicit sizes, use current rcParams.
    labelsize = labelsize or plt.rcParams["axes.labelsize"]
    ticksize = ticksize or plt.rcParams["xtick.labelsize"]
    titlesize = titlesize or plt.rcParams["axes.titlesize"]

    ax.xaxis.label.set_size(labelsize)
    ax.yaxis.label.set_size(labelsize)

    ax.tick_params(axis="both", which="major", labelsize=ticksize)
    ax.tick_params(axis="both", which="minor", labelsize=max(6, ticksize - 4))

    # Title size
    ax.title.set_size(titlesize)

    ax.grid(grid)