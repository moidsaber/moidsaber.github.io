"""Figure style and a few chart helpers shared by the notebooks.

Colors come from a validated categorical palette, used in slot order and never cycled.
The diverging map runs blue to red through a neutral gray, with the two arms
matched in lightness.
"""

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
NEUTRAL = "#f0efec"
SHADE = "#e9e8e3"  # background bands (e.g. stress regimes), deliberately not a series color

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", [
    "#0d366b", "#184f95", "#2a78d6", "#6da7ec", "#b7d3f6", NEUTRAL,
    "#fac0ba", "#ee7e77", "#d2383a", "#911e22", "#681014"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def use_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "text.color": INK, "xtick.color": BASELINE, "ytick.color": BASELINE,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 5,
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 9.5,
        "legend.frameon": False, "legend.fontsize": 9,
        "figure.dpi": 100, "savefig.dpi": 100, "figure.constrained_layout.use": True,
    })


def _ink_for(rgba):
    r, g, b = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgba[:3]]
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.33 else INK


def heatmap(ax, df, cmap, vmin, vmax, fmt="{:+.2f}", na_text="n/a", fontsize=8, clip_text=None):
    """Annotated heatmap of a DataFrame. NaN cells render as 'n/a' on the surface color.

    ``clip_text``: absolute values beyond it print as e.g. '>20' (the color already saturates).
    """
    data = df.to_numpy(dtype=float)
    cm = cmap.copy()
    cm.set_bad(SURFACE)
    im = ax.imshow(np.ma.masked_invalid(data), cmap=cm, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(df.shape[1]), [str(c) for c in df.columns], rotation=35, ha="right")
    ax.set_yticks(range(df.shape[0]), [str(i) for i in df.index])
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, df.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, df.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.0)
    ax.tick_params(which="both", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    norm = mpl.colors.Normalize(vmin, vmax)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, na_text, ha="center", va="center", fontsize=fontsize - 1, color=MUTED)
                continue
            txt = fmt.format(v)
            if clip_text is not None and abs(v) > clip_text:
                txt = (">" if v > 0 else "<-") + f"{clip_text:g}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=fontsize,
                    color=_ink_for(to_rgba(cm(norm(v)))))
    return im


def label_end(ax, x, y, text, dx=4, dy=0):
    """Direct label at the end of a line, in ink (the line beside it carries the color)."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                va="center", ha="left", fontsize=8.5, color=INK_2)
