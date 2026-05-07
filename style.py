"""J. Fuentes Aguilar, 2025-2026."""

import matplotlib as _mpl

PURPLE   = "#7951D7"
ORANGE   = "#E8883B"
GREEN    = "#4CAE74"
RED      = "#C0392B"
BLUE     = "#4C72B0"

UI_COLOR = "#555555"
FC       = "#777777"

FS       = 18
FS_TICK  = FS - 2
FS_PANEL = 20

FIG_WIDTH = 14.0

H_1x3 = 5.2
H_2x2 = 12.0
H_2x3 = 9.0
H_1x4 = 4.5

WSPACE_TOP    = 0.10
WSPACE_BOTTOM = 0.12
CB_FRACTION   = 0.045
CB_PAD        = 0.015

def tint(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor(UI_COLOR)
        spine.set_linewidth(0.9)
    ax.tick_params(colors=UI_COLOR, which="both", labelsize=FS_TICK)
    ax.xaxis.label.set_color(FC)
    ax.yaxis.label.set_color(FC)
    ax.title.set_color(FC)

def panel_label(ax, letter, *, dx=-4, dy=28):
    ax.annotate(
        letter,
        xy=(0, 1), xycoords="axes fraction",
        xytext=(dx, dy), textcoords="offset points",
        fontsize=FS_PANEL, fontweight="bold",
        va="bottom", ha="right",
        color=UI_COLOR, annotation_clip=False,
    )

def tint_colorbar(cb):
    cb.ax.tick_params(colors=UI_COLOR, labelsize=FS_TICK)
    cb.outline.set_edgecolor(UI_COLOR)
    cb.outline.set_linewidth(0.9)

def thin_ticks(ax, n=5, which="both"):
    from matplotlib.ticker import MaxNLocator
    if which in ("both", "x") and ax.get_xscale() == "linear":
        ax.xaxis.set_major_locator(MaxNLocator(nbins=n, min_n_ticks=3))
    if which in ("both", "y") and ax.get_yscale() == "linear":
        ax.yaxis.set_major_locator(MaxNLocator(nbins=n, min_n_ticks=3))

def raincloud(ax, data_groups, labels, colors, *, orient="v",
              violin_width=0.75, jitter=0.10, point_alpha=0.35,
              point_size=12, box_width=0.12, rng=None):
    import numpy as np
    if rng is None:
        rng = np.random.default_rng(0)
    n_groups = len(data_groups)
    positions = np.arange(n_groups, dtype=float)
    for pos, values, colour in zip(positions, data_groups, colors):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        parts = ax.violinplot(
            [values], positions=[pos], widths=violin_width,
            showmeans=False, showmedians=False, showextrema=False,
            vert=(orient == "v"),
        )
        for body in parts["bodies"]:
            verts = body.get_paths()[0].vertices
            if orient == "v":
                verts[:, 0] = np.clip(verts[:, 0], pos, np.inf)
            else:
                verts[:, 1] = np.clip(verts[:, 1], pos, np.inf)
            body.set_facecolor(colour)
            body.set_edgecolor(colour)
            body.set_alpha(0.35)
            body.set_linewidth(0.8)
        box_offset = 0.12
        ax.boxplot(
            [values], positions=[pos + box_offset], widths=box_width,
            vert=(orient == "v"), showfliers=False, patch_artist=True,
            medianprops=dict(color=UI_COLOR, linewidth=1.4),
            boxprops=dict(facecolor="white", edgecolor=UI_COLOR,
                          linewidth=1.0, alpha=0.9),
            whiskerprops=dict(color=UI_COLOR, linewidth=1.0),
            capprops=dict(color=UI_COLOR, linewidth=1.0),
        )
        jitter_offset = rng.uniform(-jitter, 0.0, size=values.size) - 0.12
        if orient == "v":
            ax.scatter(pos + jitter_offset, values,
                       s=point_size, color=colour, alpha=point_alpha,
                       edgecolors="none", zorder=3)
        else:
            ax.scatter(values, pos + jitter_offset,
                       s=point_size, color=colour, alpha=point_alpha,
                       edgecolors="none", zorder=3)
    if orient == "v":
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlim(positions.min() - 0.6, positions.max() + 0.7)
    else:
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.set_ylim(positions.min() - 0.6, positions.max() + 0.7)
