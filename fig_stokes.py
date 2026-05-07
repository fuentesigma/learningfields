"""J. Fuentes Aguilar, 2025-2026."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from style import (
    PURPLE, ORANGE, GREEN, RED, UI_COLOR, FC, FS, FS_TICK, FIG_WIDTH,
    tint, panel_label, tint_colorbar,
)

from topological_signature import (
    prepare_ensemble,
    build_edge_transport_cache,
    random_rectangles,
    check_abelian_stokes,
)

OUT_PDF = HERE / "figs" / "fig_char_stokes.pdf"
CS_NPZ = HERE / "figs" / "character_spectrum.npz"
N_EVAL = 30
N_MODELS = 10
SEED0 = 0

def load_char_data():
    if not CS_NPZ.exists():
        raise FileNotFoundError(
            f"character_spectrum.npz not found at {CS_NPZ}. "
            "Run character_spectrum.py first."
        )
    d = np.load(CS_NPZ, allow_pickle=True)
    return {
        "p_plaq": d["p_plaq"],
        "hop_plaq": d["hop_plaq"],
        "eta_plaq": d["eta_plaq"],
        "det_plaq": d["det_plaq"],
        "u_eval": d["u_eval"],
        "v_eval": d["v_eval"],
    }

def compute_stokes_on_rectangles(n_rect=500):
    print("[stokes] loading ensemble and building open-grid Hungarian cache ...")
    preds, jacs, E, s_y, s_g, u_eval, v_eval = prepare_ensemble(
        n_eval=N_EVAL, n_models=N_MODELS, seed0=SEED0,
    )
    cache_open = build_edge_transport_cache(
        preds, jacs, E, N_EVAL, s_y, s_g,
        periodic=False, mode="hungarian",
    )
    rng = np.random.default_rng(42)
    rects = random_rectangles(N_EVAL, n_rect, rng)
    print(f"[stokes] checking abelian Stokes on {n_rect} random rectangles ...")
    etaB, etaF, detB, detFprod, sizes = check_abelian_stokes(
        cache_open, N_EVAL, rects,
    )
    return {
        "eta_boundary": etaB,
        "eta_face_product": etaF,
        "det_boundary": detB,
        "det_face_product": detFprod,
        "sizes": sizes,
    }

def raincloud(ax, data_groups, labels, colors, *, orient="v",
              violin_width=0.7, jitter=0.08, point_alpha=0.35,
              point_size=14, box_width=0.12, rng=None):
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
        if orient == "v":
            box_position = [pos + box_offset]
            bp = ax.boxplot(
                [values], positions=box_position, widths=box_width,
                vert=True, showfliers=False, patch_artist=True,
                medianprops=dict(color=UI_COLOR, linewidth=1.4),
                boxprops=dict(facecolor="white", edgecolor=UI_COLOR,
                              linewidth=1.0, alpha=0.9),
                whiskerprops=dict(color=UI_COLOR, linewidth=1.0),
                capprops=dict(color=UI_COLOR, linewidth=1.0),
            )
        else:
            box_position = [pos + box_offset]
            bp = ax.boxplot(
                [values], positions=box_position, widths=box_width,
                vert=False, showfliers=False, patch_artist=True,
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

def render(cs, stokes):

    fig, axes = plt.subplots(
        2, 2, figsize=(FIG_WIDTH, FIG_WIDTH * 1),
        constrained_layout=True,
    )

    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]
    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_box_aspect(1.0)

    p_plaq = cs["p_plaq"]
    re_p2 = np.real(p_plaq[:, :, 1])
    u_plaq = cs["u_eval"][:-1]
    v_plaq = cs["v_eval"][:-1]
    vabs = float(np.max(np.abs(re_p2)))
    im_a = ax_a.contourf(
        u_plaq, v_plaq, re_p2.T, levels=20, cmap="Pastel2",
        vmin=-vabs, vmax=vabs,
    )
    ax_a.set_xlabel(r"$u$", fontsize=FS)
    ax_a.set_ylabel(r"$v$", fontsize=FS)
    ax_a.set_title(r"$\mathrm{Re}\,p_2(c)$ field", fontsize=FS, color=FC)
    cb_a = fig.colorbar(im_a, ax=ax_a, fraction=0.045, pad=0.02)
    cb_a.set_label(r"$\mathrm{Re}\,p_2$", fontsize=FS_TICK, color=FC)
    cb_a.locator = MaxNLocator(nbins=4)
    cb_a.update_ticks()
    tint_colorbar(cb_a)
    panel_label(ax_a, "a")
    tint(ax_a)
    ax_a.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_a.yaxis.set_major_locator(MaxNLocator(nbins=4))

    re_p1 = np.real(p_plaq[:, :, 0]).ravel()
    hop = cs["hop_plaq"].ravel()
    pred = np.sqrt(np.clip(2.0 * (1.0 - re_p1) / N_MODELS, 0.0, None))
    ax_b.scatter(pred, hop, s=14, color=PURPLE, alpha=0.55, edgecolors="none")
    lim = float(max(pred.max(), hop.max())) * 1.05 + 1e-6
    ax_b.plot([0, lim], [0, lim], "--", color=UI_COLOR, linewidth=1.2,
              alpha=0.6,
              label=r"$h_{\mathrm{op}}=\sqrt{2(1-\mathrm{Re}\,p_1)/N}$")
    ax_b.set_xlim(0, lim)
    ax_b.set_ylim(0, lim)
    ax_b.set_xlabel(r"$\sqrt{2(1-\mathrm{Re}\,p_1)/N}$", fontsize=FS)
    ax_b.set_ylabel(r"$h_{\mathrm{op}}$", fontsize=FS)
    ax_b.set_title(r"regime-(C) identity test", fontsize=FS, color=FC)
    ax_b.legend(fontsize=FS_TICK - 2, loc="lower right", frameon=False)
    ax_b.grid(True, alpha=0.25)
    panel_label(ax_b, "b")
    tint(ax_b)
    ax_b.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_b.yaxis.set_major_locator(MaxNLocator(nbins=4))

    det_all = np.abs(cs["det_plaq"]).ravel()
    det_all = det_all[det_all > 0]
    logdet = np.log10(det_all)
    raincloud(
        ax_c, [logdet], labels=[""], colors=[PURPLE],
        orient="h", violin_width=0.85, point_alpha=0.4, point_size=10,
        rng=np.random.default_rng(0),
    )
    ax_c.axvline(-12, color=RED, linestyle="--", linewidth=1.2, alpha=0.8,
                 label=r"tolerance $10^{-12}$")
    n_total = int(np.abs(cs["det_plaq"]).size)
    n_res = int(np.sum(np.abs(cs["det_plaq"]).ravel() >= 1e-12))
    ax_c.text(0.02, 0.96,
              rf"resolved: ${n_res}/{n_total}$",
              transform=ax_c.transAxes, ha="left", va="top",
              fontsize=FS_TICK, color=FC,
              bbox=dict(facecolor="white", edgecolor=UI_COLOR, alpha=0.85,
                        boxstyle="round,pad=0.3"))
    ax_c.set_xlabel(r"$\log_{10}|\det H|$", fontsize=FS)
    ax_c.set_ylabel("")
    ax_c.set_yticks([])
    ax_c.set_title(r"regime-(A) $|\det H|$ distribution",
                   fontsize=FS, color=FC)
    ax_c.legend(fontsize=FS_TICK - 2, loc="lower right", frameon=False)
    ax_c.grid(True, axis="x", alpha=0.25)
    panel_label(ax_c, "c")
    tint(ax_c)
    ax_c.xaxis.set_major_locator(MaxNLocator(nbins=5))

    etaB = np.asarray(stokes["eta_boundary"], dtype=int)
    etaF = np.asarray(stokes["eta_face_product"], dtype=int)
    sizes = np.asarray(stokes["sizes"], dtype=float)
    agree = int(np.sum(etaB == etaF))
    total = etaB.size

    log_area = np.log10(np.maximum(sizes, 1.0))
    group_neg = log_area[etaB == -1]
    group_pos = log_area[etaB == +1]
    raincloud(
        ax_d, [group_neg, group_pos],
        labels=[r"$\eta(\partial D)=-1$", r"$\eta(\partial D)=+1$"],
        colors=[PURPLE, ORANGE],
        orient="v", violin_width=0.85, point_alpha=0.4, point_size=10,
        rng=np.random.default_rng(1),
    )
    ax_d.set_ylabel(r"$\log_{10}\,\mathrm{area}(D)$", fontsize=FS)
    ax_d.set_title("abelian Stokes closure (regime C)",
                   fontsize=FS, color=FC)
    ax_d.text(0.02, 0.98,
              f"sign agreement: {agree}/{total}\nresidue $= 0$ (exact)",
              transform=ax_d.transAxes, ha="left", va="top",
              fontsize=FS_TICK - 2, color=FC,
              bbox=dict(facecolor="white", edgecolor=UI_COLOR, alpha=0.85,
                        boxstyle="round,pad=0.3"))
    ax_d.grid(True, axis="y", alpha=0.25)
    panel_label(ax_d, "d")
    tint(ax_d)
    ax_d.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_d.tick_params(axis="x", labelsize=FS_TICK)

    fig.savefig(OUT_PDF, format="pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] saved -> {OUT_PDF}")

def main():
    cs = load_char_data()
    stokes = compute_stokes_on_rectangles(n_rect=500)
    render(cs, stokes)

if __name__ == "__main__":
    main()
