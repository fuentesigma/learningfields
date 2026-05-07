"""J. Fuentes Aguilar, 2025-2026."""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from style import (
    UI_COLOR, FC, FS, FS_TICK, FS_PANEL, FIG_WIDTH,
    tint, panel_label, tint_colorbar, thin_ticks,
)

Color_A = "#8C838E"
Color_B = "#9AD6DF"

from topological_signature import (
    prepare_ensemble,
    build_edge_transport_cache,
    closed_torus_sum_rule,
    measure_signature,
    permutation_matrix,
)

FS = FS_TICK = FS_PANEL = 18

CS_NPZ = HERE / "figs" / "character_spectrum.npz"
OUT_PDF = HERE / "figs" / "body_figure.pdf"
N_EVAL = 30
N_MODELS = 10
SEED0 = 0

def load_character_spectrum():
    if not CS_NPZ.exists():
        raise FileNotFoundError(
            f"character_spectrum.npz not found at {CS_NPZ}. "
            "Run character_spectrum.py first."
        )
    return np.load(CS_NPZ, allow_pickle=True)

def compute_panels_c_and_d():
    print("[data] loading / training ensemble ...")
    preds, jacs, E, s_y, s_g, u_eval, v_eval = prepare_ensemble(
        n_eval=N_EVAL, n_models=N_MODELS, seed0=SEED0,
    )
    print("[data] building periodic Hungarian transport cache on T^2 ...")
    cache = build_edge_transport_cache(
        preds, jacs, E, N_EVAL, s_y, s_g,
        periodic=True, mode="hungarian",
    )
    print("[c] closed-surface sum rule ...")
    etas, dets, prod_eta, n_zero = closed_torus_sum_rule(cache, N_EVAL)
    n_pos = int(np.sum(etas == +1))
    n_neg = int(np.sum(etas == -1))
    print(
        f"    prod_eta = {prod_eta}, split = {n_pos}+/{n_neg}- "
        f"(zero underflow cells = {n_zero})"
    )

    print("[d] topological signatures across four ensembles ...")
    sigma = permutation_matrix([1, 0] + list(range(2, N_MODELS)), N_MODELS)
    sig_canonical, _ = measure_signature(cache, N_EVAL)
    sig_twisted_a, _ = measure_signature(cache, N_EVAL, twist="alpha", sigma=sigma)
    sig_twisted_b, _ = measure_signature(cache, N_EVAL, twist="beta", sigma=sigma)
    sig_zero = (+1, +1)
    sectors = [
        ("canonical MLP",   sig_canonical),
        ("bias-tag zero",   sig_zero),
        (r"twisted $\alpha$", sig_twisted_a),
        (r"twisted $\beta$",  sig_twisted_b),
    ]
    for name, s in sectors:
        print(f"    {name:20s}  (eta_alpha, eta_beta) = {s}")
    return etas, prod_eta, n_pos, n_neg, sectors, u_eval, v_eval

def render(etas, prod_eta, n_pos, n_neg, sectors, u_eval, v_eval, cs):
    fig, axes = plt.subplots(
        2, 2, figsize=(FIG_WIDTH, FIG_WIDTH * 1),
        constrained_layout=True,
    )
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]
    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_box_aspect(1.0)

    p_plaq = cs["p_plaq"]
    re_p = np.real(p_plaq).reshape(-1, p_plaq.shape[-1])
    re_p1 = re_p[:, 0]
    ks = np.arange(1, re_p.shape[1] + 1)
    corrs = np.array([
        np.corrcoef(re_p1, re_p[:, k - 1])[0, 1] for k in ks
    ])
    ax_a.plot(ks, corrs, "o-", color=Color_A, linewidth=3.0, markersize=8)
    ax_a.axhline(0.0, linestyle=":", color=UI_COLOR, alpha=0.5)
    ax_a.axhline(1.0, linestyle=":", color=UI_COLOR, alpha=0.5)
    ax_a.set_xticks(ks)
    ax_a.set_ylim(-0.1, 1.05)
    ax_a.set_xlabel(r"power $k$", fontsize=FS)
    ax_a.set_ylabel(r"$\mathrm{corr}(\mathrm{Re}\,p_k,\ \mathrm{Re}\,p_1)$",
                    fontsize=FS)
    ax_a.set_title(r"higher characters beyond $h_{\mathrm{op}}$",
                   fontsize=FS, color=FC)
    ax_a.grid(True, alpha=0.25)
    panel_label(ax_a, "a")
    tint(ax_a)
    thin_ticks(ax_a, n=5, which="y")

    p_alpha = np.asarray(cs["p_alpha"]).reshape(-1)
    p_beta = np.asarray(cs["p_beta"]).reshape(-1)
    ks = np.arange(1, p_alpha.size + 1)
    width = 0.4
    ax_b.bar(ks - width / 2, np.real(p_alpha), width=width,
             color=Color_A, edgecolor=UI_COLOR, linewidth=0.6,
             label=r"$\gamma_\alpha$")
    ax_b.bar(ks + width / 2, np.real(p_beta), width=width,
             color=Color_B, edgecolor=UI_COLOR, linewidth=0.6,
             label=r"$\gamma_\beta$")
    ax_b.axhline(1.0 / N_MODELS, linestyle="--", linewidth=1.1,
                 color=UI_COLOR, alpha=0.65,
                 label=r"$1/N$ (stationary $J$)")
    ax_b.set_xlabel(r"power $k$", fontsize=FS)
    ax_b.set_ylabel(r"$\mathrm{Re}\,p_k(\gamma)$", fontsize=FS)
    ax_b.set_title(r"non-contractible loops, regime (A)", fontsize=FS, color=FC)
    ax_b.set_xticks(ks)

    ax_b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=FS_TICK - 2)

    ax_b.grid(True, axis="y", alpha=0.25)
    panel_label(ax_b, "b")
    tint(ax_b)
    thin_ticks(ax_b, n=5, which="y")

    cmap_eta = ListedColormap([Color_A, "#ECECEC", Color_B])
    im_c = ax_c.imshow(
        etas.T, origin="lower", aspect="equal",
        cmap=cmap_eta, vmin=-1.5, vmax=1.5,
        extent=[0, 2 * np.pi, 0, 2 * np.pi],
    )

    ax_c.set_xlabel(r"$u$", fontsize=FS)
    ax_c.set_ylabel(r"$v$", fontsize=FS)
    title_c = (
        rf"$\eta(c)$ on $T^2$: $\prod_c \, \eta(c)={prod_eta:+d}$"
    )

    ax_c.set_title(title_c, fontsize=FS, color=FC)
    cb_c = fig.colorbar(
        im_c, ax=ax_c, fraction=0.045, pad=0.02,
        ticks=[-1, 0, +1],
    )

    cb_c.ax.set_yticklabels([r"$-1$", r"$0$", r"$+1$"])
    cb_c.set_label(r"$\eta(c)$", fontsize=FS_TICK, color=FC)
    tint_colorbar(cb_c)
    panel_label(ax_c, "c")
    tint(ax_c)
    thin_ticks(ax_c, n=5)

    n_row = len(sectors)
    sig_mat = np.array([s for _, s in sectors], dtype=float)
    cmap_sig = ListedColormap([Color_A, Color_B])
    ax_d.imshow(
        sig_mat, cmap=cmap_sig, vmin=-1, vmax=1, aspect="auto",
    )
    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels([r"$\eta_\alpha$", r"$\eta_\beta$"], fontsize=FS)
    ax_d.set_yticks(range(n_row))
    ax_d.set_yticklabels([name for name, _ in sectors], fontsize=FS_TICK)
    for i, (_, (ea, eb)) in enumerate(sectors):
        ax_d.text(0, i, rf"${ea:+d}$", ha="center", va="center",
                  color="white", fontsize=FS, fontweight="bold")
        ax_d.text(1, i, rf"${eb:+d}$", ha="center", va="center",
                  color="white", fontsize=FS, fontweight="bold")
    ax_d.set_title(r"signature $(\eta_\alpha,\eta_\beta)$",
                   fontsize=FS, color=FC)
    ax_d.tick_params(axis="both", length=0)
    panel_label(ax_d, "d")
    tint(ax_d)

    fig.savefig(OUT_PDF, format="pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] saved -> {OUT_PDF}")

def main():
    cs = load_character_spectrum()
    etas, prod_eta, n_pos, n_neg, sectors, u_eval, v_eval = compute_panels_c_and_d()
    render(etas, prod_eta, n_pos, n_neg, sectors, u_eval, v_eval, cs)

if __name__ == "__main__":
    main()
