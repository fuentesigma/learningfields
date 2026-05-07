"""J. Fuentes Aguilar, 2025-2026."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from style import (
    PURPLE, ORANGE, GREEN, RED, UI_COLOR, FC, FS, FS_TICK, FIG_WIDTH,
    tint, panel_label, thin_ticks, raincloud,
)

from gap_distribution import (
    cost_matrix_on_edge as _cost_matrix_on_edge_periodic,
    second_best_assignment_cost,
    torus_edges,
)

OUT_PDF = HERE / "figs" / "fig_gap_diagnostics.pdf"
SWEEP_NPZ = HERE / "figs" / "gap_enforced_sweep.npz"
ENSEMBLE_NPZ = HERE / "ensemble_cache_seed0_N10_nev30.npz"

EPSILONS = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

def load_top_row_data():
    if not ENSEMBLE_NPZ.exists():
        raise FileNotFoundError(
            f"Ensemble cache not found: {ENSEMBLE_NPZ}. "
            "Run gap_distribution.py first to produce it."
        )
    d = np.load(ENSEMBLE_NPZ)
    predictions = d["predictions"]
    jacobians = d["jacobians"]
    E_all = d["E_all"]
    n_eval = int(d["u_eval"].size)
    s_y = float(predictions.std() + 1e-12)
    s_g = float(jacobians.std() + 1e-12)

    edges = torus_edges(n_eval)
    print(f"  computing Delta-hat on {len(edges)} edges ...")
    gaps = np.zeros(len(edges))
    C_scales = np.zeros(len(edges))
    for k, ((i_A, j_A), (i_B, j_B)) in enumerate(edges):
        C = _cost_matrix_on_edge_periodic(
            i_A, j_A, i_B, j_B,
            predictions, jacobians, E_all, n_eval, s_y, s_g,
        )
        L_star, L_two = second_best_assignment_cost(C)
        gaps[k] = max(L_two - L_star, 0.0)
        C_scales[k] = float(np.mean(C))
        if (k + 1) % 400 == 0:
            print(f"    {k+1}/{len(edges)}")

    frac_below = {e: float(np.mean(gaps < e)) for e in EPSILONS}
    return gaps, C_scales, frac_below

def load_bottom_row_data():
    if not SWEEP_NPZ.exists():
        raise FileNotFoundError(
            f"Sweep cache not found: {SWEEP_NPZ}. "
            "Run gap_sweep.py first to produce it."
        )
    d = np.load(SWEEP_NPZ, allow_pickle=True)
    original = d["original"].item()
    gap_sweeps = list(d["gap_sweeps"])
    return original, gap_sweeps

def render(gaps, C_scales, frac_below, original, gap_sweeps):
    fig, axes = plt.subplots(
        2, 3, figsize=(FIG_WIDTH, FIG_WIDTH * 0.62),
        constrained_layout=True,
    )
    for ax in axes.ravel():
        ax.set_box_aspect(1.0)

    ax = axes[0, 0]
    pos = gaps[gaps > 0]
    log_gaps = np.log10(pos)
    raincloud(
        ax, [log_gaps], labels=[""], colors=[PURPLE],
        orient="h", violin_width=0.85, point_alpha=0.35, point_size=10,
        rng=np.random.default_rng(0),
    )
    for eps in EPSILONS:
        ax.axvline(np.log10(eps), color=RED, alpha=0.4, linewidth=1.0,
                   linestyle="--")
    ax.set_xlabel(r"$\log_{10}\hat\Delta(x,x')$", fontsize=FS)
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_title(r"$\hat\Delta$ across torus edges",
                 fontsize=FS, color=FC)
    ax.grid(True, axis="x", alpha=0.25)
    panel_label(ax, "a")
    tint(ax)
    thin_ticks(ax, n=5, which="x")

    ax = axes[0, 1]
    eps_grid = np.logspace(
        np.log10(max(gaps.min() / 2, 1e-6)),
        np.log10(max(gaps.max() * 2, max(EPSILONS) * 2)),
        200,
    )
    p_below = np.array([float(np.mean(gaps < e)) for e in eps_grid])
    p_below_plot = np.maximum(p_below, 1.0 / (2 * len(gaps)))
    ax.loglog(eps_grid, p_below_plot, "-", color=PURPLE, linewidth=2.0,
              label=r"$\mathbb{P}(\hat\Delta<\varepsilon)$")
    ax.loglog(
        list(EPSILONS),
        [max(frac_below[e], 1.0 / (2 * len(gaps))) for e in EPSILONS],
        "o", color=RED, markersize=8, label=r"sweep points",
    )
    eps_ref = np.array(EPSILONS)
    ax.loglog(eps_ref, eps_ref / eps_ref.max(), "--",
              color=UI_COLOR, alpha=0.5, linewidth=1.2,
              label=r"slope 1 reference")
    ax.set_xlabel(r"temperature $\varepsilon$", fontsize=FS)
    ax.set_ylabel(r"fraction of edges with $\hat\Delta<\varepsilon$",
                  fontsize=FS)
    ax.set_title(r"failure rate vs $\varepsilon$",
                 fontsize=FS, color=FC)
    ax.legend(fontsize=FS_TICK - 2, loc="lower right", frameon=False)
    ax.grid(True, which="both", alpha=0.25)
    panel_label(ax, "b")
    tint(ax)

    ax = axes[0, 2]
    ax.scatter(C_scales, gaps, s=12, alpha=0.35, color=PURPLE,
               edgecolor="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"edge cost scale $\langle C\rangle$", fontsize=FS)
    ax.set_ylabel(r"uniqueness gap $\hat\Delta$", fontsize=FS)
    ax.set_title(r"$\hat\Delta$ vs cost scale",
                 fontsize=FS, color=FC)
    ax.grid(True, which="both", alpha=0.25)
    panel_label(ax, "c")
    tint(ax)

    colors_tau = [PURPLE, ORANGE, GREEN]
    ax_d = axes[1, 0]
    eps_orig = np.asarray(original["epsilons"])
    ax_d.loglog(eps_orig, np.asarray(original["mean_entropy"]),
                "o-", color=UI_COLOR, linewidth=2.0, markersize=7,
                label=r"canonical (B-deg)")
    for k, sw in enumerate(gap_sweeps):
        ax_d.loglog(np.asarray(sw["epsilons"]), np.asarray(sw["mean_entropy"]),
                    "s-", color=colors_tau[k % len(colors_tau)],
                    linewidth=2.0, markersize=7,
                    label=fr"gap-enforced $\tau={sw['tau']:g}$")
    ax_d.set_xlabel(r"$\varepsilon$", fontsize=FS)
    ax_d.set_ylabel(r"mean row entropy $\bar H(T)$", fontsize=FS)
    ax_d.set_title("row entropy vs $\\varepsilon$",
                   fontsize=FS, color=FC)
    ax_d.legend(fontsize=FS_TICK - 2, loc="lower right", frameon=False)
    ax_d.grid(True, which="both", alpha=0.25)
    panel_label(ax_d, "d")
    tint(ax_d)

    ax_e = axes[1, 1]
    ax_e.loglog(eps_orig, np.asarray(original["mean_frobenius"]),
                "o-", color=UI_COLOR, linewidth=2.0, markersize=7,
                label=r"canonical (B-deg)")
    for k, sw in enumerate(gap_sweeps):
        ax_e.loglog(np.asarray(sw["epsilons"]), np.asarray(sw["mean_frobenius"]),
                    "s-", color=colors_tau[k % len(colors_tau)],
                    linewidth=2.0, markersize=7,
                    label=fr"gap-enforced $\tau={sw['tau']:g}$")
    ax_e.set_xlabel(r"$\varepsilon$", fontsize=FS)
    ax_e.set_ylabel(r"$\|T^{(\varepsilon)}-T^{\mathrm{hung}}\|_F$",
                    fontsize=FS)
    ax_e.set_title(r"distance to $T^{\mathrm{hung}}$",
                   fontsize=FS, color=FC)
    ax_e.legend(fontsize=FS_TICK - 2, loc="lower right", frameon=False)
    ax_e.grid(True, which="both", alpha=0.25)
    panel_label(ax_e, "e")
    tint(ax_e)

    ax_f = axes[1, 2]
    if gap_sweeps:
        taus = np.array([float(sw["tau"]) for sw in gap_sweeps])
        realised = np.array([float(np.mean(sw["per_edge_gap"]))
                             for sw in gap_sweeps])

        rates_f = []
        for sw in gap_sweeps:
            eps_arr = np.asarray(sw["epsilons"], dtype=float)
            f_arr = np.asarray(sw["mean_frobenius"], dtype=float)
            finite = f_arr > 0
            if finite.sum() >= 2:
                slope, _ = np.polyfit(1.0 / eps_arr[finite],
                                      np.log(f_arr[finite]), 1)
                rates_f.append(-slope)
            else:
                rates_f.append(np.nan)
        rates_f = np.array(rates_f)

        rates_h = []
        for sw in gap_sweeps:
            eps_arr = np.asarray(sw["epsilons"], dtype=float)
            h_arr = np.asarray(sw["mean_entropy"], dtype=float)
            finite = h_arr > 0
            if finite.sum() >= 2:
                slope, _ = np.polyfit(1.0 / eps_arr[finite],
                                      np.log(h_arr[finite]), 1)
                rates_h.append(-slope)
            else:
                rates_h.append(np.nan)
        rates_h = np.array(rates_h)

        ax_f.plot(realised, rates_h, "o-", color=PURPLE, linewidth=2.0,
                  markersize=9, label=r"$\hat\Delta$ from $\bar H$")
        ax_f.plot(realised, rates_f, "s-", color=ORANGE, linewidth=2.0,
                  markersize=9, label=r"$\hat\Delta$ from $\|T-T_{\mathrm{hung}}\|_F$")
        lim = float(max(realised.max(), rates_f.max(), rates_h.max()) * 1.1)
        ax_f.plot([0, lim], [0, lim], "--", color=UI_COLOR, alpha=0.5,
                  linewidth=1.2, label=r"$\hat\Delta=\Delta_\star$")
        ax_f.set_xlim(0, lim)
        ax_f.set_ylim(0, lim)
    ax_f.set_xlabel(r"realised gap $\Delta_\star$", fontsize=FS)
    ax_f.set_ylabel(r"fitted decay rate $\hat\Delta$", fontsize=FS)
    ax_f.set_title(r"$\hat\Delta\sim\Delta_\star$ (Lemma)",
                   fontsize=FS, color=FC)
    ax_f.legend(fontsize=FS_TICK - 2, loc="upper left", frameon=False)
    ax_f.grid(True, alpha=0.25)
    panel_label(ax_f, "f")
    tint(ax_f)
    thin_ticks(ax_f, n=4)

    fig.savefig(OUT_PDF, format="pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] saved -> {OUT_PDF}")

def main():
    gaps, C_scales, frac_below = load_top_row_data()
    original, gap_sweeps = load_bottom_row_data()
    render(gaps, C_scales, frac_below, original, gap_sweeps)

if __name__ == "__main__":
    main()
