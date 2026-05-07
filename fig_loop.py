"""J. Fuentes Aguilar, 2025-2026."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from style import (
    PURPLE, ORANGE, RED, UI_COLOR, FC, FS, FS_TICK, FIG_WIDTH,
    tint, panel_label, tint_colorbar, thin_ticks, raincloud,
)

from core import (
    torus_tangent,
    holonomy_field,
    fibre_jets,
    loop_holonomy,
    rotation_AB,
    jet_costm,
    transfer_operator_entropic,
)

OUT_PDF = HERE / "figs" / "fig_loop_diagnostics.pdf"
ENSEMBLE_NPZ = HERE / "ensemble_cache_seed0_N10_nev30.npz"
N_EVAL = 30
EPSILONS_SWEEP = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
EPS_MAIN = 0.5

def load_ensemble():
    if not ENSEMBLE_NPZ.exists():
        raise FileNotFoundError(str(ENSEMBLE_NPZ))
    d = np.load(ENSEMBLE_NPZ)
    preds = d["predictions"]
    jacs = d["jacobians"]
    E_all = d["E_all"]
    u_eval = d["u_eval"]
    v_eval = d["v_eval"]
    s_y = float(preds.std() + 1e-12)
    s_g = float(jacs.std() + 1e-12)
    return preds, jacs, E_all, u_eval, v_eval, s_y, s_g

def top_row_data(preds, jacs, E_all, u_eval, v_eval, s_y, s_g):
    print("[top] computing holonomy field at eps =", EPS_MAIN, "...")
    h_field = holonomy_field(
        preds, jacs, E_all, N_EVAL, s_y, s_g,
        eps=EPS_MAIN, side=1, mode="entropic",
    )

    pred_std = np.zeros_like(h_field)
    grad_fro = np.zeros_like(h_field)
    n_pl = N_EVAL - 1
    for i in range(n_pl):
        for j in range(n_pl):

            idx = i * N_EVAL + j

            pred_std[i, j] = float(np.std(preds[:, idx]))

            g_vec = jacs[:, idx, :]
            g_centered = g_vec - g_vec.mean(axis=0, keepdims=True)
            cov = g_centered.T @ g_centered / max(1, g_vec.shape[0] - 1)
            grad_fro[i, j] = float(np.linalg.norm(cov, ord="fro"))

    print("[top] epsilon sweep for mean h ...")
    eps_sweep = list(EPSILONS_SWEEP)
    mean_h = []
    for eps in eps_sweep:
        hf = holonomy_field(
            preds, jacs, E_all, N_EVAL, s_y, s_g,
            eps=float(eps), side=1, mode="entropic",
        )
        mean_h.append(float(hf.mean()))
        print(f"    eps={eps:.2f}  mean h = {mean_h[-1]:.4f}")
    return h_field, pred_std, grad_fro, np.asarray(eps_sweep), np.asarray(mean_h)

def bottom_row_data(preds, jacs, E_all, u_eval, v_eval, s_y, s_g,
                    beta_y=3.0, beta_g=3.0, eps=1e-4):
    N = preds.shape[0]

    a = np.arange(N) - (N - 1) / 2.0
    delta = beta_y * s_y * a

    theta = 2.0 * np.pi * np.arange(N) / N
    v = beta_g * s_g * np.stack([np.cos(theta), np.sin(theta)], axis=-1)

    tagged_preds = preds + delta[:, None]
    tagged_jacs = jacs + v[:, None, :]

    print("[bottom] computing zero-baseline holonomy field ...")
    h_zero = holonomy_field(
        tagged_preds, tagged_jacs, E_all, N_EVAL, s_y, s_g,
        eps=eps, side=1, mode="entropic",
    )

    print("[bottom] computing barycentric displacement field ...")
    n_pl = N_EVAL - 1
    bary = np.zeros((n_pl, n_pl))
    for i in range(n_pl):
        for j in range(n_pl):
            _, disp = loop_holonomy(
                i, j, 1, tagged_preds, tagged_jacs, E_all, N_EVAL,
                s_y, s_g, eps=eps, mode="entropic",
            )
            bary[i, j] = float(disp)
    return h_zero, bary

def render(top, bottom, u_eval, v_eval):
    h_field, pred_std, grad_fro, eps_sweep, mean_h = top
    h_zero, bary = bottom

    fig, axes = plt.subplots(
        2, 3, figsize=(FIG_WIDTH, FIG_WIDTH * 0.62),
        constrained_layout=True,
    )
    for ax in axes.ravel():
        ax.set_box_aspect(1.0)

    ax = axes[0, 0]
    rho_a, _ = spearmanr(h_field.ravel(), pred_std.ravel())
    ax.scatter(pred_std.ravel(), h_field.ravel(),
               s=14, alpha=0.5, color=PURPLE, edgecolors="none")
    ax.set_xlabel(r"prediction std $\sigma_y(x)$", fontsize=FS)
    ax.set_ylabel(r"$h_{\mathrm{op}}(c)$", fontsize=FS)
    ax.set_title(rf"$h_{{\mathrm{{op}}}}$ vs spread, $\rho={rho_a:+.3f}$",
                 fontsize=FS, color=FC)
    ax.grid(True, alpha=0.25)
    panel_label(ax, "a")
    tint(ax)
    thin_ticks(ax, n=4)

    ax = axes[0, 1]
    rho_b, _ = spearmanr(h_field.ravel(), grad_fro.ravel())
    ax.scatter(grad_fro.ravel(), h_field.ravel(),
               s=14, alpha=0.5, color=PURPLE, edgecolors="none")
    ax.set_xlabel(r"$\|\mathrm{Cov}(\nabla f)\|_{\mathrm{F}}$", fontsize=FS)
    ax.set_ylabel(r"$h_{\mathrm{op}}(c)$", fontsize=FS)
    ax.set_title(rf"$h_{{\mathrm{{op}}}}$ vs grad. cov., $\rho={rho_b:+.3f}$",
                 fontsize=FS, color=FC)
    ax.grid(True, alpha=0.25)
    panel_label(ax, "b")
    tint(ax)
    thin_ticks(ax, n=4)

    ax = axes[0, 2]
    ax.plot(eps_sweep, mean_h, "o-", color=PURPLE, linewidth=2.2, markersize=8)
    ax.set_xscale("log")
    ax.set_xlabel(r"entropic temperature $\varepsilon$", fontsize=FS)
    ax.set_ylabel(r"mean $h_{\mathrm{op}}$", fontsize=FS)
    ax.set_title(r"$\bar h_{\mathrm{op}}$ as a function of $\varepsilon$",
                 fontsize=FS, color=FC)
    ax.grid(True, which="both", alpha=0.25)
    panel_label(ax, "c")
    tint(ax)
    thin_ticks(ax, n=4, which="y")

    ax = axes[1, 0]
    extent = [u_eval[0], u_eval[-2], v_eval[0], v_eval[-2]]
    im_d = ax.imshow(h_zero.T, origin="lower", aspect="equal",
                     cmap="Pastel2", extent=extent)
    ax.set_xlabel(r"$u$", fontsize=FS)
    ax.set_ylabel(r"$v$", fontsize=FS)
    ax.set_title(r"zero-baseline $h_{\mathrm{op}}$ field",
                 fontsize=FS, color=FC)
    cb_d = fig.colorbar(im_d, ax=ax, fraction=0.045, pad=0.02)
    cb_d.set_label(r"$h_{\mathrm{op}}$", fontsize=FS_TICK, color=FC)
    cb_d.locator = plt.matplotlib.ticker.MaxNLocator(nbins=4)
    cb_d.update_ticks()
    tint_colorbar(cb_d)
    panel_label(ax, "d")
    tint(ax)
    thin_ticks(ax, n=4)

    ax = axes[1, 1]
    im_e = ax.imshow(bary.T, origin="lower", aspect="equal",
                     cmap="Pastel2", extent=extent)
    ax.set_xlabel(r"$u$", fontsize=FS)
    ax.set_ylabel(r"$v$", fontsize=FS)
    ax.set_title(r"zero-baseline displacement $\bar\delta$",
                 fontsize=FS, color=FC)
    cb_e = fig.colorbar(im_e, ax=ax, fraction=0.045, pad=0.02)
    cb_e.set_label(r"$\bar\delta$", fontsize=FS_TICK, color=FC)
    cb_e.locator = plt.matplotlib.ticker.MaxNLocator(nbins=4)
    cb_e.update_ticks()
    tint_colorbar(cb_e)
    panel_label(ax, "e")
    tint(ax)
    thin_ticks(ax, n=4)

    ax = axes[1, 2]
    vals = h_zero.ravel()
    vals = vals[np.isfinite(vals) & (vals > 0)]
    log_vals = np.log10(vals)
    raincloud(
        ax, [log_vals], labels=[""], colors=[PURPLE],
        orient="h", violin_width=0.85, point_alpha=0.35, point_size=10,
        rng=np.random.default_rng(0),
    )
    mean_hz = float(h_zero.mean())
    med_hz = float(np.median(h_zero))
    ax.axvline(np.log10(max(mean_hz, 1e-30)), linestyle="--",
               color=UI_COLOR, alpha=0.7, linewidth=1.2,
               label=rf"mean $={mean_hz:.2e}$")
    ax.axvline(np.log10(max(med_hz, 1e-30)), linestyle=":",
               color=UI_COLOR, alpha=0.7, linewidth=1.2,
               label=rf"median $={med_hz:.2e}$")
    ax.set_xlabel(r"$\log_{10}\,h_{\mathrm{op}}$", fontsize=FS)
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_title(r"$h_{\mathrm{op}}$ at numerical floor",
                 fontsize=FS, color=FC)
    ax.legend(fontsize=FS_TICK - 2, frameon=False, loc="lower right")
    ax.grid(True, axis="x", alpha=0.25)
    panel_label(ax, "f")
    tint(ax)
    thin_ticks(ax, n=5, which="x")

    fig.savefig(OUT_PDF, format="pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] saved -> {OUT_PDF}")

def main():
    preds, jacs, E_all, u_eval, v_eval, s_y, s_g = load_ensemble()
    top = top_row_data(preds, jacs, E_all, u_eval, v_eval, s_y, s_g)
    bottom = bottom_row_data(preds, jacs, E_all, u_eval, v_eval, s_y, s_g)
    render(top, bottom, u_eval, v_eval)

if __name__ == "__main__":
    main()
