"""J. Fuentes Aguilar, 2025-2026."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from core import (
    evaluate_ensemble,
    fibre_jets,
    ground_truth_field,
    jet_costm,
    rotation_AB,
    torus_tangent,
    train_ensemble,
)

def cost_matrix_on_edge(i_A, j_A, i_B, j_B,
                        predictions, jacobians, E_all, n_eval, s_y, s_g):
    A = fibre_jets(i_A, j_A, predictions, jacobians, n_eval)
    B = fibre_jets(i_B, j_B, predictions, jacobians, n_eval)
    R_AB = rotation_AB(i_A, j_A, i_B, j_B, E_all, n_eval)
    return jet_costm(A, B, s_y, s_g, R_AB)

def second_best_assignment_cost(C):
    n = C.shape[0]
    row, col = linear_sum_assignment(C)
    L_star = float(C[row, col].sum())

    BIG = 1e18
    L_two = np.inf
    for k in range(n):
        C_fb = C.copy()
        C_fb[row[k], col[k]] = BIG
        r, c = linear_sum_assignment(C_fb)
        cost_fb = float(C_fb[r, c].sum())
        if cost_fb < L_two:
            L_two = cost_fb

    return L_star, L_two

def torus_edges(n_eval):
    edges = []
    for i in range(n_eval):
        for j in range(n_eval):

            edges.append(((i, j), (i, (j + 1) % n_eval)))

            edges.append((((i, j), ((i + 1) % n_eval, j))))
    return edges

def run(
    R_param=1.0,
    r_param=1.0,
    n_samples=1000,
    n_eval=30,
    n_models=10,
    seed0=0,
    epochs=200,
    epsilons=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
    fig_path="figs/gap_distribution.pdf",
    quiet=False,
):

    def log(*a):
        if not quiet:
            print(*a)

    log("=" * 72)
    log("PHASE 0 — Uniqueness-gap distribution on the torus")
    log("=" * 72)
    log(f"  R={R_param}, r={r_param}, n_samples={n_samples}, "
        f"n_eval={n_eval}, N={n_models}, seeds={seed0}..{seed0 + n_models - 1}")

    cache_file = (Path(__file__).resolve().parent
                  / f"ensemble_cache_seed{seed0}_N{n_models}_nev{n_eval}.npz")
    if cache_file.exists():
        log(f"Loading cached ensemble evaluation from {cache_file.name}")
        d = np.load(cache_file)
        predictions = d["predictions"]
        jacobians = d["jacobians"]
        E_all = d["E_all"]
        u_eval = d["u_eval"]
        v_eval = d["v_eval"]
    else:
        log("Training ensemble...")
        torch.manual_seed(seed0)
        np.random.seed(seed0)
        u_samples = np.random.uniform(0, 2 * np.pi, n_samples)
        v_samples = np.random.uniform(0, 2 * np.pi, n_samples)
        targets = ground_truth_field(u_samples, v_samples)
        models = train_ensemble(
            u_samples, v_samples, targets,
            R_param=R_param, r_param=r_param,
            n_models=n_models, epochs=epochs, seed0=seed0,
        )
        log("Evaluating ensemble on 30x30 grid...")
        u_eval = np.linspace(0, 2 * np.pi, n_eval)
        v_eval = np.linspace(0, 2 * np.pi, n_eval)
        u_grid, v_grid = np.meshgrid(u_eval, v_eval)
        predictions, jacobians = evaluate_ensemble(
            models, u_grid, v_grid, R_param=R_param, r_param=r_param
        )
        u_flat = u_grid.ravel()
        v_flat = v_grid.ravel()
        E_all = torus_tangent(u_flat, v_flat, R_param=R_param, r_param=r_param)

    s_y = predictions.std() + 1e-12
    s_g = jacobians.std() + 1e-12
    log(f"  calibration scales: s_y={s_y:.4g}, s_g={s_g:.4g}")

    edges = torus_edges(n_eval)
    log(f"Computing Δ̂ on {len(edges)} torus edges "
        f"(second-best via Murty-level-1, N={n_models} LAPs per edge)...")

    gaps = np.zeros(len(edges))
    L_stars = np.zeros(len(edges))
    L_twos = np.zeros(len(edges))
    C_scales = np.zeros(len(edges))
    C_max = np.zeros(len(edges))

    for k, ((i_A, j_A), (i_B, j_B)) in enumerate(edges):
        C = cost_matrix_on_edge(
            i_A, j_A, i_B, j_B,
            predictions, jacobians, E_all, n_eval, s_y, s_g,
        )
        L_star, L_two = second_best_assignment_cost(C)
        L_stars[k] = L_star
        L_twos[k] = L_two
        gaps[k] = L_two - L_star
        C_scales[k] = C.mean()
        C_max[k] = C.max()
        if (k + 1) % 300 == 0:
            log(f"  ... {k + 1}/{len(edges)} edges")

    gaps = np.maximum(gaps, 0.0)

    mean = float(gaps.mean())
    median = float(np.median(gaps))
    p05 = float(np.percentile(gaps, 5))
    p95 = float(np.percentile(gaps, 95))
    log("")
    log("Uniqueness-gap Δ̂ summary (raw units of cost = squared calibrated jet distance):")
    log(f"  mean   = {mean:.4e}")
    log(f"  median = {median:.4e}")
    log(f"  5th %  = {p05:.4e}")
    log(f"  95th % = {p95:.4e}")
    log(f"  min    = {gaps.min():.4e}")
    log(f"  max    = {gaps.max():.4e}")

    frac_below = {eps: float(np.mean(gaps < eps)) for eps in epsilons}
    log("")
    log("Fraction of edges with Δ̂ < ε   (Lemma-hypothesis failure rate at temperature ε):")
    for eps in epsilons:
        log(f"  ε = {eps:<5g}   frac(Δ̂<ε) = {frac_below[eps]:.3f}")

    uniform_gap_holds = bool(gaps.min() > max(epsilons))
    log("")
    log(f"Uniform gap Δ⋆ > max(ε) holds?  {uniform_gap_holds}")
    log(f"  (min Δ̂ = {gaps.min():.4e};  max ε in sweep = {max(epsilons)})")

    log("")
    log(f"Writing figure to {fig_path}...")
    _plot(gaps, C_scales, epsilons, frac_below, fig_path)

    return {
        "gaps": gaps,
        "L_stars": L_stars,
        "L_twos": L_twos,
        "C_scales": C_scales,
        "C_max": C_max,
        "epsilons": list(epsilons),
        "frac_below": frac_below,
        "stats": {"mean": mean, "median": median, "p05": p05, "p95": p95,
                  "min": float(gaps.min()), "max": float(gaps.max())},
        "s_y": float(s_y),
        "s_g": float(s_g),
        "uniform_gap_holds": uniform_gap_holds,
        "n_edges": len(edges),
        "n_models": n_models,
        "n_eval": n_eval,
    }

def _plot(gaps, C_scales, epsilons, frac_below, fig_path):
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)

    from style import FC, FS, PURPLE, RED, UI_COLOR, tint, panel_label
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.4), constrained_layout=True)

    ax = axes[0]
    pos = gaps[gaps > 0]
    if pos.size > 1:
        bins = np.logspace(np.log10(pos.min()),
                           np.log10(pos.max()), 60)
        ax.hist(pos, bins=bins, color=PURPLE,
                edgecolor="white", alpha=0.85)
        ax.set_xscale("log")
    else:
        ax.hist(gaps, bins=40, color=PURPLE, edgecolor="white", alpha=0.85)
    for eps in epsilons:
        ax.axvline(eps, color=RED, alpha=0.4, linewidth=1, linestyle="--")
    ax.set_xlabel(r"uniqueness gap $\hat\Delta(x,x')$", fontsize=FS)
    ax.set_ylabel("edge count", fontsize=FS)
    ax.set_title(r"distribution of $\hat\Delta$ over torus edges",
                 fontsize=FS, color=FC)
    ax.grid(True, which="both", alpha=0.3)
    panel_label(ax, "a")
    tint(ax)

    ax = axes[1]
    eps_grid = np.logspace(
        np.log10(max(gaps.min() / 2, 1e-6)),
        np.log10(max(gaps.max() * 2, max(epsilons) * 2)),
        200,
    )
    p_below = np.array([np.mean(gaps < e) for e in eps_grid])

    p_below_plot = np.maximum(p_below, 1.0 / (2 * len(gaps)))
    ax.loglog(eps_grid, p_below_plot, "-", color=PURPLE, linewidth=2,
              label=r"$\mathbb{P}(\hat\Delta < \varepsilon)$")

    ax.loglog(list(epsilons), [max(frac_below[e], 1.0 / (2 * len(gaps)))
                                for e in epsilons],
              "o", color=RED, markersize=9,
              label=r"Fig. 2(d) sweep points")

    eps_ref = np.array(list(epsilons))
    ax.loglog(eps_ref, eps_ref / eps_ref.max(), "--", color="gray",
              alpha=0.4, linewidth=1, label=r"slope 1 reference $\propto\varepsilon$")
    ax.set_xlabel(r"temperature $\varepsilon$", fontsize=FS)
    ax.set_ylabel(r"fraction of edges with $\hat\Delta < \varepsilon$",
                  fontsize=FS)
    ax.set_title(r"Lemma-hypothesis failure rate vs $\varepsilon$",
                 fontsize=FS, color=FC)
    ax.legend(fontsize=FS - 3, loc="lower right", frameon=False)
    ax.grid(True, which="both", alpha=0.3)
    panel_label(ax, "b")
    tint(ax)

    ax = axes[2]
    ax.scatter(C_scales, gaps, s=10, alpha=0.35, color=PURPLE,
               edgecolor="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("edge cost scale $\\langle C\\rangle$", fontsize=FS)
    ax.set_ylabel(r"uniqueness gap $\hat\Delta$", fontsize=FS)
    ax.set_title(r"$\hat\Delta$ vs cost-matrix scale",
                 fontsize=FS, color=FC)
    ax.grid(True, which="both", alpha=0.3)
    panel_label(ax, "c")
    tint(ax)
    fig.savefig(fig_path, format="pdf", dpi=150)
    plt.close(fig)

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-models", type=int, default=10,
                   help="ensemble size N (default 10 matches §8)")
    p.add_argument("--n-eval", type=int, default=30,
                   help="grid side (default 30)")
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--fig-path", type=str,
                   default="figs/gap_distribution.pdf")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    run(
        n_models=args.n_models,
        n_eval=args.n_eval,
        n_samples=args.n_samples,
        epochs=args.epochs,
        seed0=args.seed0,
        fig_path=args.fig_path,
        quiet=args.quiet,
    )
