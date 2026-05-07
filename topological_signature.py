"""J. Fuentes Aguilar, 2025-2026."""

from __future__ import annotations

import json
import time
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
        L_two = min(L_two, float(C_fb[r, c].sum()))
    return L_star, L_two

def hungarian_permutation(C):
    n = C.shape[0]
    row, col = linear_sum_assignment(C)
    T = np.zeros((n, n))
    T[row, col] = 1.0
    return T

def non_periodic_edges(n_eval):
    edges = []
    for i in range(n_eval):
        for j in range(n_eval):
            if i + 1 < n_eval:
                edges.append(((i, j), (i + 1, j)))
            if j + 1 < n_eval:
                edges.append(((i, j), (i, j + 1)))
    return edges

def _idx(i, j, n):
    return i % n, j % n

def build_edge_transport_cache(predictions, jacobians, E_all, n_eval,
                                s_y, s_g, *, periodic: bool,
                                mode: str = "hungarian", eps: float = 0.5):
    from core import transfer_operator_entropic

    cache = {}
    if periodic:

        for i in range(n_eval):
            for j in range(n_eval):
                for di, dj in ((1, 0), (0, 1)):
                    i2 = (i + di) % n_eval
                    j2 = (j + dj) % n_eval
                    C = cost_matrix_on_edge(i, j, i2, j2,
                                            predictions, jacobians, E_all,
                                            n_eval, s_y, s_g)
                    if mode == "hungarian":
                        T = hungarian_permutation(C)
                    elif mode == "entropic":
                        T = transfer_operator_entropic(C, eps=eps)
                    else:
                        raise ValueError(f"unknown mode {mode}")
                    cache[((i, j), (i2, j2))] = T
    else:
        for i in range(n_eval):
            for j in range(n_eval):
                if i + 1 < n_eval:
                    C = cost_matrix_on_edge(i, j, i + 1, j,
                                            predictions, jacobians, E_all,
                                            n_eval, s_y, s_g)
                    if mode == "hungarian":
                        T = hungarian_permutation(C)
                    else:
                        T = transfer_operator_entropic(C, eps=eps)
                    cache[((i, j), (i + 1, j))] = T
                if j + 1 < n_eval:
                    C = cost_matrix_on_edge(i, j, i, j + 1,
                                            predictions, jacobians, E_all,
                                            n_eval, s_y, s_g)
                    if mode == "hungarian":
                        T = hungarian_permutation(C)
                    else:
                        T = transfer_operator_entropic(C, eps=eps)
                    cache[((i, j), (i, j + 1))] = T
    return cache

def get_edge_transport(cache, a, b):
    if (a, b) in cache:
        return cache[(a, b)]
    if (b, a) in cache:
        return cache[(b, a)].T
    raise KeyError(f"no transport for edge ({a}, {b}) in cache")

def loop_holonomy_from_cache(vertex_path, cache):
    T_list = [get_edge_transport(cache, a, b)
              for a, b in zip(vertex_path[:-1], vertex_path[1:])]
    N = T_list[0].shape[0]
    H = np.eye(N)
    for T in T_list:
        H = T.T @ H
    return H

def rectangle_boundary_path(i0, j0, H_rows, W_cols):
    path = [(i0, j0)]
    for k in range(1, W_cols + 1):
        path.append((i0, j0 + k))
    for k in range(1, H_rows + 1):
        path.append((i0 + k, j0 + W_cols))
    for k in range(1, W_cols + 1):
        path.append((i0 + k + (-1), j0 + W_cols - k))

    path = [(i0, j0)]
    for k in range(1, W_cols + 1):
        path.append((i0, j0 + k))
    for k in range(1, H_rows + 1):
        path.append((i0 + k, j0 + W_cols))
    for k in range(1, W_cols + 1):
        path.append((i0 + H_rows, j0 + W_cols - k))
    for k in range(1, H_rows + 1):
        path.append((i0 + H_rows - k, j0))
    return path

def plaquette_boundary_path(i0, j0):
    return [(i0, j0), (i0, j0 + 1), (i0 + 1, j0 + 1), (i0 + 1, j0), (i0, j0)]

def eta_of(H, threshold=1e-12):
    d = float(np.linalg.det(H))
    if abs(d) < threshold:
        return 0, d
    return int(np.sign(d)), d

def permutation_matrix(sigma, N):
    P = np.zeros((N, N))
    for i, j in enumerate(sigma):
        P[i, j] = 1.0
    return P

def is_alpha_seam(a, b, n_eval):
    (ia, ja), (ib, jb) = a, b
    return ia == ib and ja == n_eval - 1 and jb == 0

def is_beta_seam(a, b, n_eval):
    (ia, ja), (ib, jb) = a, b
    return ia == n_eval - 1 and ib == 0 and ja == jb

def twisted_loop_holonomy(vertex_path, cache, sigma_matrix, n_eval,
                          axis: str):
    H = np.eye(sigma_matrix.shape[0])
    for a, b in zip(vertex_path[:-1], vertex_path[1:]):
        T_ab = get_edge_transport(cache, a, b)

        twist = False
        if axis == 'alpha':
            if is_alpha_seam(a, b, n_eval):
                twist = True
                sign = +1
            elif is_alpha_seam(b, a, n_eval):
                twist = True
                sign = -1
            else:
                twist = False
        elif axis == 'beta':
            if is_beta_seam(a, b, n_eval):
                twist = True
                sign = +1
            elif is_beta_seam(b, a, n_eval):
                twist = True
                sign = -1
            else:
                twist = False
        else:
            raise ValueError(axis)

        if twist:
            if sign == +1:
                T_ab = sigma_matrix @ T_ab
            else:

                T_ab = sigma_matrix.T @ T_ab

        H = T_ab.T @ H
    return H

def prepare_ensemble(R_param=1.0, r_param=1.0, n_samples=1000, n_eval=30,
                     n_models=10, epochs=200, seed0=0, cache_dir=None):
    cache_dir = Path(cache_dir) if cache_dir else Path(__file__).resolve().parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"ensemble_cache_seed{seed0}_N{n_models}_nev{n_eval}.npz"

    if cache_file.exists():
        print(f"  Loading cached ensemble evaluation from {cache_file.name}")
        d = np.load(cache_file)
        return (d["predictions"], d["jacobians"], d["E_all"],
                float(d["s_y"]), float(d["s_g"]),
                d["u_eval"], d["v_eval"])

    print("  Training ensemble (not cached)...")
    torch.manual_seed(seed0)
    np.random.seed(seed0)
    u_samples = np.random.uniform(0, 2 * np.pi, n_samples)
    v_samples = np.random.uniform(0, 2 * np.pi, n_samples)
    targets = ground_truth_field(u_samples, v_samples)
    models = train_ensemble(u_samples, v_samples, targets,
                            R_param=R_param, r_param=r_param,
                            n_models=n_models, epochs=epochs, seed0=seed0)
    u_eval = np.linspace(0, 2 * np.pi, n_eval)
    v_eval = np.linspace(0, 2 * np.pi, n_eval)
    u_grid, v_grid = np.meshgrid(u_eval, v_eval)
    predictions, jacobians = evaluate_ensemble(models, u_grid, v_grid,
                                                R_param=R_param, r_param=r_param)
    u_flat = u_grid.ravel(); v_flat = v_grid.ravel()
    E_all = torus_tangent(u_flat, v_flat, R_param=R_param, r_param=r_param)
    s_y = float(predictions.std() + 1e-12)
    s_g = float(jacobians.std() + 1e-12)
    np.savez_compressed(cache_file, predictions=predictions, jacobians=jacobians,
                         E_all=E_all, s_y=s_y, s_g=s_g,
                         u_eval=u_eval, v_eval=v_eval)
    print(f"    cached -> {cache_file.name}")
    return predictions, jacobians, E_all, s_y, s_g, u_eval, v_eval

def compute_gap_distribution(predictions, jacobians, E_all, n_eval, s_y, s_g,
                              *, quiet=False):
    edges = non_periodic_edges(n_eval)
    gaps = np.zeros(len(edges))
    for k, (a, b) in enumerate(edges):
        C = cost_matrix_on_edge(*a, *b, predictions, jacobians, E_all,
                                n_eval, s_y, s_g)
        L_star, L_two = second_best_assignment_cost(C)
        gaps[k] = max(L_two - L_star, 0.0)
        if (not quiet) and (k + 1) % 400 == 0:
            print(f"    ... Delta_hat on {k+1}/{len(edges)} edges")
    return gaps, edges

def random_rectangles(n_eval, n_rect, rng, *, max_side=None):
    max_H = (max_side or n_eval - 1)
    max_W = (max_side or n_eval - 1)
    rect = []
    for _ in range(n_rect):
        H = int(rng.integers(1, min(max_H, n_eval - 1) + 1))
        W = int(rng.integers(1, min(max_W, n_eval - 1) + 1))
        i0 = int(rng.integers(0, n_eval - H))
        j0 = int(rng.integers(0, n_eval - W))
        rect.append((i0, j0, H, W))
    return rect

def check_abelian_stokes(cache, n_eval, rects):
    etaB = []
    etaF = []
    detB = []
    detFprod = []
    sizes = []
    for (i0, j0, H, W) in rects:

        path = rectangle_boundary_path(i0, j0, H, W)
        H_bd = loop_holonomy_from_cache(path, cache)
        e_b, d_b = eta_of(H_bd)
        etaB.append(e_b)
        detB.append(d_b)

        prod = 1
        prod_det = 1.0
        for a in range(H):
            for b in range(W):
                H_f = loop_holonomy_from_cache(
                    plaquette_boundary_path(i0 + a, j0 + b), cache)
                e_f, d_f = eta_of(H_f)
                prod *= e_f if e_f != 0 else +1
                prod_det *= d_f
        etaF.append(prod)
        detFprod.append(prod_det)
        sizes.append(H * W)
    return (np.asarray(etaB), np.asarray(etaF),
            np.asarray(detB), np.asarray(detFprod),
            np.asarray(sizes))

def closed_torus_sum_rule(cache_periodic, n_eval):
    etas = np.zeros((n_eval, n_eval), dtype=np.int64)
    dets = np.zeros((n_eval, n_eval), dtype=np.float64)
    for i in range(n_eval):
        for j in range(n_eval):

            path = [(i, j),
                    ((i) % n_eval, (j + 1) % n_eval),
                    ((i + 1) % n_eval, (j + 1) % n_eval),
                    ((i + 1) % n_eval, (j) % n_eval),
                    (i, j)]
            H = loop_holonomy_from_cache(path, cache_periodic)
            e, d = eta_of(H)
            etas[i, j] = e
            dets[i, j] = d

    zero_mask = (etas == 0)
    n_zero = int(np.sum(zero_mask))
    safe = np.where(zero_mask, 1, etas)
    prod = int(np.prod(safe))
    return etas, dets, prod, n_zero

def alpha_beta_vertex_paths(n_eval, row0=0, col0=0):
    alpha = [(row0, k) for k in range(n_eval)] + [(row0, 0)]
    beta = [(k, col0) for k in range(n_eval)] + [(0, col0)]
    return alpha, beta

def measure_signature(cache_periodic, n_eval, *, twist=None, sigma=None):
    alpha, beta = alpha_beta_vertex_paths(n_eval)
    if twist is None:
        H_a = loop_holonomy_from_cache(alpha, cache_periodic)
        H_b = loop_holonomy_from_cache(beta, cache_periodic)
    else:
        H_a = twisted_loop_holonomy(alpha, cache_periodic, sigma, n_eval, twist)
        H_b = twisted_loop_holonomy(beta, cache_periodic, sigma, n_eval, twist)
    eta_a, det_a = eta_of(H_a)
    eta_b, det_b = eta_of(H_b)
    return (eta_a, eta_b), (det_a, det_b)

def bias_tag_zero_signature():
    return (+1, +1), (1.0, 1.0)

from style import (
    PURPLE, ORANGE, GREEN, RED, UI_COLOR, FC, FS,
    tint as _tint, panel_label as _panel_letter, tint_colorbar,
)

def render_figure(out_path, panel_data):
    fig = plt.figure(figsize=(20, 12.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])

    ax_a = fig.add_subplot(gs[0, 0])
    gaps = panel_data["a"]["gaps"]
    ax_a.hist(np.log10(np.maximum(gaps, 1e-16)), bins=40,
              color=PURPLE, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax_a.set_xlabel(r"$\log_{10}\ \hat\Delta(x, x')$", fontsize=FS, color=FC)
    ax_a.set_ylabel("edges", fontsize=FS, color=FC)
    ax_a.set_title("empirical gap distribution on torus edges",
                   fontsize=FS, color=FC)
    ax_a.grid(True, alpha=0.25)
    _panel_letter(ax_a, "a")
    _tint(ax_a)

    ax_b = fig.add_subplot(gs[0, 1])
    data_b = panel_data["b"]
    epsilons = data_b["epsilons"]
    ax_b.plot(epsilons, data_b["original_frobenius"], marker="o",
              linewidth=1.8, color=UI_COLOR, markersize=7,
              label=r"canonical (B-deg)")
    colors = [PURPLE, ORANGE, GREEN]
    for k, sweep in enumerate(data_b["gap_sweeps"]):
        ax_b.plot(sweep["epsilons"], sweep["frobenius"],
                  marker="s", linewidth=1.8, color=colors[k], markersize=7,
                  label=fr"gap-enforced $\tau={sweep['tau']:g}$")
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel(r"$\varepsilon$", fontsize=FS, color=FC)
    ax_b.set_ylabel(r"$\|T^{(\varepsilon)} - T^{\rm hung}\|_F$",
                    fontsize=FS, color=FC)
    ax_b.set_title("gap-enforced $\\varepsilon$-sweep: exponential rate",
                   fontsize=FS, color=FC)
    ax_b.legend(fontsize=FS - 4, frameon=False, loc="lower right")
    ax_b.grid(True, which="both", alpha=0.25)
    _panel_letter(ax_b, "b")
    _tint(ax_b)

    ax_c = fig.add_subplot(gs[0, 2])
    data_c = panel_data["c"]
    sizes = data_c["sizes"]
    ok = data_c["eta_boundary"] == data_c["eta_face_product"]
    ax_c.scatter(sizes[ok], data_c["eta_boundary"][ok],
                 s=18, color=GREEN, alpha=0.7,
                 label=f"agree ({int(np.sum(ok))}/{len(ok)})")
    if np.any(~ok):
        ax_c.scatter(sizes[~ok], data_c["eta_boundary"][~ok],
                     s=22, color=RED, alpha=0.95, marker="x",
                     label=f"disagree ({int(np.sum(~ok))})")
    ax_c.set_xlabel("plaquettes in $D$", fontsize=FS, color=FC)
    ax_c.set_ylabel(r"$\eta(\partial D)$", fontsize=FS, color=FC)
    ax_c.set_yticks([-1, 0, +1])
    ax_c.set_title(r"$\eta(\partial D)=\prod_c\eta(c)$ on random $D$",
                   fontsize=FS, color=FC)
    ax_c.legend(fontsize=FS - 3, frameon=False, loc="lower right")
    ax_c.grid(True, alpha=0.25)
    _panel_letter(ax_c, "c")
    _tint(ax_c)

    ax_d = fig.add_subplot(gs[1, 0])
    data_d = panel_data["d"]
    etas = data_d["etas"]
    cmap = plt.matplotlib.colors.ListedColormap(
        [PURPLE, "#ECECEC", ORANGE])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    im = ax_d.imshow(etas.T, origin="lower", cmap=cmap, norm=norm,
                     extent=[0, etas.shape[0], 0, etas.shape[1]])
    ax_d.set_xlabel("plaquette column", fontsize=FS, color=FC)
    ax_d.set_ylabel("plaquette row", fontsize=FS, color=FC)
    ax_d.set_title(
        fr"$\eta(c)$ on $30\times30$ periodic torus;  "
        fr"$\prod_c\eta(c)={data_d['prod']:+d}$",
        fontsize=FS, color=FC)
    cbar = fig.colorbar(im, ax=ax_d, ticks=[-1, 0, 1], fraction=0.045, pad=0.04)
    cbar.ax.set_yticklabels([r"$-1$", r"$0$", r"$+1$"])
    tint_colorbar(cbar)
    _panel_letter(ax_d, "d")
    _tint(ax_d)

    ax_e = fig.add_subplot(gs[1, 1:])
    ax_e.axis("off")
    rows = panel_data["e"]["rows"]
    cell_text = [[r["name"],
                  _fmt_sign(r["eta_alpha"]),
                  _fmt_sign(r["eta_beta"]),
                  r["det_alpha_fmt"],
                  r["det_beta_fmt"],
                  r["note"]] for r in rows]
    col_labels = ["ensemble",
                  r"$\eta_\alpha$",
                  r"$\eta_\beta$",
                  r"$\det H_{\gamma_\alpha}$",
                  r"$\det H_{\gamma_\beta}$",
                  "construction"]
    table = ax_e.table(cellText=cell_text, colLabels=col_labels,
                       cellLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(FS - 4)
    table.scale(1.0, 2.0)
    ax_e.set_title(r"topological signature $(\eta_\alpha, \eta_\beta)$"
                   r" across four ensembles (Corollary on T$^2$)",
                   fontsize=FS, color=FC, pad=22)
    _panel_letter(ax_e, "e")

    fig.savefig(out_path, format="pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)

def _fmt_sign(e):
    if e == 0:
        return r"$0$"
    return r"$+1$" if e > 0 else r"$-1$"

def load_sweep_npz(path):
    d = np.load(path, allow_pickle=True)
    orig = d["original"].item()
    sweeps = [s for s in d["gap_sweeps"]]
    out = {
        "epsilons": np.asarray(orig["epsilons"]),
        "original_frobenius": np.asarray(orig["mean_frobenius"]),
        "gap_sweeps": [{"tau": float(s["tau"]),
                         "epsilons": np.asarray(s["epsilons"]),
                         "frobenius": np.asarray(s["mean_frobenius"])}
                        for s in sweeps],
    }
    return out

def main(R_param=1.0, r_param=1.0, n_samples=1000, n_eval=30,
         n_models=10, epochs=200, seed0=0,
         epsilon_A=0.5,
         n_rects=500, rng_seed=42,
         fig_path=None, json_path=None):

    out_dir = Path(__file__).resolve().parent / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    if fig_path is None:
        fig_path = str(out_dir / "topological_signature.pdf")
    if json_path is None:
        json_path = str(out_dir / "topological_signature.json")

    print("=" * 72)
    print("Phase 3 - Closing figure: topological signature of learning fields")
    print("=" * 72)
    print(f"  N={n_models}, grid={n_eval}x{n_eval}, seed0={seed0}")

    predictions, jacobians, E_all, s_y, s_g, u_eval, v_eval = prepare_ensemble(
        R_param=R_param, r_param=r_param, n_samples=n_samples, n_eval=n_eval,
        n_models=n_models, epochs=epochs, seed0=seed0)

    print("[a] gap distribution on all non-wrap edges...")
    t0 = time.time()
    gaps, _edges = compute_gap_distribution(predictions, jacobians, E_all,
                                             n_eval, s_y, s_g)
    print(f"    {len(gaps)} edges; min={gaps.min():.2e}, "
          f"median={np.median(gaps):.2e}, max={gaps.max():.2e} "
          f"(wall {time.time()-t0:.1f}s)")

    print("[b] loading Phase 0 gap-enforced sweep...")
    b_data = load_sweep_npz(out_dir / "gap_enforced_sweep.npz")

    print("[tc] building Hungarian transport caches...")
    t0 = time.time()
    cache_open = build_edge_transport_cache(
        predictions, jacobians, E_all, n_eval, s_y, s_g,
        periodic=False, mode="hungarian")
    cache_periodic = build_edge_transport_cache(
        predictions, jacobians, E_all, n_eval, s_y, s_g,
        periodic=True, mode="hungarian")
    print(f"    open={len(cache_open)} edges; "
          f"periodic={len(cache_periodic)} edges "
          f"(wall {time.time()-t0:.1f}s)")

    print(f"[c] Abelian Stokes on {n_rects} random rectangles (regime C)...")
    rng = np.random.default_rng(rng_seed)
    rects = random_rectangles(n_eval, n_rects, rng)
    etaB, etaF, detB, detFprod, sizes = check_abelian_stokes(
        cache_open, n_eval, rects)
    agree = int(np.sum(etaB == etaF))
    print(f"    sign agreement eta(partial D)=prod eta(c): {agree}/{n_rects}")

    print("[d] Closed-surface sum rule on the 30x30 periodic torus (regime C)...")
    etas_d, dets_d, prod_d, n_zero_d = closed_torus_sum_rule(
        cache_periodic, n_eval)
    print(f"    prod_c eta(c) = {prod_d:+d}, "
          f"n_plaq={etas_d.size}, zeros (det underflow)={n_zero_d}")

    print("[e] Topological signature across four ensembles (regime C)...")

    (ea, eb), (da, db) = measure_signature(cache_periodic, n_eval,
                                            twist=None)
    print(f"    canonical     : (eta_a, eta_b) = ({ea:+d}, {eb:+d})  "
          f"det_a={da:+.3f}  det_b={db:+.3f}")

    (ea_bz, eb_bz), (da_bz, db_bz) = bias_tag_zero_signature()
    print(f"    bias-tag zero : (eta_a, eta_b) = ({ea_bz:+d}, {eb_bz:+d}) "
          f" (by construction, T=I)")

    sigma_perm = np.arange(n_models); sigma_perm[0], sigma_perm[1] = 1, 0
    sigma_M = permutation_matrix(sigma_perm, n_models)
    (ea_ta, eb_ta), (da_ta, db_ta) = measure_signature(
        cache_periodic, n_eval, twist='alpha', sigma=sigma_M)
    print(f"    twisted alpha : (eta_a, eta_b) = ({ea_ta:+d}, {eb_ta:+d})  "
          f"det_a={da_ta:+.3f}  det_b={db_ta:+.3f}")

    (ea_tb, eb_tb), (da_tb, db_tb) = measure_signature(
        cache_periodic, n_eval, twist='beta', sigma=sigma_M)
    print(f"    twisted beta  : (eta_a, eta_b) = ({ea_tb:+d}, {eb_tb:+d})  "
          f"det_a={da_tb:+.3f}  det_b={db_tb:+.3f}")

    panel_data = {
        "a": {"gaps": gaps},
        "b": b_data,
        "c": {"eta_boundary": etaB, "eta_face_product": etaF,
              "det_boundary": detB, "det_face_prod": detFprod,
              "sizes": sizes, "rects": rects},
        "d": {"etas": etas_d, "dets": dets_d, "prod": prod_d,
              "n_zero": n_zero_d},
        "e": {"rows": [
            {"name": "canonical MLP",
             "eta_alpha": ea, "eta_beta": eb,
             "det_alpha_fmt": f"{da:+.3f}",
             "det_beta_fmt": f"{db:+.3f}",
             "note": "Hungarian on trained ensemble"},
            {"name": "bias-tag zero",
             "eta_alpha": ea_bz, "eta_beta": eb_bz,
             "det_alpha_fmt": f"{da_bz:+.3f}",
             "det_beta_fmt": f"{db_bz:+.3f}",
             "note": r"$T_e=I$ for all $e$"},
            {"name": r"twisted-$\alpha$",
             "eta_alpha": ea_ta, "eta_beta": eb_ta,
             "det_alpha_fmt": f"{da_ta:+.3f}",
             "det_beta_fmt": f"{db_ta:+.3f}",
             "note": r"$\sigma=(0\,1)$ on $\alpha$-seam"},
            {"name": r"twisted-$\beta$",
             "eta_alpha": ea_tb, "eta_beta": eb_tb,
             "det_alpha_fmt": f"{da_tb:+.3f}",
             "det_beta_fmt": f"{db_tb:+.3f}",
             "note": r"$\sigma=(0\,1)$ on $\beta$-seam"},
        ]},
    }

    print(f"Writing figure -> {fig_path}")
    render_figure(fig_path, panel_data)

    summary = {
        "metadata": {
            "N": n_models, "n_eval": n_eval, "seed0": seed0,
            "epsilon_regime_A": epsilon_A,
            "rng_seed_rects": rng_seed,
            "n_random_rectangles": n_rects,
        },
        "panel_a": {
            "n_edges": int(len(gaps)),
            "gap_stats": {
                "min": float(gaps.min()),
                "median": float(np.median(gaps)),
                "max": float(gaps.max()),
                "p05": float(np.percentile(gaps, 5)),
                "p95": float(np.percentile(gaps, 95)),
            },
        },
        "panel_c": {
            "n_rectangles": int(len(rects)),
            "n_agree_sign": int(np.sum(etaB == etaF)),
            "n_disagree_sign": int(np.sum(etaB != etaF)),
            "note": "regime-C Hungarian; abelian Stokes is an exact identity",
        },
        "panel_d": {
            "n_plaquettes": int(etas_d.size),
            "prod_eta": int(prod_d),
            "n_positive": int(np.sum(etas_d == +1)),
            "n_negative": int(np.sum(etas_d == -1)),
            "n_zero": int(np.sum(etas_d == 0)),
            "note": "regime-C Hungarian; closed-torus sum rule expected +1",
        },
        "panel_e": {
            "canonical":        {"eta_alpha": ea,    "eta_beta": eb},
            "bias_tag_zero":    {"eta_alpha": ea_bz, "eta_beta": eb_bz},
            "twisted_alpha":    {"eta_alpha": ea_ta, "eta_beta": eb_ta},
            "twisted_beta":     {"eta_alpha": ea_tb, "eta_beta": eb_tb},
            "sigma": "(0 1) transposition, det = -1",
        },
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Writing summary -> {json_path}")

    assert prod_d == +1, "closed-surface sum rule failed in regime C"
    assert agree == n_rects, (
        f"abelian Stokes failed on {n_rects - agree} of {n_rects} rectangles")
    assert ea_ta == -ea and eb_ta == eb, (
        "twisted-alpha did not flip eta_alpha (and only eta_alpha)")
    assert ea_tb == ea and eb_tb == -eb, (
        "twisted-beta did not flip eta_beta (and only eta_beta)")
    assert ea_bz == +1 and eb_bz == +1
    print("\nAll Phase 3 consistency checks passed.\n")

if __name__ == "__main__":
    main()
