"""J. Fuentes Aguilar, 2025-2026."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from core import (
    evaluate_ensemble,
    fibre_jets,
    ground_truth_field,
    jet_costm,
    rotation_AB,
    torus_tangent,
    train_ensemble,
    transfer_operator_entropic,
)

def edge_transport(i_A, j_A, i_B, j_B,
                   predictions, jacobians, E_all, n_eval,
                   s_y, s_g, eps):
    A = fibre_jets(i_A, j_A, predictions, jacobians, n_eval)
    B = fibre_jets(i_B, j_B, predictions, jacobians, n_eval)
    R_AB = rotation_AB(i_A, j_A, i_B, j_B, E_all, n_eval)
    C = jet_costm(A, B, s_y, s_g, R_AB)
    return transfer_operator_entropic(C, eps=eps)

def plaquette_holonomy_matrix(i0, j0, predictions, jacobians, E_all, n_eval,
                              s_y, s_g, eps):
    path = [
        (i0,       j0),
        (i0,       j0 + 1),
        (i0 + 1,   j0 + 1),
        (i0 + 1,   j0),
        (i0,       j0),
    ]
    Ts = []
    for (ia, ja), (ib, jb) in zip(path[:-1], path[1:]):
        Ts.append(edge_transport(ia, ja, ib, jb,
                                 predictions, jacobians, E_all, n_eval,
                                 s_y, s_g, eps))
    H = Ts[-1].T @ Ts[-2].T @ Ts[-3].T @ Ts[-4].T
    return H

def noncontractible_loop_matrix(axis, fixed_index, n_eval,
                                predictions, jacobians, E_all,
                                s_y, s_g, eps):
    assert axis in ('u', 'v')

    path = []
    if axis == 'u':

        for k in range(n_eval):
            path.append((fixed_index, k))
        path.append((fixed_index, 0))
    else:

        for k in range(n_eval):
            path.append((k, fixed_index))
        path.append((0, fixed_index))

    Ts = []
    for (ia, ja), (ib, jb) in zip(path[:-1], path[1:]):
        Ts.append(edge_transport(ia, ja, ib, jb,
                                 predictions, jacobians, E_all, n_eval,
                                 s_y, s_g, eps))

    N = Ts[0].shape[0]
    H = np.eye(N)
    for T in Ts:
        H = T.T @ H
    return H

def character_invariants(H):
    N = H.shape[0]
    eigs = np.linalg.eigvals(H)

    ks = np.arange(1, N)

    powers = eigs[None, :] ** ks[:, None]
    p = powers.sum(axis=1) / N
    det_val = float(np.real(np.prod(eigs)))

    if abs(det_val) < 1e-14:
        eta = 0
    else:
        eta = int(np.sign(det_val))
    I = np.eye(N)
    h_op = float(np.linalg.norm(H - I, ord='fro') / N)
    return {
        'p': p,
        'eta': eta,
        'h_op': h_op,
        'det': det_val,
        'eigs': eigs,
    }

def run_character_spectrum(
    R_param: float = 1.0,
    r_param: float = 1.0,
    n_samples: int = 1000,
    n_eval: int = 30,
    n_models: int = 10,
    seed0: int = 0,
    epsilon: float = 0.5,
    save_figs: bool = True,
    fig_path: str | None = None,
    json_path: str | None = None,
    npz_path: str | None = None,
):
    out_dir = Path(__file__).resolve().parent / 'figs'
    out_dir.mkdir(parents=True, exist_ok=True)
    if fig_path is None:
        fig_path = str(out_dir / 'character_spectrum.pdf')
    if json_path is None:
        json_path = str(out_dir / 'character_spectrum.json')
    if npz_path is None:
        npz_path = str(out_dir / 'character_spectrum.npz')

    print('=' * 70)
    print('Phase 1: Character spectrum of the holonomy representation')
    print('=' * 70)
    print(f'  N = {n_models}, grid = {n_eval} x {n_eval}, eps = {epsilon}')

    from pathlib import Path as _P
    cache_ensemble = (_P(__file__).resolve().parent
                      / f"ensemble_cache_seed{seed0}_N{n_models}_nev{n_eval}.npz")
    if cache_ensemble.exists():
        print(f'  Loading cached ensemble evaluation from {cache_ensemble.name}')
        d = np.load(cache_ensemble)
        predictions = d['predictions']
        jacobians = d['jacobians']
        E_all = d['E_all']
        u_eval = d['u_eval']
        v_eval = d['v_eval']
        s_y = float(predictions.std() + 1e-12)
        s_g = float(jacobians.std() + 1e-12)
    else:
        torch.manual_seed(seed0)
        np.random.seed(seed0)
        u_samples = np.random.uniform(0, 2 * np.pi, n_samples)
        v_samples = np.random.uniform(0, 2 * np.pi, n_samples)
        targets = ground_truth_field(u_samples, v_samples)
        print('  Training ensemble...')
        models = train_ensemble(
            u_samples, v_samples, targets,
            R_param=R_param, r_param=r_param,
            n_models=n_models, epochs=200, seed0=seed0,
        )
        print('  Evaluating ensemble on the 30 x 30 parameter grid...')
        u_eval = np.linspace(0, 2 * np.pi, n_eval)
        v_eval = np.linspace(0, 2 * np.pi, n_eval)
        u_grid, v_grid = np.meshgrid(u_eval, v_eval)
        predictions, jacobians = evaluate_ensemble(
            models, u_grid, v_grid, R_param=R_param, r_param=r_param,
        )
        u_flat = u_grid.ravel()
        v_flat = v_grid.ravel()
        E_all = torus_tangent(u_flat, v_flat, R_param=R_param, r_param=r_param)
        s_y = predictions.std() + 1e-12
        s_g = jacobians.std() + 1e-12

    n_plaq = n_eval - 1
    N = n_models

    print(f'  Computing {n_plaq * n_plaq} per-plaquette holonomy matrices...')
    H_plaq = np.zeros((n_plaq, n_plaq, N, N), dtype=np.float64)
    p_plaq = np.zeros((n_plaq, n_plaq, N - 1), dtype=np.complex128)
    eta_plaq = np.zeros((n_plaq, n_plaq), dtype=np.int64)
    hop_plaq = np.zeros((n_plaq, n_plaq), dtype=np.float64)
    det_plaq = np.zeros((n_plaq, n_plaq), dtype=np.float64)

    for i in range(n_plaq):
        for j in range(n_plaq):
            H = plaquette_holonomy_matrix(
                i, j, predictions, jacobians, E_all, n_eval,
                s_y, s_g, epsilon,
            )
            H_plaq[i, j] = H
            inv = character_invariants(H)
            p_plaq[i, j] = inv['p']
            eta_plaq[i, j] = inv['eta']
            hop_plaq[i, j] = inv['h_op']
            det_plaq[i, j] = inv['det']

    print(f'    h_op: mean={hop_plaq.mean():.3e}, max={hop_plaq.max():.3e}')
    n_plus = int(np.sum(eta_plaq == +1))
    n_minus = int(np.sum(eta_plaq == -1))
    n_zero = int(np.sum(eta_plaq == 0))
    print(f'    eta distribution: +1: {n_plus}, -1: {n_minus}, 0: {n_zero}')

    print('  Computing non-contractible loop holonomies (30 edges each)...')

    H_alpha = noncontractible_loop_matrix(
        axis='u', fixed_index=0, n_eval=n_eval,
        predictions=predictions, jacobians=jacobians, E_all=E_all,
        s_y=s_y, s_g=s_g, eps=epsilon,
    )
    H_beta = noncontractible_loop_matrix(
        axis='v', fixed_index=0, n_eval=n_eval,
        predictions=predictions, jacobians=jacobians, E_all=E_all,
        s_y=s_y, s_g=s_g, eps=epsilon,
    )
    inv_alpha = character_invariants(H_alpha)
    inv_beta = character_invariants(H_beta)

    print(f'    gamma_alpha: eta = {inv_alpha["eta"]:+d}, '
          f'h_op = {inv_alpha["h_op"]:.3e}, '
          f'Re p_1 = {inv_alpha["p"][0].real:+.4f}')
    print(f'    gamma_beta:  eta = {inv_beta["eta"]:+d}, '
          f'h_op = {inv_beta["h_op"]:.3e}, '
          f'Re p_1 = {inv_beta["p"][0].real:+.4f}')

    p_flat = p_plaq.reshape(-1, N - 1)
    pk_stats = []
    for k in range(N - 1):
        col = p_flat[:, k]
        pk_stats.append({
            'k': k + 1,
            'Re_mean': float(col.real.mean()),
            'Re_std': float(col.real.std()),
            'Re_min': float(col.real.min()),
            'Re_max': float(col.real.max()),
            'Im_mean': float(col.imag.mean()),
            'Im_std': float(col.imag.std()),
            'abs_mean': float(np.abs(col).mean()),
            'abs_max': float(np.abs(col).max()),
        })

    re_p1 = p_flat[:, 0].real
    corr_p1_pk = []
    for k in range(N - 1):
        re_pk = p_flat[:, k].real
        if re_pk.std() < 1e-14 or re_p1.std() < 1e-14:
            corr = float('nan')
        else:
            corr = float(np.corrcoef(re_p1, re_pk)[0, 1])
        corr_p1_pk.append(corr)

    summary = {
        'metadata': {
            'N': n_models,
            'n_eval': n_eval,
            'n_plaquettes': int(n_plaq * n_plaq),
            'epsilon': epsilon,
            'seed0': seed0,
            'regime': 'A (entropic, Sinkhorn)',
            'basepoint': 'x_0 = (u_0, v_0) = (0, 0)',
        },
        'noncontractible_signature': {
            'eta_alpha': int(inv_alpha['eta']),
            'eta_beta': int(inv_beta['eta']),
            'p_alpha_real': [float(x) for x in inv_alpha['p'].real],
            'p_alpha_imag': [float(x) for x in inv_alpha['p'].imag],
            'p_beta_real': [float(x) for x in inv_beta['p'].real],
            'p_beta_imag': [float(x) for x in inv_beta['p'].imag],
            'h_op_alpha': float(inv_alpha['h_op']),
            'h_op_beta': float(inv_beta['h_op']),
            'det_alpha': float(inv_alpha['det']),
            'det_beta': float(inv_beta['det']),
        },
        'plaquette_pk_stats': pk_stats,
        'plaquette_eta_distribution': {
            '+1': n_plus,
            '-1': n_minus,
            '0': n_zero,
        },
        'plaquette_hop': {
            'mean': float(hop_plaq.mean()),
            'std': float(hop_plaq.std()),
            'min': float(hop_plaq.min()),
            'max': float(hop_plaq.max()),
            'median': float(np.median(hop_plaq)),
        },
        'plaquette_re_p1_vs_re_pk_corr': {
            f'p{k+1}': corr_p1_pk[k] for k in range(N - 1)
        },
    }

    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  Wrote summary -> {json_path}')

    np.savez_compressed(
        npz_path,
        H_plaq=H_plaq,
        p_plaq=p_plaq,
        eta_plaq=eta_plaq,
        hop_plaq=hop_plaq,
        det_plaq=det_plaq,
        H_alpha=H_alpha,
        H_beta=H_beta,
        p_alpha=inv_alpha['p'],
        p_beta=inv_beta['p'],
        eta_alpha=inv_alpha['eta'],
        eta_beta=inv_beta['eta'],
        hop_alpha=inv_alpha['h_op'],
        hop_beta=inv_beta['h_op'],
        u_eval=u_eval,
        v_eval=v_eval,
    )
    print(f'  Wrote raw arrays -> {npz_path}')

    print('  Rendering 4-panel figure...')
    from style import (
        PURPLE, ORANGE, GREEN, RED, UI_COLOR, FC, FS, tint, panel_label,
        tint_colorbar,
    )

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])

    ax_a = fig.add_subplot(gs[0, 0])
    re_p2_map = p_plaq[:, :, 1].real

    im_a = ax_a.contourf(
        u_eval[:-1], v_eval[:-1], re_p2_map.T, levels=20, cmap='Pastel2',
    )
    ax_a.set_xlabel('$u$', fontsize=FS)
    ax_a.set_ylabel('$v$', fontsize=FS)
    ax_a.set_title('spatial field of $\\mathrm{Re}\\,p_2(\\gamma)$ on plaquettes',
                   fontsize=FS, color=FC)
    ax_a.set_aspect('equal')
    cbar_a = fig.colorbar(im_a, ax=ax_a, shrink=0.88)
    cbar_a.set_label('$\\mathrm{Re}\\,p_2$', fontsize=FS - 2, color=FC)
    tint_colorbar(cbar_a)
    panel_label(ax_a, 'a')
    tint(ax_a)

    ax_b = fig.add_subplot(gs[0, 1])
    re_p1_flat = p_flat[:, 0].real
    h_op_flat = hop_plaq.ravel()

    regC_pred = np.sqrt(np.clip(2.0 * (1.0 - re_p1_flat) / N, 0.0, None))
    ax_b.scatter(regC_pred, h_op_flat, s=14, alpha=0.55, color=PURPLE,
                 edgecolors='none')

    lim = max(h_op_flat.max(), regC_pred.max()) * 1.05 + 1e-6
    ax_b.plot([0, lim], [0, lim], '--', color=UI_COLOR, alpha=0.55,
              label='regime (C): $h_{op} = \\sqrt{2(1-\\mathrm{Re}\\,p_1)/N}$')
    ax_b.set_xlim(0, lim)
    ax_b.set_ylim(0, lim)
    ax_b.set_xlabel('$\\sqrt{2(1-\\mathrm{Re}\\,p_1)/N}$', fontsize=FS)
    ax_b.set_ylabel('$h_{\\mathrm{op}}$', fontsize=FS)
    ax_b.set_title('$h_{\\mathrm{op}}$ versus the regime-(C) identity',
                   fontsize=FS, color=FC)
    ax_b.legend(fontsize=FS - 3, loc='lower right', frameon=False)
    ax_b.grid(True, alpha=0.25)
    ax_b.set_aspect('equal')
    panel_label(ax_b, 'b')
    tint(ax_b)

    ax_c = fig.add_subplot(gs[1, 0])
    labels = ['$\\eta = -1$', '$\\eta = 0$', '$\\eta = +1$']
    counts = [n_minus, n_zero, n_plus]
    colors = [PURPLE, UI_COLOR, ORANGE]
    bars = ax_c.bar(labels, counts, color=colors, alpha=0.85,
                    edgecolor=UI_COLOR, linewidth=0.6)
    for b, c in zip(bars, counts):
        ax_c.text(b.get_x() + b.get_width() / 2.0, b.get_height(),
                  f' {c}', ha='center', va='bottom', fontsize=FS, color=FC)
    ax_c.set_ylabel('count (plaquettes)', fontsize=FS)
    total = n_plus + n_minus + n_zero
    ax_c.set_title(f'abelian invariant $\\eta$ across {total} plaquettes',
                   fontsize=FS, color=FC)
    ax_c.set_ylim(0, max(counts) * 1.12 + 1)
    ax_c.grid(True, alpha=0.25, axis='y')
    panel_label(ax_c, 'c')
    tint(ax_c)

    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis('off')
    panel_label(ax_d, 'd')
    ax_d.set_title(
        'non-contractible loop invariants (canonical ensemble)',
        fontsize=FS, color=FC,
    )

    header = (['loop', '$\\eta$', '$h_{\\mathrm{op}}$']
              + [f'Re $p_{{{k+1}}}$' for k in range(N - 1)])
    row_alpha = (['$\\gamma_\\alpha$',
                  f'{inv_alpha["eta"]:+d}',
                  f'{inv_alpha["h_op"]:.2e}']
                 + [f'{inv_alpha["p"][k].real:+.3f}' for k in range(N - 1)])
    row_beta = (['$\\gamma_\\beta$',
                 f'{inv_beta["eta"]:+d}',
                 f'{inv_beta["h_op"]:.2e}']
                + [f'{inv_beta["p"][k].real:+.3f}' for k in range(N - 1)])

    cell_text = [row_alpha, row_beta]
    table = ax_d.table(
        cellText=cell_text,
        colLabels=header,
        cellLoc='center',
        colLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)

    max_im_alpha = float(np.max(np.abs(inv_alpha['p'].imag)))
    max_im_beta = float(np.max(np.abs(inv_beta['p'].imag)))
    ax_d.text(
        0.5, 0.04,
        f'max |Im $p_k$|: $\\gamma_\\alpha$ = {max_im_alpha:.2e},   '
        f'$\\gamma_\\beta$ = {max_im_beta:.2e}',
        ha='center', va='center', transform=ax_d.transAxes,
        fontsize=FS - 2, color=FC,
    )

    if save_figs:
        fig.savefig(fig_path, format='pdf', dpi=150, bbox_inches='tight')
        print(f'  Wrote figure -> {fig_path}')

    plt.close(fig)

    print()
    print('Summary:')
    print(f'  topological signature (eta_alpha, eta_beta) = '
          f'({inv_alpha["eta"]:+d}, {inv_beta["eta"]:+d})')
    print('  correlation(Re p_1, Re p_k) across plaquettes:')
    for k in range(N - 1):
        print(f'    p_{k+1}: {corr_p1_pk[k]:+.4f}')

    return {
        'summary': summary,
        'hop_plaq': hop_plaq,
        'p_plaq': p_plaq,
        'eta_plaq': eta_plaq,
        'H_plaq': H_plaq,
        'H_alpha': H_alpha,
        'H_beta': H_beta,
        'inv_alpha': inv_alpha,
        'inv_beta': inv_beta,
    }

if __name__ == '__main__':
    run_character_spectrum(
        R_param=1.0,
        r_param=1.0,
        n_samples=1000,
        n_eval=30,
        n_models=10,
        seed0=0,
        epsilon=0.5,
        save_figs=True,
    )
