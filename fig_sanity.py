"""J. Fuentes Aguilar, 2025-2026."""

import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import spearmanr

from style import (
    PURPLE, ORANGE, UI_COLOR, FC,
    FS, FS_TICK, FIG_WIDTH,
    tint, panel_label, tint_colorbar,
)

from core import (
    torus_parametrisation,
    torus_tangent,
    ground_truth_field,
    train_ensemble,
    evaluate_ensemble,
    fibre_jets,
    rotation_AB,
    jet_costm,
    transfer_operator_entropic,
    transfer_operator_hungarian,
    holonomy_field,
    randomly_permute_ensemble,
    permutation_distance_entropy,
    permutation_distance_frobenius,
)

CB_FRACTION = 0.045
CB_PAD = 0.02

def _thin_ticks(ax, n=5, which='both'):
    if which in ('both', 'x') and ax.get_xscale() == 'linear':
        ax.xaxis.set_major_locator(MaxNLocator(nbins=n, min_n_ticks=3))
    if which in ('both', 'y') and ax.get_yscale() == 'linear':
        ax.yaxis.set_major_locator(MaxNLocator(nbins=n, min_n_ticks=3))

def prepare_data(R_param=1.0, r_param=1.0, n_samples=1000, n_eval=30, n_models=10, seed0=0, epochs=200):
    cache_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f'ensemble_cache_seed{seed0}_N{n_models}_nev{n_eval}.npz',
    )
    if os.path.exists(cache_file):
        print(f'Loading cached ensemble evaluation from {os.path.basename(cache_file)}')
        d = np.load(cache_file)
        predictions = d['predictions']
        jacobians = d['jacobians']
        E_all = d['E_all']
        u_eval = d['u_eval']
        v_eval = d['v_eval']
        u_grid, v_grid = np.meshgrid(u_eval, v_eval)
        s_y = float(predictions.std() + 1e-12)
        s_g = float(jacobians.std() + 1e-12)
        return (predictions, jacobians, E_all, n_eval, s_y, s_g,
                u_eval, v_eval, u_grid, v_grid)

    torch.manual_seed(seed0)
    np.random.seed(seed0)

    u_samples = np.random.uniform(0, 2*np.pi, n_samples)
    v_samples = np.random.uniform(0, 2*np.pi, n_samples)
    targets   = ground_truth_field(u_samples, v_samples)

    print('Training ensemble …')
    models = train_ensemble(
        u_samples, v_samples, targets,
        R_param=R_param, r_param=r_param,
        n_models=n_models, epochs=epochs, seed0=seed0)

    u_eval = np.linspace(0, 2*np.pi, n_eval)
    v_eval = np.linspace(0, 2*np.pi, n_eval)
    u_grid, v_grid = np.meshgrid(u_eval, v_eval)

    print('Evaluating ensemble …')
    predictions, jacobians = evaluate_ensemble(
        models, u_grid, v_grid, R_param=R_param, r_param=r_param)

    u_flat = u_grid.ravel()
    v_flat = v_grid.ravel()
    E_all  = torus_tangent(u_flat, v_flat, R_param=R_param, r_param=r_param)
    s_y    = predictions.std() + 1e-12
    s_g    = jacobians.std()   + 1e-12

    return (predictions, jacobians, E_all, n_eval, s_y, s_g,
            u_eval, v_eval, u_grid, v_grid)

def task1_data(predictions, jacobians, E_all, n_eval, s_y, s_g, eps=0.5, side=1):
    print('Computing entropic holonomy field …')
    hol_ent = holonomy_field(predictions, jacobians, E_all, n_eval,
                             s_y, s_g, eps=eps, side=side, mode='entropic')
    print('Computing Hungarian holonomy field …')
    hol_hun = holonomy_field(predictions, jacobians, E_all, n_eval,
                             s_y, s_g, eps=eps, side=side, mode='hungarian')
    return hol_ent, hol_hun

def task2_data(predictions, jacobians, E_all, n_eval, s_y, s_g, epsilons=None, n_edges=50, seed0=0):
    if epsilons is None:
        epsilons = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

    all_edges = []
    for i in range(n_eval - 1):
        for j in range(n_eval - 1):
            all_edges.append(((i, j), (i, j + 1)))
            all_edges.append(((i, j), (i + 1, j)))

    rng = np.random.default_rng(seed0 + 100)
    chosen = rng.choice(len(all_edges), size=min(n_edges, len(all_edges)),
                        replace=False)
    edges = [all_edges[k] for k in chosen]

    mean_ent = np.zeros(len(epsilons))
    mean_fro = np.zeros(len(epsilons))

    print(f'Epsilon sweep over {len(edges)} edges …')
    for idx, ((iA, jA), (iB, jB)) in enumerate(edges):
        if idx % 10 == 0:
            print(f'  edge {idx+1}/{len(edges)}')
        A   = fibre_jets(iA, jA, predictions, jacobians, n_eval)
        B   = fibre_jets(iB, jB, predictions, jacobians, n_eval)
        R   = rotation_AB(iA, jA, iB, jB, E_all, n_eval)
        C   = jet_costm(A, B, s_y, s_g, R)
        T_h = transfer_operator_hungarian(C)

        for ei, eps in enumerate(epsilons):
            T_e = transfer_operator_entropic(C, eps=eps)
            mean_ent[ei] += permutation_distance_entropy(T_e)
            mean_fro[ei] += permutation_distance_frobenius(T_e, T_h)

    mean_ent /= len(edges)
    mean_fro /= len(edges)
    return epsilons, mean_ent, mean_fro

def task3_data(predictions, jacobians, E_all, n_eval, s_y, s_g, eps=0.5, side=1, n_trials=5):
    print('Computing baseline holonomy for gauge test …')
    hol_base = holonomy_field(predictions, jacobians, E_all, n_eval, s_y, s_g, eps=eps, side=side, mode='entropic')

    hol_perm_list = []
    for t in range(n_trials):
        rng = np.random.default_rng(42 + t)
        p_perm, j_perm = randomly_permute_ensemble(predictions, jacobians, rng=rng)
        h = holonomy_field(p_perm, j_perm, E_all, n_eval, s_y, s_g, eps=eps, side=side, mode='entropic')
        hol_perm_list.append(h)
        l2 = np.sqrt(np.mean((h - hol_base)**2))
        print(f'  trial {t+1}/{n_trials}: L2 = {l2:.3e}')

    return hol_base, hol_perm_list

def make_figure(hol_ent, hol_hun,
                epsilons, mean_ent, mean_fro,
                hol_base, hol_perm_list,
                u_eval, v_eval,
                outpath='figs/sanity_checks.pdf'):

    mosaic = [
        ['a', 'b', 'c'],
        ['d', 'd', 'e'],
    ]

    fig, axd = plt.subplot_mosaic(
        mosaic,
        figsize=(FIG_WIDTH, FIG_WIDTH * 0.62),
        constrained_layout=True,
    )
    ax_a, ax_b, ax_c = axd['a'], axd['b'], axd['c']
    ax_d, ax_e = axd['d'], axd['e']

    for _ax in (ax_a, ax_b, ax_c, ax_e):
        _ax.set_box_aspect(1.0)
    ax_d.set_box_aspect(0.5)

    vmin = min(hol_ent.min(), hol_hun.min())
    vmax = max(hol_ent.max(), hol_hun.max())
    extent = [u_eval[0], u_eval[-2], v_eval[0], v_eval[-2]]

    def _attach_cbar(ax, mappable, label):

        cax = ax.inset_axes([1.04, 0.0, 0.05, 1.0])
        cb = fig.colorbar(mappable, cax=cax)
        cb.set_label(label, fontsize=FS_TICK, color=FC)
        tint_colorbar(cb)
        return cb

    im_a = ax_a.imshow(hol_ent, origin='lower', aspect='equal',
                       cmap='Pastel2', vmin=vmin, vmax=vmax, extent=extent)
    ax_a.set_xlabel(r'$u$', fontsize=FS)
    ax_a.set_ylabel(r'$v$', fontsize=FS)
    ax_a.set_title('Entropic holonomy', fontsize=FS, color=FC)
    _attach_cbar(ax_a, im_a, r'$h_{\mathrm{op}}$')
    panel_label(ax_a, 'a')
    tint(ax_a)
    _thin_ticks(ax_a, n=4)

    im_b = ax_b.imshow(hol_hun, origin='lower', aspect='equal',
                       cmap='Pastel2', vmin=vmin, vmax=vmax, extent=extent)
    ax_b.set_xlabel(r'$u$', fontsize=FS)
    ax_b.set_ylabel(r'$v$', fontsize=FS)
    ax_b.set_title('Hungarian holonomy', fontsize=FS, color=FC)
    _attach_cbar(ax_b, im_b, r'$h_{\mathrm{op}}$')
    panel_label(ax_b, 'b')
    tint(ax_b)
    _thin_ticks(ax_b, n=4)

    diff = hol_ent - hol_hun
    vlim = max(abs(diff.min()), abs(diff.max()))
    im_c = ax_c.imshow(diff, origin='lower', aspect='equal',
                       cmap='RdBu_r', vmin=-vlim, vmax=vlim, extent=extent)
    ax_c.set_xlabel(r'$u$', fontsize=FS)
    ax_c.set_ylabel(r'$v$', fontsize=FS)
    ax_c.set_title('Difference', fontsize=FS, color=FC)
    _attach_cbar(ax_c, im_c, r'$\Delta h$')
    panel_label(ax_c, 'c')
    tint(ax_c)
    _thin_ticks(ax_c, n=4)

    ax_d.loglog(epsilons, mean_ent, 'o-', linewidth=2.2, markersize=7,
                color=PURPLE, label='row entropy')
    ax_d.loglog(epsilons, mean_fro, 's-', linewidth=2.2, markersize=7,
                color=ORANGE, label='Frobenius distance')

    eps_ref = np.array(epsilons)
    mid = len(epsilons) // 2
    scale_lin = mean_ent[mid] / eps_ref[mid]
    scale_sqrt = mean_fro[mid] / np.sqrt(eps_ref[mid])
    ax_d.loglog(eps_ref, scale_lin * eps_ref, '--', alpha=0.45,
                linewidth=1.1, color=UI_COLOR, label=r'$O(\varepsilon)$')
    ax_d.loglog(eps_ref, scale_sqrt * np.sqrt(eps_ref), '-.', alpha=0.45,
                linewidth=1.1, color=UI_COLOR, label=r'$O(\sqrt{\varepsilon})$')

    ax_d.set_xlabel(r'entropy temperature $\varepsilon$', fontsize=FS)
    ax_d.set_ylabel('distance to permutation', fontsize=FS)
    ax_d.set_title('assignment-limit convergence', fontsize=FS, color=FC)
    ax_d.legend(fontsize=FS_TICK - 2, loc='upper left', frameon=False)
    ax_d.grid(True, alpha=0.25, which='both')
    panel_label(ax_d, 'd')
    tint(ax_d)

    hol_perm = hol_perm_list[0]
    base_flat = hol_base.ravel()
    perm_flat = hol_perm.ravel()
    l2_err = np.sqrt(np.mean((perm_flat - base_flat) ** 2))
    rho, _ = spearmanr(base_flat, perm_flat)

    ax_e.scatter(base_flat, perm_flat, s=14, alpha=0.5, color=PURPLE,
                 edgecolors='none')

    lo = min(base_flat.min(), perm_flat.min())
    hi = max(base_flat.max(), perm_flat.max())
    pad = 0.05 * (hi - lo)
    ax_e.plot([lo - pad, hi + pad], [lo - pad, hi + pad], '--',
              color=UI_COLOR, linewidth=1.1, alpha=0.6, label=r'$y = x$')
    ax_e.set_xlim(lo - pad, hi + pad)
    ax_e.set_ylim(lo - pad, hi + pad)

    ax_e.set_xlabel(r'baseline $h_{\mathrm{op}}$', fontsize=FS)
    ax_e.set_ylabel(r'relabelled $h_{\mathrm{op}}$', fontsize=FS)
    ax_e.set_title('gauge invariance', fontsize=FS, color=FC)
    ax_e.legend(fontsize=FS_TICK - 2, loc='upper left', frameon=False)

    txt = f'$L^2 = {l2_err:.2e}$\n$\\rho = {rho:.4f}$'
    ax_e.text(0.96, 0.08, txt, transform=ax_e.transAxes,
              fontsize=FS_TICK - 1, ha='right', va='bottom', color=FC,
              bbox=dict(facecolor='white', edgecolor=UI_COLOR,
                        alpha=0.85, boxstyle='round,pad=0.3'))
    ax_e.set_aspect('equal', adjustable='box')
    ax_e.grid(True, alpha=0.25)
    panel_label(ax_e, 'e')
    tint(ax_e)
    _thin_ticks(ax_e, n=5)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, format='pdf', dpi=150, bbox_inches='tight')
    print(f'\nFigure saved to {outpath}')
    plt.close(fig)

if __name__ == '__main__':

    (predictions, jacobians, E_all, n_eval, s_y, s_g,
     u_eval, v_eval, u_grid, v_grid) = prepare_data(
        R_param=1.0, r_param=1.0, n_samples=1000,
        n_eval=30, n_models=10, seed0=0, epochs=200)

    hol_ent, hol_hun = task1_data(
        predictions, jacobians, E_all, n_eval, s_y, s_g,
        eps=0.5, side=1)

    epsilons, mean_ent, mean_fro = task2_data(
        predictions, jacobians, E_all, n_eval, s_y, s_g,
        epsilons=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
        n_edges=50, seed0=0)

    hol_base, hol_perm_list = task3_data(
        predictions, jacobians, E_all, n_eval, s_y, s_g,
        eps=0.5, side=1, n_trials=5)

    make_figure(hol_ent, hol_hun,
                epsilons, mean_ent, mean_fro,
                hol_base, hol_perm_list,
                u_eval, v_eval,
                outpath=os.path.join(os.path.dirname(__file__),
                                     'figs', 'sanity_checks.pdf'))

    print('\n' + '='*70)
    print('SUMMARY STATISTICS')
    print('='*70)
    rho_ab, _ = spearmanr(hol_ent.ravel(), hol_hun.ravel())
    print(f'  Entropic  holonomy: mean = {hol_ent.mean():.4f}, '
          f'range = [{hol_ent.min():.4f}, {hol_ent.max():.4f}]')
    print(f'  Hungarian holonomy: mean = {hol_hun.mean():.4f}, '
          f'range = [{hol_hun.min():.4f}, {hol_hun.max():.4f}]')
    print(f'  Spearman(entropic, hungarian) = {rho_ab:.4f}')
    diff = hol_ent - hol_hun
    print(f'  Difference: mean = {diff.mean():.4e}, '
          f'range = [{diff.min():.4e}, {diff.max():.4e}]')
    print()
    print(f'  Epsilon sweep:')
    for i, eps in enumerate(epsilons):
        print(f'    eps={eps:.2f}: row_entropy={mean_ent[i]:.4e}, '
              f'frobenius={mean_fro[i]:.4e}')
    print()
    for t, h in enumerate(hol_perm_list):
        l2 = np.sqrt(np.mean((h - hol_base)**2))
        print(f'  Gauge trial {t+1}: L2 = {l2:.3e}')
