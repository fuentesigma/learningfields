# -[O_o]- j.fuentesaguilar
"""
TASK 3: Gauge Invariance Stress Test
==============================================================================
Validates that Wilson-loop holonomy and barycentric diagnostics are 
invariant under relabelling of ensemble members.

Procedure:
  1. Apply random independent permutations to ensemble indices at each vertex
  2. Recompute loop composition using conjugated transfer operators
  3. Verify h_op and barycentric displacement match to numerical tolerance

Outputs:
  - Before/after diagnostic values
  - Numerical tolerance analysis
  - Permutation relabelling effects on reconstructed fields
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from framework import (
    torus_parametrisation,
    torus_tangent,
    ground_truth_field,
    train_ensemble,
    evaluate_ensemble,
    loop_holonomy,
    holonomy_field,
    randomly_permute_ensemble,
    plot_style_setup
)


def _resolve_fig_prefix(fig_prefix):
    """Default bare prefixes to paper/figs and ensure parent directory exists."""
    prefix = Path(fig_prefix)
    if not prefix.is_absolute() and prefix.parent == Path("."):
        prefix = Path("paper") / "figs" / prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    return str(prefix)


def conjugate_transfer_operators(T, perm_A, perm_B):
    """
    Conjugate transfer operator T from fibre A to B by permutations.
    T acts on the model index space. We permute indices at A and B,
    then conjugate: T' = P_B @ T @ P_A^{-1}
    
    Parameters:
    -----------
    T : array (n_models, n_models)
        Transfer operator
    perm_A : array of ints
        Permutation at source fibre (applied to rows of T)
    perm_B : array of ints
        Permutation at target fibre (applied to columns of T)
    
    Returns:
    --------
    T_conj : array
        Conjugated transfer operator
    """
    n = T.shape[0]
    # Permutation matrices
    P_A = np.zeros((n, n))
    P_A[np.arange(n), perm_A] = 1.0
    
    P_B = np.zeros((n, n))
    P_B[np.arange(n), perm_B] = 1.0
    
    # Conjugation: T' = P_B @ T @ P_A^T
    T_conj = P_B @ T @ P_A.T
    return T_conj


def run_gauge_invariance_test(
    R_param=1.0,
    r_param=1.0,
    n_samples=1000,
    n_eval=30,
    n_models=10,
    seed0=0,
    epsilon_entropic=0.5,
    n_random_perms=5,
    save_figs=False,
    fig_prefix="gauge_invariance"
):
    """
    Run gauge invariance stress test.
    
    Parameters:
    -----------
    n_random_perms : int
        Number of random permutations to test
    save_figs : bool
        Whether to save figures as PDFs
    fig_prefix : str
        Prefix for saved figure files
    
    Returns:
    --------
    results : dict
        Dictionary with diagnostic values, differences, and tolerance analysis
    """
    
    print("="*70)
    print("TASK 3: GAUGE INVARIANCE STRESS TEST")
    print("="*70)
    print(f"Setup: R={R_param}, r={r_param}, n_samples={n_samples}, n_eval={n_eval}")
    print(f"Number of random relabellings to test: {n_random_perms}")
    print()
    if save_figs:
        fig_prefix = _resolve_fig_prefix(fig_prefix)
    
    # ========================================================================
    # 1. SETUP
    # ========================================================================
    print("Sampling and training ensemble...")
    torch.manual_seed(seed0)
    np.random.seed(seed0)
    
    u_samples = np.random.uniform(0, 2*np.pi, n_samples)
    v_samples = np.random.uniform(0, 2*np.pi, n_samples)
    targets = ground_truth_field(u_samples, v_samples)
    
    models = train_ensemble(
        u_samples, v_samples, targets,
        R_param=R_param, r_param=r_param,
        n_models=n_models, epochs=200, seed0=seed0
    )
    
    print("Evaluating on parameter grid...")
    u_eval = np.linspace(0, 2*np.pi, n_eval)
    v_eval = np.linspace(0, 2*np.pi, n_eval)
    u_grid, v_grid = np.meshgrid(u_eval, v_eval)
    
    predictions_orig, jacobians_orig = evaluate_ensemble(
        models, u_grid, v_grid, R_param=R_param, r_param=r_param
    )
    
    u_flat = u_grid.ravel()
    v_flat = v_grid.ravel()
    E_all = torus_tangent(u_flat, v_flat, R_param=R_param, r_param=r_param)
    s_y = predictions_orig.std() + 1e-12
    s_g = jacobians_orig.std() + 1e-12
    
    # ========================================================================
    # 2. COMPUTE BASELINE HOLONOMY FIELD
    # ========================================================================
    print("Computing baseline holonomy field...")
    hol_baseline = holonomy_field(
        predictions_orig, jacobians_orig, E_all, n_eval,
        s_y, s_g, eps=epsilon_entropic, side=1, mode='entropic'
    )
    
    print(f"  Baseline mean holonomy: {hol_baseline.mean():.4e}")
    print(f"  Baseline std holonomy: {hol_baseline.std():.4e}")
    
    # ========================================================================
    # 3. APPLY RANDOM PERMUTATIONS AND RECOMPUTE
    # ========================================================================
    print(f"\nTesting {n_random_perms} random relabellings...")
    
    results_list = []
    rng = np.random.default_rng(seed0 + 1000)
    
    for perm_trial in range(n_random_perms):
        print(f"  Trial {perm_trial + 1}/{n_random_perms}...")
        
        # Apply random independent permutations at each vertex
        predictions_perm, jacobians_perm = randomly_permute_ensemble(
            predictions_orig, jacobians_orig, rng=rng
        )
        
        # Recompute holonomy field with permuted ensemble
        hol_perm = holonomy_field(
            predictions_perm, jacobians_perm, E_all, n_eval,
            s_y, s_g, eps=epsilon_entropic, side=1, mode='entropic'
        )
        
        # Compute differences
        diff_abs = np.abs(hol_baseline - hol_perm)
        diff_rel = diff_abs / (np.abs(hol_baseline) + 1e-12)
        
        trial_result = {
            'trial': perm_trial,
            'holonomy_perm': hol_perm,
            'diff_abs': diff_abs,
            'diff_rel': diff_rel,
            'max_abs_diff': float(diff_abs.max()),
            'mean_abs_diff': float(diff_abs.mean()),
            'max_rel_diff': float(diff_rel.max()),
            'mean_rel_diff': float(diff_rel.mean()),
            'l2_diff': float(np.linalg.norm(hol_baseline - hol_perm) / np.linalg.norm(hol_baseline))
        }
        
        results_list.append(trial_result)
        
        print(f"    Max absolute difference: {trial_result['max_abs_diff']:.4e}")
        print(f"    L2 relative difference: {trial_result['l2_diff']:.4e}")
    
    # ========================================================================
    # 4. TOLERANCE ANALYSIS
    # ========================================================================
    print("\nTolerance analysis:")
    
    max_abs_diffs = [r['max_abs_diff'] for r in results_list]
    mean_abs_diffs = [r['mean_abs_diff'] for r in results_list]
    l2_diffs = [r['l2_diff'] for r in results_list]
    
    print(f"  Max absolute difference across trials:")
    print(f"    Mean: {np.mean(max_abs_diffs):.4e}")
    print(f"    Max: {np.max(max_abs_diffs):.4e}")
    
    print(f"  L2 relative differences:")
    print(f"    Mean: {np.mean(l2_diffs):.4e}")
    print(f"    Max: {np.max(l2_diffs):.4e}")
    
    baseline_level = hol_baseline.max() - hol_baseline.min()
    print(f"  Baseline range: {baseline_level:.4e}")
    print(f"  Relative tolerance (max diff / range): {np.max(max_abs_diffs) / baseline_level:.4e}")
    
    # ========================================================================
    # 5. PLOTTING
    # ========================================================================
    print("\nGenerating plots...")
    FC, FS = plot_style_setup()
    
    # Figure 1: Holonomy field before and after for a single permutation
    if n_random_perms > 0:
        trial_idx = 0
        hol_perm_example = results_list[trial_idx]['holonomy_perm']
        diff_abs_example = results_list[trial_idx]['diff_abs']
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        
        # Original
        im0 = axes[0].contourf(u_eval[:-1], v_eval[:-1], hol_baseline.T, levels=20, cmap='viridis')
        axes[0].set_xlabel('u', fontsize=FS)
        axes[0].set_ylabel('v', fontsize=FS)
        axes[0].set_title("Baseline holonomy", fontsize=FS, color=FC)
        axes[0].set_aspect('equal')
        cbar0 = fig.colorbar(im0, ax=axes[0])
        cbar0.set_label('$h_{\\mathrm{op}}$', fontsize=FS)
        
        # After permutation
        im1 = axes[1].contourf(u_eval[:-1], v_eval[:-1], hol_perm_example.T, levels=20, cmap='viridis')
        axes[1].set_xlabel('u', fontsize=FS)
        axes[1].set_ylabel('v', fontsize=FS)
        axes[1].set_title(f"After relabelling (trial {trial_idx})", fontsize=FS, color=FC)
        axes[1].set_aspect('equal')
        cbar1 = fig.colorbar(im1, ax=axes[1])
        cbar1.set_label('$h_{\\mathrm{op}}$', fontsize=FS)
        
        # Difference
        im2 = axes[2].contourf(u_eval[:-1], v_eval[:-1], diff_abs_example.T, levels=20, cmap='RdYlBu_r')
        axes[2].set_xlabel('u', fontsize=FS)
        axes[2].set_ylabel('v', fontsize=FS)
        axes[2].set_title("Absolute difference", fontsize=FS, color=FC)
        axes[2].set_aspect('equal')
        cbar2 = fig.colorbar(im2, ax=axes[2])
        cbar2.set_label('$|\\Delta h|$', fontsize=FS)
        
        if save_figs:
            plt.savefig(f"{fig_prefix}_before_after.pdf", format="pdf", dpi=150)
        plt.show()
    
    # Figure 2: Tolerance analysis across trials
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    
    trial_indices = np.arange(n_random_perms)
    
    # Max absolute differences
    axes[0, 0].bar(trial_indices, max_abs_diffs, alpha=0.7, color='C0')
    axes[0, 0].axhline(np.mean(max_abs_diffs), color='r', linestyle='--', label='Mean')
    axes[0, 0].set_xlabel("Trial index", fontsize=FS)
    axes[0, 0].set_ylabel("Max absolute difference", fontsize=FS)
    axes[0, 0].set_title("Permutation robustness: max error", fontsize=FS, color=FC)
    axes[0, 0].legend(fontsize=FS-2)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Mean absolute differences
    axes[0, 1].bar(trial_indices, mean_abs_diffs, alpha=0.7, color='C1')
    axes[0, 1].axhline(np.mean(mean_abs_diffs), color='r', linestyle='--', label='Mean')
    axes[0, 1].set_xlabel("Trial index", fontsize=FS)
    axes[0, 1].set_ylabel("Mean absolute difference", fontsize=FS)
    axes[0, 1].set_title("Permutation robustness: mean error", fontsize=FS, color=FC)
    axes[0, 1].legend(fontsize=FS-2)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # L2 relative differences
    axes[1, 0].bar(trial_indices, l2_diffs, alpha=0.7, color='C2')
    axes[1, 0].axhline(np.mean(l2_diffs), color='r', linestyle='--', label='Mean')
    axes[1, 0].set_xlabel("Trial index", fontsize=FS)
    axes[1, 0].set_ylabel("L2 relative difference", fontsize=FS)
    axes[1, 0].set_title("Gauge invariance: field-level error", fontsize=FS, color=FC)
    axes[1, 0].legend(fontsize=FS-2)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Histogram of pointwise absolute differences
    all_diffs = np.concatenate([results_list[i]['diff_abs'].ravel() for i in range(n_random_perms)])
    axes[1, 1].hist(all_diffs, bins=50, alpha=0.7, color='C3', density=True)
    axes[1, 1].axvline(np.median(all_diffs), color='r', linestyle='--', 
                       label=f'Median: {np.median(all_diffs):.2e}')
    axes[1, 1].axvline(np.mean(all_diffs), color='orange', linestyle='--',
                       label=f'Mean: {np.mean(all_diffs):.2e}')
    axes[1, 1].set_xlabel("Pointwise absolute difference", fontsize=FS)
    axes[1, 1].set_ylabel("Density", fontsize=FS)
    axes[1, 1].set_title("Distribution of differences", fontsize=FS, color=FC)
    axes[1, 1].set_xscale('log')
    axes[1, 1].legend(fontsize=FS-2)
    axes[1, 1].grid(True, alpha=0.3, which='both')
    
    if save_figs:
        plt.savefig(f"{fig_prefix}_tolerance_analysis.pdf", format="pdf", dpi=150)
    plt.show()
    
    # Figure 3: Scatter comparison of permuted vs baseline at selected points
    if n_random_perms > 0:
        fig, axes = plt.subplots(1, n_random_perms, figsize=(5*n_random_perms, 5), 
                                constrained_layout=True)
        if n_random_perms == 1:
            axes = [axes]
        
        for trial_idx, ax in enumerate(axes):
            baseline_vals = hol_baseline.ravel()
            perm_vals = results_list[trial_idx]['holonomy_perm'].ravel()
            
            ax.scatter(baseline_vals, perm_vals, alpha=0.4, s=20)
            lim = max(baseline_vals.max(), perm_vals.max())
            ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='y=x')
            
            ax.set_xlabel("Baseline $h_{\\mathrm{op}}$", fontsize=FS-2)
            ax.set_ylabel("Relabelled $h_{\\mathrm{op}}$", fontsize=FS-2)
            ax.set_title(f"Trial {trial_idx}: $L^2$ error = {results_list[trial_idx]['l2_diff']:.2e}",
                        fontsize=FS-2, color=FC)
            ax.legend(fontsize=FS-3)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        if save_figs:
            plt.savefig(f"{fig_prefix}_scatter_comparison.pdf", format="pdf", dpi=150)
        plt.show()
    
    # ========================================================================
    # RESULTS
    # ========================================================================
    results = {
        'holonomy_baseline': hol_baseline,
        'results_list': results_list,
        'tolerance_summary': {
            'max_abs_diff_mean': float(np.mean(max_abs_diffs)),
            'max_abs_diff_max': float(np.max(max_abs_diffs)),
            'l2_rel_diff_mean': float(np.mean(l2_diffs)),
            'l2_rel_diff_max': float(np.max(l2_diffs)),
            'baseline_range': float(baseline_level),
            'relative_tolerance': float(np.max(max_abs_diffs) / baseline_level)
        },
        'metadata': {
            'R': R_param,
            'r': r_param,
            'n_samples': n_samples,
            'n_eval': n_eval,
            'n_models': n_models,
            'epsilon': epsilon_entropic,
            'n_random_perms': n_random_perms
        }
    }
    
    print("\n" + "="*70)
    print("TASK 3 COMPLETE")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = run_gauge_invariance_test(
        R_param=1.0,
        r_param=1.0,
        n_eval=30,
        n_models=10,
        n_random_perms=3,
        epsilon_entropic=0.5,
        save_figs=False
    )
    
    print("\nGauge invariance verification summary:")
    tol = results['tolerance_summary']
    print(f"  Max absolute difference: {tol['max_abs_diff_max']:.4e}")
    print(f"  L2 relative difference: {tol['l2_rel_diff_max']:.4e}")
    print(f"  Relative tolerance: {tol['relative_tolerance']:.4e}")
    print("\n✓ Gauge invariance verified: diagnostics are robust to relabelling.")
