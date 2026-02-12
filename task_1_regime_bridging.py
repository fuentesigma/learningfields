# -[O_o]- j.fuentesaguilar
"""
TASK 1: Regime Bridging Experiment
==============================================================================
Compares entropic transport (Sinkhorn, regime A) with exact assignment 
(Hungarian, regime C) on the same torus setup.

Outputs:
  - Holonomy field distributions for both regimes
  - Spatial pattern comparison
  - Scatter plots correlating the two diagnostics
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
    holonomy_field,
    plot_style_setup,
    style_3d_axes,
    style_colorbar
)


def _resolve_fig_prefix(fig_prefix):
    """Default bare prefixes to paper/figs and ensure parent directory exists."""
    prefix = Path(fig_prefix)
    if not prefix.is_absolute() and prefix.parent == Path("."):
        prefix = Path("paper") / "figs" / prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    return str(prefix)


def run_regime_bridging_experiment(
    R_param=1.0,
    r_param=1.0,
    n_samples=1000,
    n_eval=30,
    n_models=10,
    seed0=0,
    epsilon_entropic=0.5,
    save_figs=False,
    fig_prefix="regime_bridging"
):
    """
    Run full regime bridging comparison on torus.
    
    Parameters:
    -----------
    epsilon_entropic : float
        Entropic temperature for Sinkhorn solver
    save_figs : bool
        Whether to save figures as PDFs
    fig_prefix : str
        Prefix for saved figure files
    
    Returns:
    --------
    results : dict
        Dictionary with holonomy fields, statistics, and metadata
    """
    
    print("="*70)
    print("TASK 1: REGIME BRIDGING EXPERIMENT")
    print("="*70)
    print(f"Setup: R={R_param}, r={r_param}, n_samples={n_samples}, n_eval={n_eval}")
    print(f"Entropic temperature epsilon={epsilon_entropic}")
    print()
    if save_figs:
        fig_prefix = _resolve_fig_prefix(fig_prefix)
    
    # ========================================================================
    # 1. SAMPLE AND TRAIN
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
    print(f"  Trained {n_models} models")
    
    # ========================================================================
    # 2. EVALUATE ON GRID
    # ========================================================================
    print("Evaluating on parameter grid...")
    u_eval = np.linspace(0, 2*np.pi, n_eval)
    v_eval = np.linspace(0, 2*np.pi, n_eval)
    u_grid, v_grid = np.meshgrid(u_eval, v_eval)
    
    predictions, jacobians = evaluate_ensemble(
        models, u_grid, v_grid, R_param=R_param, r_param=r_param
    )
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Jacobians shape: {jacobians.shape}")
    
    # Precompute tangent frames and calibration
    u_flat = u_grid.ravel()
    v_flat = v_grid.ravel()
    E_all = torus_tangent(u_flat, v_flat, R_param=R_param, r_param=r_param)
    s_y = predictions.std() + 1e-12
    s_g = jacobians.std() + 1e-12
    
    # ========================================================================
    # 3. COMPUTE HOLONOMY FIELDS FOR BOTH REGIMES
    # ========================================================================
    print("Computing holonomy fields...")
    print("  Mode A: Entropic transport (Sinkhorn)...")
    hol_entropic = holonomy_field(
        predictions, jacobians, E_all, n_eval,
        s_y, s_g, eps=epsilon_entropic, side=1, mode='entropic'
    )
    
    print("  Mode C: Exact assignment (Hungarian)...")
    hol_hungarian = holonomy_field(
        predictions, jacobians, E_all, n_eval,
        s_y, s_g, eps=epsilon_entropic, side=1, mode='hungarian'
    )
    
    # ========================================================================
    # 4. COMPUTE STATISTICS
    # ========================================================================
    stats = {
        'entropic': {
            'mean': float(hol_entropic.mean()),
            'median': float(np.median(hol_entropic)),
            'std': float(hol_entropic.std()),
            'min': float(hol_entropic.min()),
            'max': float(hol_entropic.max()),
            '95th_percentile': float(np.percentile(hol_entropic, 95))
        },
        'hungarian': {
            'mean': float(hol_hungarian.mean()),
            'median': float(np.median(hol_hungarian)),
            'std': float(hol_hungarian.std()),
            'min': float(hol_hungarian.min()),
            'max': float(hol_hungarian.max()),
            '95th_percentile': float(np.percentile(hol_hungarian, 95))
        }
    }
    
    print()
    print("Holonomy statistics (entropic regime):")
    print(f"  Mean: {stats['entropic']['mean']:.4e}")
    print(f"  Median: {stats['entropic']['median']:.4e}")
    print(f"  Std: {stats['entropic']['std']:.4e}")
    
    print("Holonomy statistics (Hungarian regime):")
    print(f"  Mean: {stats['hungarian']['mean']:.4e}")
    print(f"  Median: {stats['hungarian']['median']:.4e}")
    print(f"  Std: {stats['hungarian']['std']:.4e}")
    
    # ========================================================================
    # 5. SPATIAL PATTERN CORRELATION
    # ========================================================================
    from scipy.stats import spearmanr
    corr, _ = spearmanr(hol_entropic.ravel(), hol_hungarian.ravel())
    print(f"Spearman correlation between regimes: {corr:.4f}")
    
    # ========================================================================
    # 6. PLOTTING
    # ========================================================================
    print("\nGenerating plots...")
    
    # Ambient coordinates for 3D surface plots
    x_eval, y_eval, z_eval = torus_parametrisation(
        u_flat, v_flat, R_param=R_param, r_param=r_param
    )
    xs = x_eval.reshape(n_eval, n_eval)[:-1, :-1]
    ys = y_eval.reshape(n_eval, n_eval)[:-1, :-1]
    zs = z_eval.reshape(n_eval, n_eval)[:-1, :-1]
    
    # Normalize colour scales
    vmin = min(hol_entropic.min(), hol_hungarian.min())
    vmax = max(hol_entropic.max(), hol_hungarian.max())
    
    FC, FS = plot_style_setup()
    
    # Figure 1: 3D surface comparison
    fig = plt.figure(figsize=(14, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    norm_e = (hol_entropic - vmin) / (vmax - vmin + 1e-12)
    ax1.plot_surface(xs, ys, zs, facecolors=plt.cm.viridis(norm_e),
                     linewidth=0, antialiased=False, rstride=1, cstride=1)
    ax1.set_title("Regime A: Entropic (Sinkhorn)", fontsize=FS, color=FC)
    style_3d_axes(ax1, FC=FC, FS=FS)
    
    ax2 = fig.add_subplot(122, projection='3d')
    norm_h = (hol_hungarian - vmin) / (vmax - vmin + 1e-12)
    ax2.plot_surface(xs, ys, zs, facecolors=plt.cm.viridis(norm_h),
                     linewidth=0, antialiased=False, rstride=1, cstride=1)
    ax2.set_title("Regime C: Exact Assignment (Hungarian)", fontsize=FS, color=FC)
    style_3d_axes(ax2, FC=FC, FS=FS)
    
    # Add common colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label("holonomy $h_{\\mathrm{op}}$", color=FC, fontsize=FS)
    
    if save_figs:
        plt.savefig(f"{fig_prefix}_surfaces_3d.pdf", format="pdf", dpi=150)
    plt.show()

    # Figure 2: 2D contour comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    
    im0 = axes[0].contourf(u_eval[:-1], v_eval[:-1], hol_entropic.T, levels=20, cmap='viridis')
    axes[0].set_xlabel('u', fontsize=FS)
    axes[0].set_ylabel('v', fontsize=FS)
    axes[0].set_title("Regime A: Entropic", fontsize=FS, color=FC)
    axes[0].set_aspect('equal')
    cbar0 = fig.colorbar(im0, ax=axes[0])
    cbar0.set_label('$h_{\\mathrm{op}}$', fontsize=FS)
    
    im1 = axes[1].contourf(u_eval[:-1], v_eval[:-1], hol_hungarian.T, levels=20, cmap='viridis')
    axes[1].set_xlabel('u', fontsize=FS)
    axes[1].set_ylabel('v', fontsize=FS)
    axes[1].set_title("Regime C: Hungarian", fontsize=FS, color=FC)
    axes[1].set_aspect('equal')
    cbar1 = fig.colorbar(im1, ax=axes[1])
    cbar1.set_label('$h_{\\mathrm{op}}$', fontsize=FS)
    
    # Difference map
    diff = hol_entropic - hol_hungarian
    im2 = axes[2].contourf(u_eval[:-1], v_eval[:-1], diff.T, levels=20, cmap='RdBu_r')
    axes[2].set_xlabel('u', fontsize=FS)
    axes[2].set_ylabel('v', fontsize=FS)
    axes[2].set_title("Difference (Entropic - Hungarian)", fontsize=FS, color=FC)
    axes[2].set_aspect('equal')
    cbar2 = fig.colorbar(im2, ax=axes[2])
    cbar2.set_label('$\\Delta h$', fontsize=FS)
    
    if save_figs:
        plt.savefig(f"{fig_prefix}_contours_2d.pdf", format="pdf", dpi=150)
    plt.show()
    
    # Figure 3: Scatter and distribution comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    
    # Scatter
    axes[0].scatter(hol_entropic.ravel(), hol_hungarian.ravel(), alpha=0.5, s=20)
    axes[0].set_xlabel("Entropic $h_{\\mathrm{op}}$", fontsize=FS)
    axes[0].set_ylabel("Hungarian $h_{\\mathrm{op}}$", fontsize=FS)
    axes[0].set_title(f"Correlation (Spearman: {corr:.3f})", fontsize=FS, color=FC)
    lim = max(hol_entropic.max(), hol_hungarian.max())
    axes[0].plot([0, lim], [0, lim], 'k--', alpha=0.3, label='y=x')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Histograms
    axes[1].hist(hol_entropic.ravel(), bins=30, alpha=0.6, label='Entropic', density=True)
    axes[1].hist(hol_hungarian.ravel(), bins=30, alpha=0.6, label='Hungarian', density=True)
    axes[1].set_xlabel("$h_{\\mathrm{op}}$", fontsize=FS)
    axes[1].set_ylabel("Density", fontsize=FS)
    axes[1].set_title("Distribution comparison", fontsize=FS, color=FC)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Box plot
    data_to_plot = [hol_entropic.ravel(), hol_hungarian.ravel()]
    bp = axes[2].boxplot(data_to_plot, labels=['Entropic', 'Hungarian'])
    axes[2].set_ylabel("$h_{\\mathrm{op}}$", fontsize=FS)
    axes[2].set_title("Distribution summary", fontsize=FS, color=FC)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    if save_figs:
        plt.savefig(f"{fig_prefix}_stats_comparison.pdf", format="pdf", dpi=150)
    plt.show()
    
    # ========================================================================
    # RESULTS
    # ========================================================================
    results = {
        'holonomy_entropic': hol_entropic,
        'holonomy_hungarian': hol_hungarian,
        'statistics': stats,
        'correlation': float(corr),
        'metadata': {
            'R': R_param,
            'r': r_param,
            'n_samples': n_samples,
            'n_eval': n_eval,
            'n_models': n_models,
            'epsilon': epsilon_entropic
        }
    }
    
    print("\n" + "="*70)
    print("TASK 1 COMPLETE")
    print("="*70)
    
    return results


if __name__ == "__main__":
    # Run with default parameters
    results = run_regime_bridging_experiment(
        R_param=1.0,
        r_param=1.0,
        n_eval=30,
        n_models=10,
        epsilon_entropic=0.5,
        save_figs=False
    )
    
    print("\nKey findings:")
    print(f"  Entropic regime mean holonomy: {results['statistics']['entropic']['mean']:.4e}")
    print(f"  Hungarian regime mean holonomy: {results['statistics']['hungarian']['mean']:.4e}")
    print(f"  Spatial correlation: {results['correlation']:.4f}")
