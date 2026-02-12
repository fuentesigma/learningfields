# -[O_o]- j.fuentesaguilar
"""
TASK 2: Assignment Limit Diagnostic
==============================================================================
Demonstrates regime (B) behaviour: as ε → 0, the entropic coupling 
concentrates near the Hungarian solution.

Measures two metrics:
  1. Row entropy: ∑_b -p_b log(p_b) for each row of the coupling
  2. Frobenius distance: ||P_entropic - P_hungarian||_F

Outputs:
  - Mean metrics vs ε
  - Scatter showing "closeness to permutation" per edge
  - Convergence plots
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
    fibre_jets,
    rotation_AB,
    jet_costm,
    transfer_operator_entropic,
    transfer_operator_hungarian,
    permutation_distance_entropy,
    permutation_distance_frobenius,
    plot_style_setup
)


def _resolve_fig_prefix(fig_prefix):
    """Default bare prefixes to paper/figs and ensure parent directory exists."""
    prefix = Path(fig_prefix)
    if not prefix.is_absolute() and prefix.parent == Path("."):
        prefix = Path("paper") / "figs" / prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    return str(prefix)


def compute_edge_metrics_epsilon_sweep(
    i_A, j_A, i_B, j_B,
    predictions, jacobians, E_all, n_eval,
    s_y, s_g,
    epsilons
):
    """
    For a single edge, compute transport couplings at multiple epsilon values.
    Returns dictionaries of metrics for each epsilon.
    """
    A = fibre_jets(i_A, j_A, predictions, jacobians, n_eval)
    B = fibre_jets(i_B, j_B, predictions, jacobians, n_eval)
    R_AB = rotation_AB(i_A, j_A, i_B, j_B, E_all, n_eval)
    C = jet_costm(A, B, s_y, s_g, R_AB)
    
    # Hungarian solution (single permutation)
    T_hungarian = transfer_operator_hungarian(C)
    
    results = {
        'epsilons': [],
        'entropies': [],
        'frobenius_dists': [],
        'mean_holonomy': []
    }
    
    for eps in epsilons:
        T_entropic = transfer_operator_entropic(C, eps=eps)
        
        # Entropy of rows as "distance to permutation"
        ent = permutation_distance_entropy(T_entropic)
        
        # Frobenius distance to Hungarian
        fro = permutation_distance_frobenius(T_entropic, T_hungarian)
        
        results['epsilons'].append(eps)
        results['entropies'].append(ent)
        results['frobenius_dists'].append(fro)
        results['mean_holonomy'].append(0)  # Placeholder
    
    return results, T_hungarian


def run_assignment_limit_diagnostic(
    R_param=1.0,
    r_param=1.0,
    n_samples=1000,
    n_eval=30,
    n_models=10,
    seed0=0,
    epsilons=None,
    save_figs=False,
    fig_prefix="assignment_limit"
):
    """
    Run epsilon sweep showing concentration to permutation as ε → 0.
    
    Parameters:
    -----------
    epsilons : list or None
        Epsilon values to sweep. If None, uses [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    save_figs : bool
        Whether to save figures as PDFs
    fig_prefix : str
        Prefix for saved figure files
    
    Returns:
    --------
    results : dict
        Dictionary with epsilon sweep data, statistics, and diagnostics
    """
    
    if epsilons is None:
        epsilons = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    
    print("="*70)
    print("TASK 2: ASSIGNMENT LIMIT DIAGNOSTIC")
    print("="*70)
    print(f"Setup: R={R_param}, r={r_param}, n_samples={n_samples}, n_eval={n_eval}")
    print(f"Epsilon sweep: {epsilons}")
    print()
    if save_figs:
        fig_prefix = _resolve_fig_prefix(fig_prefix)
    
    # ========================================================================
    # 1. SETUP (reuse Task 1 infrastructure)
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
    
    predictions, jacobians = evaluate_ensemble(
        models, u_grid, v_grid, R_param=R_param, r_param=r_param
    )
    
    u_flat = u_grid.ravel()
    v_flat = v_grid.ravel()
    E_all = torus_tangent(u_flat, v_flat, R_param=R_param, r_param=r_param)
    s_y = predictions.std() + 1e-12
    s_g = jacobians.std() + 1e-12
    
    # ========================================================================
    # 2. COLLECT EDGE METRICS OVER EPSILON SWEEP
    # ========================================================================
    print(f"Computing metrics for {len(epsilons)} epsilon values...")
    
    # Sample a subset of edges to avoid overwhelming computation
    n_edges_sample = min(50, (n_eval - 1) * (n_eval - 1) * 4)
    all_edges = []

    np.random.seed(seed0 + 100)
    for i in range(n_eval - 1):
        for j in range(n_eval - 1):
            # Horizontal edges
            all_edges.append(((i, j), (i, j + 1)))
            # Vertical edges
            all_edges.append(((i, j), (i + 1, j)))

    chosen_idx = np.random.choice(len(all_edges), size=min(n_edges_sample, len(all_edges)), replace=False)
    edges_to_sample = [all_edges[idx] for idx in chosen_idx]
    
    edge_metrics_list = []
    
    for idx, ((i_A, j_A), (i_B, j_B)) in enumerate(edges_to_sample):
        if idx % 10 == 0:
            print(f"  Edge {idx+1}/{len(edges_to_sample)}...")
        metrics, _ = compute_edge_metrics_epsilon_sweep(
            i_A, j_A, i_B, j_B,
            predictions, jacobians, E_all, n_eval,
            s_y, s_g, epsilons
        )
        edge_metrics_list.append(metrics)
    
    # ========================================================================
    # 3. AGGREGATE STATISTICS
    # ========================================================================
    print("Aggregating statistics...")
    
    mean_entropy = np.zeros(len(epsilons))
    mean_frobenius = np.zeros(len(epsilons))
    std_entropy = np.zeros(len(epsilons))
    std_frobenius = np.zeros(len(epsilons))
    
    for eps_idx in range(len(epsilons)):
        entropies = [m['entropies'][eps_idx] for m in edge_metrics_list]
        frobenius = [m['frobenius_dists'][eps_idx] for m in edge_metrics_list]
        
        mean_entropy[eps_idx] = np.mean(entropies)
        std_entropy[eps_idx] = np.std(entropies)
        mean_frobenius[eps_idx] = np.mean(frobenius)
        std_frobenius[eps_idx] = np.std(frobenius)
    
    # ========================================================================
    # 4. PLOTTING
    # ========================================================================
    print("Generating plots...")
    FC, FS = plot_style_setup()
    
    # Figure 1: Epsilon sweep with error bands
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    
    # Row entropy vs epsilon
    axes[0].loglog(epsilons, mean_entropy, 'o-', linewidth=2, markersize=8, label='mean')
    axes[0].fill_between(epsilons,
                         mean_entropy - std_entropy,
                         mean_entropy + std_entropy,
                         alpha=0.3, label='±1 std')
    axes[0].set_xlabel("Entropy temperature ε", fontsize=FS)
    axes[0].set_ylabel("Mean row entropy", fontsize=FS)
    axes[0].set_title("Distance to permutation: Row entropy", fontsize=FS, color=FC)
    axes[0].legend(fontsize=FS-2)
    axes[0].grid(True, alpha=0.3, which='both')
    
    # Frobenius distance vs epsilon
    axes[1].loglog(epsilons, mean_frobenius, 's-', linewidth=2, markersize=8, label='mean')
    axes[1].fill_between(epsilons,
                         mean_frobenius - std_frobenius,
                         mean_frobenius + std_frobenius,
                         alpha=0.3, label='±1 std')
    axes[1].set_xlabel("Entropy temperature ε", fontsize=FS)
    axes[1].set_ylabel("Mean Frobenius distance", fontsize=FS)
    axes[1].set_title("Distance to permutation: Frobenius norm", fontsize=FS, color=FC)
    axes[1].legend(fontsize=FS-2)
    axes[1].grid(True, alpha=0.3, which='both')
    
    if save_figs:
        plt.savefig(f"{fig_prefix}_epsilon_sweep.pdf", format="pdf", dpi=150)
    plt.show()
    
    # Figure 2: Scatter of per-edge metrics at extremal epsilon values
    if len(epsilons) >= 2:
        eps_small_idx = 0
        eps_large_idx = -1
        eps_small = epsilons[eps_small_idx]
        eps_large = epsilons[eps_large_idx]
        
        ent_small = [m['entropies'][eps_small_idx] for m in edge_metrics_list]
        ent_large = [m['entropies'][eps_large_idx] for m in edge_metrics_list]
        fro_small = [m['frobenius_dists'][eps_small_idx] for m in edge_metrics_list]
        fro_large = [m['frobenius_dists'][eps_large_idx] for m in edge_metrics_list]
        
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        
        # Entropy comparison
        axes[0].scatter(ent_small, ent_large, alpha=0.5, s=30)
        axes[0].set_xlabel(f"Row entropy at ε={eps_small}", fontsize=FS)
        axes[0].set_ylabel(f"Row entropy at ε={eps_large}", fontsize=FS)
        axes[0].set_title("Row entropy: small ε vs large ε", fontsize=FS, color=FC)
        axes[0].grid(True, alpha=0.3)
        
        # Frobenius comparison
        axes[1].scatter(fro_small, fro_large, alpha=0.5, s=30)
        axes[1].set_xlabel(f"Frobenius distance at ε={eps_small}", fontsize=FS)
        axes[1].set_ylabel(f"Frobenius distance at ε={eps_large}", fontsize=FS)
        axes[1].set_title("Frobenius distance: small ε vs large ε", fontsize=FS, color=FC)
        axes[1].grid(True, alpha=0.3)
        
        if save_figs:
            plt.savefig(f"{fig_prefix}_extremal_comparison.pdf", format="pdf", dpi=150)
        plt.show()
    
    # Figure 3: Convergence rate on log-log scale
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    
    ax.loglog(epsilons, mean_entropy, 'o-', linewidth=2.5, markersize=10,
              label='Row entropy', color='C0')
    ax.loglog(epsilons, mean_frobenius, 's-', linewidth=2.5, markersize=10,
              label='Frobenius distance', color='C1')
    
    # Overlay reference rates
    eps_ref = np.array(epsilons)
    ax.loglog(eps_ref, eps_ref, '--', alpha=0.4, linewidth=1, label='O(ε)', color='gray')
    ax.loglog(eps_ref, eps_ref**0.5, '-.', alpha=0.4, linewidth=1, label='O(√ε)', color='gray')
    
    ax.set_xlabel("Entropy temperature ε", fontsize=FS)
    ax.set_ylabel("Distance to permutation", fontsize=FS)
    ax.set_title("Assignment limit convergence (regime B)", fontsize=FS, color=FC)
    ax.legend(fontsize=FS-2, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    if save_figs:
        plt.savefig(f"{fig_prefix}_convergence_rates.pdf", format="pdf", dpi=150)
    plt.show()
    
    # ========================================================================
    # RESULTS
    # ========================================================================
    results = {
        'epsilons': list(epsilons),
        'mean_entropy': mean_entropy.tolist(),
        'std_entropy': std_entropy.tolist(),
        'mean_frobenius': mean_frobenius.tolist(),
        'std_frobenius': std_frobenius.tolist(),
        'edge_metrics_list': edge_metrics_list,
        'metadata': {
            'R': R_param,
            'r': r_param,
            'n_samples': n_samples,
            'n_eval': n_eval,
            'n_models': n_models,
            'n_edges_sampled': len(edges_to_sample)
        }
    }
    
    print("\n" + "="*70)
    print("TASK 2 COMPLETE")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = run_assignment_limit_diagnostic(
        R_param=1.0,
        r_param=1.0,
        n_eval=30,
        n_models=10,
        epsilons=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
        save_figs=False
    )
    
    print("\nKey findings:")
    print(f"  At ε={results['epsilons'][0]}: mean row entropy = {results['mean_entropy'][0]:.4e}")
    print(f"  At ε={results['epsilons'][-1]}: mean row entropy = {results['mean_entropy'][-1]:.4e}")
    print(f"  Entropy decreases by factor: {results['mean_entropy'][-1] / (results['mean_entropy'][0] + 1e-12):.2f}")
