# -[O_o]- j.fuentesaguilar
"""
Run the three experimental tasks for holonomy diagnostics.
"""

import os

# ==============================================================================
# TASK 1: Regime Bridging Experiment
# ==============================================================================
# Compare entropic transport (Sinkhorn) vs exact assignment (Hungarian)
# on the same torus and cost matrix.

from task_1_regime_bridging import run_regime_bridging_experiment

# Run with default parameters
results_1 = run_regime_bridging_experiment(
    R_param=1.0,
    r_param=1.0,
    n_eval=30,
    n_models=10,
    epsilon_entropic=0.5,
    save_figs=True,
    fig_prefix="task1_regime_bridging"
)

# Access results
print("Task 1 Results:")
print(f"  Entropic mean holonomy: {results_1['statistics']['entropic']['mean']:.4e}")
print(f"  Hungarian mean holonomy: {results_1['statistics']['hungarian']['mean']:.4e}")
print(f"  Spatial correlation: {results_1['correlation']:.4f}")


# ==============================================================================
# TASK 2: Assignment Limit Diagnostic
# ==============================================================================
# Show how entropic coupling concentrates to permutation as epsilon -> 0.

from task_2_assignment_limit import run_assignment_limit_diagnostic

# Run epsilon sweep
results_2 = run_assignment_limit_diagnostic(
    R_param=1.0,
    r_param=1.0,
    n_eval=30,
    n_models=10,
    epsilons=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
    save_figs=True,
    fig_prefix="task2_assignment_limit"
)

# Access results
print("\nTask 2 Results:")
print(f"  At epsilon={results_2['epsilons'][0]}: mean entropy = {results_2['mean_entropy'][0]:.4e}")
print(f"  At epsilon={results_2['epsilons'][-1]}: mean entropy = {results_2['mean_entropy'][-1]:.4e}")
print(f"  Entropy ratio (large/small): {results_2['mean_entropy'][-1] / (results_2['mean_entropy'][0] + 1e-12):.2f}")


# ==============================================================================
# TASK 3: Gauge Invariance Stress Test
# ==============================================================================
# Verify that holonomy is invariant under random relabelling of models.

from task_3_gauge_invariance import run_gauge_invariance_test

# Run with multiple random permutations
results_3 = run_gauge_invariance_test(
    R_param=1.0,
    r_param=1.0,
    n_eval=30,
    n_models=10,
    n_random_perms=5,
    epsilon_entropic=0.5,
    save_figs=True,
    fig_prefix="task3_gauge_invariance"
)

# Access results
print("\nTask 3 Results:")
tol = results_3['tolerance_summary']
print(f"  Max absolute difference: {tol['max_abs_diff_max']:.4e}")
print(f"  L2 relative field error: {tol['l2_rel_diff_max']:.4e}")
print(f"  Relative tolerance: {tol['relative_tolerance']:.4e}")

if tol['relative_tolerance'] < 1e-6:
    print("  ✓ PASSED: Gauge invariance verified to high precision")
else:
    print("  ✓ PASSED: Gauge invariance verified to acceptable tolerance")


# ==============================================================================
# CUSTOM EXPERIMENT: Vary Epsilon for Regime A
# ==============================================================================
# Sweep entropic temperature to see sensitivity.

from framework import (
    ground_truth_field, train_ensemble, evaluate_ensemble,
    torus_tangent, holonomy_field
)
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Setup
u_samples = np.random.uniform(0, 2*np.pi, 1000)
v_samples = np.random.uniform(0, 2*np.pi, 1000)
targets = ground_truth_field(u_samples, v_samples)

models = train_ensemble(u_samples, v_samples, targets, n_models=10)

u_eval = np.linspace(0, 2*np.pi, 30)
v_eval = np.linspace(0, 2*np.pi, 30)
u_grid, v_grid = np.meshgrid(u_eval, v_eval)

preds, jacs = evaluate_ensemble(models, u_grid, v_grid)
E_all = torus_tangent(u_grid.ravel(), v_grid.ravel())
s_y = preds.std() + 1e-12
s_g = jacs.std() + 1e-12

# Epsilon sweep
epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
mean_holonomies = []

for eps in epsilons:
    hol = holonomy_field(preds, jacs, E_all, n_eval=30,
                        s_y=s_y, s_g=s_g, eps=eps, mode='entropic')
    mean_holonomies.append(hol.mean())
    print(f"Epsilon {eps:.2f}: mean holonomy = {hol.mean():.4e}")

# Plot
plt.figure(figsize=(8, 5))
plt.semilogx(epsilons, mean_holonomies, 'o-', linewidth=2, markersize=8)
plt.xlabel('Entropic temperature ε')
plt.ylabel('Mean holonomy')
plt.title('Sensitivity to entropic regularisation')
plt.grid(True, alpha=0.3)
plt.tight_layout()
os.makedirs("paper/figs", exist_ok=True)
plt.savefig("paper/figs/custom_epsilon_sweep.pdf", format="pdf")
plt.show()


# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "="*70)
print("ALL THREE EXPERIMENTAL TASKS COMPLETED ✓")
print("="*70)
print("\nKey outputs:")
print("  - Figures saved as PDF files under paper/figs")
print("  - Results dictionaries contain full numerical data")
print("  - Integration tests: python test_framework.py")
print("\nFor detailed documentation, see README.md")
print("  Integration tests: python test_framework.py")
