# -[O_o]- j.fuentesaguilar
"""
Integration tests for framework components.
Run before full experiments: python test_framework.py
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing

from framework import (
    torus_parametrisation,
    torus_tangent,
    ground_truth_field,
    train_ensemble,
    evaluate_ensemble,
    rotation_SO2_closest,
    sinkhorn_uniform,
    transfer_operator_entropic,
    transfer_operator_hungarian,
    loop_holonomy,
    holonomy_field,
    randomly_permute_ensemble
)


def test_geometry():
    """Test torus geometry functions."""
    print("Testing geometry...")
    u = np.array([0.0, np.pi/2, np.pi])
    v = np.array([0.0, np.pi/4, np.pi/2])
    
    X, Y, Z = torus_parametrisation(u, v, R_param=1.0, r_param=1.0)
    assert X.shape == u.shape, "X shape mismatch"
    assert Y.shape == u.shape, "Y shape mismatch"
    assert Z.shape == u.shape, "Z shape mismatch"
    
    E = torus_tangent(u, v, R_param=1.0, r_param=1.0)
    assert E.shape == (3, 2, 3), f"E shape should be (3, 2, 3), got {E.shape}"
    
    # Check orthonormality
    for i in range(E.shape[0]):
        e1 = E[i, 0, :]
        e2 = E[i, 1, :]
        assert np.abs(np.dot(e1, e1) - 1.0) < 1e-10, "e1 not normalised"
        assert np.abs(np.dot(e2, e2) - 1.0) < 1e-10, "e2 not normalised"
        assert np.abs(np.dot(e1, e2)) < 1e-10, "e1, e2 not orthogonal"
    
    print("  ✓ Geometry tests passed")


def test_ground_truth():
    """Test ground truth field."""
    print("Testing ground truth field...")
    u = np.linspace(0, 2*np.pi, 10)
    v = np.linspace(0, 2*np.pi, 10)
    U, V = np.meshgrid(u, v)
    
    g = ground_truth_field(U, V)
    assert g.shape == U.shape, "Field shape mismatch"
    assert np.all(np.isfinite(g)), "Field contains NaN/Inf"
    
    print("  ✓ Ground truth field tests passed")


def test_neural_network():
    """Test MLP training and evaluation."""
    print("Testing neural network training...")
    
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Small dataset
    u_samples = np.random.uniform(0, 2*np.pi, 100)
    v_samples = np.random.uniform(0, 2*np.pi, 100)
    targets = ground_truth_field(u_samples, v_samples)
    
    models = train_ensemble(
        u_samples, v_samples, targets,
        n_models=2, epochs=10, seed0=0
    )
    
    assert len(models) == 2, "Ensemble size mismatch"
    
    # Evaluate
    u_eval = np.linspace(0, 2*np.pi, 5)
    v_eval = np.linspace(0, 2*np.pi, 5)
    u_grid, v_grid = np.meshgrid(u_eval, v_eval)
    
    preds, jacs = evaluate_ensemble(models, u_grid, v_grid)
    
    assert preds.shape == (2, 25), f"Predictions shape mismatch: {preds.shape}"
    assert jacs.shape == (2, 25, 2), f"Jacobians shape mismatch: {jacs.shape}"
    assert np.all(np.isfinite(preds)), "Predictions contain NaN/Inf"
    assert np.all(np.isfinite(jacs)), "Jacobians contain NaN/Inf"
    
    print("  ✓ Neural network tests passed")


def test_so2_alignment():
    """Test SO(2) polar decomposition."""
    print("Testing SO(2) alignment...")
    
    # Test with a known rotation
    theta = np.pi / 4
    R_true = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Add small noise
    R = R_true + 0.01 * np.random.randn(2, 2)
    
    R_so = rotation_SO2_closest(R)
    
    # Check SO(2) properties
    assert np.abs(np.linalg.det(R_so) - 1.0) < 1e-10, "Determinant not 1"
    assert np.allclose(R_so @ R_so.T, np.eye(2), atol=1e-10), "Not orthogonal"
    
    print("  ✓ SO(2) alignment tests passed")


def test_sinkhorn():
    """Test Sinkhorn algorithm."""
    print("Testing Sinkhorn algorithm...")
    
    n = 5
    C = np.random.rand(n, n)
    
    for eps in [0.01, 0.1, 1.0, 10.0]:
        P = sinkhorn_uniform(C, eps=eps, max_iter=1000, tol=1e-9)
        
        assert P.shape == (n, n), "Coupling shape mismatch"
        assert np.all(P >= 0), "Negative coupling entries"
        
        # Check marginals (higher tolerance for very small eps)
        row_sum = P.sum(axis=1)
        col_sum = P.sum(axis=0)
        tol = 1e-4 if eps < 0.1 else 1e-6
        assert np.allclose(row_sum, 1.0/n, atol=tol), f"Row sums not uniform (eps={eps})"
        assert np.allclose(col_sum, 1.0/n, atol=tol), f"Col sums not uniform (eps={eps})"
    
    print("  ✓ Sinkhorn tests passed")


def test_transfer_operators():
    """Test transfer operator construction."""
    print("Testing transfer operators...")
    
    n = 4
    C = np.random.rand(n, n)
    
    # Entropic
    T_ent = transfer_operator_entropic(C, eps=0.5)
    assert T_ent.shape == (n, n), "Transfer operator shape mismatch"
    assert np.all(T_ent >= 0), "Negative entries"
    assert np.allclose(T_ent.sum(axis=1), 1.0), "Rows don't sum to 1"
    assert np.allclose(T_ent.sum(axis=0), 1.0), "Columns don't sum to 1"
    
    # Hungarian (returns permutation matrix, doubly stochastic)
    T_hung = transfer_operator_hungarian(C)
    assert T_hung.shape == (n, n), "Transfer operator shape mismatch"
    # Should be a permutation matrix: entries are 0 or 1
    assert np.allclose(T_hung, np.round(T_hung)), "Not a permutation matrix"
    assert np.allclose(T_hung.sum(axis=1), 1.0), "Rows don't sum to 1"
    assert np.allclose(T_hung.sum(axis=0), 1.0), "Columns don't sum to 1"
    
    print("  ✓ Transfer operator tests passed")


def test_permutation_ensemble():
    """Test random ensemble permutation."""
    print("Testing ensemble permutation...")
    
    torch.manual_seed(0)
    np.random.seed(0)
    
    u_samples = np.random.uniform(0, 2*np.pi, 50)
    v_samples = np.random.uniform(0, 2*np.pi, 50)
    targets = ground_truth_field(u_samples, v_samples)
    
    models = train_ensemble(u_samples, v_samples, targets, n_models=3, epochs=5)
    
    u_eval = np.linspace(0, 2*np.pi, 5)
    v_eval = np.linspace(0, 2*np.pi, 5)
    u_grid, v_grid = np.meshgrid(u_eval, v_eval)
    
    preds, jacs = evaluate_ensemble(models, u_grid, v_grid)
    
    # Apply permutation
    rng = np.random.default_rng(42)
    preds_perm, jacs_perm = randomly_permute_ensemble(preds, jacs, rng=rng)
    
    # Check shape preservation
    assert preds_perm.shape == preds.shape, "Predictions shape changed"
    assert jacs_perm.shape == jacs.shape, "Jacobians shape changed"
    
    # Check values changed (with high probability)
    if not np.allclose(preds, preds_perm):
        print("    (Permutation detected as expected)")
    
    print("  ✓ Ensemble permutation tests passed")


def test_holonomy_computation():
    """Test holonomy field computation (end-to-end)."""
    print("Testing holonomy field computation...")
    
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Minimal setup
    u_samples = np.random.uniform(0, 2*np.pi, 50)
    v_samples = np.random.uniform(0, 2*np.pi, 50)
    targets = ground_truth_field(u_samples, v_samples)
    
    models = train_ensemble(u_samples, v_samples, targets, n_models=3, epochs=5)
    
    u_eval = np.linspace(0, 2*np.pi, 8)
    v_eval = np.linspace(0, 2*np.pi, 8)
    u_grid, v_grid = np.meshgrid(u_eval, v_eval)
    
    preds, jacs = evaluate_ensemble(models, u_grid, v_grid)
    
    u_flat = u_grid.ravel()
    v_flat = v_grid.ravel()
    E_all = torus_tangent(u_flat, v_flat)
    s_y = preds.std() + 1e-12
    s_g = jacs.std() + 1e-12
    
    # Holonomy field
    hol = holonomy_field(preds, jacs, E_all, n_eval=8, s_y=s_y, s_g=s_g,
                        eps=0.5, side=1, mode='entropic')
    
    assert hol.shape == (7, 7), f"Holonomy field shape mismatch: {hol.shape}"
    assert np.all(np.isfinite(hol)), "Holonomy contains NaN/Inf"
    assert np.all(hol >= 0), "Holonomy should be non-negative"
    
    print(f"    Holonomy range: [{hol.min():.4e}, {hol.max():.4e}]")
    print("  ✓ Holonomy field computation tests passed")


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("INTEGRATION TESTS: FRAMEWORK MODULE")
    print("="*70 + "\n")
    
    try:
        test_geometry()
        test_ground_truth()
        test_neural_network()
        test_so2_alignment()
        test_sinkhorn()
        test_transfer_operators()
        test_permutation_ensemble()
        test_holonomy_computation()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70 + "\n")
        return True
    
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
