# -[O_o]- j.fuentesaguilar
"""
Unified framework for fibre jets, entropic transport, and Wilson-loop holonomy diagnostics.

Functions handle:
  - Torus geometry, tangent frames, and parametrisation
  - Ensemble training and predictions
  - Jet-space metrics with gauge alignment
  - Entropic optimal transport via Sinkhorn
  - Transfer operators and Wilson-loop holonomy
  - Diagnostic computations over parameter grids
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.optimize import linear_sum_assignment


# ==============================================================================
# GEOMETRY: Torus, tangent frames, embeddings
# ==============================================================================

def torus_parametrisation(u_param, v_param, R_param=1.0, r_param=1.0):
    """
    Embed torus into R^3 using (u,v) intrinsic coordinates.
    Returns (X, Y, Z) coordinates in ambient space.
    """
    X = (R_param + r_param * np.cos(v_param)) * np.cos(u_param)
    Y = (R_param + r_param * np.cos(v_param)) * np.sin(u_param)
    Z = r_param * np.sin(v_param)
    return X, Y, Z


def torus_tangent(u_param, v_param, R_param=1.0, r_param=1.0):
    """
    Compute orthonormal tangent frame at each point on the torus.
    Returns array of shape (..., 2, 3) where rows are e_1, e_2 basis vectors.
    """
    # Partial derivatives
    du = np.stack([
        -(R_param + r_param * np.cos(v_param)) * np.sin(u_param),
         (R_param + r_param * np.cos(v_param)) * np.cos(u_param),
         np.zeros_like(u_param)
    ], axis=-1)

    dv = np.stack([
        -r_param * np.sin(v_param) * np.cos(u_param),
        -r_param * np.sin(v_param) * np.sin(u_param),
         r_param * np.cos(v_param)
    ], axis=-1)

    # Gram–Schmidt: orthonormalise
    norm_du = np.linalg.norm(du, axis=-1, keepdims=True)
    norm_du = np.maximum(norm_du, 1e-12)  # Avoid division by zero
    e1 = du / norm_du
    
    dv_proj = dv - np.sum(dv * e1, axis=-1, keepdims=True) * e1
    norm_dv = np.linalg.norm(dv_proj, axis=-1, keepdims=True)
    norm_dv = np.maximum(norm_dv, 1e-12)  # Avoid division by zero
    e2 = dv_proj / norm_dv
    
    return np.stack([e1, e2], axis=-2)


def ground_truth_field(u, v):
    """Default ground truth scalar field on the torus."""
    return np.sin(2*u + 4*np.sin(v)) + 0.5*np.cos(8*v - u)


# ==============================================================================
# NEURAL NETWORK: Simple MLP ensemble
# ==============================================================================

class SimpleMLP(nn.Module):
    """Single-hidden-layer MLP: 3 -> 10 -> 1."""
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.net(x)


def train_ensemble(u_samples, v_samples, targets, R_param=1.0, r_param=1.0,
                   n_models=10, batch_size=64, learning_rate=1e-2, epochs=200,
                   seed0=0, weight_decay=0.0):
    """
    Train an ensemble of MLP models on sampled torus data.
    Returns list of trained models.
    """
    # Convert to ambient coordinates
    x_samples, y_samples, z_samples = torus_parametrisation(
        u_samples, v_samples, R_param=R_param, r_param=r_param
    )
    
    X_data = torch.tensor(
        np.vstack([x_samples, y_samples, z_samples]).T,
        dtype=torch.float32
    )
    y_data = torch.tensor(targets, dtype=torch.float32).view(-1, 1)
    
    dataset = TensorDataset(X_data, y_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    models = []
    for seed in range(n_models):
        torch.manual_seed(seed0 + seed)
        model = SimpleMLP()
        optimiser = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        loss_fn = nn.MSELoss()
        
        for _ in range(epochs):
            for xb, yb in dataloader:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
        
        models.append(model)
    
    return models


def evaluate_ensemble(models, u_eval, v_eval, R_param=1.0, r_param=1.0):
    """
    Evaluate ensemble predictions and intrinsic gradients on a grid.
    Returns predictions (n_models, n_points) and jacobians (n_models, n_points, 2).
    """
    u_flat = u_eval.ravel()
    v_flat = v_eval.ravel()
    
    x_eval, y_eval, z_eval = torus_parametrisation(
        u_flat, v_flat, R_param=R_param, r_param=r_param
    )
    
    X_eval = torch.tensor(
        np.vstack([x_eval, y_eval, z_eval]).T,
        dtype=torch.float32
    )
    
    E_all = torus_tangent(u_flat, v_flat, R_param=R_param, r_param=r_param)
    
    predictions = []
    jacobians = []
    
    for model in models:
        model.eval()
        Xi = X_eval.detach().clone().requires_grad_(True)
        y_pred = model(Xi)
        
        grad = torch.autograd.grad(
            outputs=y_pred,
            inputs=Xi,
            grad_outputs=torch.ones_like(y_pred),
            retain_graph=False,
            create_graph=False
        )[0]
        
        y_np = y_pred.detach().cpu().numpy().squeeze()
        g_np = grad.detach().cpu().numpy()
        
        e1 = E_all[:, 0, :]
        e2 = E_all[:, 1, :]
        g1 = np.sum(g_np * e1, axis=1)
        g2 = np.sum(g_np * e2, axis=1)
        g_tan = np.stack([g1, g2], axis=1)
        
        predictions.append(y_np)
        jacobians.append(g_tan)
    
    return np.stack(predictions), np.stack(jacobians)


# ==============================================================================
# JET METRIC AND GAUGE ALIGNMENT
# ==============================================================================

def rotation_SO2_closest(R):
    """
    Compute closest SO(2) element to a 2x2 matrix R in Frobenius norm.
    """
    U, _, Vt = np.linalg.svd(R)
    Rso = U @ Vt
    if np.linalg.det(Rso) < 0:
        U[:, 1] *= -1
        Rso = U @ Vt
    return Rso


def rotation_AB(i_A, j_A, i_B, j_B, E_all, n_eval):
    """
    Compute SO(2) alignment from frame B to frame A.
    """
    k_A = i_A * n_eval + j_A
    k_B = i_B * n_eval + j_B
    E_A = E_all[k_A]
    E_B = E_all[k_B]
    R = E_A @ E_B.T
    return rotation_SO2_closest(R)


def jet_costm(fibre_A, fibre_B, s_y, s_g, R_AB):
    """
    Compute pairwise squared costs between jets in two fibres.
    fibre_A, fibre_B: (n_models, 3) with columns [y, g1, g2]
    R_AB: 2x2 SO(2) rotation matrix
    Returns cost matrix of shape (n_models, n_models).
    """
    G_B_rot = fibre_B[:, 1:3] @ R_AB.T
    dY = (fibre_A[:, None, 0] - fibre_B[None, :, 0]) / s_y
    dG = (fibre_A[:, None, 1:3] - G_B_rot[None, :, :]) / s_g
    C = dY**2 + np.sum(dG**2, axis=2)
    return C


# ==============================================================================
# ENTROPIC OPTIMAL TRANSPORT: Sinkhorn algorithm
# ==============================================================================

def logsumexp(A, axis=None, keepdims=False):
    """Numerically stable log-sum-exp computation."""
    m = np.max(A, axis=axis, keepdims=True)
    S = np.log(np.sum(np.exp(A - m), axis=axis, keepdims=True)) + m
    if keepdims:
        return S
    return np.squeeze(S, axis=axis)


def sinkhorn_uniform(C, eps=0.5, max_iter=1000, tol=1e-9):
    """
    Entropic optimal transport with uniform marginals via Sinkhorn.
    Returns coupling matrix P of shape (n, n).
    """
    n = C.shape[0]
    eps = max(float(eps), 1e-8)
    
    Cc = C - np.min(C)
    log_a = -np.log(n)
    log_b = -np.log(n)
    
    f = np.zeros(n)
    g = np.zeros(n)
    
    for _ in range(max_iter):
        f_prev, g_prev = f.copy(), g.copy()
        
        S = (f[:, None] + g[None, :] - Cc) / eps
        lse_rows = logsumexp(S, axis=1)
        f = f + eps * (log_a - lse_rows)
        
        S = (f[:, None] + g[None, :] - Cc) / eps
        lse_cols = logsumexp(S, axis=0)
        g = g + eps * (log_b - lse_cols)
        
        if max(np.max(np.abs(f - f_prev)), np.max(np.abs(g - g_prev))) < tol:
            break
    
    P_log = (f[:, None] + g[None, :] - Cc) / eps
    P = np.exp(P_log)
    P = np.maximum(P, 0)
    return P


def transfer_operator_entropic(C, eps=0.5):
    """Transfer operator from entropic OT coupling with uniform marginals."""
    n = C.shape[0]
    P = sinkhorn_uniform(C, eps=eps)
    T = n * P
    return T


# ==============================================================================
# HUNGARIAN ALGORITHM: Exact assignment
# ==============================================================================

def transfer_operator_hungarian(C):
    """Transfer operator from exact linear assignment (Hungarian algorithm)."""
    n = C.shape[0]
    row_ind, col_ind = linear_sum_assignment(C)
    P = np.zeros((n, n))
    P[row_ind, col_ind] = 1.0
    T = P
    return T


# ==============================================================================
# WILSON-LOOP HOLONOMY
# ==============================================================================

def fibre_jets(i, j, predictions, jacobians, n_eval):
    """Extract fibre jets (stacked [y, g1, g2]) at grid index (i,j)."""
    k = i * n_eval + j
    return np.hstack([
        predictions[:, k:k+1],
        jacobians[:, k, :]
    ])


def loop_holonomy(i0, j0, side, predictions, jacobians, E_all, n_eval,
                  s_y, s_g, eps=0.5, mode='entropic'):
    """
    Compute Wilson-loop holonomy and diagnostics for a rectangular loop.
    
    Parameters:
    -----------
    mode : str
        'entropic' or 'hungarian'
    
    Returns:
    --------
    h_fro : float
        Normalized Frobenius gap of holonomy
    disp : float
        Mean barycentric displacement
    """
    loop_indices = [
        (i0, j0),
        (i0, j0 + side),
        (i0 + side, j0 + side),
        (i0 + side, j0),
        (i0, j0)
    ]
    
    Ts = []
    for (ia, ja), (ib, jb) in zip(loop_indices[:-1], loop_indices[1:]):
        A = fibre_jets(ia, ja, predictions, jacobians, n_eval)
        B = fibre_jets(ib, jb, predictions, jacobians, n_eval)
        R_AB = rotation_AB(ia, ja, ib, jb, E_all, n_eval)
        C = jet_costm(A, B, s_y, s_g, R_AB)
        
        if mode == 'entropic':
            T = transfer_operator_entropic(C, eps=eps)
        elif mode == 'hungarian':
            T = transfer_operator_hungarian(C)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        Ts.append(T)
    
    H = Ts[-1].T @ Ts[-2].T @ Ts[-3].T @ Ts[-4].T
    I = np.eye(H.shape[0])
    h_fro = np.linalg.norm(H - I, ord='fro') / H.shape[0]

    Z0 = fibre_jets(i0, j0, predictions, jacobians, n_eval)
    Z_loop = H @ Z0
    
    DY = (Z_loop[:, 0] - Z0[:, 0]) / s_y
    DG1 = (Z_loop[:, 1] - Z0[:, 1]) / s_g
    DG2 = (Z_loop[:, 2] - Z0[:, 2]) / s_g
    per_model_disp = np.sqrt(DY**2 + DG1**2 + DG2**2)
    disp = per_model_disp.mean()
    
    return h_fro, disp


def holonomy_field(predictions, jacobians, E_all, n_eval, s_y, s_g,
                   eps=0.5, side=1, mode='entropic'):
    """Compute holonomy diagnostic over all unit plaquettes in parameter grid."""
    hol_map = np.zeros((n_eval - side, n_eval - side))
    for i in range(n_eval - side):
        for j in range(n_eval - side):
            hol_map[i, j], _ = loop_holonomy(
                i, j, side, predictions, jacobians, E_all, n_eval,
                s_y, s_g, eps=eps, mode=mode
            )
    return hol_map


# ==============================================================================
# DISTANCE-TO-PERMUTATION DIAGNOSTICS
# ==============================================================================

def permutation_distance_entropy(T):
    """Row entropy as measure of distance from permutation matrix."""
    # Entropy of each row
    eps_safe = 1e-12
    entropies = []
    for row in T:
        p = np.maximum(row / row.sum(), eps_safe)
        h = -np.sum(p * np.log(p))
        entropies.append(h)
    return np.mean(entropies)


def permutation_distance_frobenius(T, T_hungarian):
    """Frobenius norm between entropic coupling and Hungarian solution."""
    return np.linalg.norm(T - T_hungarian, ord='fro')


# ==============================================================================
# GAUGE INVARIANCE CHECK
# ==============================================================================

def randomly_permute_ensemble(predictions, jacobians, rng=None):
    """
    Apply independent random permutations to ensemble indices at each point.
    Returns permuted copies of predictions and jacobians.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    
    n_models, n_points = predictions.shape
    preds_perm = predictions.copy()
    jacs_perm = jacobians.copy()
    
    for k in range(n_points):
        perm = rng.permutation(n_models)
        preds_perm[:, k] = predictions[perm, k]
        jacs_perm[:, k, :] = jacobians[perm, k, :]
    
    return preds_perm, jacs_perm


# ==============================================================================
# PLOTTING UTILITIES
# ==============================================================================

def plot_style_setup():
    """Configure common plotting styles."""
    FC = "#777777"  # Foreground color
    FS = 16         # Font size
    return FC, FS


def style_3d_axes(ax, FC="#777777", FS=16):
    """Apply consistent styling to 3D matplotlib axes."""
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.9, 0.9, 0.9, 0.5))
        axis._axinfo["grid"]["linewidth"] = 1
    ax.grid(True)
    
    ax.set_xlabel("x", color=FC, fontsize=FS)
    ax.set_ylabel("y", color=FC, fontsize=FS)
    ax.set_zlabel("z", color=FC, fontsize=FS)
    ax.tick_params(axis="both", which="major", colors=FC, labelsize=FS-1)
    
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=3))
    
    ax.set_box_aspect((1, 1, 0.8))


def style_colorbar(cb, label, FC="#777777", FS=16):
    """Apply consistent styling to colorbars."""
    cb.set_label(label, color=FC, fontsize=FS)
    cb.ax.yaxis.set_tick_params(color=FC, labelsize=FS-1)
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=FC)
